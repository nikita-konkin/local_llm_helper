"""Transcribe audio locally, then clean the transcript up with the local LLM.

Two stages, because they fail differently. faster-whisper produces a timestamped
but messy transcript — no punctuation to speak of, technical terms mangled. The
LLM pass fixes readability without being allowed to invent content.

Both the raw and the cleaned transcript are written to the outbox. The raw one is
the record; the cleaned one is the convenience. Keeping both means a bad cleanup
pass is always recoverable.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ..chunk import count_tokens, split_lines
from ..config import Tier
from ..guard import resolve_read, resolve_write, slug
from .base import DATA_RULE, Ctx, HandlerResult, fence

CLEANUP_SYSTEM = f"""You clean up raw speech-to-text output.

{DATA_RULE}

Do exactly this and nothing more:
- Add punctuation, capitalisation and paragraph breaks.
- Remove filler words and false starts ("um", "you know", repeated words).
- Fix words the recogniser clearly misheard, using context. If a technical term
  is ambiguous, leave it and mark it [sic].

Never summarise, never reorder, never add content that was not spoken. The output
must contain everything the speaker said."""

AUDIO_SUFFIXES = frozenset(
    ".mp3 .wav .m4a .flac .ogg .opus .wma .aac .mp4 .mkv .mov .avi .webm".split()
)


class TranscribeHandler:
    kind = "transcribe"

    def run(self, spec: dict[str, Any], ctx: Ctx, tier: Tier) -> HandlerResult:
        raw_path = spec.get("path") or spec.get("paths")
        if isinstance(raw_path, list):
            raw_path = raw_path[0] if raw_path else None
        if not raw_path:
            raise ValueError("transcribe needs spec['path'] pointing at an audio/video file")

        # resolve_read, not expand: audio is binary and expand() filters binaries out.
        audio = resolve_read(raw_path, ctx.cfg)
        if not audio.is_file():
            raise FileNotFoundError(f"{audio} is not a file")
        if audio.suffix.lower() not in AUDIO_SUFFIXES:
            raise ValueError(
                f"{audio.suffix!r} is not a recognised audio/video extension. "
                f"Expected one of: {', '.join(sorted(AUDIO_SUFFIXES))}"
            )

        model_size = spec.get("model_size") or "small"
        language = spec.get("language")
        do_cleanup = spec.get("cleanup", True)

        ctx.progress(f"transcribing {audio.name} with faster-whisper ({model_size})")
        started = time.monotonic()
        segments, info = self._transcribe(audio, model_size, language)
        transcribe_s = time.monotonic() - started

        if not segments:
            raise RuntimeError(f"no speech recognised in {audio.name}")

        timestamped = "\n".join(f"[{self._ts(s[0])}] {s[2].strip()}" for s in segments)
        plain = " ".join(s[2].strip() for s in segments)
        stem = slug(audio.stem, 40)

        raw_target = resolve_write(f"transcripts/{stem}.raw.txt", ctx.cfg)
        raw_target.write_text(timestamped, encoding="utf-8")

        detail = [
            f"# Transcript — {audio.name}\n",
            f"- language: {info.get('language')} "
            f"(confidence {info.get('language_probability', 0):.2f})",
            f"- duration: {info.get('duration', 0):.0f}s, "
            f"transcribed in {transcribe_s:.0f}s",
            f"- segments: {len(segments)}",
            f"- raw transcript: {raw_target}\n",
        ]

        cleaned = ""
        clean_target: Path | None = None
        if do_cleanup:
            ctx.progress("cleaning transcript with the local model")
            cleaned = self._cleanup(ctx, tier, plain)
            clean_target = resolve_write(f"transcripts/{stem}.clean.md", ctx.cfg)
            clean_target.write_text(cleaned, encoding="utf-8")
            detail += [f"- cleaned transcript: {clean_target}\n", "## Cleaned\n", cleaned]
        else:
            detail += ["## Raw\n", timestamped]

        summary = (
            f"Transcribed {audio.name} — {info.get('duration', 0):.0f}s of audio, "
            f"{len(segments)} segment(s), language {info.get('language')}, "
            f"in {transcribe_s:.0f}s.\n"
            f"Raw: {raw_target}\n"
            + (f"Cleaned: {clean_target}\n" if clean_target else "")
            + f"\nFirst lines:\n{chr(10).join(timestamped.splitlines()[:5])}"
        )

        spoken_tokens = count_tokens(cleaned or plain)
        return HandlerResult(
            summary=summary,
            detail="\n".join(detail),
            meta={
                "audio": str(audio),
                "language": info.get("language"),
                "duration_s": round(float(info.get("duration", 0)), 1),
                "segments": len(segments),
                "raw_path": str(raw_target),
                "model_size": model_size,
            },
            input_tokens=spoken_tokens,
            output_tokens=count_tokens(summary),
            local_prompt_tokens=ctx.local_prompt_tokens,
            local_completion_tokens=ctx.local_completion_tokens,
            duration_s=ctx.elapsed,
        )

    # ------------------------------------------------------------- internals

    @staticmethod
    def _ts(seconds: float) -> str:
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

    def _transcribe(self, audio: Path, model_size: str, language: str | None):
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "faster-whisper is not importable in this environment"
            ) from exc

        # GPU first; ctranslate2 needs cuDNN and quietly may not have it.
        attempts = [("cuda", "float16"), ("cpu", "int8")]
        last: Exception | None = None
        for device, compute_type in attempts:
            try:
                model = WhisperModel(model_size, device=device, compute_type=compute_type)
                seg_iter, info = model.transcribe(
                    str(audio), beam_size=5, language=language, vad_filter=True
                )
                segments = [(s.start, s.end, s.text) for s in seg_iter]
                return segments, {
                    "language": info.language,
                    "language_probability": info.language_probability,
                    "duration": info.duration,
                    "device": device,
                }
            except Exception as exc:  # noqa: BLE001 - fall through to CPU
                last = exc
                continue
        raise RuntimeError(f"transcription failed on both GPU and CPU: {last}")

    def _cleanup(self, ctx: Ctx, tier: Tier, text: str) -> str:
        budget = tier.context // 2
        if count_tokens(text) <= budget:
            return ctx.generate(
                f"Clean up this transcript:\n{fence(text)}",
                tier,
                system=CLEANUP_SYSTEM,
                max_tokens=budget,
            ).text

        pieces = split_lines(text, Path("transcript"), max_tokens=budget // 2)
        out: list[str] = []
        for n, piece in enumerate(pieces, start=1):
            ctx.progress(f"  cleanup part {n}/{len(pieces)}")
            out.append(
                ctx.generate(
                    f"Clean up part {n} of {len(pieces)} of a transcript. "
                    f"Do not add an introduction or conclusion.\n{fence(piece.text)}",
                    tier,
                    system=CLEANUP_SYSTEM,
                    max_tokens=budget,
                ).text
            )
        return "\n\n".join(out)
