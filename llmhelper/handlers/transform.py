"""Apply an instruction to file content and write the result to the outbox.

Where ``digest`` compresses many files into one short answer, ``transform``
preserves structure: each input file produces one output file. Reformatting,
translating, restructuring notes, extracting tables — the kind of mechanical
work that is tedious at volume and does not need frontier reasoning.

Nothing is written outside ``results/``. Moving output back into a project is a
decision for a human or a reviewing agent, never for this handler.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..chunk import count_tokens, split_lines
from ..config import Tier
from ..guard import expand, read_text, resolve_write, slug
from .base import DATA_RULE, Ctx, HandlerResult, fence

SYSTEM = f"""You transform documents exactly as instructed.

{DATA_RULE}

Rules:
- Output ONLY the transformed content. No preamble, no explanation, no code fence
  around the whole answer unless the instruction asks for one.
- Preserve all facts, numbers and identifiers precisely. Never invent content.
- If a section cannot be transformed as asked, copy it through unchanged rather
  than dropping it."""


class TransformHandler:
    kind = "transform"

    def run(self, spec: dict[str, Any], ctx: Ctx, tier: Tier) -> HandlerResult:
        paths = spec.get("paths") or []
        if isinstance(paths, str):
            paths = [paths]
        instruction = (spec.get("instruction") or "").strip()
        if not paths or not instruction:
            raise ValueError("transform needs spec['paths'] and spec['instruction']")

        suffix = spec.get("suffix") or ".md"
        max_out = int(spec.get("max_out_tokens") or 4000)

        files, skipped = expand(paths, ctx.cfg)
        if not files:
            raise FileNotFoundError(
                f"no readable files matched {paths!r}. " + "; ".join(skipped[:3])
            )

        written: list[tuple[Path, Path, int]] = []
        failures: list[str] = []
        source_tokens = 0

        for i, path in enumerate(files, start=1):
            ctx.progress(f"transform {i}/{len(files)} — {path.name}")
            try:
                text = read_text(path)
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{path}: unreadable ({exc})")
                continue
            if not text.strip():
                continue

            source_tokens += count_tokens(text)
            try:
                output = self._transform_one(ctx, tier, text, instruction, path, max_out)
            except Exception as exc:  # noqa: BLE001 - one bad file must not lose the rest
                failures.append(f"{path}: {type(exc).__name__}: {exc}")
                continue

            target = resolve_write(
                f"transform/{slug(instruction, 30)}/{path.stem}{suffix}", ctx.cfg
            )
            target.write_text(output, encoding="utf-8")
            written.append((path, target, count_tokens(output)))

        if not written:
            raise RuntimeError(
                "transform produced nothing. " + ("; ".join(failures[:3]) or "no reason recorded")
            )

        lines = [f"Transformed {len(written)} file(s) with: {instruction}\n"]
        for src, dst, tok in written:
            lines.append(f"- {src.name} -> {dst}  ({tok:,} tokens)")
        if failures:
            lines.append("\nFailed:")
            lines += [f"- {f}" for f in failures]
        if skipped:
            lines.append(f"\nSkipped {len(skipped)} path(s) during expansion.")
        summary = "\n".join(lines)

        return HandlerResult(
            summary=summary,
            detail="",  # the outputs are the detail; they already live in the outbox
            meta={
                "written": [str(d) for _, d, _ in written],
                "failed": failures,
                "skipped": skipped[:20],
                "instruction": instruction,
            },
            input_tokens=source_tokens,
            output_tokens=count_tokens(summary),
            local_prompt_tokens=ctx.local_prompt_tokens,
            local_completion_tokens=ctx.local_completion_tokens,
            duration_s=ctx.elapsed,
        )

    def _transform_one(
        self, ctx: Ctx, tier: Tier, text: str, instruction: str, path: Path, max_out: int
    ) -> str:
        """Transform a file, in slices when it does not fit the context."""
        budget = tier.context - max_out - 1000
        if count_tokens(text) <= budget:
            return ctx.generate(
                f"Instruction: {instruction}\n\n"
                f"Document ({path.name}):\n{fence(text)}\n\n"
                f"Produce the transformed document.",
                tier,
                system=SYSTEM,
                max_tokens=max_out,
            ).text

        pieces = split_lines(text, path, max_tokens=budget // 2)
        out: list[str] = []
        for n, piece in enumerate(pieces, start=1):
            ctx.progress(f"  {path.name} part {n}/{len(pieces)}")
            out.append(
                ctx.generate(
                    f"Instruction: {instruction}\n\n"
                    f"This is part {n} of {len(pieces)} of {path.name} "
                    f"(lines {piece.start_line}-{piece.end_line}). Transform this part only; "
                    f"do not add an introduction or conclusion.\n{fence(piece.text)}",
                    tier,
                    system=SYSTEM,
                    max_tokens=max_out,
                ).text
            )
        return "\n\n".join(out)
