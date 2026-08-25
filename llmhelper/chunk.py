"""Token counting and provenance-preserving chunking.

Chunks carry the file and line range they came from, so a digest can cite
``file:line`` instead of asserting things from nowhere. That citation is what
makes an offloaded answer checkable without re-reading the source.

On the tokenizer: we count with tiktoken's ``cl100k_base``. It is not the local
model's tokenizer and not exactly Claude's either, but the ledger measures *what
the expensive model would have paid*, and cl100k is a good proxy for that. When
tiktoken is unavailable we fall back to a chars/4 estimate rather than failing.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

_ENCODER = None
_ENCODER_TRIED = False


@lru_cache(maxsize=1)
def _encoder():
    """Load tiktoken once; None means 'use the heuristic'."""
    global _ENCODER_TRIED
    _ENCODER_TRIED = True
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception:  # noqa: BLE001 - offline, missing cache, any of it: degrade quietly
        return None


def count_tokens(text: str) -> int:
    """Token count, exact when tiktoken is available and estimated otherwise."""
    if not text:
        return 0
    enc = _encoder()
    if enc is None:
        return max(1, len(text) // 4)
    return len(enc.encode(text, disallowed_special=()))


def exact_counting() -> bool:
    """True when counts come from tiktoken rather than the chars/4 fallback."""
    return _encoder() is not None


@dataclass
class Chunk:
    """A slice of one file, with enough provenance to cite it."""

    text: str
    path: Path
    index: int
    start_line: int
    end_line: int
    tokens: int

    @property
    def citation(self) -> str:
        if self.start_line == self.end_line:
            return f"{self.path.name}:{self.start_line}"
        return f"{self.path.name}:{self.start_line}-{self.end_line}"

    def header(self) -> str:
        return f"[{self.path} lines {self.start_line}-{self.end_line}]"


def split_lines(
    text: str,
    path: Path,
    max_tokens: int,
    overlap_tokens: int = 0,
    number_lines: bool = False,
) -> list[Chunk]:
    """Split one file's text into token-bounded chunks along line boundaries.

    Lines are never broken mid-way unless a single line exceeds the budget on its
    own, which keeps line numbers meaningful.

    ``number_lines`` makes the budget count what the caller will actually send:
    a line shown to the model as ``1142: text`` costs more than ``text``. Sizing
    chunks on the raw text and then sending a numbered version overflows the
    context — by 25-45% on logs and source, which is enough to blow an 8K window.
    Estimating that inflation with a fudge factor is guesswork; measuring the
    real prefix cost per line is exact.
    """
    lines = text.splitlines()
    if not lines:
        return []

    chunks: list[Chunk] = []
    buf: list[str] = []
    buf_tokens = 0
    start_line = 1
    idx = 0

    def flush(end_line: int) -> None:
        nonlocal buf, buf_tokens, idx, start_line
        if not buf:
            return
        body = "\n".join(buf)
        chunks.append(
            Chunk(
                text=body,
                path=path,
                index=idx,
                start_line=start_line,
                end_line=end_line,
                tokens=buf_tokens,
            )
        )
        idx += 1
        if overlap_tokens > 0:
            # Carry the tail forward so a fact split across a boundary survives.
            carried: list[str] = []
            carried_tokens = 0
            for line in reversed(buf):
                t = count_tokens(line) + (4 if number_lines else 0) + 1
                if carried_tokens + t > overlap_tokens:
                    break
                carried.insert(0, line)
                carried_tokens += t
            buf = carried
            buf_tokens = carried_tokens
            start_line = max(1, end_line - len(carried) + 1)
        else:
            buf = []
            buf_tokens = 0
            start_line = end_line + 1

    for lineno, line in enumerate(lines, start=1):
        # Count the line as it will be *sent*, prefix included.
        t = count_tokens(f"{lineno}: {line}" if number_lines else line) + 1
        if t > max_tokens:
            # A single monstrous line (minified JSON, a base64 blob). Hard-split it.
            flush(lineno - 1)
            enc = _encoder()
            if enc is not None:
                ids = enc.encode(line, disallowed_special=())
                pieces = [
                    enc.decode(ids[i : i + max_tokens]) for i in range(0, len(ids), max_tokens)
                ]
            else:
                step = max_tokens * 4
                pieces = [line[i : i + step] for i in range(0, len(line), step)]
            for piece in pieces:
                chunks.append(
                    Chunk(
                        text=piece,
                        path=path,
                        index=idx,
                        start_line=lineno,
                        end_line=lineno,
                        tokens=count_tokens(piece),
                    )
                )
                idx += 1
            start_line = lineno + 1
            continue

        if buf_tokens + t > max_tokens and buf:
            flush(lineno - 1)

        if not buf:
            start_line = lineno
        buf.append(line)
        buf_tokens += t

    flush(len(lines))
    return chunks


def chunk_files(
    files: list[Path],
    max_tokens: int,
    overlap_tokens: int = 0,
    read_text=None,
    max_chunks: int | None = None,
    number_lines: bool = False,
) -> tuple[list[Chunk], list[str], int]:
    """Chunk many files.

    Returns ``(chunks, notes, total_tokens)``. ``notes`` explains anything that
    was truncated or skipped so the caller can say so instead of silently
    dropping material. ``read_text`` defaults to the guarded reader.
    """
    if read_text is None:
        from .guard import read_text as guarded_read

        read_text = guarded_read

    chunks: list[Chunk] = []
    notes: list[str] = []
    total = 0

    for path in files:
        try:
            text = read_text(path)
        except Exception as exc:  # noqa: BLE001 - one bad file must not kill the batch
            notes.append(f"{path}: unreadable ({exc})")
            continue
        if not text.strip():
            continue
        produced = split_lines(
            text, Path(path), max_tokens, overlap_tokens, number_lines=number_lines
        )
        total += sum(c.tokens for c in produced)
        chunks.extend(produced)
        if max_chunks is not None and len(chunks) > max_chunks:
            notes.append(
                f"input exceeded max_map_chunks={max_chunks}; "
                f"stopped after {path} — narrow the glob or raise the budget"
            )
            chunks = chunks[:max_chunks]
            break

    return chunks, notes, total
