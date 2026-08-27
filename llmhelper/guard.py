"""The filesystem boundary for everything the local model touches.

Two rules, enforced in one place so they cannot drift apart:

  * reads  — only under ``paths.read_roots``, never matching ``paths.deny_globs``
  * writes — only under ``paths.outbox``

Both checks happen *after* ``Path.resolve()``, so ``..`` traversal, symlinks and
junctions cannot escape: whatever the path claims to be, we judge where it
actually lands.
"""

from __future__ import annotations

import fnmatch
import os
import re
from collections.abc import Iterable, Iterator
from pathlib import Path

from .config import Config, get_config
from .extract import MARKUP_SUFFIXES, extract, extractable

_SLUG = re.compile(r"[^a-z0-9]+")


def slug(text: str, limit: int = 40) -> str:
    """Filesystem-safe stem derived from free text, for naming outbox files."""
    s = _SLUG.sub("-", (text or "").lower()).strip("-")
    return s[:limit].rstrip("-") or "job"

# Read far enough to be confident about the file type without slurping a huge blob.
_SNIFF_BYTES = 8192
# Extensions we never treat as text, checked before sniffing to save the open().
_BINARY_SUFFIXES = frozenset(
    """.gguf .bin .safetensors .pt .pth .onnx .zip .gz .bz2 .xz .7z .rar .tar
       .png .jpg .jpeg .gif .bmp .tiff .webp .ico .mp3 .mp4 .wav .avi .mkv .mov
       .pdf .docx .xlsx .pptx .exe .dll .so .dylib .pyc .pyd .db .sqlite .wal""".split()
)


class GuardError(PermissionError):
    """A path fell outside the sandbox. Never catch this to 'try anyway'."""


def _norm(p: Path) -> str:
    """Case- and separator-normalised absolute path, for comparison only."""
    return os.path.normcase(os.path.normpath(str(p)))


def _is_unc(path: str | Path) -> bool:
    """True for ``\\\\server\\share`` style paths, in either slash spelling."""
    s = str(path).replace("/", "\\")
    return s.startswith("\\\\")


def _resolved(path: str | Path) -> Path:
    """Absolute, normalised path — without touching the network.

    ``Path.resolve()`` on a UNC path can consult the network to resolve the host,
    which makes it slow and, worse, *nondeterministic*: a security decision must
    not depend on whether a DNS lookup happened to succeed. UNC paths are
    normalised textually instead. They can still be matched against a UNC root if
    one is configured; they simply never reach the resolver.
    """
    p = Path(path).expanduser()
    if _is_unc(p):
        return Path(os.path.normpath(str(p).replace("/", "\\")))
    # strict=False so we can also validate not-yet-existing write targets.
    return p.resolve(strict=False)


def _under(child: Path, parent: Path) -> bool:
    c, p = _norm(child), _norm(parent)
    if c == p:
        return True
    return c.startswith(p if p.endswith(os.sep) else p + os.sep)


def is_denied(path: Path, cfg: Config | None = None) -> str | None:
    """Return the deny pattern this path matches, or None if it is clean."""
    cfg = cfg or get_config()
    posix = _resolved(path).as_posix()
    for pattern in cfg.paths.deny_globs:
        if fnmatch.fnmatch(posix, pattern):
            return pattern
    return None


def resolve_read(path: str | Path, cfg: Config | None = None) -> Path:
    """Absolute path if the local model may read it, else raise GuardError."""
    cfg = cfg or get_config()
    target = _resolved(path)

    if not any(_under(target, root) for root in cfg.paths.read_roots):
        roots = ", ".join(str(r) for r in cfg.paths.read_roots)
        raise GuardError(f"refusing to read outside the allowed roots ({roots}): {target}")

    if pattern := is_denied(target, cfg):
        raise GuardError(f"refusing to read {target}: matches deny rule {pattern!r}")

    return target


def resolve_write(path: str | Path, cfg: Config | None = None) -> Path:
    """Absolute path if it lands inside the outbox, else raise GuardError.

    Parent directories are created so handlers can write straight away.
    """
    cfg = cfg or get_config()
    outbox = _resolved(cfg.paths.outbox)
    target = _resolved(outbox / path if not Path(path).is_absolute() else path)

    if not _under(target, outbox):
        raise GuardError(f"refusing to write outside the outbox ({outbox}): {target}")
    if target == outbox:
        raise GuardError(f"refusing to write to the outbox root itself: {target}")

    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def looks_binary(path: Path) -> bool:
    """Cheap text/binary sniff: known suffix, or a NUL byte in the first 8 KB.

    "Binary" here means "no text can be got out of it", not "the bytes are not
    ASCII". A PDF is binary on disk and prose underneath, so anything with an
    extractor is readable as far as callers are concerned.
    """
    if extractable(path):
        return False
    if path.suffix.lower() in _BINARY_SUFFIXES:
        return True
    try:
        with path.open("rb") as fh:
            return b"\x00" in fh.read(_SNIFF_BYTES)
    except OSError:
        return True


def read_text(path: str | Path, max_bytes: int | None = None, cfg: Config | None = None) -> str:
    """Guarded text read with an encoding fallback chain.

    ``max_bytes`` truncates rather than raising: a digest of the first N bytes of
    a giant log is still useful, and the caller records the truncation.
    """
    target = resolve_read(path, cfg)

    # Documents are extracted whole: a PDF truncated mid-stream is unparseable,
    # so max_bytes is applied to the resulting text instead of the raw file.
    if extractable(target) and target.suffix.lower() not in MARKUP_SUFFIXES:
        text = extract(target)
        return text[:max_bytes] if max_bytes is not None else text

    raw = target.read_bytes() if max_bytes is None else target.open("rb").read(max_bytes)
    for encoding in ("utf-8", "utf-8-sig", "cp1251", "cp1252", "latin-1"):
        try:
            decoded = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        decoded = raw.decode("utf-8", errors="replace")
    return extract(target, decoded) if extractable(target) else decoded


def expand(
    patterns: Iterable[str],
    cfg: Config | None = None,
    skip_binary: bool = True,
) -> tuple[list[Path], list[str]]:
    """Expand paths and globs into guarded files.

    Returns ``(files, skipped)`` — skipped carries a human-readable reason per
    entry so callers can report *why* something was left out instead of silently
    dropping it.
    """
    cfg = cfg or get_config()
    files: list[Path] = []
    skipped: list[str] = []
    seen: set[str] = set()

    def consider(p: Path) -> None:
        try:
            target = resolve_read(p, cfg)
        except GuardError as exc:
            skipped.append(str(exc))
            return
        if not target.is_file():
            return
        key = _norm(target)
        if key in seen:
            return
        if skip_binary and looks_binary(target):
            skipped.append(f"{target}: looks binary")
            return
        seen.add(key)
        files.append(target)

    for pattern in patterns:
        pattern = str(pattern).strip()
        if not pattern:
            continue
        candidate = Path(pattern).expanduser()
        if any(ch in pattern for ch in "*?["):
            # Anchor the glob at its first non-magic parent so we never walk from cwd.
            anchor = candidate.anchor or "."
            rel = str(candidate)[len(anchor):] if candidate.anchor else str(candidate)
            try:
                matches = sorted(Path(anchor).glob(rel))
            except (OSError, ValueError) as exc:
                skipped.append(f"{pattern}: bad glob ({exc})")
                continue
            if not matches:
                skipped.append(f"{pattern}: matched nothing")
            for m in matches:
                consider(m)
        elif candidate.is_dir():
            try:
                target_dir = resolve_read(candidate, cfg)
            except GuardError as exc:
                skipped.append(str(exc))
                continue
            for m in sorted(target_dir.rglob("*")):
                consider(m)
        else:
            # Check the boundary before existence. Reporting "does not exist" for
            # a path outside the allowed roots answers a question the caller was
            # not entitled to ask — whether some file elsewhere on the disk is
            # there. Out-of-root always reports as out-of-root.
            try:
                resolve_read(candidate, cfg)
            except GuardError as exc:
                skipped.append(str(exc))
                continue
            if not candidate.exists():
                skipped.append(f"{pattern}: does not exist")
                continue
            consider(candidate)

    return files, skipped


def iter_outbox(cfg: Config | None = None) -> Iterator[Path]:
    """Every file currently in the outbox."""
    cfg = cfg or get_config()
    outbox = _resolved(cfg.paths.outbox)
    if outbox.is_dir():
        yield from (p for p in outbox.rglob("*") if p.is_file())
