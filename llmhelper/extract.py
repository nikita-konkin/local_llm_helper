"""Text extraction for formats that are binary on disk but text underneath.

The guard refuses binary files, which is right for a .gguf and wrong for a PDF:
the bytes are opaque, but the *content* is exactly the kind of bulk prose this
project exists to keep out of an expensive context. Refusing it just moved the
work back to the caller, who had to extract the text by hand first.

Every extractor here returns plain text with a positional marker (page, sheet,
slide) on its own line, so the line numbers the digest handler adds still lead a
human back to the right place in the original document.

Backends are imported lazily. A missing one is a clear install instruction, not
an ImportError at startup - none of these are needed to run the rest of the tool.
"""

from __future__ import annotations

from pathlib import Path


class ExtractError(RuntimeError):
    """The file is a known format but its text could not be recovered."""


def _need(module: str, package: str):
    try:
        return __import__(module)
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ExtractError(
            f"reading this file needs the {package} package: pip install {package}"
        ) from exc


def _pdf_pages(path: Path):
    """Yield page text, preferring accuracy over speed.

    pdfplumber lays glyphs out by position; pypdf reads the content stream in
    order. On letter-spaced typography - LaTeX title pages, most theses - pypdf
    produces "PONTIFICI A UNIVERSI DAD", which a model then cannot match against
    anything. Measured on a 992-page book the difference is 12s against 66s,
    which is noise beside the minutes of inference that follow, so the accurate
    one is the default and the fast one is the fallback.
    """
    try:
        import pdfplumber
    except ImportError:
        pdfplumber = None

    if pdfplumber is not None:
        try:
            with pdfplumber.open(str(path)) as pdf:
                for n, page in enumerate(pdf.pages, 1):
                    try:
                        yield n, (page.extract_text() or "").strip()
                    except Exception as exc:
                        yield n, f"[page could not be extracted: {exc}]"
            return
        except Exception:
            pass  # fall through to pypdf, which handles some damaged files better

    _need("pypdf", "pypdf")
    from pypdf import PdfReader

    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            reader.decrypt("")  # many PDFs are "encrypted" with an empty password
    except Exception as exc:
        raise ExtractError(f"cannot open PDF: {exc}") from exc

    for n, page in enumerate(reader.pages, 1):
        try:
            yield n, (page.extract_text() or "").strip()
        except Exception as exc:
            yield n, f"[page could not be extracted: {exc}]"


def _pdf(path: Path) -> str:
    """Text per page, in page order.

    Scanned PDFs are images and yield nothing; say so rather than returning an
    empty string, which reads as 'the document is blank'.
    """
    pages = list(_pdf_pages(path))
    if not pages:
        raise ExtractError("PDF has no pages")
    if not any(body for _, body in pages):
        raise ExtractError(
            f"no text layer in this PDF ({len(pages)} pages) - it is most likely "
            "scanned images and needs OCR first"
        )
    return "\n\n".join(f"[page {n}]\n{body}" for n, body in pages).strip()


def _docx(path: Path) -> str:
    _need("docx", "python-docx")
    import docx

    try:
        doc = docx.Document(str(path))
    except Exception as exc:
        raise ExtractError(f"cannot open .docx: {exc}") from exc

    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    for i, table in enumerate(doc.tables, 1):
        parts.append(f"[table {i}]")
        for row in table.rows:
            parts.append("\t".join(c.text.strip() for c in row.cells))
    return "\n".join(parts).strip()


def _xlsx(path: Path) -> str:
    _need("openpyxl", "openpyxl")
    import openpyxl

    try:
        wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
    except Exception as exc:
        raise ExtractError(f"cannot open .xlsx: {exc}") from exc

    parts: list[str] = []
    try:
        for ws in wb.worksheets:
            parts.append(f"[sheet {ws.title}]")
            for row in ws.iter_rows(values_only=True):
                if row is None or all(c is None for c in row):
                    continue
                parts.append("\t".join("" if c is None else str(c) for c in row))
    finally:
        wb.close()
    return "\n".join(parts).strip()


def _pptx(path: Path) -> str:
    _need("pptx", "python-pptx")
    from pptx import Presentation

    try:
        deck = Presentation(str(path))
    except Exception as exc:
        raise ExtractError(f"cannot open .pptx: {exc}") from exc

    parts: list[str] = []
    for n, slide in enumerate(deck.slides, 1):
        parts.append(f"[slide {n}]")
        for shape in slide.shapes:
            if getattr(shape, "has_text_frame", False) and shape.text_frame.text.strip():
                parts.append(shape.text_frame.text.strip())
    return "\n".join(parts).strip()


def _html(raw: str) -> str:
    """Strip markup so the model reads prose instead of a tag soup."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        # Not worth a hard failure: raw HTML is still readable, just wasteful.
        return raw
    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return "\n".join(
        line.strip() for line in soup.get_text("\n").splitlines() if line.strip()
    )


# Suffix -> extractor. Membership here is what tells the guard a file is worth
# opening despite being binary.
EXTRACTORS = {
    ".pdf": _pdf,
    ".docx": _docx,
    ".xlsx": _xlsx,
    ".xlsm": _xlsx,
    ".pptx": _pptx,
}

# Text on disk, but better read with the markup removed.
MARKUP_SUFFIXES = {".html", ".htm", ".xhtml"}

EXTRACTABLE_SUFFIXES = frozenset(EXTRACTORS) | MARKUP_SUFFIXES


# Extraction is not cheap - up to a minute on a 900-page book - and it happens
# twice: once when the MCP layer sizes the request, once when the handler runs it.
# Keyed on mtime and size so an edited file is re-read. Small, because these
# strings can be megabytes.
_CACHE: dict[tuple, str] = {}
_CACHE_MAX = 4


def _cache_key(path: Path) -> tuple | None:
    try:
        st = path.stat()
    except OSError:
        return None
    return (str(path), st.st_mtime_ns, st.st_size)


def extractable(path: str | Path) -> bool:
    return Path(path).suffix.lower() in EXTRACTABLE_SUFFIXES


def extract(path: str | Path, raw_text: str | None = None) -> str:
    """Return the text of a document. Assumes the path is already guarded.

    ``raw_text`` is the decoded bytes for markup formats, which the caller has
    usually read already; binary formats ignore it and re-open the file.
    """
    target = Path(path)
    suffix = target.suffix.lower()
    if suffix in MARKUP_SUFFIXES:
        return _html(raw_text if raw_text is not None else target.read_text(
            encoding="utf-8", errors="replace"))
    extractor = EXTRACTORS.get(suffix)
    if extractor is None:
        raise ExtractError(f"no extractor for {suffix or 'this file'}")

    key = _cache_key(target)
    if key is not None and key in _CACHE:
        return _CACHE[key]

    text = extractor(target)
    if key is not None:
        if len(_CACHE) >= _CACHE_MAX:
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[key] = text
    if not text.strip():
        raise ExtractError(f"{target.name} contained no extractable text")
    return text
