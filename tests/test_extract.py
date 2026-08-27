"""Document extraction, and the boundary it must not weaken.

PDFs, .docx and .xlsx used to be refused as binary, which pushed the extraction
work back onto the caller - exactly the bulk-text handling this project exists to
absorb. Making them readable must not make them a way around the sandbox.
"""

from __future__ import annotations

import zipfile

import pytest

from llmhelper import guard
from llmhelper.extract import ExtractError, extract, extractable


def test_document_formats_are_extractable_and_opaque_ones_are_not():
    assert extractable("paper.pdf")
    assert extractable("notes.DOCX"), "suffix matching must be case-insensitive"
    assert extractable("book.xlsx")
    assert extractable("page.html")
    assert not extractable("model.gguf")
    assert not extractable("photo.png")
    assert not extractable("notes.txt"), "plain text needs no extractor"


def test_extractable_files_are_not_treated_as_binary(tmp_path):
    """looks_binary gates whether expand() will even offer the file."""
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4\x00 not really a pdf")
    assert guard.looks_binary(pdf) is False

    blob = tmp_path / "x.gguf"
    blob.write_bytes(b"\x00\x01\x02")
    assert guard.looks_binary(blob) is True


def test_html_extraction_strips_markup_and_scripts():
    html = "<html><head><style>p{color:red}</style></head><body>" \
           "<script>steal()</script><h1>Title</h1><p>Body text.</p></body></html>"
    out = extract("page.html", html)
    assert "Title" in out and "Body text." in out
    assert "steal()" not in out, "script contents are noise, not prose"
    assert "<" not in out


def test_a_corrupt_document_raises_rather_than_returning_mojibake(tmp_path):
    """Silently handing the model garbage bytes wastes a minute of GPU time."""
    bad = tmp_path / "broken.docx"
    bad.write_bytes(b"this is not a zip archive at all")
    with pytest.raises(ExtractError):
        extract(bad)


def test_an_empty_document_is_reported_not_returned_blank(tmp_path):
    empty = tmp_path / "empty.docx"
    with zipfile.ZipFile(empty, "w") as z:
        z.writestr("[Content_Types].xml", "<Types/>")
    with pytest.raises(ExtractError):
        extract(empty)


def test_unsupported_suffix_is_refused_by_name():
    with pytest.raises(ExtractError, match="no extractor"):
        extract("archive.zip")


# ------------------------------------------------------- the boundary holds


def test_a_pdf_outside_the_read_roots_is_still_refused(sandbox, tmp_path):
    """Extraction runs after resolve_read, never instead of it."""
    outside = tmp_path / "elsewhere.pdf"
    outside.write_bytes(b"%PDF-1.4 whatever")
    with pytest.raises(guard.GuardError):
        guard.read_text(outside, cfg=sandbox)


def test_a_denied_document_inside_a_root_is_still_refused(sandbox):
    root = sandbox.paths.read_roots[0]
    denied = root / "project" / "__pycache__" / "notes.docx"
    denied.write_bytes(b"PK\x03\x04")
    with pytest.raises(guard.GuardError):
        guard.read_text(denied, cfg=sandbox)


def test_plain_text_reading_is_unchanged(sandbox):
    root = sandbox.paths.read_roots[0]
    got = guard.read_text(root / "project" / "readme.txt", cfg=sandbox)
    assert got.strip() == "hello from the sandbox"
