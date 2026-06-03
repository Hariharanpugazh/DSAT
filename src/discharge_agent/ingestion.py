from __future__ import annotations

import hashlib
from pathlib import Path

import fitz

from .models import DocumentPage


def inspect_pdfs(paths: list[Path]) -> list[DocumentPage]:
    pages: list[DocumentPage] = []
    for path in paths:
        document_id = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:12]
        with fitz.open(path) as document:
            for index, page in enumerate(document):
                text = page.get_text("text").strip()
                pages.append(
                    DocumentPage(
                        document_id=document_id,
                        source_file=path.name,
                        page_number=index + 1,
                        text=text,
                        extraction_method="embedded_text" if text else "none",
                    )
                )
    return pages


def render_page(path: Path, page_number: int, dpi: int = 150) -> bytes:
    with fitz.open(path) as document:
        page = document[page_number - 1]
        pixmap = page.get_pixmap(dpi=dpi, alpha=False)
        return pixmap.tobytes("png")

