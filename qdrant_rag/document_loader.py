"""Document loading and text-chunking utilities."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


@dataclass
class Document:
    """A single chunk of text with metadata."""

    text: str
    metadata: dict = field(default_factory=dict)


class DocumentLoader:
    """Load plain-text and PDF files and split them into overlapping chunks.

    Parameters
    ----------
    chunk_size:
        Maximum number of characters per chunk.
    chunk_overlap:
        Number of characters that consecutive chunks share.
    """

    SUPPORTED_EXTENSIONS = {".txt", ".md", ".rst", ".pdf"}

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_file(self, path: str | Path) -> list[Document]:
        """Load a single file and return its chunks."""
        path = Path(path)

        ext = path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type '{ext}'. "
                f"Supported types: {sorted(self.SUPPORTED_EXTENSIONS)}"
            )

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        text = self._read_file(path)
        chunks = list(self._chunk_text(text))
        return [
            Document(
                text=chunk,
                metadata={
                    "source": str(path),
                    "filename": path.name,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                },
            )
            for i, chunk in enumerate(chunks)
        ]

    def load_directory(self, directory: str | Path) -> list[Document]:
        """Recursively load all supported files from *directory*."""
        directory = Path(directory)
        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        documents: list[Document] = []
        for root, _, files in os.walk(directory):
            for filename in sorted(files):
                filepath = Path(root) / filename
                if filepath.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                    documents.extend(self.load_file(filepath))
        return documents

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_file(self, path: Path) -> str:
        if path.suffix.lower() == ".pdf":
            return self._read_pdf(path)
        return path.read_text(encoding="utf-8", errors="replace")

    @staticmethod
    def _read_pdf(path: Path) -> str:
        try:
            import PyPDF2  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "PyPDF2 is required for PDF support. Install it with: pip install PyPDF2"
            ) from exc

        pages: list[str] = []
        with open(path, "rb") as fh:
            reader = PyPDF2.PdfReader(fh)
            for page in reader.pages:
                text = page.extract_text() or ""
                pages.append(text)
        return "\n".join(pages)

    def _chunk_text(self, text: str) -> Iterator[str]:
        """Yield overlapping character-level chunks."""
        if not text:
            return

        start = 0
        while start < len(text):
            end = start + self.chunk_size
            yield text[start:end]
            if end >= len(text):
                break
            start += self.chunk_size - self.chunk_overlap
