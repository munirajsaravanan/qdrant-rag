"""Tests for qdrant_rag.document_loader."""

import os
import tempfile
import pytest
from qdrant_rag.document_loader import Document, DocumentLoader


# ---------------------------------------------------------------------------
# DocumentLoader construction
# ---------------------------------------------------------------------------

def test_invalid_overlap_raises():
    with pytest.raises(ValueError, match="chunk_overlap must be smaller than chunk_size"):
        DocumentLoader(chunk_size=100, chunk_overlap=100)


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------

def test_chunk_text_exact_fit():
    loader = DocumentLoader(chunk_size=10, chunk_overlap=0)
    chunks = list(loader._chunk_text("1234567890"))
    assert chunks == ["1234567890"]


def test_chunk_text_multiple_chunks_no_overlap():
    loader = DocumentLoader(chunk_size=5, chunk_overlap=0)
    chunks = list(loader._chunk_text("ABCDEFGHIJ"))
    assert chunks == ["ABCDE", "FGHIJ"]


def test_chunk_text_with_overlap():
    loader = DocumentLoader(chunk_size=5, chunk_overlap=2)
    # "ABCDE", then step = 5-2=3, next start=3 → "DEFGH", step 3, start=6 → "GHIJ"
    chunks = list(loader._chunk_text("ABCDEFGHIJ"))
    assert chunks[0] == "ABCDE"
    assert chunks[1] == "DEFGH"
    assert chunks[2] == "GHIJ"


def test_chunk_text_empty():
    loader = DocumentLoader()
    assert list(loader._chunk_text("")) == []


# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------

def test_load_txt_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as fh:
        fh.write("Hello World " * 50)
        tmp_path = fh.name

    try:
        loader = DocumentLoader(chunk_size=50, chunk_overlap=0)
        docs = loader.load_file(tmp_path)
        assert len(docs) > 0
        assert all(isinstance(d, Document) for d in docs)
        # All chunks together should reconstruct the original text
        reconstructed = "".join(d.text for d in docs)
        assert reconstructed == "Hello World " * 50
    finally:
        os.unlink(tmp_path)


def test_load_nonexistent_file():
    loader = DocumentLoader()
    with pytest.raises(FileNotFoundError):
        loader.load_file("/nonexistent/path/file.txt")


def test_load_unsupported_extension():
    loader = DocumentLoader()
    with pytest.raises(ValueError, match="Unsupported file type"):
        loader.load_file("/tmp/not_supported.xyz")


def test_load_file_metadata():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as fh:
        fh.write("x" * 10)
        tmp_path = fh.name

    try:
        loader = DocumentLoader(chunk_size=5, chunk_overlap=0)
        docs = loader.load_file(tmp_path)
        assert docs[0].metadata["chunk_index"] == 0
        assert docs[0].metadata["total_chunks"] == len(docs)
        assert docs[0].metadata["source"] == tmp_path
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Directory loading
# ---------------------------------------------------------------------------

def test_load_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(3):
            path = os.path.join(tmpdir, f"file{i}.txt")
            with open(path, "w") as fh:
                fh.write(f"Content of file {i}. " * 10)

        loader = DocumentLoader(chunk_size=50, chunk_overlap=0)
        docs = loader.load_directory(tmpdir)
        assert len(docs) > 0
        sources = {d.metadata["filename"] for d in docs}
        assert sources == {"file0.txt", "file1.txt", "file2.txt"}


def test_load_directory_not_a_dir():
    loader = DocumentLoader()
    with pytest.raises(NotADirectoryError):
        loader.load_directory("/nonexistent/directory")
