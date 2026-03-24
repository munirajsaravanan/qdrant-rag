"""Tests for qdrant_rag.vector_store."""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from qdrant_rag.document_loader import Document
from qdrant_rag.vector_store import VectorStore


def _make_store(mock_client: MagicMock) -> VectorStore:
    with patch("qdrant_rag.vector_store.QdrantClient", return_value=mock_client):
        store = VectorStore(
            url="http://localhost:6333",
            collection_name="test_col",
            vector_dimension=3,
        )
    return store


# ---------------------------------------------------------------------------
# ensure_collection
# ---------------------------------------------------------------------------

def test_ensure_collection_creates_when_missing():
    mock_client = MagicMock()
    mock_client.get_collections.return_value.collections = []  # no collections
    store = _make_store(mock_client)
    store.ensure_collection()
    mock_client.create_collection.assert_called_once()


def test_ensure_collection_skips_when_exists():
    mock_client = MagicMock()
    existing = MagicMock()
    existing.name = "test_col"
    mock_client.get_collections.return_value.collections = [existing]
    store = _make_store(mock_client)
    store.ensure_collection()
    mock_client.create_collection.assert_not_called()


# ---------------------------------------------------------------------------
# upsert
# ---------------------------------------------------------------------------

def test_upsert_calls_client():
    mock_client = MagicMock()
    store = _make_store(mock_client)

    docs = [Document(text="hello", metadata={"source": "a.txt"})]
    embeddings = [[0.1, 0.2, 0.3]]
    store.upsert(docs, embeddings)
    mock_client.upsert.assert_called_once()


def test_upsert_length_mismatch_raises():
    mock_client = MagicMock()
    store = _make_store(mock_client)

    docs = [Document(text="hello")]
    embeddings = [[0.1, 0.2], [0.3, 0.4]]
    with pytest.raises(ValueError, match="same length"):
        store.upsert(docs, embeddings)


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

def test_search_returns_hits():
    mock_client = MagicMock()
    hit = MagicMock()
    hit.score = 0.95
    hit.payload = {"text": "relevant chunk", "source": "doc.txt", "chunk_index": 0}
    mock_client.search.return_value = [hit]

    store = _make_store(mock_client)
    results = store.search(query_vector=[0.1, 0.2, 0.3], top_k=1)

    assert len(results) == 1
    assert results[0]["text"] == "relevant chunk"
    assert results[0]["score"] == 0.95
    assert results[0]["metadata"]["source"] == "doc.txt"


def test_search_empty():
    mock_client = MagicMock()
    mock_client.search.return_value = []
    store = _make_store(mock_client)
    results = store.search(query_vector=[0.1, 0.2, 0.3])
    assert results == []


# ---------------------------------------------------------------------------
# count
# ---------------------------------------------------------------------------

def test_count():
    mock_client = MagicMock()
    mock_client.count.return_value.count = 42
    store = _make_store(mock_client)
    assert store.count() == 42
