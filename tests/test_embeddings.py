"""Tests for qdrant_rag.embeddings."""

import pytest
from unittest.mock import MagicMock, patch
from qdrant_rag.embeddings import EmbeddingService


# ---------------------------------------------------------------------------
# OpenAI backend
# ---------------------------------------------------------------------------

def _make_openai_mock(vectors: list[list[float]]):
    """Return a mock OpenAI client whose embeddings.create returns *vectors*."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.data = [MagicMock(embedding=v) for v in vectors]
    mock_client.embeddings.create.return_value = mock_response
    return mock_client


def test_openai_embed(monkeypatch):
    vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    mock_client = _make_openai_mock(vectors)

    with patch("qdrant_rag.embeddings.OpenAI", return_value=mock_client):
        svc = EmbeddingService(model_name="openai", openai_api_key="test-key")
        result = svc.embed(["hello", "world"])

    assert result == vectors
    mock_client.embeddings.create.assert_called_once()


def test_openai_embed_no_key():
    svc = EmbeddingService(model_name="openai", openai_api_key=None)
    with pytest.raises(ValueError, match="openai_api_key must be set"):
        svc.embed(["hello"])


def test_openai_dimension_small():
    svc = EmbeddingService(
        model_name="openai",
        openai_api_key="key",
        openai_embedding_model="text-embedding-3-small",
    )
    assert svc.dimension == 1536


def test_openai_dimension_large():
    svc = EmbeddingService(
        model_name="openai",
        openai_api_key="key",
        openai_embedding_model="text-embedding-3-large",
    )
    assert svc.dimension == 3072


# ---------------------------------------------------------------------------
# Sentence-transformers backend
# ---------------------------------------------------------------------------

def _make_st_mock(dim: int = 4):
    mock_model = MagicMock()
    import numpy as np
    mock_model.encode.return_value = np.array([[0.1] * dim, [0.2] * dim])
    mock_model.get_sentence_embedding_dimension.return_value = dim
    return mock_model


def test_st_embed(monkeypatch):
    dim = 4
    mock_model = _make_st_mock(dim)

    with patch("qdrant_rag.embeddings.SentenceTransformer", return_value=mock_model):
        svc = EmbeddingService(model_name="all-MiniLM-L6-v2")
        result = svc.embed(["text one", "text two"])

    assert len(result) == 2
    assert len(result[0]) == dim


def test_st_dimension(monkeypatch):
    mock_model = _make_st_mock(dim=384)
    with patch("qdrant_rag.embeddings.SentenceTransformer", return_value=mock_model):
        svc = EmbeddingService(model_name="all-MiniLM-L6-v2")
        assert svc.dimension == 384


def test_embed_empty_list():
    svc = EmbeddingService(model_name="openai", openai_api_key="key")
    assert svc.embed([]) == []
