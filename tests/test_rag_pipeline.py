"""Tests for qdrant_rag.rag_pipeline."""

import pytest
from unittest.mock import MagicMock, patch
from qdrant_rag.config import Config
from qdrant_rag.rag_pipeline import RAGPipeline, RAGResult


def _pipeline_with_mocks(
    embed_return=None,
    search_return=None,
    generate_return="Test answer.",
    dimension=3,
):
    """Build a RAGPipeline with all external dependencies mocked out."""
    if embed_return is None:
        embed_return = [[0.1, 0.2, 0.3]]
    if search_return is None:
        search_return = [
            {"text": "chunk text", "score": 0.9, "metadata": {"source": "a.txt"}}
        ]

    config = Config()
    config.openai_api_key = "test-key"

    mock_embedding_svc = MagicMock()
    mock_embedding_svc.embed.return_value = embed_return
    mock_embedding_svc.dimension = dimension

    mock_vector_store = MagicMock()
    mock_vector_store.search.return_value = search_return

    mock_openai_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices[0].message.content = generate_return
    mock_openai_client.chat.completions.create.return_value = mock_response

    with (
        patch("qdrant_rag.rag_pipeline.EmbeddingService", return_value=mock_embedding_svc),
        patch("qdrant_rag.rag_pipeline.VectorStore", return_value=mock_vector_store),
    ):
        pipeline = RAGPipeline(config=config)

    # Store mocks for assertion in tests
    pipeline._embedding_service = mock_embedding_svc
    pipeline._vector_store = mock_vector_store
    pipeline._mock_openai = mock_openai_client
    return pipeline


# ---------------------------------------------------------------------------
# query
# ---------------------------------------------------------------------------

def test_query_returns_rag_result():
    pipeline = _pipeline_with_mocks(generate_return="The answer is 42.")

    with patch("qdrant_rag.rag_pipeline.OpenAI", return_value=pipeline._mock_openai):
        result = pipeline.query("What is the answer?")

    assert isinstance(result, RAGResult)
    assert result.answer == "The answer is 42."
    assert result.query == "What is the answer?"
    assert len(result.sources) == 1


def test_query_calls_embed_once():
    pipeline = _pipeline_with_mocks()
    with patch("qdrant_rag.rag_pipeline.OpenAI", return_value=pipeline._mock_openai):
        pipeline.query("anything")
    pipeline._embedding_service.embed.assert_called_once_with(["anything"])


def test_query_no_openai_key():
    pipeline = _pipeline_with_mocks()
    pipeline.config.openai_api_key = None

    with pytest.raises(ValueError, match="OPENAI_API_KEY must be set"):
        pipeline.query("hello")


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------

def test_ingest_file(tmp_path):
    txt_file = tmp_path / "sample.txt"
    txt_file.write_text("Hello world! " * 20)

    pipeline = _pipeline_with_mocks(embed_return=[[0.1, 0.2, 0.3]] * 10)
    pipeline._vector_store.ensure_collection = MagicMock()
    pipeline._vector_store.upsert = MagicMock()

    count = pipeline.ingest_file(str(txt_file))
    assert count > 0
    pipeline._vector_store.ensure_collection.assert_called_once()
    pipeline._vector_store.upsert.assert_called_once()


def test_ingest_empty_file(tmp_path):
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("")

    pipeline = _pipeline_with_mocks()
    count = pipeline.ingest_file(str(empty_file))
    assert count == 0
    pipeline._vector_store.ensure_collection.assert_not_called()
