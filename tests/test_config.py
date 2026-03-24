"""Tests for qdrant_rag.config."""

import os
import pytest
from qdrant_rag.config import Config


def test_defaults():
    """Config should provide sensible defaults without any env vars."""
    # Temporarily clear env vars that might be set in the test environment
    keys_to_clear = [
        "QDRANT_URL", "QDRANT_API_KEY", "OPENAI_API_KEY", "OPENAI_MODEL",
        "EMBEDDING_MODEL", "QDRANT_COLLECTION", "CHUNK_SIZE", "CHUNK_OVERLAP", "TOP_K",
    ]
    saved = {k: os.environ.pop(k, None) for k in keys_to_clear}
    try:
        config = Config()
        assert config.qdrant_url == "http://localhost:6333"
        assert config.qdrant_api_key is None
        assert config.openai_api_key is None
        assert config.openai_model == "gpt-4o-mini"
        assert config.embedding_model == "all-MiniLM-L6-v2"
        assert config.collection_name == "rag_documents"
        assert config.chunk_size == 500
        assert config.chunk_overlap == 50
        assert config.top_k == 5
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


def test_env_overrides(monkeypatch):
    """Values set via env vars should be picked up by Config."""
    monkeypatch.setenv("QDRANT_URL", "http://qdrant:6333")
    monkeypatch.setenv("QDRANT_COLLECTION", "my_collection")
    monkeypatch.setenv("CHUNK_SIZE", "1000")
    monkeypatch.setenv("CHUNK_OVERLAP", "100")
    monkeypatch.setenv("TOP_K", "10")

    config = Config()
    assert config.qdrant_url == "http://qdrant:6333"
    assert config.collection_name == "my_collection"
    assert config.chunk_size == 1000
    assert config.chunk_overlap == 100
    assert config.top_k == 10
