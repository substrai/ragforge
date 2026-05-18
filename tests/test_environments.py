"""Tests for multi-environment configuration resolution."""

import pytest

from ragforge.core.config import (
    RAGConfig,
    DataSourceConfig,
    ChunkingConfig,
    EmbeddingConfig,
    StorageConfig,
    RetrievalConfig,
    QueryConfig,
)
from ragforge.deployment.environments import EnvironmentResolver


class TestEnvironmentResolver:
    """Tests for EnvironmentResolver class."""

    def _create_base_config(self) -> RAGConfig:
        """Create a base test config."""
        return RAGConfig(
            project_name="test-project",
            data_sources=[
                DataSourceConfig(name="docs", type="s3", update_frequency="daily")
            ],
            chunking=ChunkingConfig(strategy="auto", max_chunk_size=512),
            embedding=EmbeddingConfig(model="bedrock/titan-embed-v2", dimensions=1024),
            storage=StorageConfig(provider="opensearch-serverless", index_name="test-index"),
            retrieval=RetrievalConfig(method="hybrid", top_k=5),
            query=QueryConfig(timeout_ms=3000),
        )

    def test_resolve_without_overrides(self):
        """Test resolving with no overrides returns copy of base config."""
        resolver = EnvironmentResolver()
        config = self._create_base_config()

        resolved = resolver.resolve(config, "dev")

        assert resolved.project_name == "test-project"
        assert resolved.embedding.model == "bedrock/titan-embed-v2"
        # Should be a different object (deep copy)
        assert resolved is not config

    def test_resolve_with_embedding_override(self):
        """Test overriding embedding config."""
        resolver = EnvironmentResolver()
        config = self._create_base_config()

        overrides = {
            "embedding": {
                "model": "local/dev",
                "dimensions": 384,
            }
        }

        resolved = resolver.resolve(config, "dev", overrides)

        assert resolved.embedding.model == "local/dev"
        assert resolved.embedding.dimensions == 384
        # Original should be unchanged
        assert config.embedding.model == "bedrock/titan-embed-v2"

    def test_resolve_with_storage_override(self):
        """Test overriding storage config."""
        resolver = EnvironmentResolver()
        config = self._create_base_config()

        overrides = {
            "storage": {
                "provider": "faiss",
                "index_name": "dev-index",
            }
        }

        resolved = resolver.resolve(config, "staging", overrides)

        assert resolved.storage.provider == "faiss"
        assert resolved.storage.index_name == "dev-index"

    def test_resolve_with_retrieval_override(self):
        """Test overriding retrieval config."""
        resolver = EnvironmentResolver()
        config = self._create_base_config()

        overrides = {
            "retrieval": {
                "top_k": 10,
                "method": "semantic",
            }
        }

        resolved = resolver.resolve(config, "prod", overrides)

        assert resolved.retrieval.top_k == 10
        assert resolved.retrieval.method == "semantic"
        # Unmodified fields should remain
        assert resolved.retrieval.semantic_weight == 0.7

    def test_resolve_with_project_name_override(self):
        """Test overriding project name."""
        resolver = EnvironmentResolver()
        config = self._create_base_config()

        overrides = {"project_name": "test-project-prod"}

        resolved = resolver.resolve(config, "prod", overrides)

        assert resolved.project_name == "test-project-prod"

    def test_resolve_invalid_environment(self):
        """Test that invalid environment raises ValueError."""
        resolver = EnvironmentResolver()
        config = self._create_base_config()

        with pytest.raises(ValueError, match="Invalid environment"):
            resolver.resolve(config, "invalid_env")

    def test_resolve_valid_environments(self):
        """Test all valid environment names."""
        resolver = EnvironmentResolver()
        config = self._create_base_config()

        for env in ("dev", "staging", "prod"):
            resolved = resolver.resolve(config, env)
            assert resolved is not None

    def test_resolve_with_data_sources_override(self):
        """Test overriding data sources entirely."""
        resolver = EnvironmentResolver()
        config = self._create_base_config()

        overrides = {
            "data_sources": [
                {"name": "dev-docs", "type": "local", "config": {"path": "./dev-data"}},
            ]
        }

        resolved = resolver.resolve(config, "dev", overrides)

        assert len(resolved.data_sources) == 1
        assert resolved.data_sources[0].name == "dev-docs"
        assert resolved.data_sources[0].type == "local"

    def test_resolve_with_query_override(self):
        """Test overriding query config."""
        resolver = EnvironmentResolver()
        config = self._create_base_config()

        overrides = {
            "query": {
                "timeout_ms": 10000,
            }
        }

        resolved = resolver.resolve(config, "prod", overrides)

        assert resolved.query.timeout_ms == 10000

    def test_resolve_does_not_mutate_original(self):
        """Test that resolving does not mutate the original config."""
        resolver = EnvironmentResolver()
        config = self._create_base_config()

        overrides = {
            "embedding": {"model": "local/dev"},
            "retrieval": {"top_k": 20},
        }

        resolver.resolve(config, "dev", overrides)

        # Original should be unchanged
        assert config.embedding.model == "bedrock/titan-embed-v2"
        assert config.retrieval.top_k == 5

    def test_resolve_shallow_merge(self):
        """Test that merge is shallow (replaces nested dicts)."""
        resolver = EnvironmentResolver()
        config = self._create_base_config()

        overrides = {
            "chunking": {
                "max_chunk_size": 1024,
            }
        }

        resolved = resolver.resolve(config, "dev", overrides)

        # Overridden field
        assert resolved.chunking.max_chunk_size == 1024
        # Non-overridden fields should remain from base
        assert resolved.chunking.strategy == "auto"
