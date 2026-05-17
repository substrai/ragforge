"""Tests for config loading and validation."""

import tempfile
from pathlib import Path

import pytest

from ragforge.core.config import (
    ChunkingConfig,
    EmbeddingConfig,
    RAGConfig,
    RetrievalConfig,
    StorageConfig,
    load_config,
)


class TestRAGConfig:
    """Tests for RAGConfig validation."""

    def test_valid_config_no_errors(self):
        """A valid config should produce no validation errors."""
        config = RAGConfig(
            data_sources=[
                type("DS", (), {"name": "test", "type": "local", "config": {}, "file_types": ["txt"]})()
            ],
        )
        # Manually create a proper DataSourceConfig
        from ragforge.core.config import DataSourceConfig

        config.data_sources = [DataSourceConfig(name="test", type="local")]
        errors = config.validate()
        assert errors == []

    def test_empty_data_sources_error(self):
        """Empty data sources should produce a validation error."""
        config = RAGConfig(data_sources=[])
        errors = config.validate()
        assert any("data source" in e.lower() for e in errors)

    def test_invalid_data_source_type(self):
        """Invalid data source type should produce an error."""
        from ragforge.core.config import DataSourceConfig

        config = RAGConfig(
            data_sources=[DataSourceConfig(name="test", type="invalid_type")]
        )
        errors = config.validate()
        assert any("invalid data source type" in e.lower() for e in errors)

    def test_chunk_size_validation(self):
        """max_chunk_size < min_chunk_size should produce an error."""
        from ragforge.core.config import DataSourceConfig

        config = RAGConfig(
            data_sources=[DataSourceConfig(name="test", type="local")],
            chunking=ChunkingConfig(max_chunk_size=10, min_chunk_size=100),
        )
        errors = config.validate()
        assert any("max_chunk_size" in e for e in errors)

    def test_overlap_exceeds_chunk_size(self):
        """Overlap >= max_chunk_size should produce an error."""
        from ragforge.core.config import DataSourceConfig

        config = RAGConfig(
            data_sources=[DataSourceConfig(name="test", type="local")],
            chunking=ChunkingConfig(max_chunk_size=100, overlap=100),
        )
        errors = config.validate()
        assert any("overlap" in e for e in errors)

    def test_hybrid_weights_must_sum_to_one(self):
        """Hybrid retrieval weights must sum to 1.0."""
        from ragforge.core.config import DataSourceConfig

        config = RAGConfig(
            data_sources=[DataSourceConfig(name="test", type="local")],
            retrieval=RetrievalConfig(
                method="hybrid", semantic_weight=0.5, keyword_weight=0.3
            ),
        )
        errors = config.validate()
        assert any("weights" in e.lower() for e in errors)

    def test_invalid_chunking_strategy(self):
        """Invalid chunking strategy should produce an error."""
        from ragforge.core.config import DataSourceConfig

        config = RAGConfig(
            data_sources=[DataSourceConfig(name="test", type="local")],
            chunking=ChunkingConfig(strategy="nonexistent"),
        )
        errors = config.validate()
        assert any("strategy" in e.lower() for e in errors)


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_valid_yaml(self, tmp_path):
        """Should load a valid YAML config file."""
        config_content = """\
project:
  name: test-project
  version: "1.0.0"

data_sources:
  - name: local-docs
    type: local
    config:
      path: ./data
    file_types: [md, txt]

chunking:
  strategy: recursive
  max_chunk_size: 512
  overlap: 50

embedding:
  model: local/dev
  dimensions: 384

storage:
  provider: faiss
  index_name: test-index

retrieval:
  method: hybrid
  semantic_weight: 0.7
  keyword_weight: 0.3
  top_k: 5
"""
        config_file = tmp_path / "ragforge.yaml"
        config_file.write_text(config_content)

        config = load_config(config_file)

        assert config.project_name == "test-project"
        assert config.chunking.strategy == "recursive"
        assert config.embedding.dimensions == 384
        assert config.retrieval.semantic_weight == 0.7
        assert len(config.data_sources) == 1

    def test_file_not_found(self):
        """Should raise FileNotFoundError for missing config."""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/ragforge.yaml")

    def test_empty_config_raises(self, tmp_path):
        """Should raise ValueError for empty config file."""
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("")

        with pytest.raises(ValueError, match="empty"):
            load_config(config_file)

    def test_validation_errors_raise(self, tmp_path):
        """Should raise ValueError when validation fails."""
        config_content = """\
project:
  name: bad-project

chunking:
  strategy: recursive
  max_chunk_size: 10
  min_chunk_size: 100
"""
        config_file = tmp_path / "bad.yaml"
        config_file.write_text(config_content)

        with pytest.raises(ValueError, match="validation failed"):
            load_config(config_file)
