"""RAGForge configuration parser and validator."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class DataSourceConfig:
    """Configuration for a single data source."""

    name: str
    type: str  # s3 | dynamodb | api | confluence | local
    config: Dict[str, Any] = field(default_factory=dict)
    file_types: List[str] = field(default_factory=lambda: ["pdf", "md", "txt"])
    update_frequency: str = "daily"
    access_control: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChunkingConfig:
    """Configuration for chunking strategy."""

    strategy: str = "auto"  # auto | recursive | semantic | sentence | fixed
    max_chunk_size: int = 512
    min_chunk_size: int = 50
    overlap: int = 50
    metadata_extraction: Dict[str, bool] = field(
        default_factory=lambda: {"title": True, "date": True, "author": True}
    )
    strategies: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding model."""

    model: str = "bedrock/titan-embed-v2"
    dimensions: int = 1024
    batch_size: int = 100
    quantization: str = "float32"  # float32 | float16 | int8


@dataclass
class StorageConfig:
    """Configuration for vector store."""

    provider: str = "opensearch-serverless"  # opensearch-serverless | faiss-lambda | pgvector
    index_name: str = "ragforge-index"
    replicas: int = 1
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalConfig:
    """Configuration for retrieval strategy."""

    method: str = "hybrid"  # semantic | keyword | hybrid
    semantic_weight: float = 0.7
    keyword_weight: float = 0.3
    top_k: int = 5
    reranking: Dict[str, Any] = field(
        default_factory=lambda: {"enabled": False, "model": "bedrock/cohere-rerank", "top_n": 3}
    )


@dataclass
class QueryConfig:
    """Configuration for query endpoint."""

    endpoint: bool = True
    timeout_ms: int = 3000
    cache: Dict[str, Any] = field(
        default_factory=lambda: {"enabled": True, "ttl_seconds": 300}
    )


@dataclass
class QualityConfig:
    """Configuration for quality monitoring."""

    evaluation: Dict[str, Any] = field(default_factory=dict)
    drift_detection: Dict[str, Any] = field(default_factory=dict)
    feedback: Dict[str, Any] = field(default_factory=dict)
    alerts: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class CostConfig:
    """Configuration for cost management."""

    budget: Dict[str, float] = field(default_factory=lambda: {"monthly_total": 100.0})
    optimization: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGConfig:
    """Complete RAGForge configuration."""

    project_name: str = "ragforge-project"
    project_version: str = "1.0.0"
    data_sources: List[DataSourceConfig] = field(default_factory=list)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    query: QueryConfig = field(default_factory=QueryConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    cost: CostConfig = field(default_factory=CostConfig)

    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []

        if not self.data_sources:
            errors.append("At least one data source must be configured")

        for ds in self.data_sources:
            if not ds.name:
                errors.append("Data source name is required")
            if ds.type not in ("s3", "dynamodb", "api", "confluence", "notion", "github", "local"):
                errors.append(f"Invalid data source type: {ds.type}")

        if self.chunking.max_chunk_size < self.chunking.min_chunk_size:
            errors.append("max_chunk_size must be >= min_chunk_size")

        if self.chunking.overlap >= self.chunking.max_chunk_size:
            errors.append("overlap must be < max_chunk_size")

        if self.embedding.dimensions < 1:
            errors.append("embedding dimensions must be positive")

        if self.retrieval.top_k < 1:
            errors.append("retrieval top_k must be >= 1")

        if self.retrieval.method == "hybrid":
            total_weight = self.retrieval.semantic_weight + self.retrieval.keyword_weight
            if abs(total_weight - 1.0) > 0.01:
                errors.append(
                    f"Hybrid retrieval weights must sum to 1.0, got {total_weight}"
                )

        valid_strategies = ("auto", "recursive", "semantic", "sentence", "fixed", "adaptive", "code_aware", "table", "hierarchical")
        if self.chunking.strategy not in valid_strategies:
            errors.append(f"Invalid chunking strategy: {self.chunking.strategy}")

        return errors


def load_config(path: str | Path = "ragforge.yaml") -> RAGConfig:
    """Load and validate RAGForge configuration from YAML file."""
    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    if not raw:
        raise ValueError("Configuration file is empty")

    # Parse project info
    project = raw.get("project", {})
    project_name = project.get("name", "ragforge-project")
    project_version = project.get("version", "1.0.0")

    # Parse data sources
    data_sources = []
    for ds_raw in raw.get("data_sources", []):
        ds = DataSourceConfig(
            name=ds_raw.get("name", ""),
            type=ds_raw.get("type", "s3"),
            config=ds_raw.get("config", {}),
            file_types=ds_raw.get("file_types", ["pdf", "md", "txt"]),
            update_frequency=ds_raw.get("update_frequency", "daily"),
            access_control=ds_raw.get("access_control", {}),
        )
        data_sources.append(ds)

    # Parse chunking
    chunking_raw = raw.get("chunking", {})
    chunking = ChunkingConfig(
        strategy=chunking_raw.get("strategy", "auto"),
        max_chunk_size=chunking_raw.get("max_chunk_size", 512),
        min_chunk_size=chunking_raw.get("min_chunk_size", 50),
        overlap=chunking_raw.get("overlap", 50),
        metadata_extraction=chunking_raw.get("metadata_extraction", {}),
        strategies=chunking_raw.get("strategies", {}),
    )

    # Parse embedding
    embedding_raw = raw.get("embedding", {})
    embedding = EmbeddingConfig(
        model=embedding_raw.get("model", "bedrock/titan-embed-v2"),
        dimensions=embedding_raw.get("dimensions", 1024),
        batch_size=embedding_raw.get("batch_size", 100),
        quantization=embedding_raw.get("quantization", "float32"),
    )

    # Parse storage
    storage_raw = raw.get("storage", {})
    storage = StorageConfig(
        provider=storage_raw.get("provider", "opensearch-serverless"),
        index_name=storage_raw.get("index_name", "ragforge-index"),
        replicas=storage_raw.get("replicas", 1),
        config=storage_raw.get("config", {}),
    )

    # Parse retrieval
    retrieval_raw = raw.get("retrieval", {})
    retrieval = RetrievalConfig(
        method=retrieval_raw.get("method", "hybrid"),
        semantic_weight=retrieval_raw.get("semantic_weight", 0.7),
        keyword_weight=retrieval_raw.get("keyword_weight", 0.3),
        top_k=retrieval_raw.get("top_k", 5),
        reranking=retrieval_raw.get("reranking", {"enabled": False}),
    )

    # Parse query
    query_raw = raw.get("query", {})
    query = QueryConfig(
        endpoint=query_raw.get("endpoint", True),
        timeout_ms=query_raw.get("timeout_ms", 3000),
        cache=query_raw.get("cache", {"enabled": True, "ttl_seconds": 300}),
    )

    # Parse quality
    quality_raw = raw.get("quality", {})
    quality = QualityConfig(
        evaluation=quality_raw.get("evaluation", {}),
        drift_detection=quality_raw.get("drift_detection", {}),
        feedback=quality_raw.get("feedback", {}),
        alerts=quality_raw.get("alerts", []),
    )

    # Parse cost
    cost_raw = raw.get("cost", {})
    cost = CostConfig(
        budget=cost_raw.get("budget", {"monthly_total": 100.0}),
        optimization=cost_raw.get("optimization", {}),
    )

    config = RAGConfig(
        project_name=project_name,
        project_version=project_version,
        data_sources=data_sources,
        chunking=chunking,
        embedding=embedding,
        storage=storage,
        retrieval=retrieval,
        query=query,
        quality=quality,
        cost=cost,
    )

    errors = config.validate()
    if errors:
        raise ValueError(f"Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

    return config
