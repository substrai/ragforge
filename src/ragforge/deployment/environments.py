"""Multi-environment configuration support for RAGForge."""

from __future__ import annotations

import copy
from dataclasses import asdict, fields
from typing import Any, Dict, Optional

from ragforge.core.config import (
    ChunkingConfig,
    CostConfig,
    DataSourceConfig,
    EmbeddingConfig,
    QueryConfig,
    QualityConfig,
    RAGConfig,
    RetrievalConfig,
    StorageConfig,
)


class EnvironmentResolver:
    """Resolves environment-specific configuration overrides.

    Merges base config with environment-specific overrides from the
    `environments:` section of ragforge.yaml.

    Supports dev/staging/prod environments with shallow merge semantics.
    """

    VALID_ENVIRONMENTS = ("dev", "staging", "prod")

    def resolve(self, config: RAGConfig, env_name: str, overrides: Optional[Dict[str, Any]] = None) -> RAGConfig:
        """Resolve configuration for a specific environment.

        Performs a shallow merge of the base config with environment overrides.

        Args:
            config: Base RAGConfig instance.
            env_name: Environment name (dev, staging, prod).
            overrides: Optional dictionary of environment-specific overrides.

        Returns:
            New RAGConfig with environment overrides applied.

        Raises:
            ValueError: If env_name is not a valid environment.
        """
        if env_name not in self.VALID_ENVIRONMENTS:
            raise ValueError(
                f"Invalid environment: {env_name}. "
                f"Valid environments: {', '.join(self.VALID_ENVIRONMENTS)}"
            )

        if not overrides:
            return copy.deepcopy(config)

        # Start with a deep copy of the base config
        resolved = copy.deepcopy(config)

        # Apply overrides using shallow merge
        self._apply_overrides(resolved, overrides)

        return resolved

    def _apply_overrides(self, config: RAGConfig, overrides: Dict[str, Any]) -> None:
        """Apply override dictionary to config object (shallow merge)."""
        if "project_name" in overrides:
            config.project_name = overrides["project_name"]

        if "embedding" in overrides:
            self._merge_dataclass(config.embedding, overrides["embedding"])

        if "storage" in overrides:
            self._merge_dataclass(config.storage, overrides["storage"])

        if "retrieval" in overrides:
            self._merge_dataclass(config.retrieval, overrides["retrieval"])

        if "chunking" in overrides:
            self._merge_dataclass(config.chunking, overrides["chunking"])

        if "query" in overrides:
            self._merge_dataclass(config.query, overrides["query"])

        if "cost" in overrides:
            self._merge_dataclass(config.cost, overrides["cost"])

        if "quality" in overrides:
            self._merge_dataclass(config.quality, overrides["quality"])

        if "data_sources" in overrides:
            # Replace data sources entirely if specified
            config.data_sources = [
                DataSourceConfig(**ds) if isinstance(ds, dict) else ds
                for ds in overrides["data_sources"]
            ]

    def _merge_dataclass(self, target: Any, overrides: Dict[str, Any]) -> None:
        """Shallow merge a dictionary into a dataclass instance."""
        for key, value in overrides.items():
            if hasattr(target, key):
                setattr(target, key, value)
