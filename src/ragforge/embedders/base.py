"""Base embedder interface for RAGForge."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class EmbeddingResult:
    """Result of an embedding operation."""

    embedding: List[float]
    token_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseEmbedder(ABC):
    """Abstract base class for embedding models.

    Embedders convert text into dense vector representations suitable
    for similarity search in vector stores.
    """

    def __init__(self, dimensions: int = 1024):
        self.dimensions = dimensions

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Embed a single text string.

        Args:
            text: The text to embed.

        Returns:
            A list of floats representing the embedding vector.
        """
        ...

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of text strings.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        ...
