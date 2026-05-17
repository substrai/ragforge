"""Base vector store interface for RAGForge."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SearchResult:
    """A single search result from the vector store."""

    chunk_id: str
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseVectorStore(ABC):
    """Abstract base class for vector storage backends.

    Vector stores persist embeddings and support similarity search
    for retrieval operations.
    """

    @abstractmethod
    def upsert(
        self,
        chunk_id: str,
        embedding: List[float],
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Insert or update a vector in the store.

        Args:
            chunk_id: Unique identifier for the chunk.
            embedding: The embedding vector.
            content: The text content of the chunk.
            metadata: Optional metadata to store with the vector.
        """
        ...

    @abstractmethod
    def search(
        self,
        embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search for similar vectors.

        Args:
            embedding: The query embedding vector.
            top_k: Number of results to return.
            filters: Optional metadata filters.

        Returns:
            List of SearchResult objects ordered by similarity.
        """
        ...

    @abstractmethod
    def delete(self, chunk_id: str) -> None:
        """Delete a vector from the store.

        Args:
            chunk_id: The unique identifier of the chunk to delete.
        """
        ...
