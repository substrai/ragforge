"""Base retriever interface for RAGForge."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List

from ragforge.storage.base import SearchResult


@dataclass
class RetrievalResult:
    """A single retrieval result with relevance score."""

    chunk_id: str
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseRetriever(ABC):
    """Abstract base class for retrieval strategies.

    Retrievers take raw search results and apply scoring, reranking,
    or filtering to produce the final ranked results.
    """

    @abstractmethod
    def retrieve(
        self,
        query_text: str,
        query_embedding: List[float],
        candidates: List[SearchResult],
        top_k: int = 5,
    ) -> List[RetrievalResult]:
        """Retrieve and rank results from candidates.

        Args:
            query_text: The original query string.
            query_embedding: The query embedding vector.
            candidates: Raw search results from the vector store.
            top_k: Number of results to return.

        Returns:
            List of RetrievalResult objects ranked by relevance.
        """
        ...
