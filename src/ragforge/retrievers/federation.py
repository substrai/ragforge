"""Federated retriever that queries multiple vector stores with RRF merging."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ragforge.retrievers.base import BaseRetriever, RetrievalResult
from ragforge.storage.base import BaseVectorStore, SearchResult


class FederatedRetriever(BaseRetriever):
    """Queries multiple vector stores and merges results using Reciprocal Rank Fusion.

    Supports per-source weights and deduplication of results across sources.
    RRF formula: score = sum(weight / (k + rank)) across all sources where k=60.
    """

    def __init__(
        self,
        sources: List[Tuple[BaseVectorStore, float]],
        k: int = 60,
        top_k: int = 5,
        deduplicate: bool = True,
    ):
        """Initialize the federated retriever.

        Args:
            sources: List of (vector_store, weight) tuples.
            k: RRF constant (default 60).
            top_k: Number of results to return.
            deduplicate: Whether to deduplicate results across sources.
        """
        self.sources = sources
        self.k = k
        self.top_k = top_k
        self.deduplicate = deduplicate

    def retrieve(
        self,
        query_text: str,
        query_embedding: List[float],
        candidates: List[SearchResult] | None = None,
        top_k: int = 5,
    ) -> List[RetrievalResult]:
        """Retrieve results from all federated sources using RRF.

        Args:
            query_text: The original query string.
            query_embedding: The query embedding vector.
            candidates: Ignored for federated retrieval (queries stores directly).
            top_k: Number of results to return.

        Returns:
            List of RetrievalResult objects ranked by RRF score.
        """
        k = top_k or self.top_k
        all_ranked_results: List[List[SearchResult]] = []
        weights: List[float] = []

        # Query each source
        for store, weight in self.sources:
            results = store.search(
                embedding=query_embedding,
                top_k=k * 2,  # Over-fetch for better fusion
            )
            all_ranked_results.append(results)
            weights.append(weight)

        # Apply RRF fusion
        fused_scores = self._reciprocal_rank_fusion(all_ranked_results, weights)

        # Deduplicate if enabled
        if self.deduplicate:
            fused_scores = self._deduplicate(fused_scores)

        # Sort by fused score descending
        sorted_results = sorted(fused_scores.items(), key=lambda x: x[1][1], reverse=True)

        # Convert to RetrievalResult
        results: List[RetrievalResult] = []
        for chunk_id, (result, score) in sorted_results[:k]:
            results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    content=result.content,
                    score=score,
                    metadata=result.metadata,
                )
            )

        return results

    def retrieve_from_candidates(
        self,
        query_text: str,
        query_embedding: List[float],
        candidates: List[SearchResult],
        top_k: int = 5,
    ) -> List[RetrievalResult]:
        """Fallback retrieve method using pre-fetched candidates.

        This is used when candidates are already provided (e.g., from a single store).
        Simply converts and returns top-k candidates.
        """
        results = []
        for i, candidate in enumerate(candidates[:top_k]):
            results.append(
                RetrievalResult(
                    chunk_id=candidate.chunk_id,
                    content=candidate.content,
                    score=candidate.score,
                    metadata=candidate.metadata,
                )
            )
        return results

    def _reciprocal_rank_fusion(
        self,
        ranked_lists: List[List[SearchResult]],
        weights: List[float],
    ) -> Dict[str, Tuple[SearchResult, float]]:
        """Apply Reciprocal Rank Fusion across multiple ranked lists.

        RRF score for a document = sum(weight / (k + rank)) across all lists
        where rank is 1-indexed.

        Args:
            ranked_lists: List of ranked result lists from each source.
            weights: Per-source weights.

        Returns:
            Dict mapping chunk_id to (SearchResult, fused_score).
        """
        fused: Dict[str, Tuple[SearchResult, float]] = {}

        for source_idx, results in enumerate(ranked_lists):
            weight = weights[source_idx] if source_idx < len(weights) else 1.0

            for rank, result in enumerate(results, start=1):
                rrf_score = weight / (self.k + rank)

                if result.chunk_id in fused:
                    existing_result, existing_score = fused[result.chunk_id]
                    fused[result.chunk_id] = (existing_result, existing_score + rrf_score)
                else:
                    fused[result.chunk_id] = (result, rrf_score)

        return fused

    def _deduplicate(
        self, fused_scores: Dict[str, Tuple[SearchResult, float]]
    ) -> Dict[str, Tuple[SearchResult, float]]:
        """Remove duplicate results based on content similarity.

        Uses exact content matching for deduplication. Results with the same
        content but different chunk_ids are merged, keeping the higher score.
        """
        seen_content: Dict[str, str] = {}  # content -> best chunk_id
        deduplicated: Dict[str, Tuple[SearchResult, float]] = {}

        # Sort by score descending to keep highest-scored version
        sorted_items = sorted(fused_scores.items(), key=lambda x: x[1][1], reverse=True)

        for chunk_id, (result, score) in sorted_items:
            content_key = result.content.strip()
            if content_key not in seen_content:
                seen_content[content_key] = chunk_id
                deduplicated[chunk_id] = (result, score)

        return deduplicated
