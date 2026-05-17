"""Tests for FederatedRetriever with RRF merging."""

import pytest
from typing import Any, Dict, List, Optional

from ragforge.retrievers.federation import FederatedRetriever
from ragforge.storage.base import BaseVectorStore, SearchResult


class MockVectorStore(BaseVectorStore):
    """Mock vector store for testing federation."""

    def __init__(self, results: List[SearchResult]):
        self._results = results

    def upsert(
        self,
        chunk_id: str,
        embedding: List[float],
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        pass

    def search(
        self,
        embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        return self._results[:top_k]

    def delete(self, chunk_id: str) -> None:
        pass


def make_result(chunk_id: str, content: str, score: float) -> SearchResult:
    """Helper to create a SearchResult."""
    return SearchResult(
        chunk_id=chunk_id,
        content=content,
        score=score,
        metadata={"source": f"source_{chunk_id}"},
    )


class TestFederatedRetriever:
    """Test federated retrieval with RRF merging."""

    def test_single_source(self):
        results = [
            make_result("a1", "Document A", 0.9),
            make_result("a2", "Document B", 0.8),
            make_result("a3", "Document C", 0.7),
        ]
        store = MockVectorStore(results)
        retriever = FederatedRetriever(sources=[(store, 1.0)], top_k=3)

        merged = retriever.retrieve(
            query_text="test query",
            query_embedding=[0.1, 0.2, 0.3],
            top_k=3,
        )

        assert len(merged) == 3
        # First result should be "a1" (rank 1 in only source)
        assert merged[0].chunk_id == "a1"

    def test_multiple_sources_rrf(self):
        store1_results = [
            make_result("doc1", "First document", 0.9),
            make_result("doc2", "Second document", 0.8),
            make_result("doc3", "Third document", 0.7),
        ]
        store2_results = [
            make_result("doc2", "Second document", 0.95),
            make_result("doc4", "Fourth document", 0.85),
            make_result("doc1", "First document", 0.75),
        ]

        store1 = MockVectorStore(store1_results)
        store2 = MockVectorStore(store2_results)

        retriever = FederatedRetriever(
            sources=[(store1, 1.0), (store2, 1.0)],
            top_k=5,
        )

        merged = retriever.retrieve(
            query_text="test",
            query_embedding=[0.1, 0.2],
            top_k=5,
        )

        # doc2 appears at rank 2 in store1 and rank 1 in store2
        # doc1 appears at rank 1 in store1 and rank 3 in store2
        # doc2 RRF = 1/(60+2) + 1/(60+1) = 0.01613 + 0.01639 = 0.03252
        # doc1 RRF = 1/(60+1) + 1/(60+3) = 0.01639 + 0.01587 = 0.03226
        # doc2 should rank higher than doc1
        doc2_idx = next(i for i, r in enumerate(merged) if r.chunk_id == "doc2")
        doc1_idx = next(i for i, r in enumerate(merged) if r.chunk_id == "doc1")
        assert doc2_idx < doc1_idx

    def test_weighted_sources(self):
        store1_results = [
            make_result("a", "Doc A", 0.9),
            make_result("b", "Doc B", 0.8),
        ]
        store2_results = [
            make_result("b", "Doc B", 0.95),
            make_result("a", "Doc A", 0.85),
        ]

        store1 = MockVectorStore(store1_results)
        store2 = MockVectorStore(store2_results)

        # Give store2 much higher weight
        retriever = FederatedRetriever(
            sources=[(store1, 0.1), (store2, 10.0)],
            top_k=2,
        )

        merged = retriever.retrieve(
            query_text="test",
            query_embedding=[0.1],
            top_k=2,
        )

        # With store2 having 100x weight, "b" (rank 1 in store2) should be first
        assert merged[0].chunk_id == "b"

    def test_deduplication(self):
        store1_results = [
            make_result("id1", "Same content here", 0.9),
        ]
        store2_results = [
            make_result("id2", "Same content here", 0.85),
        ]

        store1 = MockVectorStore(store1_results)
        store2 = MockVectorStore(store2_results)

        retriever = FederatedRetriever(
            sources=[(store1, 1.0), (store2, 1.0)],
            deduplicate=True,
            top_k=5,
        )

        merged = retriever.retrieve(
            query_text="test",
            query_embedding=[0.1],
            top_k=5,
        )

        # Should deduplicate identical content
        contents = [r.content for r in merged]
        assert contents.count("Same content here") == 1

    def test_no_deduplication(self):
        store1_results = [
            make_result("id1", "Same content here", 0.9),
        ]
        store2_results = [
            make_result("id2", "Same content here", 0.85),
        ]

        store1 = MockVectorStore(store1_results)
        store2 = MockVectorStore(store2_results)

        retriever = FederatedRetriever(
            sources=[(store1, 1.0), (store2, 1.0)],
            deduplicate=False,
            top_k=5,
        )

        merged = retriever.retrieve(
            query_text="test",
            query_embedding=[0.1],
            top_k=5,
        )

        # Without deduplication, both should appear
        assert len(merged) == 2

    def test_rrf_score_calculation(self):
        """Verify RRF formula: score = weight / (k + rank)."""
        results = [
            make_result("first", "First", 0.9),
            make_result("second", "Second", 0.8),
        ]
        store = MockVectorStore(results)

        k = 60
        retriever = FederatedRetriever(sources=[(store, 1.0)], k=k, top_k=2)

        merged = retriever.retrieve(
            query_text="test",
            query_embedding=[0.1],
            top_k=2,
        )

        # First result: score = 1.0 / (60 + 1) = 0.01639...
        expected_score_1 = 1.0 / (k + 1)
        assert abs(merged[0].score - expected_score_1) < 1e-6

        # Second result: score = 1.0 / (60 + 2) = 0.01613...
        expected_score_2 = 1.0 / (k + 2)
        assert abs(merged[1].score - expected_score_2) < 1e-6

    def test_top_k_limits_results(self):
        results = [make_result(f"doc{i}", f"Content {i}", 0.9 - i * 0.1) for i in range(10)]
        store = MockVectorStore(results)

        retriever = FederatedRetriever(sources=[(store, 1.0)], top_k=3)

        merged = retriever.retrieve(
            query_text="test",
            query_embedding=[0.1],
            top_k=3,
        )

        assert len(merged) == 3

    def test_empty_sources(self):
        store = MockVectorStore([])
        retriever = FederatedRetriever(sources=[(store, 1.0)], top_k=5)

        merged = retriever.retrieve(
            query_text="test",
            query_embedding=[0.1],
            top_k=5,
        )

        assert merged == []

    def test_metadata_preserved(self):
        results = [
            SearchResult(
                chunk_id="x1",
                content="Test content",
                score=0.9,
                metadata={"source": "file.txt", "page": 3},
            )
        ]
        store = MockVectorStore(results)
        retriever = FederatedRetriever(sources=[(store, 1.0)], top_k=5)

        merged = retriever.retrieve(
            query_text="test",
            query_embedding=[0.1],
            top_k=5,
        )

        assert merged[0].metadata["source"] == "file.txt"
        assert merged[0].metadata["page"] == 3

    def test_three_sources_fusion(self):
        store1 = MockVectorStore([make_result("a", "A", 0.9), make_result("b", "B", 0.8)])
        store2 = MockVectorStore([make_result("b", "B", 0.9), make_result("c", "C", 0.8)])
        store3 = MockVectorStore([make_result("a", "A", 0.9), make_result("c", "C", 0.8)])

        retriever = FederatedRetriever(
            sources=[(store1, 1.0), (store2, 1.0), (store3, 1.0)],
            top_k=3,
        )

        merged = retriever.retrieve(
            query_text="test",
            query_embedding=[0.1],
            top_k=3,
        )

        # "a" appears in store1 rank1 and store3 rank1: RRF = 2/(60+1) = 0.03279
        # "b" appears in store1 rank2 and store2 rank1: RRF = 1/(60+2) + 1/(60+1) = 0.03252
        # "a" should be ranked first
        assert merged[0].chunk_id == "a"
        assert len(merged) == 3
