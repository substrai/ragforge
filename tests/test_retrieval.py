"""Tests for hybrid retriever."""

import pytest

from ragforge.retrievers.base import RetrievalResult
from ragforge.retrievers.hybrid import HybridRetriever
from ragforge.storage.base import SearchResult


@pytest.fixture
def retriever():
    """Create a hybrid retriever with default weights."""
    return HybridRetriever(semantic_weight=0.7, keyword_weight=0.3, top_k=5)


@pytest.fixture
def sample_candidates():
    """Create sample search results as candidates."""
    return [
        SearchResult(
            chunk_id="chunk-1",
            content="Python is a high-level programming language known for readability.",
            score=0.95,
            metadata={"source": "python.md"},
        ),
        SearchResult(
            chunk_id="chunk-2",
            content="Java is a statically typed programming language used in enterprise.",
            score=0.85,
            metadata={"source": "java.md"},
        ),
        SearchResult(
            chunk_id="chunk-3",
            content="Machine learning algorithms can classify and predict outcomes.",
            score=0.60,
            metadata={"source": "ml.md"},
        ),
        SearchResult(
            chunk_id="chunk-4",
            content="The weather today is sunny with a high of 75 degrees.",
            score=0.30,
            metadata={"source": "weather.md"},
        ),
    ]


class TestHybridRetriever:
    """Tests for HybridRetriever."""

    def test_returns_retrieval_results(self, retriever, sample_candidates):
        """Should return RetrievalResult objects."""
        results = retriever.retrieve(
            query_text="Python programming",
            query_embedding=[0.1] * 128,
            candidates=sample_candidates,
            top_k=3,
        )

        assert all(isinstance(r, RetrievalResult) for r in results)

    def test_respects_top_k(self, retriever, sample_candidates):
        """Should return at most top_k results."""
        results = retriever.retrieve(
            query_text="programming language",
            query_embedding=[0.1] * 128,
            candidates=sample_candidates,
            top_k=2,
        )

        assert len(results) <= 2

    def test_results_sorted_by_score(self, retriever, sample_candidates):
        """Results should be sorted by combined score descending."""
        results = retriever.retrieve(
            query_text="programming",
            query_embedding=[0.1] * 128,
            candidates=sample_candidates,
            top_k=4,
        )

        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_keyword_boost(self, retriever, sample_candidates):
        """Keyword matching should boost relevant results."""
        results = retriever.retrieve(
            query_text="Python programming language readability",
            query_embedding=[0.1] * 128,
            candidates=sample_candidates,
            top_k=4,
        )

        # Python chunk should rank high due to keyword match + high semantic score
        assert results[0].chunk_id == "chunk-1"

    def test_empty_candidates(self, retriever):
        """Should return empty list for empty candidates."""
        results = retriever.retrieve(
            query_text="anything",
            query_embedding=[0.1] * 128,
            candidates=[],
            top_k=5,
        )

        assert results == []

    def test_combined_score_range(self, retriever, sample_candidates):
        """Combined scores should be in reasonable range."""
        results = retriever.retrieve(
            query_text="programming",
            query_embedding=[0.1] * 128,
            candidates=sample_candidates,
            top_k=4,
        )

        for result in results:
            assert 0.0 <= result.score <= 1.0

    def test_metadata_preserved(self, retriever, sample_candidates):
        """Metadata from candidates should be preserved in results."""
        results = retriever.retrieve(
            query_text="Python",
            query_embedding=[0.1] * 128,
            candidates=sample_candidates,
            top_k=4,
        )

        for result in results:
            assert "source" in result.metadata

    def test_custom_weights(self, sample_candidates):
        """Custom weights should affect scoring."""
        # Pure semantic retriever
        semantic_retriever = HybridRetriever(
            semantic_weight=1.0, keyword_weight=0.0
        )
        semantic_results = semantic_retriever.retrieve(
            query_text="weather sunny",
            query_embedding=[0.1] * 128,
            candidates=sample_candidates,
            top_k=4,
        )

        # Pure keyword retriever
        keyword_retriever = HybridRetriever(
            semantic_weight=0.0, keyword_weight=1.0
        )
        keyword_results = keyword_retriever.retrieve(
            query_text="weather sunny",
            query_embedding=[0.1] * 128,
            candidates=sample_candidates,
            top_k=4,
        )

        # With pure semantic, chunk-1 (score 0.95) should be first
        assert semantic_results[0].chunk_id == "chunk-1"

        # With pure keyword, weather chunk should rank higher
        weather_rank_keyword = next(
            i for i, r in enumerate(keyword_results) if r.chunk_id == "chunk-4"
        )
        weather_rank_semantic = next(
            i for i, r in enumerate(semantic_results) if r.chunk_id == "chunk-4"
        )
        assert weather_rank_keyword < weather_rank_semantic

    def test_single_candidate(self, retriever):
        """Should handle a single candidate correctly."""
        candidates = [
            SearchResult(
                chunk_id="only-one",
                content="The only document in the store.",
                score=0.8,
                metadata={},
            )
        ]

        results = retriever.retrieve(
            query_text="document",
            query_embedding=[0.1] * 128,
            candidates=candidates,
            top_k=5,
        )

        assert len(results) == 1
        assert results[0].chunk_id == "only-one"
