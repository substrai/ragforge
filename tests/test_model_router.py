"""Tests for embedding model router."""

import pytest

from ragforge.cost.model_router import EmbeddingModelRouter, RoutingResult


class TestEmbeddingModelRouter:
    """Tests for EmbeddingModelRouter class."""

    def test_simple_query_routes_to_lite(self):
        """Test that simple queries route to lite model."""
        router = EmbeddingModelRouter(
            full_model="bedrock/titan-embed-v2",
            lite_model="local/dev",
            complexity_threshold=0.5,
        )
        # Short, simple query
        model = router.route("hello")
        assert model == "local/dev"

    def test_complex_query_routes_to_full(self):
        """Test that complex queries route to full model."""
        router = EmbeddingModelRouter(
            full_model="bedrock/titan-embed-v2",
            lite_model="local/dev",
            complexity_threshold=0.3,
        )
        # Long, complex query with question and special chars
        query = (
            "What are the architectural implications of using event-driven "
            "microservices with CQRS pattern for high-throughput data processing?"
        )
        model = router.route(query)
        assert model == "bedrock/titan-embed-v2"

    def test_complexity_empty_query(self):
        """Test complexity of empty query."""
        router = EmbeddingModelRouter()
        assert router.estimate_complexity("") == 0.0
        assert router.estimate_complexity("   ") == 0.0

    def test_complexity_short_query(self):
        """Test complexity of short query."""
        router = EmbeddingModelRouter()
        complexity = router.estimate_complexity("hello")
        assert 0.0 <= complexity <= 1.0
        # Short query should have low complexity
        assert complexity < 0.5

    def test_complexity_question_increases_score(self):
        """Test that questions increase complexity score."""
        router = EmbeddingModelRouter()
        without_q = router.estimate_complexity("explain RAG architecture")
        with_q = router.estimate_complexity("explain RAG architecture?")
        assert with_q > without_q

    def test_complexity_longer_query_higher(self):
        """Test that longer queries have higher complexity."""
        router = EmbeddingModelRouter()
        short = router.estimate_complexity("hello world")
        long_query = router.estimate_complexity(
            "explain the detailed architectural implications of using "
            "retrieval augmented generation with vector databases"
        )
        assert long_query > short

    def test_complexity_clamped_to_range(self):
        """Test that complexity is always in [0, 1]."""
        router = EmbeddingModelRouter()

        # Very long query with lots of special chars
        extreme = "?" * 100 + " " + "supercalifragilisticexpialidocious " * 50
        complexity = router.estimate_complexity(extreme)
        assert 0.0 <= complexity <= 1.0

    def test_custom_threshold(self):
        """Test custom complexity threshold."""
        # Very low threshold - almost everything goes to full model
        router = EmbeddingModelRouter(complexity_threshold=0.01)
        model = router.route("hello world test")
        assert model == "bedrock/titan-embed-v2"

        # Very high threshold - almost everything goes to lite model
        router = EmbeddingModelRouter(complexity_threshold=0.99)
        model = router.route("what is the meaning of life?")
        assert model == "local/dev"

    def test_route_with_details(self):
        """Test routing with detailed result."""
        router = EmbeddingModelRouter(
            full_model="full-model",
            lite_model="lite-model",
            complexity_threshold=0.5,
        )
        result = router.route_with_details("hello")
        assert isinstance(result, RoutingResult)
        assert result.model == "lite-model"
        assert 0.0 <= result.complexity <= 1.0
        assert "Simple" in result.reason or "simple" in result.reason.lower()

    def test_special_chars_affect_complexity(self):
        """Test that special characters affect complexity."""
        router = EmbeddingModelRouter()
        plain = router.estimate_complexity("hello world test")
        special = router.estimate_complexity("hello! @world #test $$$")
        assert special > plain

    def test_default_models(self):
        """Test default model names."""
        router = EmbeddingModelRouter()
        assert router.full_model == "bedrock/titan-embed-v2"
        assert router.lite_model == "local/dev"
        assert router.complexity_threshold == 0.5
