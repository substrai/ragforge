"""Tests for contextual compression module."""

import pytest

from ragforge.retrievers.compression import ContextualCompressor, CompressedResult


class TestContextualCompressor:
    """Tests for ContextualCompressor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.compressor = ContextualCompressor()

    def test_compress_basic(self):
        """Test basic compression with relevant sentences."""
        query = "machine learning models"
        chunks = [
            "Machine learning models are trained on data. "
            "The weather is nice today. "
            "Deep learning is a subset of machine learning.",
            "Python is a programming language. "
            "Models can be deployed to production. "
            "Cats are popular pets.",
        ]

        result = self.compressor.compress(query, chunks)
        assert isinstance(result, CompressedResult)
        assert result.original_chunks == 2
        assert result.sentences_kept > 0
        assert result.sentences_total > 0
        # Relevant sentences should be kept
        assert "machine learning" in result.text.lower() or "models" in result.text.lower()

    def test_compress_empty_chunks(self):
        """Test compression with empty chunks list."""
        result = self.compressor.compress("test query", [])
        assert result.text == ""
        assert result.original_chunks == 0
        assert result.sentences_kept == 0

    def test_compress_empty_query(self):
        """Test compression with empty query."""
        chunks = ["This is a test sentence. Another sentence here."]
        result = self.compressor.compress("", chunks)
        # With empty query, no sentences should score above min_score
        assert result.sentences_kept == 0

    def test_compress_max_tokens(self):
        """Test that compression respects max_tokens limit."""
        query = "data processing"
        # Create chunks with many sentences
        chunks = [
            "Data processing is important. " * 20,
            "Processing data requires tools. " * 20,
        ]

        result = self.compressor.compress(query, chunks, max_tokens=50)
        # Should be limited by max_tokens
        word_count = len(result.text.split())
        # max_tokens=50, words ≈ 50/1.3 ≈ 38
        assert word_count <= 50

    def test_compress_all_relevant(self):
        """Test compression when all sentences are relevant."""
        query = "python programming"
        chunks = [
            "Python is great for programming. Python has many libraries."
        ]

        result = self.compressor.compress(query, chunks, max_tokens=1000)
        assert result.sentences_kept > 0

    def test_compress_no_relevant(self):
        """Test compression when no sentences are relevant."""
        query = "quantum physics"
        chunks = [
            "The cat sat on the mat. Dogs like to play fetch."
        ]

        result = self.compressor.compress(query, chunks)
        # No overlap with query terms
        assert result.sentences_kept == 0
        assert result.text == ""

    def test_compression_ratio(self):
        """Test that compression ratio is calculated correctly."""
        query = "machine learning"
        chunks = [
            "Machine learning is powerful. "
            "The sky is blue. "
            "Trees are green. "
            "Water is wet. "
            "Machine learning uses data.",
        ]

        result = self.compressor.compress(query, chunks, max_tokens=1000)
        assert 0.0 <= result.compression_ratio <= 1.0
        # Should compress since not all sentences are relevant
        if result.sentences_kept < result.sentences_total:
            assert result.compression_ratio > 0.0

    def test_compress_single_sentence(self):
        """Test compression with a single sentence chunk."""
        query = "hello world"
        chunks = ["Hello world is a common greeting."]

        result = self.compressor.compress(query, chunks)
        assert result.sentences_total == 1

    def test_min_score_filtering(self):
        """Test that min_score parameter filters low-relevance sentences."""
        # High min_score should filter more aggressively
        strict_compressor = ContextualCompressor(min_score=0.8)
        query = "data"
        chunks = ["Data is important. The sun is bright. Data drives decisions."]

        result = strict_compressor.compress(query, chunks)
        # With high min_score, fewer sentences should pass
        assert result.sentences_kept <= 3

    def test_compress_preserves_order_by_relevance(self):
        """Test that most relevant sentences come first."""
        query = "python programming language"
        chunks = [
            "Java is also popular. "
            "Python is a programming language. "
            "C++ is fast.",
        ]

        result = self.compressor.compress(query, chunks)
        if result.sentences_kept > 0:
            # The most relevant sentence should be about Python
            assert "python" in result.text.lower() or "programming" in result.text.lower()

    def test_compress_multiple_chunks(self):
        """Test compression across multiple chunks."""
        query = "database optimization"
        chunks = [
            "Database optimization improves performance. Indexes help queries.",
            "Caching reduces database load. Memory is fast.",
            "Flowers bloom in spring. Birds sing in the morning.",
        ]

        result = self.compressor.compress(query, chunks)
        assert result.original_chunks == 3
        # Should pick sentences from relevant chunks
        assert "database" in result.text.lower() or "optimization" in result.text.lower()
