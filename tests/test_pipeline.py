"""Tests for the RAG pipeline (ingest + query with local embedder and FAISS)."""

import pytest

from ragforge.core.models import Chunk, Document, QueryResult
from ragforge.core.pipeline import RAGPipeline
from ragforge.core.config import (
    ChunkingConfig,
    DataSourceConfig,
    EmbeddingConfig,
    RAGConfig,
    RetrievalConfig,
    StorageConfig,
)


@pytest.fixture
def test_config():
    """Create a test configuration using local embedder and FAISS."""
    return RAGConfig(
        project_name="test-pipeline",
        data_sources=[DataSourceConfig(name="test", type="local")],
        chunking=ChunkingConfig(
            strategy="recursive",
            max_chunk_size=200,
            overlap=20,
        ),
        embedding=EmbeddingConfig(
            model="local/dev",
            dimensions=128,
            batch_size=10,
        ),
        storage=StorageConfig(
            provider="faiss",
            index_name="test-index",
        ),
        retrieval=RetrievalConfig(
            method="hybrid",
            semantic_weight=0.7,
            keyword_weight=0.3,
            top_k=3,
        ),
    )


@pytest.fixture
def pipeline(test_config):
    """Create a pipeline instance."""
    return RAGPipeline(test_config)


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        Document(
            content="Python is a programming language known for its simplicity and readability. "
            "It supports multiple programming paradigms including procedural, object-oriented, "
            "and functional programming.",
            source="python.md",
            doc_type="md",
            metadata={"topic": "programming"},
        ),
        Document(
            content="Machine learning is a subset of artificial intelligence that enables systems "
            "to learn from data. Deep learning uses neural networks with many layers.",
            source="ml.md",
            doc_type="md",
            metadata={"topic": "ai"},
        ),
        Document(
            content="AWS Lambda is a serverless compute service that runs code in response to events. "
            "It automatically manages the compute resources required by that code.",
            source="aws.md",
            doc_type="md",
            metadata={"topic": "cloud"},
        ),
    ]


class TestPipelineIngestion:
    """Tests for the ingestion pipeline."""

    def test_ingest_documents(self, pipeline, sample_documents):
        """Should successfully ingest documents."""
        stats = pipeline.ingest(documents=sample_documents)

        assert stats["documents_processed"] == 3
        assert stats["chunks_created"] > 0
        assert stats["embeddings_generated"] > 0
        assert stats["errors"] == []

    def test_ingest_empty_list(self, pipeline):
        """Should handle empty document list."""
        stats = pipeline.ingest(documents=[])

        assert stats["documents_processed"] == 0
        assert stats["chunks_created"] == 0

    def test_ingest_creates_chunks(self, pipeline, sample_documents):
        """Ingestion should create chunks from documents."""
        stats = pipeline.ingest(documents=sample_documents)

        assert stats["chunks_created"] >= len(sample_documents)

    def test_ingest_generates_embeddings(self, pipeline, sample_documents):
        """Ingestion should generate embeddings for all chunks."""
        stats = pipeline.ingest(documents=sample_documents)

        assert stats["embeddings_generated"] == stats["chunks_created"]


class TestPipelineQuery:
    """Tests for the query pipeline."""

    def test_query_returns_results(self, pipeline, sample_documents):
        """Query should return results after ingestion."""
        pipeline.ingest(documents=sample_documents)
        results = pipeline.query("What is Python?")

        assert len(results) > 0
        assert all(isinstance(r, QueryResult) for r in results)

    def test_query_respects_top_k(self, pipeline, sample_documents):
        """Query should respect the top_k parameter."""
        pipeline.ingest(documents=sample_documents)
        results = pipeline.query("programming language", top_k=2)

        assert len(results) <= 2

    def test_query_results_have_scores(self, pipeline, sample_documents):
        """Query results should have relevance scores."""
        pipeline.ingest(documents=sample_documents)
        results = pipeline.query("machine learning")

        for result in results:
            assert isinstance(result.score, float)
            assert result.score >= 0

    def test_query_results_have_content(self, pipeline, sample_documents):
        """Query results should include content."""
        pipeline.ingest(documents=sample_documents)
        results = pipeline.query("serverless")

        for result in results:
            assert result.content
            assert result.chunk_id

    def test_query_empty_store(self, pipeline):
        """Query on empty store should return empty results."""
        results = pipeline.query("anything")

        assert results == []

    def test_query_with_filters(self, pipeline, sample_documents):
        """Query with metadata filters should filter results."""
        pipeline.ingest(documents=sample_documents)
        results = pipeline.query("programming", filters={"topic": "programming"})

        for result in results:
            assert result.metadata.get("topic") == "programming"


class TestPipelineEndToEnd:
    """End-to-end pipeline tests."""

    def test_ingest_then_query(self, pipeline):
        """Full ingest → query cycle should work."""
        docs = [
            Document(
                content="The quick brown fox jumps over the lazy dog. " * 5,
                source="fox.txt",
                doc_type="txt",
            ),
            Document(
                content="A cat sat on the mat and looked at the hat. " * 5,
                source="cat.txt",
                doc_type="txt",
            ),
        ]

        stats = pipeline.ingest(documents=docs)
        assert stats["documents_processed"] == 2

        results = pipeline.query("fox jumping")
        assert len(results) > 0

    def test_multiple_ingestions(self, pipeline):
        """Multiple ingestion calls should accumulate data."""
        docs1 = [Document(content="First batch of content.", source="a.txt")]
        docs2 = [Document(content="Second batch of content.", source="b.txt")]

        pipeline.ingest(documents=docs1)
        pipeline.ingest(documents=docs2)

        results = pipeline.query("batch content")
        assert len(results) > 0
