"""RAGForge pipeline orchestrator - coordinates ingestion, retrieval, and evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from ragforge.core.config import RAGConfig, load_config
from ragforge.core.models import Chunk, Document, QueryResult


class RAGPipeline:
    """Main RAG pipeline orchestrator.

    Coordinates the full RAG lifecycle:
    1. Ingestion: documents → chunks → embeddings → vector store
    2. Retrieval: query → embed → search → rank → results
    3. Evaluation: measure retrieval quality metrics
    """

    def __init__(self, config: RAGConfig):
        self.config = config
        self._chunker_registry = None
        self._embedder = None
        self._vector_store = None
        self._retriever = None
        self._metadata_enricher = None
        self._ingestion_tracker = None
        self._initialized = False

    @classmethod
    def from_config(cls, path: str | Path = "ragforge.yaml") -> "RAGPipeline":
        """Create a RAGPipeline from a YAML configuration file."""
        config = load_config(path)
        return cls(config)

    def _initialize(self) -> None:
        """Lazy initialization of pipeline components."""
        if self._initialized:
            return

        # Initialize chunker registry
        from ragforge.chunkers.registry import ChunkerRegistry
        self._chunker_registry = ChunkerRegistry()

        # Initialize metadata enricher
        from ragforge.chunkers.metadata import MetadataEnricher
        extraction_config = self.config.chunking.metadata_extraction
        self._metadata_enricher = MetadataEnricher(
            extract_title=extraction_config.get("title", True),
            extract_dates=extraction_config.get("date", True),
            extract_word_count=True,
        )

        # Initialize ingestion tracker
        from ragforge.ingestion.tracker import IngestionTracker
        self._ingestion_tracker = IngestionTracker()

        # Initialize embedder based on config
        self._embedder = self._create_embedder()

        # Initialize vector store based on config
        self._vector_store = self._create_vector_store()

        # Initialize retriever based on config
        self._retriever = self._create_retriever()

        self._initialized = True

    def _create_embedder(self) -> BaseEmbedder:
        """Create embedder based on configuration."""
        from ragforge.embedders.local import LocalEmbedder

        model = self.config.embedding.model
        if model.startswith("bedrock/"):
            try:
                from ragforge.embedders.bedrock import BedrockEmbedder

                return BedrockEmbedder(
                    model_id=model.replace("bedrock/", ""),
                    dimensions=self.config.embedding.dimensions,
                )
            except ImportError:
                pass

        # Fallback to local embedder for development
        return LocalEmbedder(dimensions=self.config.embedding.dimensions)

    def _create_vector_store(self) -> BaseVectorStore:
        """Create vector store based on configuration."""
        provider = self.config.storage.provider

        if provider == "faiss-lambda" or provider == "faiss":
            from ragforge.storage.faiss_store import FAISSVectorStore

            return FAISSVectorStore(
                index_name=self.config.storage.index_name,
                dimensions=self.config.embedding.dimensions,
            )

        if provider == "opensearch-serverless":
            try:
                from ragforge.storage.opensearch_store import OpenSearchVectorStore

                return OpenSearchVectorStore(
                    index_name=self.config.storage.index_name,
                    dimensions=self.config.embedding.dimensions,
                    config=self.config.storage.config,
                )
            except ImportError:
                pass

        # Fallback to in-memory FAISS
        from ragforge.storage.faiss_store import FAISSVectorStore

        return FAISSVectorStore(
            index_name=self.config.storage.index_name,
            dimensions=self.config.embedding.dimensions,
        )

    def _create_retriever(self) -> BaseRetriever:
        """Create retriever based on configuration."""
        from ragforge.retrievers.hybrid import HybridRetriever

        return HybridRetriever(
            semantic_weight=self.config.retrieval.semantic_weight,
            keyword_weight=self.config.retrieval.keyword_weight,
            top_k=self.config.retrieval.top_k,
        )

    def ingest(self, documents: Optional[List[Document]] = None) -> Dict[str, Any]:
        """Run the ingestion pipeline: documents → chunks → embeddings → store.

        Args:
            documents: Optional list of documents. If None, loads from configured sources.

        Returns:
            Ingestion statistics (documents processed, chunks created, etc.)
        """
        self._initialize()

        if documents is None:
            documents = self._load_from_sources()

        stats = {
            "documents_processed": 0,
            "chunks_created": 0,
            "embeddings_generated": 0,
            "errors": [],
        }

        all_chunks: List[Chunk] = []

        for doc in documents:
            try:
                # Select chunker based on document type
                chunker = self._chunker_registry.get_chunker(
                    doc_type=doc.doc_type,
                    strategy=self.config.chunking.strategy,
                    max_chunk_size=self.config.chunking.max_chunk_size,
                    overlap=self.config.chunking.overlap,
                )

                # Chunk the document
                chunks = chunker.chunk(doc.content, source=doc.source, metadata=doc.metadata)

                # Enrich chunk metadata
                if self._metadata_enricher:
                    chunks = self._metadata_enricher.enrich(chunks)

                all_chunks.extend(chunks)
                stats["documents_processed"] += 1
                stats["chunks_created"] += len(chunks)

            except Exception as e:
                stats["errors"].append({"source": doc.source, "error": str(e)})

        # Embed all chunks in batches
        if all_chunks:
            batch_size = self.config.embedding.batch_size
            for i in range(0, len(all_chunks), batch_size):
                batch = all_chunks[i : i + batch_size]
                texts = [c.content for c in batch]
                embeddings = self._embedder.embed_batch(texts)

                # Store embeddings
                for chunk, embedding in zip(batch, embeddings):
                    self._vector_store.upsert(
                        chunk_id=chunk.chunk_id,
                        embedding=embedding,
                        content=chunk.content,
                        metadata={
                            "source": chunk.source,
                            **chunk.metadata,
                        },
                    )
                    stats["embeddings_generated"] += 1

        return stats

    def ingest_incremental(
        self, documents: Optional[List[Document]] = None
    ) -> Dict[str, Any]:
        """Run incremental ingestion - only processes documents that have changed.

        Uses IngestionTracker to compare document hashes and skip unchanged documents.

        Args:
            documents: Optional list of documents. If None, loads from configured sources.

        Returns:
            Ingestion statistics including skipped count.
        """
        self._initialize()

        if documents is None:
            documents = self._load_from_sources()

        # Filter to only changed documents
        changed_docs = self._ingestion_tracker.filter_changed(documents)
        skipped_count = len(documents) - len(changed_docs)

        # Ingest only changed documents
        stats = self.ingest(changed_docs) if changed_docs else {
            "documents_processed": 0,
            "chunks_created": 0,
            "embeddings_generated": 0,
            "errors": [],
        }

        # Mark successfully ingested documents
        if changed_docs:
            successfully_ingested = [
                doc for doc in changed_docs
                if doc.source not in [e.get("source") for e in stats["errors"]]
            ]
            self._ingestion_tracker.mark_batch_ingested(successfully_ingested)

        stats["documents_skipped"] = skipped_count
        stats["documents_changed"] = len(changed_docs)

        return stats

    def query(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[QueryResult]:
        """Query the RAG pipeline.

        Args:
            query_text: The query string.
            top_k: Number of results to return (overrides config).
            filters: Optional metadata filters.

        Returns:
            List of QueryResult objects ranked by relevance.
        """
        self._initialize()

        k = top_k or self.config.retrieval.top_k

        # Embed the query
        query_embedding = self._embedder.embed(query_text)

        # Retrieve from vector store
        raw_results = self._vector_store.search(
            embedding=query_embedding,
            top_k=k * 2,  # Over-fetch for reranking
            filters=filters,
        )

        # Apply retrieval strategy (hybrid scoring, reranking)
        ranked_results = self._retriever.retrieve(
            query_text=query_text,
            query_embedding=query_embedding,
            candidates=raw_results,
            top_k=k,
        )

        return [
            QueryResult(
                content=r.content,
                score=r.score,
                source=r.metadata.get("source", "unknown"),
                chunk_id=r.chunk_id,
                metadata=r.metadata,
            )
            for r in ranked_results
        ]

    def _load_from_sources(self) -> List[Document]:
        """Load documents from configured data sources."""
        documents = []

        for source_config in self.config.data_sources:
            if source_config.type == "local":
                docs = self._load_local_source(source_config)
                documents.extend(docs)
            elif source_config.type == "s3":
                docs = self._load_s3_source(source_config)
                documents.extend(docs)
            # Additional source types can be added here

        return documents

    def _load_local_source(self, source_config) -> List[Document]:
        """Load documents from local filesystem."""
        documents = []
        path = Path(source_config.config.get("path", "."))

        if not path.exists():
            return documents

        for file_type in source_config.file_types:
            for file_path in path.rglob(f"*.{file_type}"):
                try:
                    content = file_path.read_text(encoding="utf-8")
                    documents.append(
                        Document(
                            content=content,
                            source=str(file_path),
                            doc_type=file_type,
                            metadata={"filename": file_path.name},
                        )
                    )
                except (UnicodeDecodeError, IOError):
                    continue

        return documents

    def _load_s3_source(self, source_config) -> List[Document]:
        """Load documents from S3 (requires boto3)."""
        try:
            import boto3
        except ImportError:
            raise ImportError("boto3 is required for S3 sources. Install with: pip install substrai-ragforge[aws]")

        documents = []
        s3 = boto3.client("s3")
        bucket = source_config.config.get("bucket", "")
        prefix = source_config.config.get("prefix", "")

        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                ext = key.rsplit(".", 1)[-1].lower() if "." in key else ""

                if ext in source_config.file_types:
                    response = s3.get_object(Bucket=bucket, Key=key)
                    content = response["Body"].read().decode("utf-8", errors="ignore")
                    documents.append(
                        Document(
                            content=content,
                            source=f"s3://{bucket}/{key}",
                            doc_type=ext,
                            metadata={"bucket": bucket, "key": key},
                        )
                    )

        return documents
