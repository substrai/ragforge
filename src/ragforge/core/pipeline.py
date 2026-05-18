"""RAGForge pipeline orchestrator - coordinates ingestion, retrieval, and evaluation."""

from __future__ import annotations

import time
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
        self._query_expander = None
        self._query_analytics = None
        self._cost_tracker = None
        self._budget_enforcer = None
        self._model_router = None
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

        # Initialize query expander
        from ragforge.retrievers.query_expansion import QueryExpander
        self._query_expander = QueryExpander()

        # Initialize query analytics
        from ragforge.evaluation.analytics import QueryAnalytics
        analytics_path = self.config.quality.evaluation.get(
            "analytics_path", "ragforge_analytics.json"
        )
        self._query_analytics = QueryAnalytics(storage_path=analytics_path)

        # Initialize cost tracking
        self._init_cost_components()

        self._initialized = True

    def _init_cost_components(self) -> None:
        """Initialize cost tracking, budget enforcement, and model routing."""
        from ragforge.cost.tracker import CostTracker
        from ragforge.cost.enforcer import BudgetEnforcer
        from ragforge.cost.model_router import EmbeddingModelRouter

        cost_path = self.config.cost.optimization.get(
            "cost_path", "ragforge_costs.json"
        )
        self._cost_tracker = CostTracker(storage_path=cost_path)

        # Budget enforcer
        daily_budget = self.config.cost.budget.get("daily_total", 10.0)
        monthly_budget = self.config.cost.budget.get("monthly_total", 100.0)
        action = self.config.cost.optimization.get("action_on_exceed", "alert")
        self._budget_enforcer = BudgetEnforcer(
            cost_tracker=self._cost_tracker,
            daily_budget=daily_budget,
            monthly_budget=monthly_budget,
            action_on_exceed=action,
        )

        # Model router (only if cost optimization is enabled)
        if self.config.cost.optimization.get("model_routing", False):
            lite_model = self.config.cost.optimization.get("lite_model", "local/dev")
            threshold = self.config.cost.optimization.get("complexity_threshold", 0.5)
            self._model_router = EmbeddingModelRouter(
                full_model=self.config.embedding.model,
                lite_model=lite_model,
                complexity_threshold=threshold,
            )

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
            # Check budget before embedding
            if self._budget_enforcer:
                self._budget_enforcer.check_budget()

            batch_size = self.config.embedding.batch_size
            for i in range(0, len(all_chunks), batch_size):
                batch = all_chunks[i : i + batch_size]
                texts = [c.content for c in batch]
                embeddings = self._embedder.embed_batch(texts)

                # Record embedding cost
                if self._cost_tracker:
                    token_estimate = sum(int(len(t.split()) * 1.3) for t in texts)
                    self._cost_tracker.record_embedding_cost(
                        tokens=token_estimate,
                        model=self.config.embedding.model,
                    )

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
        expand_query: Optional[bool] = None,
    ) -> List[QueryResult]:
        """Query the RAG pipeline.

        Args:
            query_text: The query string.
            top_k: Number of results to return (overrides config).
            filters: Optional metadata filters.
            expand_query: Whether to use query expansion (overrides config).

        Returns:
            List of QueryResult objects ranked by relevance.
        """
        self._initialize()

        k = top_k or self.config.retrieval.top_k

        # Check budget before query
        if self._budget_enforcer:
            budget_action = self._budget_enforcer.check_budget()
        else:
            budget_action = "ok"

        # Determine if query expansion is enabled
        use_expansion = expand_query
        if use_expansion is None:
            use_expansion = self.config.quality.evaluation.get("query_expansion", False)

        # Expand query if enabled
        queries_to_run = [query_text]
        if use_expansion and self._query_expander:
            queries_to_run = self._query_expander.expand(query_text)

        all_results: List[QueryResult] = []
        seen_chunk_ids: set = set()

        for q_text in queries_to_run:
            # Use model router if available and not downgraded
            embedding_model = self.config.embedding.model
            if self._model_router and budget_action != "downgrade":
                embedding_model = self._model_router.route(q_text)
            elif budget_action == "downgrade" and self._model_router:
                embedding_model = self._model_router.lite_model

            # Embed the query
            query_embedding = self._embedder.embed(q_text)

            # Record query cost
            if self._cost_tracker:
                token_estimate = int(len(q_text.split()) * 1.3)
                self._cost_tracker.record_query_cost(
                    query=q_text,
                    tokens=token_estimate,
                    model=embedding_model,
                )

            # Retrieve from vector store
            raw_results = self._vector_store.search(
                embedding=query_embedding,
                top_k=k * 2,  # Over-fetch for reranking
                filters=filters,
            )

            # Apply retrieval strategy (hybrid scoring, reranking)
            ranked_results = self._retriever.retrieve(
                query_text=q_text,
                query_embedding=query_embedding,
                candidates=raw_results,
                top_k=k,
            )

            for r in ranked_results:
                if r.chunk_id not in seen_chunk_ids:
                    seen_chunk_ids.add(r.chunk_id)
                    all_results.append(
                        QueryResult(
                            content=r.content,
                            score=r.score,
                            source=r.metadata.get("source", "unknown"),
                            chunk_id=r.chunk_id,
                            metadata=r.metadata,
                        )
                    )

        # Sort all results by score and return top-k
        all_results.sort(key=lambda r: r.score, reverse=True)
        return all_results[:k]

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

    def evaluate(self, golden_dataset_path: str | Path) -> "EvaluationReport":
        """Run evaluation against a golden dataset.

        Args:
            golden_dataset_path: Path to the golden dataset JSON file.

        Returns:
            EvaluationReport with per-query and aggregate metrics.
        """
        self._initialize()

        from ragforge.evaluation.runner import EvaluationRunner

        k = self.config.quality.evaluation.get("k", self.config.retrieval.top_k)
        runner = EvaluationRunner(pipeline=self, k=k)
        return runner.run(golden_dataset_path)

    def query_with_analytics(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[QueryResult]:
        """Query the pipeline and record analytics.

        Same as query() but also records latency and result metrics.

        Args:
            query_text: The query string.
            top_k: Number of results to return (overrides config).
            filters: Optional metadata filters.

        Returns:
            List of QueryResult objects ranked by relevance.
        """
        self._initialize()

        start_time = time.time()
        results = self.query(query_text, top_k=top_k, filters=filters)
        latency_ms = (time.time() - start_time) * 1000

        if self._query_analytics:
            self._query_analytics.record_query(
                query=query_text,
                results=results,
                latency_ms=latency_ms,
            )

        return results

    def get_cost_report(self) -> Dict[str, Any]:
        """Get a cost report for the pipeline.

        Returns:
            Dictionary with total cost, breakdown by category,
            daily cost, monthly forecast, and budget status.
        """
        self._initialize()

        if not self._cost_tracker:
            return {"error": "Cost tracking not initialized"}

        report: Dict[str, Any] = {
            "total_cost": self._cost_tracker.get_total_cost(),
            "breakdown": self._cost_tracker.get_cost_breakdown(),
            "daily_cost": self._cost_tracker.get_daily_cost(),
            "monthly_forecast": self._cost_tracker.get_monthly_forecast(),
        }

        if self._budget_enforcer:
            status = self._budget_enforcer.get_budget_status()
            report["budget_status"] = {
                "daily_limit": status.daily_limit,
                "daily_spent": status.daily_spent,
                "daily_remaining": status.daily_remaining,
                "monthly_limit": status.monthly_limit,
                "monthly_spent": status.monthly_spent,
                "monthly_remaining": status.monthly_remaining,
                "is_exceeded": status.is_exceeded,
            }

        return report

    def query_with_audit(
        self,
        query_text: str,
        tenant_id: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[QueryResult]:
        """Query the pipeline with audit logging and access control.

        Logs the retrieval to the audit trail and applies tenant-based
        access control filtering.

        Args:
            query_text: The query string.
            tenant_id: Identifier of the tenant making the query.
            top_k: Number of results to return (overrides config).
            filters: Optional metadata filters.

        Returns:
            List of QueryResult objects (filtered by access control).
        """
        from ragforge.deployment.audit import AuditLogger

        results = self.query(query_text, top_k=top_k, filters=filters)

        # Log to audit trail
        audit_logger = AuditLogger()
        audit_logger.log_retrieval(
            tenant_id=tenant_id,
            query=query_text,
            results=results,
        )

        return results

    def get_deployment_template(self) -> str:
        """Generate a CloudFormation deployment template for this pipeline.

        Returns:
            YAML string of the CloudFormation/SAM template.
        """
        from ragforge.deployment.cloudformation import CloudFormationGenerator

        generator = CloudFormationGenerator()
        return generator.generate(self.config)
