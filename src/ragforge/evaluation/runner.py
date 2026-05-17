"""Evaluation runner for RAGForge.

Loads golden datasets and runs evaluation metrics against the pipeline.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ragforge.evaluation.metrics import (
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


@dataclass
class QueryEvalResult:
    """Evaluation result for a single query."""

    query: str
    precision_at_k: float
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    retrieved_ids: List[str] = field(default_factory=list)
    relevant_ids: List[str] = field(default_factory=list)


@dataclass
class EvaluationReport:
    """Complete evaluation report with per-query and aggregate scores."""

    metrics: Dict[str, float]
    per_query_results: List[QueryEvalResult]
    timestamp: float
    config_used: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary for serialization."""
        return {
            "metrics": self.metrics,
            "per_query_results": [
                {
                    "query": r.query,
                    "precision_at_k": r.precision_at_k,
                    "recall_at_k": r.recall_at_k,
                    "mrr": r.mrr,
                    "ndcg_at_k": r.ndcg_at_k,
                    "retrieved_ids": r.retrieved_ids,
                    "relevant_ids": r.relevant_ids,
                }
                for r in self.per_query_results
            ],
            "timestamp": self.timestamp,
            "config_used": self.config_used,
        }


class EvaluationRunner:
    """Runs evaluation against a golden dataset.

    Loads a golden dataset (JSON file with queries and expected results),
    runs each query through the pipeline, and computes retrieval metrics.
    """

    def __init__(self, pipeline: Any, k: int = 5):
        """Initialize the evaluation runner.

        Args:
            pipeline: A RAGPipeline instance (or any object with a query() method).
            k: The k value for precision@k, recall@k, and ndcg@k.
        """
        self.pipeline = pipeline
        self.k = k

    def load_golden_dataset(self, path: str | Path) -> List[Dict[str, Any]]:
        """Load a golden dataset from a JSON file.

        Expected format:
        [
            {
                "query": "...",
                "relevant_ids": ["chunk_id_1", "chunk_id_2"],
                "expected_answer": "..."  (optional)
            }
        ]

        Args:
            path: Path to the golden dataset JSON file.

        Returns:
            List of golden dataset entries.
        """
        dataset_path = Path(path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Golden dataset not found: {dataset_path}")

        with open(dataset_path, "r") as f:
            dataset = json.load(f)

        if not isinstance(dataset, list):
            raise ValueError("Golden dataset must be a JSON array")

        return dataset

    def run(self, golden_dataset_path: str | Path) -> EvaluationReport:
        """Run evaluation against a golden dataset.

        Args:
            golden_dataset_path: Path to the golden dataset JSON file.

        Returns:
            EvaluationReport with per-query and aggregate metrics.
        """
        dataset = self.load_golden_dataset(golden_dataset_path)
        return self.run_from_dataset(dataset)

    def run_from_dataset(self, dataset: List[Dict[str, Any]]) -> EvaluationReport:
        """Run evaluation from an already-loaded dataset.

        Args:
            dataset: List of golden dataset entries.

        Returns:
            EvaluationReport with per-query and aggregate metrics.
        """
        per_query_results: List[QueryEvalResult] = []

        for entry in dataset:
            query_text = entry["query"]
            relevant_ids = entry.get("relevant_ids", [])

            # Run query through pipeline
            results = self.pipeline.query(query_text, top_k=self.k)
            retrieved_ids = [r.chunk_id for r in results]

            # Compute metrics
            p_at_k = precision_at_k(retrieved_ids, relevant_ids, self.k)
            r_at_k = recall_at_k(retrieved_ids, relevant_ids, self.k)
            mrr_score = mrr(retrieved_ids, relevant_ids)

            # For NDCG, assign binary relevance (1.0 if relevant, 0.0 otherwise)
            relevant_set = set(relevant_ids)
            relevance_scores = [
                1.0 if rid in relevant_set else 0.0 for rid in retrieved_ids
            ]
            ndcg_score = ndcg_at_k(retrieved_ids, relevance_scores, self.k)

            per_query_results.append(
                QueryEvalResult(
                    query=query_text,
                    precision_at_k=p_at_k,
                    recall_at_k=r_at_k,
                    mrr=mrr_score,
                    ndcg_at_k=ndcg_score,
                    retrieved_ids=retrieved_ids,
                    relevant_ids=relevant_ids,
                )
            )

        # Compute aggregate metrics
        n = len(per_query_results)
        if n > 0:
            aggregate_metrics = {
                "avg_precision_at_k": sum(r.precision_at_k for r in per_query_results) / n,
                "avg_recall_at_k": sum(r.recall_at_k for r in per_query_results) / n,
                "avg_mrr": sum(r.mrr for r in per_query_results) / n,
                "avg_ndcg_at_k": sum(r.ndcg_at_k for r in per_query_results) / n,
                "num_queries": n,
                "k": self.k,
            }
        else:
            aggregate_metrics = {
                "avg_precision_at_k": 0.0,
                "avg_recall_at_k": 0.0,
                "avg_mrr": 0.0,
                "avg_ndcg_at_k": 0.0,
                "num_queries": 0,
                "k": self.k,
            }

        return EvaluationReport(
            metrics=aggregate_metrics,
            per_query_results=per_query_results,
            timestamp=time.time(),
            config_used={"k": self.k},
        )
