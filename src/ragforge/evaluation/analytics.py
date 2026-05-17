"""Query analytics tracking for RAGForge.

Tracks query patterns, latency, zero-result queries, and low-confidence retrievals.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class QueryRecord:
    """A single recorded query with its results."""

    query: str
    num_results: int
    max_score: float
    latency_ms: float
    timestamp: float
    sources: List[str] = field(default_factory=list)


class QueryAnalytics:
    """Tracks and analyzes query patterns for monitoring.

    Records query metrics including latency, result counts, and confidence
    scores. Persists analytics to a JSON file for historical analysis.
    """

    def __init__(self, storage_path: str | Path = "ragforge_analytics.json"):
        """Initialize query analytics.

        Args:
            storage_path: Path to the JSON file for persisting analytics.
        """
        self.storage_path = Path(storage_path)
        self._records: List[QueryRecord] = []
        self._load()

    def _load(self) -> None:
        """Load existing analytics from storage."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r") as f:
                    data = json.load(f)
                self._records = [
                    QueryRecord(
                        query=r["query"],
                        num_results=r["num_results"],
                        max_score=r["max_score"],
                        latency_ms=r["latency_ms"],
                        timestamp=r["timestamp"],
                        sources=r.get("sources", []),
                    )
                    for r in data.get("records", [])
                ]
            except (json.JSONDecodeError, KeyError):
                self._records = []

    def _save(self) -> None:
        """Persist analytics to storage."""
        data = {
            "records": [
                {
                    "query": r.query,
                    "num_results": r.num_results,
                    "max_score": r.max_score,
                    "latency_ms": r.latency_ms,
                    "timestamp": r.timestamp,
                    "sources": r.sources,
                }
                for r in self._records
            ]
        }
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, "w") as f:
            json.dump(data, f, indent=2)

    def record_query(
        self,
        query: str,
        results: List[Any],
        latency_ms: float,
    ) -> None:
        """Record a query and its results.

        Args:
            query: The query text.
            results: List of QueryResult objects (or any objects with score/source attrs).
            latency_ms: Query latency in milliseconds.
        """
        max_score = 0.0
        sources: List[str] = []

        for r in results:
            score = getattr(r, "score", 0.0)
            if score > max_score:
                max_score = score
            source = getattr(r, "source", "unknown")
            if source not in sources:
                sources.append(source)

        record = QueryRecord(
            query=query,
            num_results=len(results),
            max_score=max_score,
            latency_ms=latency_ms,
            timestamp=time.time(),
            sources=sources,
        )
        self._records.append(record)
        self._save()

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of query analytics.

        Returns:
            Dictionary with total queries, avg latency, queries per source,
            zero-result count, and low-confidence count.
        """
        if not self._records:
            return {
                "total_queries": 0,
                "avg_latency_ms": 0.0,
                "queries_per_source": {},
                "zero_result_count": 0,
                "low_confidence_count": 0,
            }

        total = len(self._records)
        avg_latency = sum(r.latency_ms for r in self._records) / total

        # Queries per source
        source_counter: Counter = Counter()
        for record in self._records:
            for source in record.sources:
                source_counter[source] += 1

        zero_result_count = sum(1 for r in self._records if r.num_results == 0)
        low_confidence_count = sum(
            1 for r in self._records if r.max_score < 0.5 and r.num_results > 0
        )

        return {
            "total_queries": total,
            "avg_latency_ms": round(avg_latency, 2),
            "queries_per_source": dict(source_counter),
            "zero_result_count": zero_result_count,
            "low_confidence_count": low_confidence_count,
        }

    def get_zero_result_queries(self) -> List[str]:
        """Get all queries that returned no results.

        Returns:
            List of query strings with zero results.
        """
        return [r.query for r in self._records if r.num_results == 0]

    def get_low_confidence_queries(self, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Get queries with low confidence scores.

        Args:
            threshold: Maximum score threshold (default 0.5).

        Returns:
            List of dicts with query, max_score, and timestamp.
        """
        return [
            {
                "query": r.query,
                "max_score": r.max_score,
                "timestamp": r.timestamp,
            }
            for r in self._records
            if r.max_score < threshold and r.num_results > 0
        ]

    def get_query_frequency(self) -> Dict[str, int]:
        """Get query frequency distribution.

        Returns:
            Dictionary mapping query text to occurrence count.
        """
        counter: Counter = Counter(r.query for r in self._records)
        return dict(counter.most_common())
