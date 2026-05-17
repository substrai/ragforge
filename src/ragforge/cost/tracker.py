"""Cost tracking for RAGForge operations.

Tracks embedding, storage, and retrieval costs with persistence to JSON.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional


# Default pricing per 1K tokens
DEFAULT_PRICING: Dict[str, float] = {
    "titan-embed-v2": 0.0001,
    "titan-embed-text-v1": 0.0001,
    "cohere-embed-english": 0.0001,
    "cohere-embed-multilingual": 0.0001,
    "local/dev": 0.0,
    "dev": 0.0,
    "default": 0.0001,
}


@dataclass
class CostRecord:
    """A single cost record."""

    category: str  # embedding | storage | retrieval
    amount: float
    timestamp: float
    details: Dict[str, Any] = field(default_factory=dict)


class CostTracker:
    """Tracks costs for RAGForge operations.

    Records per-operation costs for embeddings, storage, and retrieval.
    Persists cost records to a JSON file for historical analysis.
    """

    def __init__(
        self,
        storage_path: str | Path = "ragforge_costs.json",
        pricing: Optional[Dict[str, float]] = None,
    ):
        """Initialize cost tracker.

        Args:
            storage_path: Path to the JSON file for persisting cost records.
            pricing: Custom pricing table (model_name -> cost per 1K tokens).
        """
        self.storage_path = Path(storage_path)
        self.pricing = {**DEFAULT_PRICING, **(pricing or {})}
        self._records: List[CostRecord] = []
        self._load()

    def _load(self) -> None:
        """Load existing cost records from storage."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r") as f:
                    data = json.load(f)
                self._records = [
                    CostRecord(
                        category=r["category"],
                        amount=r["amount"],
                        timestamp=r["timestamp"],
                        details=r.get("details", {}),
                    )
                    for r in data.get("records", [])
                ]
            except (json.JSONDecodeError, KeyError):
                self._records = []

    def _save(self) -> None:
        """Persist cost records to storage."""
        data = {
            "records": [
                {
                    "category": r.category,
                    "amount": r.amount,
                    "timestamp": r.timestamp,
                    "details": r.details,
                }
                for r in self._records
            ]
        }
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, "w") as f:
            json.dump(data, f, indent=2)

    def _get_model_price(self, model: str) -> float:
        """Get price per 1K tokens for a model.

        Args:
            model: Model name (may include provider prefix like 'bedrock/').

        Returns:
            Price per 1K tokens.
        """
        # Strip provider prefix
        clean_model = model.split("/")[-1] if "/" in model else model
        return self.pricing.get(clean_model, self.pricing.get("default", 0.0001))

    def record_embedding_cost(self, tokens: int, model: str) -> float:
        """Record cost for an embedding operation.

        Args:
            tokens: Number of tokens embedded.
            model: Model name used for embedding.

        Returns:
            The cost amount recorded.
        """
        price_per_1k = self._get_model_price(model)
        cost = (tokens / 1000.0) * price_per_1k

        record = CostRecord(
            category="embedding",
            amount=cost,
            timestamp=time.time(),
            details={"tokens": tokens, "model": model, "price_per_1k": price_per_1k},
        )
        self._records.append(record)
        self._save()
        return cost

    def record_storage_cost(
        self, vector_count: int, dimensions: int, price_per_gb_month: float = 0.25
    ) -> float:
        """Record estimated storage cost.

        Args:
            vector_count: Number of vectors stored.
            dimensions: Embedding dimensions.
            price_per_gb_month: Price per GB per month.

        Returns:
            The estimated monthly storage cost.
        """
        # float32 = 4 bytes per dimension
        bytes_per_vector = dimensions * 4
        total_bytes = vector_count * bytes_per_vector
        total_gb = total_bytes / (1024**3)
        cost = total_gb * price_per_gb_month

        record = CostRecord(
            category="storage",
            amount=cost,
            timestamp=time.time(),
            details={
                "vector_count": vector_count,
                "dimensions": dimensions,
                "total_gb": total_gb,
            },
        )
        self._records.append(record)
        self._save()
        return cost

    def record_query_cost(self, query: str, tokens: Optional[int] = None, model: Optional[str] = None) -> float:
        """Record cost for a query/retrieval operation.

        Args:
            query: The query text.
            tokens: Number of tokens in the query (estimated from word count if not provided).
            model: Model used for query embedding.

        Returns:
            The cost amount recorded.
        """
        if tokens is None:
            # Rough estimate: ~1.3 tokens per word
            tokens = int(len(query.split()) * 1.3)

        price_per_1k = self._get_model_price(model or "default")
        cost = (tokens / 1000.0) * price_per_1k

        record = CostRecord(
            category="retrieval",
            amount=cost,
            timestamp=time.time(),
            details={"query": query, "tokens": tokens, "model": model},
        )
        self._records.append(record)
        self._save()
        return cost

    def get_total_cost(self) -> float:
        """Get total cost across all categories.

        Returns:
            Total cost in dollars.
        """
        return sum(r.amount for r in self._records)

    def get_cost_breakdown(self) -> Dict[str, float]:
        """Get cost breakdown by category.

        Returns:
            Dictionary mapping category to total cost.
        """
        breakdown: Dict[str, float] = {"embedding": 0.0, "storage": 0.0, "retrieval": 0.0}
        for r in self._records:
            breakdown[r.category] = breakdown.get(r.category, 0.0) + r.amount
        return breakdown

    def get_daily_cost(self, date: Optional[datetime] = None) -> float:
        """Get total cost for a specific day.

        Args:
            date: The date to query. Defaults to today.

        Returns:
            Total cost for the specified day.
        """
        if date is None:
            date = datetime.now()

        day_start = datetime(date.year, date.month, date.day).timestamp()
        day_end = day_start + 86400  # 24 hours in seconds

        return sum(
            r.amount for r in self._records
            if day_start <= r.timestamp < day_end
        )

    def get_monthly_forecast(self) -> float:
        """Forecast monthly cost based on recent daily spending.

        Uses the last 7 days of spending to project monthly cost.

        Returns:
            Projected monthly cost.
        """
        now = datetime.now()
        seven_days_ago = (now - timedelta(days=7)).timestamp()

        recent_cost = sum(
            r.amount for r in self._records
            if r.timestamp >= seven_days_ago
        )

        # Project to 30 days
        if recent_cost == 0:
            return 0.0

        daily_avg = recent_cost / 7.0
        return daily_avg * 30.0

    def get_records(self, category: Optional[str] = None) -> List[CostRecord]:
        """Get cost records, optionally filtered by category.

        Args:
            category: Optional category filter.

        Returns:
            List of cost records.
        """
        if category:
            return [r for r in self._records if r.category == category]
        return list(self._records)
