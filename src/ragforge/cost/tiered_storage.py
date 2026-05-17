"""Tiered storage management for cost optimization.

Manages hot/warm/cold storage tiers based on access frequency.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class AccessRecord:
    """Access record for a chunk."""

    chunk_id: str
    access_count: int = 0
    last_access: float = 0.0
    tier: str = "hot"  # hot | warm | cold


class TieredStorageManager:
    """Manages storage tiers based on access frequency.

    Tiers:
        - hot: Full precision, fast access. Frequently accessed chunks.
        - warm: Quantized storage. Moderately accessed chunks.
        - cold: Archived/removed from index. Rarely accessed chunks.

    Chunks are promoted/demoted based on access frequency within
    a configurable time window.
    """

    def __init__(
        self,
        storage_path: str | Path = "ragforge_tiers.json",
        hot_threshold: int = 5,
        warm_threshold: int = 1,
        window_days: int = 7,
    ):
        """Initialize tiered storage manager.

        Args:
            storage_path: Path to JSON file for persisting tier data.
            hot_threshold: Minimum accesses in window to stay in hot tier.
            warm_threshold: Minimum accesses in window to stay in warm tier.
            window_days: Time window in days for access counting.
        """
        self.storage_path = Path(storage_path)
        self.hot_threshold = hot_threshold
        self.warm_threshold = warm_threshold
        self.window_days = window_days
        self._records: Dict[str, AccessRecord] = {}
        self._access_history: Dict[str, List[float]] = {}
        self._load()

    def _load(self) -> None:
        """Load existing tier data from storage."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r") as f:
                    data = json.load(f)
                for r in data.get("records", []):
                    chunk_id = r["chunk_id"]
                    self._records[chunk_id] = AccessRecord(
                        chunk_id=chunk_id,
                        access_count=r.get("access_count", 0),
                        last_access=r.get("last_access", 0.0),
                        tier=r.get("tier", "hot"),
                    )
                self._access_history = data.get("access_history", {})
            except (json.JSONDecodeError, KeyError):
                self._records = {}
                self._access_history = {}

    def _save(self) -> None:
        """Persist tier data to storage."""
        data = {
            "records": [
                {
                    "chunk_id": r.chunk_id,
                    "access_count": r.access_count,
                    "last_access": r.last_access,
                    "tier": r.tier,
                }
                for r in self._records.values()
            ],
            "access_history": self._access_history,
        }
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, "w") as f:
            json.dump(data, f, indent=2)

    def _get_recent_access_count(self, chunk_id: str) -> int:
        """Get number of accesses within the time window.

        Args:
            chunk_id: The chunk identifier.

        Returns:
            Number of accesses in the configured window.
        """
        history = self._access_history.get(chunk_id, [])
        cutoff = time.time() - (self.window_days * 86400)
        return sum(1 for ts in history if ts >= cutoff)

    def record_access(self, chunk_id: str) -> None:
        """Record an access to a chunk.

        Args:
            chunk_id: The chunk identifier.
        """
        now = time.time()

        if chunk_id not in self._records:
            self._records[chunk_id] = AccessRecord(
                chunk_id=chunk_id,
                access_count=0,
                last_access=0.0,
                tier="hot",
            )

        record = self._records[chunk_id]
        record.access_count += 1
        record.last_access = now

        # Track access history for windowed counting
        if chunk_id not in self._access_history:
            self._access_history[chunk_id] = []
        self._access_history[chunk_id].append(now)

        # Auto-promote if accessed
        if record.tier != "hot":
            recent_count = self._get_recent_access_count(chunk_id)
            if recent_count >= self.hot_threshold:
                record.tier = "hot"
            elif recent_count >= self.warm_threshold and record.tier == "cold":
                record.tier = "warm"

        self._save()

    def get_tier(self, chunk_id: str) -> str:
        """Get the current tier for a chunk.

        Args:
            chunk_id: The chunk identifier.

        Returns:
            Tier name: "hot", "warm", or "cold".
        """
        if chunk_id not in self._records:
            return "hot"  # New chunks default to hot
        return self._records[chunk_id].tier

    def promote(self, chunk_id: str) -> str:
        """Promote a chunk to a higher tier.

        cold → warm → hot

        Args:
            chunk_id: The chunk identifier.

        Returns:
            The new tier after promotion.
        """
        if chunk_id not in self._records:
            self._records[chunk_id] = AccessRecord(
                chunk_id=chunk_id, tier="hot"
            )
            self._save()
            return "hot"

        record = self._records[chunk_id]
        if record.tier == "cold":
            record.tier = "warm"
        elif record.tier == "warm":
            record.tier = "hot"
        # hot stays hot

        self._save()
        return record.tier

    def demote(self, chunk_id: str) -> str:
        """Demote a chunk to a lower tier.

        hot → warm → cold

        Args:
            chunk_id: The chunk identifier.

        Returns:
            The new tier after demotion.
        """
        if chunk_id not in self._records:
            self._records[chunk_id] = AccessRecord(
                chunk_id=chunk_id, tier="warm"
            )
            self._save()
            return "warm"

        record = self._records[chunk_id]
        if record.tier == "hot":
            record.tier = "warm"
        elif record.tier == "warm":
            record.tier = "cold"
        # cold stays cold

        self._save()
        return record.tier

    def evaluate_tiers(self) -> Dict[str, List[str]]:
        """Evaluate all chunks and suggest tier changes based on access patterns.

        Returns:
            Dictionary with 'demote' and 'promote' lists of chunk_ids.
        """
        suggestions: Dict[str, List[str]] = {"demote": [], "promote": []}

        for chunk_id, record in self._records.items():
            recent_count = self._get_recent_access_count(chunk_id)

            if record.tier == "hot" and recent_count < self.warm_threshold:
                suggestions["demote"].append(chunk_id)
            elif record.tier == "warm" and recent_count < self.warm_threshold:
                suggestions["demote"].append(chunk_id)
            elif record.tier == "warm" and recent_count >= self.hot_threshold:
                suggestions["promote"].append(chunk_id)
            elif record.tier == "cold" and recent_count >= self.warm_threshold:
                suggestions["promote"].append(chunk_id)

        return suggestions

    def get_tier_stats(self) -> Dict[str, Any]:
        """Get statistics about tier distribution.

        Returns:
            Dictionary with counts and percentages per tier.
        """
        total = len(self._records)
        if total == 0:
            return {
                "total": 0,
                "hot": {"count": 0, "percentage": 0.0},
                "warm": {"count": 0, "percentage": 0.0},
                "cold": {"count": 0, "percentage": 0.0},
            }

        hot_count = sum(1 for r in self._records.values() if r.tier == "hot")
        warm_count = sum(1 for r in self._records.values() if r.tier == "warm")
        cold_count = sum(1 for r in self._records.values() if r.tier == "cold")

        return {
            "total": total,
            "hot": {"count": hot_count, "percentage": round(hot_count / total * 100, 1)},
            "warm": {"count": warm_count, "percentage": round(warm_count / total * 100, 1)},
            "cold": {"count": cold_count, "percentage": round(cold_count / total * 100, 1)},
        }
