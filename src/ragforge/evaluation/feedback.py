"""Relevance feedback collection for RAGForge.

Collects user feedback on retrieval results and applies feedback-based
score boosting to improve future retrievals.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class FeedbackEntry:
    """A single feedback entry."""

    query: str
    chunk_id: str
    relevant: bool
    timestamp: float


class FeedbackCollector:
    """Collects and manages user relevance feedback.

    Records whether retrieved chunks were relevant to the user's query,
    and uses this feedback to boost scores of positively-rated chunks.
    """

    def __init__(self, storage_path: str | Path = "ragforge_feedback.json"):
        """Initialize feedback collector.

        Args:
            storage_path: Path to the JSON file for persisting feedback.
        """
        self.storage_path = Path(storage_path)
        self._entries: List[FeedbackEntry] = []
        self._load()

    def _load(self) -> None:
        """Load existing feedback from storage."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r") as f:
                    data = json.load(f)
                self._entries = [
                    FeedbackEntry(
                        query=e["query"],
                        chunk_id=e["chunk_id"],
                        relevant=e["relevant"],
                        timestamp=e["timestamp"],
                    )
                    for e in data.get("entries", [])
                ]
            except (json.JSONDecodeError, KeyError):
                self._entries = []

    def _save(self) -> None:
        """Persist feedback to storage."""
        data = {
            "entries": [
                {
                    "query": e.query,
                    "chunk_id": e.chunk_id,
                    "relevant": e.relevant,
                    "timestamp": e.timestamp,
                }
                for e in self._entries
            ]
        }
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, "w") as f:
            json.dump(data, f, indent=2)

    def record_feedback(self, query: str, chunk_id: str, relevant: bool) -> None:
        """Record user feedback for a query-chunk pair.

        Args:
            query: The query text.
            chunk_id: The chunk ID that was retrieved.
            relevant: Whether the user found this chunk relevant.
        """
        entry = FeedbackEntry(
            query=query,
            chunk_id=chunk_id,
            relevant=relevant,
            timestamp=time.time(),
        )
        self._entries.append(entry)
        self._save()

    def get_feedback_for_query(self, query: str) -> List[Dict[str, Any]]:
        """Get all feedback entries for a specific query.

        Args:
            query: The query text to look up.

        Returns:
            List of feedback entries as dictionaries.
        """
        return [
            {
                "chunk_id": e.chunk_id,
                "relevant": e.relevant,
                "timestamp": e.timestamp,
            }
            for e in self._entries
            if e.query == query
        ]

    def get_positive_chunks(self) -> List[str]:
        """Get all chunk IDs that received positive feedback.

        Returns:
            List of unique chunk IDs with at least one positive feedback.
        """
        positive_ids: set = set()
        for entry in self._entries:
            if entry.relevant:
                positive_ids.add(entry.chunk_id)
        return list(positive_ids)

    def apply_feedback_boost(
        self, results: List[Any], boost_factor: float = 0.1
    ) -> List[Any]:
        """Boost scores of positively-rated chunks in results.

        Increases the score of results whose chunk_id has received positive
        feedback. Results are re-sorted by boosted score.

        Args:
            results: List of result objects with chunk_id and score attributes.
            boost_factor: Amount to add to scores of positive chunks (default 0.1).

        Returns:
            Results list with boosted scores, re-sorted descending.
        """
        positive_chunks = set(self.get_positive_chunks())

        boosted = []
        for result in results:
            chunk_id = getattr(result, "chunk_id", None)
            score = getattr(result, "score", 0.0)

            if chunk_id in positive_chunks:
                # Create a copy-like approach: modify score in place or wrap
                if hasattr(result, "_replace"):
                    # namedtuple
                    result = result._replace(score=score + boost_factor)
                elif hasattr(result, "__dataclass_fields__"):
                    # dataclass - create new instance
                    from dataclasses import asdict

                    d = asdict(result)
                    d["score"] = score + boost_factor
                    result = type(result)(**d)
                else:
                    # Try direct attribute setting
                    try:
                        result.score = score + boost_factor
                    except AttributeError:
                        pass

            boosted.append(result)

        # Sort by score descending
        boosted.sort(key=lambda r: getattr(r, "score", 0.0), reverse=True)
        return boosted
