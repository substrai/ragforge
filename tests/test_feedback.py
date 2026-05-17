"""Tests for relevance feedback collection."""

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import pytest

from ragforge.evaluation.feedback import FeedbackCollector


@dataclass
class MockResult:
    """Mock query result for testing feedback boost."""

    chunk_id: str
    score: float
    content: str = ""
    source: str = ""
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class TestFeedbackCollector:
    """Tests for FeedbackCollector class."""

    def _create_collector(self) -> tuple:
        """Create a feedback collector with a temp file."""
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        tmp.close()
        path = Path(tmp.name)
        path.unlink()
        collector = FeedbackCollector(storage_path=path)
        return collector, path

    def test_record_positive_feedback(self):
        """Test recording positive feedback."""
        collector, path = self._create_collector()
        try:
            collector.record_feedback("test query", "chunk1", relevant=True)

            feedback = collector.get_feedback_for_query("test query")
            assert len(feedback) == 1
            assert feedback[0]["chunk_id"] == "chunk1"
            assert feedback[0]["relevant"] is True
        finally:
            if path.exists():
                path.unlink()

    def test_record_negative_feedback(self):
        """Test recording negative feedback."""
        collector, path = self._create_collector()
        try:
            collector.record_feedback("test query", "chunk1", relevant=False)

            feedback = collector.get_feedback_for_query("test query")
            assert len(feedback) == 1
            assert feedback[0]["relevant"] is False
        finally:
            if path.exists():
                path.unlink()

    def test_multiple_feedback_same_query(self):
        """Test multiple feedback entries for same query."""
        collector, path = self._create_collector()
        try:
            collector.record_feedback("q1", "c1", relevant=True)
            collector.record_feedback("q1", "c2", relevant=False)
            collector.record_feedback("q1", "c3", relevant=True)

            feedback = collector.get_feedback_for_query("q1")
            assert len(feedback) == 3
        finally:
            if path.exists():
                path.unlink()

    def test_get_feedback_different_queries(self):
        """Test that feedback is filtered by query."""
        collector, path = self._create_collector()
        try:
            collector.record_feedback("q1", "c1", relevant=True)
            collector.record_feedback("q2", "c2", relevant=True)

            feedback_q1 = collector.get_feedback_for_query("q1")
            feedback_q2 = collector.get_feedback_for_query("q2")

            assert len(feedback_q1) == 1
            assert feedback_q1[0]["chunk_id"] == "c1"
            assert len(feedback_q2) == 1
            assert feedback_q2[0]["chunk_id"] == "c2"
        finally:
            if path.exists():
                path.unlink()

    def test_get_positive_chunks(self):
        """Test getting all positively-rated chunks."""
        collector, path = self._create_collector()
        try:
            collector.record_feedback("q1", "c1", relevant=True)
            collector.record_feedback("q1", "c2", relevant=False)
            collector.record_feedback("q2", "c3", relevant=True)
            collector.record_feedback("q2", "c1", relevant=True)  # duplicate positive

            positive = collector.get_positive_chunks()
            assert set(positive) == {"c1", "c3"}
        finally:
            if path.exists():
                path.unlink()

    def test_apply_feedback_boost(self):
        """Test that positive chunks get score boost."""
        collector, path = self._create_collector()
        try:
            collector.record_feedback("q1", "c1", relevant=True)

            results = [
                MockResult(chunk_id="c1", score=0.5),
                MockResult(chunk_id="c2", score=0.7),
            ]

            boosted = collector.apply_feedback_boost(results, boost_factor=0.1)

            # c1 should be boosted from 0.5 to 0.6
            # c2 stays at 0.7
            # After sorting: c2 (0.7), c1 (0.6)
            assert boosted[0].chunk_id == "c2"
            assert boosted[0].score == 0.7
            assert boosted[1].chunk_id == "c1"
            assert boosted[1].score == pytest.approx(0.6)
        finally:
            if path.exists():
                path.unlink()

    def test_apply_feedback_boost_reorders(self):
        """Test that boost can reorder results."""
        collector, path = self._create_collector()
        try:
            collector.record_feedback("q1", "c2", relevant=True)

            results = [
                MockResult(chunk_id="c1", score=0.6),
                MockResult(chunk_id="c2", score=0.55),
            ]

            boosted = collector.apply_feedback_boost(results, boost_factor=0.1)

            # c2 boosted from 0.55 to 0.65, now higher than c1 (0.6)
            assert boosted[0].chunk_id == "c2"
            assert boosted[0].score == pytest.approx(0.65)
        finally:
            if path.exists():
                path.unlink()

    def test_persistence(self):
        """Test that feedback persists to file."""
        collector, path = self._create_collector()
        try:
            collector.record_feedback("q1", "c1", relevant=True)

            # Create new instance from same file
            collector2 = FeedbackCollector(storage_path=path)
            positive = collector2.get_positive_chunks()
            assert "c1" in positive
        finally:
            if path.exists():
                path.unlink()

    def test_no_feedback_for_query(self):
        """Test getting feedback for a query with no entries."""
        collector, path = self._create_collector()
        try:
            feedback = collector.get_feedback_for_query("nonexistent")
            assert feedback == []
        finally:
            if path.exists():
                path.unlink()

    def test_empty_positive_chunks(self):
        """Test getting positive chunks when none exist."""
        collector, path = self._create_collector()
        try:
            collector.record_feedback("q1", "c1", relevant=False)
            positive = collector.get_positive_chunks()
            assert positive == []
        finally:
            if path.exists():
                path.unlink()
