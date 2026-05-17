"""Tests for query analytics tracking."""

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import pytest

from ragforge.evaluation.analytics import QueryAnalytics


@dataclass
class MockResult:
    """Mock query result for testing."""

    score: float
    source: str


class TestQueryAnalytics:
    """Tests for QueryAnalytics class."""

    def _create_analytics(self) -> tuple:
        """Create analytics with a temp file."""
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        tmp.close()
        path = Path(tmp.name)
        # Remove the file so analytics starts fresh
        path.unlink()
        analytics = QueryAnalytics(storage_path=path)
        return analytics, path

    def test_record_query(self):
        """Test recording a query."""
        analytics, path = self._create_analytics()
        try:
            results = [MockResult(score=0.8, source="doc1")]
            analytics.record_query("test query", results, latency_ms=50.0)

            summary = analytics.get_summary()
            assert summary["total_queries"] == 1
            assert summary["avg_latency_ms"] == 50.0
        finally:
            if path.exists():
                path.unlink()

    def test_multiple_queries(self):
        """Test recording multiple queries."""
        analytics, path = self._create_analytics()
        try:
            analytics.record_query("q1", [MockResult(score=0.9, source="s1")], 30.0)
            analytics.record_query("q2", [MockResult(score=0.7, source="s2")], 50.0)
            analytics.record_query("q3", [MockResult(score=0.3, source="s1")], 70.0)

            summary = analytics.get_summary()
            assert summary["total_queries"] == 3
            assert summary["avg_latency_ms"] == 50.0
        finally:
            if path.exists():
                path.unlink()

    def test_zero_result_queries(self):
        """Test tracking zero-result queries."""
        analytics, path = self._create_analytics()
        try:
            analytics.record_query("found", [MockResult(score=0.8, source="s1")], 30.0)
            analytics.record_query("not found", [], 20.0)
            analytics.record_query("also not found", [], 25.0)

            zero_queries = analytics.get_zero_result_queries()
            assert len(zero_queries) == 2
            assert "not found" in zero_queries
            assert "also not found" in zero_queries

            summary = analytics.get_summary()
            assert summary["zero_result_count"] == 2
        finally:
            if path.exists():
                path.unlink()

    def test_low_confidence_queries(self):
        """Test tracking low-confidence queries."""
        analytics, path = self._create_analytics()
        try:
            analytics.record_query("high", [MockResult(score=0.9, source="s1")], 30.0)
            analytics.record_query("low", [MockResult(score=0.3, source="s1")], 30.0)
            analytics.record_query("medium", [MockResult(score=0.5, source="s1")], 30.0)

            low_conf = analytics.get_low_confidence_queries(threshold=0.5)
            assert len(low_conf) == 1
            assert low_conf[0]["query"] == "low"
            assert low_conf[0]["max_score"] == 0.3
        finally:
            if path.exists():
                path.unlink()

    def test_queries_per_source(self):
        """Test source tracking."""
        analytics, path = self._create_analytics()
        try:
            analytics.record_query("q1", [MockResult(score=0.8, source="doc1")], 30.0)
            analytics.record_query("q2", [MockResult(score=0.7, source="doc1")], 30.0)
            analytics.record_query("q3", [MockResult(score=0.6, source="doc2")], 30.0)

            summary = analytics.get_summary()
            assert summary["queries_per_source"]["doc1"] == 2
            assert summary["queries_per_source"]["doc2"] == 1
        finally:
            if path.exists():
                path.unlink()

    def test_persistence(self):
        """Test that analytics persist to file and reload."""
        analytics, path = self._create_analytics()
        try:
            analytics.record_query("persisted", [MockResult(score=0.8, source="s1")], 40.0)

            # Create new instance from same file
            analytics2 = QueryAnalytics(storage_path=path)
            summary = analytics2.get_summary()
            assert summary["total_queries"] == 1
        finally:
            if path.exists():
                path.unlink()

    def test_empty_summary(self):
        """Test summary with no recorded queries."""
        analytics, path = self._create_analytics()
        try:
            summary = analytics.get_summary()
            assert summary["total_queries"] == 0
            assert summary["avg_latency_ms"] == 0.0
            assert summary["zero_result_count"] == 0
        finally:
            if path.exists():
                path.unlink()

    def test_query_frequency(self):
        """Test query frequency distribution."""
        analytics, path = self._create_analytics()
        try:
            analytics.record_query("common", [MockResult(score=0.8, source="s")], 30.0)
            analytics.record_query("common", [MockResult(score=0.7, source="s")], 30.0)
            analytics.record_query("rare", [MockResult(score=0.6, source="s")], 30.0)

            freq = analytics.get_query_frequency()
            assert freq["common"] == 2
            assert freq["rare"] == 1
        finally:
            if path.exists():
                path.unlink()

    def test_multiple_sources_per_query(self):
        """Test query with results from multiple sources."""
        analytics, path = self._create_analytics()
        try:
            results = [
                MockResult(score=0.9, source="doc1"),
                MockResult(score=0.7, source="doc2"),
            ]
            analytics.record_query("multi", results, 30.0)

            summary = analytics.get_summary()
            assert "doc1" in summary["queries_per_source"]
            assert "doc2" in summary["queries_per_source"]
        finally:
            if path.exists():
                path.unlink()
