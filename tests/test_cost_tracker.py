"""Tests for cost tracking module."""

import json
import tempfile
import time
from pathlib import Path

import pytest

from ragforge.cost.tracker import CostTracker, DEFAULT_PRICING


class TestCostTracker:
    """Tests for CostTracker class."""

    def _create_tracker(self) -> tuple:
        """Create a tracker with a temp file."""
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        tmp.close()
        path = Path(tmp.name)
        path.unlink()
        tracker = CostTracker(storage_path=path)
        return tracker, path

    def test_record_embedding_cost(self):
        """Test recording embedding cost."""
        tracker, path = self._create_tracker()
        try:
            cost = tracker.record_embedding_cost(tokens=1000, model="titan-embed-v2")
            assert cost == pytest.approx(0.0001)
            assert tracker.get_total_cost() == pytest.approx(0.0001)
        finally:
            if path.exists():
                path.unlink()

    def test_record_embedding_cost_with_prefix(self):
        """Test recording cost with provider prefix in model name."""
        tracker, path = self._create_tracker()
        try:
            cost = tracker.record_embedding_cost(tokens=2000, model="bedrock/titan-embed-v2")
            assert cost == pytest.approx(0.0002)
        finally:
            if path.exists():
                path.unlink()

    def test_record_query_cost(self):
        """Test recording query cost."""
        tracker, path = self._create_tracker()
        try:
            cost = tracker.record_query_cost(query="what is RAG?", tokens=5)
            assert cost > 0
        finally:
            if path.exists():
                path.unlink()

    def test_record_query_cost_auto_tokens(self):
        """Test query cost with automatic token estimation."""
        tracker, path = self._create_tracker()
        try:
            cost = tracker.record_query_cost(query="hello world test query")
            # 4 words * 1.3 ≈ 5 tokens
            assert cost > 0
        finally:
            if path.exists():
                path.unlink()

    def test_record_storage_cost(self):
        """Test recording storage cost."""
        tracker, path = self._create_tracker()
        try:
            cost = tracker.record_storage_cost(
                vector_count=10000, dimensions=1024, price_per_gb_month=0.25
            )
            assert cost > 0
        finally:
            if path.exists():
                path.unlink()

    def test_get_cost_breakdown(self):
        """Test cost breakdown by category."""
        tracker, path = self._create_tracker()
        try:
            tracker.record_embedding_cost(tokens=1000, model="titan-embed-v2")
            tracker.record_query_cost(query="test", tokens=100)
            tracker.record_storage_cost(vector_count=100, dimensions=384)

            breakdown = tracker.get_cost_breakdown()
            assert "embedding" in breakdown
            assert "retrieval" in breakdown
            assert "storage" in breakdown
            assert breakdown["embedding"] > 0
            assert breakdown["retrieval"] > 0
            assert breakdown["storage"] > 0
        finally:
            if path.exists():
                path.unlink()

    def test_get_daily_cost(self):
        """Test daily cost calculation."""
        tracker, path = self._create_tracker()
        try:
            tracker.record_embedding_cost(tokens=1000, model="titan-embed-v2")
            tracker.record_embedding_cost(tokens=2000, model="titan-embed-v2")

            daily = tracker.get_daily_cost()
            assert daily == pytest.approx(0.0003)
        finally:
            if path.exists():
                path.unlink()

    def test_get_monthly_forecast(self):
        """Test monthly forecast calculation."""
        tracker, path = self._create_tracker()
        try:
            # Record some costs
            tracker.record_embedding_cost(tokens=10000, model="titan-embed-v2")

            forecast = tracker.get_monthly_forecast()
            # Should project based on last 7 days
            assert forecast >= 0
        finally:
            if path.exists():
                path.unlink()

    def test_persistence(self):
        """Test that cost records persist to file."""
        tracker, path = self._create_tracker()
        try:
            tracker.record_embedding_cost(tokens=1000, model="titan-embed-v2")

            # Create new tracker from same file
            tracker2 = CostTracker(storage_path=path)
            assert tracker2.get_total_cost() == pytest.approx(0.0001)
        finally:
            if path.exists():
                path.unlink()

    def test_custom_pricing(self):
        """Test custom pricing table."""
        tracker, path = self._create_tracker()
        try:
            custom_pricing = {"my-model": 0.001}
            tracker = CostTracker(storage_path=path, pricing=custom_pricing)
            cost = tracker.record_embedding_cost(tokens=1000, model="my-model")
            assert cost == pytest.approx(0.001)
        finally:
            if path.exists():
                path.unlink()

    def test_free_local_model(self):
        """Test that local/dev model has zero cost."""
        tracker, path = self._create_tracker()
        try:
            cost = tracker.record_embedding_cost(tokens=1000, model="local/dev")
            assert cost == 0.0
        finally:
            if path.exists():
                path.unlink()

    def test_get_records_filtered(self):
        """Test getting records filtered by category."""
        tracker, path = self._create_tracker()
        try:
            tracker.record_embedding_cost(tokens=1000, model="titan-embed-v2")
            tracker.record_query_cost(query="test", tokens=100)

            embedding_records = tracker.get_records(category="embedding")
            assert len(embedding_records) == 1
            assert embedding_records[0].category == "embedding"

            all_records = tracker.get_records()
            assert len(all_records) == 2
        finally:
            if path.exists():
                path.unlink()

    def test_empty_tracker(self):
        """Test empty tracker returns zeros."""
        tracker, path = self._create_tracker()
        try:
            assert tracker.get_total_cost() == 0.0
            assert tracker.get_daily_cost() == 0.0
            assert tracker.get_monthly_forecast() == 0.0
            breakdown = tracker.get_cost_breakdown()
            assert all(v == 0.0 for v in breakdown.values())
        finally:
            if path.exists():
                path.unlink()
