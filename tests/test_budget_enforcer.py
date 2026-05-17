"""Tests for budget enforcement module."""

import tempfile
import time
from pathlib import Path

import pytest

from ragforge.cost.enforcer import BudgetEnforcer, BudgetExceededError, BudgetStatus
from ragforge.cost.tracker import CostTracker


class TestBudgetEnforcer:
    """Tests for BudgetEnforcer class."""

    def _create_enforcer(self, daily=10.0, monthly=100.0, action="alert") -> tuple:
        """Create an enforcer with a temp cost tracker."""
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        tmp.close()
        path = Path(tmp.name)
        path.unlink()
        tracker = CostTracker(storage_path=path)
        enforcer = BudgetEnforcer(
            cost_tracker=tracker,
            daily_budget=daily,
            monthly_budget=monthly,
            action_on_exceed=action,
        )
        return enforcer, tracker, path

    def test_within_budget(self):
        """Test that operations proceed when within budget."""
        enforcer, tracker, path = self._create_enforcer()
        try:
            tracker.record_embedding_cost(tokens=1000, model="titan-embed-v2")
            assert enforcer.can_proceed() is True
            assert enforcer.check_budget() == "ok"
        finally:
            if path.exists():
                path.unlink()

    def test_budget_exceeded_block(self):
        """Test that BudgetExceededError is raised when action is 'block'."""
        enforcer, tracker, path = self._create_enforcer(daily=0.0001, action="block")
        try:
            # Exceed the tiny budget
            tracker.record_embedding_cost(tokens=10000, model="titan-embed-v2")

            assert enforcer.can_proceed() is False

            with pytest.raises(BudgetExceededError) as exc_info:
                enforcer.check_budget()

            assert exc_info.value.budget_type == "daily"
            assert exc_info.value.limit == 0.0001
        finally:
            if path.exists():
                path.unlink()

    def test_budget_exceeded_downgrade(self):
        """Test downgrade action when budget exceeded."""
        enforcer, tracker, path = self._create_enforcer(daily=0.0001, action="downgrade")
        try:
            tracker.record_embedding_cost(tokens=10000, model="titan-embed-v2")

            assert enforcer.can_proceed() is True
            result = enforcer.check_budget()
            assert result == "downgrade"
        finally:
            if path.exists():
                path.unlink()

    def test_budget_exceeded_alert(self):
        """Test alert action when budget exceeded."""
        enforcer, tracker, path = self._create_enforcer(daily=0.0001, action="alert")
        try:
            tracker.record_embedding_cost(tokens=10000, model="titan-embed-v2")

            assert enforcer.can_proceed() is True
            result = enforcer.check_budget()
            assert result == "alert"
        finally:
            if path.exists():
                path.unlink()

    def test_get_budget_status(self):
        """Test getting budget status."""
        enforcer, tracker, path = self._create_enforcer(daily=10.0, monthly=100.0)
        try:
            tracker.record_embedding_cost(tokens=1000, model="titan-embed-v2")

            status = enforcer.get_budget_status()
            assert isinstance(status, BudgetStatus)
            assert status.daily_limit == 10.0
            assert status.monthly_limit == 100.0
            assert status.daily_spent > 0
            assert status.daily_remaining < 10.0
            assert status.is_exceeded is False
        finally:
            if path.exists():
                path.unlink()

    def test_get_remaining_budget(self):
        """Test getting remaining budget amounts."""
        enforcer, tracker, path = self._create_enforcer(daily=10.0, monthly=100.0)
        try:
            remaining = enforcer.get_remaining_budget()
            assert remaining["daily"] == 10.0
            assert remaining["monthly"] == 100.0

            tracker.record_embedding_cost(tokens=1000, model="titan-embed-v2")
            remaining = enforcer.get_remaining_budget()
            assert remaining["daily"] < 10.0
            assert remaining["monthly"] < 100.0
        finally:
            if path.exists():
                path.unlink()

    def test_monthly_budget_exceeded(self):
        """Test monthly budget detection."""
        enforcer, tracker, path = self._create_enforcer(
            daily=100.0, monthly=0.0001, action="block"
        )
        try:
            tracker.record_embedding_cost(tokens=10000, model="titan-embed-v2")

            status = enforcer.get_budget_status()
            assert status.is_exceeded is True
            assert status.exceeded_type == "monthly"
        finally:
            if path.exists():
                path.unlink()

    def test_zero_spend_within_budget(self):
        """Test that zero spend is always within budget."""
        enforcer, tracker, path = self._create_enforcer()
        try:
            status = enforcer.get_budget_status()
            assert status.is_exceeded is False
            assert status.daily_spent == 0.0
            assert status.monthly_spent == 0.0
        finally:
            if path.exists():
                path.unlink()

    def test_budget_exceeded_error_attributes(self):
        """Test BudgetExceededError has correct attributes."""
        error = BudgetExceededError(
            "test message",
            budget_type="daily",
            limit=10.0,
            current=15.0,
        )
        assert str(error) == "test message"
        assert error.budget_type == "daily"
        assert error.limit == 10.0
        assert error.current == 15.0
