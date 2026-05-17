"""Budget enforcement for RAGForge operations.

Checks budgets before operations and takes configured actions when exceeded.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

from ragforge.cost.tracker import CostTracker

logger = logging.getLogger(__name__)


class BudgetExceededError(Exception):
    """Raised when a budget limit is exceeded and action is 'block'."""

    def __init__(self, message: str, budget_type: str, limit: float, current: float):
        super().__init__(message)
        self.budget_type = budget_type
        self.limit = limit
        self.current = current


@dataclass
class BudgetStatus:
    """Current budget status."""

    daily_limit: float
    daily_spent: float
    daily_remaining: float
    monthly_limit: float
    monthly_spent: float
    monthly_remaining: float
    is_exceeded: bool
    exceeded_type: Optional[str] = None  # "daily" | "monthly" | None


class BudgetEnforcer:
    """Enforces budget limits on RAGForge operations.

    Checks daily and monthly budgets before operations and takes
    configured actions when limits are exceeded.

    Actions on exceed:
        - "block": Raise BudgetExceededError
        - "downgrade": Switch to cheaper model (returns downgrade signal)
        - "alert": Log warning but allow operation to continue
    """

    def __init__(
        self,
        cost_tracker: CostTracker,
        daily_budget: float = 10.0,
        monthly_budget: float = 100.0,
        action_on_exceed: str = "alert",
    ):
        """Initialize budget enforcer.

        Args:
            cost_tracker: CostTracker instance for reading current costs.
            daily_budget: Maximum daily spend in dollars.
            monthly_budget: Maximum monthly spend in dollars.
            action_on_exceed: Action when budget exceeded ("block", "downgrade", "alert").
        """
        self.cost_tracker = cost_tracker
        self.daily_budget = daily_budget
        self.monthly_budget = monthly_budget
        self.action_on_exceed = action_on_exceed

    def _get_monthly_spent(self) -> float:
        """Get total spending for the current month."""
        now = datetime.now()
        month_start = datetime(now.year, now.month, 1).timestamp()
        return sum(
            r.amount for r in self.cost_tracker.get_records()
            if r.timestamp >= month_start
        )

    def get_budget_status(self) -> BudgetStatus:
        """Get current budget status.

        Returns:
            BudgetStatus with current spending and remaining budget.
        """
        daily_spent = self.cost_tracker.get_daily_cost()
        monthly_spent = self._get_monthly_spent()

        daily_remaining = max(0.0, self.daily_budget - daily_spent)
        monthly_remaining = max(0.0, self.monthly_budget - monthly_spent)

        is_exceeded = daily_spent >= self.daily_budget or monthly_spent >= self.monthly_budget
        exceeded_type = None
        if daily_spent >= self.daily_budget:
            exceeded_type = "daily"
        elif monthly_spent >= self.monthly_budget:
            exceeded_type = "monthly"

        return BudgetStatus(
            daily_limit=self.daily_budget,
            daily_spent=daily_spent,
            daily_remaining=daily_remaining,
            monthly_limit=self.monthly_budget,
            monthly_spent=monthly_spent,
            monthly_remaining=monthly_remaining,
            is_exceeded=is_exceeded,
            exceeded_type=exceeded_type,
        )

    def can_proceed(self) -> bool:
        """Check if operations can proceed within budget.

        Returns:
            True if within budget or action is not "block".
        """
        status = self.get_budget_status()
        if not status.is_exceeded:
            return True
        return self.action_on_exceed != "block"

    def check_budget(self) -> str:
        """Check budget and take configured action if exceeded.

        Returns:
            "ok" if within budget, "downgrade" if should switch models,
            "alert" if warning was logged.

        Raises:
            BudgetExceededError: If budget exceeded and action is "block".
        """
        status = self.get_budget_status()

        if not status.is_exceeded:
            return "ok"

        if self.action_on_exceed == "block":
            limit = (
                self.daily_budget if status.exceeded_type == "daily"
                else self.monthly_budget
            )
            current = (
                status.daily_spent if status.exceeded_type == "daily"
                else status.monthly_spent
            )
            raise BudgetExceededError(
                f"Budget exceeded: {status.exceeded_type} limit ${limit:.4f}, "
                f"current spend ${current:.4f}",
                budget_type=status.exceeded_type or "unknown",
                limit=limit,
                current=current,
            )

        if self.action_on_exceed == "downgrade":
            logger.warning(
                "Budget exceeded (%s). Switching to cheaper model.",
                status.exceeded_type,
            )
            return "downgrade"

        # Default: alert
        logger.warning(
            "Budget warning: %s budget exceeded (spent: $%.4f / limit: $%.4f)",
            status.exceeded_type,
            status.daily_spent if status.exceeded_type == "daily" else status.monthly_spent,
            self.daily_budget if status.exceeded_type == "daily" else self.monthly_budget,
        )
        return "alert"

    def get_remaining_budget(self) -> Dict[str, float]:
        """Get remaining budget for daily and monthly periods.

        Returns:
            Dictionary with 'daily' and 'monthly' remaining amounts.
        """
        status = self.get_budget_status()
        return {
            "daily": status.daily_remaining,
            "monthly": status.monthly_remaining,
        }
