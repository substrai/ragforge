"""Drift detection for RAGForge retrieval quality.

Monitors retrieval metrics over time and detects significant quality drops
compared to a stored baseline.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class DriftReport:
    """Report for a single metric's drift status."""

    metric_name: str
    baseline_value: float
    current_value: float
    drop_percentage: float
    is_drifted: bool


class DriftDetector:
    """Detects quality drift by comparing current metrics against a baseline.

    Stores a baseline set of metrics (from an initial evaluation run) and
    compares subsequent evaluation results to detect significant drops.
    """

    def __init__(
        self,
        threshold: float = 0.10,
        storage_path: str | Path = "ragforge_baseline.json",
    ):
        """Initialize drift detector.

        Args:
            threshold: Percentage drop threshold to trigger drift alert (default 10%).
            storage_path: Path to the JSON file for persisting baseline metrics.
        """
        self.threshold = threshold
        self.storage_path = Path(storage_path)
        self._baseline: Optional[Dict[str, float]] = None
        self._load_baseline()

    def _load_baseline(self) -> None:
        """Load baseline from storage if it exists."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r") as f:
                    data = json.load(f)
                self._baseline = data.get("metrics")
            except (json.JSONDecodeError, KeyError):
                self._baseline = None

    def set_baseline(self, metrics: Dict[str, float]) -> None:
        """Set the baseline metrics for drift comparison.

        Args:
            metrics: Dictionary of metric names to their baseline values.
        """
        self._baseline = dict(metrics)
        self.save_baseline()

    def save_baseline(self) -> None:
        """Persist baseline metrics to storage."""
        if self._baseline is None:
            return

        data = {
            "metrics": self._baseline,
            "timestamp": time.time(),
        }
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, "w") as f:
            json.dump(data, f, indent=2)

    def load_baseline(self) -> Optional[Dict[str, float]]:
        """Load and return the stored baseline metrics.

        Returns:
            Dictionary of baseline metrics, or None if no baseline exists.
        """
        self._load_baseline()
        return self._baseline

    def check_drift(self, current_metrics: Dict[str, float]) -> List[DriftReport]:
        """Compare current metrics against baseline and detect drift.

        Args:
            current_metrics: Dictionary of current metric values.

        Returns:
            List of DriftReport objects, one per metric that exists in both
            baseline and current metrics.

        Raises:
            ValueError: If no baseline has been set.
        """
        if self._baseline is None:
            raise ValueError(
                "No baseline set. Call set_baseline() first or load from storage."
            )

        reports: List[DriftReport] = []

        for metric_name, baseline_value in self._baseline.items():
            if metric_name not in current_metrics:
                continue

            current_value = current_metrics[metric_name]

            # Calculate drop percentage
            if baseline_value > 0:
                drop_percentage = (baseline_value - current_value) / baseline_value
            else:
                drop_percentage = 0.0

            is_drifted = drop_percentage >= self.threshold

            reports.append(
                DriftReport(
                    metric_name=metric_name,
                    baseline_value=baseline_value,
                    current_value=current_value,
                    drop_percentage=drop_percentage,
                    is_drifted=is_drifted,
                )
            )

        return reports
