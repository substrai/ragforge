"""Tests for drift detection."""

import tempfile
from pathlib import Path

import pytest

from ragforge.evaluation.drift import DriftDetector, DriftReport


class TestDriftDetector:
    """Tests for DriftDetector class."""

    def _create_detector(self, threshold: float = 0.10) -> tuple:
        """Create a drift detector with a temp file."""
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        tmp.close()
        path = Path(tmp.name)
        path.unlink()
        detector = DriftDetector(threshold=threshold, storage_path=path)
        return detector, path

    def test_set_baseline(self):
        """Test setting baseline metrics."""
        detector, path = self._create_detector()
        try:
            metrics = {"precision": 0.8, "recall": 0.7, "mrr": 0.9}
            detector.set_baseline(metrics)

            loaded = detector.load_baseline()
            assert loaded == metrics
        finally:
            if path.exists():
                path.unlink()

    def test_no_drift(self):
        """Test that stable metrics don't trigger drift."""
        detector, path = self._create_detector(threshold=0.10)
        try:
            detector.set_baseline({"precision": 0.8, "recall": 0.7})

            current = {"precision": 0.78, "recall": 0.68}
            reports = detector.check_drift(current)

            assert len(reports) == 2
            for report in reports:
                assert report.is_drifted is False
        finally:
            if path.exists():
                path.unlink()

    def test_drift_detected(self):
        """Test that significant drops trigger drift."""
        detector, path = self._create_detector(threshold=0.10)
        try:
            detector.set_baseline({"precision": 0.8, "recall": 0.7})

            # 25% drop in precision, 28% drop in recall
            current = {"precision": 0.6, "recall": 0.5}
            reports = detector.check_drift(current)

            assert len(reports) == 2
            for report in reports:
                assert report.is_drifted is True
                assert report.drop_percentage > 0.10
        finally:
            if path.exists():
                path.unlink()

    def test_mixed_drift(self):
        """Test that only drifted metrics are flagged."""
        detector, path = self._create_detector(threshold=0.10)
        try:
            detector.set_baseline({"precision": 0.8, "recall": 0.7, "mrr": 0.9})

            # precision drops 25%, recall stable, mrr drops 5%
            current = {"precision": 0.6, "recall": 0.68, "mrr": 0.855}
            reports = detector.check_drift(current)

            drift_map = {r.metric_name: r for r in reports}
            assert drift_map["precision"].is_drifted is True
            assert drift_map["recall"].is_drifted is False
            assert drift_map["mrr"].is_drifted is False
        finally:
            if path.exists():
                path.unlink()

    def test_no_baseline_raises(self):
        """Test that checking drift without baseline raises error."""
        detector, path = self._create_detector()
        try:
            with pytest.raises(ValueError, match="No baseline set"):
                detector.check_drift({"precision": 0.5})
        finally:
            if path.exists():
                path.unlink()

    def test_drop_percentage_calculation(self):
        """Test correct drop percentage calculation."""
        detector, path = self._create_detector(threshold=0.10)
        try:
            detector.set_baseline({"metric": 1.0})

            current = {"metric": 0.75}
            reports = detector.check_drift(current)

            assert len(reports) == 1
            assert reports[0].drop_percentage == pytest.approx(0.25)
            assert reports[0].baseline_value == 1.0
            assert reports[0].current_value == 0.75
        finally:
            if path.exists():
                path.unlink()

    def test_improvement_not_drift(self):
        """Test that improvement (negative drop) is not flagged as drift."""
        detector, path = self._create_detector(threshold=0.10)
        try:
            detector.set_baseline({"precision": 0.7})

            # Improvement
            current = {"precision": 0.85}
            reports = detector.check_drift(current)

            assert len(reports) == 1
            assert reports[0].is_drifted is False
            assert reports[0].drop_percentage < 0  # Negative = improvement
        finally:
            if path.exists():
                path.unlink()

    def test_persistence(self):
        """Test that baseline persists to file."""
        detector, path = self._create_detector()
        try:
            detector.set_baseline({"precision": 0.8})

            # Create new instance from same file
            detector2 = DriftDetector(storage_path=path)
            baseline = detector2.load_baseline()
            assert baseline == {"precision": 0.8}
        finally:
            if path.exists():
                path.unlink()

    def test_missing_metric_in_current(self):
        """Test that metrics not in current are skipped."""
        detector, path = self._create_detector()
        try:
            detector.set_baseline({"precision": 0.8, "recall": 0.7})

            # Only precision in current
            current = {"precision": 0.75}
            reports = detector.check_drift(current)

            assert len(reports) == 1
            assert reports[0].metric_name == "precision"
        finally:
            if path.exists():
                path.unlink()

    def test_zero_baseline_value(self):
        """Test handling of zero baseline value."""
        detector, path = self._create_detector()
        try:
            detector.set_baseline({"metric": 0.0})

            current = {"metric": 0.5}
            reports = detector.check_drift(current)

            assert len(reports) == 1
            assert reports[0].drop_percentage == 0.0
            assert reports[0].is_drifted is False
        finally:
            if path.exists():
                path.unlink()

    def test_custom_threshold(self):
        """Test with custom threshold."""
        detector, path = self._create_detector(threshold=0.05)
        try:
            detector.set_baseline({"precision": 1.0})

            # 6% drop - should trigger with 5% threshold
            current = {"precision": 0.94}
            reports = detector.check_drift(current)

            assert reports[0].is_drifted is True
        finally:
            if path.exists():
                path.unlink()

    def test_drift_report_fields(self):
        """Test DriftReport dataclass fields."""
        report = DriftReport(
            metric_name="precision",
            baseline_value=0.8,
            current_value=0.6,
            drop_percentage=0.25,
            is_drifted=True,
        )
        assert report.metric_name == "precision"
        assert report.baseline_value == 0.8
        assert report.current_value == 0.6
        assert report.drop_percentage == 0.25
        assert report.is_drifted is True
