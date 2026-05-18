"""Tests for access control module."""

import json
import tempfile
from pathlib import Path

import pytest

from ragforge.core.models import QueryResult
from ragforge.deployment.access_control import AccessController


class TestAccessController:
    """Tests for AccessController class."""

    def _create_controller(self) -> tuple:
        """Create a controller with a temp file."""
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        tmp.close()
        path = Path(tmp.name)
        path.unlink()
        controller = AccessController(policy_path=path)
        return controller, path

    def _make_result(self, source: str, score: float = 0.9) -> QueryResult:
        """Create a test QueryResult."""
        return QueryResult(
            content="test content",
            score=score,
            source=source,
            chunk_id=f"chunk-{source}",
            metadata={"source": source},
        )

    def test_add_and_check_policy(self):
        """Test adding a policy and checking access."""
        controller, path = self._create_controller()
        try:
            controller.add_policy("tenant-1", ["source-a", "source-b"])

            assert controller.check_access("tenant-1", "source-a") is True
            assert controller.check_access("tenant-1", "source-b") is True
            assert controller.check_access("tenant-1", "source-c") is False
        finally:
            if path.exists():
                path.unlink()

    def test_no_policy_means_open_access(self):
        """Test that no policy means open access."""
        controller, path = self._create_controller()
        try:
            # No policy for tenant-2
            assert controller.check_access("tenant-2", "any-source") is True
        finally:
            if path.exists():
                path.unlink()

    def test_wildcard_access(self):
        """Test wildcard access policy."""
        controller, path = self._create_controller()
        try:
            controller.add_policy("admin", ["*"])

            assert controller.check_access("admin", "source-a") is True
            assert controller.check_access("admin", "source-b") is True
            assert controller.check_access("admin", "anything") is True
        finally:
            if path.exists():
                path.unlink()

    def test_filter_results(self):
        """Test filtering query results by access policy."""
        controller, path = self._create_controller()
        try:
            controller.add_policy("tenant-1", ["source-a", "source-b"])

            results = [
                self._make_result("source-a"),
                self._make_result("source-b"),
                self._make_result("source-c"),
                self._make_result("source-d"),
            ]

            filtered = controller.filter_results(results, "tenant-1")

            assert len(filtered) == 2
            assert all(r.source in ("source-a", "source-b") for r in filtered)
        finally:
            if path.exists():
                path.unlink()

    def test_filter_results_no_policy(self):
        """Test that filtering with no policy returns all results."""
        controller, path = self._create_controller()
        try:
            results = [
                self._make_result("source-a"),
                self._make_result("source-b"),
            ]

            filtered = controller.filter_results(results, "unknown-tenant")

            assert len(filtered) == 2
        finally:
            if path.exists():
                path.unlink()

    def test_remove_policy(self):
        """Test removing a policy."""
        controller, path = self._create_controller()
        try:
            controller.add_policy("tenant-1", ["source-a"])
            assert controller.check_access("tenant-1", "source-b") is False

            controller.remove_policy("tenant-1")
            # After removal, open access
            assert controller.check_access("tenant-1", "source-b") is True
        finally:
            if path.exists():
                path.unlink()

    def test_persistence(self):
        """Test that policies persist to file."""
        controller, path = self._create_controller()
        try:
            controller.add_policy("tenant-1", ["source-a", "source-b"])

            # Create new controller from same file
            controller2 = AccessController(policy_path=path)
            assert controller2.check_access("tenant-1", "source-a") is True
            assert controller2.check_access("tenant-1", "source-c") is False
        finally:
            if path.exists():
                path.unlink()

    def test_update_policy(self):
        """Test updating an existing policy."""
        controller, path = self._create_controller()
        try:
            controller.add_policy("tenant-1", ["source-a"])
            assert controller.check_access("tenant-1", "source-b") is False

            controller.add_policy("tenant-1", ["source-a", "source-b"])
            assert controller.check_access("tenant-1", "source-b") is True
        finally:
            if path.exists():
                path.unlink()

    def test_get_policy(self):
        """Test getting a policy."""
        controller, path = self._create_controller()
        try:
            controller.add_policy("tenant-1", ["source-a", "source-b"])

            policy = controller.get_policy("tenant-1")
            assert policy == ["source-a", "source-b"]

            assert controller.get_policy("nonexistent") is None
        finally:
            if path.exists():
                path.unlink()

    def test_list_tenants(self):
        """Test listing all tenants."""
        controller, path = self._create_controller()
        try:
            controller.add_policy("tenant-1", ["source-a"])
            controller.add_policy("tenant-2", ["source-b"])

            tenants = controller.list_tenants()
            assert "tenant-1" in tenants
            assert "tenant-2" in tenants
            assert len(tenants) == 2
        finally:
            if path.exists():
                path.unlink()
