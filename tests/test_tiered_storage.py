"""Tests for tiered storage management."""

import tempfile
import time
from pathlib import Path

import pytest

from ragforge.cost.tiered_storage import TieredStorageManager, AccessRecord


class TestTieredStorageManager:
    """Tests for TieredStorageManager class."""

    def _create_manager(self, hot_threshold=5, warm_threshold=1, window_days=7) -> tuple:
        """Create a manager with a temp file."""
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        tmp.close()
        path = Path(tmp.name)
        path.unlink()
        manager = TieredStorageManager(
            storage_path=path,
            hot_threshold=hot_threshold,
            warm_threshold=warm_threshold,
            window_days=window_days,
        )
        return manager, path

    def test_new_chunk_defaults_to_hot(self):
        """Test that new chunks default to hot tier."""
        manager, path = self._create_manager()
        try:
            assert manager.get_tier("chunk-1") == "hot"
        finally:
            if path.exists():
                path.unlink()

    def test_record_access(self):
        """Test recording chunk access."""
        manager, path = self._create_manager()
        try:
            manager.record_access("chunk-1")
            assert manager.get_tier("chunk-1") == "hot"
            assert manager._records["chunk-1"].access_count == 1
        finally:
            if path.exists():
                path.unlink()

    def test_multiple_accesses(self):
        """Test multiple accesses to same chunk."""
        manager, path = self._create_manager()
        try:
            for _ in range(5):
                manager.record_access("chunk-1")
            assert manager._records["chunk-1"].access_count == 5
        finally:
            if path.exists():
                path.unlink()

    def test_demote_hot_to_warm(self):
        """Test demoting from hot to warm."""
        manager, path = self._create_manager()
        try:
            manager.record_access("chunk-1")
            new_tier = manager.demote("chunk-1")
            assert new_tier == "warm"
            assert manager.get_tier("chunk-1") == "warm"
        finally:
            if path.exists():
                path.unlink()

    def test_demote_warm_to_cold(self):
        """Test demoting from warm to cold."""
        manager, path = self._create_manager()
        try:
            manager.record_access("chunk-1")
            manager.demote("chunk-1")  # hot -> warm
            new_tier = manager.demote("chunk-1")  # warm -> cold
            assert new_tier == "cold"
            assert manager.get_tier("chunk-1") == "cold"
        finally:
            if path.exists():
                path.unlink()

    def test_demote_cold_stays_cold(self):
        """Test that cold chunks stay cold on demote."""
        manager, path = self._create_manager()
        try:
            manager.record_access("chunk-1")
            manager.demote("chunk-1")  # hot -> warm
            manager.demote("chunk-1")  # warm -> cold
            new_tier = manager.demote("chunk-1")  # cold -> cold
            assert new_tier == "cold"
        finally:
            if path.exists():
                path.unlink()

    def test_promote_cold_to_warm(self):
        """Test promoting from cold to warm."""
        manager, path = self._create_manager()
        try:
            manager.record_access("chunk-1")
            manager.demote("chunk-1")  # hot -> warm
            manager.demote("chunk-1")  # warm -> cold
            new_tier = manager.promote("chunk-1")  # cold -> warm
            assert new_tier == "warm"
        finally:
            if path.exists():
                path.unlink()

    def test_promote_warm_to_hot(self):
        """Test promoting from warm to hot."""
        manager, path = self._create_manager()
        try:
            manager.record_access("chunk-1")
            manager.demote("chunk-1")  # hot -> warm
            new_tier = manager.promote("chunk-1")  # warm -> hot
            assert new_tier == "hot"
        finally:
            if path.exists():
                path.unlink()

    def test_promote_hot_stays_hot(self):
        """Test that hot chunks stay hot on promote."""
        manager, path = self._create_manager()
        try:
            manager.record_access("chunk-1")
            new_tier = manager.promote("chunk-1")
            assert new_tier == "hot"
        finally:
            if path.exists():
                path.unlink()

    def test_promote_unknown_chunk(self):
        """Test promoting an unknown chunk defaults to hot."""
        manager, path = self._create_manager()
        try:
            new_tier = manager.promote("unknown-chunk")
            assert new_tier == "hot"
        finally:
            if path.exists():
                path.unlink()

    def test_demote_unknown_chunk(self):
        """Test demoting an unknown chunk defaults to warm."""
        manager, path = self._create_manager()
        try:
            new_tier = manager.demote("unknown-chunk")
            assert new_tier == "warm"
        finally:
            if path.exists():
                path.unlink()

    def test_get_tier_stats_empty(self):
        """Test tier stats with no chunks."""
        manager, path = self._create_manager()
        try:
            stats = manager.get_tier_stats()
            assert stats["total"] == 0
            assert stats["hot"]["count"] == 0
            assert stats["warm"]["count"] == 0
            assert stats["cold"]["count"] == 0
        finally:
            if path.exists():
                path.unlink()

    def test_get_tier_stats(self):
        """Test tier stats with mixed tiers."""
        manager, path = self._create_manager()
        try:
            manager.record_access("chunk-1")  # hot
            manager.record_access("chunk-2")  # hot
            manager.record_access("chunk-3")  # hot
            manager.demote("chunk-2")  # warm
            manager.demote("chunk-3")  # warm
            manager.demote("chunk-3")  # cold

            stats = manager.get_tier_stats()
            assert stats["total"] == 3
            assert stats["hot"]["count"] == 1
            assert stats["warm"]["count"] == 1
            assert stats["cold"]["count"] == 1
        finally:
            if path.exists():
                path.unlink()

    def test_persistence(self):
        """Test that tier data persists to file."""
        manager, path = self._create_manager()
        try:
            manager.record_access("chunk-1")
            manager.demote("chunk-1")

            # Create new manager from same file
            manager2 = TieredStorageManager(storage_path=path)
            assert manager2.get_tier("chunk-1") == "warm"
        finally:
            if path.exists():
                path.unlink()

    def test_auto_promote_on_access(self):
        """Test that accessing a cold chunk promotes it."""
        manager, path = self._create_manager(hot_threshold=5, warm_threshold=1)
        try:
            manager.record_access("chunk-1")
            manager.demote("chunk-1")  # warm
            manager.demote("chunk-1")  # cold

            # Access should promote from cold to warm (meets warm_threshold=1)
            manager.record_access("chunk-1")
            assert manager.get_tier("chunk-1") == "warm"
        finally:
            if path.exists():
                path.unlink()

    def test_evaluate_tiers(self):
        """Test tier evaluation suggestions."""
        manager, path = self._create_manager(hot_threshold=5, warm_threshold=1)
        try:
            # Create a chunk in hot tier with no recent accesses
            manager.record_access("chunk-1")
            # Manually set access history to empty (simulating old accesses)
            manager._access_history["chunk-1"] = []
            manager._save()

            suggestions = manager.evaluate_tiers()
            # chunk-1 is hot but has 0 recent accesses (< warm_threshold)
            assert "chunk-1" in suggestions["demote"]
        finally:
            if path.exists():
                path.unlink()
