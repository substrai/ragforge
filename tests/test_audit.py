"""Tests for audit trail module."""

import json
import tempfile
import time
from pathlib import Path

import pytest

from ragforge.core.models import QueryResult
from ragforge.deployment.audit import AuditLogger


class TestAuditLogger:
    """Tests for AuditLogger class."""

    def _create_logger(self) -> tuple:
        """Create a logger with a temp file."""
        tmp = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
        tmp.close()
        path = Path(tmp.name)
        path.unlink()
        logger = AuditLogger(audit_path=path)
        return logger, path

    def _make_results(self, count: int = 3) -> list:
        """Create test QueryResult objects."""
        return [
            QueryResult(
                content=f"Content for result {i}",
                score=0.9 - (i * 0.1),
                source=f"source-{i}",
                chunk_id=f"chunk-{i}",
                metadata={"source": f"source-{i}"},
            )
            for i in range(count)
        ]

    def test_log_retrieval(self):
        """Test logging a retrieval operation."""
        logger, path = self._create_logger()
        try:
            results = self._make_results()
            record = logger.log_retrieval(
                tenant_id="tenant-1",
                query="what is RAG?",
                results=results,
            )

            assert record["tenant_id"] == "tenant-1"
            assert record["query"] == "what is RAG?"
            assert record["num_results"] == 3
            assert len(record["results"]) == 3
            assert "timestamp" in record
            assert "epoch" in record
        finally:
            if path.exists():
                path.unlink()

    def test_log_retrieval_persists(self):
        """Test that log records are persisted to file."""
        logger, path = self._create_logger()
        try:
            results = self._make_results(2)
            logger.log_retrieval("tenant-1", "query 1", results)
            logger.log_retrieval("tenant-2", "query 2", results)

            # Read file directly
            lines = path.read_text().strip().split("\n")
            assert len(lines) == 2

            record1 = json.loads(lines[0])
            assert record1["tenant_id"] == "tenant-1"

            record2 = json.loads(lines[1])
            assert record2["tenant_id"] == "tenant-2"
        finally:
            if path.exists():
                path.unlink()

    def test_get_audit_trail_all(self):
        """Test getting all audit records."""
        logger, path = self._create_logger()
        try:
            results = self._make_results()
            logger.log_retrieval("tenant-1", "query 1", results)
            logger.log_retrieval("tenant-2", "query 2", results)
            logger.log_retrieval("tenant-1", "query 3", results)

            trail = logger.get_audit_trail()
            assert len(trail) == 3
        finally:
            if path.exists():
                path.unlink()

    def test_get_audit_trail_by_tenant(self):
        """Test filtering audit trail by tenant."""
        logger, path = self._create_logger()
        try:
            results = self._make_results()
            logger.log_retrieval("tenant-1", "query 1", results)
            logger.log_retrieval("tenant-2", "query 2", results)
            logger.log_retrieval("tenant-1", "query 3", results)

            trail = logger.get_audit_trail(tenant_id="tenant-1")
            assert len(trail) == 2
            assert all(r["tenant_id"] == "tenant-1" for r in trail)
        finally:
            if path.exists():
                path.unlink()

    def test_get_audit_trail_by_time_range(self):
        """Test filtering audit trail by time range."""
        logger, path = self._create_logger()
        try:
            results = self._make_results()

            before = time.time()
            logger.log_retrieval("tenant-1", "query 1", results)
            time.sleep(0.05)
            mid = time.time()
            time.sleep(0.05)
            logger.log_retrieval("tenant-1", "query 2", results)
            after = time.time()

            # Get records after mid
            trail = logger.get_audit_trail(start_time=mid)
            assert len(trail) == 1
            assert trail[0]["query"] == "query 2"

            # Get records before mid
            trail = logger.get_audit_trail(end_time=mid)
            assert len(trail) == 1
            assert trail[0]["query"] == "query 1"
        finally:
            if path.exists():
                path.unlink()

    def test_get_audit_trail_empty(self):
        """Test getting audit trail when no records exist."""
        logger, path = self._create_logger()
        try:
            trail = logger.get_audit_trail()
            assert trail == []
        finally:
            if path.exists():
                path.unlink()

    def test_export_csv(self):
        """Test exporting audit trail to CSV."""
        logger, path = self._create_logger()
        try:
            results = self._make_results()
            logger.log_retrieval("tenant-1", "query 1", results)
            logger.log_retrieval("tenant-2", "query 2", results)

            csv_path = path.parent / "audit_export.csv"
            count = logger.export_csv(csv_path)

            assert count == 2
            assert csv_path.exists()

            content = csv_path.read_text()
            lines = content.strip().split("\n")
            assert len(lines) == 3  # header + 2 records
            assert "timestamp" in lines[0]
            assert "tenant_id" in lines[0]
            assert "query" in lines[0]

            csv_path.unlink()
        finally:
            if path.exists():
                path.unlink()

    def test_log_retrieval_with_metadata(self):
        """Test logging with additional metadata."""
        logger, path = self._create_logger()
        try:
            results = self._make_results()
            record = logger.log_retrieval(
                tenant_id="tenant-1",
                query="test query",
                results=results,
                metadata={"session_id": "abc123", "ip": "192.168.1.1"},
            )

            assert record["metadata"]["session_id"] == "abc123"
            assert record["metadata"]["ip"] == "192.168.1.1"
        finally:
            if path.exists():
                path.unlink()

    def test_result_scores_in_audit(self):
        """Test that result scores are captured in audit."""
        logger, path = self._create_logger()
        try:
            results = self._make_results()
            record = logger.log_retrieval("tenant-1", "test", results)

            assert record["results"][0]["score"] == pytest.approx(0.9)
            assert record["results"][1]["score"] == pytest.approx(0.8)
            assert record["results"][2]["score"] == pytest.approx(0.7)
        finally:
            if path.exists():
                path.unlink()

    def test_content_preview_truncated(self):
        """Test that content preview is truncated to 100 chars."""
        logger, path = self._create_logger()
        try:
            long_content = "x" * 200
            results = [
                QueryResult(
                    content=long_content,
                    score=0.9,
                    source="test",
                    chunk_id="chunk-1",
                )
            ]
            record = logger.log_retrieval("tenant-1", "test", results)

            assert len(record["results"][0]["content_preview"]) == 100
        finally:
            if path.exists():
                path.unlink()
