"""Audit trail for RAGForge retrieval operations."""

from __future__ import annotations

import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ragforge.core.models import QueryResult


class AuditLogger:
    """Logs every retrieval operation for compliance and debugging.

    Records are stored in JSON Lines format (one JSON object per line)
    for efficient append-only writes.

    Each record includes: tenant_id, query, results returned,
    relevance scores, and timestamp.
    """

    def __init__(self, audit_path: str | Path = "ragforge_audit.jsonl"):
        """Initialize the audit logger.

        Args:
            audit_path: Path to the JSON Lines audit file.
        """
        self._audit_path = Path(audit_path)
        self._audit_path.parent.mkdir(parents=True, exist_ok=True)

    def log_retrieval(
        self,
        tenant_id: str,
        query: str,
        results: List[QueryResult],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Log a retrieval operation.

        Args:
            tenant_id: Identifier of the user/tenant making the query.
            query: The query text.
            results: List of QueryResult objects returned.
            metadata: Optional additional metadata to include.

        Returns:
            The audit record that was logged.
        """
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "epoch": time.time(),
            "tenant_id": tenant_id,
            "query": query,
            "num_results": len(results),
            "results": [
                {
                    "chunk_id": r.chunk_id,
                    "source": r.source,
                    "score": r.score,
                    "content_preview": r.content[:100] if r.content else "",
                }
                for r in results
            ],
            "metadata": metadata or {},
        }

        # Append to file (JSON Lines format)
        with open(self._audit_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        return record

    def get_audit_trail(
        self,
        tenant_id: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve audit records with optional filtering.

        Args:
            tenant_id: Filter by tenant ID (None for all tenants).
            start_time: Filter records after this epoch timestamp.
            end_time: Filter records before this epoch timestamp.

        Returns:
            List of audit records matching the filters.
        """
        if not self._audit_path.exists():
            return []

        records = []
        with open(self._audit_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Apply filters
                if tenant_id and record.get("tenant_id") != tenant_id:
                    continue
                if start_time and record.get("epoch", 0) < start_time:
                    continue
                if end_time and record.get("epoch", 0) > end_time:
                    continue

                records.append(record)

        return records

    def export_csv(self, path: str | Path) -> int:
        """Export audit trail to CSV format.

        Args:
            path: Output CSV file path.

        Returns:
            Number of records exported.
        """
        records = self.get_audit_trail()
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "timestamp",
            "tenant_id",
            "query",
            "num_results",
            "top_score",
            "sources",
        ]

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for record in records:
                results = record.get("results", [])
                top_score = max((r["score"] for r in results), default=0.0)
                sources = ", ".join(r["source"] for r in results)

                writer.writerow({
                    "timestamp": record.get("timestamp", ""),
                    "tenant_id": record.get("tenant_id", ""),
                    "query": record.get("query", ""),
                    "num_results": record.get("num_results", 0),
                    "top_score": f"{top_score:.4f}",
                    "sources": sources,
                })

        return len(records)
