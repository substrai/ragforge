"""Incremental ingestion tracker using content hashing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ragforge.core.models import Document


class IngestionTracker:
    """Tracks document hashes to enable incremental ingestion.

    Stores SHA-256 hashes of document content in a JSON file. On re-ingestion,
    only documents whose content hash has changed are processed.
    """

    def __init__(self, state_path: str | Path = ".ragforge_ingestion_state.json"):
        """Initialize the ingestion tracker.

        Args:
            state_path: Path to the JSON file storing document hashes.
        """
        self.state_path = Path(state_path)
        self._state: Dict[str, Dict[str, Any]] = {}
        self._load_state()

    def _load_state(self) -> None:
        """Load existing state from disk."""
        if self.state_path.exists():
            try:
                with open(self.state_path, "r") as f:
                    self._state = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._state = {}

    def _save_state(self) -> None:
        """Persist state to disk."""
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_path, "w") as f:
            json.dump(self._state, f, indent=2)

    def compute_hash(self, content: str) -> str:
        """Compute SHA-256 hash of document content.

        Args:
            content: The document content to hash.

        Returns:
            Hex-encoded SHA-256 hash string.
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def has_changed(self, document: Document) -> bool:
        """Check if a document has changed since last ingestion.

        Args:
            document: The document to check.

        Returns:
            True if the document is new or has changed, False otherwise.
        """
        source = document.source
        current_hash = self.compute_hash(document.content)

        if source not in self._state:
            return True

        stored_hash = self._state[source].get("hash", "")
        return current_hash != stored_hash

    def filter_changed(self, documents: List[Document]) -> List[Document]:
        """Filter documents to only those that have changed.

        Args:
            documents: List of documents to filter.

        Returns:
            List of documents that are new or have changed content.
        """
        return [doc for doc in documents if self.has_changed(doc)]

    def mark_ingested(self, document: Document) -> None:
        """Mark a document as successfully ingested.

        Args:
            document: The document that was ingested.
        """
        self._state[document.source] = {
            "hash": self.compute_hash(document.content),
            "doc_type": document.doc_type,
            "metadata": document.metadata,
        }
        self._save_state()

    def mark_batch_ingested(self, documents: List[Document]) -> None:
        """Mark multiple documents as successfully ingested.

        Args:
            documents: List of documents that were ingested.
        """
        for doc in documents:
            self._state[doc.source] = {
                "hash": self.compute_hash(doc.content),
                "doc_type": doc.doc_type,
                "metadata": doc.metadata,
            }
        self._save_state()

    def remove(self, source: str) -> None:
        """Remove a document from tracking.

        Args:
            source: The source identifier of the document to remove.
        """
        if source in self._state:
            del self._state[source]
            self._save_state()

    def get_tracked_sources(self) -> List[str]:
        """Get all tracked document sources.

        Returns:
            List of source identifiers being tracked.
        """
        return list(self._state.keys())

    def clear(self) -> None:
        """Clear all tracking state."""
        self._state = {}
        self._save_state()

    @property
    def tracked_count(self) -> int:
        """Number of documents currently tracked."""
        return len(self._state)
