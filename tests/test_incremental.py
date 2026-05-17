"""Tests for IngestionTracker - incremental ingestion."""

import json
import os
import tempfile

import pytest

from ragforge.core.models import Document
from ragforge.ingestion.tracker import IngestionTracker


@pytest.fixture
def temp_state_file():
    """Create a temporary state file for testing."""
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def tracker(temp_state_file):
    """Create an IngestionTracker with a temporary state file."""
    return IngestionTracker(state_path=temp_state_file)


def make_doc(source: str, content: str, doc_type: str = "text") -> Document:
    """Helper to create a Document."""
    return Document(content=content, source=source, doc_type=doc_type)


class TestIngestionTracker:
    """Test incremental ingestion tracking."""

    def test_new_document_is_changed(self, tracker):
        doc = make_doc("file.txt", "Hello world")
        assert tracker.has_changed(doc) is True

    def test_ingested_document_not_changed(self, tracker):
        doc = make_doc("file.txt", "Hello world")
        tracker.mark_ingested(doc)
        assert tracker.has_changed(doc) is False

    def test_modified_document_is_changed(self, tracker):
        doc = make_doc("file.txt", "Hello world")
        tracker.mark_ingested(doc)

        # Modify content
        modified_doc = make_doc("file.txt", "Hello world updated")
        assert tracker.has_changed(modified_doc) is True

    def test_filter_changed_new_docs(self, tracker):
        docs = [
            make_doc("a.txt", "Content A"),
            make_doc("b.txt", "Content B"),
            make_doc("c.txt", "Content C"),
        ]

        changed = tracker.filter_changed(docs)
        assert len(changed) == 3  # All new

    def test_filter_changed_some_ingested(self, tracker):
        docs = [
            make_doc("a.txt", "Content A"),
            make_doc("b.txt", "Content B"),
            make_doc("c.txt", "Content C"),
        ]

        # Mark first two as ingested
        tracker.mark_ingested(docs[0])
        tracker.mark_ingested(docs[1])

        changed = tracker.filter_changed(docs)
        assert len(changed) == 1
        assert changed[0].source == "c.txt"

    def test_filter_changed_all_ingested(self, tracker):
        docs = [
            make_doc("a.txt", "Content A"),
            make_doc("b.txt", "Content B"),
        ]

        tracker.mark_batch_ingested(docs)

        changed = tracker.filter_changed(docs)
        assert len(changed) == 0

    def test_filter_changed_content_modified(self, tracker):
        original = make_doc("file.txt", "Original content")
        tracker.mark_ingested(original)

        modified = make_doc("file.txt", "Modified content")
        changed = tracker.filter_changed([modified])
        assert len(changed) == 1

    def test_compute_hash_deterministic(self, tracker):
        hash1 = tracker.compute_hash("test content")
        hash2 = tracker.compute_hash("test content")
        assert hash1 == hash2

    def test_compute_hash_different_content(self, tracker):
        hash1 = tracker.compute_hash("content A")
        hash2 = tracker.compute_hash("content B")
        assert hash1 != hash2

    def test_state_persists_to_disk(self, temp_state_file):
        tracker1 = IngestionTracker(state_path=temp_state_file)
        doc = make_doc("persist.txt", "Persistent content")
        tracker1.mark_ingested(doc)

        # Create new tracker instance reading same file
        tracker2 = IngestionTracker(state_path=temp_state_file)
        assert tracker2.has_changed(doc) is False

    def test_remove_tracked_document(self, tracker):
        doc = make_doc("file.txt", "Content")
        tracker.mark_ingested(doc)
        assert tracker.has_changed(doc) is False

        tracker.remove("file.txt")
        assert tracker.has_changed(doc) is True

    def test_get_tracked_sources(self, tracker):
        docs = [
            make_doc("a.txt", "A"),
            make_doc("b.txt", "B"),
            make_doc("c.txt", "C"),
        ]
        tracker.mark_batch_ingested(docs)

        sources = tracker.get_tracked_sources()
        assert set(sources) == {"a.txt", "b.txt", "c.txt"}

    def test_tracked_count(self, tracker):
        assert tracker.tracked_count == 0

        tracker.mark_ingested(make_doc("a.txt", "A"))
        assert tracker.tracked_count == 1

        tracker.mark_ingested(make_doc("b.txt", "B"))
        assert tracker.tracked_count == 2

    def test_clear_state(self, tracker):
        tracker.mark_ingested(make_doc("a.txt", "A"))
        tracker.mark_ingested(make_doc("b.txt", "B"))
        assert tracker.tracked_count == 2

        tracker.clear()
        assert tracker.tracked_count == 0
        assert tracker.has_changed(make_doc("a.txt", "A")) is True

    def test_mark_batch_ingested(self, tracker):
        docs = [
            make_doc("x.txt", "X"),
            make_doc("y.txt", "Y"),
            make_doc("z.txt", "Z"),
        ]
        tracker.mark_batch_ingested(docs)

        for doc in docs:
            assert tracker.has_changed(doc) is False

    def test_state_file_created_if_not_exists(self):
        path = tempfile.mktemp(suffix=".json")
        try:
            tracker = IngestionTracker(state_path=path)
            tracker.mark_ingested(make_doc("test.txt", "test"))
            assert os.path.exists(path)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_handles_corrupted_state_file(self, temp_state_file):
        # Write invalid JSON
        with open(temp_state_file, "w") as f:
            f.write("not valid json{{{")

        # Should handle gracefully
        tracker = IngestionTracker(state_path=temp_state_file)
        assert tracker.tracked_count == 0

    def test_hash_is_sha256(self, tracker):
        hash_val = tracker.compute_hash("hello")
        # SHA-256 produces 64 hex characters
        assert len(hash_val) == 64
        assert all(c in "0123456789abcdef" for c in hash_val)
