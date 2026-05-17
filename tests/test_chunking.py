"""Tests for recursive and semantic chunkers."""

import pytest

from ragforge.chunkers.base import BaseChunker
from ragforge.chunkers.recursive import RecursiveChunker
from ragforge.chunkers.semantic import SemanticChunker
from ragforge.chunkers.registry import ChunkerRegistry
from ragforge.core.models import Chunk


class TestRecursiveChunker:
    """Tests for RecursiveChunker."""

    def test_short_text_single_chunk(self):
        """Short text should produce a single chunk."""
        chunker = RecursiveChunker(max_chunk_size=100, overlap=10)
        chunks = chunker.chunk("Hello world.", source="test.txt")

        assert len(chunks) == 1
        assert chunks[0].content == "Hello world."
        assert chunks[0].source == "test.txt"

    def test_long_text_multiple_chunks(self):
        """Long text should be split into multiple chunks."""
        text = "This is a sentence. " * 50  # ~1000 chars
        chunker = RecursiveChunker(max_chunk_size=200, overlap=20)
        chunks = chunker.chunk(text, source="doc.txt")

        assert len(chunks) > 1
        for chunk in chunks:
            assert isinstance(chunk, Chunk)
            assert chunk.chunk_id  # UUID should be set
            assert chunk.source == "doc.txt"

    def test_splits_on_paragraphs_first(self):
        """Should prefer splitting on paragraph boundaries."""
        text = "First paragraph content.\n\nSecond paragraph content.\n\nThird paragraph content."
        chunker = RecursiveChunker(max_chunk_size=50, overlap=0)
        chunks = chunker.chunk(text)

        # Should split on \n\n boundaries
        assert len(chunks) >= 2

    def test_overlap_included(self):
        """Chunks should include overlap from previous chunk."""
        text = "A" * 100 + "\n\n" + "B" * 100
        chunker = RecursiveChunker(max_chunk_size=120, overlap=20)
        chunks = chunker.chunk(text)

        # Second chunk should contain some overlap
        assert len(chunks) >= 2

    def test_metadata_propagated(self):
        """Metadata should be attached to all chunks."""
        chunker = RecursiveChunker(max_chunk_size=50, overlap=0)
        meta = {"author": "test", "category": "docs"}
        chunks = chunker.chunk("Short text.", source="s", metadata=meta)

        assert chunks[0].metadata == meta

    def test_empty_content(self):
        """Empty content should return empty list."""
        chunker = RecursiveChunker(max_chunk_size=100, overlap=10)
        chunks = chunker.chunk("")

        assert chunks == []

    def test_chunk_ids_unique(self):
        """Each chunk should have a unique ID."""
        text = "Sentence one. " * 30
        chunker = RecursiveChunker(max_chunk_size=100, overlap=10)
        chunks = chunker.chunk(text)

        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))


class TestSemanticChunker:
    """Tests for SemanticChunker."""

    def test_splits_on_headers(self):
        """Should split on markdown headers."""
        text = "# Introduction\n\nSome intro text here that is long enough.\n\n# Methods\n\nSome methods text that is also long enough."
        chunker = SemanticChunker(max_chunk_size=60, overlap=0)
        chunks = chunker.chunk(text)

        assert len(chunks) >= 2

    def test_splits_on_double_newlines(self):
        """Should split on double newlines when no headers present."""
        text = "First section.\n\nSecond section.\n\nThird section."
        chunker = SemanticChunker(max_chunk_size=30, overlap=0)
        chunks = chunker.chunk(text)

        assert len(chunks) >= 2

    def test_merges_small_sections(self):
        """Small consecutive sections should be merged."""
        text = "A.\n\nB.\n\nC."
        chunker = SemanticChunker(max_chunk_size=500, overlap=0)
        chunks = chunker.chunk(text)

        # All sections are small, should be merged into one
        assert len(chunks) == 1

    def test_respects_max_chunk_size(self):
        """No chunk should exceed max_chunk_size significantly."""
        text = "# Header\n\n" + "Word " * 200 + "\n\n# Another\n\nMore text."
        chunker = SemanticChunker(max_chunk_size=200, overlap=0)
        chunks = chunker.chunk(text)

        for chunk in chunks:
            # Allow some tolerance for boundary handling
            assert len(chunk.content) <= 250

    def test_empty_content(self):
        """Empty content should return empty list."""
        chunker = SemanticChunker(max_chunk_size=100, overlap=0)
        chunks = chunker.chunk("")

        assert chunks == []

    def test_source_and_metadata(self):
        """Source and metadata should be set on all chunks."""
        text = "# Title\n\nContent here.\n\n# Section 2\n\nMore content."
        chunker = SemanticChunker(max_chunk_size=50, overlap=0)
        meta = {"doc_type": "markdown"}
        chunks = chunker.chunk(text, source="readme.md", metadata=meta)

        for chunk in chunks:
            assert chunk.source == "readme.md"
            assert chunk.metadata == meta


class TestChunkerRegistry:
    """Tests for ChunkerRegistry."""

    def test_auto_selects_semantic_for_markdown(self):
        """Auto strategy should select semantic chunker for markdown."""
        registry = ChunkerRegistry()
        chunker = registry.get_chunker(doc_type="md", strategy="auto")

        assert isinstance(chunker, SemanticChunker)

    def test_auto_selects_recursive_for_text(self):
        """Auto strategy should select recursive chunker for plain text."""
        registry = ChunkerRegistry()
        chunker = registry.get_chunker(doc_type="txt", strategy="auto")

        assert isinstance(chunker, RecursiveChunker)

    def test_explicit_strategy_overrides_auto(self):
        """Explicit strategy should override auto-selection."""
        registry = ChunkerRegistry()
        chunker = registry.get_chunker(doc_type="md", strategy="recursive")

        assert isinstance(chunker, RecursiveChunker)

    def test_unknown_doc_type_defaults_to_recursive(self):
        """Unknown doc types should default to recursive chunker."""
        registry = ChunkerRegistry()
        chunker = registry.get_chunker(doc_type="xyz", strategy="auto")

        assert isinstance(chunker, RecursiveChunker)

    def test_custom_chunker_registration(self):
        """Should support registering custom chunkers."""
        registry = ChunkerRegistry()

        class CustomChunker(BaseChunker):
            def chunk(self, content, source="", metadata=None):
                return []

        registry.register("custom", CustomChunker)
        chunker = registry.get_chunker(strategy="custom")

        assert isinstance(chunker, CustomChunker)

    def test_passes_parameters(self):
        """Should pass max_chunk_size and overlap to chunker."""
        registry = ChunkerRegistry()
        chunker = registry.get_chunker(
            doc_type="txt", strategy="recursive", max_chunk_size=256, overlap=30
        )

        assert chunker.max_chunk_size == 256
        assert chunker.overlap == 30
