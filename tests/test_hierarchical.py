"""Tests for HierarchicalChunker."""

import pytest

from ragforge.chunkers.hierarchical import HierarchicalChunker
from ragforge.core.models import Chunk


MARKDOWN_SAMPLE = """# Introduction

This is the introduction paragraph. It provides an overview of the document
and sets the context for what follows.

## Background

The background section explains the history and context.
There are multiple paragraphs here.

This is the second paragraph of the background section.
It contains additional details.

## Methods

We used the following methods in our research.

### Data Collection

Data was collected from multiple sources over a period of six months.

### Analysis

Statistical analysis was performed using standard techniques.

## Results

The results show significant improvements across all metrics.

## Conclusion

In conclusion, our approach demonstrates clear benefits.
"""

PLAIN_TEXT_SAMPLE = """This is a plain text document without any headings.

It has multiple paragraphs separated by blank lines.

Each paragraph should become a child chunk.

The entire document is one section since there are no headings.
"""

SINGLE_SECTION = """# Only One Section

This document has only one section with a single paragraph.
"""


class TestHierarchicalChunker:
    """Test hierarchical parent-child chunking."""

    def test_creates_parent_and_child_chunks(self):
        chunker = HierarchicalChunker(max_chunk_size=2000)
        chunks = chunker.chunk(MARKDOWN_SAMPLE, source="doc.md")

        parent_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "parent"]
        child_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "child"]

        assert len(parent_chunks) > 0
        assert len(child_chunks) > 0

    def test_child_has_parent_chunk_id(self):
        chunker = HierarchicalChunker(max_chunk_size=2000)
        chunks = chunker.chunk(MARKDOWN_SAMPLE, source="doc.md")

        child_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "child"]
        parent_ids = {c.chunk_id for c in chunks if c.metadata.get("chunk_type") == "parent"}

        for child in child_chunks:
            assert "parent_chunk_id" in child.metadata
            assert child.metadata["parent_chunk_id"] in parent_ids

    def test_section_titles_in_metadata(self):
        chunker = HierarchicalChunker(max_chunk_size=2000)
        chunks = chunker.chunk(MARKDOWN_SAMPLE, source="doc.md")

        parent_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "parent"]
        titles = [c.metadata.get("section_title", "") for c in parent_chunks]

        assert "Introduction" in titles
        assert "Background" in titles
        assert "Methods" in titles
        assert "Results" in titles
        assert "Conclusion" in titles

    def test_parent_contains_section_content(self):
        chunker = HierarchicalChunker(max_chunk_size=2000)
        chunks = chunker.chunk(MARKDOWN_SAMPLE, source="doc.md")

        parent_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "parent"]
        intro_parent = next(
            (c for c in parent_chunks if c.metadata.get("section_title") == "Introduction"),
            None,
        )

        assert intro_parent is not None
        assert "introduction paragraph" in intro_parent.content.lower()

    def test_children_are_paragraphs(self):
        chunker = HierarchicalChunker(max_chunk_size=2000)
        chunks = chunker.chunk(MARKDOWN_SAMPLE, source="doc.md")

        # Find children of Background section
        bg_parent = next(
            (c for c in chunks if c.metadata.get("section_title") == "Background"
             and c.metadata.get("chunk_type") == "parent"),
            None,
        )
        assert bg_parent is not None

        bg_children = [
            c for c in chunks
            if c.metadata.get("parent_chunk_id") == bg_parent.chunk_id
        ]
        assert len(bg_children) >= 2  # Background has 2 paragraphs

    def test_plain_text_without_headings(self):
        chunker = HierarchicalChunker(max_chunk_size=2000)
        chunks = chunker.chunk(PLAIN_TEXT_SAMPLE, source="plain.txt")

        # Should still create chunks (one parent section, multiple children)
        assert len(chunks) > 0
        child_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "child"]
        assert len(child_chunks) >= 3  # 4 paragraphs

    def test_include_parents_false(self):
        chunker = HierarchicalChunker(max_chunk_size=2000, include_parents=False)
        chunks = chunker.chunk(MARKDOWN_SAMPLE, source="doc.md")

        parent_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "parent"]
        child_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "child"]

        assert len(parent_chunks) == 0
        assert len(child_chunks) > 0
        # Children should still have parent_chunk_id
        for child in child_chunks:
            assert "parent_chunk_id" in child.metadata

    def test_chunk_ids_unique(self):
        chunker = HierarchicalChunker(max_chunk_size=2000)
        chunks = chunker.chunk(MARKDOWN_SAMPLE, source="doc.md")

        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_metadata_chunker_field(self):
        chunker = HierarchicalChunker(max_chunk_size=2000)
        chunks = chunker.chunk(MARKDOWN_SAMPLE, source="doc.md")

        for chunk in chunks:
            assert chunk.metadata["chunker"] == "hierarchical"

    def test_source_preserved(self):
        chunker = HierarchicalChunker(max_chunk_size=2000)
        chunks = chunker.chunk(MARKDOWN_SAMPLE, source="docs/readme.md")

        for chunk in chunks:
            assert chunk.source == "docs/readme.md"

    def test_custom_metadata_preserved(self):
        chunker = HierarchicalChunker(max_chunk_size=2000)
        chunks = chunker.chunk(
            MARKDOWN_SAMPLE, source="doc.md", metadata={"version": "1.0"}
        )

        for chunk in chunks:
            assert chunk.metadata["version"] == "1.0"

    def test_single_section_document(self):
        chunker = HierarchicalChunker(max_chunk_size=2000)
        chunks = chunker.chunk(SINGLE_SECTION, source="single.md")

        parent_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "parent"]
        assert len(parent_chunks) >= 1

    def test_empty_content(self):
        chunker = HierarchicalChunker(max_chunk_size=2000)
        chunks = chunker.chunk("", source="empty.md")

        # Empty content should produce no meaningful chunks
        # (may produce empty section)
        for chunk in chunks:
            # If any chunks are produced, they should have proper metadata
            assert "chunker" in chunk.metadata

    def test_nested_headings(self):
        chunker = HierarchicalChunker(max_chunk_size=2000)
        chunks = chunker.chunk(MARKDOWN_SAMPLE, source="doc.md")

        # Should detect sub-headings like "Data Collection" and "Analysis"
        titles = [c.metadata.get("section_title", "") for c in chunks]
        assert "Data Collection" in titles or "Analysis" in titles
