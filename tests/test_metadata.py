"""Tests for MetadataEnricher."""

import pytest

from ragforge.chunkers.metadata import MetadataEnricher
from ragforge.core.models import Chunk


def make_chunk(content: str, chunk_id: str = "test-id") -> Chunk:
    """Helper to create a Chunk for testing."""
    return Chunk(
        content=content,
        chunk_id=chunk_id,
        source="test.txt",
        metadata={},
    )


class TestMetadataEnricher:
    """Test metadata extraction from chunks."""

    def test_extracts_markdown_title(self):
        enricher = MetadataEnricher()
        chunk = make_chunk("# My Document Title\n\nSome content here.")
        enricher.enrich_single(chunk)

        assert chunk.metadata["title"] == "My Document Title"

    def test_extracts_h2_title(self):
        enricher = MetadataEnricher()
        chunk = make_chunk("## Section Title\n\nParagraph text.")
        enricher.enrich_single(chunk)

        assert chunk.metadata["title"] == "Section Title"

    def test_falls_back_to_first_line(self):
        enricher = MetadataEnricher()
        chunk = make_chunk("This is the first line\n\nMore content follows.")
        enricher.enrich_single(chunk)

        assert chunk.metadata["title"] == "This is the first line"

    def test_truncates_long_first_line(self):
        enricher = MetadataEnricher()
        long_line = "A" * 150
        chunk = make_chunk(long_line + "\n\nMore content.")
        enricher.enrich_single(chunk)

        assert len(chunk.metadata["title"]) <= 104  # 100 + "..."

    def test_extracts_yyyy_mm_dd_date(self):
        enricher = MetadataEnricher()
        chunk = make_chunk("Published on 2024-01-15 by the team.")
        enricher.enrich_single(chunk)

        assert "2024-01-15" in chunk.metadata["dates"]

    def test_extracts_multiple_dates(self):
        enricher = MetadataEnricher()
        chunk = make_chunk("From 2024-01-01 to 2024-12-31 the project ran.")
        enricher.enrich_single(chunk)

        assert "2024-01-01" in chunk.metadata["dates"]
        assert "2024-12-31" in chunk.metadata["dates"]

    def test_extracts_slash_date_format(self):
        enricher = MetadataEnricher()
        chunk = make_chunk("Date: 2024/03/20 was the deadline.")
        enricher.enrich_single(chunk)

        assert "2024/03/20" in chunk.metadata["dates"]

    def test_no_dates_found(self):
        enricher = MetadataEnricher()
        chunk = make_chunk("No dates in this content at all.")
        enricher.enrich_single(chunk)

        # dates key should not be set if no dates found
        assert "dates" not in chunk.metadata

    def test_word_count(self):
        enricher = MetadataEnricher()
        chunk = make_chunk("one two three four five")
        enricher.enrich_single(chunk)

        assert chunk.metadata["word_count"] == 5

    def test_word_count_with_punctuation(self):
        enricher = MetadataEnricher()
        chunk = make_chunk("Hello, world! This is a test.")
        enricher.enrich_single(chunk)

        assert chunk.metadata["word_count"] == 6

    def test_word_count_empty(self):
        enricher = MetadataEnricher()
        chunk = make_chunk("")
        enricher.enrich_single(chunk)

        assert chunk.metadata["word_count"] == 0

    def test_enrich_batch(self):
        enricher = MetadataEnricher()
        chunks = [
            make_chunk("# Title One\n\nContent 2024-01-01", "id1"),
            make_chunk("# Title Two\n\nMore content here", "id2"),
            make_chunk("Plain text without heading", "id3"),
        ]

        result = enricher.enrich(chunks)

        assert result is chunks  # Same list returned
        assert chunks[0].metadata["title"] == "Title One"
        assert chunks[1].metadata["title"] == "Title Two"
        assert chunks[2].metadata["title"] == "Plain text without heading"

    def test_disable_title_extraction(self):
        enricher = MetadataEnricher(extract_title=False)
        chunk = make_chunk("# Heading\n\nContent")
        enricher.enrich_single(chunk)

        assert "title" not in chunk.metadata

    def test_disable_date_extraction(self):
        enricher = MetadataEnricher(extract_dates=False)
        chunk = make_chunk("Date: 2024-01-15")
        enricher.enrich_single(chunk)

        assert "dates" not in chunk.metadata

    def test_disable_word_count(self):
        enricher = MetadataEnricher(extract_word_count=False)
        chunk = make_chunk("Some words here")
        enricher.enrich_single(chunk)

        assert "word_count" not in chunk.metadata

    def test_preserves_existing_metadata(self):
        enricher = MetadataEnricher()
        chunk = make_chunk("# Title\n\nContent 2024-05-01")
        chunk.metadata["existing_key"] = "existing_value"
        enricher.enrich_single(chunk)

        assert chunk.metadata["existing_key"] == "existing_value"
        assert chunk.metadata["title"] == "Title"

    def test_deduplicates_dates(self):
        enricher = MetadataEnricher()
        chunk = make_chunk("Date 2024-01-01 repeated 2024-01-01 again.")
        enricher.enrich_single(chunk)

        assert chunk.metadata["dates"].count("2024-01-01") == 1

    def test_empty_content_title(self):
        enricher = MetadataEnricher()
        chunk = make_chunk("   \n\n   ")
        enricher.enrich_single(chunk)

        assert chunk.metadata.get("title", "") == ""

    def test_setext_heading_extraction(self):
        enricher = MetadataEnricher()
        chunk = make_chunk("My Setext Title\n===\n\nContent below.")
        enricher.enrich_single(chunk)

        assert chunk.metadata["title"] == "My Setext Title"
