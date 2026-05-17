"""Metadata enrichment for chunks as a post-processing step."""

from __future__ import annotations

import re
from typing import Any, Dict, List

from ragforge.core.models import Chunk


# Date patterns to detect in content
DATE_PATTERNS = [
    re.compile(r"\b(\d{4}-\d{2}-\d{2})\b"),  # YYYY-MM-DD
    re.compile(r"\b(\d{4}/\d{2}/\d{2})\b"),  # YYYY/MM/DD
    re.compile(r"\b(\d{2}-\d{2}-\d{4})\b"),  # DD-MM-YYYY or MM-DD-YYYY
]

# Heading patterns for title extraction
TITLE_PATTERNS = [
    re.compile(r"^#{1,6}\s+(.+)$", re.MULTILINE),  # Markdown headings
    re.compile(r"^(.+)\n[=]{3,}$", re.MULTILINE),  # Setext h1
]


class MetadataEnricher:
    """Extracts and enriches chunk metadata as a post-processing step.

    Extracts:
    - title: First heading or first line of content
    - dates: Any YYYY-MM-DD patterns found in content
    - word_count: Number of words in the chunk
    """

    def __init__(
        self,
        extract_title: bool = True,
        extract_dates: bool = True,
        extract_word_count: bool = True,
    ):
        self.extract_title = extract_title
        self.extract_dates = extract_dates
        self.extract_word_count = extract_word_count

    def enrich(self, chunks: List[Chunk]) -> List[Chunk]:
        """Enrich a list of chunks with extracted metadata.

        Args:
            chunks: List of Chunk objects to enrich.

        Returns:
            The same list of chunks with enriched metadata.
        """
        for chunk in chunks:
            self._enrich_chunk(chunk)
        return chunks

    def enrich_single(self, chunk: Chunk) -> Chunk:
        """Enrich a single chunk with extracted metadata.

        Args:
            chunk: A Chunk object to enrich.

        Returns:
            The chunk with enriched metadata.
        """
        self._enrich_chunk(chunk)
        return chunk

    def _enrich_chunk(self, chunk: Chunk) -> None:
        """Apply all metadata extraction to a single chunk."""
        content = chunk.content

        if self.extract_title:
            title = self._extract_title(content)
            if title:
                chunk.metadata["title"] = title

        if self.extract_dates:
            dates = self._extract_dates(content)
            if dates:
                chunk.metadata["dates"] = dates

        if self.extract_word_count:
            chunk.metadata["word_count"] = self._count_words(content)

    def _extract_title(self, content: str) -> str:
        """Extract title from content (first heading or first line).

        Args:
            content: The text content to extract title from.

        Returns:
            Extracted title string, or empty string if none found.
        """
        # Try markdown headings first
        for pattern in TITLE_PATTERNS:
            match = pattern.search(content)
            if match:
                return match.group(1).strip()

        # Fall back to first non-empty line
        lines = content.strip().split("\n")
        for line in lines:
            stripped = line.strip()
            if stripped:
                # Truncate long first lines
                if len(stripped) > 100:
                    return stripped[:100] + "..."
                return stripped

        return ""

    def _extract_dates(self, content: str) -> List[str]:
        """Extract date patterns from content.

        Args:
            content: The text content to search for dates.

        Returns:
            List of date strings found in the content.
        """
        dates: List[str] = []
        seen: set = set()

        for pattern in DATE_PATTERNS:
            for match in pattern.finditer(content):
                date_str = match.group(1)
                if date_str not in seen:
                    dates.append(date_str)
                    seen.add(date_str)

        return dates

    def _count_words(self, content: str) -> int:
        """Count words in content.

        Args:
            content: The text content to count words in.

        Returns:
            Number of words.
        """
        words = re.findall(r"\b\w+\b", content)
        return len(words)
