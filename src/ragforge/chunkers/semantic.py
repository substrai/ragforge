"""Semantic chunker for RAGForge."""

from __future__ import annotations

import re
import uuid
from typing import Any, Dict, List

from ragforge.chunkers.base import BaseChunker
from ragforge.core.models import Chunk


class SemanticChunker(BaseChunker):
    """Splits text on topic boundaries.

    Uses structural cues (double newlines, markdown headers) to identify
    natural topic boundaries. Merges small sections and splits large ones
    to stay within size constraints.
    """

    HEADER_PATTERN = re.compile(r"^#{1,6}\s+.+$", re.MULTILINE)

    def __init__(self, max_chunk_size: int = 512, overlap: int = 50):
        super().__init__(max_chunk_size=max_chunk_size, overlap=overlap)

    def chunk(
        self,
        content: str,
        source: str = "",
        metadata: Dict[str, Any] | None = None,
    ) -> List[Chunk]:
        """Split content on semantic boundaries (headers, double newlines)."""
        metadata = metadata or {}
        sections = self._split_on_boundaries(content)
        merged = self._merge_small_sections(sections)

        chunks: List[Chunk] = []
        current_index = 0

        for text in merged:
            start = content.find(text.strip(), max(0, current_index - self.overlap))
            if start == -1:
                start = current_index
            end = start + len(text.strip())

            chunks.append(
                Chunk(
                    content=text.strip(),
                    chunk_id=str(uuid.uuid4()),
                    source=source,
                    metadata=metadata.copy(),
                    start_index=start,
                    end_index=end,
                )
            )
            current_index = end

        return [c for c in chunks if c.content]

    def _split_on_boundaries(self, text: str) -> List[str]:
        """Split text on double newlines and markdown headers."""
        # First split on headers
        parts: List[str] = []
        header_positions = [m.start() for m in self.HEADER_PATTERN.finditer(text)]

        if not header_positions:
            # No headers, split on double newlines
            return [s for s in text.split("\n\n") if s.strip()]

        # Split at header positions
        prev = 0
        for pos in header_positions:
            if pos > prev:
                segment = text[prev:pos]
                if segment.strip():
                    parts.append(segment)
            prev = pos

        # Add the last segment
        if prev < len(text):
            parts.append(text[prev:])

        # Further split large sections on double newlines
        result: List[str] = []
        for part in parts:
            if len(part) > self.max_chunk_size:
                sub_parts = [s for s in part.split("\n\n") if s.strip()]
                result.extend(sub_parts)
            else:
                result.append(part)

        return result

    def _merge_small_sections(self, sections: List[str]) -> List[str]:
        """Merge consecutive small sections to avoid tiny chunks."""
        if not sections:
            return []

        min_size = self.max_chunk_size // 4
        merged: List[str] = []
        current = ""

        for section in sections:
            candidate = current + "\n\n" + section if current else section

            if len(candidate) <= self.max_chunk_size:
                current = candidate
            else:
                if current:
                    merged.append(current)
                # If section itself is too large, split it
                if len(section) > self.max_chunk_size:
                    sub_chunks = self._hard_split(section)
                    merged.extend(sub_chunks)
                    current = ""
                else:
                    current = section

        if current:
            merged.append(current)

        return merged

    def _hard_split(self, text: str) -> List[str]:
        """Hard split text that exceeds max_chunk_size by sentences, then by words."""
        sentences = re.split(r"(?<=[.!?])\s+", text)

        # If no sentence boundaries found (single long sentence), split by words
        if len(sentences) == 1 and len(text) > self.max_chunk_size:
            words = text.split()
            chunks: List[str] = []
            current = ""
            for word in words:
                candidate = current + " " + word if current else word
                if len(candidate) <= self.max_chunk_size:
                    current = candidate
                else:
                    if current:
                        chunks.append(current)
                    current = word
            if current:
                chunks.append(current)
            return chunks if chunks else [text[:self.max_chunk_size]]

        chunks: List[str] = []
        current = ""

        for sentence in sentences:
            candidate = current + " " + sentence if current else sentence
            if len(candidate) <= self.max_chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                # If a single sentence is still too long, split by words
                if len(sentence) > self.max_chunk_size:
                    words = sentence.split()
                    sub_current = ""
                    for word in words:
                        sub_candidate = sub_current + " " + word if sub_current else word
                        if len(sub_candidate) <= self.max_chunk_size:
                            sub_current = sub_candidate
                        else:
                            if sub_current:
                                chunks.append(sub_current)
                            sub_current = word
                    current = sub_current
                else:
                    current = sentence

        if current:
            chunks.append(current)

        return chunks
