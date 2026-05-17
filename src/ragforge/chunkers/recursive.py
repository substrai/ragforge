"""Recursive text chunker for RAGForge."""

from __future__ import annotations

import uuid
from typing import Any, Dict, List

from ragforge.chunkers.base import BaseChunker
from ragforge.core.models import Chunk


class RecursiveChunker(BaseChunker):
    """Splits text recursively by separators with overlap.

    Tries to split on paragraph boundaries first, then sentences,
    then words, preserving semantic coherence at each level.
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(
        self,
        max_chunk_size: int = 512,
        overlap: int = 50,
        separators: List[str] | None = None,
    ):
        super().__init__(max_chunk_size=max_chunk_size, overlap=overlap)
        self.separators = separators or self.DEFAULT_SEPARATORS

    def chunk(
        self,
        content: str,
        source: str = "",
        metadata: Dict[str, Any] | None = None,
    ) -> List[Chunk]:
        """Split content recursively by separators with overlap."""
        metadata = metadata or {}
        raw_chunks = self._split_recursive(content, self.separators)
        merged = self._merge_with_overlap(raw_chunks)

        chunks: List[Chunk] = []
        current_index = 0

        for text in merged:
            start = content.find(text, max(0, current_index - self.overlap))
            if start == -1:
                start = current_index
            end = start + len(text)

            chunks.append(
                Chunk(
                    content=text,
                    chunk_id=str(uuid.uuid4()),
                    source=source,
                    metadata=metadata.copy(),
                    start_index=start,
                    end_index=end,
                )
            )
            current_index = end

        return chunks

    def _split_recursive(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text using the separator hierarchy."""
        if not text:
            return []

        if len(text) <= self.max_chunk_size:
            return [text]

        if not separators:
            # Last resort: hard split at max_chunk_size
            return [
                text[i : i + self.max_chunk_size]
                for i in range(0, len(text), self.max_chunk_size)
            ]

        separator = separators[0]
        remaining_separators = separators[1:]

        if separator == "":
            # Character-level split
            return [
                text[i : i + self.max_chunk_size]
                for i in range(0, len(text), self.max_chunk_size)
            ]

        parts = text.split(separator)
        results: List[str] = []
        current = ""

        for part in parts:
            candidate = current + separator + part if current else part

            if len(candidate) <= self.max_chunk_size:
                current = candidate
            else:
                if current:
                    results.append(current)
                # If the part itself is too large, recurse with next separator
                if len(part) > self.max_chunk_size:
                    sub_parts = self._split_recursive(part, remaining_separators)
                    results.extend(sub_parts)
                    current = ""
                else:
                    current = part

        if current:
            results.append(current)

        return results

    def _merge_with_overlap(self, chunks: List[str]) -> List[str]:
        """Merge chunks to include overlap from previous chunk."""
        if not chunks or self.overlap <= 0:
            return chunks

        merged: List[str] = [chunks[0]]

        for i in range(1, len(chunks)):
            prev = chunks[i - 1]
            overlap_text = prev[-self.overlap :] if len(prev) > self.overlap else prev
            merged_text = overlap_text + chunks[i]

            # Trim if merged text exceeds max size
            if len(merged_text) > self.max_chunk_size:
                merged_text = merged_text[: self.max_chunk_size]

            merged.append(merged_text)

        return merged
