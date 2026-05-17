"""Hierarchical chunker that creates parent-child chunk relationships."""

from __future__ import annotations

import re
import uuid
from typing import Any, Dict, List

from ragforge.chunkers.base import BaseChunker
from ragforge.core.models import Chunk


# Patterns for detecting section boundaries (headings)
HEADING_PATTERNS = [
    re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE),  # Markdown headings
    re.compile(r"^(.+)\n[=]{3,}$", re.MULTILINE),  # Setext h1
    re.compile(r"^(.+)\n[-]{3,}$", re.MULTILINE),  # Setext h2
]


class HierarchicalChunker(BaseChunker):
    """Creates parent-child chunk relationships.

    Parent chunks represent sections (identified by headings), and child
    chunks are paragraphs within those sections. Each child chunk's metadata
    includes a reference to its parent_chunk_id.
    """

    def __init__(
        self,
        max_chunk_size: int = 512,
        overlap: int = 50,
        include_parents: bool = True,
    ):
        super().__init__(max_chunk_size=max_chunk_size, overlap=overlap)
        self.include_parents = include_parents

    def chunk(
        self,
        content: str,
        source: str = "",
        metadata: Dict[str, Any] | None = None,
    ) -> List[Chunk]:
        """Split content into hierarchical parent-child chunks.

        Args:
            content: The text content to chunk.
            source: Source identifier for the document.
            metadata: Optional metadata to attach to each chunk.

        Returns:
            List of Chunk objects with parent-child relationships in metadata.
        """
        metadata = metadata or {}

        sections = self._split_into_sections(content)
        chunks: List[Chunk] = []

        for section_title, section_content, section_start in sections:
            parent_id = str(uuid.uuid4())

            # Create parent chunk (the full section)
            parent_metadata = metadata.copy()
            parent_metadata["chunker"] = "hierarchical"
            parent_metadata["chunk_type"] = "parent"
            parent_metadata["section_title"] = section_title

            parent_text = section_content
            if section_title:
                parent_text = section_title + "\n\n" + section_content

            if self.include_parents:
                # Truncate parent if too large (store summary)
                parent_content = parent_text
                if len(parent_content) > self.max_chunk_size * 2:
                    parent_content = parent_content[: self.max_chunk_size * 2]

                chunks.append(
                    Chunk(
                        content=parent_content,
                        chunk_id=parent_id,
                        source=source,
                        metadata=parent_metadata,
                        start_index=section_start,
                        end_index=section_start + len(parent_content),
                    )
                )

            # Create child chunks (paragraphs within the section)
            children = self._split_into_paragraphs(section_content, section_start)

            for para_text, para_start in children:
                child_metadata = metadata.copy()
                child_metadata["chunker"] = "hierarchical"
                child_metadata["chunk_type"] = "child"
                child_metadata["parent_chunk_id"] = parent_id
                child_metadata["section_title"] = section_title

                # Split paragraph further if too large
                if len(para_text) > self.max_chunk_size:
                    sub_chunks = self._split_large_paragraph(
                        para_text, para_start, source, child_metadata, parent_id
                    )
                    chunks.extend(sub_chunks)
                else:
                    chunks.append(
                        Chunk(
                            content=para_text,
                            chunk_id=str(uuid.uuid4()),
                            source=source,
                            metadata=child_metadata,
                            start_index=para_start,
                            end_index=para_start + len(para_text),
                        )
                    )

        return chunks

    def _split_into_sections(self, content: str) -> List[tuple]:
        """Split content into sections based on headings.

        Returns list of (title, content, start_index) tuples.
        """
        # Find all markdown headings
        heading_positions: List[tuple] = []

        for pattern in HEADING_PATTERNS:
            for match in pattern.finditer(content):
                heading_positions.append((match.start(), match.group(0)))

        heading_positions.sort(key=lambda x: x[0])

        if not heading_positions:
            # No headings found, treat entire content as one section
            return [("", content, 0)]

        sections: List[tuple] = []

        # Content before first heading
        first_pos = heading_positions[0][0]
        if first_pos > 0:
            preamble = content[:first_pos].strip()
            if preamble:
                sections.append(("", preamble, 0))

        # Sections between headings
        for i, (pos, heading_text) in enumerate(heading_positions):
            # Extract title from heading
            title = self._extract_title(heading_text)

            # Section content is from after heading to next heading
            content_start = pos + len(heading_text)
            if i + 1 < len(heading_positions):
                content_end = heading_positions[i + 1][0]
            else:
                content_end = len(content)

            section_content = content[content_start:content_end].strip()
            if section_content or title:
                sections.append((title, section_content, pos))

        return sections if sections else [("", content, 0)]

    def _extract_title(self, heading_text: str) -> str:
        """Extract clean title from a heading line."""
        # Remove markdown heading markers
        title = re.sub(r"^#{1,6}\s+", "", heading_text.strip())
        # Remove setext underlines
        title = title.split("\n")[0].strip()
        return title

    def _split_into_paragraphs(
        self, content: str, base_offset: int
    ) -> List[tuple]:
        """Split section content into paragraphs.

        Returns list of (text, start_index) tuples.
        """
        if not content.strip():
            return []

        paragraphs: List[tuple] = []
        # Split on double newlines (paragraph boundaries)
        parts = re.split(r"\n\s*\n", content)

        current_offset = 0
        for part in parts:
            stripped = part.strip()
            if stripped:
                # Find actual position in content
                actual_pos = content.find(part, current_offset)
                if actual_pos == -1:
                    actual_pos = current_offset
                paragraphs.append((stripped, base_offset + actual_pos))
                current_offset = actual_pos + len(part)

        return paragraphs

    def _split_large_paragraph(
        self,
        text: str,
        start_idx: int,
        source: str,
        base_metadata: Dict[str, Any],
        parent_id: str,
    ) -> List[Chunk]:
        """Split a large paragraph into smaller chunks with overlap."""
        chunks: List[Chunk] = []
        sentences = re.split(r"(?<=[.!?])\s+", text)

        current = ""
        current_start = start_idx

        for sentence in sentences:
            candidate = current + " " + sentence if current else sentence
            if len(candidate) > self.max_chunk_size and current:
                child_metadata = base_metadata.copy()
                child_metadata["parent_chunk_id"] = parent_id
                chunks.append(
                    Chunk(
                        content=current,
                        chunk_id=str(uuid.uuid4()),
                        source=source,
                        metadata=child_metadata,
                        start_index=current_start,
                        end_index=current_start + len(current),
                    )
                )
                current_start += len(current)
                current = sentence
            else:
                current = candidate

        if current.strip():
            child_metadata = base_metadata.copy()
            child_metadata["parent_chunk_id"] = parent_id
            chunks.append(
                Chunk(
                    content=current,
                    chunk_id=str(uuid.uuid4()),
                    source=source,
                    metadata=child_metadata,
                    start_index=current_start,
                    end_index=current_start + len(current),
                )
            )

        return chunks
