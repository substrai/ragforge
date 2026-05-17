"""Code-aware chunker that splits on function/class boundaries."""

from __future__ import annotations

import re
import uuid
from typing import Any, Dict, List

from ragforge.chunkers.base import BaseChunker
from ragforge.core.models import Chunk


# Patterns for detecting code boundaries
PYTHON_PATTERNS = [
    re.compile(r"^(class\s+\w+)", re.MULTILINE),
    re.compile(r"^(def\s+\w+)", re.MULTILINE),
    re.compile(r"^(async\s+def\s+\w+)", re.MULTILINE),
]

JS_TS_PATTERNS = [
    re.compile(r"^(class\s+\w+)", re.MULTILINE),
    re.compile(r"^(function\s+\w+)", re.MULTILINE),
    re.compile(r"^(export\s+function\s+\w+)", re.MULTILINE),
    re.compile(r"^(export\s+default\s+function\s+\w+)", re.MULTILINE),
    re.compile(r"^(const\s+\w+\s*=\s*\(.*?\)\s*=>)", re.MULTILINE),
    re.compile(r"^(const\s+\w+\s*=\s*async\s*\(.*?\)\s*=>)", re.MULTILINE),
    re.compile(r"^(export\s+const\s+\w+\s*=\s*\(.*?\)\s*=>)", re.MULTILINE),
]

LANGUAGE_PATTERNS: Dict[str, List[re.Pattern]] = {
    "py": PYTHON_PATTERNS,
    "python": PYTHON_PATTERNS,
    "js": JS_TS_PATTERNS,
    "javascript": JS_TS_PATTERNS,
    "ts": JS_TS_PATTERNS,
    "typescript": JS_TS_PATTERNS,
}


class CodeAwareChunker(BaseChunker):
    """Splits code on function/class boundaries.

    Detects language-specific patterns (Python def/class, JS/TS function/const arrow)
    and creates chunks that align with logical code units.
    """

    def __init__(
        self,
        max_chunk_size: int = 512,
        overlap: int = 50,
        language: str = "py",
    ):
        super().__init__(max_chunk_size=max_chunk_size, overlap=overlap)
        self.language = language

    def chunk(
        self,
        content: str,
        source: str = "",
        metadata: Dict[str, Any] | None = None,
    ) -> List[Chunk]:
        """Split code content on function/class boundaries.

        Args:
            content: The source code to chunk.
            source: Source identifier for the document.
            metadata: Optional metadata to attach to each chunk.

        Returns:
            List of Chunk objects aligned to code boundaries.
        """
        metadata = metadata or {}

        # Detect language from source extension or explicit setting
        language = self._detect_language(source)
        patterns = LANGUAGE_PATTERNS.get(language, PYTHON_PATTERNS)

        # Find all boundary positions
        boundaries = self._find_boundaries(content, patterns)

        # Split content at boundaries
        segments = self._split_at_boundaries(content, boundaries)

        # Merge small segments and split large ones
        merged_segments = self._merge_segments(segments)

        chunks: List[Chunk] = []
        for segment_text, start_idx in merged_segments:
            chunk_metadata = metadata.copy()
            chunk_metadata["language"] = language
            chunk_metadata["chunker"] = "code_aware"

            chunks.append(
                Chunk(
                    content=segment_text,
                    chunk_id=str(uuid.uuid4()),
                    source=source,
                    metadata=chunk_metadata,
                    start_index=start_idx,
                    end_index=start_idx + len(segment_text),
                )
            )

        return chunks

    def _detect_language(self, source: str) -> str:
        """Detect programming language from file extension."""
        if not source:
            return self.language

        ext = source.rsplit(".", 1)[-1].lower() if "." in source else ""
        if ext in LANGUAGE_PATTERNS:
            return ext
        return self.language

    def _find_boundaries(self, content: str, patterns: List[re.Pattern]) -> List[int]:
        """Find all code boundary positions in the content."""
        boundaries: set = set()

        for pattern in patterns:
            for match in pattern.finditer(content):
                boundaries.add(match.start())

        return sorted(boundaries)

    def _split_at_boundaries(
        self, content: str, boundaries: List[int]
    ) -> List[tuple]:
        """Split content at detected boundaries, returning (text, start_index) tuples."""
        if not boundaries:
            # No boundaries found, treat as single segment
            return [(content, 0)] if content.strip() else []

        segments: List[tuple] = []

        # Content before first boundary (imports, module docstring, etc.)
        if boundaries[0] > 0:
            preamble = content[: boundaries[0]]
            if preamble.strip():
                segments.append((preamble, 0))

        # Content between boundaries
        for i, start in enumerate(boundaries):
            end = boundaries[i + 1] if i + 1 < len(boundaries) else len(content)
            segment = content[start:end]
            if segment.strip():
                segments.append((segment, start))

        return segments

    def _merge_segments(
        self, segments: List[tuple]
    ) -> List[tuple]:
        """Merge small segments and split large ones to respect size limits."""
        if not segments:
            return []

        result: List[tuple] = []
        current_text = ""
        current_start = 0

        for text, start_idx in segments:
            if not current_text:
                current_text = text
                current_start = start_idx
            elif len(current_text) + len(text) <= self.max_chunk_size:
                current_text += text
            else:
                # Flush current
                if len(current_text) > self.max_chunk_size:
                    # Split oversized segment
                    for sub_chunk in self._hard_split(current_text, current_start):
                        result.append(sub_chunk)
                else:
                    result.append((current_text, current_start))
                current_text = text
                current_start = start_idx

        # Flush remaining
        if current_text:
            if len(current_text) > self.max_chunk_size:
                for sub_chunk in self._hard_split(current_text, current_start):
                    result.append(sub_chunk)
            else:
                result.append((current_text, current_start))

        return result

    def _hard_split(self, text: str, start_idx: int) -> List[tuple]:
        """Hard-split oversized text at line boundaries."""
        lines = text.split("\n")
        chunks: List[tuple] = []
        current = ""
        current_offset = start_idx

        for line in lines:
            candidate = current + line + "\n" if current else line + "\n"
            if len(candidate) > self.max_chunk_size and current:
                chunks.append((current, current_offset))
                current_offset += len(current)
                current = line + "\n"
            else:
                current = candidate

        if current.strip():
            chunks.append((current, current_offset))

        return chunks
