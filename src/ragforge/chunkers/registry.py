"""Chunker registry for auto-selecting chunking strategy."""

from __future__ import annotations

from typing import Dict, Optional

from ragforge.chunkers.base import BaseChunker
from ragforge.chunkers.recursive import RecursiveChunker
from ragforge.chunkers.semantic import SemanticChunker
from ragforge.chunkers.code_aware import CodeAwareChunker
from ragforge.chunkers.table import TableChunker
from ragforge.chunkers.hierarchical import HierarchicalChunker


# Mapping of document types to preferred chunking strategies
DOC_TYPE_STRATEGY_MAP: Dict[str, str] = {
    "md": "semantic",
    "markdown": "semantic",
    "rst": "semantic",
    "txt": "recursive",
    "text": "recursive",
    "pdf": "recursive",
    "html": "semantic",
    "py": "code_aware",
    "js": "code_aware",
    "ts": "code_aware",
    "json": "recursive",
    "yaml": "recursive",
    "yml": "recursive",
    "csv": "table",
}


class ChunkerRegistry:
    """Registry that auto-selects chunking strategy based on document type.

    Provides a factory method to get the appropriate chunker for a given
    document type and strategy configuration.
    """

    def __init__(self):
        self._chunkers: Dict[str, type] = {
            "recursive": RecursiveChunker,
            "semantic": SemanticChunker,
            "code_aware": CodeAwareChunker,
            "table": TableChunker,
            "hierarchical": HierarchicalChunker,
        }

    def register(self, name: str, chunker_class: type) -> None:
        """Register a custom chunker class."""
        self._chunkers[name] = chunker_class

    def get_chunker(
        self,
        doc_type: str = "text",
        strategy: str = "auto",
        max_chunk_size: int = 512,
        overlap: int = 50,
    ) -> BaseChunker:
        """Get the appropriate chunker for the given document type and strategy.

        Args:
            doc_type: The document type (e.g., 'md', 'pdf', 'txt').
            strategy: Chunking strategy. 'auto' selects based on doc_type.
            max_chunk_size: Maximum chunk size in characters.
            overlap: Number of overlapping characters between chunks.

        Returns:
            An instance of the appropriate BaseChunker subclass.
        """
        if strategy == "auto":
            strategy = DOC_TYPE_STRATEGY_MAP.get(doc_type, "recursive")

        chunker_class = self._chunkers.get(strategy, RecursiveChunker)
        return chunker_class(max_chunk_size=max_chunk_size, overlap=overlap)
