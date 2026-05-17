"""Base chunker interface for RAGForge."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from ragforge.core.models import Chunk


class BaseChunker(ABC):
    """Abstract base class for all chunking strategies.

    Chunkers split documents into smaller pieces suitable for embedding
    and retrieval. Each implementation defines its own splitting logic.
    """

    def __init__(self, max_chunk_size: int = 512, overlap: int = 50):
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap

    @abstractmethod
    def chunk(
        self,
        content: str,
        source: str = "",
        metadata: Dict[str, Any] | None = None,
    ) -> List[Chunk]:
        """Split content into chunks.

        Args:
            content: The text content to chunk.
            source: Source identifier for the document.
            metadata: Optional metadata to attach to each chunk.

        Returns:
            List of Chunk objects.
        """
        ...
