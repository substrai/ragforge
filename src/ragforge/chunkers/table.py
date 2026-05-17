"""Table-aware chunker for CSV/tabular data."""

from __future__ import annotations

import uuid
from typing import Any, Dict, List

from ragforge.chunkers.base import BaseChunker
from ragforge.core.models import Chunk


class TableChunker(BaseChunker):
    """Splits CSV/table data while preserving headers with each chunk.

    Each chunk contains the header row followed by a subset of data rows,
    ensuring that every chunk is self-contained and interpretable.
    """

    def __init__(
        self,
        max_chunk_size: int = 512,
        overlap: int = 0,
        rows_per_chunk: int = 20,
        delimiter: str = ",",
    ):
        super().__init__(max_chunk_size=max_chunk_size, overlap=overlap)
        self.rows_per_chunk = rows_per_chunk
        self.delimiter = delimiter

    def chunk(
        self,
        content: str,
        source: str = "",
        metadata: Dict[str, Any] | None = None,
    ) -> List[Chunk]:
        """Split tabular content into chunks, each with headers.

        Args:
            content: The CSV/table content to chunk.
            source: Source identifier for the document.
            metadata: Optional metadata to attach to each chunk.

        Returns:
            List of Chunk objects, each containing headers + data rows.
        """
        metadata = metadata or {}

        stripped = content.strip()
        if not stripped:
            return []

        lines = stripped.split("\n")

        # First line is the header
        header = lines[0]
        data_rows = lines[1:]

        if not data_rows:
            # Only header, return as single chunk
            chunk_metadata = metadata.copy()
            chunk_metadata["chunker"] = "table"
            chunk_metadata["row_count"] = 0
            chunk_metadata["has_header"] = True
            return [
                Chunk(
                    content=header,
                    chunk_id=str(uuid.uuid4()),
                    source=source,
                    metadata=chunk_metadata,
                    start_index=0,
                    end_index=len(header),
                )
            ]

        # Determine column count from header
        column_count = len(header.split(self.delimiter))

        chunks: List[Chunk] = []
        current_index = len(header) + 1  # +1 for newline after header

        # Calculate effective rows per chunk based on max_chunk_size
        effective_rows = self._calculate_rows_per_chunk(header, data_rows)

        for i in range(0, len(data_rows), effective_rows):
            batch = data_rows[i : i + effective_rows]
            chunk_content = header + "\n" + "\n".join(batch)

            chunk_metadata = metadata.copy()
            chunk_metadata["chunker"] = "table"
            chunk_metadata["row_start"] = i
            chunk_metadata["row_end"] = i + len(batch)
            chunk_metadata["row_count"] = len(batch)
            chunk_metadata["column_count"] = column_count
            chunk_metadata["has_header"] = True

            start_idx = current_index
            end_idx = start_idx + sum(len(row) + 1 for row in batch)

            chunks.append(
                Chunk(
                    content=chunk_content,
                    chunk_id=str(uuid.uuid4()),
                    source=source,
                    metadata=chunk_metadata,
                    start_index=start_idx,
                    end_index=end_idx,
                )
            )

            current_index = end_idx

        return chunks

    def _calculate_rows_per_chunk(self, header: str, data_rows: List[str]) -> int:
        """Calculate how many rows fit in max_chunk_size including header."""
        header_size = len(header) + 1  # +1 for newline

        if not data_rows:
            return self.rows_per_chunk

        # Estimate average row size
        sample_size = min(10, len(data_rows))
        avg_row_size = sum(len(row) + 1 for row in data_rows[:sample_size]) / sample_size

        available_size = self.max_chunk_size - header_size
        if available_size <= 0:
            return 1

        calculated_rows = max(1, int(available_size / avg_row_size))
        return min(calculated_rows, self.rows_per_chunk)
