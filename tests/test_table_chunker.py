"""Tests for TableChunker."""

import pytest

from ragforge.chunkers.table import TableChunker
from ragforge.core.models import Chunk


CSV_SAMPLE = """name,age,city,occupation
Alice,30,New York,Engineer
Bob,25,San Francisco,Designer
Charlie,35,Chicago,Manager
Diana,28,Boston,Analyst
Eve,32,Seattle,Developer
Frank,40,Austin,Director
Grace,27,Denver,Researcher
Henry,33,Portland,Architect
Iris,29,Miami,Consultant
Jack,31,Dallas,Product Manager"""

LARGE_CSV = "id,value,category\n" + "\n".join(
    f"{i},{i*10},cat_{i%5}" for i in range(100)
)

HEADER_ONLY_CSV = "name,age,city"

SIMPLE_CSV = """product,price,quantity
Widget,9.99,100
Gadget,19.99,50
Doohickey,4.99,200"""


class TestTableChunker:
    """Test table chunking with CSV data."""

    def test_basic_chunking(self):
        chunker = TableChunker(max_chunk_size=500, rows_per_chunk=5)
        chunks = chunker.chunk(CSV_SAMPLE, source="data.csv")

        assert len(chunks) > 0
        # Each chunk should contain the header
        for chunk in chunks:
            assert "name,age,city,occupation" in chunk.content

    def test_header_preserved_in_all_chunks(self):
        chunker = TableChunker(max_chunk_size=500, rows_per_chunk=3)
        chunks = chunker.chunk(CSV_SAMPLE, source="data.csv")

        header = "name,age,city,occupation"
        for chunk in chunks:
            lines = chunk.content.split("\n")
            assert lines[0] == header

    def test_rows_per_chunk_respected(self):
        chunker = TableChunker(max_chunk_size=5000, rows_per_chunk=3)
        chunks = chunker.chunk(CSV_SAMPLE, source="data.csv")

        for chunk in chunks:
            lines = chunk.content.split("\n")
            data_rows = lines[1:]  # Exclude header
            assert len(data_rows) <= 3

    def test_all_data_rows_present(self):
        chunker = TableChunker(max_chunk_size=5000, rows_per_chunk=3)
        chunks = chunker.chunk(CSV_SAMPLE, source="data.csv")

        # Collect all data rows across chunks
        all_rows = set()
        for chunk in chunks:
            lines = chunk.content.split("\n")
            for line in lines[1:]:  # Skip header
                all_rows.add(line)

        # Original has 10 data rows
        assert len(all_rows) == 10

    def test_metadata_row_info(self):
        chunker = TableChunker(max_chunk_size=5000, rows_per_chunk=3)
        chunks = chunker.chunk(CSV_SAMPLE, source="data.csv")

        for chunk in chunks:
            assert chunk.metadata["chunker"] == "table"
            assert chunk.metadata["has_header"] is True
            assert "row_count" in chunk.metadata
            assert chunk.metadata["row_count"] <= 3

    def test_metadata_column_count(self):
        chunker = TableChunker(max_chunk_size=5000, rows_per_chunk=5)
        chunks = chunker.chunk(CSV_SAMPLE, source="data.csv")

        for chunk in chunks:
            assert chunk.metadata["column_count"] == 4

    def test_large_csv_chunking(self):
        chunker = TableChunker(max_chunk_size=500, rows_per_chunk=10)
        chunks = chunker.chunk(LARGE_CSV, source="large.csv")

        assert len(chunks) > 1
        # Each chunk should have the header
        for chunk in chunks:
            assert chunk.content.startswith("id,value,category")

    def test_header_only_csv(self):
        chunker = TableChunker(max_chunk_size=500, rows_per_chunk=5)
        chunks = chunker.chunk(HEADER_ONLY_CSV, source="empty.csv")

        assert len(chunks) == 1
        assert chunks[0].content == "name,age,city"
        assert chunks[0].metadata["row_count"] == 0

    def test_max_chunk_size_limits_rows(self):
        # With a very small max_chunk_size, fewer rows should fit
        chunker = TableChunker(max_chunk_size=100, rows_per_chunk=100)
        chunks = chunker.chunk(CSV_SAMPLE, source="data.csv")

        # Should create multiple chunks due to size constraint
        assert len(chunks) > 1

    def test_chunk_ids_unique(self):
        chunker = TableChunker(max_chunk_size=500, rows_per_chunk=3)
        chunks = chunker.chunk(CSV_SAMPLE, source="data.csv")

        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_source_preserved(self):
        chunker = TableChunker(max_chunk_size=500, rows_per_chunk=5)
        chunks = chunker.chunk(CSV_SAMPLE, source="reports/data.csv")

        for chunk in chunks:
            assert chunk.source == "reports/data.csv"

    def test_custom_metadata_preserved(self):
        chunker = TableChunker(max_chunk_size=500, rows_per_chunk=5)
        chunks = chunker.chunk(
            CSV_SAMPLE, source="data.csv", metadata={"department": "sales"}
        )

        for chunk in chunks:
            assert chunk.metadata["department"] == "sales"

    def test_simple_csv(self):
        chunker = TableChunker(max_chunk_size=5000, rows_per_chunk=10)
        chunks = chunker.chunk(SIMPLE_CSV, source="products.csv")

        assert len(chunks) == 1
        assert "Widget" in chunks[0].content
        assert "Gadget" in chunks[0].content
        assert "Doohickey" in chunks[0].content
        assert chunks[0].metadata["row_count"] == 3

    def test_empty_content(self):
        chunker = TableChunker(max_chunk_size=500, rows_per_chunk=5)
        chunks = chunker.chunk("", source="empty.csv")

        assert chunks == []
