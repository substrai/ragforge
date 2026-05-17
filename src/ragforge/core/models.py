"""Core data models for RAGForge."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class Document:
    """A document to be ingested into the RAG pipeline."""

    content: str
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    doc_type: str = "text"


@dataclass
class Chunk:
    """A chunk of text with metadata."""

    content: str
    chunk_id: str
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    start_index: int = 0
    end_index: int = 0


@dataclass
class QueryResult:
    """A single retrieval result."""

    content: str
    score: float
    source: str
    chunk_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
