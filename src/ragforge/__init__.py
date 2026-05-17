"""RAGForge - Config-driven enterprise RAG architecture generator."""

from ragforge.core.config import RAGConfig, load_config
from ragforge.core.models import Chunk, Document, QueryResult

__version__ = "0.1.0"
__all__ = [
    "RAGConfig",
    "Chunk",
    "Document",
    "QueryResult",
    "load_config",
]


def __getattr__(name):
    """Lazy imports for pipeline components to avoid circular imports."""
    if name == "RAGPipeline":
        from ragforge.core.pipeline import RAGPipeline
        return RAGPipeline
    if name == "ChunkerRegistry":
        from ragforge.chunkers.registry import ChunkerRegistry
        return ChunkerRegistry
    if name == "BaseEmbedder":
        from ragforge.embedders.base import BaseEmbedder
        return BaseEmbedder
    if name == "BaseRetriever":
        from ragforge.retrievers.base import BaseRetriever
        return BaseRetriever
    if name == "BaseVectorStore":
        from ragforge.storage.base import BaseVectorStore
        return BaseVectorStore
    raise AttributeError(f"module 'ragforge' has no attribute {name!r}")
