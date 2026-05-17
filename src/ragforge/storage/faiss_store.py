"""FAISS-based vector store for RAGForge."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from ragforge.storage.base import BaseVectorStore, SearchResult


class FAISSVectorStore(BaseVectorStore):
    """Vector store using FAISS for local/dev similarity search.

    Supports in-memory operation with optional persistence to disk.
    Falls back to numpy-based cosine similarity if faiss-cpu is not installed.
    """

    def __init__(
        self,
        index_name: str = "ragforge-index",
        dimensions: int = 1024,
        persist_dir: Optional[str] = None,
    ):
        self.index_name = index_name
        self.dimensions = dimensions
        self.persist_dir = persist_dir

        # Internal storage
        self._vectors: Dict[str, List[float]] = {}
        self._contents: Dict[str, str] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}

        # Try to use FAISS, fall back to numpy cosine similarity
        self._faiss_index = None
        self._faiss_id_map: List[str] = []
        self._use_faiss = False

        try:
            import faiss
            self._faiss = faiss
            self._faiss_index = faiss.IndexFlatIP(dimensions)  # Inner product (cosine on normalized)
            self._use_faiss = True
        except ImportError:
            self._faiss = None

        # Load persisted data if available
        if persist_dir:
            self._load_from_disk()

    def upsert(
        self,
        chunk_id: str,
        embedding: List[float],
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Insert or update a vector in the store."""
        # Normalize the embedding for cosine similarity
        normalized = self._normalize(embedding)

        # Remove old entry if updating
        if chunk_id in self._vectors:
            self._remove_from_faiss(chunk_id)

        self._vectors[chunk_id] = normalized
        self._contents[chunk_id] = content
        self._metadata[chunk_id] = metadata or {}

        # Add to FAISS index
        if self._use_faiss:
            import numpy as np
            vector_np = np.array([normalized], dtype=np.float32)
            self._faiss_index.add(vector_np)
            self._faiss_id_map.append(chunk_id)

    def search(
        self,
        embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search for similar vectors using cosine similarity."""
        if not self._vectors:
            return []

        normalized_query = self._normalize(embedding)

        if self._use_faiss and self._faiss_index.ntotal > 0:
            results = self._search_faiss(normalized_query, top_k, filters)
        else:
            results = self._search_numpy(normalized_query, top_k, filters)

        return results

    def delete(self, chunk_id: str) -> None:
        """Delete a vector from the store."""
        if chunk_id in self._vectors:
            self._remove_from_faiss(chunk_id)
            del self._vectors[chunk_id]
            del self._contents[chunk_id]
            del self._metadata[chunk_id]

    def persist(self) -> None:
        """Persist the index to disk."""
        if not self.persist_dir:
            return

        os.makedirs(self.persist_dir, exist_ok=True)
        data = {
            "vectors": self._vectors,
            "contents": self._contents,
            "metadata": self._metadata,
        }

        path = Path(self.persist_dir) / f"{self.index_name}.json"
        with open(path, "w") as f:
            json.dump(data, f)

    def _load_from_disk(self) -> None:
        """Load persisted index from disk."""
        if not self.persist_dir:
            return

        path = Path(self.persist_dir) / f"{self.index_name}.json"
        if not path.exists():
            return

        with open(path, "r") as f:
            data = json.load(f)

        for chunk_id, vector in data.get("vectors", {}).items():
            self._vectors[chunk_id] = vector
            self._contents[chunk_id] = data["contents"].get(chunk_id, "")
            self._metadata[chunk_id] = data["metadata"].get(chunk_id, {})

            if self._use_faiss:
                import numpy as np
                vector_np = np.array([vector], dtype=np.float32)
                self._faiss_index.add(vector_np)
                self._faiss_id_map.append(chunk_id)

    def _search_faiss(
        self,
        query: List[float],
        top_k: int,
        filters: Optional[Dict[str, Any]],
    ) -> List[SearchResult]:
        """Search using FAISS index."""
        import numpy as np

        query_np = np.array([query], dtype=np.float32)
        k = min(top_k, self._faiss_index.ntotal)

        if k == 0:
            return []

        # Over-fetch if filtering
        search_k = min(k * 3, self._faiss_index.ntotal) if filters else k
        scores, indices = self._faiss_index.search(query_np, search_k)

        results: List[SearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._faiss_id_map):
                continue

            chunk_id = self._faiss_id_map[idx]
            meta = self._metadata.get(chunk_id, {})

            # Apply filters
            if filters and not self._matches_filters(meta, filters):
                continue

            results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    content=self._contents.get(chunk_id, ""),
                    score=float(score),
                    metadata=meta,
                )
            )

            if len(results) >= top_k:
                break

        return results

    def _search_numpy(
        self,
        query: List[float],
        top_k: int,
        filters: Optional[Dict[str, Any]],
    ) -> List[SearchResult]:
        """Fallback search using numpy-style cosine similarity."""
        scored: List[tuple] = []

        for chunk_id, vector in self._vectors.items():
            meta = self._metadata.get(chunk_id, {})

            # Apply filters
            if filters and not self._matches_filters(meta, filters):
                continue

            score = self._cosine_similarity(query, vector)
            scored.append((chunk_id, score))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        results: List[SearchResult] = []
        for chunk_id, score in scored[:top_k]:
            results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    content=self._contents.get(chunk_id, ""),
                    score=score,
                    metadata=self._metadata.get(chunk_id, {}),
                )
            )

        return results

    def _remove_from_faiss(self, chunk_id: str) -> None:
        """Remove a vector from the FAISS index (rebuild required)."""
        if not self._use_faiss or chunk_id not in self._faiss_id_map:
            return

        # FAISS IndexFlatIP doesn't support removal, rebuild
        idx = self._faiss_id_map.index(chunk_id)
        self._faiss_id_map.pop(idx)

        # Rebuild index
        import numpy as np
        self._faiss_index = self._faiss.IndexFlatIP(self.dimensions)
        if self._faiss_id_map:
            vectors = [self._vectors[cid] for cid in self._faiss_id_map if cid in self._vectors]
            if vectors:
                vectors_np = np.array(vectors, dtype=np.float32)
                self._faiss_index.add(vectors_np)

    @staticmethod
    def _normalize(vector: List[float]) -> List[float]:
        """Normalize a vector to unit length."""
        magnitude = math.sqrt(sum(x * x for x in vector))
        if magnitude > 0:
            return [x / magnitude for x in vector]
        return vector

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two normalized vectors (dot product)."""
        return sum(x * y for x, y in zip(a, b))

    @staticmethod
    def _matches_filters(metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if metadata matches all filter criteria."""
        for key, value in filters.items():
            if key not in metadata:
                return False
            if metadata[key] != value:
                return False
        return True
