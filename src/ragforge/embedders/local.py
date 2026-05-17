"""Local embedder for development and testing."""

from __future__ import annotations

import hashlib
import math
import random
from typing import List

from ragforge.embedders.base import BaseEmbedder


class LocalEmbedder(BaseEmbedder):
    """Generates deterministic normalized vectors for dev/testing.

    Uses a hash-based approach to generate consistent embeddings for
    the same input text, without requiring any external API calls.
    This is useful for local development and testing.
    """

    def __init__(self, dimensions: int = 1024, seed: int = 42):
        super().__init__(dimensions=dimensions)
        self.seed = seed

    def embed(self, text: str) -> List[float]:
        """Generate a deterministic normalized embedding for text.

        Uses SHA-256 hash of the text as a seed to produce consistent
        vectors for the same input.
        """
        # Create a deterministic seed from the text content
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        text_seed = int(text_hash[:16], 16)
        rng = random.Random(text_seed)

        # Generate random vector
        vector = [rng.gauss(0, 1) for _ in range(self.dimensions)]

        # Normalize to unit length
        magnitude = math.sqrt(sum(x * x for x in vector))
        if magnitude > 0:
            vector = [x / magnitude for x in vector]

        return vector

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""
        return [self.embed(text) for text in texts]
