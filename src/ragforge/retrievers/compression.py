"""Contextual compression for retrieved chunks.

Compresses retrieved chunks by extracting only the most relevant
sentences based on query overlap, reducing token usage for LLM calls.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class CompressedResult:
    """Result of contextual compression."""

    text: str
    original_chunks: int
    sentences_kept: int
    sentences_total: int
    compression_ratio: float


class ContextualCompressor:
    """Compresses retrieved chunks to extract only query-relevant content.

    Uses sentence-level relevance scoring based on term overlap with
    the query to select the most informative sentences from each chunk.
    """

    def __init__(self, min_score: float = 0.1):
        """Initialize contextual compressor.

        Args:
            min_score: Minimum relevance score for a sentence to be kept.
        """
        self.min_score = min_score

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences.

        Args:
            text: Input text.

        Returns:
            List of sentence strings.
        """
        # Split on sentence-ending punctuation followed by space or end
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def _tokenize(self, text: str) -> set:
        """Tokenize text into lowercase word set.

        Args:
            text: Input text.

        Returns:
            Set of lowercase words.
        """
        words = re.findall(r'\b\w+\b', text.lower())
        return set(words)

    def _score_sentence(self, sentence: str, query_tokens: set) -> float:
        """Score a sentence based on term overlap with query.

        Args:
            sentence: The sentence to score.
            query_tokens: Set of query terms.

        Returns:
            Relevance score between 0.0 and 1.0.
        """
        if not query_tokens:
            return 0.0

        sentence_tokens = self._tokenize(sentence)
        if not sentence_tokens:
            return 0.0

        overlap = sentence_tokens & query_tokens
        # Jaccard-like score weighted toward query coverage
        query_coverage = len(overlap) / len(query_tokens)
        sentence_density = len(overlap) / len(sentence_tokens)

        # Weighted combination favoring query coverage
        return query_coverage * 0.7 + sentence_density * 0.3

    def compress(
        self,
        query: str,
        chunks: List[str],
        max_tokens: int = 1000,
    ) -> CompressedResult:
        """Compress chunks by extracting query-relevant sentences.

        Args:
            query: The query text for relevance scoring.
            chunks: List of chunk text strings.
            max_tokens: Approximate maximum tokens in output
                (estimated as words * 1.3).

        Returns:
            CompressedResult with compressed text and statistics.
        """
        query_tokens = self._tokenize(query)

        # Score all sentences across all chunks
        scored_sentences: List[tuple] = []  # (score, sentence, chunk_idx)
        total_sentences = 0

        for chunk_idx, chunk in enumerate(chunks):
            sentences = self._split_sentences(chunk)
            total_sentences += len(sentences)

            for sentence in sentences:
                score = self._score_sentence(sentence, query_tokens)
                if score >= self.min_score:
                    scored_sentences.append((score, sentence, chunk_idx))

        # Sort by score descending
        scored_sentences.sort(key=lambda x: x[0], reverse=True)

        # Select sentences up to max_tokens
        max_words = int(max_tokens / 1.3)  # Approximate tokens to words
        selected: List[str] = []
        current_words = 0

        for score, sentence, chunk_idx in scored_sentences:
            sentence_words = len(sentence.split())
            if current_words + sentence_words > max_words:
                break
            selected.append(sentence)
            current_words += sentence_words

        compressed_text = " ".join(selected)
        original_words = sum(len(chunk.split()) for chunk in chunks)
        compressed_words = len(compressed_text.split()) if compressed_text else 0

        compression_ratio = (
            1.0 - (compressed_words / original_words) if original_words > 0 else 0.0
        )

        return CompressedResult(
            text=compressed_text,
            original_chunks=len(chunks),
            sentences_kept=len(selected),
            sentences_total=total_sentences,
            compression_ratio=round(compression_ratio, 3),
        )
