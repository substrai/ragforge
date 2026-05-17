"""Hybrid retriever combining semantic and keyword scoring."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List

from ragforge.retrievers.base import BaseRetriever, RetrievalResult
from ragforge.storage.base import SearchResult


class HybridRetriever(BaseRetriever):
    """Combines semantic similarity scores with keyword matching (BM25-style).

    Produces a weighted combination of vector similarity and term-frequency
    based relevance for improved retrieval quality.
    """

    def __init__(
        self,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        top_k: int = 5,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        self.top_k = top_k
        # BM25 parameters
        self.k1 = k1
        self.b = b

    def retrieve(
        self,
        query_text: str,
        query_embedding: List[float],
        candidates: List[SearchResult],
        top_k: int = 5,
    ) -> List[RetrievalResult]:
        """Retrieve and rank results using hybrid scoring.

        Combines the semantic similarity score from the vector store with
        a BM25-style keyword relevance score.

        Args:
            query_text: The original query string.
            query_embedding: The query embedding vector (unused in keyword scoring).
            candidates: Raw search results from the vector store.
            top_k: Number of results to return.

        Returns:
            List of RetrievalResult objects ranked by combined score.
        """
        if not candidates:
            return []

        k = top_k or self.top_k
        query_terms = self._tokenize(query_text)

        # Compute BM25-style keyword scores
        doc_lengths = [len(self._tokenize(c.content)) for c in candidates]
        avg_doc_length = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 1.0

        # Document frequency for IDF
        df: Dict[str, int] = Counter()
        for candidate in candidates:
            terms = set(self._tokenize(candidate.content))
            for term in terms:
                df[term] += 1

        n_docs = len(candidates)
        scored_results: List[RetrievalResult] = []

        for i, candidate in enumerate(candidates):
            # Semantic score (already from vector store, normalize to 0-1)
            semantic_score = max(0.0, min(1.0, candidate.score))

            # BM25-style keyword score
            keyword_score = self._bm25_score(
                query_terms=query_terms,
                doc_content=candidate.content,
                doc_length=doc_lengths[i],
                avg_doc_length=avg_doc_length,
                df=df,
                n_docs=n_docs,
            )

            # Combine scores
            combined_score = (
                self.semantic_weight * semantic_score
                + self.keyword_weight * keyword_score
            )

            scored_results.append(
                RetrievalResult(
                    chunk_id=candidate.chunk_id,
                    content=candidate.content,
                    score=combined_score,
                    metadata=candidate.metadata,
                )
            )

        # Sort by combined score descending
        scored_results.sort(key=lambda r: r.score, reverse=True)
        return scored_results[:k]

    def _bm25_score(
        self,
        query_terms: List[str],
        doc_content: str,
        doc_length: int,
        avg_doc_length: float,
        df: Dict[str, int],
        n_docs: int,
    ) -> float:
        """Compute BM25-style relevance score for a document."""
        doc_terms = self._tokenize(doc_content)
        tf: Dict[str, int] = Counter(doc_terms)

        score = 0.0
        for term in query_terms:
            if term not in tf:
                continue

            # IDF component
            doc_freq = df.get(term, 0)
            idf = math.log((n_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)

            # TF component with length normalization
            term_freq = tf[term]
            tf_norm = (term_freq * (self.k1 + 1)) / (
                term_freq + self.k1 * (1 - self.b + self.b * doc_length / avg_doc_length)
            )

            score += idf * tf_norm

        # Normalize score to 0-1 range
        if query_terms:
            max_possible = len(query_terms) * math.log(n_docs + 1) * (self.k1 + 1)
            if max_possible > 0:
                score = min(1.0, score / max_possible)

        return score

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple whitespace + punctuation tokenizer."""
        text = text.lower()
        tokens = re.findall(r"\b\w+\b", text)
        return tokens
