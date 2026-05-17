"""Retrieval evaluation metrics for RAGForge.

Provides standard IR metrics for measuring retrieval quality:
- Precision@k, Recall@k
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (NDCG@k)
- Answer relevancy score (placeholder for LLM judge)
"""

from __future__ import annotations

import math
from typing import List


def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """Compute Precision@k: fraction of top-k results that are relevant.

    Args:
        retrieved: Ordered list of retrieved chunk IDs or content strings.
        relevant: List of relevant chunk IDs or content strings.
        k: Number of top results to consider.

    Returns:
        Precision score between 0.0 and 1.0.
    """
    if k <= 0:
        return 0.0

    top_k = retrieved[:k]
    relevant_set = set(relevant)
    relevant_in_top_k = sum(1 for item in top_k if item in relevant_set)
    return relevant_in_top_k / k


def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """Compute Recall@k: fraction of relevant docs found in top-k.

    Args:
        retrieved: Ordered list of retrieved chunk IDs or content strings.
        relevant: List of relevant chunk IDs or content strings.
        k: Number of top results to consider.

    Returns:
        Recall score between 0.0 and 1.0.
    """
    if not relevant:
        return 0.0
    if k <= 0:
        return 0.0

    top_k = retrieved[:k]
    relevant_set = set(relevant)
    relevant_in_top_k = sum(1 for item in top_k if item in relevant_set)
    return relevant_in_top_k / len(relevant_set)


def mrr(retrieved: List[str], relevant: List[str]) -> float:
    """Compute Mean Reciprocal Rank: 1/rank of first relevant result.

    Args:
        retrieved: Ordered list of retrieved chunk IDs or content strings.
        relevant: List of relevant chunk IDs or content strings.

    Returns:
        MRR score between 0.0 and 1.0. Returns 0.0 if no relevant result found.
    """
    relevant_set = set(relevant)

    for i, item in enumerate(retrieved, start=1):
        if item in relevant_set:
            return 1.0 / i

    return 0.0


def ndcg_at_k(retrieved: List[str], relevance_scores: List[float], k: int) -> float:
    """Compute Normalized Discounted Cumulative Gain at k.

    DCG = sum(relevance_i / log2(i+1)) for i=1..k
    NDCG = DCG / ideal_DCG

    Args:
        retrieved: Ordered list of retrieved chunk IDs or content strings.
        relevance_scores: Relevance score for each retrieved item (same order).
            Items not in this list are assumed to have relevance 0.
        k: Number of top results to consider.

    Returns:
        NDCG score between 0.0 and 1.0.
    """
    if k <= 0 or not retrieved or not relevance_scores:
        return 0.0

    # Get scores for top-k items
    scores = relevance_scores[:k]

    # Pad with zeros if fewer scores than k
    while len(scores) < min(k, len(retrieved)):
        scores.append(0.0)

    # Compute DCG
    dcg = 0.0
    for i, score in enumerate(scores):
        dcg += score / math.log2(i + 2)  # i+2 because i is 0-indexed, formula uses 1-indexed (i+1 in log)

    # Compute ideal DCG (sort scores descending)
    ideal_scores = sorted(relevance_scores[:], reverse=True)[:k]
    idcg = 0.0
    for i, score in enumerate(ideal_scores):
        idcg += score / math.log2(i + 2)

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


def answer_relevancy_score(query: str, answer: str, context: str) -> float:
    """Compute answer relevancy score (placeholder for LLM judge integration).

    This is a placeholder that returns a heuristic score based on term overlap.
    In production, this would call an LLM to judge answer quality.

    Args:
        query: The original query string.
        answer: The generated answer.
        context: The retrieved context used to generate the answer.

    Returns:
        Relevancy score between 0.0 and 1.0.
    """
    if not query or not answer:
        return 0.0

    # Simple heuristic: term overlap between query and answer
    query_terms = set(query.lower().split())
    answer_terms = set(answer.lower().split())

    if not query_terms:
        return 0.0

    overlap = len(query_terms & answer_terms)
    return min(1.0, overlap / len(query_terms))
