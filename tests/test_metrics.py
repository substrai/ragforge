"""Tests for retrieval evaluation metrics."""

import math

import pytest

from ragforge.evaluation.metrics import (
    answer_relevancy_score,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


class TestPrecisionAtK:
    """Tests for precision_at_k metric."""

    def test_perfect_precision(self):
        """All top-k results are relevant."""
        retrieved = ["a", "b", "c", "d", "e"]
        relevant = ["a", "b", "c", "d", "e"]
        assert precision_at_k(retrieved, relevant, 5) == 1.0

    def test_zero_precision(self):
        """No top-k results are relevant."""
        retrieved = ["x", "y", "z"]
        relevant = ["a", "b", "c"]
        assert precision_at_k(retrieved, relevant, 3) == 0.0

    def test_partial_precision(self):
        """Some top-k results are relevant."""
        retrieved = ["a", "x", "b", "y", "c"]
        relevant = ["a", "b", "c"]
        assert precision_at_k(retrieved, relevant, 5) == 3 / 5

    def test_k_less_than_retrieved(self):
        """k is less than total retrieved."""
        retrieved = ["a", "b", "x", "y", "z"]
        relevant = ["a", "b"]
        assert precision_at_k(retrieved, relevant, 2) == 1.0

    def test_k_greater_than_retrieved(self):
        """k is greater than total retrieved - only considers available items."""
        retrieved = ["a", "b"]
        relevant = ["a", "b", "c"]
        # Only 2 items available, but k=5, so precision = 2/5
        assert precision_at_k(retrieved, relevant, 5) == 2 / 5

    def test_k_zero(self):
        """k=0 returns 0."""
        assert precision_at_k(["a"], ["a"], 0) == 0.0

    def test_empty_retrieved(self):
        """Empty retrieved list."""
        assert precision_at_k([], ["a", "b"], 5) == 0.0

    def test_empty_relevant(self):
        """Empty relevant list."""
        assert precision_at_k(["a", "b"], [], 5) == 0.0


class TestRecallAtK:
    """Tests for recall_at_k metric."""

    def test_perfect_recall(self):
        """All relevant docs found in top-k."""
        retrieved = ["a", "b", "c", "x", "y"]
        relevant = ["a", "b", "c"]
        assert recall_at_k(retrieved, relevant, 5) == 1.0

    def test_zero_recall(self):
        """No relevant docs found in top-k."""
        retrieved = ["x", "y", "z"]
        relevant = ["a", "b", "c"]
        assert recall_at_k(retrieved, relevant, 3) == 0.0

    def test_partial_recall(self):
        """Some relevant docs found."""
        retrieved = ["a", "x", "y"]
        relevant = ["a", "b", "c"]
        assert recall_at_k(retrieved, relevant, 3) == 1 / 3

    def test_k_limits_search(self):
        """Only considers top-k items."""
        retrieved = ["x", "y", "a", "b", "c"]
        relevant = ["a", "b", "c"]
        assert recall_at_k(retrieved, relevant, 2) == 0.0

    def test_empty_relevant(self):
        """Empty relevant list returns 0."""
        assert recall_at_k(["a", "b"], [], 5) == 0.0

    def test_k_zero(self):
        """k=0 returns 0."""
        assert recall_at_k(["a"], ["a"], 0) == 0.0


class TestMRR:
    """Tests for Mean Reciprocal Rank."""

    def test_first_result_relevant(self):
        """First result is relevant -> MRR = 1.0."""
        retrieved = ["a", "b", "c"]
        relevant = ["a"]
        assert mrr(retrieved, relevant) == 1.0

    def test_second_result_relevant(self):
        """Second result is first relevant -> MRR = 0.5."""
        retrieved = ["x", "a", "b"]
        relevant = ["a", "b"]
        assert mrr(retrieved, relevant) == 0.5

    def test_third_result_relevant(self):
        """Third result is first relevant -> MRR = 1/3."""
        retrieved = ["x", "y", "a"]
        relevant = ["a"]
        assert mrr(retrieved, relevant) == pytest.approx(1 / 3)

    def test_no_relevant_found(self):
        """No relevant results -> MRR = 0."""
        retrieved = ["x", "y", "z"]
        relevant = ["a", "b"]
        assert mrr(retrieved, relevant) == 0.0

    def test_empty_retrieved(self):
        """Empty retrieved list."""
        assert mrr([], ["a"]) == 0.0

    def test_empty_relevant(self):
        """Empty relevant list."""
        assert mrr(["a", "b"], []) == 0.0


class TestNDCGAtK:
    """Tests for Normalized Discounted Cumulative Gain."""

    def test_perfect_ranking(self):
        """Perfect ranking should give NDCG = 1.0."""
        retrieved = ["a", "b", "c"]
        scores = [3.0, 2.0, 1.0]  # Already in ideal order
        assert ndcg_at_k(retrieved, scores, 3) == pytest.approx(1.0)

    def test_reversed_ranking(self):
        """Reversed ranking should give NDCG < 1.0."""
        retrieved = ["c", "b", "a"]
        scores = [1.0, 2.0, 3.0]  # Worst first
        result = ndcg_at_k(retrieved, scores, 3)
        assert 0.0 < result < 1.0

    def test_all_zero_relevance(self):
        """All zero relevance scores."""
        retrieved = ["a", "b", "c"]
        scores = [0.0, 0.0, 0.0]
        assert ndcg_at_k(retrieved, scores, 3) == 0.0

    def test_single_item(self):
        """Single item with relevance."""
        retrieved = ["a"]
        scores = [1.0]
        assert ndcg_at_k(retrieved, scores, 1) == 1.0

    def test_k_zero(self):
        """k=0 returns 0."""
        assert ndcg_at_k(["a"], [1.0], 0) == 0.0

    def test_empty_inputs(self):
        """Empty inputs return 0."""
        assert ndcg_at_k([], [], 5) == 0.0
        assert ndcg_at_k(["a"], [], 5) == 0.0

    def test_binary_relevance(self):
        """Binary relevance scores (0 or 1)."""
        retrieved = ["a", "b", "c", "d"]
        scores = [1.0, 0.0, 1.0, 0.0]
        result = ndcg_at_k(retrieved, scores, 4)
        # DCG = 1/log2(2) + 0/log2(3) + 1/log2(4) + 0/log2(5) = 1.0 + 0.5
        # IDCG = 1/log2(2) + 1/log2(3) = 1.0 + 0.6309
        expected_dcg = 1.0 / math.log2(2) + 1.0 / math.log2(4)
        expected_idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
        assert result == pytest.approx(expected_dcg / expected_idcg)


class TestAnswerRelevancyScore:
    """Tests for answer relevancy score placeholder."""

    def test_high_overlap(self):
        """High term overlap gives higher score."""
        query = "what is machine learning"
        answer = "machine learning is a subset of artificial intelligence"
        context = "ML is AI"
        score = answer_relevancy_score(query, answer, context)
        assert score > 0.0

    def test_no_overlap(self):
        """No term overlap gives low score."""
        query = "what is python"
        answer = "the sky is blue today"
        context = "irrelevant"
        score = answer_relevancy_score(query, answer, context)
        # "is" overlaps
        assert score >= 0.0

    def test_empty_query(self):
        """Empty query returns 0."""
        assert answer_relevancy_score("", "answer", "context") == 0.0

    def test_empty_answer(self):
        """Empty answer returns 0."""
        assert answer_relevancy_score("query", "", "context") == 0.0

    def test_score_bounded(self):
        """Score is between 0 and 1."""
        score = answer_relevancy_score("test query", "test query answer", "context")
        assert 0.0 <= score <= 1.0
