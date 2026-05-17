"""RAGForge evaluation and quality monitoring.

Provides retrieval metrics, evaluation runners, query analytics,
relevance feedback, and drift detection.
"""

from ragforge.evaluation.analytics import QueryAnalytics
from ragforge.evaluation.drift import DriftDetector, DriftReport
from ragforge.evaluation.feedback import FeedbackCollector
from ragforge.evaluation.metrics import (
    answer_relevancy_score,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from ragforge.evaluation.runner import EvaluationReport, EvaluationRunner

__all__ = [
    "precision_at_k",
    "recall_at_k",
    "mrr",
    "ndcg_at_k",
    "answer_relevancy_score",
    "EvaluationRunner",
    "EvaluationReport",
    "QueryAnalytics",
    "FeedbackCollector",
    "DriftDetector",
    "DriftReport",
]
