"""Embedding model router for cost optimization.

Routes queries to cheaper or more expensive embedding models based on
query complexity heuristics.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class RoutingResult:
    """Result of model routing decision."""

    model: str
    complexity: float
    reason: str


class EmbeddingModelRouter:
    """Routes queries to appropriate embedding models based on complexity.

    Simple queries are routed to cheaper/lighter models while complex
    queries use the full-capability model.

    Complexity heuristic:
        score = (word_count/20) * 0.3
              + has_question * 0.2
              + special_char_ratio * 0.2
              + (avg_word_length/10) * 0.3
        clamped to [0.0, 1.0]
    """

    def __init__(
        self,
        full_model: str = "bedrock/titan-embed-v2",
        lite_model: str = "local/dev",
        complexity_threshold: float = 0.5,
    ):
        """Initialize model router.

        Args:
            full_model: Model name for complex queries.
            lite_model: Model name for simple queries.
            complexity_threshold: Threshold for routing (0.0-1.0).
                Queries below this go to lite_model, above to full_model.
        """
        self.full_model = full_model
        self.lite_model = lite_model
        self.complexity_threshold = complexity_threshold

    def estimate_complexity(self, query_text: str) -> float:
        """Estimate query complexity on a 0.0-1.0 scale.

        Args:
            query_text: The query text to analyze.

        Returns:
            Complexity score between 0.0 and 1.0.
        """
        if not query_text or not query_text.strip():
            return 0.0

        words = query_text.split()
        word_count = len(words)

        # Factor 1: Word count (longer = more complex)
        word_count_factor = min(word_count / 20.0, 1.0)

        # Factor 2: Is it a question?
        has_question = 1.0 if "?" in query_text else 0.0

        # Factor 3: Special character ratio
        special_chars = sum(1 for c in query_text if not c.isalnum() and not c.isspace())
        total_chars = len(query_text)
        special_char_ratio = special_chars / total_chars if total_chars > 0 else 0.0

        # Factor 4: Average word length (technical terms tend to be longer)
        avg_word_length = (
            sum(len(w) for w in words) / word_count if word_count > 0 else 0.0
        )
        word_length_factor = min(avg_word_length / 10.0, 1.0)

        # Weighted combination
        complexity = (
            word_count_factor * 0.3
            + has_question * 0.2
            + special_char_ratio * 0.2
            + word_length_factor * 0.3
        )

        # Clamp to [0.0, 1.0]
        return max(0.0, min(1.0, complexity))

    def route(self, query_text: str) -> str:
        """Route a query to the appropriate embedding model.

        Args:
            query_text: The query text to route.

        Returns:
            Model name to use for embedding.
        """
        complexity = self.estimate_complexity(query_text)

        if complexity < self.complexity_threshold:
            return self.lite_model
        return self.full_model

    def route_with_details(self, query_text: str) -> RoutingResult:
        """Route a query and return detailed routing information.

        Args:
            query_text: The query text to route.

        Returns:
            RoutingResult with model, complexity score, and reason.
        """
        complexity = self.estimate_complexity(query_text)

        if complexity < self.complexity_threshold:
            return RoutingResult(
                model=self.lite_model,
                complexity=complexity,
                reason=f"Simple query (complexity={complexity:.3f} < threshold={self.complexity_threshold})",
            )
        return RoutingResult(
            model=self.full_model,
            complexity=complexity,
            reason=f"Complex query (complexity={complexity:.3f} >= threshold={self.complexity_threshold})",
        )
