"""Tests for the evaluation runner."""

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from ragforge.evaluation.runner import EvaluationReport, EvaluationRunner, QueryEvalResult


@dataclass
class MockQueryResult:
    """Mock query result for testing."""

    content: str
    score: float
    source: str
    chunk_id: str
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class MockPipeline:
    """Mock pipeline that returns predefined results."""

    def __init__(self, results_map: Dict[str, List[MockQueryResult]]):
        """Initialize with a mapping of query -> results."""
        self.results_map = results_map
        self.queries_received: List[str] = []

    def query(self, query_text: str, top_k: Optional[int] = None) -> List[MockQueryResult]:
        """Return predefined results for the query."""
        self.queries_received.append(query_text)
        return self.results_map.get(query_text, [])


class TestEvaluationRunner:
    """Tests for EvaluationRunner."""

    def _create_golden_dataset(self, entries: List[Dict]) -> Path:
        """Create a temporary golden dataset file."""
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(entries, tmp)
        tmp.close()
        return Path(tmp.name)

    def test_run_perfect_retrieval(self):
        """Test evaluation with perfect retrieval."""
        results_map = {
            "what is RAG": [
                MockQueryResult(content="RAG is...", score=0.9, source="doc1", chunk_id="c1"),
                MockQueryResult(content="RAG stands for...", score=0.8, source="doc2", chunk_id="c2"),
            ]
        }
        pipeline = MockPipeline(results_map)
        runner = EvaluationRunner(pipeline=pipeline, k=2)

        dataset = [{"query": "what is RAG", "relevant_ids": ["c1", "c2"]}]
        golden_path = self._create_golden_dataset(dataset)

        try:
            report = runner.run(golden_path)

            assert report.metrics["avg_precision_at_k"] == 1.0
            assert report.metrics["avg_recall_at_k"] == 1.0
            assert report.metrics["avg_mrr"] == 1.0
            assert report.metrics["num_queries"] == 1
        finally:
            golden_path.unlink()

    def test_run_no_relevant_results(self):
        """Test evaluation when no relevant results are retrieved."""
        results_map = {
            "test query": [
                MockQueryResult(content="irrelevant", score=0.5, source="doc1", chunk_id="x1"),
            ]
        }
        pipeline = MockPipeline(results_map)
        runner = EvaluationRunner(pipeline=pipeline, k=5)

        dataset = [{"query": "test query", "relevant_ids": ["c1", "c2"]}]
        golden_path = self._create_golden_dataset(dataset)

        try:
            report = runner.run(golden_path)

            assert report.metrics["avg_precision_at_k"] == 0.0
            assert report.metrics["avg_recall_at_k"] == 0.0
            assert report.metrics["avg_mrr"] == 0.0
        finally:
            golden_path.unlink()

    def test_run_multiple_queries(self):
        """Test evaluation with multiple queries."""
        results_map = {
            "query1": [
                MockQueryResult(content="a", score=0.9, source="s1", chunk_id="c1"),
            ],
            "query2": [
                MockQueryResult(content="b", score=0.8, source="s2", chunk_id="c3"),
            ],
        }
        pipeline = MockPipeline(results_map)
        runner = EvaluationRunner(pipeline=pipeline, k=5)

        dataset = [
            {"query": "query1", "relevant_ids": ["c1"]},
            {"query": "query2", "relevant_ids": ["c2"]},  # c3 not relevant
        ]
        golden_path = self._create_golden_dataset(dataset)

        try:
            report = runner.run(golden_path)

            assert report.metrics["num_queries"] == 2
            assert len(report.per_query_results) == 2
            # First query: perfect, second: zero
            assert report.per_query_results[0].precision_at_k > 0
            assert report.per_query_results[1].precision_at_k == 0.0
        finally:
            golden_path.unlink()

    def test_run_empty_dataset(self):
        """Test evaluation with empty dataset."""
        pipeline = MockPipeline({})
        runner = EvaluationRunner(pipeline=pipeline, k=5)

        dataset: List[Dict] = []
        golden_path = self._create_golden_dataset(dataset)

        try:
            report = runner.run(golden_path)

            assert report.metrics["num_queries"] == 0
            assert report.metrics["avg_precision_at_k"] == 0.0
        finally:
            golden_path.unlink()

    def test_file_not_found(self):
        """Test that missing golden dataset raises error."""
        pipeline = MockPipeline({})
        runner = EvaluationRunner(pipeline=pipeline, k=5)

        with pytest.raises(FileNotFoundError):
            runner.run("/nonexistent/path.json")

    def test_invalid_json(self):
        """Test that invalid JSON raises error."""
        pipeline = MockPipeline({})
        runner = EvaluationRunner(pipeline=pipeline, k=5)

        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        tmp.write("not valid json{{{")
        tmp.close()

        try:
            with pytest.raises((json.JSONDecodeError, ValueError)):
                runner.run(tmp.name)
        finally:
            Path(tmp.name).unlink()

    def test_report_has_timestamp(self):
        """Test that report includes a timestamp."""
        pipeline = MockPipeline({"q": []})
        runner = EvaluationRunner(pipeline=pipeline, k=5)

        dataset = [{"query": "q", "relevant_ids": []}]
        golden_path = self._create_golden_dataset(dataset)

        try:
            report = runner.run(golden_path)
            assert report.timestamp > 0
        finally:
            golden_path.unlink()

    def test_report_to_dict(self):
        """Test EvaluationReport serialization."""
        report = EvaluationReport(
            metrics={"avg_mrr": 0.5},
            per_query_results=[
                QueryEvalResult(
                    query="test",
                    precision_at_k=0.5,
                    recall_at_k=0.5,
                    mrr=0.5,
                    ndcg_at_k=0.5,
                    retrieved_ids=["c1"],
                    relevant_ids=["c1"],
                )
            ],
            timestamp=1000.0,
            config_used={"k": 5},
        )
        d = report.to_dict()
        assert d["metrics"]["avg_mrr"] == 0.5
        assert len(d["per_query_results"]) == 1
        assert d["timestamp"] == 1000.0

    def test_run_from_dataset(self):
        """Test run_from_dataset with in-memory dataset."""
        results_map = {
            "hello": [
                MockQueryResult(content="hi", score=0.9, source="s", chunk_id="c1"),
            ]
        }
        pipeline = MockPipeline(results_map)
        runner = EvaluationRunner(pipeline=pipeline, k=5)

        dataset = [{"query": "hello", "relevant_ids": ["c1"]}]
        report = runner.run_from_dataset(dataset)

        assert report.metrics["avg_mrr"] == 1.0
