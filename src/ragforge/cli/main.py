"""RAGForge CLI entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def cmd_init(args: argparse.Namespace) -> None:
    """Initialize a new RAGForge project."""
    config_path = Path(args.output or "ragforge.yaml")

    if config_path.exists() and not args.force:
        print(f"Error: {config_path} already exists. Use --force to overwrite.")
        sys.exit(1)

    template = """\
project:
  name: {name}
  version: "1.0.0"

data_sources:
  - name: local-docs
    type: local
    config:
      path: ./data
    file_types: [md, txt, pdf]

chunking:
  strategy: auto
  max_chunk_size: 512
  overlap: 50

embedding:
  model: local/dev
  dimensions: 384
  batch_size: 100

storage:
  provider: faiss
  index_name: {name}-index

retrieval:
  method: hybrid
  semantic_weight: 0.7
  keyword_weight: 0.3
  top_k: 5
""".format(name=args.name or "my-rag-project")

    config_path.write_text(template)
    print(f"Initialized RAGForge project at {config_path}")


def cmd_ingest(args: argparse.Namespace) -> None:
    """Run the ingestion pipeline."""
    from ragforge.core.pipeline import RAGPipeline

    config_path = args.config or "ragforge.yaml"
    pipeline = RAGPipeline.from_config(config_path)
    stats = pipeline.ingest()

    print("Ingestion complete:")
    print(f"  Documents processed: {stats['documents_processed']}")
    print(f"  Chunks created: {stats['chunks_created']}")
    print(f"  Embeddings generated: {stats['embeddings_generated']}")

    if stats["errors"]:
        print(f"  Errors: {len(stats['errors'])}")
        for err in stats["errors"]:
            print(f"    - {err['source']}: {err['error']}")


def cmd_query(args: argparse.Namespace) -> None:
    """Query the RAG pipeline."""
    from ragforge.core.pipeline import RAGPipeline

    config_path = args.config or "ragforge.yaml"
    pipeline = RAGPipeline.from_config(config_path)
    results = pipeline.query(args.text, top_k=args.top_k)

    if not results:
        print("No results found.")
        return

    for i, result in enumerate(results, 1):
        print(f"\n--- Result {i} (score: {result.score:.4f}) ---")
        print(f"Source: {result.source}")
        print(f"Content: {result.content[:200]}...")


def cmd_eval(args: argparse.Namespace) -> None:
    """Run evaluation on the RAG pipeline."""
    from ragforge.core.pipeline import RAGPipeline

    config_path = args.config or "ragforge.yaml"
    golden_path = args.golden

    if not golden_path:
        print("Error: --golden dataset path is required.")
        print("Usage: ragforge eval --golden ./eval/golden_qa.json")
        sys.exit(1)

    golden_file = Path(golden_path)
    if not golden_file.exists():
        print(f"Error: Golden dataset not found: {golden_file}")
        sys.exit(1)

    pipeline = RAGPipeline.from_config(config_path)
    report = pipeline.evaluate(golden_file)

    print("Evaluation Report")
    print("=" * 50)
    print(f"Queries evaluated: {report.metrics.get('num_queries', 0)}")
    print(f"k = {report.metrics.get('k', 5)}")
    print()
    print("Aggregate Metrics:")
    print(f"  Precision@k:  {report.metrics.get('avg_precision_at_k', 0):.4f}")
    print(f"  Recall@k:     {report.metrics.get('avg_recall_at_k', 0):.4f}")
    print(f"  MRR:          {report.metrics.get('avg_mrr', 0):.4f}")
    print(f"  NDCG@k:       {report.metrics.get('avg_ndcg_at_k', 0):.4f}")

    if args.verbose:
        print()
        print("Per-Query Results:")
        print("-" * 50)
        for result in report.per_query_results:
            print(f"  Query: {result.query}")
            print(f"    P@k={result.precision_at_k:.4f}  R@k={result.recall_at_k:.4f}  "
                  f"MRR={result.mrr:.4f}  NDCG={result.ndcg_at_k:.4f}")


def cmd_analytics(args: argparse.Namespace) -> None:
    """Show query analytics summary."""
    from ragforge.evaluation.analytics import QueryAnalytics

    analytics_path = args.path or "ragforge_analytics.json"
    analytics = QueryAnalytics(storage_path=analytics_path)
    summary = analytics.get_summary()

    print("Query Analytics Summary")
    print("=" * 50)
    print(f"Total queries:         {summary['total_queries']}")
    print(f"Avg latency (ms):      {summary['avg_latency_ms']}")
    print(f"Zero-result queries:   {summary['zero_result_count']}")
    print(f"Low-confidence queries: {summary['low_confidence_count']}")

    if summary["queries_per_source"]:
        print()
        print("Queries per source:")
        for source, count in summary["queries_per_source"].items():
            print(f"  {source}: {count}")

    if args.zero_results:
        zero_queries = analytics.get_zero_result_queries()
        if zero_queries:
            print()
            print("Zero-result queries:")
            for q in zero_queries:
                print(f"  - {q}")

    if args.low_confidence:
        threshold = args.threshold or 0.5
        low_conf = analytics.get_low_confidence_queries(threshold=threshold)
        if low_conf:
            print()
            print(f"Low-confidence queries (threshold={threshold}):")
            for item in low_conf:
                print(f"  - {item['query']} (max_score={item['max_score']:.4f})")


def cmd_deploy(args: argparse.Namespace) -> None:
    """Deploy the RAG pipeline to AWS."""
    print("Deployment module not yet implemented.")
    print("Coming soon: AWS CDK/SAM deployment for Lambda + OpenSearch Serverless")


def cmd_cost(args: argparse.Namespace) -> None:
    """Show cost breakdown and monthly forecast."""
    from ragforge.cost.tracker import CostTracker

    cost_path = args.path or "ragforge_costs.json"
    tracker = CostTracker(storage_path=cost_path)

    total = tracker.get_total_cost()
    breakdown = tracker.get_cost_breakdown()
    daily = tracker.get_daily_cost()
    forecast = tracker.get_monthly_forecast()

    print("RAGForge Cost Report")
    print("=" * 50)
    print(f"Total cost:        ${total:.6f}")
    print(f"Today's cost:      ${daily:.6f}")
    print(f"Monthly forecast:  ${forecast:.6f}")
    print()
    print("Breakdown by category:")
    for category, amount in breakdown.items():
        print(f"  {category:12s}: ${amount:.6f}")

    if args.detailed:
        records = tracker.get_records()
        if records:
            print()
            print(f"Recent records ({min(10, len(records))} of {len(records)}):")
            for r in records[-10:]:
                print(f"  [{r.category}] ${r.amount:.6f} - {r.details}")


def cmd_optimize(args: argparse.Namespace) -> None:
    """Show optimization recommendations."""
    from ragforge.cost.tracker import CostTracker
    from ragforge.cost.quantization import EmbeddingQuantizer

    cost_path = args.path or "ragforge_costs.json"
    tracker = CostTracker(storage_path=cost_path)
    quantizer = EmbeddingQuantizer()

    print("RAGForge Optimization Recommendations")
    print("=" * 50)

    breakdown = tracker.get_cost_breakdown()
    total = tracker.get_total_cost()

    recommendations = []

    # Check if embedding costs dominate
    if total > 0 and breakdown.get("embedding", 0) / total > 0.5:
        recommendations.append(
            "Embedding costs are >50% of total. Consider:\n"
            "  - Enable model routing (route simple queries to cheaper models)\n"
            "  - Increase batch sizes to reduce API call overhead"
        )

    # Quantization savings estimate
    dimensions = args.dimensions or 1024
    vectors = args.vectors or 10000
    savings_f16 = quantizer.estimate_savings(vectors, dimensions, "float16")
    savings_i8 = quantizer.estimate_savings(vectors, dimensions, "int8")

    recommendations.append(
        f"Storage quantization savings (for {vectors:,} vectors, {dimensions}d):\n"
        f"  - float16: {savings_f16['savings_percentage']}% savings "
        f"({savings_f16['savings_bytes'] / 1024 / 1024:.1f} MB saved)\n"
        f"  - int8:    {savings_i8['savings_percentage']}% savings "
        f"({savings_i8['savings_bytes'] / 1024 / 1024:.1f} MB saved)"
    )

    # Model routing recommendation
    if not args.routing_enabled:
        recommendations.append(
            "Enable model routing in config to automatically use cheaper\n"
            "  models for simple queries:\n"
            "  cost:\n"
            "    optimization:\n"
            "      model_routing: true\n"
            "      lite_model: local/dev\n"
            "      complexity_threshold: 0.5"
        )

    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec}")

    if not recommendations:
        print("\nNo optimization recommendations at this time.")


def cmd_status(args: argparse.Namespace) -> None:
    """Show pipeline status."""
    from ragforge.core.config import load_config

    config_path = args.config or "ragforge.yaml"

    try:
        config = load_config(config_path)
    except FileNotFoundError:
        print(f"No configuration found at {config_path}")
        print("Run 'ragforge init' to create a new project.")
        sys.exit(1)

    print(f"Project: {config.project_name} v{config.project_version}")
    print(f"Data sources: {len(config.data_sources)}")
    for ds in config.data_sources:
        print(f"  - {ds.name} ({ds.type})")
    print(f"Chunking: {config.chunking.strategy} (max={config.chunking.max_chunk_size})")
    print(f"Embedding: {config.embedding.model} ({config.embedding.dimensions}d)")
    print(f"Storage: {config.storage.provider}")
    print(f"Retrieval: {config.retrieval.method} (top_k={config.retrieval.top_k})")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="ragforge",
        description="RAGForge - Config-driven enterprise RAG architecture generator",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # init
    init_parser = subparsers.add_parser("init", help="Initialize a new RAGForge project")
    init_parser.add_argument("--name", type=str, help="Project name")
    init_parser.add_argument("--output", "-o", type=str, help="Output config file path")
    init_parser.add_argument("--force", action="store_true", help="Overwrite existing config")
    init_parser.set_defaults(func=cmd_init)

    # ingest
    ingest_parser = subparsers.add_parser("ingest", help="Run the ingestion pipeline")
    ingest_parser.add_argument("--config", "-c", type=str, help="Config file path")
    ingest_parser.set_defaults(func=cmd_ingest)

    # query
    query_parser = subparsers.add_parser("query", help="Query the RAG pipeline")
    query_parser.add_argument("text", type=str, help="Query text")
    query_parser.add_argument("--config", "-c", type=str, help="Config file path")
    query_parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    query_parser.set_defaults(func=cmd_query)

    # eval
    eval_parser = subparsers.add_parser("eval", help="Run evaluation metrics")
    eval_parser.add_argument("--golden", "-g", type=str, help="Path to golden dataset JSON")
    eval_parser.add_argument("--config", "-c", type=str, help="Config file path")
    eval_parser.add_argument("--verbose", "-v", action="store_true", help="Show per-query results")
    eval_parser.set_defaults(func=cmd_eval)

    # analytics
    analytics_parser = subparsers.add_parser("analytics", help="Show query analytics")
    analytics_parser.add_argument("--path", "-p", type=str, help="Analytics file path")
    analytics_parser.add_argument("--zero-results", action="store_true", help="Show zero-result queries")
    analytics_parser.add_argument("--low-confidence", action="store_true", help="Show low-confidence queries")
    analytics_parser.add_argument("--threshold", type=float, help="Confidence threshold")
    analytics_parser.set_defaults(func=cmd_analytics)

    # deploy
    deploy_parser = subparsers.add_parser("deploy", help="Deploy to AWS")
    deploy_parser.add_argument("--config", "-c", type=str, help="Config file path")
    deploy_parser.add_argument("--stage", type=str, default="dev", help="Deployment stage")
    deploy_parser.set_defaults(func=cmd_deploy)

    # cost
    cost_parser = subparsers.add_parser("cost", help="Show cost breakdown and forecast")
    cost_parser.add_argument("--path", "-p", type=str, help="Cost data file path")
    cost_parser.add_argument("--detailed", "-d", action="store_true", help="Show detailed records")
    cost_parser.set_defaults(func=cmd_cost)

    # optimize
    optimize_parser = subparsers.add_parser("optimize", help="Show optimization recommendations")
    optimize_parser.add_argument("--path", "-p", type=str, help="Cost data file path")
    optimize_parser.add_argument("--dimensions", type=int, help="Embedding dimensions")
    optimize_parser.add_argument("--vectors", type=int, help="Number of vectors")
    optimize_parser.add_argument("--routing-enabled", action="store_true", help="Model routing is enabled")
    optimize_parser.set_defaults(func=cmd_optimize)

    # status
    status_parser = subparsers.add_parser("status", help="Show pipeline status")
    status_parser.add_argument("--config", "-c", type=str, help="Config file path")
    status_parser.set_defaults(func=cmd_status)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
