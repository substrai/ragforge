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
    print("Evaluation module not yet implemented.")
    print("Coming soon: retrieval quality metrics (MRR, NDCG, recall@k)")


def cmd_deploy(args: argparse.Namespace) -> None:
    """Deploy the RAG pipeline to AWS."""
    print("Deployment module not yet implemented.")
    print("Coming soon: AWS CDK/SAM deployment for Lambda + OpenSearch Serverless")


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
    eval_parser.add_argument("--config", "-c", type=str, help="Config file path")
    eval_parser.set_defaults(func=cmd_eval)

    # deploy
    deploy_parser = subparsers.add_parser("deploy", help="Deploy to AWS")
    deploy_parser.add_argument("--config", "-c", type=str, help="Config file path")
    deploy_parser.add_argument("--stage", type=str, default="dev", help="Deployment stage")
    deploy_parser.set_defaults(func=cmd_deploy)

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
