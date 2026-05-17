# RAGForge

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/substrai-ragforge.svg)](https://pypi.org/project/substrai-ragforge/)

**Config-driven enterprise RAG architecture generator for serverless infrastructure.**

RAGForge transforms a single YAML configuration into a production-ready Retrieval-Augmented Generation pipeline. Define your data sources, chunking strategy, embedding model, and retrieval method — RAGForge handles the rest.

## Features

- **Config-driven** — Define your entire RAG pipeline in a single `ragforge.yaml`
- **Auto-adaptive chunking** — Automatically selects chunking strategy based on document type (semantic for markdown, recursive for code)
- **Hybrid retrieval** — Combines semantic similarity with BM25-style keyword matching for better recall
- **Pluggable embedders** — Local dev embedder (zero API calls) or AWS Bedrock Titan Embed
- **FAISS vector store** — In-memory similarity search with optional disk persistence, no external services needed
- **Minimal dependencies** — Only `pyyaml` required; AWS, FAISS, and OpenSearch are optional
- **CLI interface** — `ragforge init`, `ingest`, `query`, `eval`, `deploy`, `status`
- **Cost-aware** — Built-in budget tracking and optimization recommendations

## Installation

```bash
pip install substrai-ragforge
```

With optional dependencies:

```bash
# AWS Bedrock embeddings
pip install substrai-ragforge[aws]

# FAISS vector search
pip install substrai-ragforge[faiss]

# All optional dependencies
pip install substrai-ragforge[all]
```

## Quickstart

### 1. Initialize a project

```bash
ragforge init --name my-rag-project
```

This creates a `ragforge.yaml` configuration file.

### 2. Configure your pipeline

```yaml
project:
  name: my-rag-project
  version: "1.0.0"

data_sources:
  - name: documentation
    type: local
    config:
      path: ./docs
    file_types: [md, txt, pdf]

chunking:
  strategy: auto
  max_chunk_size: 512
  overlap: 50

embedding:
  model: bedrock/amazon.titan-embed-text-v2:0
  dimensions: 1024

storage:
  provider: faiss
  index_name: my-project-index

retrieval:
  method: hybrid
  semantic_weight: 0.7
  keyword_weight: 0.3
  top_k: 5
```

### 3. Ingest documents

```bash
ragforge ingest
```

### 4. Query your pipeline

```bash
ragforge query "How do I configure authentication?"
```

## Python API

```python
from ragforge.core.pipeline import RAGPipeline, Document

# From config file
pipeline = RAGPipeline.from_config("ragforge.yaml")

# Ingest documents
docs = [
    Document(content="Your document text...", source="doc.md", doc_type="md")
]
stats = pipeline.ingest(documents=docs)

# Query
results = pipeline.query("What is RAG?", top_k=5)
for result in results:
    print(f"[{result.score:.3f}] {result.content[:100]}")
```

## Architecture

```
ragforge.yaml → Pipeline Orchestrator
                    ├── Chunkers (recursive, semantic, auto-select)
                    ├── Embedders (local dev, Bedrock Titan)
                    ├── Vector Store (FAISS, OpenSearch Serverless)
                    └── Retrievers (hybrid semantic + keyword)
```

## Project Structure

```
src/ragforge/
├── core/           # Config parser, pipeline orchestrator
├── chunkers/       # Text chunking strategies
├── embedders/      # Embedding model integrations
├── storage/        # Vector store backends
├── retrievers/     # Retrieval and ranking strategies
├── ingestion/      # Document loading and preprocessing
├── evaluation/     # Quality metrics (MRR, NDCG, recall@k)
├── cost/           # Cost tracking and optimization
└── cli/            # Command-line interface
```

## Development

```bash
# Clone and install in development mode
git clone https://github.com/substrai/ragforge.git
cd ragforge
pip install -e ".[dev]"

# Run tests
pytest

# Run specific test module
pytest tests/test_chunking.py -v
```

## License

MIT License — see [LICENSE](LICENSE) for details.
