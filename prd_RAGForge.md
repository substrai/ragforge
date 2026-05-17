# RAGForge - Product Requirements Document (PRD)

> **The first config-driven enterprise RAG architecture generator** — describe your data sources, query patterns, and accuracy requirements in a YAML file, and RAGForge generates the complete retrieval-augmented generation infrastructure: chunking pipelines, embedding workflows, vector store provisioning, retrieval optimization, and continuous quality monitoring.

---

## 1. Problem Statement

Every RAG implementation is built from scratch. Teams spend weeks making the same decisions and writing the same boilerplate:

| Current Approach | Problem |
|---|---|
| LlamaIndex / LangChain RAG | Library, not infrastructure — no deployment, no scaling, no monitoring |
| AWS Bedrock Knowledge Bases | Rigid, limited chunking strategies, no quality feedback loop |
| Custom Lambda pipelines | Rebuilt from scratch every time, no reusability across projects |
| Vector DB vendor tutorials | Vendor-locked, no multi-source federation, no cost optimization |
| Manual chunking + embedding | No adaptive strategies, no evaluation, no drift detection |
| No RAG evaluation at all | Retrieval quality degrades silently, no evidence for stakeholders |

**The gap:** Tools like LlamaIndex exist as *libraries* for building RAG. AWS Bedrock Knowledge Bases is a managed service with limited control. Nobody has built the *infrastructure framework* that auto-generates and deploys the entire RAG pipeline as serverless infrastructure based on your data profile and query patterns.

### Who Needs This

- **ML Engineers:** Production RAG without rebuilding ingestion/retrieval infrastructure from scratch
- **Platform Teams:** Standardized RAG architecture across all GenAI applications in the org
- **Data Engineers:** Automated chunking and embedding pipelines with quality monitoring
- **Enterprise Teams:** Multi-source RAG with compliance, cost controls, and audit trails
- **Consultants:** Deploy client RAG solutions quickly with production-grade infrastructure

---

## 2. Vision and Positioning

> **RAGForge is to RAG what create-react-app is to React projects** — describe WHAT you want to retrieve and from WHERE, and the framework generates HOW to chunk, embed, store, retrieve, and monitor it, then deploys the entire pipeline as serverless infrastructure.

### The Key Insight

Every RAG pipeline follows the same pattern:

```
Data Sources → Ingestion → Chunking → Embedding → Storage → Retrieval → Augmentation → Generation → Evaluation
```

RAGForge automates this entire pattern. You describe your data and requirements; it generates everything else.

### Design Principles (Applied)

| Principle | How RAGForge Implements It |
|---|---|
| Convention Over Configuration | Describe data sources → framework selects optimal chunking, embedding, and retrieval strategies |
| Inversion of Control | Framework owns the pipeline lifecycle; user provides data config only |
| Plugin Architecture | Custom chunkers, embedders, retrievers, and rankers as plugins |
| Declarative Over Imperative | YAML defines what to retrieve; framework handles execution |
| Observable by Default | Every retrieval logged with relevance scores, latency, cost, and quality metrics |
| Infrastructure as Byproduct | `ragforge deploy` provisions Lambda + OpenSearch/FAISS + S3 + Step Functions |
| Escape Hatches | Export raw SAM/CDK; use individual components as standalone library |

---

## 3. Core Architecture

```
+------------------------------------------------------------------+
|                        RAGForge Framework                          |
+------------------------------------------------------------------+
|                                                                    |
|  +------------------+  +------------------+  +-----------------+  |
|  |   CLI Tool       |  |  Pipeline Engine |  |  Infra Gen      |  |
|  |  (init/deploy/   |  |  (ingestion,     |  |  (CDK/SAM,      |  |
|  |   ingest/query)  |  |   retrieval,     |  |   Lambda,       |  |
|  |                  |  |   evaluation)    |  |   OpenSearch)   |  |
|  +--------+---------+  +--------+---------+  +--------+--------+  |
|           |                      |                     |           |
+-----------+----------------------+---------------------+-----------+
|                      PLUGIN LAYER                                  |
|  +---------------------------------------------------------------+|
|  | Chunkers | Embedders | Retrievers | Rankers | Evaluators      ||
|  +---------------------------------------------------------------+|
|                                                                    |
+--------------------------------------------------------------------+
|                     AWS SERVICES LAYER                              |
|  +--------+ +----------+ +--------+ +-----+ +----------+ +-----+ |
|  | Lambda | | Step Fn  | |OpenSrch| |  S3 | | Bedrock  | | CW  | |
|  +--------+ +----------+ +--------+ +-----+ +----------+ +-----+ |
|  +--------+ +----------+ +--------+                               |
|  |DynamoDB| |EventBrdg | |  SQS   |                               |
|  +--------+ +----------+ +--------+                               |
+--------------------------------------------------------------------+
```

### Pipeline Execution Model

```
[Data Source Config]
       |
       v
[Source Analysis] ──→ [Strategy Selection] ──→ [Pipeline Assembly]
       |                        |                         |
       v                        v                         v
[Profile data:           [Auto-select:            [Step Functions:
 doc types, sizes,        chunking strategy,       scheduled ingestion,
 update frequency]        embedding model,         parallel processing,
                          retrieval method]         checkpoints]
       |                        |                         |
       +────────────────────────+─────────────────────────+
                                |
                                v
                    [Deploy RAG Infrastructure]
                                |
                    ┌───────────┼───────────┐
                    v           v           v
              [Ingestion   [Retrieval  [Quality
               Pipeline]    API]        Monitor]
                    |           |           |
                    v           v           v
              [S3 → Chunk  [Query →    [Evaluate
               → Embed →    Retrieve    retrieval
               Store]       → Rank]     quality]
```

---

## 4. Feature Breakdown by Phase

### Phase 1: Core Engine (MVP - Weeks 1-4)

**Goal:** A developer can generate a complete RAG pipeline for their data sources in under 15 minutes.

| Feature | Description | Priority |
|---|---|---|
| Data source config (YAML) | Declarative description of data sources and query requirements | P0 |
| Chunking strategy auto-selection | Framework picks optimal chunking based on document type | P0 |
| Built-in chunkers | Recursive, semantic, sentence-based, fixed-size, document-aware | P0 |
| Embedding model abstraction | Unified interface for Bedrock Titan, Cohere, custom models | P0 |
| Vector store provisioning | Auto-provision OpenSearch Serverless or FAISS-on-Lambda | P0 |
| Ingestion pipeline | S3 trigger → chunk → embed → store (Step Functions) | P0 |
| Retrieval API | Lambda-backed query endpoint with configurable retrieval | P0 |
| CLI: init | `ragforge init` scaffolds project with data-source-specific config | P0 |
| CLI: deploy | `ragforge deploy` deploys entire RAG infrastructure | P0 |

**MVP Config:**

```yaml
# ragforge.yaml
project:
  name: "product-docs-rag"
  version: "1.0.0"

data_sources:
  - name: product-documentation
    type: s3
    bucket: "company-docs"
    prefix: "product/"
    file_types: [pdf, md, html, docx]
    update_frequency: daily

  - name: support-tickets
    type: dynamodb
    table: "support-tickets"
    fields: [title, description, resolution]
    update_frequency: hourly

chunking:
  strategy: auto  # auto | recursive | semantic | sentence | fixed
  # Auto-selection based on document type:
  # PDF → semantic chunking (respects sections/headers)
  # Markdown → header-based recursive
  # Support tickets → per-record (no chunking needed)
  max_chunk_size: 512
  overlap: 50

embedding:
  model: bedrock/titan-embed-v2
  dimensions: 1024
  batch_size: 100

storage:
  provider: opensearch-serverless  # opensearch-serverless | faiss-lambda | pgvector
  index_name: "product-docs"

retrieval:
  method: hybrid  # semantic | keyword | hybrid
  top_k: 5
  reranking:
    enabled: true
    model: bedrock/cohere-rerank

query:
  endpoint: true  # deploy as API Gateway + Lambda
  timeout_ms: 3000
  cache:
    enabled: true
    ttl_seconds: 300
```

**MVP Code Usage:**

```python
from ragforge import RAGPipeline

# Load config and deploy
pipeline = RAGPipeline.from_config("ragforge.yaml")

# Ingest data
pipeline.ingest()  # Runs full ingestion pipeline

# Query
results = pipeline.query("What is the return policy for electronics?")
print(results)
# [
#   {"content": "...", "score": 0.94, "source": "product-docs/returns.pdf", "chunk_id": "..."},
#   {"content": "...", "score": 0.87, "source": "product-docs/faq.md", "chunk_id": "..."},
# ]
```

---

### Phase 2: Adaptive Chunking & Multi-Source Federation (Weeks 5-8)

**Goal:** Intelligent chunking that adapts to document structure and federated retrieval across multiple sources.

| Feature | Description | Priority |
|---|---|---|
| Document-aware chunking | Different strategies per document type (PDF sections, code blocks, tables) | P0 |
| Semantic chunking | Split on topic boundaries using embedding similarity | P0 |
| Hierarchical chunking | Parent-child chunks for context preservation | P0 |
| Multi-source federation | Query across multiple knowledge bases with relevance-weighted merging | P0 |
| Source-specific retrieval | Different retrieval strategies per data source | P1 |
| Chunk metadata enrichment | Auto-extract titles, dates, authors, categories from chunks | P1 |
| Incremental ingestion | Only re-process changed/new documents | P1 |
| Deduplication | Detect and handle duplicate content across sources | P2 |

**Adaptive Chunking Config:**

```yaml
chunking:
  strategy: adaptive  # Uses document-type-specific strategies

  strategies:
    pdf:
      method: semantic
      respect_headers: true
      respect_pages: false
      max_chunk_size: 512
      min_chunk_size: 100

    markdown:
      method: header_recursive
      split_on: [h1, h2, h3]
      include_parent_context: true
      max_chunk_size: 800

    code:
      method: function_aware
      languages: [python, typescript]
      include_docstrings: true
      max_chunk_size: 1000

    table:
      method: row_based
      include_headers: true
      max_rows_per_chunk: 20

    support_ticket:
      method: per_record
      fields_to_chunk: [title, description, resolution]
      concatenate: true

  fallback:
    method: recursive
    max_chunk_size: 512
    overlap: 50
```

**Multi-Source Federation:**

```yaml
federation:
  strategy: relevance_weighted  # relevance_weighted | round_robin | source_priority

  sources:
    - name: product-docs
      weight: 1.0
      retrieval_method: hybrid

    - name: support-tickets
      weight: 0.8
      retrieval_method: semantic

    - name: api-docs
      weight: 0.6
      retrieval_method: keyword

  merging:
    method: reciprocal_rank_fusion  # rrf | linear_combination | max_score
    top_k: 10
    deduplicate: true
    diversity_threshold: 0.3  # ensure result diversity
```

---

### Phase 3: Retrieval Optimization & Quality Monitoring (Weeks 9-12)

**Goal:** Continuously optimize retrieval quality and detect degradation.

| Feature | Description | Priority |
|---|---|---|
| Retrieval evaluation | Automated quality scoring (precision, recall, MRR, NDCG) | P0 |
| Query analytics | Track query patterns, zero-result queries, low-confidence retrievals | P0 |
| Relevance feedback loop | User feedback improves retrieval ranking over time | P0 |
| A/B testing retrieval | Compare retrieval strategies with traffic splitting | P1 |
| Embedding drift detection | Detect when embeddings become stale vs new content | P1 |
| Auto-reindexing | Trigger re-embedding when quality metrics drop | P1 |
| Query expansion | Automatic query reformulation for better recall | P1 |
| Caching layer | Intelligent caching of frequent queries with invalidation | P2 |

**Quality Monitoring Config:**

```yaml
quality:
  evaluation:
    enabled: true
    frequency: daily
    metrics:
      - precision_at_k  # k=5
      - recall_at_k
      - mrr  # Mean Reciprocal Rank
      - ndcg  # Normalized Discounted Cumulative Gain
      - answer_relevancy  # end-to-end with LLM judge

    thresholds:
      precision_at_5: 0.75
      recall_at_5: 0.80
      mrr: 0.70
      answer_relevancy: 0.85

    golden_dataset: "./eval/golden_qa.json"

  drift_detection:
    enabled: true
    baseline_window: 7d
    alert_on_drop: 0.10  # 10% quality drop triggers alert

  feedback:
    enabled: true
    collection_method: implicit  # implicit (click tracking) | explicit (thumbs up/down)
    apply_after: 100  # apply feedback after 100 signals

  alerts:
    - channel: sns
      topic_arn: "arn:aws:sns:us-east-1:123:rag-quality-alerts"
      on: [quality_drop, zero_results_spike, high_latency]
```

---

### Phase 4: Cost Optimization & Advanced Retrieval (Weeks 13-16)

**Goal:** Minimize RAG infrastructure costs while maintaining quality.

| Feature | Description | Priority |
|---|---|---|
| Tiered storage | Hot/warm/cold embedding tiers based on access frequency | P0 |
| Embedding model routing | Use cheaper models for simple queries, expensive for complex | P0 |
| Chunk deduplication | Detect and deduplicate similar chunks to reduce storage | P1 |
| Lazy embedding | Only embed documents when first queried (on-demand) | P1 |
| Compression | Quantize embeddings (float32 → int8) for storage savings | P1 |
| Multi-vector retrieval | ColBERT-style late interaction for better quality at lower cost | P2 |
| Hypothetical Document Embedding (HyDE) | Generate hypothetical answers for better retrieval | P2 |
| Contextual compression | Compress retrieved chunks before passing to LLM | P1 |

**Cost Optimization Config:**

```yaml
cost:
  budget:
    monthly_embedding: 50.00  # USD for embedding API calls
    monthly_storage: 20.00    # USD for vector storage
    monthly_retrieval: 30.00  # USD for query processing

  optimization:
    embedding_quantization: int8  # float32 | float16 | int8
    tiered_storage:
      hot: 30d    # full-precision, fast retrieval
      warm: 90d   # quantized, slightly slower
      cold: 365d  # archived, on-demand re-embedding

    model_routing:
      simple_queries: bedrock/titan-embed-lite  # cheaper
      complex_queries: bedrock/titan-embed-v2   # better quality
      complexity_threshold: 0.6

    caching:
      enabled: true
      strategy: lru
      max_entries: 10000
      ttl_seconds: 3600

  alerts:
    on_budget_80_percent: notify
    on_budget_exceeded: downgrade_model
```

---

### Phase 5: Deployment & Enterprise Features (Weeks 17-20)

| Feature | Description | Priority |
|---|---|---|
| One-command deploy | `ragforge deploy` provisions all infrastructure | P0 |
| Multi-environment | dev/staging/prod with environment-specific configs | P0 |
| Access control | Per-source, per-tenant document access policies | P1 |
| Audit trail | Log every retrieval with user, query, results, and relevance | P1 |
| Scheduled ingestion | EventBridge-triggered periodic re-ingestion | P0 |
| Webhook notifications | Notify on ingestion complete, quality alerts, errors | P1 |
| CI/CD integration | GitHub Actions for deploy-on-push with quality gates | P0 |
| Eject | Export raw SAM/CDK templates | P1 |

---

## 5. Chunking Strategies & Auto-Selection

RAGForge auto-selects chunking strategy based on document type and content analysis:

| Document Type | Default Strategy | Chunk Size | Rationale |
|---|---|---|---|
| **PDF (structured)** | Semantic (section-aware) | 512 tokens | Respects document structure |
| **PDF (unstructured)** | Recursive with overlap | 512 tokens | Handles flowing text |
| **Markdown** | Header-recursive | 800 tokens | Natural section boundaries |
| **HTML** | DOM-aware semantic | 512 tokens | Respects HTML structure |
| **Code files** | Function/class-aware | 1000 tokens | Preserves logical units |
| **CSV/Tables** | Row-based with headers | 20 rows | Maintains tabular context |
| **JSON/YAML** | Object-level | 1 object | Preserves data integrity |
| **Support tickets** | Per-record | Full record | Each ticket is a unit |
| **Chat transcripts** | Conversation-turn | 5 turns | Preserves dialogue context |
| **Legal documents** | Clause-based | 300 tokens | Respects legal structure |

---

## 6. Configuration Schema (ragforge.yaml)

```yaml
project:
  name: "my-rag-system"
  version: "1.0.0"
  organization: substrai

data_sources:
  - name: primary-docs
    type: s3  # s3 | dynamodb | api | confluence | notion | github
    config:
      bucket: "my-docs-bucket"
      prefix: "knowledge-base/"
      file_types: [pdf, md, html, docx, txt]
    update_frequency: daily
    access_control:
      public: false
      allowed_tenants: [team-a, team-b]

chunking:
  strategy: auto
  max_chunk_size: 512
  overlap: 50
  metadata_extraction:
    title: true
    date: true
    author: true
    category: true

embedding:
  model: bedrock/titan-embed-v2
  dimensions: 1024
  batch_size: 100
  quantization: float16

storage:
  provider: opensearch-serverless
  index_name: "my-rag-index"
  replicas: 1

retrieval:
  method: hybrid
  semantic_weight: 0.7
  keyword_weight: 0.3
  top_k: 5
  reranking:
    enabled: true
    model: bedrock/cohere-rerank
    top_n: 3

query:
  endpoint: true
  timeout_ms: 3000
  cache:
    enabled: true
    ttl_seconds: 300

quality:
  evaluation:
    enabled: true
    frequency: daily
    golden_dataset: "./eval/golden_qa.json"
  drift_detection:
    enabled: true
    baseline_window: 7d

cost:
  budget:
    monthly_total: 100.00
  optimization:
    embedding_quantization: float16
    caching: true

deployment:
  runtime: lambda
  memory_mb: 512
  timeout: 30

environments:
  dev:
    storage.provider: faiss-lambda  # cheaper for dev
    retrieval.top_k: 3
  staging:
    storage.provider: opensearch-serverless
  prod:
    storage.provider: opensearch-serverless
    quality.evaluation.frequency: hourly
```

---

## 7. Project Structure (Generated by `ragforge init`)

```
my-rag-system/
├── ragforge.yaml              # Pipeline configuration
├── sources/
│   ├── __init__.py
│   ├── s3_connector.py        # S3 data source connector
│   └── custom_connector.py    # Custom data source (optional)
├── chunkers/
│   ├── __init__.py
│   └── custom_chunker.py     # Custom chunking logic (optional)
├── retrievers/
│   ├── __init__.py
│   └── custom_retriever.py   # Custom retrieval logic (optional)
├── eval/
│   ├── golden_qa.json         # Golden Q&A dataset for evaluation
│   └── test_queries.json      # Test queries for quality checks
├── infrastructure/            # Auto-generated (or ejected)
│   └── template.yaml
├── tests/
│   ├── test_chunking.py
│   ├── test_retrieval.py
│   └── test_ingestion.py
└── README.md
```

---

## 8. CLI Commands

```bash
# Project scaffolding
ragforge init [project-name]
ragforge init --source s3              # Start with S3 source template
ragforge init --source confluence      # Start with Confluence template

# Data management
ragforge ingest                         # Run full ingestion pipeline
ragforge ingest --source primary-docs   # Ingest specific source only
ragforge ingest --incremental           # Only process new/changed docs
ragforge status                         # Show ingestion status and stats

# Querying
ragforge query "What is the return policy?"  # Test query locally
ragforge query --explain                     # Show retrieval reasoning

# Evaluation
ragforge eval                            # Run quality evaluation
ragforge eval --golden                   # Evaluate against golden dataset
ragforge eval --report                   # Generate quality report

# Deployment
ragforge deploy --env dev
ragforge deploy --env prod --approve
ragforge logs --tail                     # Tail ingestion/query logs

# Optimization
ragforge optimize                        # Analyze and suggest optimizations
ragforge optimize --apply                # Apply recommended optimizations
ragforge cost                            # Show cost breakdown
ragforge cost --forecast                 # Predict monthly spend

# Maintenance
ragforge reindex                         # Force full re-embedding
ragforge eject                           # Export raw infrastructure
ragforge upgrade                         # Upgrade framework version
```

---

## 9. Integration with Substrai Ecosystem

RAGForge integrates with LambdaLLM, GuardrailGraph, EvalForge, and CostSentinel:

```yaml
# ragforge.yaml
integrations:
  lambdallm:
    enabled: true
    # RAGForge provides retrieval context to LambdaLLM handlers
    context_injection: automatic

  guardrailgraph:
    enabled: true
    # Apply guardrails to retrieved content before passing to LLM
    pipeline: content-safety
    check_retrieved_content: true

  evalforge:
    enabled: true
    # Use EvalForge for end-to-end RAG quality evaluation
    evaluate_on_deploy: true

  costsentinel:
    enabled: true
    # Track embedding and retrieval costs
    budget_enforcement: true
```

```python
# Programmatic integration with LambdaLLM
from ragforge import RAGPipeline
from lambdallm import handler, Model

rag = RAGPipeline.from_config("ragforge.yaml")

@handler(model=Model.CLAUDE_3_SONNET)
def answer_question(event, context):
    query = event["body"]["question"]

    # RAGForge handles retrieval
    retrieved = rag.query(query, top_k=5)

    # LambdaLLM handles generation with retrieved context
    response = context.invoke(
        prompt=f"Answer based on this context:\n{retrieved}\n\nQuestion: {query}",
    )
    return {"statusCode": 200, "body": response}
```

---

## 10. Technical Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Language | Python (primary) | ML/NLP ecosystem, Lambda runtime |
| Ingestion orchestration | Step Functions | Serverless, visual, retry-native, handles long pipelines |
| Embedding compute | Lambda (small batches), Batch (large) | Cost-optimized per workload size |
| Default embedding model | Bedrock Titan Embed v2 | Native AWS, no API key, good quality |
| Vector store (prod) | OpenSearch Serverless | Managed, scalable, hybrid search support |
| Vector store (dev) | FAISS on Lambda | Zero cost, fast iteration |
| Document storage | S3 | Cheap, durable, event-driven triggers |
| Metadata store | DynamoDB | Fast lookups, TTL, serverless |
| Scheduling | EventBridge | Serverless cron, no infrastructure |
| Query API | API Gateway + Lambda | Serverless, auto-scaling |
| Package size | <8MB core | Lambda cold start optimization |
| Config format | YAML (ragforge.yaml) | Consistent with Substrai ecosystem |

---

## 11. Differentiation from Existing Tools

| Capability | LlamaIndex | LangChain RAG | Bedrock KB | Pinecone Canopy | RAGForge |
|---|---|---|---|---|---|
| Auto chunking selection | No | No | Basic | No | **Yes (document-aware)** |
| Infrastructure generation | No | No | Managed | No | **Step Functions + Lambda** |
| Multi-source federation | Basic | Basic | No | No | **Weighted + RRF merging** |
| Quality monitoring | No | No | No | No | **Continuous + drift detection** |
| Cost optimization | No | No | No | No | **Tiered storage + model routing** |
| Adaptive strategies | No | No | No | No | **Auto-selects per doc type** |
| Serverless-native | No | No | Yes | No | **Lambda-optimized** |
| One-command deploy | No | No | Console | No | **CLI deploy** |
| Incremental ingestion | Manual | Manual | Yes | Manual | **Built-in** |
| Retrieval A/B testing | No | No | No | No | **Built-in** |
| Open source | Yes | Yes | No | Partial | **Yes (MIT)** |
| Provider agnostic | Yes | Yes | No | No | **Yes (Bedrock default)** |

---

## 12. Success Metrics

| Metric | Target (6 months) | Target (12 months) |
|---|---|---|
| GitHub stars | 300+ | 1,500+ |
| PyPI weekly downloads | 500+ | 5,000+ |
| Enterprise adopters | 3+ | 15+ |
| Data source connectors | 5 | 12+ |
| Chunking strategies | 8 | 15+ |
| Conference talks | 2+ | 5+ |
| Time to first RAG deploy | <15 minutes | <5 minutes |
| Retrieval quality (MRR) | 0.75+ | 0.85+ |

---

## 13. EB1A Evidence This Generates

- **Original contribution of major significance:** First config-driven RAG architecture generator — establishes a new category of tooling that bridges the gap between RAG libraries and production infrastructure
- **Published material:** "Config-Driven RAG: From Data Description to Production Pipeline in Minutes" — arXiv, MLOps community talks
- **Judging:** Invited to review RAG architecture standards, vector search benchmarks
- **Leading role:** Organizations depend on framework for production knowledge retrieval systems
- **High remuneration:** RAG architecture consulting commands premium ($250-400/hr)

### The EB1A Narrative

> "I developed RAGForge, the first framework that auto-generates complete retrieval-augmented generation infrastructure from data source descriptions. By treating RAG as declarative infrastructure — analogous to how Terraform treats cloud resources — RAGForge reduced enterprise knowledge retrieval setup time from weeks to minutes. The framework has been adopted by X organizations, enabling production RAG systems that serve Y queries daily with Z% retrieval accuracy."

---

## 14. Go-to-Market Timeline

| Month | Action | Milestone |
|---|---|---|
| Month 1-2 | Build core engine + chunking + embedding + retrieval | MVP: describe data, deploy RAG |
| Month 2-3 | Adaptive chunking + multi-source federation | Enterprise data complexity |
| Month 3-4 | Quality monitoring + retrieval optimization | Continuous improvement |
| Month 4-5 | Cost optimization + tiered storage | Enterprise cost control |
| Month 5-6 | Blog: "Why Every RAG is Built from Scratch (And How to Fix It)" | Published material |
| Month 6-7 | arXiv paper on config-driven RAG infrastructure | Academic credibility |
| Month 7-8 | Conference talk: "From Data Sources to Production RAG in 15 Minutes" | Speaking evidence |
| Month 8-9 | Enterprise pilot (2-3 organizations) | Adoption evidence |

---

## 15. Getting Started (Day 1 Action Plan)

```bash
# 1. Create the repository
gh repo create substrai/ragforge --public \
  --description "Config-driven enterprise RAG architecture generator"

# 2. Set up Python project
mkdir -p src/ragforge/{core,chunkers,embedders,retrievers,storage,ingestion,evaluation,cost,cli}
touch src/ragforge/__init__.py

# 3. Start with these files (the heart of the framework):
# src/ragforge/core/config.py           — YAML config parser and validator
# src/ragforge/core/pipeline.py         — RAG pipeline orchestrator
# src/ragforge/chunkers/registry.py     — Chunking strategy auto-selection
# src/ragforge/chunkers/recursive.py    — Recursive text chunker
# src/ragforge/chunkers/semantic.py     — Semantic boundary chunker
# src/ragforge/embedders/bedrock.py     — Bedrock Titan embedding
# src/ragforge/storage/opensearch.py    — OpenSearch Serverless adapter
# src/ragforge/storage/faiss_lambda.py  — FAISS-on-Lambda adapter
# src/ragforge/retrievers/hybrid.py     — Hybrid retrieval (semantic + keyword)
# src/ragforge/ingestion/pipeline.py    — Step Functions ingestion pipeline
# src/ragforge/evaluation/metrics.py    — Retrieval quality metrics
# src/ragforge/cli/main.py              — CLI entry point

# 4. Write first test
mkdir tests && touch tests/test_chunking.py
```

**First file to write:** `src/ragforge/core/config.py` — the YAML config parser that reads data source descriptions and determines which chunking, embedding, and retrieval strategies to use.

---

*This PRD is a living document. Update it as you build, learn, and get community feedback.*
