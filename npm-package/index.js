/**
 * RAGForge - Config-driven enterprise RAG architecture generator
 *
 * This is the npm placeholder package for RAGForge.
 * The primary implementation is in Python: pip install substrai-ragforge
 *
 * TypeScript SDK coming soon.
 *
 * @see https://github.com/substrai/ragforge
 * @see https://pypi.org/project/substrai-ragforge/
 */

"use strict";

const VERSION = "0.1.0";

/**
 * @typedef {Object} RAGForgeConfig
 * @property {string} projectName - Project name
 * @property {Array<DataSource>} dataSources - Data source configurations
 * @property {ChunkingConfig} chunking - Chunking strategy configuration
 * @property {EmbeddingConfig} embedding - Embedding model configuration
 * @property {StorageConfig} storage - Vector store configuration
 * @property {RetrievalConfig} retrieval - Retrieval strategy configuration
 */

/**
 * @typedef {Object} DataSource
 * @property {string} name - Source name
 * @property {'s3'|'dynamodb'|'local'|'api'} type - Source type
 * @property {Object} config - Source-specific configuration
 */

/**
 * @typedef {Object} ChunkingConfig
 * @property {'auto'|'recursive'|'semantic'|'sentence'|'fixed'} strategy
 * @property {number} maxChunkSize - Maximum chunk size in tokens
 * @property {number} overlap - Overlap between chunks
 */

/**
 * @typedef {Object} QueryResult
 * @property {string} content - Retrieved content
 * @property {number} score - Relevance score
 * @property {string} source - Source document
 * @property {string} chunkId - Chunk identifier
 * @property {Object} metadata - Additional metadata
 */

module.exports = {
  VERSION,
  info: () => ({
    name: "substrai-ragforge",
    version: VERSION,
    description: "Config-driven enterprise RAG architecture generator",
    python_package: "pip install substrai-ragforge",
    repository: "https://github.com/substrai/ragforge",
    documentation: "https://docs.substrai.dev/ragforge",
  }),
};
