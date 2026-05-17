/**
 * RAGForge - Config-driven enterprise RAG architecture generator
 *
 * Primary implementation is in Python: pip install substrai-ragforge
 * TypeScript SDK coming soon.
 */

export interface RAGForgeInfo {
  name: string;
  version: string;
  description: string;
  python_package: string;
  repository: string;
  documentation: string;
}

export const VERSION: string;
export function info(): RAGForgeInfo;
