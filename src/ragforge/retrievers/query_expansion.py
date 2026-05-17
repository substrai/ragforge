"""Query expansion for improved retrieval recall.

Expands queries using synonyms, acronym expansion, and sub-query generation
to improve recall without requiring external dependencies.
"""

from __future__ import annotations

import re
from typing import Dict, List, Set


# Built-in synonym map for common terms
_SYNONYM_MAP: Dict[str, List[str]] = {
    "fast": ["quick", "rapid", "speedy"],
    "quick": ["fast", "rapid", "speedy"],
    "error": ["bug", "issue", "problem", "fault"],
    "bug": ["error", "issue", "defect"],
    "fix": ["repair", "resolve", "patch"],
    "create": ["make", "build", "generate"],
    "delete": ["remove", "drop", "erase"],
    "update": ["modify", "change", "edit"],
    "get": ["fetch", "retrieve", "obtain"],
    "send": ["transmit", "deliver", "dispatch"],
    "start": ["begin", "launch", "initiate"],
    "stop": ["halt", "end", "terminate"],
    "big": ["large", "huge", "massive"],
    "small": ["tiny", "little", "compact"],
    "show": ["display", "present", "render"],
    "hide": ["conceal", "mask", "obscure"],
    "search": ["find", "lookup", "query"],
    "install": ["setup", "deploy", "configure"],
    "test": ["verify", "validate", "check"],
    "run": ["execute", "launch", "invoke"],
}

# Common tech acronym expansions
_ACRONYM_MAP: Dict[str, str] = {
    "rag": "retrieval augmented generation",
    "llm": "large language model",
    "api": "application programming interface",
    "sdk": "software development kit",
    "cli": "command line interface",
    "db": "database",
    "sql": "structured query language",
    "nosql": "non-relational database",
    "ml": "machine learning",
    "ai": "artificial intelligence",
    "nlp": "natural language processing",
    "aws": "amazon web services",
    "gcp": "google cloud platform",
    "ci": "continuous integration",
    "cd": "continuous deployment",
    "k8s": "kubernetes",
    "vpc": "virtual private cloud",
    "iam": "identity and access management",
    "cdn": "content delivery network",
    "dns": "domain name system",
    "http": "hypertext transfer protocol",
    "rest": "representational state transfer",
    "grpc": "google remote procedure call",
    "jwt": "json web token",
    "oauth": "open authorization",
    "sso": "single sign on",
    "etl": "extract transform load",
    "crud": "create read update delete",
}


class QueryExpander:
    """Expands queries for improved retrieval recall.

    Generates expanded query variants using synonym substitution,
    acronym expansion, and sub-query splitting.
    """

    def __init__(
        self,
        synonym_map: Dict[str, List[str]] | None = None,
        acronym_map: Dict[str, str] | None = None,
    ):
        """Initialize query expander.

        Args:
            synonym_map: Custom synonym map (uses built-in if None).
            acronym_map: Custom acronym map (uses built-in if None).
        """
        self.synonym_map = synonym_map or _SYNONYM_MAP
        self.acronym_map = acronym_map or _ACRONYM_MAP

    def expand_synonyms(self, query: str) -> List[str]:
        """Expand query with synonym substitutions.

        For each word in the query that has synonyms, generates a variant
        with that word replaced by each synonym.

        Args:
            query: The original query string.

        Returns:
            List of expanded query strings (includes original).
        """
        results: List[str] = [query]
        words = query.lower().split()

        for i, word in enumerate(words):
            clean_word = re.sub(r"[^\w]", "", word)
            if clean_word in self.synonym_map:
                for synonym in self.synonym_map[clean_word]:
                    expanded_words = words.copy()
                    expanded_words[i] = synonym
                    expanded = " ".join(expanded_words)
                    if expanded not in results:
                        results.append(expanded)

        return results

    def expand_acronyms(self, query: str) -> List[str]:
        """Expand acronyms in the query.

        Replaces recognized acronyms with their full form and returns
        both the original and expanded versions.

        Args:
            query: The original query string.

        Returns:
            List of expanded query strings (includes original).
        """
        results: List[str] = [query]
        words = query.lower().split()

        for i, word in enumerate(words):
            clean_word = re.sub(r"[^\w]", "", word)
            if clean_word in self.acronym_map:
                expanded_words = words.copy()
                expanded_words[i] = self.acronym_map[clean_word]
                expanded = " ".join(expanded_words)
                if expanded not in results:
                    results.append(expanded)

        return results

    def generate_sub_queries(self, query: str) -> List[str]:
        """Split compound queries on conjunctions.

        Splits queries containing "and" or "or" into sub-queries
        for individual retrieval.

        Args:
            query: The original query string.

        Returns:
            List of sub-query strings (includes original).
        """
        results: List[str] = [query]

        # Split on " and " or " or " (case-insensitive)
        parts = re.split(r"\s+(?:and|or)\s+", query, flags=re.IGNORECASE)

        if len(parts) > 1:
            for part in parts:
                stripped = part.strip()
                if stripped and stripped not in results:
                    results.append(stripped)

        return results

    def expand(self, query: str) -> List[str]:
        """Apply all expansion strategies to a query.

        Combines synonyms, acronyms, and sub-query expansion.

        Args:
            query: The original query string.

        Returns:
            Deduplicated list of all expanded query variants.
        """
        all_expansions: List[str] = [query]
        seen: Set[str] = {query}

        for expanded in self.expand_synonyms(query):
            if expanded not in seen:
                all_expansions.append(expanded)
                seen.add(expanded)

        for expanded in self.expand_acronyms(query):
            if expanded not in seen:
                all_expansions.append(expanded)
                seen.add(expanded)

        for expanded in self.generate_sub_queries(query):
            if expanded not in seen:
                all_expansions.append(expanded)
                seen.add(expanded)

        return all_expansions
