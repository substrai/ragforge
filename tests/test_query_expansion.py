"""Tests for query expansion."""

import pytest

from ragforge.retrievers.query_expansion import QueryExpander


class TestQueryExpander:
    """Tests for QueryExpander class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.expander = QueryExpander()

    def test_expand_synonyms_basic(self):
        """Test basic synonym expansion."""
        results = self.expander.expand_synonyms("fix the error")
        assert "fix the error" in results
        # Should have expansions with synonyms of "fix" and "error"
        assert len(results) > 1
        # Check some expected expansions
        assert any("repair" in r for r in results)
        assert any("bug" in r or "issue" in r or "problem" in r for r in results)

    def test_expand_synonyms_no_match(self):
        """Test synonym expansion with no matching words."""
        results = self.expander.expand_synonyms("quantum entanglement")
        assert results == ["quantum entanglement"]

    def test_expand_synonyms_preserves_original(self):
        """Test that original query is always first."""
        results = self.expander.expand_synonyms("create a test")
        assert results[0] == "create a test"

    def test_expand_acronyms_basic(self):
        """Test basic acronym expansion."""
        results = self.expander.expand_acronyms("what is rag")
        assert "what is rag" in results
        assert any("retrieval augmented generation" in r for r in results)

    def test_expand_acronyms_multiple(self):
        """Test expanding multiple acronyms."""
        results = self.expander.expand_acronyms("llm api")
        assert "llm api" in results
        assert any("large language model" in r for r in results)
        assert any("application programming interface" in r for r in results)

    def test_expand_acronyms_no_match(self):
        """Test acronym expansion with no matching acronyms."""
        results = self.expander.expand_acronyms("hello world")
        assert results == ["hello world"]

    def test_expand_acronyms_preserves_original(self):
        """Test that original query is always first."""
        results = self.expander.expand_acronyms("use the api")
        assert results[0] == "use the api"

    def test_generate_sub_queries_and(self):
        """Test splitting on 'and'."""
        results = self.expander.generate_sub_queries("python and javascript")
        assert "python and javascript" in results
        assert "python" in results
        assert "javascript" in results

    def test_generate_sub_queries_or(self):
        """Test splitting on 'or'."""
        results = self.expander.generate_sub_queries("docker or kubernetes")
        assert "docker or kubernetes" in results
        assert "docker" in results
        assert "kubernetes" in results

    def test_generate_sub_queries_no_conjunction(self):
        """Test query without conjunctions."""
        results = self.expander.generate_sub_queries("simple query")
        assert results == ["simple query"]

    def test_generate_sub_queries_multiple_parts(self):
        """Test splitting with multiple conjunctions."""
        results = self.expander.generate_sub_queries("a and b and c")
        assert "a and b and c" in results
        assert "a" in results
        assert "b" in results
        assert "c" in results

    def test_expand_all(self):
        """Test combined expansion."""
        results = self.expander.expand("fix the api error")
        assert "fix the api error" in results
        # Should have synonym expansions
        assert len(results) > 1
        # Should be deduplicated
        assert len(results) == len(set(results))

    def test_expand_empty_query(self):
        """Test expansion of empty query."""
        results = self.expander.expand_synonyms("")
        assert results == [""]

    def test_custom_synonym_map(self):
        """Test with custom synonym map."""
        custom_map = {"hello": ["hi", "greetings"]}
        expander = QueryExpander(synonym_map=custom_map)
        results = expander.expand_synonyms("hello world")
        assert any("hi" in r for r in results)
        assert any("greetings" in r for r in results)

    def test_custom_acronym_map(self):
        """Test with custom acronym map."""
        custom_map = {"xyz": "extended yellow zone"}
        expander = QueryExpander(acronym_map=custom_map)
        results = expander.expand_acronyms("check xyz")
        assert any("extended yellow zone" in r for r in results)

    def test_no_duplicates_in_expand(self):
        """Test that expand() returns no duplicates."""
        results = self.expander.expand("test the api")
        assert len(results) == len(set(results))

    def test_case_insensitive_synonyms(self):
        """Test that synonym matching is case-insensitive."""
        results = self.expander.expand_synonyms("Fix the Error")
        # The expansion works on lowercased input
        assert len(results) > 1
