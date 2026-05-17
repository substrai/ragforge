"""Tests for CodeAwareChunker."""

import pytest

from ragforge.chunkers.code_aware import CodeAwareChunker
from ragforge.core.models import Chunk


PYTHON_SAMPLE = '''"""Module docstring."""

import os
import sys


class MyClass:
    """A sample class."""

    def __init__(self, name):
        self.name = name

    def greet(self):
        return f"Hello, {self.name}"


def standalone_function(x, y):
    """Add two numbers."""
    return x + y


async def async_function():
    """An async function."""
    await something()
    return True
'''

JS_SAMPLE = '''import { useState } from 'react';

function greet(name) {
    return `Hello, ${name}`;
}

const add = (a, b) => {
    return a + b;
};

class Calculator {
    constructor() {
        this.result = 0;
    }

    add(n) {
        this.result += n;
        return this;
    }
}

export const multiply = (a, b) => {
    return a * b;
};
'''

TS_SAMPLE = '''interface User {
    name: string;
    age: number;
}

function createUser(name: string, age: number): User {
    return { name, age };
}

const getAge = (user: User) => {
    return user.age;
};

class UserService {
    private users: User[] = [];

    addUser(user: User): void {
        this.users.push(user);
    }
}
'''


class TestCodeAwareChunkerPython:
    """Test code-aware chunking with Python samples."""

    def test_splits_on_class_boundary(self):
        chunker = CodeAwareChunker(max_chunk_size=2000, language="py")
        chunks = chunker.chunk(PYTHON_SAMPLE, source="test.py")

        assert len(chunks) > 0
        # Should have chunks for imports/preamble, class, and standalone functions
        contents = [c.content for c in chunks]
        # At least one chunk should contain the class
        assert any("class MyClass" in c for c in contents)

    def test_splits_on_function_boundary(self):
        chunker = CodeAwareChunker(max_chunk_size=2000, language="py")
        chunks = chunker.chunk(PYTHON_SAMPLE, source="test.py")

        contents = [c.content for c in chunks]
        # standalone_function should be in the chunks
        assert any("standalone_function" in c for c in contents)

    def test_detects_async_functions(self):
        chunker = CodeAwareChunker(max_chunk_size=2000, language="py")
        chunks = chunker.chunk(PYTHON_SAMPLE, source="test.py")

        contents = [c.content for c in chunks]
        assert any("async_function" in c for c in contents)

    def test_respects_max_chunk_size(self):
        chunker = CodeAwareChunker(max_chunk_size=100, language="py")
        chunks = chunker.chunk(PYTHON_SAMPLE, source="test.py")

        for chunk in chunks:
            assert len(chunk.content) <= 200  # Allow some flexibility for line boundaries

    def test_metadata_includes_language(self):
        chunker = CodeAwareChunker(max_chunk_size=2000, language="py")
        chunks = chunker.chunk(PYTHON_SAMPLE, source="test.py")

        for chunk in chunks:
            assert chunk.metadata["language"] == "py"
            assert chunk.metadata["chunker"] == "code_aware"

    def test_detects_language_from_source(self):
        chunker = CodeAwareChunker(max_chunk_size=2000)
        chunks = chunker.chunk(PYTHON_SAMPLE, source="module.py")

        for chunk in chunks:
            assert chunk.metadata["language"] == "py"

    def test_chunk_ids_are_unique(self):
        chunker = CodeAwareChunker(max_chunk_size=2000, language="py")
        chunks = chunker.chunk(PYTHON_SAMPLE, source="test.py")

        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_empty_content(self):
        chunker = CodeAwareChunker(max_chunk_size=2000, language="py")
        chunks = chunker.chunk("", source="empty.py")

        assert chunks == []

    def test_preserves_source(self):
        chunker = CodeAwareChunker(max_chunk_size=2000, language="py")
        chunks = chunker.chunk(PYTHON_SAMPLE, source="my/module.py")

        for chunk in chunks:
            assert chunk.source == "my/module.py"


class TestCodeAwareChunkerJS:
    """Test code-aware chunking with JavaScript/TypeScript samples."""

    def test_splits_js_functions(self):
        chunker = CodeAwareChunker(max_chunk_size=2000, language="js")
        chunks = chunker.chunk(JS_SAMPLE, source="app.js")

        contents = [c.content for c in chunks]
        assert any("function greet" in c for c in contents)

    def test_splits_js_arrow_functions(self):
        chunker = CodeAwareChunker(max_chunk_size=2000, language="js")
        chunks = chunker.chunk(JS_SAMPLE, source="app.js")

        contents = [c.content for c in chunks]
        assert any("const add" in c for c in contents)

    def test_splits_js_classes(self):
        chunker = CodeAwareChunker(max_chunk_size=2000, language="js")
        chunks = chunker.chunk(JS_SAMPLE, source="app.js")

        contents = [c.content for c in chunks]
        assert any("class Calculator" in c for c in contents)

    def test_detects_ts_from_extension(self):
        chunker = CodeAwareChunker(max_chunk_size=2000)
        chunks = chunker.chunk(TS_SAMPLE, source="service.ts")

        for chunk in chunks:
            assert chunk.metadata["language"] == "ts"

    def test_ts_function_detection(self):
        chunker = CodeAwareChunker(max_chunk_size=2000, language="ts")
        chunks = chunker.chunk(TS_SAMPLE, source="service.ts")

        contents = [c.content for c in chunks]
        assert any("function createUser" in c for c in contents)

    def test_ts_class_detection(self):
        chunker = CodeAwareChunker(max_chunk_size=2000, language="ts")
        chunks = chunker.chunk(TS_SAMPLE, source="service.ts")

        contents = [c.content for c in chunks]
        assert any("class UserService" in c for c in contents)

    def test_custom_metadata_preserved(self):
        chunker = CodeAwareChunker(max_chunk_size=2000, language="js")
        chunks = chunker.chunk(JS_SAMPLE, source="app.js", metadata={"author": "test"})

        for chunk in chunks:
            assert chunk.metadata["author"] == "test"
