"""Tests for embedding quantization module."""

import math

import pytest

from ragforge.cost.quantization import EmbeddingQuantizer, QuantizedEmbedding


class TestEmbeddingQuantizer:
    """Tests for EmbeddingQuantizer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.quantizer = EmbeddingQuantizer()

    def test_quantize_float16_basic(self):
        """Test basic float16 quantization."""
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        result = self.quantizer.quantize(embedding, target="float16")

        assert isinstance(result, QuantizedEmbedding)
        assert result.target_type == "float16"
        assert result.original_type == "float32"
        assert result.dimensions == 5
        # float16 = 2 bytes per value
        assert len(result.data) == 10

    def test_dequantize_float16_roundtrip(self):
        """Test float16 quantize/dequantize roundtrip."""
        embedding = [0.1, 0.5, -0.3, 1.0, -1.0]
        quantized = self.quantizer.quantize(embedding, target="float16")
        restored = self.quantizer.dequantize(quantized)

        assert len(restored) == len(embedding)
        # float16 has limited precision, allow some tolerance
        for orig, rest in zip(embedding, restored):
            assert abs(orig - rest) < 0.01

    def test_quantize_int8_basic(self):
        """Test basic int8 quantization."""
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        result = self.quantizer.quantize(embedding, target="int8")

        assert isinstance(result, QuantizedEmbedding)
        assert result.target_type == "int8"
        assert result.dimensions == 5
        # int8 = 1 byte per value
        assert len(result.data) == 5
        assert result.scale > 0

    def test_dequantize_int8_roundtrip(self):
        """Test int8 quantize/dequantize roundtrip."""
        embedding = [0.0, 0.25, 0.5, 0.75, 1.0]
        quantized = self.quantizer.quantize(embedding, target="int8")
        restored = self.quantizer.dequantize(quantized)

        assert len(restored) == len(embedding)
        # int8 has lower precision, allow more tolerance
        for orig, rest in zip(embedding, restored):
            assert abs(orig - rest) < 0.05

    def test_quantize_negative_values(self):
        """Test quantization with negative values."""
        embedding = [-1.0, -0.5, 0.0, 0.5, 1.0]

        # float16
        q16 = self.quantizer.quantize(embedding, target="float16")
        r16 = self.quantizer.dequantize(q16)
        for orig, rest in zip(embedding, r16):
            assert abs(orig - rest) < 0.01

        # int8
        q8 = self.quantizer.quantize(embedding, target="int8")
        r8 = self.quantizer.dequantize(q8)
        for orig, rest in zip(embedding, r8):
            assert abs(orig - rest) < 0.05

    def test_quantize_empty_embedding(self):
        """Test quantization of empty embedding."""
        embedding: list = []

        q16 = self.quantizer.quantize(embedding, target="float16")
        assert q16.dimensions == 0
        assert self.quantizer.dequantize(q16) == []

        q8 = self.quantizer.quantize(embedding, target="int8")
        assert q8.dimensions == 0
        assert self.quantizer.dequantize(q8) == []

    def test_quantize_uniform_values(self):
        """Test quantization when all values are the same."""
        embedding = [0.5, 0.5, 0.5, 0.5]

        q8 = self.quantizer.quantize(embedding, target="int8")
        r8 = self.quantizer.dequantize(q8)
        for orig, rest in zip(embedding, r8):
            assert abs(orig - rest) < 0.05

    def test_quantize_invalid_target(self):
        """Test that invalid target raises ValueError."""
        embedding = [0.1, 0.2, 0.3]
        with pytest.raises(ValueError, match="Unsupported quantization target"):
            self.quantizer.quantize(embedding, target="int4")

    def test_estimate_savings_float16(self):
        """Test storage savings estimation for float16."""
        savings = self.quantizer.estimate_savings(
            num_vectors=10000, dimensions=1024, target="float16"
        )

        assert savings["original_bytes"] == 10000 * 1024 * 4
        assert savings["quantized_bytes"] == 10000 * 1024 * 2
        assert savings["savings_percentage"] == 50.0

    def test_estimate_savings_int8(self):
        """Test storage savings estimation for int8."""
        savings = self.quantizer.estimate_savings(
            num_vectors=10000, dimensions=1024, target="int8"
        )

        assert savings["original_bytes"] == 10000 * 1024 * 4
        # int8: 1 byte per dim + 8 bytes overhead per vector
        expected_quantized = 10000 * (1024 * 1 + 8)
        assert savings["quantized_bytes"] == expected_quantized
        assert savings["savings_percentage"] > 74.0  # ~75% savings

    def test_estimate_savings_invalid_target(self):
        """Test savings estimation with invalid target."""
        with pytest.raises(ValueError):
            self.quantizer.estimate_savings(1000, 384, target="int4")

    def test_large_embedding(self):
        """Test quantization of a large embedding (1024 dimensions)."""
        import random
        random.seed(42)
        embedding = [random.uniform(-1.0, 1.0) for _ in range(1024)]

        # float16 roundtrip
        q16 = self.quantizer.quantize(embedding, target="float16")
        r16 = self.quantizer.dequantize(q16)
        assert len(r16) == 1024
        max_error_16 = max(abs(a - b) for a, b in zip(embedding, r16))
        assert max_error_16 < 0.01

        # int8 roundtrip
        q8 = self.quantizer.quantize(embedding, target="int8")
        r8 = self.quantizer.dequantize(q8)
        assert len(r8) == 1024
        max_error_8 = max(abs(a - b) for a, b in zip(embedding, r8))
        assert max_error_8 < 0.1

    def test_float16_size_is_half(self):
        """Test that float16 data is exactly half the size of float32."""
        embedding = [0.1] * 100
        quantized = self.quantizer.quantize(embedding, target="float16")
        # float32 would be 400 bytes, float16 should be 200
        assert len(quantized.data) == 200
