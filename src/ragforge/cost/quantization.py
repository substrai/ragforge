"""Embedding quantization for storage cost reduction.

Converts float32 embeddings to lower precision formats for storage savings.
Uses pure Python (struct module) for zero-dependency operation.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class QuantizedEmbedding:
    """A quantized embedding with metadata for dequantization."""

    data: bytes
    original_type: str  # "float32"
    target_type: str  # "float16" | "int8"
    dimensions: int
    scale: float = 1.0  # Used for int8 dequantization
    offset: float = 0.0  # Used for int8 dequantization


class EmbeddingQuantizer:
    """Quantizes embeddings for storage cost reduction.

    Supports:
        - float32 → float16: 50% storage savings, minimal quality loss
        - float32 → int8: 75% storage savings, some quality loss

    Uses pure Python struct module for conversions (no numpy dependency).
    """

    def quantize(
        self, embedding: List[float], target: str = "float16"
    ) -> QuantizedEmbedding:
        """Quantize a float32 embedding to a lower precision format.

        Args:
            embedding: List of float32 values.
            target: Target format ("float16" or "int8").

        Returns:
            QuantizedEmbedding with compressed data.

        Raises:
            ValueError: If target format is not supported.
        """
        if target == "float16":
            return self._quantize_float16(embedding)
        elif target == "int8":
            return self._quantize_int8(embedding)
        else:
            raise ValueError(f"Unsupported quantization target: {target}. Use 'float16' or 'int8'.")

    def dequantize(self, quantized: QuantizedEmbedding) -> List[float]:
        """Dequantize an embedding back to float32.

        Args:
            quantized: QuantizedEmbedding to decompress.

        Returns:
            List of float32 values.
        """
        if quantized.target_type == "float16":
            return self._dequantize_float16(quantized)
        elif quantized.target_type == "int8":
            return self._dequantize_int8(quantized)
        else:
            raise ValueError(f"Unsupported quantization type: {quantized.target_type}")

    def estimate_savings(
        self, num_vectors: int, dimensions: int, target: str = "float16"
    ) -> dict:
        """Estimate storage savings from quantization.

        Args:
            num_vectors: Number of vectors to quantize.
            dimensions: Embedding dimensions.
            target: Target quantization format.

        Returns:
            Dictionary with original_bytes, quantized_bytes, savings_bytes,
            savings_percentage.
        """
        original_bytes = num_vectors * dimensions * 4  # float32 = 4 bytes

        if target == "float16":
            quantized_bytes = num_vectors * dimensions * 2  # float16 = 2 bytes
        elif target == "int8":
            # int8 = 1 byte per dim + 8 bytes overhead (scale + offset) per vector
            quantized_bytes = num_vectors * (dimensions * 1 + 8)
        else:
            raise ValueError(f"Unsupported target: {target}")

        savings_bytes = original_bytes - quantized_bytes
        savings_percentage = (savings_bytes / original_bytes * 100) if original_bytes > 0 else 0.0

        return {
            "original_bytes": original_bytes,
            "quantized_bytes": quantized_bytes,
            "savings_bytes": savings_bytes,
            "savings_percentage": round(savings_percentage, 1),
        }

    def _quantize_float16(self, embedding: List[float]) -> QuantizedEmbedding:
        """Quantize float32 to float16 using struct half-precision format.

        Args:
            embedding: List of float32 values.

        Returns:
            QuantizedEmbedding with float16 data.
        """
        # Pack each float as half-precision (2 bytes each)
        data = struct.pack(f"<{len(embedding)}e", *embedding)

        return QuantizedEmbedding(
            data=data,
            original_type="float32",
            target_type="float16",
            dimensions=len(embedding),
        )

    def _dequantize_float16(self, quantized: QuantizedEmbedding) -> List[float]:
        """Dequantize float16 back to float32.

        Args:
            quantized: QuantizedEmbedding with float16 data.

        Returns:
            List of float32 values.
        """
        values = struct.unpack(f"<{quantized.dimensions}e", quantized.data)
        return [float(v) for v in values]

    def _quantize_int8(self, embedding: List[float]) -> QuantizedEmbedding:
        """Quantize float32 to int8 using linear scaling.

        Maps the range [min, max] of the embedding to [-128, 127].

        Args:
            embedding: List of float32 values.

        Returns:
            QuantizedEmbedding with int8 data.
        """
        if not embedding:
            return QuantizedEmbedding(
                data=b"",
                original_type="float32",
                target_type="int8",
                dimensions=0,
                scale=1.0,
                offset=0.0,
            )

        min_val = min(embedding)
        max_val = max(embedding)

        # Avoid division by zero
        value_range = max_val - min_val
        if value_range == 0:
            # All values are the same - store zero int8 values with the
            # original value as offset and scale=0 to signal uniform case
            scale = 0.0
            offset = min_val
            int8_values = [0] * len(embedding)
        else:
            scale = value_range / 255.0  # Map to 0-255 range, then shift to -128..127
            offset = min_val
            int8_values = [
                max(-128, min(127, int(round((v - offset) / scale)) - 128))
                for v in embedding
            ]

        # Pack as signed bytes
        data = struct.pack(f"<{len(int8_values)}b", *int8_values)

        return QuantizedEmbedding(
            data=data,
            original_type="float32",
            target_type="int8",
            dimensions=len(embedding),
            scale=scale,
            offset=offset,
        )

    def _dequantize_int8(self, quantized: QuantizedEmbedding) -> List[float]:
        """Dequantize int8 back to float32.

        Args:
            quantized: QuantizedEmbedding with int8 data.

        Returns:
            List of float32 values (approximate).
        """
        if quantized.dimensions == 0:
            return []

        # Uniform case: all values were the same
        if quantized.scale == 0.0:
            return [quantized.offset] * quantized.dimensions

        int8_values = struct.unpack(f"<{quantized.dimensions}b", quantized.data)

        return [
            (v + 128) * quantized.scale + quantized.offset
            for v in int8_values
        ]
