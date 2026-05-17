"""AWS Bedrock embedder for RAGForge."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from ragforge.embedders.base import BaseEmbedder


class BedrockEmbedder(BaseEmbedder):
    """Embedder using AWS Bedrock Runtime with Titan Embed models.

    Requires boto3 and valid AWS credentials configured in the environment.
    """

    def __init__(
        self,
        model_id: str = "amazon.titan-embed-text-v2:0",
        dimensions: int = 1024,
        region_name: Optional[str] = None,
    ):
        super().__init__(dimensions=dimensions)
        self.model_id = model_id
        self.region_name = region_name
        self._client = None

    @property
    def client(self):
        """Lazy initialization of the Bedrock Runtime client."""
        if self._client is None:
            try:
                import boto3
            except ImportError:
                raise ImportError(
                    "boto3 is required for BedrockEmbedder. "
                    "Install with: pip install substrai-ragforge[aws]"
                )

            kwargs: Dict[str, Any] = {"service_name": "bedrock-runtime"}
            if self.region_name:
                kwargs["region_name"] = self.region_name

            self._client = boto3.client(**kwargs)

        return self._client

    def embed(self, text: str) -> List[float]:
        """Embed text using AWS Bedrock Titan Embed model.

        Args:
            text: The text to embed.

        Returns:
            Embedding vector as a list of floats.
        """
        body = json.dumps({
            "inputText": text,
            "dimensions": self.dimensions,
            "normalize": True,
        })

        response = self.client.invoke_model(
            modelId=self.model_id,
            body=body,
            contentType="application/json",
            accept="application/json",
        )

        response_body = json.loads(response["body"].read())
        return response_body["embedding"]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts using Bedrock.

        Note: Bedrock Titan Embed does not natively support batch embedding,
        so this processes texts sequentially.
        """
        return [self.embed(text) for text in texts]
