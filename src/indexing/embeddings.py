"""Embedding generation, configured from config/llm_config.yaml.
Provided, complete - not part of any activity.

The same LLM_BASE_URL provider switch as src/llm/llm_client.py applies here.
OpenRouter's /embeddings endpoint lives at the same base URL as its chat
endpoint.
"""
import os
from typing import List, Optional

from openai import OpenAI

from src.utils.config_loader import load_yaml_config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class EmbeddingGenerator:
    """Generate embeddings via an OpenAI-compatible API."""

    def __init__(self, model: Optional[str] = None, base_path: Optional[str] = None):
        config = load_yaml_config("config/llm_config.yaml", base_path=base_path)
        api_config = config.get("api", {})
        embedding_config = config.get("models", {}).get("embedding", {})

        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL") or None,
            timeout=api_config.get("timeout", 60),
            max_retries=api_config.get("max_retries", 2),
        )
        self.model = model or embedding_config.get("model_name", "text-embedding-3-small")
        self.dimensions = embedding_config.get("dimensions", 1536)

    def embed_query(self, text: str) -> List[float]:
        """Embed a single string."""
        try:
            response = self.client.embeddings.create(model=self.model, input=text)
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Embedding generation error: {e}")
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of strings, preserving order."""
        try:
            response = self.client.embeddings.create(model=self.model, input=texts)
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"Batch embedding generation error: {e}")
            raise
