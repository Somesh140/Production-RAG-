"""OpenAI-compatible LLM client, configured from config/llm_config.yaml.
Provided, complete - not part of any activity."""
import os
from typing import Dict, List, Optional

from openai import OpenAI

from src.utils.config_loader import load_yaml_config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class LLMClient:
    """Wrapper for chat completions.

    Model, temperature and token settings come from config/llm_config.yaml by
    default. Pass explicit arguments to override the config for one instance
    (e.g. in tests) without editing the YAML file.

    LLM_BASE_URL is optional. If it's unset, this is a normal OpenAI client.
    Set it (e.g. to https://openrouter.ai/api/v1) to send requests to an
    OpenAI-API-compatible provider, using the same OPENAI_API_KEY variable.
    """

    def __init__(self, model: Optional[str] = None, temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None, base_path: Optional[str] = None):
        config = load_yaml_config("config/llm_config.yaml", base_path=base_path)
        chat_config = config.get("models", {}).get("chat", {})
        api_config = config.get("api", {})

        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL") or None,
            timeout=api_config.get("timeout", 60),
            max_retries=api_config.get("max_retries", 2),
        )
        self.model = model or chat_config.get("model_name", "gpt-4")
        self.temperature = temperature if temperature is not None else chat_config.get("temperature", 0.2)
        self.max_tokens = max_tokens if max_tokens is not None else chat_config.get("max_tokens", 800)

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate a completion from a list of chat messages."""
        try:
            response = self.client.chat.completions.create(
                model=kwargs.get("model", self.model),
                messages=messages,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            raise
