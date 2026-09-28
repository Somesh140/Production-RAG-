"""Prompt template loading. Provided, complete - not part of any activity."""
from typing import Any, Dict, Optional

from src.utils.config_loader import load_yaml_config


class PromptManager:
    """Loads and formats prompt templates from config/prompts.yaml."""

    def __init__(self, base_path: Optional[str] = None):
        self.prompts = load_yaml_config("config/prompts.yaml", base_path=base_path)

    def get_prompt(self, prompt_name: str) -> Dict[str, Any]:
        return self.prompts.get(prompt_name, {})

    def format_prompt(self, template: str, **kwargs) -> str:
        return template.format(**kwargs)
