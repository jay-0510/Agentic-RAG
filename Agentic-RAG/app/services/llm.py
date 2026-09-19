import json
import re
from typing import Protocol

import boto3

from app.core.config import Settings


class LLM(Protocol):
    def complete(self, prompt: str) -> str: ...


class BedrockNovaLLM:
    """Minimal Amazon Nova Micro adapter using the Bedrock Converse API."""

    def __init__(self, settings: Settings):
        self.client = boto3.client("bedrock-runtime", region_name=settings.aws_region)
        self.model_id = settings.nova_model_id

    def complete(self, prompt: str) -> str:
        response = self.client.converse(
            modelId=self.model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": 700, "temperature": 0},
        )
        return response["output"]["message"]["content"][0]["text"]


class LocalLLM:
    """Deterministic development fallback; it keeps API/tests usable without AWS."""

    def complete(self, prompt: str) -> str:
        context_match = re.search(r"CONTEXT:\n(.*?)\n(?:QUESTION|TASK):", prompt, re.S)
        if context_match:
            context = context_match.group(1).strip()
            return f"Based on the supplied knowledge base: {context[:900]}"
        return "I can help with that. Ask me a specific question about the sports knowledge base."


def get_llm(settings: Settings) -> LLM:
    if settings.llm_backend == "bedrock":
        return BedrockNovaLLM(settings)
    return LocalLLM()


def parse_json_response(text: str, default: dict) -> dict:
    """Extract JSON safely because model responses may include Markdown fences."""
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return default
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return default
