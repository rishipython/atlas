"""Thin wrapper around the OpenAI chat completions API.

Works with any OpenAI-compatible server (vLLM, SGLang, etc.)."""

from __future__ import annotations

import logging
from openai import OpenAI

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(
        self,
        model: str,
        api_base: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
    ):
        self.model = model
        self.client = OpenAI(base_url=api_base, api_key=api_key)

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.6,
        max_tokens: int = 4096,
        stop: list[str] | None = None,
    ) -> str:
        logger.debug(
            "LLMClient.chat  model=%s  temp=%.2f  max_tokens=%d",
            self.model, temperature, max_tokens,
        )
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
        )
        return resp.choices[0].message.content or ""
