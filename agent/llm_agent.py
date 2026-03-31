"""Basic LLM agent — single-rollout generation via an OpenAI-compatible API."""

from __future__ import annotations

import logging
import re

from agent.base import AgentResult, BaseAgent
from utils.llm_client import LLMClient

logger = logging.getLogger(__name__)


class LLMAgent(BaseAgent):
    """Generates a solution in one shot (no search / TTT)."""

    def __init__(
        self,
        llm: LLMClient,
        system_prompt: str | None = None,
        temperature: float = 0.6,
        max_tokens: int = 4096,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.llm = llm
        self.system_prompt = system_prompt or (
            "You are an expert programmer specializing in GPU kernel optimization "
            "with Triton. Return ONLY valid Python code with no markdown formatting."
        )
        self.temperature = temperature
        self.max_tokens = max_tokens

    def solve(self, problem_id: str, prompt: str) -> AgentResult:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        logger.info("LLMAgent: generating solution for %s", problem_id)
        response = self.llm.chat(
            messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        solution = _strip_markdown_fences(response)
        logger.debug("LLMAgent raw response length: %d chars", len(response))

        trajectory = [
            {"role": "prompt", "content": prompt},
            {"role": "response", "content": response},
        ]
        return AgentResult(
            problem_id=problem_id,
            solution=solution,
            trajectory=trajectory,
        )


def _strip_markdown_fences(text: str) -> str:
    """Remove ```python ... ``` wrappers if present."""
    pattern = r"```(?:python)?\s*\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()
