"""Basic LLM agent — single-rollout generation via an OpenAI-compatible API."""

from __future__ import annotations

import logging

from agent.base import AgentResult, BaseAgent
from agent.prompts import TRITON_SYSTEM_PROMPT
from utils.extract import extract_code
from utils.llm_client import LLMClient

logger = logging.getLogger(__name__)

__all__ = ["LLMAgent", "TRITON_SYSTEM_PROMPT"]


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
        self.system_prompt = system_prompt or TRITON_SYSTEM_PROMPT
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
        solution = extract_code(response)
        logger.debug(
            "LLMAgent response: %d chars raw -> %d chars extracted",
            len(response), len(solution),
        )

        trajectory = [
            {"role": "prompt", "content": prompt},
            {"role": "response", "content": response},
        ]
        return AgentResult(
            problem_id=problem_id,
            solution=solution,
            trajectory=trajectory,
        )
