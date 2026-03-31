from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

from benchmark.base import EvalResult


@dataclass
class AgentResult:
    problem_id: str
    solution: str
    eval_result: EvalResult | None = None
    trajectory: list[dict[str, Any]] = field(default_factory=list)


class BaseAgent(ABC):
    """Base class for all agents.

    Parameters
    ----------
    eval_fn : callable, optional
        ``eval_fn(problem_id, solution) -> EvalResult``.
        Benchmark-provided evaluation callback that TTT agents can call
        during their search loop.  Basic agents may ignore it.
    """

    def __init__(self, eval_fn: Callable[[str, str], EvalResult] | None = None):
        self.eval_fn = eval_fn

    @abstractmethod
    def solve(self, problem_id: str, prompt: str) -> AgentResult:
        """Generate a solution for the given problem.

        Parameters
        ----------
        problem_id : str
            Unique problem identifier.
        prompt : str
            The full problem description / prompt to present to the model.

        Returns
        -------
        AgentResult
        """
        ...
