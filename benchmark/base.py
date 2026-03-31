from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Problem:
    problem_id: str
    description: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    problem_id: str
    correct: bool
    score: float = 0.0
    feedback: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseBenchmark(ABC):
    """Base class for all benchmarks.

    A benchmark provides problems to an agent, evaluates solutions,
    and exposes an evaluation API that agents can call during search
    (needed for test-time training methods).
    """

    @abstractmethod
    def get_problems(self) -> list[Problem]:
        """Return all problems in this benchmark."""
        ...

    @abstractmethod
    def evaluate(self, problem_id: str, solution: str) -> EvalResult:
        """Evaluate a candidate solution for a given problem.

        This can be called repeatedly by TTT agents during their
        optimization loop — not just at the end.
        """
        ...

    def get_problem(self, problem_id: str) -> Problem:
        for p in self.get_problems():
            if p.problem_id == problem_id:
                return p
        raise KeyError(f"Problem {problem_id!r} not found")
