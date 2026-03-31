"""Kernel optimization benchmark."""

from __future__ import annotations

from benchmark.base import BaseBenchmark, EvalResult, Problem
from benchmark.kernel.evaluator import evaluate_kernel
from benchmark.kernel.problems import KERNEL_PROBLEMS, KernelProblem


class KernelBenchmark(BaseBenchmark):
    """Benchmark that asks an agent to produce optimized Triton kernels."""

    def __init__(
        self,
        problem_ids: list[str] | None = None,
        gpu_id: int = 0,
        timeout: int = 120,
    ):
        if problem_ids is None:
            self._problems = dict(KERNEL_PROBLEMS)
        else:
            self._problems = {
                pid: KERNEL_PROBLEMS[pid]
                for pid in problem_ids
                if pid in KERNEL_PROBLEMS
            }
        self.gpu_id = gpu_id
        self.timeout = timeout

    def get_problems(self) -> list[Problem]:
        return [self._to_problem(kp) for kp in self._problems.values()]

    def evaluate(self, problem_id: str, solution: str) -> EvalResult:
        kp = self._problems[problem_id]
        return evaluate_kernel(
            kp, solution, timeout=self.timeout, gpu_id=self.gpu_id
        )

    def get_kernel_problem(self, problem_id: str) -> KernelProblem:
        return self._problems[problem_id]

    @staticmethod
    def _to_problem(kp: KernelProblem) -> Problem:
        prompt = _build_prompt(kp)
        return Problem(
            problem_id=kp.problem_id,
            description=prompt,
            metadata={"entry_point": kp.entry_point},
        )


def _build_prompt(kp: KernelProblem) -> str:
    return f"""\
You are a GPU kernel optimization expert. Write an optimized Triton kernel \
for the following operation.

## Task
{kp.description}

## Reference Implementation
```python
{kp.reference_code.strip()}
```

## Requirements
- Use the Triton library (`import triton` and `import triton.language as tl`).
- Define a Python wrapper function named `{kp.entry_point}` that accepts the \
same inputs as the reference function and returns a `torch.Tensor`.
- The wrapper should allocate the output tensor, compute grid dimensions, and \
launch your Triton kernel.
- Output must be numerically close to the reference (atol={kp.atol}, rtol={kp.rtol}).

Return ONLY valid Python code (no markdown fences, no explanations). The code \
will be executed directly."""
