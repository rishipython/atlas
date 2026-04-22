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
- Define a Python wrapper function named exactly `{kp.entry_point}` that \
accepts the same inputs as the reference function and returns a `torch.Tensor`.
- The wrapper should allocate the output tensor, compute grid dimensions, and \
launch your Triton kernel.
- Output must be numerically close to the reference (atol={kp.atol}, rtol={kp.rtol}).
- Include any `import` statements your code needs (e.g. `import torch`, \
`import triton`, `import triton.language as tl`).

## Response format
You may reason through the problem first. When you are done, put the final, \
complete, self-contained solution as the **last** fenced code block in your \
response, in exactly this form:

```python
# your full solution here (imports + kernel + wrapper function `{kp.entry_point}`)
```

Only the contents of the last ```python ... ``` block will be executed; \
everything outside it is ignored. Do not split the solution across multiple \
blocks — the last block must be runnable on its own."""


def _main():
    """CLI helper: print the full prompt (system + user) for a given problem.

    Useful for eyeballing what the model actually sees, or for piping into
    a chat UI to smoke-test a model without spinning up vLLM/Modal.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Print the full prompt for a kernel problem.",
    )
    parser.add_argument(
        "--problem-id", "-p",
        required=True,
        choices=sorted(KERNEL_PROBLEMS.keys()),
        help="Which kernel problem to render the prompt for.",
    )
    parser.add_argument(
        "--part",
        choices=["system", "user", "both"],
        default="both",
        help="Which part of the prompt to print (default: both).",
    )
    args = parser.parse_args()

    from agent.prompts import TRITON_SYSTEM_PROMPT

    user = _build_prompt(KERNEL_PROBLEMS[args.problem_id])

    if args.part == "system":
        print(TRITON_SYSTEM_PROMPT)
    elif args.part == "user":
        print(user)
    else:
        print("=" * 20 + " SYSTEM PROMPT " + "=" * 20)
        print(TRITON_SYSTEM_PROMPT)
        print()
        print("=" * 20 + " USER PROMPT " + "=" * 20)
        print(user)


if __name__ == "__main__":
    _main()
