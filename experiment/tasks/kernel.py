"""Kernel task family — Triton kernel optimization on A100-80GB.

Keeps the existing behaviour of ``openevolve_runner.py``'s inline kernel
builders exactly, just packaged as a ``TaskSpec`` factory so the runner
can share code with the other task families (alphaevolve, algotune,
prompt_opt).
"""

from __future__ import annotations

import sys
from pathlib import Path

from .base import TaskFamily, TaskSpec

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


# Problem-specific concrete optimization strategies, modelled after the
# mlx_metal_kernel_opt example's system message.
_PROBLEM_PLAYBOOK = {
    "softmax": """\

## Problem-specific optimization playbook: 2-D softmax along last dim

### Target
Reference: `torch.nn.functional.softmax(x, dim=-1)`, dtype=float32, shapes
`{'M': 1024, 'N': 1024}`, `{'M': 2048, 'N': 4096}`, `{'M': 256, 'N': 8192}`.
A well-tuned Triton kernel on A100-80GB should land in the **1.2-1.8x
speedup** range; the PyTorch op is already a fused CUDA kernel, so naive
3-pass streaming kernels (max, sum, normalise) will run SLOWER than torch.

### Strategies that actually close the gap (try these in order)
1. **Single-kernel online softmax** (fused one-pass). Keep a running
   `(max, sum)` per row using the numerically-stable recurrence:
   ```
   new_max = max(old_max, tile_max)
   sum = sum * exp(old_max - new_max) + tile_exp_sum(shifted by new_max)
   ```
   One output pass at the end. This is the single biggest win over a 3-pass
   kernel — cuts DRAM reads from 3N to ~2N per row.

2. **One-block-per-row with large BLOCK when N fits.** For N up to ~4096
   you can load the entire row in a single `tl.arange(0, BLOCK)` vector
   (BLOCK = `triton.next_power_of_2(N)`), skipping the loop entirely.
   Switch to the online-softmax loop only for the N=8192 shape.

3. **Autotune BLOCK and num_warps.**  Register a
   `@triton.autotune` with configs at `(BLOCK=1024/2048/4096/8192,
   num_warps=4/8/16, num_stages=2/3/4)`, keyed on `N`.

4. **Store in input dtype without intermediate fp32 buffer.** Cast the
   final normalised values back to `x.dtype` inside the kernel; allocating
   `out = torch.empty_like(x, dtype=torch.float32)` and then `.to(orig)`
   doubles the memory traffic.

### Known failure modes to avoid (each of these we've already seen crash)
- `tl.load(..., dtype=...)` — `tl.load` has **no** `dtype=` kwarg; use
  `.to(tl.float32)` after the load.
- Passing `X_ptr.dtype` to `tl.cast` — pointer objects don't carry a
  dtype at kernel-runtime.
- Storing normalised values through a fresh fp32 output then casting
  after — expensive extra pass.
- Reading a row with `tl.arange(0, BLOCK)` when `BLOCK < N` without a
  loop — silently truncates the softmax.
""",
    "layernorm": """\

## Problem-specific optimization playbook: LayerNorm along last dim

### Target
Reference: `torch.nn.functional.layer_norm(x, (N,), weight, bias, eps)`,
float32 input, weight/bias length N. Target speedup on A100: **1.1-1.4x**.
Naive two-pass (mean then var) kernels typically run at 0.4-0.6x.

### Strategies
1. **Fused single-pass with Welford's online variance.** One kernel does:
   read tile -> update `(mean, m2, count)` -> at the end, compute
   `var = m2 / N`, `inv_std = rsqrt(var + eps)`, then a SECOND pass over
   the same row to write `(x - mean) * inv_std * weight + bias`. Still
   two DRAM passes, but half the latency of the naive version because
   no temporary intermediate buffer is written.

2. **Vectorised loads with `tl.arange(0, BLOCK)` and `tl.multiple_of`.**
   Layernorm is memory-bound; squeezing every bit of bandwidth matters.

3. **Autotune BLOCK / num_warps keyed on N.** Same pattern as softmax.

4. **Keep weight/bias in registers.** Load once per row (they're small
   compared to the row itself).

### Failure modes to avoid
- Computing mean and var in separate kernel launches — enormous overhead.
- Using `tl.sum` over the ENTIRE tile when N > BLOCK — doesn't reduce
  across tiles; you need a scalar accumulator loop.
- Writing `(x - mean) / sqrt(var + eps)` literally — use `rsqrt` and fuse.
""",
    "matmul": """\

## Problem-specific optimization playbook: fp16/fp32 dense matmul

### Target
Reference: `torch.matmul(A, B)`. On A100-80GB with the tested shapes you
should hit **2-4x speedup** over PyTorch's fp32 matmul by using Tensor
Cores with TF32 accumulation (the reference is allowed to change precision
to TF32, this is expected behaviour).

### Strategies
1. **Blocked `tl.dot` with fp32 accumulator.** Canonical Triton matmul
   pattern:
   ```python
   acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
   for k in range(0, K, BLOCK_K):
       a = tl.load(A_ptrs, ...)
       b = tl.load(B_ptrs, ...)
       acc += tl.dot(a, b)  # (BLOCK_M, BLOCK_K) @ (BLOCK_K, BLOCK_N)
   ```

2. **Autotune over `(BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages)`.**
   A useful grid for A100: `BLOCK_M in [64,128,256]`,
   `BLOCK_N in [64,128,256]`, `BLOCK_K in [32,64]`,
   `num_warps in [4,8]`, `num_stages in [3,4]`. Key on `(M, N, K)`.

3. **Swizzled program-id order (group_m).** Boosts L2 hit rate on A100
   for tall/skinny shapes. Standard Triton matmul tutorial pattern.

4. **Do NOT pre-transpose B.** `tl.dot` handles the layout; loading B
   with `(BLOCK_K, BLOCK_N)` shape directly is faster than transposing.

### Failure modes
- `acc = tl.zeros(..., dtype=tl.float16)` — accumulating in fp16 kills
  numerical correctness on long K.
- Forgetting to mask the K-loop tail when `K % BLOCK_K != 0`.
- Using `tl.load(ptr, mask=(m<M) & (n<N))` without parentheses.
""",
}


def _build_initial_program(problem_id: str) -> str:
    sys.path.insert(0, str(REPO_ROOT))
    from benchmark.kernel.problems import KERNEL_PROBLEMS

    kp = KERNEL_PROBLEMS[problem_id]
    renamed_ref = kp.reference_code.replace(
        f"def {kp.ref_entry_point}", f"def {kp.entry_point}"
    ).rstrip()

    return f'''\
"""Starting program for OpenEvolve — {kp.problem_id}.

Task: {kp.description}

Requirements for the evolved program:
  - Define a function named exactly `{kp.entry_point}` with the same
    signature as the reference below.
  - Numerical output must be close to the reference (atol={kp.atol}, rtol={kp.rtol}).
  - Test shapes the evaluator will run: {kp.test_shapes}.

The current body is the reference PyTorch implementation (correct but slow).
Evolve it into a Triton kernel.  Inside the evolve block you may add any
helper functions, `@triton.jit` kernels, imports, constants, etc. — as long as
`{kp.entry_point}` remains defined with the same signature.
"""

# EVOLVE-BLOCK-START
{renamed_ref}
# EVOLVE-BLOCK-END
'''


def _build_evaluator(problem_id: str) -> str:
    return f'''\
"""Auto-generated OpenEvolve evaluator for problem_id={problem_id!r}."""
from __future__ import annotations

import sys
sys.path.insert(0, "/atlas")

from benchmark.kernel.evaluator import evaluate_kernel
from benchmark.kernel.problems import KERNEL_PROBLEMS
from openevolve.evaluation_result import EvaluationResult

PROBLEM_ID = {problem_id!r}


def _per_shape_score(r: dict) -> float:
    if r.get("correct"):
        speedup = float(r.get("speedup", 0.0))
        return 0.5 + 0.5 * min(speedup, 2.0)
    if "error" in r:
        return 0.05
    return 0.15


def evaluate(program_path: str):
    with open(program_path, "r") as f:
        code = f.read()

    problem = KERNEL_PROBLEMS[PROBLEM_ID]
    result = evaluate_kernel(problem, code, timeout=180)

    per_shape = (result.metadata or {{}}).get("per_shape", [])
    if per_shape:
        shape_scores = [_per_shape_score(r) for r in per_shape]
        progress = sum(shape_scores) / len(shape_scores)
    else:
        shape_scores = []
        progress = 0.0

    metrics = {{
        "correctness": 1.0 if result.correct else 0.0,
        "speedup": float(result.score),
        "combined_score": float(progress),
    }}

    artifacts = {{
        "feedback": result.feedback[:6000],
        "per_shape": str(per_shape)[:6000],
        "shape_scores": str(shape_scores),
    }}
    return EvaluationResult(metrics=metrics, artifacts=artifacts)
'''


def _build_system_message(problem_id: str) -> str:
    sys.path.insert(0, str(REPO_ROOT))
    from agent.prompts import TRITON_SYSTEM_PROMPT
    from benchmark.kernel.problems import KERNEL_PROBLEMS

    kp = KERNEL_PROBLEMS[problem_id]

    rules_marker = "## Triton API rules"
    if rules_marker in TRITON_SYSTEM_PROMPT:
        intro, rules = TRITON_SYSTEM_PROMPT.split(rules_marker, 1)
        intro = intro.split("## Response format", 1)[0].rstrip()
        triton_cheatsheet = f"{intro}\n\n{rules_marker}{rules}"
    else:
        triton_cheatsheet = TRITON_SYSTEM_PROMPT

    header = (
        f"Your task: improve the program inside the EVOLVE-BLOCK to make "
        f"`{kp.entry_point}` faster than the reference PyTorch implementation "
        f"while remaining numerically correct on the listed test shapes. "
        f"The winning strategy is almost always to replace the PyTorch call "
        f"with a custom `@triton.jit` kernel launched from a thin Python "
        f"wrapper. If diff-style edits aren't producing improvements, feel "
        f"free to replace the entire EVOLVE-BLOCK with a full rewrite. "
        f"Follow the response format specified in the user message exactly "
        f"(OpenEvolve accepts either SEARCH/REPLACE diffs OR a full rewrite "
        f"of the EVOLVE-BLOCK).\n\n"
    )
    playbook = _PROBLEM_PLAYBOOK.get(problem_id, "")
    return header + triton_cheatsheet + playbook


def _available_problems() -> list[str]:
    sys.path.insert(0, str(REPO_ROOT))
    from benchmark.kernel.problems import KERNEL_PROBLEMS

    return sorted(KERNEL_PROBLEMS.keys())


def make_task(problem_id: str) -> TaskSpec:
    return TaskSpec(
        task_family="kernel",
        problem_id=problem_id,
        initial_program=_build_initial_program(problem_id),
        evaluator=_build_evaluator(problem_id),
        system_message=_build_system_message(problem_id),
        extra_packages=[],
        evaluator_timeout=300,
        uses_vllm_in_evaluator=False,
    )


FAMILY = TaskFamily(
    name="kernel",
    make_task=make_task,
    available_problems=_available_problems(),
)
