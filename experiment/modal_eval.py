"""Run the atlas kernel evaluator on a Modal GPU.

This is a thin wrapper around ``benchmark.kernel.evaluator.evaluate_kernel``
that executes it inside a Modal container so we can validate generated
Triton code against real hardware without a local GPU.

Usage (from repo root, `atlas` conda env):

    modal run experiment/modal_eval.py --problem-id softmax --file sol.txt
    modal run experiment/modal_eval.py --problem-id matmul --file sol.txt --gpu L4
    modal run experiment/modal_eval.py --problem-id softmax --file sol.py --raw

The solution file may be either raw Python code (pass ``--raw``) or a
full LLM response containing a ```python``` fenced block — the fenced
block is extracted client-side before the code is sent to Modal.
"""

from __future__ import annotations

import sys
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent

# Image: small Python + torch (pulls matching triton automatically).
# First build takes a few minutes (not billed); subsequent runs are cached.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "triton==3.1.0")
    .add_local_dir(str(REPO_ROOT), "/atlas", copy=True)
    .env({"PYTHONPATH": "/atlas"})
)

app = modal.App("atlas-kernel-eval", image=image)


@app.function(gpu="T4", timeout=300)
def eval_remote(problem_id: str, solution_code: str) -> dict:
    """Evaluate a candidate kernel solution on the attached GPU."""
    import sys
    sys.path.insert(0, "/atlas")

    import torch
    from benchmark.kernel.evaluator import evaluate_kernel
    from benchmark.kernel.problems import KERNEL_PROBLEMS

    assert torch.cuda.is_available(), "No CUDA device visible inside container"
    device = torch.cuda.get_device_name(0)
    print(f"[remote] evaluating {problem_id} on {device}", flush=True)

    problem = KERNEL_PROBLEMS[problem_id]
    result = evaluate_kernel(problem, solution_code, timeout=180)

    return {
        "problem_id": result.problem_id,
        "correct": result.correct,
        "score": result.score,
        "feedback": result.feedback,
        "metadata": result.metadata,
        "device": device,
    }


@app.local_entrypoint()
def main(problem_id: str, file: str, raw: bool = False):
    """Entry point: read a solution, extract its code, ship to Modal, print result."""
    sys.path.insert(0, str(REPO_ROOT))
    from benchmark.kernel.problems import KERNEL_PROBLEMS
    from utils.extract import extract_code

    if problem_id not in KERNEL_PROBLEMS:
        print(
            f"[error] unknown problem_id={problem_id!r}; "
            f"choose from {sorted(KERNEL_PROBLEMS)}",
            file=sys.stderr,
        )
        raise SystemExit(2)

    text = Path(file).read_text()
    solution = text.strip() if raw else extract_code(text)
    if not solution:
        print("[error] no solution extracted from input file", file=sys.stderr)
        raise SystemExit(2)

    if f"def {KERNEL_PROBLEMS[problem_id].entry_point}" not in solution:
        print(
            f"[warn] extracted code does not contain "
            f"`def {KERNEL_PROBLEMS[problem_id].entry_point}` — eval will "
            f"short-circuit",
            file=sys.stderr,
        )

    print(f"[local] shipping {len(solution)} chars of code to Modal...")
    result = eval_remote.remote(problem_id, solution)

    status = "PASS" if result["correct"] else "FAIL"
    print(f"\n[{status}] {result['problem_id']} on {result['device']}  "
          f"score={result['score']:.3f}")
    print(result["feedback"])
    per_shape = (result.get("metadata") or {}).get("per_shape", [])
    if per_shape:
        print("\nPer-shape details:")
        for r in per_shape:
            print(f"  {r}")
    raise SystemExit(0 if result["correct"] else 1)
