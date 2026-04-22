"""Kernel evaluator — runs generated Triton code in a subprocess
and measures correctness + wall-clock speedup vs. the reference."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from benchmark.base import EvalResult
from benchmark.kernel.problems import KERNEL_PROBLEMS, KernelProblem

logger = logging.getLogger(__name__)

# The harness is a self-contained Python script assembled from blocks.
# It prints a JSON payload prefixed with __RESULT__ on success.
_HARNESS = '''\
import json, sys, time, traceback, torch
torch.backends.cudnn.benchmark = False

# ---------- reference implementation ----------
{reference_code}

# ---------- generated implementation ----------
{generated_code}

# ---------- input generator ----------
{input_gen_code}

def main():
    results = []
    shapes = {shapes!r}
    ref_fn = {ref_fn}
    gen_fn = {gen_fn}

    for shape in shapes:
        inputs = generate_inputs(**shape)

        # -- correctness --
        try:
            ref_out = ref_fn(*inputs)
            gen_out = gen_fn(*inputs)
            if not isinstance(gen_out, torch.Tensor):
                results.append(dict(shape=shape, correct=False,
                                    error="output is not a Tensor"))
                continue
            correct = torch.allclose(gen_out, ref_out,
                                     atol={atol}, rtol={rtol})
            max_diff = (gen_out - ref_out).abs().max().item()
        except Exception:
            results.append(dict(shape=shape, correct=False,
                                error=traceback.format_exc()[-1500:]))
            continue

        if not correct:
            results.append(dict(shape=shape, correct=False,
                                max_diff=max_diff))
            continue

        # -- benchmark --
        for _ in range({num_warmup}):
            gen_fn(*inputs)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range({num_bench}):
            gen_fn(*inputs)
        torch.cuda.synchronize()
        gen_ms = (time.perf_counter() - t0) * 1000

        for _ in range({num_warmup}):
            ref_fn(*inputs)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range({num_bench}):
            ref_fn(*inputs)
        torch.cuda.synchronize()
        ref_ms = (time.perf_counter() - t0) * 1000

        speedup = ref_ms / gen_ms if gen_ms > 0 else 0.0
        results.append(dict(
            shape=shape, correct=True,
            speedup=round(speedup, 4),
            ref_ms=round(ref_ms, 3),
            gen_ms=round(gen_ms, 3),
            max_diff=max_diff,
        ))

    print("__RESULT__" + json.dumps(results))

if __name__ == "__main__":
    main()
'''


def evaluate_kernel(
    problem: KernelProblem,
    generated_code: str,
    timeout: int = 120,
    gpu_id: int = 0,
) -> EvalResult:
    """Run *generated_code* against *problem* in an isolated subprocess."""

    if not generated_code or not generated_code.strip():
        return EvalResult(
            problem_id=problem.problem_id,
            correct=False,
            score=0.0,
            feedback=(
                "No code extracted from model response. "
                "Expected a ```python fenced block containing "
                f"`def {problem.entry_point}(...)`."
            ),
        )

    if f"def {problem.entry_point}" not in generated_code:
        return EvalResult(
            problem_id=problem.problem_id,
            correct=False,
            score=0.0,
            feedback=(
                f"Extracted code does not define `{problem.entry_point}`. "
                "Make sure the wrapper function is named exactly "
                f"`{problem.entry_point}` and lives in the final ```python block."
            ),
        )

    harness = _HARNESS.format(
        reference_code=problem.reference_code,
        generated_code=generated_code,
        input_gen_code=problem.input_generator_code,
        shapes=problem.test_shapes,
        ref_fn=problem.ref_entry_point,
        gen_fn=problem.entry_point,
        atol=problem.atol,
        rtol=problem.rtol,
        num_warmup=problem.num_warmup,
        num_bench=problem.num_benchmark,
    )

    fd, script_path = tempfile.mkstemp(suffix=".py", prefix="atlas_eval_")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(harness)

        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
        proc = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return EvalResult(
            problem_id=problem.problem_id,
            correct=False,
            score=0.0,
            feedback="Evaluation timed out.",
        )
    finally:
        Path(script_path).unlink(missing_ok=True)

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""

    if proc.returncode != 0:
        return EvalResult(
            problem_id=problem.problem_id,
            correct=False,
            score=0.0,
            feedback=f"Subprocess crashed (rc={proc.returncode}):\n{stderr[-2000:]}",
        )

    marker = "__RESULT__"
    if marker not in stdout:
        return EvalResult(
            problem_id=problem.problem_id,
            correct=False,
            score=0.0,
            feedback=(
                f"No result marker in output.\n"
                f"stdout: {stdout[-1000:]}\nstderr: {stderr[-1000:]}"
            ),
        )

    payload = stdout.split(marker, 1)[1].strip()
    try:
        results: list[dict] = json.loads(payload)
    except json.JSONDecodeError as exc:
        return EvalResult(
            problem_id=problem.problem_id,
            correct=False,
            score=0.0,
            feedback=f"JSON parse error: {exc}\nraw: {payload[:500]}",
        )

    all_correct = all(r.get("correct") for r in results)
    avg_speedup = 0.0
    if all_correct and results:
        avg_speedup = sum(r.get("speedup", 0) for r in results) / len(results)

    lines = []
    for r in results:
        s = r.get("shape", "?")
        if r.get("correct"):
            lines.append(f"  {s}: correct, speedup={r.get('speedup','?')}x")
        else:
            err = r.get("error", f"max_diff={r.get('max_diff', '?')}")
            lines.append(f"  {s}: INCORRECT — {err}")

    return EvalResult(
        problem_id=problem.problem_id,
        correct=all_correct,
        score=avg_speedup if all_correct else 0.0,
        feedback="\n".join(lines),
        metadata={"per_shape": results},
    )


def _main():
    """CLI helper: evaluate a candidate solution for a single kernel problem.

    Reads the solution from a file (``-f``) or from stdin. By default the
    input is passed through ``extract_code``, so you can paste the raw
    model response (including ``<think>`` traces and surrounding prose)
    and the last ```python fenced block will be pulled out and run.

    Requires an actual GPU environment with Triton installed — this won't
    run on macOS / CPU-only machines.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Run and evaluate a candidate Triton solution.",
    )
    parser.add_argument(
        "--problem-id", "-p",
        required=True,
        choices=sorted(KERNEL_PROBLEMS.keys()),
        help="Which kernel problem to evaluate against.",
    )
    parser.add_argument(
        "--file", "-f",
        type=str, default=None,
        help="Path to a file containing the solution. If omitted, read stdin.",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help=(
            "Skip markdown extraction and pass the input verbatim to the "
            "evaluator. Use when your input is already pure Python code."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=int, default=120,
        help="Subprocess timeout in seconds (default 120).",
    )
    parser.add_argument(
        "--gpu-id",
        type=int, default=0,
        help="CUDA_VISIBLE_DEVICES value for the subprocess (default 0).",
    )
    args = parser.parse_args()

    if args.file:
        raw = Path(args.file).read_text()
    else:
        raw = sys.stdin.read()

    if args.raw:
        solution = raw.strip()
    else:
        from utils.extract import extract_code
        solution = extract_code(raw)

    if not solution:
        print("[error] no solution extracted from input", file=sys.stderr)
        sys.exit(2)

    problem = KERNEL_PROBLEMS[args.problem_id]
    result = evaluate_kernel(
        problem, solution, timeout=args.timeout, gpu_id=args.gpu_id
    )

    status = "PASS" if result.correct else "FAIL"
    print(f"[{status}] {result.problem_id}  score={result.score:.3f}")
    print(result.feedback)
    sys.exit(0 if result.correct else 1)


if __name__ == "__main__":
    _main()
