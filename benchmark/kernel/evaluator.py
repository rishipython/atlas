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
from benchmark.kernel.problems import KernelProblem

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
