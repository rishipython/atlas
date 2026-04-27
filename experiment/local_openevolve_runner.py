"""Run OpenEvolve locally on one GPU-backed machine.

This is the Colab/local analogue of `experiment/openevolve_runner.py`.
It starts a local vLLM server, generates the OpenEvolve input artifacts under
`runs/<run_name>/`, and then launches the OpenEvolve Python API against them.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agent.prompts import TRITON_SYSTEM_PROMPT
from benchmark.kernel.problems import KERNEL_PROBLEMS
from experiment.local_vllm_utils import resolve_adapter_path, start_vllm

MODEL_ID = "openai/gpt-oss-20b"


def build_initial_program(problem_id: str) -> str:
    kp = KERNEL_PROBLEMS[problem_id]
    renamed_ref = kp.reference_code.replace(
        f"def {kp.ref_entry_point}", f"def {kp.entry_point}"
    ).rstrip()
    return f'''\
"""Starting program for OpenEvolve — {kp.problem_id}."""

# EVOLVE-BLOCK-START
{renamed_ref}
# EVOLVE-BLOCK-END
'''


def build_evaluator(problem_id: str) -> str:
    return f'''\
from __future__ import annotations

import sys
sys.path.insert(0, {str(REPO_ROOT)!r})

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
    progress = sum(_per_shape_score(r) for r in per_shape) / len(per_shape) if per_shape else 0.0
    metrics = {{
        "correctness": 1.0 if result.correct else 0.0,
        "speedup": float(result.score),
        "combined_score": float(progress),
    }}
    artifacts = {{
        "feedback": result.feedback[:6000],
        "per_shape": str(per_shape)[:6000],
    }}
    return EvaluationResult(metrics=metrics, artifacts=artifacts)
'''


def build_system_message(problem_id: str) -> str:
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
        f"Follow the response format specified in the user message exactly.\n\n"
    )
    return header + triton_cheatsheet


def build_config(iterations: int, system_message: str, served_model: str) -> dict:
    return {
        "max_iterations": iterations,
        "checkpoint_interval": 5,
        "log_level": "INFO",
        "random_seed": 42,
        "diff_based_evolution": True,
        "max_code_length": 20000,
        "llm": {
            "models": [{"name": served_model, "weight": 1.0}],
            "evaluator_models": [{"name": served_model, "weight": 1.0}],
            "api_base": "http://localhost:8000/v1",
            "api_key": "EMPTY",
            "temperature": 0.7,
            "top_p": 0.95,
            "max_tokens": 16384,
            "timeout": 600,
            "retries": 2,
            "retry_delay": 5,
        },
        "prompt": {
            "system_message": system_message,
            "num_top_programs": 3,
            "num_diverse_programs": 2,
            "include_artifacts": True,
            "max_artifact_bytes": 16384,
        },
        "database": {
            "in_memory": True,
            "log_prompts": True,
            "population_size": 200,
            "archive_size": 30,
            "num_islands": 3,
            "migration_interval": 20,
            "migration_rate": 0.1,
            "elite_selection_ratio": 0.2,
            "exploration_ratio": 0.3,
            "exploitation_ratio": 0.6,
        },
        "evaluator": {
            "timeout": 300,
            "max_retries": 2,
            "parallel_evaluations": 1,
            "cascade_evaluation": False,
        },
        "evolution_trace": {
            "enabled": True,
            "format": "jsonl",
            "include_code": True,
            "include_prompts": True,
            "buffer_size": 1,
            "compress": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem-id", required=True, choices=sorted(KERNEL_PROBLEMS.keys()))
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--adapter", default=None, help="Local adapter path or name.")
    parser.add_argument("--base-model", default=MODEL_ID)
    parser.add_argument("--vllm-gpu-mem", type=float, default=0.72)
    args = parser.parse_args()

    run_name = args.run_name or f"{args.problem_id}_{int(time.time())}"
    output_dir = Path("runs") / run_name
    oe_dir = output_dir / "oe"
    oe_dir.mkdir(parents=True, exist_ok=True)

    program_path = output_dir / "initial_program.py"
    evaluator_path = output_dir / "evaluator.py"
    config_path = output_dir / "config.yaml"
    program_path.write_text(build_initial_program(args.problem_id))
    evaluator_path.write_text(build_evaluator(args.problem_id))

    adapter_path = resolve_adapter_path(args.adapter) if args.adapter else None
    served_model = "atlas" if adapter_path else args.base_model
    config_path.write_text(
        yaml.safe_dump(
            build_config(args.iterations, build_system_message(args.problem_id), served_model),
            default_flow_style=False,
            sort_keys=False,
        )
    )

    os.environ["ATLAS_REASONING_TRACE_PATH"] = str(oe_dir / "reasoning_trace.jsonl")
    vllm_proc = start_vllm(
        args.base_model,
        adapter_path=adapter_path,
        gpu_memory_utilization=args.vllm_gpu_mem,
    )
    try:
        from openevolve.api import run_evolution as oe_run

        result = oe_run(
            initial_program=str(program_path),
            evaluator=str(evaluator_path),
            config=str(config_path),
            iterations=args.iterations,
            output_dir=str(oe_dir),
            cleanup=False,
        )
        summary = {
            "run_name": run_name,
            "problem_id": args.problem_id,
            "iterations": args.iterations,
            "adapter": args.adapter,
            "served_model": served_model,
            "best_score": result.best_score,
            "best_metrics": result.metrics,
            "best_code_chars": len(result.best_code) if result.best_code else 0,
            "output_dir": str(output_dir),
            "evolution_trace": str(oe_dir / "evolution_trace.jsonl"),
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        print(json.dumps(summary, indent=2))
    finally:
        vllm_proc.terminate()
        try:
            vllm_proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            vllm_proc.kill()


if __name__ == "__main__":
    main()
