"""Run OpenEvolve on a vendored AlgoTune task with `gpt-oss-20b`."""

from __future__ import annotations

import os
import subprocess
import sys
import time
import json
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "openai/gpt-oss-20b"
VLLM_PORT = 8000

HF_CACHE_VOL = modal.Volume.from_name("atlas-hf-cache", create_if_missing=True)
OUTPUTS_VOL = modal.Volume.from_name("atlas-openevolve-outputs", create_if_missing=True)
MODELS_VOL = modal.Volume.from_name("atlas-models", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl", "build-essential")
    .pip_install(
        "vllm==0.19.1",
        "openevolve==0.2.27",
        "openai>=1.40",
        "pyyaml>=6",
        "numpy>=1.26",
        "scipy>=1.11",
        "numba>=0.60",
    )
    .add_local_dir(
        str(REPO_ROOT),
        "/atlas",
        copy=True,
        ignore=[
            ".venv",
            "__pycache__",
            ".git",
            "runs",
            "openevolve_output",
            "logs",
            "eval_runs",
            "tmp_runs",
        ],
    )
    .env({"PYTHONPATH": "/atlas", "HF_HOME": "/hf_cache"})
)

app = modal.App("atlas-openevolve-algotune", image=image)


def _wait_for_vllm(port: int, timeout: int = 900) -> None:
    import urllib.error
    import urllib.request

    url = f"http://localhost:{port}/v1/models"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(3)
    raise RuntimeError(f"vLLM did not come up within {timeout}s")


def _start_vllm(adapter_path: str | None = None) -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        MODEL_ID,
        "--port",
        str(VLLM_PORT),
        "--gpu-memory-utilization",
        "0.80",
        "--max-model-len",
        "32768",
        "--reasoning-parser",
        "openai_gptoss",
    ]
    if adapter_path:
        cmd.extend(["--enable-lora", "--lora-modules", f"atlas={adapter_path}"])
    proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
    try:
        _wait_for_vllm(VLLM_PORT)
    except Exception:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        raise
    return proc


def _build_config(iterations: int, system_message: str, served_model: str) -> dict:
    return {
        "max_iterations": iterations,
        "checkpoint_interval": 10,
        "log_level": "INFO",
        "random_seed": 42,
        "diff_based_evolution": False,
        "allow_full_rewrites": True,
        "max_code_length": 20000,
        "llm": {
            "models": [{"name": served_model, "weight": 1.0}],
            "evaluator_models": [{"name": served_model, "weight": 1.0}],
            "api_base": f"http://localhost:{VLLM_PORT}/v1",
            "api_key": "EMPTY",
            "temperature": 0.4,
            "top_p": 0.95,
            "max_tokens": 128000,
            "timeout": 150,
            "retries": 3,
        },
        "prompt": {
            "system_message": system_message,
            "num_top_programs": 5,
            "num_diverse_programs": 5,
            "use_template_stochasticity": True,
            "include_artifacts": True,
            "max_artifact_bytes": 32768,
        },
        "database": {
            "in_memory": True,
            "log_prompts": True,
            "population_size": 1000,
            "archive_size": 100,
            "num_islands": 4,
            "migration_interval": 20,
            "migration_rate": 0.1,
            "elite_selection_ratio": 0.1,
            "exploration_ratio": 0.3,
            "exploitation_ratio": 0.6,
            "feature_bins": 10,
        },
        "evaluator": {
            "timeout": 200,
            "max_retries": 3,
            "parallel_evaluations": 4,
            "cascade_evaluation": True,
            "cascade_thresholds": [0.5, 0.8],
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


@app.function(
    gpu="A100-80GB",
    timeout=2 * 3600,
    volumes={
        "/hf_cache": HF_CACHE_VOL,
        "/outputs": OUTPUTS_VOL,
        "/atlas_models": MODELS_VOL,
    },
)
def run_evolution(
    problem_id: str,
    iterations: int = 30,
    run_name: str | None = None,
    adapter_name: str | None = None,
) -> dict:
    import yaml

    sys.path.insert(0, "/atlas")
    from experiment.tasks import get_task

    spec = get_task("algotune", problem_id)
    run_name = run_name or f"{problem_id}_{int(time.time())}"
    output_dir = Path("/outputs") / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    oe_dir = output_dir / "oe"
    oe_dir.mkdir(parents=True, exist_ok=True)

    initial_program_path = output_dir / "initial_program.py"
    evaluator_path = output_dir / "evaluator.py"
    initial_program_path.write_text(spec.initial_program)
    evaluator_path.write_text(spec.evaluator)

    adapter_path = None
    served_model = MODEL_ID
    if adapter_name:
        adapter_path = f"/atlas_models/{adapter_name}"
        if not Path(adapter_path).exists():
            raise FileNotFoundError(adapter_path)
        served_model = "atlas"

    config = _build_config(iterations, spec.system_message, served_model)
    config_path = output_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    OUTPUTS_VOL.commit()
    os.environ["ATLAS_REASONING_TRACE_PATH"] = str(oe_dir / "reasoning_trace.jsonl")

    vllm_proc = _start_vllm(adapter_path=adapter_path)
    try:
        from openevolve.api import run_evolution as oe_run

        result = oe_run(
            initial_program=str(initial_program_path),
            evaluator=str(evaluator_path),
            config=str(config_path),
            iterations=iterations,
            output_dir=str(oe_dir),
            cleanup=False,
        )

        summary = {
            "run_name": run_name,
            "problem_id": problem_id,
            "adapter_name": adapter_name,
            "served_model": served_model,
            "iterations": iterations,
            "best_score": result.best_score,
            "best_metrics": result.metrics,
            "evolution_trace": str(oe_dir / "evolution_trace.jsonl"),
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        OUTPUTS_VOL.commit()
        HF_CACHE_VOL.commit()
        return summary
    finally:
        vllm_proc.terminate()
        try:
            vllm_proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            vllm_proc.kill()


@app.local_entrypoint()
def main(
    problem_id: str,
    iterations: int = 30,
    run_name: str | None = None,
    adapter_name: str | None = None,
) -> None:
    summary = run_evolution.remote(
        problem_id=problem_id,
        iterations=iterations,
        run_name=run_name,
        adapter_name=adapter_name,
    )
    print(summary)
    print(f"modal volume get atlas-openevolve-outputs {summary['run_name']} ./runs/")
