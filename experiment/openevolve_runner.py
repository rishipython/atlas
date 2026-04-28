"""Run OpenEvolve with ``openai/gpt-oss-20b`` on any registered task.

Everything lives inside a single Modal container:
  - vLLM is started as a subprocess serving gpt-oss-20b on localhost:8000
  - OpenEvolve's Python API runs in the same container, pointed at localhost
  - The evaluator is produced by the task module via ``experiment.tasks``
    and spawns isolated subprocesses per candidate (so the LLM server and
    each candidate share the GPU but never share a Python interpreter)

Task families supported today (see ``experiment/tasks/__init__.py``):
  - ``kernel``       — Triton kernel optimisation (softmax, layernorm, matmul)
  - ``alphaevolve``  — circle packing n=26 (AlphaEvolve math benchmark)
  - ``algotune``     — AlgoTune-style Python speedup (3 signal tasks)
  - ``prompt_opt``   — evolve a system prompt for GSM8K-style questions

Usage (from the repo root, ``modal`` conda env):

    # legacy kernel path (task_family defaults to "kernel")
    modal run experiment/openevolve_runner.py --problem-id softmax

    # new task families
    modal run experiment/openevolve_runner.py \\
        --task-family alphaevolve --problem-id circle_packing_26 \\
        --iterations 40 --run-name alphaevolve_pilot_v1

Outputs (initial program, evaluator, config, best program, full
evolution_trace.jsonl, and OpenEvolve checkpoints) are persisted to the
``atlas-openevolve-outputs`` Modal Volume.  Download them with::

    modal volume get atlas-openevolve-outputs <run_name> ./runs/
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent

MODEL_ID = "openai/gpt-oss-20b"
VLLM_PORT = 8000

HF_CACHE_VOL = modal.Volume.from_name("atlas-hf-cache", create_if_missing=True)
OUTPUTS_VOL = modal.Volume.from_name(
    "atlas-openevolve-outputs", create_if_missing=True
)
# Optional: LoRA adapters saved by ``train_atlas_sft.py`` live here.
# Only mounted when the caller passes ``adapter_name``.
MODELS_VOL = modal.Volume.from_name("atlas-models", create_if_missing=True)


# ---------------------------------------------------------------------------
# Container image: vLLM (gpt-oss merged into mainline since 0.10.2) + openevolve
# ---------------------------------------------------------------------------
# We used to pin ``vllm==0.10.1+gptoss`` per OpenAI's original cookbook, but
# that pin requires a specific nightly PyTorch cu128 wheel which has since
# been rotated out of the PyTorch nightly index.  gpt-oss has been in vLLM
# mainline for a long time now (0.10.2+), so we use the latest stable wheel
# which brings its own compatible torch.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl", "build-essential")
    .pip_install(
        "vllm==0.19.1",
        "openevolve==0.2.27",
        "pyyaml>=6",
        # Task-family-specific evaluator dependencies. Baked into the
        # image so an individual task's ``extra_packages`` don't need to
        # trigger a per-run pip install.
        "scipy>=1.11",
        "numba>=0.60",
        "openai>=1.40",
        "orjson>=3.10",
        "toml>=0.10",
        "pyaml>=24.0",
        "pillow>=10.0",
        # Modal runtime imports these at startup (via modal_proto/client).
        "protobuf>=4.25",
        "grpclib>=0.4.7",
        # alphaevolve_math problems need jax/optax for the jax-based
        # optimizers in several initial_program.py files, and sympy/tqdm
        # for the analytic uncertainty_ineq / heilbronn evaluators.
        "jax>=0.4.20",
        "jaxlib>=0.4.20",
        "optax>=0.1.7",
        "sympy>=1.12",
        "tqdm>=4.66",
        "matplotlib>=3.7",
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
            # Runtime-only directories — mustn't be snapshotted into the
            # image because the local Python processes monitoring Modal
            # runs write into them (log tails, eval result dumps, plots)
            # and Modal's image builder flags any mid-build mutation as
            # a fatal error.
            "logs",
            "eval_runs",
            "tmp_runs",
        ],
    )
    .env({"PYTHONPATH": "/atlas", "HF_HOME": "/hf_cache"})
)

app = modal.App("atlas-openevolve", image=image)


# ---------------------------------------------------------------------------
# vLLM subprocess management
# ---------------------------------------------------------------------------
def _wait_for_vllm(port: int, timeout: int = 900) -> None:
    """Poll vLLM's /v1/models endpoint until it responds 200 or timeout."""
    import urllib.error
    import urllib.request

    url = f"http://localhost:{port}/v1/models"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(3)
    raise RuntimeError(f"vLLM did not come up within {timeout}s")


def _start_vllm(adapter_path: str | None = None) -> subprocess.Popen:
    """Start the OpenAI-compatible vLLM server as a background subprocess.

    When ``adapter_path`` is provided, the base model is served as before
    AND the LoRA adapter is exposed under the name ``"atlas"``, so the
    OpenEvolve config can reference whichever of the two it wants.  When
    no adapter is passed we only serve the base model.
    """
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
        # 32k leaves ample headroom for 16k output after the prompt grew
        # past 16k (parent code + top-K siblings + artifact tracebacks).
        # gpt-oss-20b natively supports up to 128k, so 32k is trivial.
        "--max-model-len",
        "32768",
        # Expose the Harmony analysis channel as a separate
        # ``reasoning_content`` field on chat completions.  Without this
        # flag the analysis text is silently dropped and we only see the
        # final channel; with it, our sitecustomize-side logger can
        # persist the full chain of thought to ``reasoning_trace.jsonl``
        # alongside OpenEvolve's own ``evolution_trace.jsonl``.
        "--reasoning-parser",
        "openai_gptoss",
    ]
    if adapter_path:
        cmd.extend(["--enable-lora", "--lora-modules", f"atlas={adapter_path}"])
    print(f"[vllm] launching: {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
    try:
        _wait_for_vllm(VLLM_PORT)
        print(f"[vllm] ready at http://localhost:{VLLM_PORT}/v1", flush=True)
    except Exception:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        raise
    return proc


# Per-task-family OE config presets. These are merged *on top of* the
# kernel-oriented default below. The keys are task_family strings (matching
# ``experiment.tasks.TASK_FAMILIES``). Each preset is a shallow nested
# override: keys at ``prompt``/``database``/``llm``/``evaluator`` level
# replace the defaults, other sub-keys are left untouched.
#
# The settings here are ported from the OpenEvolve repo's own tuned
# configs:
#   * ``algotune``              ← ``examples/algotune/*/config.yaml``
#   * ``alphaevolve``,          ← ``examples/circle_packing_with_artifacts/config.yaml``
#     ``alphaevolve_math``,
#     ``circle_packing_artifacts``
# (numbers trimmed where our 40-iter/40k-token budget requires it.)
_REPO_PRESETS: dict[str, dict] = {
    "algotune": {
        "checkpoint_interval": 10,
        "diff_based_evolution": False,
        "max_code_length": 20000,
        "llm": {
            "temperature": 0.4,
            "max_tokens": 128000,
            "timeout": 150,
            "retries": 3,
        },
        "prompt": {
            "num_top_programs": 5,
            "num_diverse_programs": 5,
            "include_artifacts": True,
        },
        "database": {
            "population_size": 1000,
            "archive_size": 100,
            "num_islands": 4,
            "feature_bins": 10,
            "migration_interval": 20,
            "migration_rate": 0.1,
            "elite_selection_ratio": 0.1,
            "exploration_ratio": 0.3,
            "exploitation_ratio": 0.6,
        },
        "evaluator": {
            "timeout": 200,
            "max_retries": 3,
            "cascade_evaluation": True,
            "cascade_thresholds": [0.5, 0.8],
            "parallel_evaluations": 4,
        },
    },
    "alphaevolve": {
        "llm": {"temperature": 0.4, "max_tokens": 16000},
        "diff_based_evolution": False,
        "prompt": {"num_top_programs": 3, "num_diverse_programs": 2},
        "database": {
            "num_islands": 4,
            "archive_size": 50,
            "elite_selection_ratio": 0.1,
            "exploration_ratio": 0.4,
            "exploitation_ratio": 0.5,
            "migration_interval": 10,
            "migration_rate": 0.15,
            "feature_dimensions": ["score", "complexity"],
            "feature_bins": 10,
        },
    },
}
# Aliases — same preset for other circle-packing / alphaevolve families.
_REPO_PRESETS["alphaevolve_math"] = _REPO_PRESETS["alphaevolve"]
_REPO_PRESETS["circle_packing_artifacts"] = _REPO_PRESETS["alphaevolve"]


def _apply_preset(cfg: dict, preset: dict) -> None:
    """Shallow-nested merge: ``preset`` overrides ``cfg`` one level deep."""
    for top_key, top_val in preset.items():
        if isinstance(top_val, dict) and isinstance(cfg.get(top_key), dict):
            cfg[top_key].update(top_val)
        else:
            cfg[top_key] = top_val


def _build_config(
    iterations: int,
    system_message: str,
    served_model: str = MODEL_ID,
    evaluator_timeout: int = 300,
    parallel_evaluations: int = 1,
    task_family: str = "kernel",
    random_seed: int = 42,
) -> dict:
    """OpenEvolve YAML config, rendered as a Python dict for yaml.safe_dump.

    ``served_model`` is the name under which vLLM is serving the model —
    this is ``"atlas"`` when a LoRA adapter is registered via
    ``--lora-modules``, and the HF id (``openai/gpt-oss-20b``) otherwise.
    OpenEvolve's OpenAI client uses this as the ``model`` field in
    /v1/chat/completions calls.

    ``evaluator_timeout`` and ``parallel_evaluations`` come from the
    per-task ``TaskSpec`` so cheap CPU-only evaluators (circle packing,
    pairwise distance) don't waste 5 minutes each while GPU-heavy or
    LLM-driven evaluators (kernel, prompt_opt) keep the big budget they
    need.
    """
    cfg = {
        "max_iterations": iterations,
        "checkpoint_interval": 5,
        "log_level": "INFO",
        "random_seed": random_seed,
        "diff_based_evolution": True,
        # Let the LLM replace the entire EVOLVE-BLOCK instead of emitting
        # SEARCH/REPLACE diffs when it thinks a rewrite is warranted.
        # This kills most of the "malformed diff -> no-op iteration"
        # failures we saw on short-budget kernel runs.
        "allow_full_rewrites": True,
        "max_code_length": 20000,
        "llm": {
            "models": [{"name": served_model, "weight": 1.0}],
            "evaluator_models": [{"name": served_model, "weight": 1.0}],
            "api_base": f"http://localhost:{VLLM_PORT}/v1",
            "api_key": "EMPTY",
            "temperature": 0.7,
            "top_p": 0.95,
            # gpt-oss-20b spends a big chunk of its output budget on the
            # reasoning trace before it emits the SEARCH/REPLACE diffs, so
            # we give it plenty of room.  32k matches the mlx_metal_kernel_opt
            # example — pushes right up against --max-model-len=65536 so
            # the prompt has 32k headroom for top-programs + artifacts.
            "max_tokens": 32000,
            # Reasoning budget is left at gpt-oss's default ("high"). When
            # the first attempt returns empty content (all tokens spent in
            # the Harmony analysis channel, nothing emitted to the final
            # channel), the cascading retry installed by
            # ``sitecustomize.py`` at container startup automatically
            # re-issues the request with reasoning_effort="medium", then
            # "low", before giving up — preserving full thinking budget
            # when it's productive and forcing an answer when it isn't.
            "timeout": 600,
            "retries": 2,
            "retry_delay": 5,
        },
        "prompt": {
            "system_message": system_message,
            # More reference programs per prompt.  The default 3+2 is sized
            # for small search problems; for kernel writing where each
            # "reference" is a few-KB Triton file, the base LLM benefits
            # from seeing more correct-but-different attempts to draw from.
            "num_top_programs": 5,
            "num_diverse_programs": 3,
            # Default-on; make it explicit so the OE runner doesn't fall
            # silent on it.
            "use_template_stochasticity": True,
            "include_artifacts": True,
            # Bump this so full Triton compile tracebacks (which can run
            # 2-3KB each) make it into the LLM's prompt instead of being
            # truncated to the first line.
            "max_artifact_bytes": 32768,
            # Default 500 makes OE inject a "simplify your code" hint into
            # EVERY prompt for kernel programs, because every valid
            # triton kernel is 1-3KB.  That's actively counterproductive
            # for this task -- push the threshold way up.
            "suggest_simplification_after_chars": 8000,
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
            "timeout": evaluator_timeout,
            "max_retries": 2,
            # Most of our evaluators share the GPU with vLLM (kernel) or
            # hit the same vLLM server OE is driving (prompt_opt), so
            # parallel_evaluations=1 is the safe default.  Task specs can
            # override it when the eval is pure CPU + cheap.
            "parallel_evaluations": parallel_evaluations,
            "cascade_evaluation": False,
        },
        "evolution_trace": {
            "enabled": True,
            "format": "jsonl",
            "include_code": True,
            "include_prompts": True,
            # Flush on every record so even an interrupted run leaves a
            # useful trajectory file.
            "buffer_size": 1,
            "compress": False,
        },
    }

    preset = _REPO_PRESETS.get(task_family)
    if preset is not None:
        _apply_preset(cfg, preset)
        print(
            f"[config] applied '{task_family}' OE-repo preset "
            f"(top-level overrides: {sorted(preset.keys())})"
        )

    return cfg


# ---------------------------------------------------------------------------
# Modal function: one container runs vLLM + OpenEvolve end-to-end
# ---------------------------------------------------------------------------
# A100-80GB gives us compute capability 8.0 (MXFP4-compatible) with plenty
# of headroom (gpt-oss-20b MXFP4 is ~16GB; vLLM takes ~64GB at 0.80 util;
# that leaves ~16GB for the eval subprocess which needs <1GB).
#
# 2h timeout covers a 60-iteration run with room to spare.
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
    task_family: str = "kernel",
    random_seed: int = 42,
    rlm_memory_bank_text: str | None = None,
) -> dict:
    import yaml

    sys.path.insert(0, "/atlas")
    from experiment.tasks import TASK_FAMILIES, get_task

    if task_family not in TASK_FAMILIES:
        raise ValueError(
            f"Unknown task_family {task_family!r}; "
            f"choose from {sorted(TASK_FAMILIES)}"
        )
    spec = get_task(task_family, problem_id)

    run_name = run_name or f"{task_family}_{problem_id}_{int(time.time())}"
    output_dir = Path("/outputs") / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    oe_dir = output_dir / "oe"

    # Snapshot every input artifact into the outputs volume so a run is
    # fully reproducible from what ends up on disk.
    program_path = output_dir / "initial_program.py"
    program_path.write_text(spec.initial_program)

    evaluator_path = output_dir / "evaluator.py"
    evaluator_path.write_text(spec.evaluator)
    # Some vendored AlgoTune evaluators import a sibling `task_ref.py`.
    # Ensure it is present in the run directory for those tasks.
    if task_family == "algotune":
        task_ref_src = (
            Path("/atlas")
            / "experiment"
            / "tasks"
            / "_oe_problems"
            / "algotune_examples"
            / problem_id
            / "task_ref.py"
        )
        if task_ref_src.exists():
            (output_dir / "task_ref.py").write_text(task_ref_src.read_text())

    # If an adapter was requested, resolve the on-volume path and pick
    # the served-model name vLLM will register it under.  Base-only
    # runs keep the original ``openai/gpt-oss-20b`` id.
    adapter_path: str | None = None
    served_model = MODEL_ID
    if adapter_name:
        adapter_path = f"/atlas_models/{adapter_name}"
        assert Path(adapter_path).exists(), (
            f"Adapter directory {adapter_path} not found on atlas-models Volume"
        )
        served_model = "atlas"
        print(f"[run] adapter={adapter_name} served_model={served_model}")
    else:
        print(f"[run] served_model={served_model} (base, no adapter)")

    system_message = spec.system_message
    if rlm_memory_bank_text:
        memory = rlm_memory_bank_text.strip()
        if memory:
            # Keep prompt bloat bounded while still preserving the latest context.
            max_chars = 24_000
            if len(memory) > max_chars:
                memory = memory[-max_chars:]
            system_message = (
                f"{system_message}\n\n"
                "RLM MEMORY BANK (distilled notes from prior related problems):\n"
                "Use these as heuristics and anti-pattern reminders, but still verify "
                "correctness against the exact current task requirements.\n\n"
                f"{memory}\n"
            )

    cfg = _build_config(
        iterations,
        system_message,
        served_model=served_model,
        evaluator_timeout=spec.evaluator_timeout,
        parallel_evaluations=1,
        task_family=task_family,
        random_seed=random_seed,
    )
    cfg_path = output_dir / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, default_flow_style=False, sort_keys=False))

    print(
        f"[run] task_family={task_family} problem={problem_id} "
        f"iterations={iterations} run_name={run_name}"
    )
    print(f"[run] output_dir={output_dir}")
    print(f"[run] initial_program ({program_path.stat().st_size} bytes)")
    print(f"[run] evaluator ({evaluator_path.stat().st_size} bytes)")

    # Persist the inputs immediately so they survive even a vLLM crash.
    OUTPUTS_VOL.commit()

    # Tell ``sitecustomize._log_reasoning`` where to append reasoning
    # trace records.  Picked up by both the main process and the
    # ProcessPoolExecutor workers OpenEvolve spawns, because this env
    # var is inherited from the container.
    reasoning_trace_path = oe_dir / "reasoning_trace.jsonl"
    oe_dir.mkdir(parents=True, exist_ok=True)
    os.environ["ATLAS_REASONING_TRACE_PATH"] = str(reasoning_trace_path)

    vllm_proc = _start_vllm(adapter_path=adapter_path)
    try:
        # Import lazily so a missing openevolve install doesn't break the
        # vLLM-startup path (which is where most early failures actually are).
        from openevolve.api import run_evolution as oe_run

        print(f"[oe] starting evolution → {oe_dir}")
        result = oe_run(
            initial_program=str(program_path),
            evaluator=str(evaluator_path),
            config=str(cfg_path),
            iterations=iterations,
            output_dir=str(oe_dir),
            cleanup=False,
        )

        # Persist outputs (trace, checkpoints, best program) to the volume.
        # HF cache volume already has the model weights downloaded by vLLM;
        # commit it too so the next run skips the 16GB download.
        OUTPUTS_VOL.commit()
        HF_CACHE_VOL.commit()

        summary = {
            "run_name": run_name,
            "task_family": task_family,
            "problem_id": problem_id,
            "iterations": iterations,
            "adapter_name": adapter_name,
            "rlm_enabled": bool(rlm_memory_bank_text and rlm_memory_bank_text.strip()),
            "served_model": served_model,
            "best_score": result.best_score,
            "best_metrics": result.metrics,
            "best_code_chars": len(result.best_code) if result.best_code else 0,
            "output_dir": str(output_dir),
            "evolution_trace": str(oe_dir / "evolution_trace.jsonl"),
        }
        # Also drop a JSON summary so downloads capture it.
        (output_dir / "summary.json").write_text(
            __import__("json").dumps(summary, indent=2, default=str)
        )
        OUTPUTS_VOL.commit()
        return summary

    finally:
        print("[vllm] terminating server subprocess...")
        vllm_proc.terminate()
        try:
            vllm_proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            vllm_proc.kill()
        # Commit HF cache even on failure so a retry doesn't re-download.
        try:
            HF_CACHE_VOL.commit()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Local entry point
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    problem_id: str,
    iterations: int = 30,
    run_name: str | None = None,
    adapter_name: str | None = None,
    task_family: str = "kernel",
    random_seed: int = 42,
    rlm_memory_bank: str | None = None,
):
    """Kick off an OpenEvolve run on Modal and print the result summary."""
    print(
        f"[local] launching OpenEvolve: "
        f"task_family={task_family} problem={problem_id} "
        f"iterations={iterations} run_name={run_name} adapter={adapter_name} "
        f"seed={random_seed}"
    )
    rlm_memory_bank_text = None
    if rlm_memory_bank:
        p = Path(rlm_memory_bank)
        if not p.exists():
            raise FileNotFoundError(f"RLM memory bank not found: {p}")
        rlm_memory_bank_text = p.read_text()

    summary = run_evolution.remote(
        problem_id=problem_id,
        iterations=iterations,
        run_name=run_name,
        adapter_name=adapter_name,
        task_family=task_family,
        random_seed=random_seed,
        rlm_memory_bank_text=rlm_memory_bank_text,
    )
    print("\n=== RESULT ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print()
    print("To download all artifacts (initial program, evaluator, config,")
    print("checkpoints, best program, and the full evolution_trace.jsonl):")
    print(f"  modal volume get atlas-openevolve-outputs {summary['run_name']} ./runs/")
