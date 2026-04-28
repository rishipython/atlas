"""Run OpenEvolve on AlphaEvolve math problems with ``openai/gpt-oss-20b``.

This is the AlphaEvolve sibling of ``openevolve_runner.py``.  It targets
the problems under
``examples/alphaevolve_math_problems/`` in the upstream OpenEvolve repo
(https://github.com/algorithmicsuperintelligence/openevolve) — small
self-contained CPU-bound math problems with their own ``config.yaml``,
``initial_program.py``, and ``evaluator.py``.

Why a separate runner?  The kernel runner synthesises every problem
artifact from atlas's ``KERNEL_PROBLEMS`` table because it controls the
benchmark.  AlphaEvolve problems already ship a tuned config + per-problem
system prompt + evaluator we want to use verbatim, so this runner only
*overrides* the LLM block (api_base/api_key/models) and bolts on our
``evolution_trace`` logging block, leaving the upstream system prompt,
MAP-Elites database settings, and iteration count untouched.

Pipeline inside the container:
  - clone openevolve at image build time, exposing
    ``/openevolve_repo/examples/alphaevolve_math_problems/<problem>/``
  - vLLM is started as a subprocess on localhost:8000 (with the LoRA
    adapter mounted under name ``"atlas"`` if requested)
  - OpenEvolve's Python API runs in the same container, pointed at
    that local vLLM instance
  - The upstream evaluator imports the candidate program directly (pure
    CPU/numpy work for these problems — no GPU contention with vLLM)

Outputs (snapshotted initial program / evaluator / config, the full
``evolution_trace.jsonl`` capturing prompts + code + scores for every
candidate, the matching ``reasoning_trace.jsonl`` from sitecustomize, and
OpenEvolve checkpoints) are persisted to the
``atlas-alphaevolve-outputs`` Modal Volume.

Usage (from repo root, ``modal`` conda env)::

    # Use the upstream config's max_iterations (default behavior)
    modal run experiment/alphaevolve_runner.py --problem-id circle_packing_rect

    # Override iterations (e.g. for a smoke test)
    modal run experiment/alphaevolve_runner.py \\
        --problem-id circle_packing_rect --iterations 5 --run-name ae_smoke

    # Use an existing LoRA adapter from atlas-models
    modal run experiment/alphaevolve_runner.py \\
        --problem-id first_autocorr_ineq --adapter-name ae_distill_1

    # Sub-N variants (hexagon_packing has 11/, 12/ subdirs)
    modal run experiment/alphaevolve_runner.py --problem-id hexagon_packing/11

Download the trajectory + best program locally::

    modal volume get atlas-alphaevolve-outputs <run_name> ./runs/
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

# Pinning the upstream openevolve repo to ``main`` for now.  When we have
# a stable result worth reproducing, swap this for a specific commit SHA
# so the per-problem files (config / initial / evaluator) are frozen.
OPENEVOLVE_REPO_URL = (
    "https://github.com/algorithmicsuperintelligence/openevolve.git"
)
OPENEVOLVE_REPO_REF = "main"

HF_CACHE_VOL = modal.Volume.from_name("atlas-hf-cache", create_if_missing=True)
OUTPUTS_VOL = modal.Volume.from_name(
    "atlas-alphaevolve-outputs", create_if_missing=True
)
# Reuse the same atlas-models volume as the kernel track so adapters
# trained by ``train_atlas_sft.py`` from either track are interchangeable.
MODELS_VOL = modal.Volume.from_name("atlas-models", create_if_missing=True)


# ---------------------------------------------------------------------------
# Container image
# ---------------------------------------------------------------------------
# Base image matches the kernel runner (vLLM 0.19.1 + openevolve 0.2.27),
# plus we git-clone the upstream openevolve repo so we have access to the
# ``examples/alphaevolve_math_problems/<problem>/`` files at runtime.  We
# also pre-install numpy + scipy because almost every AE problem needs at
# least one of them; per-problem ``requirements.txt`` is still installed
# at runtime to pick up anything else.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl", "build-essential")
    .pip_install(
        "vllm==0.19.1",
        "openevolve==0.2.27",
        "pyyaml>=6",
        "numpy",
        "scipy",
        # Several AE problems (autocorr inequalities, erdos overlap)
        # have ``jax`` / ``optax`` in their requirements.txt.  We pin
        # the CPU-only jax wheel to avoid pip pulling the CUDA build
        # at runtime, which would happily pre-allocate the GPU and
        # fight vLLM for memory.  Per-problem requirements.txt is
        # still pip-installed at runtime for any problem that needs
        # something we missed; that becomes a no-op when the package
        # is already present.
        "jax[cpu]",
        "optax",
    )
    .run_commands(
        f"git clone --depth 1 --branch {OPENEVOLVE_REPO_REF} "
        f"{OPENEVOLVE_REPO_URL} /openevolve_repo"
    )
    .add_local_dir(
        str(REPO_ROOT),
        "/atlas",
        copy=True,
        # Modal's pattern matcher needs explicit ``**/`` for nested
        # matches — bare "__pycache__" / "*.pyc" only matches at the
        # repo root and would let nested ones get picked up, racing
        # with concurrent imports that regenerate them mid-build.
        ignore=[
            ".venv",
            ".git",
            "runs",
            "openevolve_output",
            "__pycache__",
            "**/__pycache__",
            "**/*.pyc",
            "*.pyc",
        ],
    )
    .env(
        {
            "PYTHONPATH": "/atlas",
            "HF_HOME": "/hf_cache",
            # Belt-and-suspenders: even if a candidate's ``import jax``
            # somehow pulls in a CUDA-aware build, this env var routes
            # all jax devices to CPU.  Disabling pre-allocation makes
            # the (rare) case of a CUDA jax import slip through merely
            # slow rather than catastrophic for vLLM.
            "JAX_PLATFORMS": "cpu",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        }
    )
)

app = modal.App("atlas-alphaevolve", image=image)


# ---------------------------------------------------------------------------
# vLLM subprocess management (duplicated from openevolve_runner.py — the
# kernel scripts are deliberately left untouched per project preference,
# so we copy the helpers here rather than refactoring into a shared module)
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

    AE problems do not contend with vLLM for the GPU (the evaluator is
    pure CPU/numpy), so we can crank ``gpu_memory_utilization`` higher
    than the kernel runner — 0.90 leaves room for KV cache spikes
    without starving the model.
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
        "0.90",
        "--max-model-len",
        "32768",
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


# ---------------------------------------------------------------------------
# Config override
# ---------------------------------------------------------------------------
def _override_llm_block(cfg: dict, served_model: str) -> dict:
    """Mutate ``cfg`` in-place to point the LLM block at our local vLLM
    and to enable evolution-trace logging.

    We deliberately preserve every OTHER block of the upstream config —
    most importantly the per-problem ``prompt.system_message`` and the
    ``database`` (MAP-Elites) settings, which are tuned per problem and
    we'd just be guessing at sensible defaults if we overrode them.
    """
    llm = cfg.setdefault("llm", {})
    llm["api_base"] = f"http://localhost:{VLLM_PORT}/v1"
    llm["api_key"] = "EMPTY"
    llm["models"] = [{"name": served_model, "weight": 1.0}]
    llm["evaluator_models"] = [{"name": served_model, "weight": 1.0}]
    # Provide reasonable defaults for fields some upstream configs omit.
    llm.setdefault("temperature", 0.7)
    llm.setdefault("top_p", 0.95)
    # gpt-oss-20b at the default ``reasoning_effort=high`` typically
    # spends 13-14k of any 16k token budget inside the Harmony analysis
    # channel before emitting the SEARCH/REPLACE diff.  Bump the cap to
    # 24k unconditionally — upstream per-problem configs ship with
    # ``max_tokens: 4000`` which truncates high-effort responses
    # mid-diff and starves the cascade.  Use ``=`` (not ``setdefault``)
    # so we always win over those small upstream defaults.
    llm["max_tokens"] = 24576
    llm.setdefault("timeout", 600)
    llm.setdefault("retries", 2)
    llm.setdefault("retry_delay", 5)

    # Always log a full per-iteration trace — this is what the SFT
    # builder reads downstream.
    cfg["evolution_trace"] = {
        "enabled": True,
        "format": "jsonl",
        "include_code": True,
        "include_prompts": True,
        "buffer_size": 1,
        "compress": False,
    }
    return cfg


# ---------------------------------------------------------------------------
# Modal function
# ---------------------------------------------------------------------------
# A100-80GB matches the kernel runner.  4h timeout covers a 200-iter run
# on the slower problems (autocorrelation inequalities, where each
# iteration's evaluator is a few seconds of numpy convolution).
@app.function(
    gpu="A100-80GB",
    timeout=4 * 3600,
    volumes={
        "/hf_cache": HF_CACHE_VOL,
        "/outputs": OUTPUTS_VOL,
        "/atlas_models": MODELS_VOL,
    },
)
def run_evolution(
    problem_id: str,
    iterations: int | None = None,
    run_name: str | None = None,
    adapter_name: str | None = None,
) -> dict:
    """Run OpenEvolve on a single AlphaEvolve math problem.

    Parameters
    ----------
    problem_id   : path under ``examples/alphaevolve_math_problems/`` —
                   either a flat problem name (``circle_packing_rect``)
                   or a sub-N path (``hexagon_packing/11``).
    iterations   : when ``None``, use the upstream config's
                   ``max_iterations``; otherwise override it (useful for
                   smoke tests).
    run_name     : subdirectory under the outputs volume; defaults to
                   ``<problem_id_slug>_<unix_ts>``.
    adapter_name : if set, look for a LoRA adapter dir at
                   ``/atlas_models/<adapter_name>`` and serve it under
                   the alias ``"atlas"`` alongside the base model.
    """
    import json
    import yaml

    problem_dir = (
        Path("/openevolve_repo/examples/alphaevolve_math_problems") / problem_id
    )
    assert problem_dir.is_dir(), (
        f"AlphaEvolve problem dir {problem_dir} does not exist. "
        f"Available: {sorted(p.name for p in problem_dir.parent.iterdir() if p.is_dir())}"
    )
    src_init = problem_dir / "initial_program.py"
    src_eval = problem_dir / "evaluator.py"
    src_cfg = problem_dir / "config.yaml"
    src_req = problem_dir / "requirements.txt"
    for required in (src_init, src_eval, src_cfg):
        assert required.exists(), f"required upstream file missing: {required}"

    # Per-problem requirements.txt — most are just ``numpy``/``scipy``
    # (already in the image), but a few pull in cvxpy / mosek-style
    # solvers we'd otherwise miss.  Quiet install so log noise stays low.
    if src_req.exists() and src_req.read_text().strip():
        print(f"[setup] installing per-problem requirements: {src_req}")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-r", str(src_req)],
            check=True,
        )

    run_name = run_name or f"{problem_id.replace('/', '_')}_{int(time.time())}"
    output_dir = Path("/outputs") / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    oe_dir = output_dir / "oe"
    oe_dir.mkdir(parents=True, exist_ok=True)

    # Snapshot the upstream input artifacts into the run dir so an
    # individual run is fully reproducible from what's on the volume.
    program_path = output_dir / "initial_program.py"
    program_path.write_text(src_init.read_text())
    evaluator_path = output_dir / "evaluator.py"
    evaluator_path.write_text(src_eval.read_text())

    cfg = yaml.safe_load(src_cfg.read_text())
    if iterations is not None:
        cfg["max_iterations"] = int(iterations)
    iters_to_run = int(cfg["max_iterations"])

    # Resolve adapter path / served-model name.
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

    cfg = _override_llm_block(cfg, served_model)
    cfg_path = output_dir / "config.yaml"
    cfg_path.write_text(
        yaml.safe_dump(cfg, default_flow_style=False, sort_keys=False)
    )

    # Tell sitecustomize._log_reasoning where to append per-attempt
    # records.  Picked up by every Python process spawned in this
    # container (including OE's ProcessPoolExecutor workers).
    reasoning_trace_path = oe_dir / "reasoning_trace.jsonl"
    os.environ["ATLAS_REASONING_TRACE_PATH"] = str(reasoning_trace_path)
    # Force the cascade's first attempt to ``reasoning_effort=medium``.
    # We tested ``high`` (v3) and it was strictly worse for OE — at
    # high effort gpt-oss-20b writes ambitious LP/scipy rewrites that
    # diff cleanly into the parent program but contain runtime logic
    # bugs (overlap-constraint violations, off-by-one array indexing,
    # bad signatures) — 96.6% of v3 iters scored 0 vs ~50% in v2.
    # The diff format inherently rewards small targeted edits, which
    # is what medium-effort outputs look like.  BoN-40 wins at high
    # because it asks for full-file rewrites — a different workflow.
    # The cascade still falls back to ``low`` if medium returns empty.
    os.environ["ATLAS_DEFAULT_REASONING_EFFORT"] = "medium"

    print(
        f"[run] problem={problem_id} iterations={iters_to_run} run_name={run_name}"
    )
    print(f"[run] output_dir={output_dir}")
    print(f"[run] initial_program ({program_path.stat().st_size} bytes)")
    print(f"[run] evaluator ({evaluator_path.stat().st_size} bytes)")
    OUTPUTS_VOL.commit()

    vllm_proc = _start_vllm(adapter_path=adapter_path)
    try:
        # Lazy import — keeps a missing ``openevolve`` install from
        # masking the more useful vLLM-startup error path above.
        from openevolve.api import run_evolution as oe_run

        print(f"[oe] starting evolution → {oe_dir}")
        result = oe_run(
            initial_program=str(program_path),
            evaluator=str(evaluator_path),
            config=str(cfg_path),
            iterations=iters_to_run,
            output_dir=str(oe_dir),
            cleanup=False,
        )
        OUTPUTS_VOL.commit()
        HF_CACHE_VOL.commit()

        summary = {
            "run_name": run_name,
            "problem_id": problem_id,
            "iterations": iters_to_run,
            "adapter_name": adapter_name,
            "served_model": served_model,
            "best_score": result.best_score,
            "best_metrics": result.metrics,
            "best_code_chars": len(result.best_code) if result.best_code else 0,
            "output_dir": str(output_dir),
            "evolution_trace": str(oe_dir / "evolution_trace.jsonl"),
            "reasoning_trace": str(reasoning_trace_path),
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, default=str)
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
    iterations: int = -1,
    run_name: str | None = None,
    adapter_name: str | None = None,
):
    """Kick off an OpenEvolve run on an AlphaEvolve math problem.

    Parameters
    ----------
    problem_id   : flat name (e.g. ``circle_packing_rect``) or sub-N path
                   (e.g. ``hexagon_packing/11``).
    iterations   : ``-1`` (default) means "use the upstream config's
                   ``max_iterations``" (typically 100-200).  Pass a
                   positive integer to override (smoke tests, ablations).
    run_name     : optional subdir name on the outputs volume.
    adapter_name : optional LoRA adapter dir from ``atlas-models``.
    """
    iter_arg: int | None = None if iterations < 0 else int(iterations)
    print(
        f"[local] launching AlphaEvolve OE: problem={problem_id} "
        f"iterations={iter_arg if iter_arg is not None else 'config-default'} "
        f"run_name={run_name} adapter={adapter_name}"
    )
    summary = run_evolution.remote(
        problem_id=problem_id,
        iterations=iter_arg,
        run_name=run_name,
        adapter_name=adapter_name,
    )
    print("\n=== RESULT ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print()
    print("Download all artifacts (initial_program, evaluator, config,")
    print("evolution_trace.jsonl, reasoning_trace.jsonl, OE checkpoints):")
    print(
        f"  modal volume get atlas-alphaevolve-outputs {summary['run_name']} ./runs/"
    )
