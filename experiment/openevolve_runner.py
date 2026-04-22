"""Run OpenEvolve with ``openai/gpt-oss-20b`` on an atlas kernel problem.

Everything lives inside a single Modal container:
  - vLLM is started as a subprocess serving gpt-oss-20b on localhost:8000
  - OpenEvolve's Python API runs in the same container, pointed at localhost
  - The evaluator is our existing ``evaluate_kernel`` which spawns an isolated
    subprocess per candidate (so the LLM server and the Triton benchmark
    share the GPU but never share a Python interpreter)

Outputs (initial program, evaluator, config, best program, full
evolution_trace.jsonl capturing prompts + code + scores for every
candidate, and OpenEvolve checkpoints) are persisted to the
``atlas-openevolve-outputs`` Modal Volume.

Usage (from the repo root, ``modal`` conda env):

    modal run experiment/openevolve_runner.py --problem-id softmax
    modal run experiment/openevolve_runner.py \\
        --problem-id softmax --iterations 30 --run-name softmax_pilot

Download the trajectory + best program locally after a run with::

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
    )
    .add_local_dir(
        str(REPO_ROOT),
        "/atlas",
        copy=True,
        ignore=[".venv", "__pycache__", ".git", "runs", "openevolve_output"],
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


# ---------------------------------------------------------------------------
# Per-problem artifact generation (initial program + evaluator file)
# ---------------------------------------------------------------------------
def _build_initial_program(problem_id: str) -> str:
    """Return the source code for ``initial_program.py`` for this problem.

    The evolve block starts as the *reference PyTorch implementation*
    renamed to the expected ``entry_point``.  OpenEvolve's first few
    iterations rewrite this torch-based body into a Triton kernel; the
    evaluator rewards only candidates that are numerically correct,
    and the speedup vs. torch becomes the fitness signal.
    """
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
    """Return the source code for ``evaluator.py`` bound to a problem_id.

    OpenEvolve imports this module and calls ``evaluate(program_path)`` for
    every candidate.  We delegate to atlas's ``evaluate_kernel``, which runs
    the candidate in a fresh Python subprocess on the GPU and measures
    correctness + wall-clock speedup vs. the reference.

    Scoring (``combined_score``) uses graduated partial credit so the
    archive / MAP-Elites front can accumulate progressively better
    attempts rather than being flooded with 0.0s:

      * Shape ran and was correct  →  0.50 + 0.50 * min(speedup, 2.0)
      * Shape ran but numerical mismatch                →  0.15
      * Shape ran and threw during the call             →  0.05
      * Subprocess crashed before printing results      →  0.00

    ``combined_score`` is the average per-shape score.  The strictly-
    correct speedup (matching the benchmark's historical definition) is
    still reported as the ``speedup`` metric, and the full per-shape
    breakdown + subprocess feedback are stored as artifacts so OpenEvolve
    can surface the exact error text to the LLM on the next mutation.
    """
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
    """Graduated per-shape credit. See module docstring for levels."""
    if r.get("correct"):
        speedup = float(r.get("speedup", 0.0))
        return 0.5 + 0.5 * min(speedup, 2.0)
    # "error" field is populated when the call raised — compile error,
    # wrong-shape output, pointer issue, etc.  This is a runnable but
    # broken attempt and is very useful context for the LLM.
    if "error" in r:
        return 0.05
    # No error string + not correct means the call returned a tensor
    # whose values didn't match the reference — closer to "almost right"
    # than a full crash, so it gets a slightly higher partial-credit score.
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
        # Subprocess crashed before printing any per-shape result. Leave
        # progress at 0 — feedback still carries the traceback as an
        # artifact so the LLM can see what went wrong.
        shape_scores = []
        progress = 0.0

    metrics = {{
        "correctness": 1.0 if result.correct else 0.0,
        "speedup": float(result.score),  # strict: 0 unless every shape correct
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
    """The system prompt OpenEvolve injects into every mutation request.

    Combines the atlas Triton API cheatsheet with an explicit goal statement.
    We deliberately drop the ``## Response format`` section of the cheatsheet:
    that section tells the LLM to put its answer in a single fenced ```python
    block, which directly conflicts with OpenEvolve's SEARCH/REPLACE diff
    format requested by its built-in user-message template.
    """
    sys.path.insert(0, str(REPO_ROOT))
    from agent.prompts import TRITON_SYSTEM_PROMPT
    from benchmark.kernel.problems import KERNEL_PROBLEMS

    kp = KERNEL_PROBLEMS[problem_id]

    # Keep the opening paragraph + strip the "## Response format" section
    # before the "## Triton API rules" section (which we want to keep).
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
        f"wrapper.  Follow the response format specified in the user "
        f"message exactly (OpenEvolve expects SEARCH/REPLACE diff blocks).\n\n"
    )
    return header + triton_cheatsheet


def _build_config(
    iterations: int, system_message: str, served_model: str = MODEL_ID
) -> dict:
    """OpenEvolve YAML config, rendered as a Python dict for yaml.safe_dump.

    ``served_model`` is the name under which vLLM is serving the model —
    this is ``"atlas"`` when a LoRA adapter is registered via
    ``--lora-modules``, and the HF id (``openai/gpt-oss-20b``) otherwise.
    OpenEvolve's OpenAI client uses this as the ``model`` field in
    /v1/chat/completions calls.
    """
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
            "api_base": f"http://localhost:{VLLM_PORT}/v1",
            "api_key": "EMPTY",
            "temperature": 0.7,
            "top_p": 0.95,
            # gpt-oss-20b spends a big chunk of its output budget on the
            # reasoning trace before it emits the SEARCH/REPLACE diffs, so
            # we give it plenty of room.  max_tokens here caps the RESPONSE
            # size; the prompt + response together must fit in
            # --max-model-len, which we bumped to 32k above.
            "max_tokens": 16384,
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
            "num_top_programs": 3,
            "num_diverse_programs": 2,
            "include_artifacts": True,
            # Bump this so full Triton compile tracebacks (which can run
            # 2-3KB each) make it into the LLM's prompt instead of being
            # truncated to the first line.
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
            # Our kernel benchmark already consumes the whole GPU during
            # eval — parallel evaluations would contend with vLLM and with
            # each other.  Keep this at 1.
            "parallel_evaluations": 1,
            # Our evaluator doesn't define evaluate_stage1 / _stage2, so
            # OpenEvolve's cascade system can't work here anyway.
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
) -> dict:
    import yaml

    sys.path.insert(0, "/atlas")
    from benchmark.kernel.problems import KERNEL_PROBLEMS

    assert problem_id in KERNEL_PROBLEMS, (
        f"Unknown problem_id {problem_id!r}; choose from {sorted(KERNEL_PROBLEMS)}"
    )

    run_name = run_name or f"{problem_id}_{int(time.time())}"
    output_dir = Path("/outputs") / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    oe_dir = output_dir / "oe"

    # Snapshot every input artifact into the outputs volume so a run is
    # fully reproducible from what ends up on disk.
    program_path = output_dir / "initial_program.py"
    program_path.write_text(_build_initial_program(problem_id))

    evaluator_path = output_dir / "evaluator.py"
    evaluator_path.write_text(_build_evaluator(problem_id))

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

    system_message = _build_system_message(problem_id)
    cfg = _build_config(iterations, system_message, served_model=served_model)
    cfg_path = output_dir / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, default_flow_style=False, sort_keys=False))

    print(f"[run] problem={problem_id} iterations={iterations} run_name={run_name}")
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
            "problem_id": problem_id,
            "iterations": iterations,
            "adapter_name": adapter_name,
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
):
    """Kick off an OpenEvolve run on Modal and print the result summary."""
    print(
        f"[local] launching OpenEvolve: "
        f"problem={problem_id} iterations={iterations} "
        f"run_name={run_name} adapter={adapter_name}"
    )
    summary = run_evolution.remote(problem_id, iterations, run_name, adapter_name)
    print("\n=== RESULT ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print()
    print("To download all artifacts (initial program, evaluator, config,")
    print("checkpoints, best program, and the full evolution_trace.jsonl):")
    print(f"  modal volume get atlas-openevolve-outputs {summary['run_name']} ./runs/")
