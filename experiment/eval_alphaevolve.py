"""Standalone (no-OpenEvolve) evaluation of a model on AlphaEvolve math problems.

Sibling of ``eval_standalone.py`` for AlphaEvolve.  Spawns a single vLLM
container (with the LoRA adapter mounted alongside the base model when
requested), draws ``--n-samples`` completions per (problem, model) leg,
and scores each generation by invoking the upstream AE evaluator in a
short-lived subprocess (so a misbehaving candidate's import-time side
effects can't poison subsequent samples).

Reported per-leg metrics
------------------------
For each (problem, model) leg we record::

    {
        "problem_id": "circle_packing_rect",
        "served_model": "openai/gpt-oss-20b" | "atlas",
        "n_samples": 40,
        "best_score":             0.74,   # max(combined_score)
        "mean_score":             0.18,   # mean over all samples (errors -> 0)
        "mean_score_when_valid":  0.41,   # mean over samples with combined_score > 0
        "pct_valid":              0.45,   # fraction with combined_score > 0
        "num_beat_sota":          0,      # count with combined_score >= 1.0
        "pass_at_1":              0.0,    # 1 if first sample beat SOTA else 0
        "pass_at_k":              0.0,    # fraction of samples that beat SOTA
    }

For our purposes ``best_score`` is the headline metric — this is the
"best-of-N" comparison the user asked for vs. the OE-driven runs.

Typical use
-----------
    # Best-of-40 base baseline on the four chain problems, single container
    modal run experiment/eval_alphaevolve.py \\
        --problems circle_packing_rect,first_autocorr_ineq,hexagon_packing/11,erdos_min_overlap \\
        --run-name ae_base_bo40 --n-samples 40

    # Distilled-model leg vs base on a single problem
    modal run experiment/eval_alphaevolve.py \\
        --problems first_autocorr_ineq \\
        --adapter-name ae_distill_1 --run-name ae_d1_vs_base_p2 --n-samples 40
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent

BASE_MODEL_DEFAULT = "openai/gpt-oss-20b"
VLLM_PORT = 8000

OPENEVOLVE_REPO_URL = (
    "https://github.com/algorithmicsuperintelligence/openevolve.git"
)
OPENEVOLVE_REPO_REF = "main"

HF_CACHE_VOL = modal.Volume.from_name("atlas-hf-cache", create_if_missing=True)
MODELS_VOL = modal.Volume.from_name("atlas-models", create_if_missing=True)
OUTPUTS_VOL = modal.Volume.from_name(
    "atlas-alphaevolve-outputs", create_if_missing=True
)

# Image mirrors ``alphaevolve_runner.py`` (same vLLM, same openevolve
# clone).  We don't need ``openevolve`` itself here — there's no
# evolutionary loop — but we do need the cloned repo's
# ``examples/alphaevolve_math_problems/<problem>/`` files.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl", "build-essential")
    .pip_install(
        "vllm==0.19.1",
        "openai>=1.50",
        "pyyaml>=6",
        "numpy",
        "scipy",
        # Same reasoning as alphaevolve_runner.py — pre-pin jax-CPU so
        # candidate code that uses jax never touches the GPU.
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
            "JAX_PLATFORMS": "cpu",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        }
    )
)

app = modal.App("atlas-alphaevolve-eval", image=image)


# ---------------------------------------------------------------------------
# vLLM management (duplicated from openevolve_runner / eval_standalone —
# kernel scripts left untouched per project preference)
# ---------------------------------------------------------------------------
def _wait_for_vllm(port: int, timeout: int = 900) -> None:
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


def _start_vllm(base_model: str, adapter_path: str | None) -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        base_model,
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
# Per-problem prompt construction (must match the Phase-2 base prompt
# emitted by ``build_alphaevolve_sft.py`` exactly — that's the whole point
# of Phase 2: train and eval prompts agree on x_base).
# ---------------------------------------------------------------------------
def _problem_dir(problem_id: str) -> Path:
    p = Path("/openevolve_repo/examples/alphaevolve_math_problems") / problem_id
    if not p.is_dir():
        avail = sorted(
            d.name for d in p.parent.iterdir() if d.is_dir()
        ) if p.parent.exists() else []
        raise AssertionError(
            f"AlphaEvolve problem dir {p} does not exist. Available: {avail}"
        )
    return p


def _build_eval_prompts(problem_id: str) -> tuple[str, str]:
    """Return (system, user) for a one-shot solve attempt.

    Reads ``config.yaml`` and ``initial_program.py`` straight from the
    upstream openevolve clone in the container so the prompt content is
    pinned to the same upstream files OE saw during search.
    """
    import yaml

    pdir = _problem_dir(problem_id)
    cfg = yaml.safe_load((pdir / "config.yaml").read_text())
    system = cfg.get("prompt", {}).get("system_message", "")
    assert system, (
        f"upstream config.yaml at {pdir / 'config.yaml'} has no "
        f"prompt.system_message — cannot build eval prompt"
    )
    initial_program = (pdir / "initial_program.py").read_text()

    user = (
        f"Below is the starter program for the `{problem_id}` problem.\n\n"
        f"```python\n{initial_program.rstrip()}\n```\n\n"
        f"Your task: write a complete, runnable Python file that improves "
        f"on this starter to MAXIMIZE the `combined_score` reported by "
        f"the problem's evaluator (a `combined_score` ≥ 1.0 means matching "
        f"or beating the AlphaEvolve state-of-the-art).  Keep the same "
        f"function signature(s) the evaluator expects; preserve safe "
        f"`if __name__ == \"__main__\":` guards so importing the file is "
        f"side-effect-free.  Return a single ```python fenced block "
        f"containing the full solution."
    )
    return system, user


# ---------------------------------------------------------------------------
# Subprocess-isolated AE evaluator wrapper
# ---------------------------------------------------------------------------
_EVAL_HARNESS = """\
import json, sys, traceback, importlib.util
EVALUATOR_PATH = {evaluator_path!r}
SAMPLE_PATH    = {sample_path!r}

try:
    spec = importlib.util.spec_from_file_location('_ae_evaluator', EVALUATOR_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    result = mod.evaluate(SAMPLE_PATH)
    if not isinstance(result, dict):
        result = {{'combined_score': 0.0, 'error': f'evaluator returned non-dict: {{type(result).__name__}}'}}
except Exception:
    result = {{'combined_score': 0.0, 'error': traceback.format_exc()[-2000:]}}

print('__RESULT__' + json.dumps(result, default=str))
"""


def _eval_ae_sample(
    evaluator_path: Path, sample_path: Path, timeout: int = 360
) -> dict:
    """Run AE's ``evaluate(sample_path)`` in a fresh subprocess.

    Subprocess isolation matters because AE evaluators import the
    candidate program with a fresh ``__import__`` and never reload it —
    a candidate that misbehaves at import time (or pollutes
    ``sys.modules``) would otherwise carry over into later samples.
    """
    harness = _EVAL_HARNESS.format(
        evaluator_path=str(evaluator_path),
        sample_path=str(sample_path),
    )
    # Inherit container env (including JAX_PLATFORMS=cpu) and pin the
    # subprocess to CPU explicitly so a candidate that does ``import jax``
    # without configuring devices first cannot grab the GPU.
    sub_env = {**os.environ}
    sub_env.setdefault("JAX_PLATFORMS", "cpu")
    sub_env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    try:
        proc = subprocess.run(
            [sys.executable, "-c", harness],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=sub_env,
        )
    except subprocess.TimeoutExpired:
        return {"combined_score": 0.0, "error": "evaluator timed out"}

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    marker = "__RESULT__"
    if marker not in stdout:
        return {
            "combined_score": 0.0,
            "error": (
                f"no result marker (rc={proc.returncode}). "
                f"stderr tail: {stderr[-1500:]}"
            ),
        }
    payload = stdout.split(marker, 1)[1].strip()
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        return {
            "combined_score": 0.0,
            "error": f"JSON parse error: {exc}; raw: {payload[:500]}",
        }


# ---------------------------------------------------------------------------
# vLLM batched sampling (mirrors eval_standalone._batched_generate)
# ---------------------------------------------------------------------------
def _batched_call(
    client,
    served_model: str,
    messages: list[dict],
    n: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    seed: int,
    reasoning_effort: str | None,
    batch_size: int,
    label: str,
) -> list[tuple[str, str | None]]:
    import concurrent.futures as cf

    def _one_call(i: int) -> tuple[str, str | None]:
        params = dict(
            model=served_model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            n=1,
            seed=seed + i,
        )
        if reasoning_effort is not None:
            params["extra_body"] = {"reasoning_effort": reasoning_effort}
        resp = client.chat.completions.create(**params)
        msg = resp.choices[0].message
        content = msg.content or ""
        reasoning = getattr(msg, "reasoning", None) or getattr(
            msg, "reasoning_content", None
        )
        return content, reasoning

    out: list[tuple[str, str | None]] = [("", None)] * n
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=batch_size) as pool:
        futures = {pool.submit(_one_call, i): i for i in range(n)}
        done_count = 0
        for fut in cf.as_completed(futures):
            i = futures[fut]
            try:
                out[i] = fut.result()
            except Exception as exc:  # noqa: BLE001
                print(f"      [batch:{label}] slot {i} errored: {exc}", flush=True)
                out[i] = ("", None)
            done_count += 1
            if done_count % max(1, batch_size) == 0 or done_count == n:
                dt = time.time() - t0
                print(
                    f"      [batch:{label}] {done_count}/{n} done "
                    f"effort={reasoning_effort or 'default'} elapsed={dt:.1f}s",
                    flush=True,
                )
    return out


def _batched_generate(
    client,
    served_model: str,
    messages: list[dict],
    n_samples: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    seed: int,
    batch_size: int = 32,
) -> list[tuple[str, str | None, str]]:
    """Cascading batched generation: default → medium → low reasoning_effort.

    gpt-oss frequently exhausts its budget in the analysis channel and
    returns empty content; we re-roll just those slots at progressively
    lower reasoning_effort.
    """
    # Start at ``medium`` instead of the server default ``high``: at
    # high, gpt-oss-20b's analysis channel routinely consumes 13-14k
    # of the 16k token budget before emitting any final-channel
    # content, which truncates the python solution mid-file.  Medium
    # caps the analysis budget so the code completes; we still fall
    # back to low if medium happens to return empty.
    cascade: list[tuple[str | None, str]] = [
        ("medium", "medium"),
        ("low", "low"),
    ]
    results: list[tuple[str, str | None, str]] = [("", None, "default")] * n_samples
    pending = list(range(n_samples))

    for pass_idx, (effort, tag) in enumerate(cascade):
        if not pending:
            break
        print(
            f"      [batch] pass {pass_idx + 1}/{len(cascade)} effort={tag} "
            f"{'rolling' if pass_idx == 0 else 're-rolling'} "
            f"{len(pending)} slots",
            flush=True,
        )
        draws = _batched_call(
            client=client,
            served_model=served_model,
            messages=messages,
            n=len(pending),
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=seed + pass_idx * 7_777_777,
            reasoning_effort=effort,
            batch_size=batch_size,
            label=f"pass{pass_idx + 1}",
        )
        next_pending: list[int] = []
        for slot, (content, reasoning) in zip(pending, draws):
            results[slot] = (content, reasoning, tag)
            if not content.strip():
                next_pending.append(slot)
        pending = next_pending

    if pending:
        print(
            f"      [batch] {len(pending)} slots remained empty after full cascade",
            flush=True,
        )
    return results


# ---------------------------------------------------------------------------
# Per-leg sample + score
# ---------------------------------------------------------------------------
def _sample_and_score(
    *,
    client,
    served_model: str,
    problem_id: str,
    system_msg: str,
    user_msg: str,
    n_samples: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    seed: int,
    output_dir: Path,
    extract_code,
    evaluator_path: Path,
    eval_timeout: int,
    outputs_vol,
    batch_size: int = 32,
) -> tuple[list[dict], dict]:
    samples: list[dict] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "system_message.txt").write_text(system_msg)
    (output_dir / "user_message.txt").write_text(user_msg)

    gen_t0 = time.time()
    generations = _batched_generate(
        client,
        served_model=served_model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        n_samples=n_samples,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        seed=seed,
        batch_size=batch_size,
    )
    total_gen_s = time.time() - gen_t0
    print(
        f"      [batch] generated {n_samples} completions in {total_gen_s:.1f}s "
        f"({n_samples / max(total_gen_s, 1e-6):.2f} seq/s)",
        flush=True,
    )

    samples_dir = output_dir / "candidates"
    samples_dir.mkdir(exist_ok=True)

    for i, (raw, reasoning, winning_effort) in enumerate(generations):
        code = extract_code(raw)
        sample_path = samples_dir / f"sample_{i:03d}.py"
        # Always write something even on empty content so the file exists
        # for inspection — eval will then return combined_score=0.
        sample_path.write_text(code or "# (empty generation)\n")

        eval_t0 = time.time()
        result = _eval_ae_sample(evaluator_path, sample_path, timeout=eval_timeout)
        eval_dt = time.time() - eval_t0

        cs = float(result.get("combined_score", 0.0))
        err = result.get("error")
        samples.append(
            {
                "sample_idx": i,
                "raw_length": len(raw),
                "code_length": len(code),
                "reasoning_length": len(reasoning) if reasoning else 0,
                "winning_reasoning_effort": winning_effort,
                "eval_time_s": round(eval_dt, 2),
                "combined_score": cs,
                "metrics": {k: v for k, v in result.items() if k != "error"},
                "error": (err[-1500:] if isinstance(err, str) else None),
            }
        )
        (output_dir / f"sample_{i:03d}_raw.txt").write_text(raw)
        if reasoning:
            (output_dir / f"sample_{i:03d}_reasoning.txt").write_text(reasoning)
        flag = "OK " if (err is None and cs > 0) else "BAD"
        print(
            f"    [{served_model}/{problem_id}] [{i + 1}/{n_samples}] "
            f"{flag} combined_score={cs:.4f} eval={eval_dt:.2f}s",
            flush=True,
        )
        if (i + 1) % 20 == 0 or (i + 1) == n_samples:
            outputs_vol.commit()

    valid = [s for s in samples if s["error"] is None and s["combined_score"] > 0]
    beat_sota = [s for s in samples if s["combined_score"] >= 1.0]
    summary = {
        "problem_id": problem_id,
        "served_model": served_model,
        "n_samples": n_samples,
        "temperature": temperature,
        "best_score": (
            max((s["combined_score"] for s in samples), default=0.0)
        ),
        "mean_score": (
            sum(s["combined_score"] for s in samples) / max(1, len(samples))
        ),
        "mean_score_when_valid": (
            (sum(s["combined_score"] for s in valid) / len(valid))
            if valid
            else 0.0
        ),
        "pct_valid": len(valid) / max(1, n_samples),
        "num_beat_sota": len(beat_sota),
        "pass_at_1": (
            1.0 if samples and samples[0]["combined_score"] >= 1.0 else 0.0
        ),
        "pass_at_k": len(beat_sota) / max(1, n_samples),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (output_dir / "samples.json").write_text(json.dumps(samples, indent=2))
    outputs_vol.commit()
    return samples, summary


# ---------------------------------------------------------------------------
# Modal entry: one container, multiple (problem × {base, adapter}) legs
# ---------------------------------------------------------------------------
@app.function(
    gpu="A100-80GB",
    timeout=4 * 3600,
    volumes={
        "/hf_cache": HF_CACHE_VOL,
        "/atlas_models": MODELS_VOL,
        "/outputs": OUTPUTS_VOL,
    },
)
def run_eval_sweep(
    problems: list[str],
    n_samples: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    adapter_name: str | None,
    run_name: str,
    base_model: str,
    seed: int,
    eval_base: bool,
    eval_adapter: bool,
    eval_timeout: int,
) -> dict:
    import openai

    sys.path.insert(0, "/atlas")
    from utils.extract import extract_code

    # Validate all problems up front + install per-problem requirements
    # so we don't burn 10 min of GPU time on vLLM startup before noticing.
    for pid in problems:
        pdir = _problem_dir(pid)
        req = pdir / "requirements.txt"
        if req.exists() and req.read_text().strip():
            print(f"[setup] installing requirements for {pid}: {req}")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "-r", str(req)],
                check=True,
            )

    adapter_path = None
    if adapter_name:
        adapter_path = f"/atlas_models/{adapter_name}"
        assert Path(adapter_path).exists(), (
            f"Adapter {adapter_path} not found on atlas-models Volume"
        )
        print(f"[eval] LoRA adapter at {adapter_path}", flush=True)

    vllm_proc = _start_vllm(base_model, adapter_path)
    try:
        client = openai.OpenAI(
            api_key="EMPTY", base_url=f"http://localhost:{VLLM_PORT}/v1"
        )

        legs: list[tuple[str, str]] = []
        if eval_base:
            legs.append((base_model, "base"))
        if eval_adapter and adapter_path:
            legs.append(("atlas", "atlas"))
        assert legs, (
            "no legs to evaluate — pass --eval-base True and/or "
            "--eval-adapter True (with --adapter-name)"
        )

        all_summaries: list[dict] = []
        base_dir = Path("/outputs") / "eval" / run_name
        base_dir.mkdir(parents=True, exist_ok=True)

        for pid in problems:
            pdir = _problem_dir(pid)
            evaluator_path = pdir / "evaluator.py"
            assert evaluator_path.exists(), (
                f"upstream evaluator missing for {pid}: {evaluator_path}"
            )

            system_msg, user_msg = _build_eval_prompts(pid)
            for served, tag in legs:
                # Slugify pid (handles "hexagon_packing/11" → "hexagon_packing_11")
                pid_slug = pid.replace("/", "_")
                leg_dir = base_dir / f"{pid_slug}__{tag}"
                print(
                    f"\n[eval] === leg: problem={pid} model={tag} "
                    f"(served='{served}') ===",
                    flush=True,
                )
                _, summary = _sample_and_score(
                    client=client,
                    served_model=served,
                    problem_id=pid,
                    system_msg=system_msg,
                    user_msg=user_msg,
                    n_samples=n_samples,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    seed=seed,
                    output_dir=leg_dir,
                    extract_code=extract_code,
                    evaluator_path=evaluator_path,
                    eval_timeout=eval_timeout,
                    outputs_vol=OUTPUTS_VOL,
                )
                summary["tag"] = tag
                all_summaries.append(summary)

        compare = {
            "run_name": run_name,
            "base_model": base_model,
            "adapter_name": adapter_name,
            "n_samples": n_samples,
            "temperature": temperature,
            "legs": all_summaries,
        }
        (base_dir / "compare.json").write_text(json.dumps(compare, indent=2))
        OUTPUTS_VOL.commit()

        # Compact comparison table.
        print("\n=== CROSS-CONDITION COMPARISON ===", flush=True)
        print(
            f"{'problem':<28} {'leg':<8} {'best':>8} {'mean':>8} "
            f"{'mean_v':>8} {'pct_v':>7} {'beats':>6}",
            flush=True,
        )
        for s in all_summaries:
            print(
                f"{s['problem_id']:<28} {s['tag']:<8} "
                f"{s['best_score']:>8.4f} "
                f"{s['mean_score']:>8.4f} "
                f"{s['mean_score_when_valid']:>8.4f} "
                f"{s['pct_valid']:>7.2f} "
                f"{s['num_beat_sota']:>6}",
                flush=True,
            )
        return compare

    finally:
        print("[vllm] terminating server subprocess...", flush=True)
        vllm_proc.terminate()
        try:
            vllm_proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            vllm_proc.kill()
        try:
            HF_CACHE_VOL.commit()
        except Exception:
            pass


@app.local_entrypoint()
def main(
    problems: str,
    run_name: str,
    n_samples: int = 40,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: int = 16384,
    adapter_name: str | None = None,
    base_model: str = BASE_MODEL_DEFAULT,
    seed: int = 42,
    eval_base: bool = True,
    eval_adapter: bool = True,
    eval_timeout: int = 360,
):
    """Best-of-N eval on AlphaEvolve math problems on Modal.

    Parameters
    ----------
    problems      : comma-separated problem ids (e.g.
                    ``circle_packing_rect,hexagon_packing/11``).
    run_name      : output subdir under ``atlas-alphaevolve-outputs/eval/``.
    n_samples     : completions per (problem, model) leg (default 40).
    adapter_name  : if set, also evaluate the LoRA adapter ``atlas-models/<adapter_name>``.
    eval_base     : if True, evaluate the base model on each problem.
    eval_adapter  : if True (and adapter_name is set), evaluate the adapter.
    eval_timeout  : per-sample evaluator subprocess timeout (seconds).
    """
    problem_list = [p.strip() for p in problems.split(",") if p.strip()]
    assert problem_list, f"No problems parsed from {problems!r}"
    print(
        f"[local] best-of-{n_samples} sweep: problems={problem_list} "
        f"run_name={run_name} base={eval_base} atlas={eval_adapter} "
        f"adapter={adapter_name}"
    )
    compare = run_eval_sweep.remote(
        problems=problem_list,
        n_samples=n_samples,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        adapter_name=adapter_name,
        run_name=run_name,
        base_model=base_model,
        seed=seed,
        eval_base=eval_base,
        eval_adapter=eval_adapter,
        eval_timeout=eval_timeout,
    )
    print("\n=== COMPARE ===")
    for leg in compare["legs"]:
        print(
            f"  {leg['problem_id']:<28} {leg.get('tag', '?'):<8} "
            f"best={leg['best_score']:.4f} mean={leg['mean_score']:.4f} "
            f"pct_valid={leg['pct_valid']:.2f}"
        )
    print(
        f"\nDownload artifacts with:\n"
        f"  modal volume get atlas-alphaevolve-outputs eval/{run_name} ./eval_runs/"
    )
