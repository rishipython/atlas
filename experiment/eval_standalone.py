"""Standalone (no-OpenEvolve) evaluation of a model on kernel problems.

Purpose
-------
Compare ``base gpt-oss-20b`` to ``ATLAS gpt-oss-20b + LoRA`` on
kernel problems **without** any evolutionary loop — just straight
sampling + correctness/speedup measurement. This is the protocol for
the final comparison the user asked for: base vs ATLAS on layernorm
and matmul (and optionally softmax, to sanity-check in-distribution
behavior).

Protocol
--------
A single container boots vLLM once with ``--enable-lora`` so the base
model and the ATLAS LoRA adapter are both served side-by-side (selected
at request time via the ``model`` field: base HF id vs. the adapter
alias). Then, for each (problem_id, model) pair we requested, we draw
``--n-samples`` completions, extract the fenced ``python`` block, and
score each via ``evaluate_kernel`` in a subprocess.

Per-sample results and one aggregate ``summary.json`` are persisted to
``/outputs/eval/<run_name>/<problem>__<model>/``. A cross-condition
``compare.json`` (pass@1, pass@k, best/mean speedup-when-correct for
each leg) is written under ``/outputs/eval/<run_name>/``.

Typical run (the one we actually use for the head-to-head)
----------------------------------------------------------
    modal run experiment/eval_standalone.py \\
        --problems layernorm,matmul \\
        --adapter-name atlas_softmax_v1 \\
        --run-name atlas_vs_base_v1 \\
        --n-samples 8

This evaluates the base gpt-oss-20b AND the ATLAS adapter on BOTH
problems in one container — so vLLM startup / model download is paid
only once.
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

HF_CACHE_VOL = modal.Volume.from_name("atlas-hf-cache", create_if_missing=True)
MODELS_VOL = modal.Volume.from_name("atlas-models", create_if_missing=True)
OUTPUTS_VOL = modal.Volume.from_name(
    "atlas-openevolve-outputs", create_if_missing=True
)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl", "build-essential")
    .pip_install(
        "vllm==0.19.1",
        "openai>=1.50",
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

app = modal.App("atlas-eval-standalone", image=image)


# --------------------------------------------------------------------------
# vLLM subprocess management (copy of the pattern in openevolve_runner.py)
# --------------------------------------------------------------------------
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
    """Launch vLLM.

    If ``adapter_path`` is set, we register the LoRA adapter via
    ``--lora-modules`` so the adapter is served under the name
    ``atlas``; otherwise we serve just the base model.  We also enable
    the Harmony reasoning parser so ``reasoning_effort`` control works
    server-side and ``reasoning_content`` is exposed separately from
    the final-channel ``content`` — matching ``openevolve_runner.py``.
    """
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        base_model,
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


# --------------------------------------------------------------------------
# Prompt construction (intentionally *exactly* what OpenEvolve used to
# sample from gpt-oss during trajectory collection — so the base-vs-ATLAS
# comparison reflects behavior the adapter was trained to improve).
# --------------------------------------------------------------------------
def _build_standalone_prompts(problem_id: str) -> tuple[str, str]:
    """Return (system_message, user_message) for a straight generation pass."""
    sys.path.insert(0, "/atlas")
    from agent.prompts import TRITON_SYSTEM_PROMPT
    from benchmark.kernel.problems import KERNEL_PROBLEMS

    kp = KERNEL_PROBLEMS[problem_id]

    system = TRITON_SYSTEM_PROMPT  # keeps Response-format rules, unlike OpenEvolve

    user = (
        f"Problem: {kp.description}\n\n"
        f"Write a complete, runnable Python file that defines a function "
        f"`{kp.entry_point}(...)` with the same signature as the reference "
        f"below. It must produce output numerically close to the reference "
        f"(atol={kp.atol}, rtol={kp.rtol}) on all of these test shapes: "
        f"{kp.test_shapes}.\n\n"
        f"Reference implementation (correct but slow — replace its body with a "
        f"Triton kernel):\n```python\n{kp.reference_code}\n```\n\n"
        f"Return a single ```python fenced block containing the full "
        f"solution."
    )
    return system, user


# --------------------------------------------------------------------------
# Main eval function: one container, many (problem, model) conditions.
# --------------------------------------------------------------------------
def _chat_with_twophase_fallback(
    client,
    served_model: str,
    messages: list[dict],
    temperature: float,
    top_p: float,
    max_tokens: int,
    seed: int,
) -> tuple[str, str | None, str]:
    """Run a chat completion with cascading ``reasoning_effort`` retry.

    gpt-oss at ``reasoning_effort="high"`` (the server default) often
    spends its whole output budget inside the Harmony analysis channel
    and emits nothing to the final channel — we'd score a blank program
    0.0 and falsely conclude the model "can't do it".  This mirrors the
    fallback we installed in ``sitecustomize.py`` for OpenEvolve: try
    the requested effort, then medium, then low.  Returns the final
    content (possibly empty after exhausting the cascade), the
    ``reasoning_content`` from the successful attempt (or None), and
    the reasoning_effort that eventually succeeded.
    """
    cascade: list[str | None] = [None, "medium", "low"]  # None = server default
    last_content = ""
    last_reasoning: str | None = None
    winning_effort = "default"

    for idx, effort in enumerate(cascade):
        params = dict(
            model=served_model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=seed,
        )
        if effort is not None:
            # OpenAI-Python uses ``extra_body`` for fields not in the
            # stable schema, and vLLM forwards them to the server.
            params["extra_body"] = {"reasoning_effort": effort}
        resp = client.chat.completions.create(**params)
        msg = resp.choices[0].message
        content = msg.content or ""
        # vLLM 0.19 renamed this field from ``reasoning_content`` to
        # ``reasoning`` — check both (mirrors ``sitecustomize.py``).
        reasoning = getattr(msg, "reasoning", None) or getattr(
            msg, "reasoning_content", None
        )

        if content.strip():
            if idx > 0:
                print(
                    f"      [two-phase] recovered on attempt {idx + 1}/{len(cascade)} "
                    f"(reasoning_effort={effort or 'default'})",
                    flush=True,
                )
            return content, reasoning, effort or "default"
        last_content = content
        last_reasoning = reasoning
        winning_effort = effort or "default"
        print(
            f"      [two-phase] attempt {idx + 1}/{len(cascade)} "
            f"(effort={effort or 'default'}) returned empty; cascading",
            flush=True,
        )
    return last_content, last_reasoning, winning_effort


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
    """Fire ``n`` concurrent chat.completions requests (n=1 each) and gather.

    vLLM's internal ``n``-completion batching turns out to serialize sequences
    when using the Harmony reasoning parser, so we get true parallelism by
    firing many independent HTTP requests and letting vLLM's continuous
    batcher schedule them.  ``batch_size`` caps the in-flight concurrency.

    Returns a list of ``(content, reasoning)`` tuples of length ``n``.
    """
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
    """Generate ``n_samples`` completions in parallel using vLLM's ``n`` param.

    Cascade strategy, but batched (not per-sample):
      1. One pass at server-default reasoning_effort.
      2. Re-roll the empties in one batched pass at ``reasoning_effort=medium``.
      3. Re-roll the still-empties in one batched pass at ``reasoning_effort=low``.

    Each pass uses vLLM's ``n``-completion batching so a leg of 100 samples
    typically finishes its generation in a handful of HTTP calls instead of
    ~300 sequential ones.
    """
    cascade: list[tuple[str | None, str]] = [
        (None, "default"),
        ("medium", "medium"),
        ("low", "low"),
    ]
    # (content, reasoning, winning_effort), initialized empty
    results: list[tuple[str, str | None, str]] = [("", None, "default")] * n_samples
    pending_idx = list(range(n_samples))

    for pass_idx, (effort, tag) in enumerate(cascade):
        if not pending_idx:
            break
        if pass_idx == 0:
            print(
                f"      [batch] pass {pass_idx + 1}/{len(cascade)} "
                f"effort={tag} rolling {len(pending_idx)} completions",
                flush=True,
            )
        else:
            print(
                f"      [batch] pass {pass_idx + 1}/{len(cascade)} effort={tag} "
                f"re-rolling {len(pending_idx)} still-empty slots",
                flush=True,
            )
        draws = _batched_call(
            client=client,
            served_model=served_model,
            messages=messages,
            n=len(pending_idx),
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            # Different seed base per pass so we don't re-sample identical
            # empty-producing paths.
            seed=seed + pass_idx * 7_777_777,
            reasoning_effort=effort,
            batch_size=batch_size,
            label=f"pass{pass_idx + 1}",
        )
        next_pending = []
        for slot, (content, reasoning) in zip(pending_idx, draws):
            if content.strip():
                results[slot] = (content, reasoning, tag)
            else:
                results[slot] = (content, reasoning, tag)
                next_pending.append(slot)
        pending_idx = next_pending

    if pending_idx:
        print(
            f"      [batch] {len(pending_idx)} slots remained empty after full cascade",
            flush=True,
        )
    return results


def _sample_and_score(
    client,
    served_model: str,
    problem,
    system_msg: str,
    user_msg: str,
    n_samples: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    seed: int,
    output_dir: "Path",
    extract_code,
    evaluate_kernel,
    outputs_vol,
    batch_size: int = 32,
) -> tuple[list[dict], dict]:
    """Generate n_samples from (served_model) for (problem) and score."""
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
        f"      [batch] generated {n_samples} completions total in {total_gen_s:.1f}s "
        f"({n_samples / total_gen_s:.2f} seq/s)",
        flush=True,
    )

    for i, (raw, reasoning, winning_effort) in enumerate(generations):
        code = extract_code(raw)
        gen_time = total_gen_s / n_samples  # amortized; eval dominates anyway

        result = evaluate_kernel(problem, code, timeout=180)
        samples.append(
            {
                "sample_idx": i,
                "raw_length": len(raw),
                "code_length": len(code),
                "reasoning_length": len(reasoning) if reasoning else 0,
                "winning_reasoning_effort": winning_effort,
                "gen_time_s": round(gen_time, 2),
                "correct": result.correct,
                "score": float(result.score),
                "feedback": result.feedback[:4000],
            }
        )
        (output_dir / f"sample_{i:03d}_raw.txt").write_text(raw)
        (output_dir / f"sample_{i:03d}_code.py").write_text(code)
        if reasoning:
            (output_dir / f"sample_{i:03d}_reasoning.txt").write_text(reasoning)
        print(
            f"    [{served_model}/{problem.problem_id}] [{i + 1}/{n_samples}] "
            f"correct={result.correct} score={result.score:.3f} gen={gen_time:.1f}s",
            flush=True,
        )
        # Only commit periodically — 100 commits per leg wastes minutes.
        if (i + 1) % 20 == 0 or (i + 1) == n_samples:
            outputs_vol.commit()

    # Aggregate
    correct = [s for s in samples if s["correct"]]
    summary = {
        "problem_id": problem.problem_id,
        "served_model": served_model,
        "n_samples": n_samples,
        "temperature": temperature,
        "pass_at_1": 1.0 if samples and samples[0]["correct"] else 0.0,
        "pass_at_k": len(correct) / max(1, n_samples),
        "num_correct": len(correct),
        "best_speedup_when_correct": max(
            (s["score"] for s in correct), default=0.0
        ),
        "mean_speedup_when_correct": (
            sum(s["score"] for s in correct) / len(correct) if correct else 0.0
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (output_dir / "samples.json").write_text(json.dumps(samples, indent=2))
    outputs_vol.commit()
    return samples, summary


@app.function(
    gpu="A100-80GB",
    timeout=2 * 3600,
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
) -> dict:
    """Run (problems × {base, adapter}) on one container with one vLLM server."""
    import openai

    sys.path.insert(0, "/atlas")
    from benchmark.kernel.evaluator import evaluate_kernel
    from benchmark.kernel.problems import KERNEL_PROBLEMS
    from utils.extract import extract_code

    for pid in problems:
        assert pid in KERNEL_PROBLEMS, f"Unknown problem_id {pid!r}"

    adapter_path = None
    if adapter_name:
        adapter_path = f"/atlas_models/{adapter_name}"
        assert Path(adapter_path).exists(), (
            f"Adapter directory {adapter_path} not found on atlas-models Volume"
        )
        print(f"[eval] LoRA adapter at {adapter_path}", flush=True)

    vllm_proc = _start_vllm(base_model, adapter_path)

    try:
        client = openai.OpenAI(
            api_key="EMPTY", base_url=f"http://localhost:{VLLM_PORT}/v1"
        )

        legs: list[tuple[str, str]] = []  # (served_model_name, leg_tag)
        if eval_base:
            legs.append((base_model, "base"))
        if eval_adapter and adapter_path:
            legs.append(("atlas", "atlas"))
        assert legs, "No legs to evaluate — set eval_base or eval_adapter + adapter"

        all_summaries: list[dict] = []
        base_dir = Path("/outputs") / "eval" / run_name
        base_dir.mkdir(parents=True, exist_ok=True)

        for pid in problems:
            problem = KERNEL_PROBLEMS[pid]
            system_msg, user_msg = _build_standalone_prompts(pid)
            for served, tag in legs:
                leg_dir = base_dir / f"{pid}__{tag}"
                print(
                    f"\n[eval] === leg: problem={pid} model={tag} (served='{served}') ===",
                    flush=True,
                )
                _, summary = _sample_and_score(
                    client=client,
                    served_model=served,
                    problem=problem,
                    system_msg=system_msg,
                    user_msg=user_msg,
                    n_samples=n_samples,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    seed=seed,
                    output_dir=leg_dir,
                    extract_code=extract_code,
                    evaluate_kernel=evaluate_kernel,
                    outputs_vol=OUTPUTS_VOL,
                )
                summary["tag"] = tag
                all_summaries.append(summary)

        # Cross-condition comparison -------------------------------------
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

        # Pretty-print a compact table for the user.
        print("\n=== CROSS-CONDITION COMPARISON ===", flush=True)
        print(
            f"{'problem':<12} {'leg':<8} {'pass@1':>7} {'pass@k':>7} "
            f"{'n_corr':>7} {'best_sp':>8} {'mean_sp':>8}",
            flush=True,
        )
        for s in all_summaries:
            print(
                f"{s['problem_id']:<12} {s['tag']:<8} {s['pass_at_1']:>7.2f} "
                f"{s['pass_at_k']:>7.2f} {s['num_correct']:>7} "
                f"{s['best_speedup_when_correct']:>8.3f} "
                f"{s['mean_speedup_when_correct']:>8.3f}",
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


# --------------------------------------------------------------------------
# Local entry point
# --------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    problems: str,
    run_name: str,
    n_samples: int = 8,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: int = 8192,
    adapter_name: str | None = None,
    base_model: str = BASE_MODEL_DEFAULT,
    seed: int = 42,
    eval_base: bool = True,
    eval_adapter: bool = True,
):
    """Head-to-head eval: base gpt-oss-20b vs ATLAS(LoRA) over multiple
    kernel problems, in a single container sharing one vLLM server.

    Parameters
    ----------
    problems      : comma-separated problem ids (e.g. ``layernorm,matmul``).
    run_name      : subdir under the outputs volume for all artifacts.
    adapter_name  : subdir name inside the ``atlas-models`` Volume
                    produced by ``train_atlas_lora.py``. Required for
                    ``eval_adapter=True``.
    eval_base     : if True, evaluate the base model on each problem.
    eval_adapter  : if True (and adapter_name is set), evaluate ATLAS.
    """
    problem_list = [p.strip() for p in problems.split(",") if p.strip()]
    assert problem_list, f"No problems parsed from {problems!r}"
    print(
        f"[local] sweep: problems={problem_list} run_name={run_name} "
        f"base={eval_base} atlas={eval_adapter} adapter={adapter_name}"
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
    )
    print("\n=== COMPARE ===")
    for leg in compare["legs"]:
        print(
            f"  {leg['problem_id']:<12} {leg.get('tag', '?'):<8} "
            f"pass@1={leg['pass_at_1']:.2f} pass@k={leg['pass_at_k']:.2f} "
            f"best_speedup={leg['best_speedup_when_correct']:.3f}"
        )
    print(
        f"\nDownload artifacts with:\n"
        f"  modal volume get atlas-openevolve-outputs eval/{run_name} ./eval_runs/"
    )
