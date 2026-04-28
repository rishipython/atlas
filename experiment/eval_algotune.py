"""Standalone (no-OpenEvolve) pass@k eval for algotune problems.

Parallels ``eval_standalone.py`` but targets the algotune task family
(CPU numpy/scipy, no Triton).  Evaluates a base model and (optionally)
an ATLAS LoRA adapter on any subset of algotune problems, with ``n``
samples per (problem, leg) and writes per-sample outputs + a summary
JSON to the outputs Volume.

Typical usage::

    modal run experiment/eval_algotune.py \\
        --problems fft_convolution,convolve2d_full_fill,affine_transform_2d \\
        --adapter-name atlas_algotune_signal_v1 \\
        --run-name atlas_algotune_v1 \\
        --n-samples 40
"""
from __future__ import annotations

import json
import os
import pickle
import re
import subprocess
import sys
import tempfile
import time
import traceback
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
        "orjson>=3.10",
        "toml>=0.10",
        "pyaml>=24.0",
        "pillow>=10.0",
        "numpy>=1.26",
        "scipy>=1.11",
        "numba>=0.60",
        # Modal runtime imports google.protobuf via modal_proto.
        "protobuf>=4.25",
        # Modal runtime also imports grpclib.client.
        "grpclib>=0.4.7",
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

app = modal.App("atlas-eval-algotune", image=image)


# ---------------------------------------------------------------------------
# vLLM subprocess launcher (identical to eval_standalone.py).
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


# ---------------------------------------------------------------------------
# Batched concurrent generation (copied from eval_standalone; identical
# semantics — true parallelism via HTTP, cascading reasoning_effort
# fallback for empty final-channel content).
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
    cascade: list[tuple[str | None, str]] = [
        (None, "default"),
        ("medium", "medium"),
        ("low", "low"),
    ]
    results: list[tuple[str, str | None, str]] = [("", None, "default")] * n_samples
    pending_idx = list(range(n_samples))

    for pass_idx, (effort, tag) in enumerate(cascade):
        if not pending_idx:
            break
        print(
            f"      [batch] pass {pass_idx + 1}/{len(cascade)} effort={tag} "
            f"rolling {len(pending_idx)} completions",
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
            seed=seed + pass_idx * 7_777_777,
            reasoning_effort=effort,
            batch_size=batch_size,
            label=f"pass{pass_idx + 1}",
        )
        next_pending = []
        for slot, (content, reasoning) in zip(pending_idx, draws):
            results[slot] = (content, reasoning, tag)
            if not content.strip():
                next_pending.append(slot)
        pending_idx = next_pending

    if pending_idx:
        print(
            f"      [batch] {len(pending_idx)} slots remained empty after full cascade",
            flush=True,
        )
    return results


# ---------------------------------------------------------------------------
# Optional RLM-style memory retrieval (search once per problem; reuse for all
# samples/legs for that problem).
# ---------------------------------------------------------------------------
_RLM_ENTRY_RE = re.compile(
    r"ENTRY\s+(\d+)\s+\[([^\]]*)\]\s*\n=+\s*\n(.*?)(?=\n={3,}\nENTRY\s+\d+|\Z)",
    re.DOTALL,
)
_RLM_INDEX_LIST_RE = re.compile(r"\[\s*(\d+(?:\s*,\s*\d+)*)\s*\]")


def _rlm_parse_entries(memory_bank_text: str) -> list[dict]:
    entries: list[dict] = []
    for m in _RLM_ENTRY_RE.finditer(memory_bank_text):
        entries.append(
            {
                "idx": int(m.group(1)),
                "tag": m.group(2).strip(),
                "body": m.group(3).strip(),
            }
        )
    return entries


def _rlm_parse_indices(response: str, max_idx: int, k: int) -> list[int]:
    hits: list[int] = []
    for m in _RLM_INDEX_LIST_RE.finditer(response):
        raw = m.group(1)
        try:
            nums = [int(x.strip()) for x in raw.split(",")]
        except ValueError:
            continue
        for n in nums:
            if 1 <= n <= max_idx and n not in hits:
                hits.append(n)
        if hits:
            break
    if not hits:
        hits = list(range(1, min(k, max_idx) + 1))
    return hits[:k]


def _rlm_build_search_messages(
    memory_bank_text: str, problem_system: str, problem_user: str, top_k: int
) -> list[dict]:
    system = (
        "You retrieve useful lessons from a memory bank of past algorithmic "
        "code attempts. Return ONLY a JSON array of the most useful entry "
        f"indices (exactly {top_k}) for solving the new task correctly and fast."
    )
    user = (
        f"# Memory bank\n{memory_bank_text}\n\n"
        f"# New task system prompt\n{problem_system[:2500]}\n\n"
        f"# New task user prompt\n{problem_user}\n\n"
        f"Return JSON array with top-{top_k} entry indices."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _rlm_build_augmented_system(
    base_system: str, entries: list[dict], selected_idx: list[int]
) -> str:
    selected = [e for e in entries if e["idx"] in selected_idx]
    if not selected:
        return base_system
    header = (
        "\n\n---\n## Memory: Lessons from prior optimization attempts\n"
        "Use these as guidance to avoid known pitfalls and prefer working designs.\n"
    )
    chunks = [header]
    for e in selected:
        chunks.append(f"\n### Memory entry {e['idx']} [{e['tag']}]\n{e['body']}\n")
    return base_system + "".join(chunks)


# ---------------------------------------------------------------------------
# Evaluator: writes candidate code to a .py file and runs the task's
# evaluator.evaluate(program_path) from the algotune task registry.
# ---------------------------------------------------------------------------
def _score_candidate(problem_id: str, candidate_code: str, timeout_s: int = 300) -> dict:
    """Score one candidate program.

    Strategy:
      1. Dump the task evaluator source to a temp file.
      2. Dump the candidate code to a temp .py file.
      3. Run ``python -c "import <eval>; print(json.dumps(eval.evaluate(prog)))"``
         in a subprocess so any candidate segfault/hang doesn't kill us.
    """
    sys.path.insert(0, "/atlas")
    from experiment.tasks import get_task  # noqa: E402

    spec = get_task("algotune", problem_id)

    with tempfile.TemporaryDirectory(prefix=f"algotune_eval_{problem_id}_") as tmpdir:
        tmp = Path(tmpdir)
        candidate_path = tmp / "candidate.py"
        candidate_path.write_text(candidate_code or "# empty\n")
        # Replace the openevolve import with a local shim so we don't have
        # to install openevolve just for a one-line data class.
        evaluator_src = spec.evaluator.replace(
            "from openevolve.evaluation_result import EvaluationResult",
            (
                "class EvaluationResult:\n"
                "    def __init__(self, metrics=None, artifacts=None):\n"
                "        self.metrics = metrics or {}\n"
                "        self.artifacts = artifacts or {}"
            ),
        )
        evaluator_path = tmp / "evaluator.py"
        evaluator_path.write_text(evaluator_src)
        # Ensure upstream evaluator picks up per-problem AlgoTune settings
        # (e.g., data_size=5 for convolve2d) instead of hardcoded fallbacks.
        config_src = (
            REPO_ROOT
            / "experiment"
            / "tasks"
            / "_oe_problems"
            / "algotune_examples"
            / problem_id
            / "config.yaml"
        )
        if config_src.exists():
            (tmp / "config.yaml").write_text(config_src.read_text())

        # Harness runs evaluate() and prints a compact JSON to stdout.
        harness = (
            "import importlib.util, json, sys\n"
            f"spec = importlib.util.spec_from_file_location('evmod', {str(evaluator_path)!r})\n"
            "mod = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(mod)\n"
            f"res = mod.evaluate({str(candidate_path)!r})\n"
            "if hasattr(res, 'metrics') and hasattr(res, 'artifacts'):\n"
            "    out = {'metrics': dict(res.metrics), 'artifacts': dict(res.artifacts)}\n"
            "elif isinstance(res, dict):\n"
            "    if 'metrics' in res or 'artifacts' in res:\n"
            "        out = {\n"
            "            'metrics': dict(res.get('metrics') or {}),\n"
            "            'artifacts': dict(res.get('artifacts') or {}),\n"
            "        }\n"
            "    else:\n"
            "        # Some evaluators return a flat metrics dict directly.\n"
            "        out = {'metrics': dict(res), 'artifacts': {}}\n"
            "else:\n"
            "    out = {'metrics': {}, 'artifacts': {'error': f'unexpected eval return type: {type(res)}'}}\n"
            "print('__OE_RESULT__' + json.dumps(out))\n"
        )
        harness_path = tmp / "harness.py"
        harness_path.write_text(harness)

        env = os.environ.copy()
        env["PYTHONPATH"] = (
            "/atlas/AlgoTune"
            + os.pathsep
            + "/atlas"
            + os.pathsep
            + env.get("PYTHONPATH", "")
        )
        try:
            proc = subprocess.run(
                [sys.executable, str(harness_path)],
                capture_output=True,
                text=True,
                timeout=timeout_s,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return {
                "correct": False,
                "score": 0.0,
                "speedup": 0.0,
                "combined_score": 0.0,
                "feedback": f"harness timeout after {timeout_s}s",
            }

        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        for line in stdout.splitlines():
            if line.startswith("__OE_RESULT__"):
                try:
                    data = json.loads(line[len("__OE_RESULT__") :])
                    metrics = data.get("metrics") or {}
                    artifacts = data.get("artifacts") or {}
                    return {
                        "correct": bool(metrics.get("correctness", 0.0) >= 0.99),
                        "score": float(metrics.get("speedup", 0.0)),
                        "speedup": float(metrics.get("speedup", 0.0)),
                        "combined_score": float(metrics.get("combined_score", 0.0)),
                        "feedback": str(artifacts.get("feedback", ""))[:4000],
                    }
                except Exception as exc:  # noqa: BLE001
                    return {
                        "correct": False,
                        "score": 0.0,
                        "speedup": 0.0,
                        "combined_score": 0.0,
                        "feedback": f"parse error: {exc}; line[:200]={line[:200]!r}",
                    }
        return {
            "correct": False,
            "score": 0.0,
            "speedup": 0.0,
            "combined_score": 0.0,
            "feedback": f"no result line; exit={proc.returncode}; "
            f"stdout={stdout[:800]!r}; stderr={stderr[:1200]!r}",
        }


# ---------------------------------------------------------------------------
# Per-(problem, leg) sample + score loop.
# ---------------------------------------------------------------------------
def _sample_and_score(
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
    batch_size: int = 32,
    harness_timeout_s: int = 300,
    resume: bool = True,
    checkpoint_every: int = 1,
    rescore_only: bool = False,
) -> tuple[list[dict], dict]:
    sys.path.insert(0, "/atlas")
    from utils.extract import extract_code  # noqa: E402

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "system_message.txt").write_text(system_msg)
    (output_dir / "user_message.txt").write_text(user_msg)

    samples_path = output_dir / "samples.json"
    partial_samples_path = output_dir / "samples.partial.json"
    partial_summary_path = output_dir / "summary.partial.json"

    samples: list[dict] = []
    if resume:
        for p in (samples_path, partial_samples_path):
            if p.exists():
                try:
                    loaded = json.loads(p.read_text())
                    if isinstance(loaded, list):
                        samples = loaded
                        print(
                            f"      [resume] loaded {len(samples)} existing samples from {p.name}",
                            flush=True,
                        )
                        break
                except Exception as exc:  # noqa: BLE001
                    print(f"      [resume] failed to load {p.name}: {exc}", flush=True)

    done_already = len(samples)
    if done_already > n_samples:
        print(
            f"      [resume] existing samples ({done_already}) exceed target ({n_samples}); truncating",
            flush=True,
        )
        samples = samples[:n_samples]
        done_already = n_samples

    remaining = n_samples - done_already
    if rescore_only:
        if not samples:
            print("      [rescore] no existing samples to rescore; skipping leg", flush=True)
        else:
            print(f"      [rescore] rescoring {len(samples)} existing samples", flush=True)
            for i, row in enumerate(samples):
                code_path = output_dir / f"sample_{i:03d}_code.py"
                code = code_path.read_text() if code_path.exists() else ""
                t0 = time.time()
                result = _score_candidate(problem_id, code, timeout_s=harness_timeout_s)
                score_dt = time.time() - t0
                row.update(
                    {
                        "score_time_s": round(score_dt, 2),
                        "correct": result.get("correct", False),
                        "score": float(result.get("score", 0.0)),
                        "speedup": float(result.get("speedup", 0.0)),
                        "combined_score": float(result.get("combined_score", 0.0)),
                        "feedback": str(result.get("feedback", ""))[:4000],
                    }
                )
                if checkpoint_every <= 0:
                    checkpoint_every = 1
                if ((i + 1) % checkpoint_every == 0) or ((i + 1) == len(samples)):
                    correct_partial = [s for s in samples if s.get("correct")]
                    partial_summary = {
                        "problem_id": problem_id,
                        "served_model": served_model,
                        "n_samples": n_samples,
                        "completed_samples": len(samples),
                        "temperature": temperature,
                        "pass_at_1": 1.0 if samples and samples[0].get("correct") else 0.0,
                        "pass_at_k": len(correct_partial) / max(1, len(samples)),
                        "num_correct": len(correct_partial),
                        "best_speedup_when_correct": max(
                            (float(s.get("speedup", 0.0)) for s in correct_partial), default=0.0
                        ),
                    }
                    partial_samples_path.write_text(json.dumps(samples, indent=2))
                    partial_summary_path.write_text(json.dumps(partial_summary, indent=2))
                    OUTPUTS_VOL.commit()

            correct = [s for s in samples if s.get("correct")]
            summary = {
                "problem_id": problem_id,
                "served_model": served_model,
                "n_samples": n_samples,
                "temperature": temperature,
                "pass_at_1": 1.0 if samples and samples[0].get("correct") else 0.0,
                "pass_at_k": len(correct) / max(1, n_samples),
                "num_correct": len(correct),
                "best_speedup_when_correct": max(
                    (float(s.get("speedup", 0.0)) for s in correct), default=0.0
                ),
                "mean_speedup_when_correct": (
                    sum(float(s.get("speedup", 0.0)) for s in correct) / len(correct)
                    if correct
                    else 0.0
                ),
                "mean_combined_when_correct": (
                    sum(float(s.get("combined_score", 0.0)) for s in correct) / len(correct)
                    if correct
                    else 0.0
                ),
                "completed_samples": len(samples),
                "rescored_only": True,
            }
            (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
            samples_path.write_text(json.dumps(samples, indent=2))
            OUTPUTS_VOL.commit()
            return samples, summary

    if remaining <= 0:
        print("      [resume] leg already complete; recomputing summary only", flush=True)
        correct = [s for s in samples if s.get("correct")]
        summary = {
            "problem_id": problem_id,
            "served_model": served_model,
            "n_samples": n_samples,
            "temperature": temperature,
            "pass_at_1": 1.0 if samples and samples[0].get("correct") else 0.0,
            "pass_at_k": len(correct) / max(1, n_samples),
            "num_correct": len(correct),
            "best_speedup_when_correct": max(
                (float(s.get("speedup", 0.0)) for s in correct), default=0.0
            ),
            "mean_speedup_when_correct": (
                sum(float(s.get("speedup", 0.0)) for s in correct) / len(correct)
                if correct
                else 0.0
            ),
            "mean_combined_when_correct": (
                sum(float(s.get("combined_score", 0.0)) for s in correct) / len(correct)
                if correct
                else 0.0
            ),
        }
        summary["completed_samples"] = len(samples)
        samples_path.write_text(json.dumps(samples, indent=2))
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        OUTPUTS_VOL.commit()
        return samples, summary

    gen_t0 = time.time()
    generations = _batched_generate(
        client,
        served_model=served_model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        n_samples=remaining,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        seed=seed + done_already,
        batch_size=batch_size,
    )
    total_gen_s = time.time() - gen_t0
    print(
        f"      [gen] {remaining} completions in {total_gen_s:.1f}s "
        f"({remaining / max(total_gen_s, 1e-6):.2f} seq/s)",
        flush=True,
    )

    for offset, (raw, reasoning, winning_effort) in enumerate(generations):
        i = done_already + offset
        try:
            code = extract_code(raw)
            t0 = time.time()
            result = _score_candidate(problem_id, code, timeout_s=harness_timeout_s)
            score_dt = time.time() - t0
        except Exception as exc:  # noqa: BLE001
            code = ""
            score_dt = 0.0
            result = {
                "correct": False,
                "score": 0.0,
                "speedup": 0.0,
                "combined_score": 0.0,
                "feedback": f"sample loop exception: {exc}\n{traceback.format_exc()[:2000]}",
            }

        sample_row = {
            "sample_idx": i,
            "raw_length": len(raw),
            "code_length": len(code),
            "reasoning_length": len(reasoning) if reasoning else 0,
            "winning_reasoning_effort": winning_effort,
            "score_time_s": round(score_dt, 2),
            **result,
        }
        samples.append(sample_row)

        (output_dir / f"sample_{i:03d}_raw.txt").write_text(raw)
        (output_dir / f"sample_{i:03d}_code.py").write_text(code)
        if reasoning:
            (output_dir / f"sample_{i:03d}_reasoning.txt").write_text(reasoning)
        print(
            f"    [{served_model}/{problem_id}] [{i + 1}/{n_samples}] "
            f"correct={result['correct']} speedup={result['speedup']:.2f}x "
            f"combined={result['combined_score']:.3f} eval={score_dt:.1f}s",
            flush=True,
        )
        if checkpoint_every <= 0:
            checkpoint_every = 1
        if ((i + 1) % checkpoint_every == 0) or ((i + 1) == n_samples):
            correct_partial = [s for s in samples if s.get("correct")]
            partial_summary = {
                "problem_id": problem_id,
                "served_model": served_model,
                "n_samples": n_samples,
                "completed_samples": len(samples),
                "temperature": temperature,
                "pass_at_1": 1.0 if samples and samples[0].get("correct") else 0.0,
                "pass_at_k": len(correct_partial) / max(1, len(samples)),
                "num_correct": len(correct_partial),
                "best_speedup_when_correct": max(
                    (float(s.get("speedup", 0.0)) for s in correct_partial), default=0.0
                ),
            }
            partial_samples_path.write_text(json.dumps(samples, indent=2))
            partial_summary_path.write_text(json.dumps(partial_summary, indent=2))
            OUTPUTS_VOL.commit()

    correct = [s for s in samples if s["correct"]]
    summary = {
        "problem_id": problem_id,
        "served_model": served_model,
        "n_samples": n_samples,
        "temperature": temperature,
        "pass_at_1": 1.0 if samples and samples[0]["correct"] else 0.0,
        "pass_at_k": len(correct) / max(1, n_samples),
        "num_correct": len(correct),
        "best_speedup_when_correct": max(
            (s["speedup"] for s in correct), default=0.0
        ),
        "mean_speedup_when_correct": (
            sum(s["speedup"] for s in correct) / len(correct) if correct else 0.0
        ),
        "mean_combined_when_correct": (
            sum(s["combined_score"] for s in correct) / len(correct)
            if correct
            else 0.0
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    samples_path.write_text(json.dumps(samples, indent=2))
    OUTPUTS_VOL.commit()
    return samples, summary


# ---------------------------------------------------------------------------
# Modal remote: sweep over (problems, legs).
# ---------------------------------------------------------------------------
@app.function(
    gpu="A100-80GB",
    timeout=3 * 3600,
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
    batch_size: int,
    harness_timeout_s: int = 300,
    resume: bool = True,
    checkpoint_every: int = 1,
    rescore_only: bool = False,
    rlm_memory_bank_text: str | None = None,
    rlm_top_k: int = 3,
) -> dict:
    import openai

    sys.path.insert(0, "/atlas")
    from experiment.algotune_prompts import build_algotune_prompts  # noqa: E402
    from experiment.tasks import get_task  # noqa: E402

    for pid in problems:
        # Validate that this problem is known to the algotune family.
        get_task("algotune", pid)

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

        legs: list[tuple[str, str]] = []
        if eval_base:
            legs.append((base_model, "base"))
        if eval_adapter and adapter_path:
            legs.append(("atlas", "atlas"))
        assert legs, "No legs to evaluate — set eval_base or eval_adapter + adapter"

        all_summaries: list[dict] = []
        base_dir = Path("/outputs") / "eval" / run_name
        base_dir.mkdir(parents=True, exist_ok=True)
        compare_partial_path = base_dir / "compare.partial.json"

        for pid in problems:
            system_msg, user_msg, _ep = build_algotune_prompts(pid)
            selected_idx: list[int] = []
            if rlm_memory_bank_text:
                try:
                    entries = _rlm_parse_entries(rlm_memory_bank_text)
                    if entries:
                        search_resp = client.chat.completions.create(
                            model=base_model,
                            messages=_rlm_build_search_messages(
                                memory_bank_text=rlm_memory_bank_text,
                                problem_system=system_msg,
                                problem_user=user_msg,
                                top_k=rlm_top_k,
                            ),
                            temperature=0.0,
                            top_p=1.0,
                            max_tokens=256,
                            seed=seed,
                        )
                        search_text = (
                            search_resp.choices[0].message.content or ""
                        )
                        selected_idx = _rlm_parse_indices(
                            search_text, max_idx=max(e["idx"] for e in entries), k=rlm_top_k
                        )
                        system_msg = _rlm_build_augmented_system(
                            base_system=system_msg,
                            entries=entries,
                            selected_idx=selected_idx,
                        )
                        print(
                            f"[rlm] problem={pid} selected_entries={selected_idx}",
                            flush=True,
                        )
                except Exception as exc:  # noqa: BLE001
                    print(f"[rlm] retrieval failed for {pid}: {exc}", flush=True)
            for served, tag in legs:
                leg_dir = base_dir / f"{pid}__{tag}"
                print(
                    f"\n[eval] === leg: problem={pid} model={tag} (served='{served}') ===",
                    flush=True,
                )
                try:
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
                        batch_size=batch_size,
                        harness_timeout_s=harness_timeout_s,
                        resume=resume,
                        checkpoint_every=checkpoint_every,
                        rescore_only=rescore_only,
                    )
                    summary["tag"] = tag
                    all_summaries.append(summary)
                except Exception as exc:  # noqa: BLE001
                    err = {
                        "problem_id": pid,
                        "tag": tag,
                        "served_model": served,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc()[:4000],
                        "pass_at_1": 0.0,
                        "pass_at_k": 0.0,
                        "num_correct": 0,
                        "best_speedup_when_correct": 0.0,
                        "mean_speedup_when_correct": 0.0,
                        "mean_combined_when_correct": 0.0,
                    }
                    all_summaries.append(err)
                    (leg_dir / "error.json").write_text(json.dumps(err, indent=2))
                    print(
                        f"[eval] leg failed but continuing: {pid} {tag} :: {exc}",
                        flush=True,
                    )
                compare_partial = {
                    "run_name": run_name,
                    "task_family": "algotune",
                    "base_model": base_model,
                    "adapter_name": adapter_name,
                    "n_samples": n_samples,
                    "temperature": temperature,
                    "legs": all_summaries,
                    "partial": True,
                    "rlm_enabled": bool(rlm_memory_bank_text),
                    "rlm_top_k": rlm_top_k if rlm_memory_bank_text else None,
                }
                compare_partial_path.write_text(json.dumps(compare_partial, indent=2))
                OUTPUTS_VOL.commit()

        compare = {
            "run_name": run_name,
            "task_family": "algotune",
            "base_model": base_model,
            "adapter_name": adapter_name,
            "n_samples": n_samples,
            "temperature": temperature,
            "legs": all_summaries,
            "rlm_enabled": bool(rlm_memory_bank_text),
            "rlm_top_k": rlm_top_k if rlm_memory_bank_text else None,
        }
        (base_dir / "compare.json").write_text(json.dumps(compare, indent=2))
        OUTPUTS_VOL.commit()

        print("\n=== CROSS-CONDITION COMPARISON ===", flush=True)
        print(
            f"{'problem':<22} {'leg':<8} {'pass@1':>7} {'pass@k':>7} "
            f"{'n_corr':>7} {'best_sp':>10} {'mean_sp':>10}",
            flush=True,
        )
        for s in all_summaries:
            print(
                f"{s['problem_id']:<22} {s['tag']:<8} {s['pass_at_1']:>7.2f} "
                f"{s['pass_at_k']:>7.2f} {s['num_correct']:>7} "
                f"{s['best_speedup_when_correct']:>10.2f} "
                f"{s['mean_speedup_when_correct']:>10.2f}",
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
    max_tokens: int = 8192,
    adapter_name: str | None = None,
    base_model: str = BASE_MODEL_DEFAULT,
    seed: int = 42,
    eval_base: bool = True,
    eval_adapter: bool = True,
    batch_size: int = 32,
    harness_timeout_s: int = 300,
    resume: bool = True,
    checkpoint_every: int = 1,
    rescore_only: bool = False,
    rlm_memory_bank: str | None = None,
    rlm_top_k: int = 3,
):
    """Head-to-head pass@k eval: base gpt-oss-20b vs ATLAS(LoRA) on algotune
    problems, in a single container sharing one vLLM server.
    """
    problem_list = [p.strip() for p in problems.split(",") if p.strip()]
    assert problem_list, f"No problems parsed from {problems!r}"
    print(
        f"[local] sweep: problems={problem_list} run_name={run_name} "
        f"base={eval_base} atlas={eval_adapter} adapter={adapter_name}"
    )
    rlm_memory_bank_text = None
    if rlm_memory_bank:
        p = Path(rlm_memory_bank)
        assert p.exists(), f"RLM memory bank not found: {p}"
        rlm_memory_bank_text = p.read_text()

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
        batch_size=batch_size,
        harness_timeout_s=harness_timeout_s,
        resume=resume,
        checkpoint_every=checkpoint_every,
        rescore_only=rescore_only,
        rlm_memory_bank_text=rlm_memory_bank_text,
        rlm_top_k=rlm_top_k,
    )
    print("\n=== COMPARE ===")
    for leg in compare["legs"]:
        print(
            f"  {leg['problem_id']:<22} {leg.get('tag', '?'):<8} "
            f"pass@1={leg['pass_at_1']:.2f} pass@k={leg['pass_at_k']:.2f} "
            f"best_speedup={leg['best_speedup_when_correct']:.2f}"
        )
    print(
        f"\nDownload artifacts with:\n"
        f"  modal volume get atlas-openevolve-outputs eval/{run_name} ./eval_runs/"
    )
