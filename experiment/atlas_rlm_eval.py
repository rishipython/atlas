"""ATLAS-RLM: recursive-language-model-style evaluation.

Unlike ATLAS-SFT/DPO, nothing is trained here.  Instead, we distill the
OpenEvolve trajectory bank into a **memory bank** (a numbered text file
produced by ``build_rlm_memory_bank.py``), and at inference time use the
base gpt-oss-20b model itself to:

  1. SEARCH: pick the K most-relevant memory entries for the new task.
  2. GENERATE: augment the system prompt with those K entries, then
     run the usual code-generation pass.

Since selection depends only on the task (not on the random seed), we
do the search call ONCE per problem and reuse the selected entries
across all N samples for that problem.

Usage::

    modal run experiment/atlas_rlm_eval.py \\
        --problems softmax,layernorm,matmul \\
        --memory-bank data/rlm_memory_softmax.txt \\
        --run-name atlas_rlm_v1 \\
        --n-samples 8 \\
        --top-k 3

The memory bank file is uploaded as a Modal secret-like blob via the
local entrypoint (we read it locally and pass its contents).
"""
from __future__ import annotations

import json
import re
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

app = modal.App("atlas-rlm-eval", image=image)


# --------------------------------------------------------------------------
# vLLM process management (same pattern as eval_standalone.py)
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


def _start_vllm(base_model: str) -> subprocess.Popen:
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
        "65536",  # RLM needs large context for memory bank + search
        "--reasoning-parser",
        "openai_gptoss",
    ]
    print(f"[vllm] launching: {' '.join(cmd)}", flush=True)
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


# --------------------------------------------------------------------------
# Memory bank parsing + search-call prompt construction
# --------------------------------------------------------------------------
_ENTRY_RE = re.compile(
    r"ENTRY\s+(\d+)\s+\[([^\]]*)\]\s*\n=+\s*\n(.*?)(?=\n={3,}\nENTRY\s+\d+|\Z)",
    re.DOTALL,
)


def _parse_entries(memory_bank_text: str) -> list[dict]:
    entries: list[dict] = []
    for m in _ENTRY_RE.finditer(memory_bank_text):
        entries.append(
            {
                "idx": int(m.group(1)),
                "tag": m.group(2).strip(),
                "body": m.group(3).strip(),
            }
        )
    return entries


_INDEX_LIST_RE = re.compile(r"\[\s*(\d+(?:\s*,\s*\d+)*)\s*\]")


def _parse_indices(response: str, max_idx: int, k: int) -> list[int]:
    """Find a JSON-ish list of ints in the response; clip to [1, max_idx]."""
    hits: list[int] = []
    for m in _INDEX_LIST_RE.finditer(response):
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
    if not hits:  # fallback: first k entries (highest-scoring already)
        hits = list(range(1, min(k, max_idx) + 1))
    return hits[:k]


def _build_search_messages(
    memory_bank_text: str, problem_system: str, problem_user: str, top_k: int
) -> list[dict]:
    system = (
        "You are helping retrieve lessons from a memory bank of past GPU "
        "kernel attempts.  You will see a memory bank (numbered entries "
        "with outcome tags and short lessons), followed by a new kernel "
        f"task.  Return a JSON array of the {top_k} entry numbers whose "
        "lessons would be MOST useful when writing a correct, fast Triton "
        "kernel for the new task.  Prefer correct entries that teach WHICH "
        "design worked, and incorrect entries that teach WHAT PITFALLS to "
        "avoid.  Output ONLY the JSON array, e.g. `[3, 7, 12]`, nothing else."
    )
    user = (
        f"# Memory bank\n{memory_bank_text}\n\n"
        f"# New task (system prompt the kernel-writer will see)\n{problem_system[:2000]}\n"
        f"...(system prompt continues; you don't need it all)\n\n"
        f"# New task (user prompt the kernel-writer will see)\n{problem_user}\n\n"
        f"Return JSON: the top-{top_k} most useful entry indices."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _build_augmented_system(
    base_system: str, entries: list[dict], selected_idx: list[int]
) -> str:
    selected = [e for e in entries if e["idx"] in selected_idx]
    if not selected:
        return base_system
    header = (
        "\n\n---\n## Memory: lessons from past attempts at similar tasks\n"
        "The following are distilled notes from previous kernel-writing "
        "attempts at this (or a closely related) task.  Use them to avoid "
        "known pitfalls and to favor designs that are known to work.\n"
    )
    chunks = [header]
    for e in selected:
        chunks.append(
            f"\n### Memory entry {e['idx']}  [{e['tag']}]\n{e['body']}\n"
        )
    return base_system + "".join(chunks)


# --------------------------------------------------------------------------
# Prompt construction (identical to eval_standalone.py)
# --------------------------------------------------------------------------
def _build_standalone_prompts(problem_id: str) -> tuple[str, str]:
    sys.path.insert(0, "/atlas")
    from agent.prompts import TRITON_SYSTEM_PROMPT
    from benchmark.kernel.problems import KERNEL_PROBLEMS

    kp = KERNEL_PROBLEMS[problem_id]
    system = TRITON_SYSTEM_PROMPT
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
# Generation (reused from eval_standalone's pattern; kept inline for
# clarity + to avoid cross-module imports at Modal container start)
# --------------------------------------------------------------------------
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
            f"rolling {len(pending_idx)}",
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
            if content.strip():
                results[slot] = (content, reasoning, tag)
            else:
                results[slot] = (content, reasoning, tag)
                next_pending.append(slot)
        pending_idx = next_pending
    return results


# --------------------------------------------------------------------------
# Main eval function
# --------------------------------------------------------------------------
@app.function(
    gpu="A100-80GB",
    timeout=3 * 3600,
    volumes={
        "/hf_cache": HF_CACHE_VOL,
        "/atlas_models": MODELS_VOL,
        "/outputs": OUTPUTS_VOL,
    },
)
def run_rlm_eval(
    problems: list[str],
    memory_bank_text: str,
    run_name: str,
    n_samples: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    seed: int,
    top_k: int,
    base_model: str,
) -> dict:
    import openai

    sys.path.insert(0, "/atlas")
    from benchmark.kernel.evaluator import evaluate_kernel
    from benchmark.kernel.problems import KERNEL_PROBLEMS
    from utils.extract import extract_code

    for pid in problems:
        assert pid in KERNEL_PROBLEMS, f"Unknown problem_id {pid!r}"

    entries = _parse_entries(memory_bank_text)
    print(f"[rlm] memory bank has {len(entries)} entries "
          f"({len(memory_bank_text)} chars)", flush=True)
    assert entries, "Memory bank parsed as empty — check format"

    vllm_proc = _start_vllm(base_model)

    try:
        client = openai.OpenAI(
            api_key="EMPTY", base_url=f"http://localhost:{VLLM_PORT}/v1"
        )

        base_dir = Path("/outputs") / "eval" / run_name
        base_dir.mkdir(parents=True, exist_ok=True)
        (base_dir / "memory_bank.txt").write_text(memory_bank_text)

        all_summaries: list[dict] = []
        for pid in problems:
            problem = KERNEL_PROBLEMS[pid]
            system_msg, user_msg = _build_standalone_prompts(pid)

            # --- Stage 1: search over memory bank -----------------------
            # Cascade reasoning_effort (default → medium → low → minimal) the
            # same way the generate path does.  gpt-oss at default effort can
            # burn the whole output budget on analysis-channel reasoning and
            # emit nothing on the final channel, which would silently fall
            # back to the first-3 memory entries — that's a degenerate RLM
            # condition we want to avoid on out-of-domain problems.
            print(f"\n[rlm] === leg {pid}: stage 1 (search) ===", flush=True)
            search_msgs = _build_search_messages(
                memory_bank_text, system_msg, user_msg, top_k
            )
            search_cascade: list[str | None] = [None, "medium", "low", "minimal"]
            search_content = ""
            search_reasoning = None
            search_effort = "default"
            search_t0 = time.time()
            for attempt_idx, effort in enumerate(search_cascade):
                params = dict(
                    model=base_model,
                    messages=search_msgs,
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=2048,
                    seed=seed + attempt_idx,
                )
                if effort is not None:
                    params["extra_body"] = {"reasoning_effort": effort}
                try:
                    search_resp = client.chat.completions.create(**params)
                except Exception as exc:  # noqa: BLE001
                    print(f"[rlm] search call failed (effort={effort}): {exc}",
                          flush=True)
                    continue
                msg = search_resp.choices[0].message
                content = msg.content or ""
                reasoning = getattr(msg, "reasoning", None) or getattr(
                    msg, "reasoning_content", None
                )
                search_content = content
                search_reasoning = reasoning
                search_effort = effort or "default"
                if content.strip():
                    if attempt_idx > 0:
                        print(
                            f"[rlm] search recovered on attempt {attempt_idx + 1} "
                            f"(effort={search_effort})",
                            flush=True,
                        )
                    break
                print(
                    f"[rlm] search attempt {attempt_idx + 1}/"
                    f"{len(search_cascade)} returned empty "
                    f"(effort={search_effort}); cascading",
                    flush=True,
                )
            selected_idx = _parse_indices(search_content, len(entries), top_k)
            print(
                f"[rlm] search returned in {time.time() - search_t0:.1f}s "
                f"effort={search_effort}; raw={search_content!r}; "
                f"selected={selected_idx}",
                flush=True,
            )

            augmented_system = _build_augmented_system(system_msg, entries, selected_idx)

            leg_dir = base_dir / f"{pid}__rlm"
            leg_dir.mkdir(parents=True, exist_ok=True)
            (leg_dir / "system_message.txt").write_text(augmented_system)
            (leg_dir / "user_message.txt").write_text(user_msg)
            (leg_dir / "search_response.txt").write_text(search_content)
            if search_reasoning:
                (leg_dir / "search_reasoning.txt").write_text(search_reasoning)
            (leg_dir / "selected_entries.json").write_text(
                json.dumps({"top_k": top_k, "selected": selected_idx}, indent=2)
            )

            # --- Stage 2: generate with augmented system prompt --------
            print(f"\n[rlm] === leg {pid}: stage 2 (generate, n={n_samples}) ===",
                  flush=True)
            gen_t0 = time.time()
            generations = _batched_generate(
                client=client,
                served_model=base_model,
                messages=[
                    {"role": "system", "content": augmented_system},
                    {"role": "user", "content": user_msg},
                ],
                n_samples=n_samples,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                seed=seed,
                batch_size=32,
            )
            total_gen_s = time.time() - gen_t0
            print(
                f"      [batch] generated {n_samples} in {total_gen_s:.1f}s "
                f"({n_samples / total_gen_s:.2f} seq/s)",
                flush=True,
            )

            samples: list[dict] = []
            for i, (raw, reasoning, winning_effort) in enumerate(generations):
                code = extract_code(raw)
                gen_time = total_gen_s / n_samples
                result = evaluate_kernel(problem, code, timeout=180)
                samples.append({
                    "sample_idx": i,
                    "raw_length": len(raw),
                    "code_length": len(code),
                    "reasoning_length": len(reasoning) if reasoning else 0,
                    "winning_reasoning_effort": winning_effort,
                    "gen_time_s": round(gen_time, 2),
                    "correct": result.correct,
                    "score": float(result.score),
                    "feedback": result.feedback[:4000],
                })
                (leg_dir / f"sample_{i:03d}_raw.txt").write_text(raw)
                (leg_dir / f"sample_{i:03d}_code.py").write_text(code)
                if reasoning:
                    (leg_dir / f"sample_{i:03d}_reasoning.txt").write_text(reasoning)
                print(
                    f"    [rlm/{pid}] [{i + 1}/{n_samples}] "
                    f"correct={result.correct} score={result.score:.3f} "
                    f"gen={gen_time:.1f}s",
                    flush=True,
                )
                if (i + 1) % 20 == 0 or (i + 1) == n_samples:
                    OUTPUTS_VOL.commit()

            correct = [s for s in samples if s["correct"]]
            summary = {
                "problem_id": pid,
                "leg": "rlm",
                "n_samples": n_samples,
                "temperature": temperature,
                "selected_entries": selected_idx,
                "pass_at_1": 1.0 if samples and samples[0]["correct"] else 0.0,
                "pass_at_k": len(correct) / max(1, n_samples),
                "num_correct": len(correct),
                "best_speedup_when_correct": max(
                    (s["score"] for s in correct), default=0.0
                ),
                "mean_speedup_when_correct": (
                    sum(s["score"] for s in correct) / len(correct)
                    if correct else 0.0
                ),
            }
            (leg_dir / "summary.json").write_text(json.dumps(summary, indent=2))
            (leg_dir / "samples.json").write_text(json.dumps(samples, indent=2))
            OUTPUTS_VOL.commit()
            all_summaries.append(summary)

        compare = {
            "run_name": run_name,
            "base_model": base_model,
            "memory_bank_entries": len(entries),
            "top_k": top_k,
            "n_samples": n_samples,
            "temperature": temperature,
            "legs": all_summaries,
        }
        (base_dir / "compare.json").write_text(json.dumps(compare, indent=2))
        OUTPUTS_VOL.commit()

        print("\n=== ATLAS-RLM RESULTS ===", flush=True)
        print(f"{'problem':<12} {'pass@1':>7} {'pass@k':>7} {'n_corr':>7} "
              f"{'best_sp':>8} {'mean_sp':>8}", flush=True)
        for s in all_summaries:
            print(
                f"{s['problem_id']:<12} {s['pass_at_1']:>7.2f} "
                f"{s['pass_at_k']:>7.2f} {s['num_correct']:>7} "
                f"{s['best_speedup_when_correct']:>8.3f} "
                f"{s['mean_speedup_when_correct']:>8.3f}",
                flush=True,
            )
        return compare

    finally:
        print("[vllm] terminating server...", flush=True)
        vllm_proc.terminate()
        try:
            vllm_proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            vllm_proc.kill()


# --------------------------------------------------------------------------
# Local entry point
# --------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    problems: str,
    run_name: str,
    memory_bank: str,
    n_samples: int = 8,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: int = 8192,
    top_k: int = 3,
    base_model: str = BASE_MODEL_DEFAULT,
    seed: int = 42,
):
    """RLM evaluation.

    problems        : comma-separated problem ids (e.g. ``softmax,layernorm,matmul``).
    run_name        : subdir under outputs volume for artifacts.
    memory_bank     : local path to memory-bank text file.
    top_k           : how many memory entries to inject into the system prompt.
    """
    problem_list = [p.strip() for p in problems.split(",") if p.strip()]
    mem_path = Path(memory_bank)
    if not mem_path.exists():
        raise FileNotFoundError(mem_path)
    mem_text = mem_path.read_text()
    print(
        f"[local] RLM eval: problems={problem_list} run_name={run_name} "
        f"mem_bank={mem_path} ({len(mem_text)} chars) top_k={top_k} "
        f"n_samples={n_samples}"
    )
    compare = run_rlm_eval.remote(
        problems=problem_list,
        memory_bank_text=mem_text,
        run_name=run_name,
        n_samples=n_samples,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        seed=seed,
        top_k=top_k,
        base_model=base_model,
    )
    print("\n=== RLM COMPARE ===")
    for leg in compare["legs"]:
        print(
            f"  {leg['problem_id']:<12} rlm  pass@1={leg['pass_at_1']:.2f} "
            f"pass@k={leg['pass_at_k']:.2f} "
            f"best_speedup={leg['best_speedup_when_correct']:.3f}"
        )
    print(
        f"\nDownload artifacts with:\n"
        f"  modal volume get atlas-openevolve-outputs eval/{run_name} ./eval_runs/"
    )
