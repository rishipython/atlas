"""Synthesize in-distribution reasoning traces from OE lineage records.

Motivation
----------
The v5 ATLAS adapter regressed on OOD problems because we trained the
model on (OE-style prompt → verbose OE-style reasoning → kernel).  That
reasoning was conditioned on OE's parent code + rendered feedback, so
at eval time (base prompt, no parent) the model still emits OE-length
analysis, over-thinks, and either runs out of token budget or
over-engineers.

The fix tested here is a *baseline* distillation recipe:

1. Reconstruct the root→best lineage from ``evolution_trace.jsonl``.
2. Build a synthesis prompt that asks **base gpt-oss-20b itself** to
   write a chain-of-thought reasoning trace, starting from the **exact
   base prompt we use at eval time**, that arrives at the final
   winning kernel code.
3. Instruct the synthesizer to match gpt-oss-20b's own reasoning
   style (stream-of-consciousness, inline code snippets, explicit
   self-correction, consideration of alternatives) and to be long
   enough (~8k chars) to be representative.
4. Save the best N samples so we can pick one and train on
   ``(base_prompt → synth_reasoning + final_code)``.

This module is *not* responsible for training — it only produces the
synth dataset.  Training is done by ``train_atlas_sft.py`` on the
JSONL this script writes.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent

BASE_MODEL_DEFAULT = "openai/gpt-oss-20b"
VLLM_PORT = 8000

HF_CACHE_VOL = modal.Volume.from_name("atlas-hf-cache", create_if_missing=True)
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

app = modal.App("atlas-synth-reasoning", image=image)


# ---------------------------------------------------------------------------
# vLLM subprocess management (same recipe as eval_standalone.py).
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


def _start_vllm(base_model: str):
    import subprocess

    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        base_model,
        "--port",
        str(VLLM_PORT),
        "--gpu-memory-utilization",
        "0.85",
        "--max-model-len",
        "32768",
        "--reasoning-parser",
        "openai_gptoss",
    ]
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
# Synthesizer prompt construction.
# ---------------------------------------------------------------------------
# Style cues lifted directly from the OE reasoning traces we captured.
# Giving the synthesizer a dense list of cues is more reliable than a
# vague "match the style" instruction.
_STYLE_CUES = """\
- Write stream-of-consciousness prose, not a polished explanation or tutorial.
- Use conversational openers: "We need to ...", "Let's ...", "Actually ...", "But ...", "Hmm ...", "Wait ...".
- Short inline snippets in single backticks are fine (e.g. `mask = col + tl.arange(0, BLOCK) < N`), but NEVER emit a full function or kernel body.
- Reconsider choices mid-thought at least a few times: "Wait — that won't work because ...", "Hmm, actually ...", "On second thought ...".
- Genuinely consider at least 2-3 alternative designs and explain why you reject each one before committing (e.g. "one-kernel-three-pass vs two-kernel with sum-buffer vs three-kernel", "2D grid vs 1D grid", "BLOCK=128 vs BLOCK=512").
- Ask yourself pointed questions and answer them ("Do we need to mask on the reduce? ... Yes, because N may not be a multiple of BLOCK, and for the max reduction the masked lanes need -inf as the identity.").
- Think about the tricky Triton details: pointer arithmetic with row/col strides, masking on partial blocks, float32 accumulation for numerical stability, avoiding overflow via max-subtraction, block-size tradeoffs, register pressure, how scalar accumulators like `row_sum += tl.sum(...)` work.
- Reference the required correctness bounds (atol, rtol) and the specific test shapes when they motivate a decision.
- Do NOT describe this as "reasoning" or "a monologue" — just think.
- Do NOT reference any external material: no "the log shows", no "the evolution logs", no "the experiment data", no meta-commentary about having been shown a solution.  You are reasoning from the task alone.
"""

_HARD_RULES = """\
ABSOLUTELY FORBIDDEN:
- Do not emit any triple-backtick fenced code block (no ```python, no ```, nothing fenced).
- Do not write a complete @triton.jit kernel body.
- Do not write a complete wrapper function body.
- Do not output the final solution in any form.
- Do not say things like "Here's the final code:" or "Below is the implementation:".

WHAT TO DO INSTEAD:
- End the trace with a single short handoff sentence like "OK, I think I have the design. Time to write the kernel."
- Then stop. The caller will append the real code after your handoff.
"""

_SYNTH_SYSTEM = (
    "You are gpt-oss-20b and you are thinking through a GPU kernel "
    "programming task.  You will be given a task and a short design brief "
    "describing the solution shape you should converge on.  Your job is to "
    "write the chain-of-thought you would go through WHILE you work out the "
    "solution — exploring, second-guessing yourself, trying an alternative, "
    "coming back to the original, worrying about edge cases.  Do NOT write "
    "a tutorial or a textbook explanation.  Do NOT write the final code.\n\n"
    "Style requirements:\n"
    + _STYLE_CUES
    + "\nLength target: roughly 6000–10000 characters of dense thinking.  "
    "Short tidy summaries are not what we want — we want a real exploration "
    "with dead ends and revisions.  Once you have worked out the design, "
    "do a brief sanity check of edge cases and STOP with the handoff "
    "sentence.  Do not pad the trace.  If you notice yourself starting "
    "sentences with the same phrase ('Now we need to...', 'We need to "
    "think about...') more than twice, you have finished — write the "
    "handoff and stop.\n\n"
    + _HARD_RULES
    + "\nOutput format: your entire response is the trace itself.  No "
    "preamble like 'Here is the reasoning:' — start directly with the first "
    "thought ('We need to...' or similar)."
)


def _build_synth_user_message(
    base_system: str,
    base_user: str,
    design_brief: str,
) -> str:
    parts = [
        "# Task (this is the exact prompt you are reasoning about)",
        "",
        "Your system prompt:",
        "---",
        base_system,
        "---",
        "",
        "User message:",
        "---",
        base_user,
        "---",
        "",
        "# Design brief you should converge toward",
        "",
        "You do not get to see the final code.  Here is only a high-level "
        "sketch of the design you should land on (you should *arrive* at "
        "this design through exploration, not by copying it):",
        "",
        design_brief,
        "",
        "# Your job",
        "",
        "Produce the chain-of-thought you would go through to arrive at a "
        "design matching the brief, starting from only the task above.  Do "
        "NOT reveal that you were given any brief — just explore, "
        "second-guess, and converge organically.  Follow the style, length, "
        "and hard-rules requirements from the system message strictly.  "
        "Begin the trace now.",
    ]
    return "\n".join(parts)


def build_design_brief(final_code: str) -> str:
    """Extract a prose design brief from the final code.

    We deliberately do NOT paste the code itself — seeing the code tempts
    the synthesizer to reproduce it verbatim.  Instead we list the key
    architectural choices the code makes, in bullet form, at a level of
    abstraction a sharp engineer could independently rediscover.
    """
    # Parse the code lightly for the key signals we want the synthesizer
    # to converge on.  The rest of the design is intentionally left to
    # the synthesizer's own thinking.
    signals = []
    n_triton_jit = final_code.count("@triton.jit")
    signals.append(
        f"- Uses **{n_triton_jit} Triton kernel{'s' if n_triton_jit != 1 else ''}**."
    )
    if "program_id(0)" in final_code and "program_id(1)" not in final_code:
        signals.append("- Launch grid is **1-D over rows**: one program per row.")
    if "BLOCK" in final_code:
        # Try to extract a block size literal
        import re

        m = re.search(r"BLOCK\s*=\s*(\d+)", final_code)
        if m:
            signals.append(
                f"- Block size along the column dimension is **{m.group(1)}** "
                "(a compile-time constexpr)."
            )
    if "to(torch.float32)" in final_code or "tl.float32" in final_code:
        signals.append(
            "- Accumulation is done in **float32** for numerical "
            "stability; if the input dtype differs, the wrapper casts to "
            "float32 and casts back at the end."
        )
    if "while col < N" in final_code or "for col in range" in final_code:
        signals.append(
            "- Inside the kernel, the row is processed in a **loop over "
            "column blocks** (N may not be a multiple of BLOCK), using "
            "masked loads/stores on the tail."
        )
    # Detect 3-pass structure heuristically
    passes = []
    if "tl.maximum" in final_code or "max_val" in final_code:
        passes.append("compute per-row maximum")
    if "tl.exp" in final_code:
        passes.append("compute exp(x - max) and accumulate a per-row sum")
    if "/ row_sum" in final_code or "/ sum_val" in final_code:
        passes.append("divide stored exponentials by the per-row sum")
    if passes:
        signals.append(
            "- The kernel makes **"
            + str(len(passes))
            + " passes** over the row inside one launch: "
            + "; ".join(f"({i + 1}) {p}" for i, p in enumerate(passes))
            + "."
        )
    if "-float" in final_code and "inf" in final_code:
        signals.append(
            "- The max reduction uses `-inf` as the masked/identity value so "
            "tail lanes don't corrupt the result."
        )
    signals.append(
        "- Pass both `stride_row` and `stride_col` into the kernel so the "
        "pointer arithmetic works for non-contiguous tensors."
    )
    return "\n".join(signals)


# ---------------------------------------------------------------------------
# Lineage reconstruction (runs locally).
# ---------------------------------------------------------------------------
def reconstruct_lineage(trace_path: Path) -> tuple[dict, list[dict]]:
    """Return (best_record, chain) where chain[0] is root and chain[-1] is best.

    The chain is reconstructed by following ``parent_id -> child_id``
    links in the evolution_trace.jsonl.  Root is defined as the record
    whose parent was never itself a child (i.e. the initial program),
    and best is the record with the highest ``child_metrics.combined_score``.
    """
    recs = [json.loads(l) for l in trace_path.read_text().splitlines() if l.strip()]
    best = max(recs, key=lambda r: r["child_metrics"].get("combined_score", 0.0))
    by_child = {r["child_id"]: r for r in recs}
    chain = [best]
    cur = best
    while cur["parent_id"] in by_child:
        cur = by_child[cur["parent_id"]]
        chain.append(cur)
    chain.reverse()
    return best, chain


def summarize_lineage(chain: list[dict]) -> str:
    """Render a short textual summary of the lineage for the synth prompt.

    Each step gets one or two bullet points describing the approach and
    the headline failure mode (from ``artifacts.feedback``).  The final
    step is flagged as the one that worked.
    """
    lines = []
    for i, rec in enumerate(chain):
        is_last = i == len(chain) - 1
        child_score = rec["child_metrics"].get("combined_score", 0.0)
        parent_score = rec["parent_metrics"].get("combined_score", 0.0)
        feedback = (rec.get("artifacts") or {}).get("feedback", "") or ""
        feedback = feedback.strip().replace("\n", " ")[:280]
        tag = "FINAL WINNING ATTEMPT" if is_last else "intermediate attempt"
        lines.append(
            f"- Step {i + 1} ({tag}): parent_score={parent_score:.2f} → "
            f"child_score={child_score:.2f}. Outcome: {feedback}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Modal remote: generate N synthesis samples.
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/hf_cache": HF_CACHE_VOL, "/outputs": OUTPUTS_VOL},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=60 * 60,
)
def synthesize_remote(
    synth_system: str,
    synth_user: str,
    n_samples: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    reasoning_effort: str | None,
    base_model: str,
    seed: int,
    out_dir_rel: str,
    target_context: dict | None = None,
) -> list[dict]:
    from openai import OpenAI

    vllm_proc = _start_vllm(base_model)
    try:
        client = OpenAI(api_key="EMPTY", base_url=f"http://localhost:{VLLM_PORT}/v1")

        messages = [
            {"role": "system", "content": synth_system},
            {"role": "user", "content": synth_user},
        ]
        print(
            f"[synth] prompt sizes: system={len(synth_system)} user={len(synth_user)}",
            flush=True,
        )

        results: list[dict] = []
        for i in range(n_samples):
            t0 = time.time()
            params = dict(
                model=base_model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                seed=seed + i,
                # Suppress late-sequence "Now we need to ..." repetition
                # attractors that the model falls into when it's approaching
                # its stamina limit on long free-form traces.
                frequency_penalty=0.25,
                presence_penalty=0.1,
            )
            if reasoning_effort is not None:
                params["extra_body"] = {"reasoning_effort": reasoning_effort}
            resp = client.chat.completions.create(**params)
            msg = resp.choices[0].message
            content = msg.content or ""
            # The reasoning trace we want lives in the `final` channel as
            # plain content — we explicitly asked for that in the system
            # message.  Any text the model emitted in the analysis channel
            # (`reasoning` / `reasoning_content`) is separate and is not
            # used as the distilled trace.
            reasoning = getattr(msg, "reasoning", None) or getattr(
                msg, "reasoning_content", None
            )
            dt = time.time() - t0
            rec = {
                "sample_idx": i,
                "temperature": temperature,
                "reasoning_effort": reasoning_effort or "default",
                "seed": seed + i,
                "gen_time_s": round(dt, 2),
                "content_len": len(content),
                "reasoning_len": len(reasoning or ""),
                "finish_reason": resp.choices[0].finish_reason,
                "content": content,
                "analysis_reasoning": reasoning or "",
            }
            print(
                f"[synth] sample {i}: content_len={len(content)} "
                f"analysis_len={len(reasoning or '')} finish={rec['finish_reason']} "
                f"gen={dt:.1f}s",
                flush=True,
            )
            results.append(rec)

        out_dir = Path("/outputs") / out_dir_rel
        out_dir.mkdir(parents=True, exist_ok=True)
        for r in results:
            (out_dir / f"sample_{r['sample_idx']:02d}_content.txt").write_text(
                r["content"]
            )
            if r["analysis_reasoning"]:
                (out_dir / f"sample_{r['sample_idx']:02d}_analysis.txt").write_text(
                    r["analysis_reasoning"]
                )
        (out_dir / "all_samples.json").write_text(json.dumps(results, indent=2))
        if target_context is not None:
            (out_dir / "target_context.json").write_text(
                json.dumps(target_context, indent=2)
            )
        OUTPUTS_VOL.commit()
        print(f"[synth] wrote {len(results)} samples to /outputs/{out_dir_rel}")
        return results
    finally:
        vllm_proc.terminate()
        try:
            vllm_proc.wait(timeout=10)
        except Exception:
            vllm_proc.kill()


# ---------------------------------------------------------------------------
# Modal remote: batch synthesis over many trajectories in one vLLM session.
# ---------------------------------------------------------------------------
# Shares the same image / secrets / volume set as ``synthesize_remote``.
# ``tasks`` is a list of per-trajectory job specs (one synth call each),
# so we boot vLLM once and run all of them back-to-back.  This is ~50x
# cheaper than doing one remote invocation per trajectory.
@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/hf_cache": HF_CACHE_VOL, "/outputs": OUTPUTS_VOL},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=2 * 3600,
)
def synthesize_batch_remote(
    tasks: list[dict],
    synth_system: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    reasoning_effort: str | None,
    base_model: str,
    seed: int,
    out_dir_rel: str,
) -> list[dict]:
    """Run one synth call per task, reusing a single vLLM session.

    Each task must contain:
      - ``task_id``:       str (used as output filename prefix)
      - ``synth_user``:    str (the rendered user prompt for this target)
      - ``target_context``: dict (saved alongside the output for reproducibility)
    """
    from openai import OpenAI

    vllm_proc = _start_vllm(base_model)
    try:
        client = OpenAI(api_key="EMPTY", base_url=f"http://localhost:{VLLM_PORT}/v1")
        out_dir = Path("/outputs") / out_dir_rel
        out_dir.mkdir(parents=True, exist_ok=True)

        results: list[dict] = []
        n = len(tasks)
        t_batch_start = time.time()
        for i, task in enumerate(tasks):
            task_id = task["task_id"]
            synth_user = task["synth_user"]
            t0 = time.time()
            messages = [
                {"role": "system", "content": synth_system},
                {"role": "user", "content": synth_user},
            ]
            params = dict(
                model=base_model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                seed=seed + i,
                frequency_penalty=0.25,
                presence_penalty=0.1,
            )
            if reasoning_effort is not None:
                params["extra_body"] = {"reasoning_effort": reasoning_effort}
            try:
                resp = client.chat.completions.create(**params)
            except Exception as e:
                print(f"[synth-batch] task {task_id} FAILED: {e}", flush=True)
                results.append(
                    {
                        "task_id": task_id,
                        "error": str(e),
                        "content": "",
                        "analysis_reasoning": "",
                    }
                )
                continue
            msg = resp.choices[0].message
            content = msg.content or ""
            reasoning = getattr(msg, "reasoning", None) or getattr(
                msg, "reasoning_content", None
            )
            dt = time.time() - t0
            rec = {
                "task_id": task_id,
                "temperature": temperature,
                "reasoning_effort": reasoning_effort or "default",
                "seed": seed + i,
                "gen_time_s": round(dt, 2),
                "content_len": len(content),
                "reasoning_len": len(reasoning or ""),
                "finish_reason": resp.choices[0].finish_reason,
                "content": content,
                "analysis_reasoning": reasoning or "",
            }
            results.append(rec)
            (out_dir / f"{task_id}_content.txt").write_text(content)
            (out_dir / f"{task_id}_context.json").write_text(
                json.dumps(task["target_context"], indent=2)
            )
            elapsed = time.time() - t_batch_start
            eta = elapsed / (i + 1) * (n - i - 1)
            print(
                f"[synth-batch] {i + 1}/{n} task={task_id} "
                f"content_len={len(content)} finish={rec['finish_reason']} "
                f"gen={dt:.1f}s elapsed={elapsed:.0f}s eta={eta:.0f}s",
                flush=True,
            )
            # Commit periodically so a mid-batch OOM doesn't lose work.
            if (i + 1) % 10 == 0:
                OUTPUTS_VOL.commit()

        (out_dir / "all_samples.json").write_text(json.dumps(results, indent=2))
        OUTPUTS_VOL.commit()
        print(
            f"[synth-batch] done: {len(results)} tasks, total "
            f"{time.time() - t_batch_start:.0f}s → /outputs/{out_dir_rel}"
        )
        return results
    finally:
        vllm_proc.terminate()
        try:
            vllm_proc.wait(timeout=10)
        except Exception:
            vllm_proc.kill()


# ---------------------------------------------------------------------------
# Algotune-flavored synth: no Triton references, numpy/scipy-style cues.
# ---------------------------------------------------------------------------
_STYLE_CUES_ALGOTUNE = """\
- Write stream-of-consciousness prose, not a polished explanation or tutorial.
- Use conversational openers: "We need to ...", "Let's ...", "Actually ...", "But ...", "Hmm ...", "Wait ...".
- Short inline snippets in single backticks are fine (e.g. `D = -2 * X @ X.T + sq[:, None] + sq[None, :]`), but NEVER emit a full function body.
- Reconsider choices mid-thought a few times: "Wait — that won't work because ...", "Hmm, actually ...", "On second thought ...".
- Genuinely consider at least 2-3 alternative designs and explain why you reject each one before committing (e.g. "Python nested loops vs vectorised numpy vs scipy's cdist vs numba", "FFT convolution vs direct vs np.convolve", "Taylor series vs Pade + scaling/squaring vs scipy.linalg.expm").
- Ask yourself pointed questions and answer them ("Do we get enough speedup from pure numpy, or do we need scipy? ... Let's see, the Gram-matrix trick is a single BLAS call, that's fine.").
- Think about the numerical details: float64 vs float32, accumulation order, tiny negatives from cancellation (clip to 0?), complex-valued FFT output (take .real), padding length (next pow-of-two for FFT speed), symmetry tricks.
- Mention the rtol/atol budget when it motivates a decision.
- Do NOT describe this as "reasoning" or "a monologue" — just think.
- Do NOT reference any external material: no "the log shows", no "the evolution logs", no meta-commentary about having been shown a solution. You are reasoning from the task alone.
"""

_HARD_RULES_ALGOTUNE = """\
ABSOLUTELY FORBIDDEN:
- Do not emit any triple-backtick fenced code block (no ```python, no ```, nothing fenced).
- Do not write a complete function body for the entry point.
- Do not output the final solution in any form.
- Do not say things like "Here's the final code:" or "Below is the implementation:".

WHAT TO DO INSTEAD:
- End the trace with a single short handoff sentence like "OK, I think I have the design. Time to write it."
- Then stop. The caller will append the real code after your handoff.
"""

_SYNTH_SYSTEM_ALGOTUNE = (
    "You are gpt-oss-20b and you are thinking through a Python "
    "numerical-computing speedup task. You will be given the task and a "
    "short design brief describing the solution shape you should converge "
    "on. Your job is to write the chain-of-thought you would go through "
    "WHILE you work out the solution — exploring, second-guessing "
    "yourself, trying an alternative, coming back to the original, "
    "worrying about numerical edge cases. Do NOT write a tutorial or a "
    "textbook explanation. Do NOT write the final code.\n\n"
    "Style requirements:\n"
    + _STYLE_CUES_ALGOTUNE
    + "\nLength target: roughly 4000–8000 characters of dense thinking. "
    "Short tidy summaries are not what we want — we want a real "
    "exploration with dead ends and revisions. Once you have worked out "
    "the design, do a brief sanity check of edge cases and STOP with the "
    "handoff sentence. Do not pad the trace. If you notice yourself "
    "starting sentences with the same phrase more than twice, you have "
    "finished — write the handoff and stop.\n\n"
    + _HARD_RULES_ALGOTUNE
    + "\nOutput format: your entire response is the trace itself. No "
    "preamble like 'Here is the reasoning:' — start directly with the "
    "first thought ('We need to...' or similar)."
)


# ---------------------------------------------------------------------------
# Local entrypoint: algotune-flavored synth (uses algotune_prompts).
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main_algotune(
    trace_path: str,
    problem_id: str,
    out_name: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_tokens: int = 3200,
    reasoning_effort: str = "low",
    base_model: str = BASE_MODEL_DEFAULT,
    seed: int = 101,
    max_trajectories: int = 0,
):
    """Synthesize one reasoning trace per OE trajectory for an algotune problem.

    Identical batching strategy to ``main_all``, but uses the
    algotune-flavored system prompt and the simple user-facing base
    prompt produced by ``experiment.algotune_prompts``.
    """
    sys.path.insert(0, str(REPO_ROOT))
    from experiment.algotune_prompts import (  # noqa: E402
        build_algotune_prompts,
        design_brief_from_code,
    )

    base_system, base_user, _ep = build_algotune_prompts(problem_id)

    records = [
        json.loads(line)
        for line in Path(trace_path).read_text().splitlines()
        if line.strip()
    ]
    if max_trajectories and len(records) > max_trajectories:
        records = records[:max_trajectories]

    tasks: list[dict] = []
    for r in records:
        iteration = r.get("iteration")
        child_code = r.get("child_code") or ""
        child_metrics = r.get("child_metrics") or {}
        feedback = (r.get("artifacts") or {}).get("feedback", "") or ""
        if not child_code:
            print(f"[local] skipping iter={iteration}: empty child_code")
            continue

        design_brief = design_brief_from_code(child_code)
        synth_user_message = _build_synth_user_message(
            base_system=base_system,
            base_user=base_user,
            design_brief=design_brief,
        )
        target_context = {
            "problem_id": problem_id,
            "task_family": "algotune",
            "iteration": iteration,
            "child_id": r.get("child_id"),
            "parent_id": r.get("parent_id"),
            "child_metrics": child_metrics,
            "feedback": feedback[:2000],
            "base_system": base_system,
            "base_user": base_user,
            "final_code": child_code,
            "design_brief": design_brief,
        }
        tasks.append(
            {
                "task_id": f"iter_{iteration:03d}",
                "synth_user": synth_user_message,
                "target_context": target_context,
            }
        )

    print(
        f"[local] queued {len(tasks)} algotune synth tasks from {trace_path} "
        f"(problem={problem_id}, out=synth/{out_name})"
    )

    results = synthesize_batch_remote.remote(
        tasks=tasks,
        synth_system=_SYNTH_SYSTEM_ALGOTUNE,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        base_model=base_model,
        seed=seed,
        out_dir_rel=f"synth/{out_name}",
    )

    ok = sum(1 for r in results if r.get("content"))
    print(f"\n=== BATCH ALGOTUNE SYNTH DONE: {ok}/{len(results)} non-empty ===")
    print(
        f"\nDownload with:\n  modal volume get atlas-openevolve-outputs "
        f"synth/{out_name} ./runs/\n"
    )


# ---------------------------------------------------------------------------
# Local entrypoint: synth one reasoning trace per trajectory in a trace.
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main_all(
    trace_path: str,
    problem_id: str = "softmax",
    out_name: str = "synth_softmax_all_v1",
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_tokens: int = 3200,
    reasoning_effort: str = "low",
    base_model: str = BASE_MODEL_DEFAULT,
    seed: int = 101,
    max_trajectories: int = 0,
):
    """Synthesize one reasoning trace per OE trajectory (not only best).

    Reads every record of ``trace_path``, builds a design brief from the
    record's ``child_code``, and queues a synth call.  All calls share a
    single vLLM session on Modal.  The outputs are named by iteration
    index so the advantage-SFT builder can join them back to the scoring
    records by iteration.
    """
    sys.path.insert(0, str(REPO_ROOT))
    from agent.prompts import TRITON_SYSTEM_PROMPT
    from benchmark.kernel.problems import KERNEL_PROBLEMS

    kp = KERNEL_PROBLEMS[problem_id]
    base_system = TRITON_SYSTEM_PROMPT
    base_user = (
        f"Problem: {kp.description}\n\n"
        f"Write a complete, runnable Python file that defines a function "
        f"`{kp.entry_point}(...)` with the same signature as the reference "
        f"below. It must produce output numerically close to the reference "
        f"(atol={kp.atol}, rtol={kp.rtol}) on all of these test shapes: "
        f"{kp.test_shapes}.\n\n"
        f"Reference implementation (correct but slow — replace its body with a "
        f"Triton kernel):\n```python\n{kp.reference_code}\n```\n\n"
        f"Return a single ```python fenced block containing the full solution."
    )

    records = [
        json.loads(line)
        for line in Path(trace_path).read_text().splitlines()
        if line.strip()
    ]
    if max_trajectories and len(records) > max_trajectories:
        records = records[:max_trajectories]

    tasks: list[dict] = []
    for r in records:
        iteration = r.get("iteration")
        child_code = r.get("child_code") or ""
        child_metrics = r.get("child_metrics") or {}
        feedback = (r.get("artifacts") or {}).get("feedback", "") or ""
        if not child_code:
            print(f"[local] skipping iter={iteration}: empty child_code")
            continue

        design_brief = build_design_brief(child_code)
        synth_user_message = _build_synth_user_message(
            base_system=base_system,
            base_user=base_user,
            design_brief=design_brief,
        )
        target_context = {
            "problem_id": problem_id,
            "iteration": iteration,
            "child_id": r.get("child_id"),
            "parent_id": r.get("parent_id"),
            "child_metrics": child_metrics,
            "feedback": feedback[:2000],
            "base_system": base_system,
            "base_user": base_user,
            "final_code": child_code,
            "design_brief": design_brief,
        }
        tasks.append(
            {
                "task_id": f"iter_{iteration:03d}",
                "synth_user": synth_user_message,
                "target_context": target_context,
            }
        )

    print(
        f"[local] queued {len(tasks)} synth tasks from {trace_path} "
        f"(problem={problem_id}, out=synth/{out_name})"
    )

    results = synthesize_batch_remote.remote(
        tasks=tasks,
        synth_system=_SYNTH_SYSTEM,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        base_model=base_model,
        seed=seed,
        out_dir_rel=f"synth/{out_name}",
    )

    ok = sum(1 for r in results if r.get("content"))
    print(f"\n=== BATCH SYNTH DONE: {ok}/{len(results)} non-empty ===")
    print(
        f"\nDownload with:\n  modal volume get atlas-openevolve-outputs "
        f"synth/{out_name} ./runs/\n"
    )


# ---------------------------------------------------------------------------
# Local entrypoint: reconstruct lineage + invoke remote synthesis.
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    trace_path: str,
    problem_id: str = "softmax",
    out_name: str = "synth_softmax_v1",
    n_samples: int = 4,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_tokens: int = 3200,
    reasoning_effort: str = "low",
    base_model: str = BASE_MODEL_DEFAULT,
    seed: int = 11,
):
    """Generate synthesis candidates for one problem's final-best lineage.

    Parameters
    ----------
    trace_path       : local path to the evolution_trace.jsonl.
    problem_id       : which kernel problem (used to fetch the exact
                       eval-time base prompt).
    reasoning_effort : we default to "low" because we want the model's
                       *final-channel* output (the synthesized trace)
                       to be long — high analysis-effort would instead
                       dump most tokens into a separate channel we
                       discard.
    """
    sys.path.insert(0, str(REPO_ROOT))
    from agent.prompts import TRITON_SYSTEM_PROMPT
    from benchmark.kernel.problems import KERNEL_PROBLEMS

    kp = KERNEL_PROBLEMS[problem_id]
    base_system = TRITON_SYSTEM_PROMPT
    base_user = (
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

    best, chain = reconstruct_lineage(Path(trace_path))
    final_code = best["child_code"]
    design_brief = build_design_brief(final_code)
    final_score = best["child_metrics"].get("combined_score", 0.0)

    print(
        f"[local] problem={problem_id} chain_len={len(chain)} "
        f"final_score={final_score:.3f} final_code_chars={len(final_code)}",
        flush=True,
    )
    print("[local] design_brief:")
    print(design_brief)

    synth_user_message = _build_synth_user_message(
        base_system=base_system,
        base_user=base_user,
        design_brief=design_brief,
    )

    # Stash the target context alongside the synth output so the
    # downstream SFT dataset builder doesn't have to re-run this entrypoint.
    target_context = {
        "problem_id": problem_id,
        "base_system": base_system,
        "base_user": base_user,
        "final_code": final_code,
        "final_score": final_score,
        "design_brief": design_brief,
        "lineage_summary": summarize_lineage(chain),
        "synth_system": _SYNTH_SYSTEM,
        "synth_user": synth_user_message,
    }

    results = synthesize_remote.remote(
        synth_system=_SYNTH_SYSTEM,
        synth_user=synth_user_message,
        n_samples=n_samples,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        base_model=base_model,
        seed=seed,
        out_dir_rel=f"synth/{out_name}",
        target_context=target_context,
    )

    # Brief summary + which sample looks usable.
    print("\n=== SYNTH RESULTS ===")
    for r in results:
        print(
            f"  sample {r['sample_idx']}: content_len={r['content_len']} "
            f"analysis_len={r['reasoning_len']} finish={r['finish_reason']}"
        )
    print(
        f"\nDownload with:\n  modal volume get atlas-openevolve-outputs "
        f"synth/{out_name} ./runs/\n"
    )
