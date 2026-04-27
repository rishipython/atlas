"""Local standalone evaluation on one machine (e.g. Colab A100).

Runs base and/or LoRA-adapted gpt-oss against the kernel benchmark by starting
one local vLLM server and issuing repeated chat-completions requests.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import openai

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agent.prompts import TRITON_SYSTEM_PROMPT
from benchmark.kernel.evaluator import evaluate_kernel
from benchmark.kernel.problems import KERNEL_PROBLEMS
from experiment.local_vllm_utils import VLLM_PORT, resolve_adapter_path, start_vllm
from experiment.research_metrics import parse_pass_ks, summarize_sample_records
from utils.extract import extract_code


BASE_MODEL_DEFAULT = "openai/gpt-oss-20b"


def build_standalone_prompts(problem_id: str) -> tuple[str, str]:
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
        f"Return a single ```python fenced block containing the full solution."
    )
    return system, user


def batched_generate(
    client: openai.OpenAI,
    *,
    served_model: str,
    messages: list[dict[str, str]],
    n_samples: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    seed: int,
) -> list[tuple[str, str | None, str]]:
    cascade: list[tuple[str | None, str]] = [(None, "default"), ("medium", "medium"), ("low", "low")]
    results: list[tuple[str, str | None, str]] = [("", None, "default")] * n_samples
    for i in range(n_samples):
        for effort, tag in cascade:
            params = dict(
                model=served_model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                seed=seed + i,
            )
            if effort is not None:
                params["extra_body"] = {"reasoning_effort": effort}
            resp = client.chat.completions.create(**params)
            msg = resp.choices[0].message
            content = msg.content or ""
            reasoning = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None)
            if content.strip():
                results[i] = (content, reasoning, tag)
                break
            results[i] = (content, reasoning, tag)
    return results


def sample_and_score(
    client: openai.OpenAI,
    *,
    served_model: str,
    tag: str,
    problem_id: str,
    n_samples: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    seed: int,
    pass_ks: list[int],
    out_dir: Path,
) -> dict:
    problem = KERNEL_PROBLEMS[problem_id]
    system_msg, user_msg = build_standalone_prompts(problem_id)
    leg_dir = out_dir / f"{problem_id}__{tag}"
    leg_dir.mkdir(parents=True, exist_ok=True)
    (leg_dir / "system_message.txt").write_text(system_msg)
    (leg_dir / "user_message.txt").write_text(user_msg)

    t0 = time.time()
    generations = batched_generate(
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
    )
    total_gen_s = time.time() - t0
    samples = []
    for i, (raw, reasoning, winning_effort) in enumerate(generations):
        code = extract_code(raw)
        result = evaluate_kernel(problem, code, timeout=180)
        rec = {
            "sample_idx": i,
            "correct": result.correct,
            "score": float(result.score),
            "feedback": result.feedback[:4000],
            "raw_length": len(raw),
            "code_length": len(code),
            "reasoning_length": len(reasoning) if reasoning else 0,
            "winning_reasoning_effort": winning_effort,
            "gen_time_s": round(total_gen_s / max(1, n_samples), 2),
        }
        samples.append(rec)
        (leg_dir / f"sample_{i:03d}_raw.txt").write_text(raw)
        (leg_dir / f"sample_{i:03d}_code.py").write_text(code)
        if reasoning:
            (leg_dir / f"sample_{i:03d}_reasoning.txt").write_text(reasoning)

    metrics = summarize_sample_records(samples, pass_ks=pass_ks)
    summary = {
        "problem_id": problem_id,
        "served_model": served_model,
        "tag": tag,
        "temperature": temperature,
        "pass_ks": pass_ks,
        **metrics,
        "best_speedup_when_correct": metrics["best_speedup"],
    }
    (leg_dir / "samples.json").write_text(json.dumps(samples, indent=2))
    (leg_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems", required=True, help="Comma-separated problem ids.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--adapter", default=None, help="Local adapter path or name.")
    parser.add_argument("--base-model", default=BASE_MODEL_DEFAULT)
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pass-ks", default="1,5,10,20,50,100")
    parser.add_argument("--eval-base", action="store_true", default=True)
    parser.add_argument("--no-eval-base", dest="eval_base", action="store_false")
    parser.add_argument("--eval-adapter", action="store_true", default=True)
    parser.add_argument("--no-eval-adapter", dest="eval_adapter", action="store_false")
    parser.add_argument("--vllm-gpu-mem", type=float, default=0.72)
    args = parser.parse_args()

    problems = [p.strip() for p in args.problems.split(",") if p.strip()]
    for pid in problems:
        assert pid in KERNEL_PROBLEMS, f"Unknown problem_id {pid!r}"
    pass_ks = parse_pass_ks(args.pass_ks)
    adapter_path = resolve_adapter_path(args.adapter) if args.adapter else None
    if args.eval_adapter and not adapter_path:
        print("[warn] --eval-adapter requested but no adapter provided; skipping adapter leg")
        args.eval_adapter = False

    vllm_proc = start_vllm(
        args.base_model,
        adapter_path=adapter_path,
        gpu_memory_utilization=args.vllm_gpu_mem,
    )
    out_dir = Path("eval_runs") / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        client = openai.OpenAI(api_key="EMPTY", base_url=f"http://localhost:{VLLM_PORT}/v1")
        legs = []
        if args.eval_base:
            legs.append((args.base_model, "base"))
        if args.eval_adapter:
            legs.append(("atlas", "atlas"))
        summaries = []
        for pid in problems:
            for served_model, tag in legs:
                summaries.append(
                    sample_and_score(
                        client,
                        served_model=served_model,
                        tag=tag,
                        problem_id=pid,
                        n_samples=args.n_samples,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_tokens=args.max_tokens,
                        seed=args.seed,
                        pass_ks=pass_ks,
                        out_dir=out_dir,
                    )
                )
        compare = {
            "run_name": args.run_name,
            "base_model": args.base_model,
            "adapter": args.adapter,
            "n_samples": args.n_samples,
            "legs": summaries,
        }
        (out_dir / "compare.json").write_text(json.dumps(compare, indent=2))
        print(json.dumps(compare, indent=2))
    finally:
        vllm_proc.terminate()
        try:
            vllm_proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            vllm_proc.kill()


if __name__ == "__main__":
    main()
