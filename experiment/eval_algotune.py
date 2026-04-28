"""Evaluate Best-of-N on AlgoTune for base and optional ATLAS adapter."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_MODEL_DEFAULT = "openai/gpt-oss-20b"
VLLM_PORT = 8000

HF_CACHE_VOL = modal.Volume.from_name("atlas-hf-cache", create_if_missing=True)
MODELS_VOL = modal.Volume.from_name("atlas-models", create_if_missing=True)
OUTPUTS_VOL = modal.Volume.from_name("atlas-openevolve-outputs", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl", "build-essential")
    .pip_install(
        "vllm==0.19.1",
        "openai>=1.50",
        "numpy>=1.26",
        "scipy>=1.11",
        "numba>=0.60",
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


def _uses_gptoss_reasoning(model_name: str) -> bool:
    name = model_name.lower()
    return "gpt-oss" in name or "gpt_oss" in name


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
    ]
    if _uses_gptoss_reasoning(base_model):
        cmd.extend(["--reasoning-parser", "openai_gptoss"])
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


def _batched_generate(
    client,
    served_model: str,
    messages: list[dict[str, str]],
    n_samples: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    seed: int,
    use_reasoning_effort: bool,
) -> list[tuple[str, str | None, str]]:
    import concurrent.futures as cf

    def one_call(sample_idx: int, reasoning_effort: str | None) -> tuple[str, str | None]:
        params: dict = {
            "model": served_model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "seed": seed + sample_idx,
        }
        if use_reasoning_effort and reasoning_effort is not None:
            params["extra_body"] = {"reasoning_effort": reasoning_effort}
        response = client.chat.completions.create(**params)
        message = response.choices[0].message
        content = message.content or ""
        reasoning = getattr(message, "reasoning", None) or getattr(message, "reasoning_content", None)
        return content, reasoning

    results: list[tuple[str, str | None, str]] = [("", None, "default")] * n_samples
    pending = list(range(n_samples))
    cascade: list[tuple[str | None, str]]
    if use_reasoning_effort:
        cascade = [(None, "default"), ("medium", "medium"), ("low", "low")]
    else:
        cascade = [(None, "default")]

    for effort, label in cascade:
        if not pending:
            break
        next_pending: list[int] = []
        with cf.ThreadPoolExecutor(max_workers=min(32, len(pending))) as pool:
            futures = {pool.submit(one_call, idx, effort): idx for idx in pending}
            for future in cf.as_completed(futures):
                idx = futures[future]
                try:
                    content, reasoning = future.result()
                except Exception:
                    content, reasoning = "", None
                results[idx] = (content, reasoning, label)
                if not content.strip():
                    next_pending.append(idx)
        pending = next_pending
    return results


def _score_candidate(problem_id: str, candidate_code: str, timeout_s: int = 300) -> dict:
    sys.path.insert(0, "/atlas")
    from experiment.tasks import get_task

    spec = get_task("algotune", problem_id)
    with tempfile.TemporaryDirectory(prefix=f"algotune_eval_{problem_id}_") as tmpdir:
        tmp = Path(tmpdir)
        candidate_path = tmp / "candidate.py"
        candidate_path.write_text(candidate_code or "# empty\n")
        evaluator_src = spec.evaluator.replace(
            "from openevolve.evaluation_result import EvaluationResult",
            (
                "class EvaluationResult:\n"
                "    def __init__(self, metrics=None, artifacts=None):\n"
                "        self.metrics = metrics or {}\n"
                "        self.artifacts = artifacts or {}\n"
            ),
        )
        evaluator_path = tmp / "evaluator.py"
        evaluator_path.write_text(evaluator_src)
        harness_path = tmp / "harness.py"
        harness_path.write_text(
            "import importlib.util, json\n"
            f"spec = importlib.util.spec_from_file_location('evmod', {str(evaluator_path)!r})\n"
            "mod = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(mod)\n"
            f"res = mod.evaluate({str(candidate_path)!r})\n"
            "if isinstance(res, dict):\n"
            "    payload = {'metrics': res, 'artifacts': {}}\n"
            "else:\n"
            "    payload = {'metrics': dict(res.metrics), 'artifacts': dict(res.artifacts)}\n"
            "print('__OE_RESULT__' + json.dumps(payload))\n"
        )

        env = os.environ.copy()
        env["PYTHONPATH"] = "/atlas" + os.pathsep + env.get("PYTHONPATH", "")
        try:
            proc = subprocess.run(
                [sys.executable, str(harness_path)],
                capture_output=True,
                text=True,
                timeout=timeout_s,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return {"correct": False, "combined_score": 0.0, "speedup": 0.0, "feedback": "timeout"}

        for line in (proc.stdout or "").splitlines():
            if not line.startswith("__OE_RESULT__"):
                continue
            payload = json.loads(line[len("__OE_RESULT__") :])
            metrics = payload.get("metrics") or {}
            artifacts = payload.get("artifacts") or {}
            correctness = metrics.get("correctness")
            if correctness is None:
                correctness = metrics.get("correctness_score", 0.0)
            speedup = metrics.get("speedup")
            if speedup is None:
                speedup = metrics.get("speedup_score", 0.0)
            return {
                "correct": float(correctness or 0.0) >= 1.0,
                "combined_score": float(metrics.get("combined_score", 0.0) or 0.0),
                "speedup": float(speedup or 0.0),
                "feedback": str(artifacts.get("feedback", ""))[:4000],
            }
        return {"correct": False, "combined_score": 0.0, "speedup": 0.0, "feedback": proc.stderr[:2000]}


def _metrics_from_samples(samples: list[dict]) -> dict:
    rewards = [float(sample["combined_score"]) for sample in samples]
    correctness = [bool(sample["correct"]) for sample in samples]
    best_reward_by_k: list[float] = []
    pass_by_k: list[float] = []

    best_reward = 0.0
    any_pass = False
    for reward, passed in zip(rewards, correctness):
        best_reward = max(best_reward, reward)
        any_pass = any_pass or passed
        best_reward_by_k.append(best_reward)
        pass_by_k.append(1.0 if any_pass else 0.0)

    return {
        "best_reward_at_n": best_reward_by_k[-1] if best_reward_by_k else 0.0,
        "pass_at_n": pass_by_k[-1] if pass_by_k else 0.0,
        "best_reward_by_k": best_reward_by_k,
        "pass_by_k": pass_by_k,
    }


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
) -> tuple[list[dict], dict]:
    from utils.extract import extract_code

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "system_message.txt").write_text(system_msg)
    (output_dir / "user_message.txt").write_text(user_msg)

    generations = _batched_generate(
        client,
        served_model,
        [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        n_samples,
        temperature,
        top_p,
        max_tokens,
        seed,
        _uses_gptoss_reasoning(served_model),
    )

    samples: list[dict] = []
    for idx, (raw, reasoning, effort) in enumerate(generations):
        code = extract_code(raw)
        result = _score_candidate(problem_id, code)
        sample = {
            "sample_idx": idx,
            "winning_reasoning_effort": effort,
            "raw_length": len(raw),
            "code_length": len(code),
            "reasoning_length": len(reasoning) if reasoning else 0,
            **result,
        }
        samples.append(sample)
        (output_dir / f"sample_{idx:03d}_raw.txt").write_text(raw)
        (output_dir / f"sample_{idx:03d}_code.py").write_text(code)
        if reasoning:
            (output_dir / f"sample_{idx:03d}_reasoning.txt").write_text(reasoning)

    metrics = _metrics_from_samples(samples)
    summary = {
        "problem_id": problem_id,
        "served_model": served_model,
        "n_samples": n_samples,
        **metrics,
    }
    (output_dir / "samples.json").write_text(json.dumps(samples, indent=2))
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    OUTPUTS_VOL.commit()
    return samples, summary


def _compute_cross_leg_metrics(compare: dict) -> None:
    summaries = compare["legs"]
    by_problem: dict[str, dict[str, dict]] = {}
    for summary in summaries:
        by_problem.setdefault(summary["problem_id"], {})[summary["tag"]] = summary

    for problem_id, legs in by_problem.items():
        base = legs.get("base")
        if not base:
            continue
        baseline_reward = float(base["best_reward_at_n"])
        for tag, leg in legs.items():
            min_k = None
            for idx, reward in enumerate(leg["best_reward_by_k"], start=1):
                if reward >= baseline_reward:
                    min_k = idx
                    break
            leg["min_k_to_base_best_reward"] = min_k


@app.function(
    gpu="A100-80GB",
    timeout=3 * 3600,
    volumes={"/hf_cache": HF_CACHE_VOL, "/atlas_models": MODELS_VOL, "/outputs": OUTPUTS_VOL},
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
    import openai

    sys.path.insert(0, "/atlas")
    from experiment.algotune_prompts import build_algotune_prompts
    from experiment.tasks import get_task

    for problem_id in problems:
        get_task("algotune", problem_id)

    adapter_path = None
    if adapter_name:
        adapter_path = f"/atlas_models/{adapter_name}"
        if not Path(adapter_path).exists():
            raise FileNotFoundError(adapter_path)

    legs: list[tuple[str, str]] = []
    if eval_base:
        legs.append((base_model, "base"))
    if eval_adapter and adapter_path:
        legs.append(("atlas", "atlas"))
    if not legs:
        raise RuntimeError("No evaluation legs requested.")

    vllm_proc = _start_vllm(base_model, adapter_path)
    try:
        client = openai.OpenAI(api_key="EMPTY", base_url=f"http://localhost:{VLLM_PORT}/v1")
        base_dir = Path("/outputs") / "eval" / run_name
        base_dir.mkdir(parents=True, exist_ok=True)

        summaries: list[dict] = []
        for problem_id in problems:
            system_msg, user_msg, _ = build_algotune_prompts(problem_id)
            for served_model, tag in legs:
                _, summary = _sample_and_score(
                    client=client,
                    served_model=served_model,
                    problem_id=problem_id,
                    system_msg=system_msg,
                    user_msg=user_msg,
                    n_samples=n_samples,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    seed=seed,
                    output_dir=base_dir / f"{problem_id}__{tag}",
                )
                summary["tag"] = tag
                summaries.append(summary)

        compare = {
            "run_name": run_name,
            "task_family": "algotune",
            "base_model": base_model,
            "adapter_name": adapter_name,
            "n_samples": n_samples,
            "temperature": temperature,
            "legs": summaries,
        }
        _compute_cross_leg_metrics(compare)
        (base_dir / "compare.json").write_text(json.dumps(compare, indent=2))
        OUTPUTS_VOL.commit()
        return compare
    finally:
        vllm_proc.terminate()
        try:
            vllm_proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            vllm_proc.kill()


@app.local_entrypoint()
def main(
    problems: str,
    run_name: str,
    n_samples: int = 50,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: int = 8192,
    adapter_name: str | None = None,
    base_model: str = BASE_MODEL_DEFAULT,
    seed: int = 42,
    eval_base: bool = True,
    eval_adapter: bool = True,
) -> None:
    problem_list = [problem.strip() for problem in problems.split(",") if problem.strip()]
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
    print(json.dumps(compare, indent=2))
    print(f"modal volume get atlas-openevolve-outputs eval/{run_name} ./eval_runs/")
