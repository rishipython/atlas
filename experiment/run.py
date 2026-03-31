"""Experiment runner — loads config, instantiates benchmark + agent, runs evaluation."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import yaml

from agent.llm_agent import LLMAgent
from benchmark.kernel.benchmark import KernelBenchmark
from utils.llm_client import LLMClient

logger = logging.getLogger(__name__)

BENCHMARK_REGISTRY: dict[str, type] = {
    "kernel": KernelBenchmark,
}


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_benchmark(cfg: dict):
    btype = cfg["benchmark"]["type"]
    cls = BENCHMARK_REGISTRY[btype]
    kwargs = {k: v for k, v in cfg["benchmark"].items() if k != "type"}
    return cls(**kwargs)


def build_agent(cfg: dict, benchmark):
    acfg = cfg["agent"]
    llm = LLMClient(
        model=acfg["model"],
        api_base=acfg.get("api_base", "http://localhost:8000/v1"),
        api_key=acfg.get("api_key", "EMPTY"),
    )
    return LLMAgent(
        llm=llm,
        temperature=acfg.get("temperature", 0.6),
        max_tokens=acfg.get("max_tokens", 4096),
        eval_fn=benchmark.evaluate,
    )


def run_experiment(cfg: dict, output_dir: Path | None = None):
    benchmark = build_benchmark(cfg)
    agent = build_agent(cfg, benchmark)
    problems = benchmark.get_problems()

    logger.info("Running %d problems", len(problems))
    results = []

    for prob in problems:
        logger.info("=== Problem: %s ===", prob.problem_id)
        agent_result = agent.solve(prob.problem_id, prob.description)

        eval_result = benchmark.evaluate(prob.problem_id, agent_result.solution)
        agent_result.eval_result = eval_result

        results.append({
            "problem_id": prob.problem_id,
            "correct": eval_result.correct,
            "score": eval_result.score,
            "feedback": eval_result.feedback,
            "solution": agent_result.solution,
            "metadata": eval_result.metadata,
        })

        status = "PASS" if eval_result.correct else "FAIL"
        logger.info(
            "[%s] %s  score=%.3f\n%s",
            status, prob.problem_id, eval_result.score, eval_result.feedback,
        )

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = output_dir / f"results_{ts}.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Results saved to %s", out_path)

    n_correct = sum(1 for r in results if r["correct"])
    logger.info(
        "Summary: %d/%d correct, avg score=%.3f",
        n_correct,
        len(results),
        sum(r["score"] for r in results) / max(len(results), 1),
    )
    return results


def main():
    parser = argparse.ArgumentParser(description="ATLAS experiment runner")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    parser.add_argument(
        "--output-dir", type=str, default="outputs",
        help="Directory to save results",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = load_config(args.config)
    run_experiment(cfg, output_dir=Path(args.output_dir))


if __name__ == "__main__":
    main()
