"""Generate and aggregate the ATLAS research experiment suite.

The goal is not to execute anything locally. Instead this script writes a small
study workspace with shell scripts that you can run on a GPU box / Colab,
covering:

- Base vs Search vs ATLAS vs ATLAS+Search
- Train-on-one, test-on-others
- pass@k / expected-speedup evaluation
- the priority ablations requested by the user
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TASKS = ["softmax", "layernorm", "matmul"]
DEFAULT_PASS_KS = "1,5,10,20,50,100"


def _sh_header(title: str) -> str:
    return (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n\n"
        f"# {title}\n"
        f"cd {REPO_ROOT}\n\n"
    )


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o755)


def build_plan(
    out_dir: Path,
    *,
    tasks: list[str],
    base_model: str,
    n_samples: int,
    search_iterations: int,
    pass_ks: str,
) -> dict:
    problems_csv = ",".join(tasks)
    adapters = {task: f"atlas_{task}_adv_sft" for task in tasks}
    manifest = {
        "tasks": tasks,
        "base_model": base_model,
        "n_samples": n_samples,
        "search_iterations": search_iterations,
        "pass_ks": pass_ks,
        "adapters": adapters,
    }

    # 1. Base vs ATLAS standalone -------------------------------------------------
    standalone = _sh_header("Standalone base vs ATLAS eval") + f"""
# Expected outputs land under ./eval_runs after `modal volume get`.
for task in {" ".join(tasks)}; do
  adapter_var="atlas_${{task}}_adv_sft"
  modal run experiment/eval_standalone.py \
    --problems "$task" \
    --run-name "study_standalone_${{task}}" \
    --adapter-name "$adapter_var" \
    --n-samples {n_samples} \
    --pass-ks "{pass_ks}" \
    --base-model "{base_model}"
done
"""
    _write(out_dir / "01_run_standalone.sh", standalone)

    # 2. Base+OE vs ATLAS+OE ------------------------------------------------------
    search = _sh_header("OpenEvolve search runs for base and ATLAS") + f"""
for task in {" ".join(tasks)}; do
  modal run experiment/openevolve_runner.py \
    --problem-id "$task" \
    --iterations {search_iterations} \
    --run-name "study_base_search_${{task}}"

  modal run experiment/openevolve_runner.py \
    --problem-id "$task" \
    --iterations {search_iterations} \
    --adapter-name "atlas_${{task}}_adv_sft" \
    --run-name "study_atlas_search_${{task}}"
done
"""
    _write(out_dir / "02_run_search.sh", search)

    # 3. Train-on-one/test-on-others ---------------------------------------------
    cross_generalization = _sh_header("Train-on-one, test-on-others") + f"""
# Assumes one OE trace per training task already exists.
for train_task in {" ".join(tasks)}; do
  python experiment/build_sft_dataset.py \
    --traces "runs/${{train_task}}_v3_twophase/oe/evolution_trace.jsonl" \
    --phase 2 \
    --out-phase2 "data/sft/${{train_task}}_trajectory_phase2.jsonl" \
    --weight-scheme raw_score \
    --selection all \
    --min-score 0.0

  modal run experiment/train_atlas_sft.py \
    --dataset "data/sft/${{train_task}}_trajectory_phase2.jsonl" \
    --phase phase2 \
    --run-name "atlas_${{train_task}}_trajectory" \
    --base-model "{base_model}"

  modal run experiment/eval_standalone.py \
    --problems "{problems_csv}" \
    --run-name "study_transfer_${{train_task}}" \
    --adapter-name "atlas_${{train_task}}_trajectory" \
    --n-samples {n_samples} \
    --pass-ks "{pass_ks}" \
    --base-model "{base_model}" \
    --eval-base false \
    --eval-adapter true
done
"""
    _write(out_dir / "03_train_on_one_test_on_others.sh", cross_generalization)

    # 4. Ablations ----------------------------------------------------------------
    ablations = _sh_header("Priority ablations") + f"""
TASK=softmax
TRACE="runs/${{TASK}}_v3_twophase/oe/evolution_trace.jsonl"

# A1. Final-only vs trajectory-aware
python experiment/build_sft_dataset.py \
  --traces "$TRACE" \
  --phase 2 \
  --out-phase2 data/sft/${{TASK}}_final_only_phase2.jsonl \
  --selection best-per-source \
  --weight-scheme raw_score \
  --min-score 0.0

python experiment/build_sft_dataset.py \
  --traces "$TRACE" \
  --phase 2 \
  --out-phase2 data/sft/${{TASK}}_trajectory_all_phase2.jsonl \
  --selection all \
  --weight-scheme raw_score \
  --min-score 0.0

modal run experiment/train_atlas_sft.py \
  --dataset data/sft/${{TASK}}_final_only_phase2.jsonl \
  --phase phase2 \
  --run-name atlas_${{TASK}}_final_only \
  --base-model "{base_model}"

modal run experiment/train_atlas_sft.py \
  --dataset data/sft/${{TASK}}_trajectory_all_phase2.jsonl \
  --phase phase2 \
  --run-name atlas_${{TASK}}_trajectory_all \
  --base-model "{base_model}"

# A2. Binary reward vs latency-aware reward
python experiment/build_sft_dataset.py \
  --traces "$TRACE" \
  --phase 2 \
  --out-phase2 data/sft/${{TASK}}_binary_reward_phase2.jsonl \
  --selection all \
  --weight-scheme binary \
  --min-score 0.0

python experiment/build_sft_dataset.py \
  --traces "$TRACE" \
  --phase 2 \
  --out-phase2 data/sft/${{TASK}}_latency_reward_phase2.jsonl \
  --selection all \
  --weight-scheme raw_score \
  --min-score 0.0

# A3. Positive-only SFT vs DPO vs advantage-weighted SFT
python experiment/build_sft_dataset.py \
  --traces "$TRACE" \
  --phase 2 \
  --out-phase2 data/sft/${{TASK}}_positive_only_phase2.jsonl \
  --selection all \
  --correct-only \
  --weight-scheme uniform \
  --min-score 0.0

python experiment/build_dpo_dataset.py \
  --traces "$TRACE" \
  --out data/dpo/${{TASK}}_study_dpo.jsonl

# Fill SYNTH_DIR with the directory produced by experiment/synth_reasoning.py
# before running the next command.
# python experiment/build_advantage_sft.py \
#   --trace "$TRACE" \
#   --synth-dir /tmp/synth_${{TASK}} \
#   --out data/sft/${{TASK}}_advantage_phase2.jsonl \
#   --problem-id "$TASK"

# A4. Successful-only vs include failures
python experiment/build_sft_dataset.py \
  --traces "$TRACE" \
  --phase 2 \
  --out-phase2 data/sft/${{TASK}}_success_only_phase2.jsonl \
  --selection all \
  --correct-only \
  --weight-scheme raw_score \
  --min-score 0.0

python experiment/build_sft_dataset.py \
  --traces "$TRACE" \
  --phase 2 \
  --out-phase2 data/sft/${{TASK}}_with_failures_phase2.jsonl \
  --selection all \
  --weight-scheme raw_score \
  --min-score 0.0

# A5. Direct trajectory vs edit trajectory
python experiment/build_sft_dataset.py \
  --traces "$TRACE" \
  --phase 2 \
  --out-phase2 data/sft/${{TASK}}_direct_phase2.jsonl \
  --selection all \
  --weight-scheme raw_score \
  --min-score 0.0

python experiment/build_sft_dataset.py \
  --traces "$TRACE" \
  --phase 1 \
  --out-phase1 data/sft/${{TASK}}_edit_phase1.jsonl \
  --selection all \
  --weight-scheme raw_score \
  --min-score 0.0
"""
    _write(out_dir / "04_run_ablations.sh", ablations)

    # 5. Aggregation ---------------------------------------------------------------
    aggregate = _sh_header("Aggregate standalone + search outputs") + f"""
python experiment/analyze_search.py \
  --traces \
  runs/study_base_search_softmax/oe/evolution_trace.jsonl \
  runs/study_atlas_search_softmax/oe/evolution_trace.jsonl \
  runs/study_base_search_layernorm/oe/evolution_trace.jsonl \
  runs/study_atlas_search_layernorm/oe/evolution_trace.jsonl \
  runs/study_base_search_matmul/oe/evolution_trace.jsonl \
  runs/study_atlas_search_matmul/oe/evolution_trace.jsonl \
  --pass-ks "{pass_ks}" \
  --out study_outputs/search_summary.json

python experiment/summarize_study.py \
  --standalone eval_runs/study_standalone_softmax/compare.json \
               eval_runs/study_standalone_layernorm/compare.json \
               eval_runs/study_standalone_matmul/compare.json \
  --search study_outputs/search_summary.json \
  --out study_outputs/final_report.json
"""
    _write(out_dir / "05_aggregate_results.sh", aggregate)

    _write(out_dir / "manifest.json", json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="study_runs/atlas_research_suite")
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--base-model", default="openai/gpt-oss-20b")
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--search-iterations", type=int, default=100)
    parser.add_argument("--pass-ks", default=DEFAULT_PASS_KS)
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    manifest = build_plan(
        Path(args.out_dir),
        tasks=tasks,
        base_model=args.base_model,
        n_samples=args.n_samples,
        search_iterations=args.search_iterations,
        pass_ks=args.pass_ks,
    )
    print(json.dumps(manifest, indent=2))
    print(f"\nWrote study plan under {args.out_dir}")


if __name__ == "__main__":
    main()
