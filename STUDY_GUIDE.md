# ATLAS Study Guide

This repo now includes a small experiment harness for the comparison matrix and
ablations discussed in the planning note.

For a single-machine Colab workflow, start with [COLAB_GUIDE.md](/Users/sandeep/Documents/atlas/COLAB_GUIDE.md).

## Main entry points

- `experiment/research_suite.py`
  - Writes runnable shell scripts for the full study.
  - Use `--backend local` on Colab, or `--backend modal` if you later switch back.
- `experiment/eval_standalone.py`
  - Standalone base vs adapter evaluation with pass@k, correctness rate,
    expected speedup, mean speedup among correct samples, and best speedup.
- `experiment/local_eval_standalone.py`
  - Local, non-Modal version of the standalone evaluator.
- `experiment/local_openevolve_runner.py`
  - Local, non-Modal OpenEvolve runner.
- `experiment/local_train_atlas_sft.py`
  - Local, non-Modal reward-weighted LoRA training.
- `experiment/analyze_search.py`
  - Summarizes OpenEvolve runs as search-efficiency curves.
- `experiment/summarize_study.py`
  - Merges standalone and search summaries into one report.
- `experiment/build_sft_dataset.py`
  - Now supports ablation-friendly switches such as:
    - `--selection all|best-per-source|best-per-problem`
    - `--weight-scheme raw_score|binary|uniform|strict_speedup`
    - `--correct-only`

## Suggested workflow

1. Generate a study workspace:
   - `python experiment/research_suite.py --backend local --out-dir study_runs/atlas_research_suite`
2. On Colab / remote GPU, run the generated shell scripts in order.
3. Download the `eval_runs/` and `runs/` artifacts.
4. Run the aggregation script:
   - `bash study_runs/atlas_research_suite/05_aggregate_results.sh`
5. Share the resulting JSON summaries for analysis.

## Notes on the ablations

- Final-only vs trajectory-aware:
  - `build_sft_dataset.py --selection best-per-source` vs `--selection all`
- Binary reward vs latency-aware reward:
  - `--weight-scheme binary` vs `--weight-scheme raw_score`
- Positive-only SFT:
  - `--correct-only --weight-scheme uniform`
- Successful-only vs include-failures:
  - `--correct-only` vs leaving failures in with `--min-score 0.0`
- Direct vs edit trajectory:
  - `--phase 2` (base prompt) vs `--phase 1` (OE edit-context prompt)

The generated shell scripts use `softmax` as the default ablation task because
that is the most mature path in the current repo, but the same commands can be
retargeted to `layernorm` or `matmul` once matching traces exist.
