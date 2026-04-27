# Colab A100 Guide

This repo can now run without Modal on a single Colab Pro A100.

## Is one A100 enough?

Yes for the current priority experiments, with a few caveats:

- `Base` vs `ATLAS` standalone eval: yes
- `Base + OpenEvolve` vs `ATLAS + OpenEvolve`: yes, but slower because vLLM and kernel evaluation share one GPU
- reward-weighted SFT LoRA training on `gpt-oss-20b`: yes, this is exactly the GPU class the training recipe was designed around
- full experiment matrix: feasible, but long-running; do the must-run subset first

Practical recommendation:

1. Start with `softmax` only to verify the stack
2. Then run the full `softmax,layernorm,matmul` comparison matrix
3. Leave the larger ablation sweep for last

## One-time setup in Colab

```bash
!git clone <your repo>
%cd atlas
!bash setup/colab_install.sh
```

If the flash-attn wheel URL mismatches your Colab Python version, run:

```bash
!python --version
```

and swap the `cp311` tag in `setup/colab_install.sh` if needed.

## Generate local study scripts

```bash
!python experiment/research_suite.py \
  --backend local \
  --out-dir study_runs/atlas_research_suite \
  --n-samples 100 \
  --search-iterations 100
```

## Run the must-run experiments first

```bash
!bash study_runs/atlas_research_suite/01_run_standalone.sh
!bash study_runs/atlas_research_suite/02_run_search.sh
```

These test:

- `Base`
- `ATLAS`
- `Base + OpenEvolve`
- `ATLAS + OpenEvolve`

on:

- `softmax`
- `layernorm`
- `matmul`

## Train on one task, test on the others

```bash
!bash study_runs/atlas_research_suite/03_train_on_one_test_on_others.sh
```

This assumes the referenced OpenEvolve traces already exist under `runs/`.

## Run the ablations

```bash
!bash study_runs/atlas_research_suite/04_run_ablations.sh
```

Default ablation target is currently `softmax`.

## Aggregate results

```bash
!bash study_runs/atlas_research_suite/05_aggregate_results.sh
```

Main outputs:

- `study_outputs/final_report.json`
- `study_outputs/search_summary.json`
- `eval_runs/.../compare.json`
- `runs/.../summary.json`

## Local entry points

- `experiment/local_eval_standalone.py`
  - repeated sampling / pass@k / expected-speedup
- `experiment/local_openevolve_runner.py`
  - local OpenEvolve search run
- `experiment/local_train_atlas_sft.py`
  - local reward-weighted LoRA training
- `experiment/analyze_search.py`
  - summarize search traces
- `experiment/summarize_study.py`
  - merge standalone + search summaries

## Suggested first smoke test

```bash
!python experiment/local_eval_standalone.py \
  --problems softmax \
  --run-name smoke_softmax_base \
  --n-samples 8 \
  --no-eval-adapter
```

Then, once you have a local adapter directory like `atlas_models/atlas_softmax_adv_sft`, run:

```bash
!python experiment/local_eval_standalone.py \
  --problems softmax \
  --run-name smoke_softmax_compare \
  --adapter atlas_models/atlas_softmax_adv_sft \
  --n-samples 8
```
