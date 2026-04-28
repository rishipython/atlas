# atlas

ATLAS, trimmed down to a single story:

- benchmark: AlgoTune
- base model: `openai/gpt-oss-20b`
- search baseline: OpenEvolve
- learned method: reward-weighted SFT on search trajectories
- final comparison: base Best-of-N vs ATLAS Best-of-N on held-out AlgoTune tasks

## What is left in this branch

This branch keeps only the code needed for the AlgoTune experiment:

- `experiment/openevolve_runner.py`
  Collect OpenEvolve trajectories on vendored AlgoTune tasks.
- `experiment/build_sft_dataset.py`
  Convert OpenEvolve traces or standalone Best-of-N samples into a single
  base-prompt SFT dataset with simple ablation controls.
- `experiment/train_atlas_sft.py`
  Train a LoRA adapter on `gpt-oss-20b` with optional reward weighting.
- `experiment/eval_algotune.py`
  Evaluate base vs adapter on AlgoTune and write reward-vs-k summaries.
- `experiment/plot_algotune_pass40.py`
  Plot the main reward-vs-k curves from `compare.json`.
- `experiment/tasks/algotune.py`
  Registry wrapper around vendored AlgoTune problems.

## Current task inventory

The repo currently vendors three runnable AlgoTune example tasks under
`experiment/tasks/_oe_problems/algotune_examples/`:

- `affine_transform_2d`
- `convolve2d_full_fill`
- `fft_convolution`

If you want the full 5-8 task setup from the experiment plan, add more
AlgoTune task folders in that directory with `initial_program.py`,
`evaluator.py`, and `config.yaml`. The rest of the pipeline will pick them up
automatically.

## Recommended workflow

1. Run OpenEvolve on train tasks to collect trajectories.
2. Build an SFT dataset from those trajectories.
3. Train one adapter.
4. Evaluate base vs adapter on held-out tasks.
5. Plot reward-vs-k.

## Example commands

Collect OpenEvolve data:

```bash
modal run experiment/openevolve_runner.py \
  --problem-id fft_convolution \
  --iterations 30 \
  --run-name oe_fft_train_v1
```

Build the main reward-weighted dataset:

```bash
python experiment/build_sft_dataset.py \
  --trace runs/oe_fft_train_v1/oe/evolution_trace.jsonl \
  --problems fft_convolution \
  --selection all \
  --weight-scheme reward \
  --out data/algotune_reward_weighted.jsonl
```

Build the final-only ablation:

```bash
python experiment/build_sft_dataset.py \
  --trace runs/oe_fft_train_v1/oe/evolution_trace.jsonl \
  --problems fft_convolution \
  --selection best_per_problem \
  --weight-scheme uniform \
  --out data/algotune_final_only.jsonl
```

Train:

```bash
modal run experiment/train_atlas_sft.py \
  --dataset data/algotune_reward_weighted.jsonl \
  --run-name atlas_algotune_reward_v1
```

Evaluate on held-out tasks:

```bash
modal run experiment/eval_algotune.py \
  --problems affine_transform_2d,convolve2d_full_fill \
  --adapter-name atlas_algotune_reward_v1 \
  --run-name algotune_eval_v1 \
  --n-samples 50
```

Plot:

```bash
python experiment/plot_algotune_pass40.py \
  --compare eval_runs/algotune_eval_v1/compare.json \
  --out eval_runs/algotune_eval_v1/reward_vs_k.png
```
