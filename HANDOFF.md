# ATLAS Handoff (FFT Distillation + OE)

Last updated: 2026-04-27 (local)

This handoff is focused on the latest working experiment set: `fft_convolution` with OpenEvolve (OE), synth-trajectory distillation, and pass@40 evaluation.

## 1) TL;DR results (latest working setup)

All numbers below are for `fft_convolution` with seed 42 and `n_samples=40` for pass@40 evals.

| Method | pass@40 | Best speedup (correct-only) |
|---|---:|---:|
| Base (`openai/gpt-oss-20b`) | 0.35 | 1.3567x |
| Distill: BestSpeed-only (`atlas_fftconv_bestspeed_only_v1`) | 0.40 | 1.4394x |
| Distill: TwoStage P2 (`atlas_fftconv_twostage_speed_p2_v1`) | 0.35 | 1.3389x |
| OpenEvolve (`oe_fft_convolution_nodiff_s42_v1`) | n/a (not pass@k) | 1.4846x |

Definition used for speedup reporting: **max speedup among trajectories with correctness >= 0.99**.

## 2) What changed to make OE better than base

For the improved FFT OE run we used repo-style settings and disabled diff-style mutation:
- `task_family=algotune`
- `problem_id=fft_convolution`
- preset-backed config in `experiment/openevolve_runner.py` (`_REPO_PRESETS['algotune']`)
- `diff_based_evolution=False`
- temperature 0.4
- 40 iterations, seed 42

## 3) End-to-end commands (repro)

Run from repo root: `/Users/rishi/cs288/atlas`.

### 3.1 OpenEvolve (FFT)

```bash
/Users/rishi/miniconda3/envs/atlas/bin/modal run -d experiment/openevolve_runner.py \
  --task-family algotune \
  --problem-id fft_convolution \
  --iterations 40 \
  --random-seed 42 \
  --run-name oe_fft_convolution_nodiff_s42_v1
```

Download trace/artifacts:

```bash
/Users/rishi/miniconda3/envs/atlas/bin/modal volume get atlas-openevolve-outputs \
  oe_fft_convolution_nodiff_s42_v1 ./runs/
```

### 3.2 Synthesize trajectories from OE trace

```bash
/Users/rishi/miniconda3/envs/atlas/bin/modal run experiment/synth_reasoning.py::main_all \
  --trace-path ./runs/oe_fft_convolution_nodiff_s42_v1/oe/evolution_trace.jsonl \
  --problem-id fft_convolution \
  --out-name synth_fft_nodiff_all_v2
```

Download synth outputs:

```bash
/Users/rishi/miniconda3/envs/atlas/bin/modal volume get atlas-openevolve-outputs \
  synth/synth_fft_nodiff_all_v2 ./runs/
```

### 3.3 Build distillation datasets (correctness + speed + best-only)

```bash
/Users/rishi/miniconda3/envs/atlas/bin/python -m experiment.build_fft_speedup_sft_datasets \
  --trace ./runs/oe_fft_convolution_nodiff_s42_v1/oe/evolution_trace.jsonl \
  --synth-dir ./runs/synth_fft_nodiff_all_v2 \
  --problem-id fft_convolution \
  --out-p1 data/sft/fftconv_nodiff_twostage_p1_v1.jsonl \
  --out-p2 data/sft/fftconv_nodiff_twostage_p2_v1.jsonl \
  --out-best data/sft/fftconv_nodiff_bestspeed_only_v1.jsonl \
  --correct-threshold 0.99
```

### 3.4 Train distilled adapters

BestSpeed-only:

```bash
/Users/rishi/miniconda3/envs/atlas/bin/modal run experiment/train_atlas_sft.py \
  --dataset data/sft/fftconv_nodiff_bestspeed_only_v1.jsonl \
  --phase phase1 \
  --run-name atlas_fftconv_bestspeed_only_v1 \
  --epochs 3 --learning-rate 5e-5
```

TwoStage phase 1 (correctness):

```bash
/Users/rishi/miniconda3/envs/atlas/bin/modal run experiment/train_atlas_sft.py \
  --dataset data/sft/fftconv_nodiff_twostage_p1_v1.jsonl \
  --phase phase1 \
  --run-name atlas_fftconv_twostage_corr_p1_v1 \
  --epochs 3 --learning-rate 5e-5
```

TwoStage phase 2 (speedup, resume from P1):

```bash
/Users/rishi/miniconda3/envs/atlas/bin/modal run experiment/train_atlas_sft.py \
  --dataset data/sft/fftconv_nodiff_twostage_p2_v1.jsonl \
  --phase phase2 \
  --resume-from atlas_fftconv_twostage_corr_p1_v1 \
  --run-name atlas_fftconv_twostage_speed_p2_v1 \
  --epochs 3 --learning-rate 5e-5
```

### 3.5 Evaluate pass@40

Base:

```bash
/Users/rishi/miniconda3/envs/atlas/bin/modal run experiment/eval_algotune.py \
  --run-name base_pass40_algotune_evalresultfix_v2 \
  --problems fft_convolution \
  --n-samples 40 --seed 42 \
  --no-eval-adapter
```

BestSpeed adapter:

```bash
/Users/rishi/miniconda3/envs/atlas/bin/modal run experiment/eval_algotune.py \
  --run-name eval_fft_bestspeed_only_v1 \
  --problems fft_convolution \
  --n-samples 40 --seed 42 \
  --adapter-name atlas_fftconv_bestspeed_only_v1 \
  --no-eval-base
```

TwoStage adapter:

```bash
/Users/rishi/miniconda3/envs/atlas/bin/modal run experiment/eval_algotune.py \
  --run-name eval_fft_twostage_p2_v1 \
  --problems fft_convolution \
  --n-samples 40 --seed 42 \
  --adapter-name atlas_fftconv_twostage_speed_p2_v1 \
  --no-eval-base
```

## 4) What is on Modal and where it is stored

## 4.1 Volume: `atlas-openevolve-outputs`

### OE runs (root-level directories)
- `oe_fft_convolution_nodiff_s42_v1`
- `oe_fft_convolution_oe_match_s42_v1`
- `oe_convolve2d_full_fill_oe_match_s42_v1`
- `oe_affine_transform_2d_oe_match_s42_v1`

For OE run `oe_fft_convolution_nodiff_s42_v1`, key file:
- `oe_fft_convolution_nodiff_s42_v1/oe/evolution_trace.jsonl`

### Synth outputs
Under `/synth`:
- `synth/synth_fft_nodiff_all_v2`
- `synth/synth_fft_nodiff_qualitycheck_v1`
- `synth/synth_pairwise_oe_repoconf_s42_all_v1`

### Eval outputs
Under `/eval`:
- `eval/base_pass40_algotune_evalresultfix_v2`
- `eval/eval_fft_bestspeed_only_v1`
- `eval/eval_fft_twostage_p2_v1`
- `eval/base_pass40_other2_evalfix_v3` (partial/incomplete)
- `eval/eval_other2_bestspeed_only_v2` (partial/incomplete)
- `eval/eval_other2_twostage_p2_v2` (partial/incomplete)

Key summary files used for latest FFT table:
- `eval/base_pass40_algotune_evalresultfix_v2/fft_convolution__base/summary.json`
- `eval/eval_fft_bestspeed_only_v1/fft_convolution__atlas/summary.json`
- `eval/eval_fft_twostage_p2_v1/fft_convolution__atlas/summary.json`

## 4.2 Volume: `atlas-models`

Adapters for latest FFT distillation:
- `atlas_fftconv_bestspeed_only_v1`
- `atlas_fftconv_twostage_corr_p1_v1`
- `atlas_fftconv_twostage_speed_p2_v1`

Each run directory contains adapter artifacts and `training_summary.json`.

## 4.3 Volume: `atlas-hf-cache`

Shared model/tokenizer cache for vLLM and training containers.
No experiment-specific logic here; just cache.

## 5) Download cheatsheet

Get OE run locally:

```bash
/Users/rishi/miniconda3/envs/atlas/bin/modal volume get atlas-openevolve-outputs \
  oe_fft_convolution_nodiff_s42_v1 ./runs/
```

Get synth run locally:

```bash
/Users/rishi/miniconda3/envs/atlas/bin/modal volume get atlas-openevolve-outputs \
  synth/synth_fft_nodiff_all_v2 ./runs/
```

Get eval outputs locally:

```bash
/Users/rishi/miniconda3/envs/atlas/bin/modal volume get atlas-openevolve-outputs \
  eval/eval_fft_bestspeed_only_v1 ./eval_runs/
```

Get adapter locally:

```bash
/Users/rishi/miniconda3/envs/atlas/bin/modal volume get atlas-models \
  atlas_fftconv_bestspeed_only_v1 ./atlas_models/
```

## 6) Runtime/cost-relevant timing from latest FFT pipeline

- OE (`oe_fft_convolution_nodiff_s42_v1`): ~8m34s from trace timestamps.
- Synth trajectory generation (`atlas-synth-reasoning` for `synth_fft_nodiff_all_v2`): ~14m wall-clock in Modal app runtime.
- Distill train runtime (from `training_summary.json`):
  - BestSpeed-only: 8.62s
  - TwoStage P1: 225.36s
  - TwoStage P2: 167.40s
  - TwoStage total train runtime: 392.76s (~6m33s)

## 7) Current caveats / incomplete items

- The multi-problem cross-generalization eval runs for `convolve2d_full_fill` and `affine_transform_2d` under:
  - `eval/base_pass40_other2_evalfix_v3`
  - `eval/eval_other2_bestspeed_only_v2`
  - `eval/eval_other2_twostage_p2_v2`
  are **incomplete** (no `summary.json`; second problem leg not reached).
- Therefore those cells should be treated as `n/a` until rerun.

## 8) Quick “run everything again” checklist

1. Launch OE (`oe_fft_convolution_nodiff_s42_v1` style).
2. Download OE trace.
3. Run synth `main_all` and download synth output.
4. Build three FFT datasets (`out-p1`, `out-p2`, `out-best`).
5. Train:
   - bestspeed-only
   - twostage p1
   - twostage p2 (resume from p1)
6. Evaluate pass@40 on FFT for base + both adapters.
7. Report with correct-only speedup metric (correctness >= 0.99).
