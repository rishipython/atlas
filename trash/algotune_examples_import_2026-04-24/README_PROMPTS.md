# AlgoTune Upstream Import (Prompt Review)

Source: `algorithmicsuperintelligence/openevolve/examples/algotune`

## Imported Problems
- `affine_transform_2d`
- `convolve2d_full_fill`
- `eigenvectors_complex`
- `fft_cmplx_scipy_fftpack`
- `fft_convolution`
- `lu_factorization`
- `polynomial_real`
- `psd_cone_projection`

Each problem folder contains:
- `config.yaml` (upstream prompt + tuned settings)
- `initial_program.py`
- `evaluator.py`
- `best_program.py`
- `best_program_info.json`

## Prompt Files
- Full upstream prompts and filtered active prompts are in `manifest.json` under each problem key:
  - `system_prompt_raw`
  - `system_prompt_filtered`
  - `filtered_out_sections`

## Active Sample (3 relatively simple)
- `fft_convolution`
- `convolve2d_full_fill`
- `affine_transform_2d`

These are stored in `manifest.json` at `sample_problem_ids` and exported in Python as `experiment.tasks.algotune.SIMPLE_SAMPLE_PROBLEMS`.

## Current Filtering Rule
For now, the active prompt removes optional all-caps sections such as:
- `PERFORMANCE OPTIMIZATION OPPORTUNITIES`

The removed sections are preserved in `system_prompt_raw` for easy re-enable later.
