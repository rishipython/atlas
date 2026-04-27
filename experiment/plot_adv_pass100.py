#!/usr/bin/env python3
"""Parse pass@100 results for base vs adv-SFT and produce histograms + table."""
import json
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = Path("/tmp/adv_pass100")
OUT_DIR = Path("/Users/rishi/cs288/atlas/eval_runs/analysis_adv_v1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROBLEMS = ["softmax", "layernorm", "matmul"]
MODELS = ["base", "atlas"]  # "atlas" == adv-SFT here
MODEL_LABEL = {"base": "base gpt-oss-20b", "atlas": "adv-SFT (GRPO-ish)"}

SPEEDUP_RE = re.compile(r"speedup=([0-9.]+)x")


def load_samples(prob: str, model: str):
    path = DATA_DIR / f"{prob}__{model}.samples.json"
    with path.open() as f:
        return json.load(f)


def per_sample_speedups(samples):
    """Return list of per-sample mean speedups, with 0.0 for failed samples."""
    per_sample = []
    per_config = []  # flat across all configs
    for s in samples:
        if s.get("correct"):
            mean_sp = float(s.get("score", 0.0))
            per_sample.append(mean_sp)
            hits = SPEEDUP_RE.findall(s.get("feedback", ""))
            per_config.extend(float(h) for h in hits)
        else:
            per_sample.append(0.0)
    return per_sample, per_config


def summarize(per_sample, per_config):
    arr = np.array(per_sample)
    n = len(arr)
    correct_mask = arr > 0
    n_correct = int(correct_mask.sum())
    best = float(arr.max()) if n else 0.0
    mean_all = float(arr.mean()) if n else 0.0
    std_all = float(arr.std(ddof=1)) if n > 1 else 0.0
    if n_correct > 0:
        mean_corr = float(arr[correct_mask].mean())
        std_corr = float(arr[correct_mask].std(ddof=1)) if n_correct > 1 else 0.0
    else:
        mean_corr = 0.0
        std_corr = 0.0
    cfg = np.array(per_config) if per_config else np.array([])
    return {
        "n": n,
        "n_correct": n_correct,
        "pass_at_k": n_correct / n if n else 0.0,
        "best": best,
        "mean_all": mean_all,
        "std_all": std_all,
        "mean_corr": mean_corr,
        "std_corr": std_corr,
        "per_config_mean": float(cfg.mean()) if cfg.size else 0.0,
        "per_config_std": float(cfg.std(ddof=1)) if cfg.size > 1 else 0.0,
        "per_config_n": int(cfg.size),
    }


def main():
    stats = {}
    per_sample = {}
    per_config = {}
    for prob in PROBLEMS:
        for model in MODELS:
            samples = load_samples(prob, model)
            ps, pc = per_sample_speedups(samples)
            per_sample[(prob, model)] = ps
            per_config[(prob, model)] = pc
            stats[(prob, model)] = summarize(ps, pc)

    # --- Histograms: one figure, 3 rows x 2 cols (problem x model) overlay style ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=False)
    colors = {"base": "#4c72b0", "atlas": "#c44e52"}
    for j, prob in enumerate(PROBLEMS):
        ax = axes[j]
        all_vals = np.concatenate([
            np.array(per_sample[(prob, "base")]),
            np.array(per_sample[(prob, "atlas")]),
        ])
        bin_max = max(1.0, float(np.ceil(all_vals.max() * 10) / 10))
        bins = np.linspace(0.0, bin_max, 25)
        for model in MODELS:
            vals = np.array(per_sample[(prob, model)])
            ax.hist(
                vals,
                bins=bins,
                alpha=0.55,
                label=MODEL_LABEL[model],
                color=colors[model],
                edgecolor="black",
                linewidth=0.4,
            )
            mean = stats[(prob, model)]["mean_all"]
            ax.axvline(mean, color=colors[model], linestyle="--", linewidth=1.2)
        ax.set_title(f"{prob}  (n=100, fails=0x)")
        ax.set_xlabel("per-sample mean speedup (x)")
        if j == 0:
            ax.set_ylabel("count")
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.25)
    fig.suptitle(
        "pass@100 speedup distribution — base vs adv-SFT (GRPO-ish, single phase)",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = OUT_DIR / "speedup_hist_all_samples.png"
    fig.savefig(out_path, dpi=140)
    print(f"wrote {out_path}")

    # --- Correct-only histograms (zoomed in on the interesting region) ---
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4.2), sharey=False)
    for j, prob in enumerate(PROBLEMS):
        ax = axes2[j]
        all_vals = []
        for model in MODELS:
            vals = np.array(per_sample[(prob, model)])
            all_vals.append(vals[vals > 0])
        concat = np.concatenate(all_vals) if any(v.size for v in all_vals) else np.array([0.1])
        bin_min = max(0.0, float(np.floor(concat.min() * 10) / 10))
        bin_max = max(bin_min + 0.1, float(np.ceil(concat.max() * 10) / 10))
        bins = np.linspace(bin_min, bin_max, 20)
        for model in MODELS:
            vals = np.array(per_sample[(prob, model)])
            vals = vals[vals > 0]
            if vals.size == 0:
                continue
            ax.hist(
                vals,
                bins=bins,
                alpha=0.55,
                label=f"{MODEL_LABEL[model]} (n={vals.size})",
                color=colors[model],
                edgecolor="black",
                linewidth=0.4,
            )
            ax.axvline(
                vals.mean(), color=colors[model], linestyle="--", linewidth=1.2
            )
        ax.axvline(1.0, color="black", linestyle=":", linewidth=1.0, label="1x (torch)")
        ax.set_title(f"{prob} — correct samples only")
        ax.set_xlabel("per-sample mean speedup (x)")
        if j == 0:
            ax.set_ylabel("count")
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.25)
    fig2.suptitle(
        "pass@100 speedup distribution (correct only) — base vs adv-SFT (GRPO-ish)",
        fontsize=13,
    )
    fig2.tight_layout(rect=[0, 0, 1, 0.95])
    out_path2 = OUT_DIR / "speedup_hist_correct_only.png"
    fig2.savefig(out_path2, dpi=140)
    print(f"wrote {out_path2}")

    # --- Table ---
    def fmt(x, nd=3):
        return f"{x:.{nd}f}"

    header = (
        "| problem   | model              |   n | correct | pass@100 | mean speedup (all, fail=0) | std (all) | mean speedup (correct) | std (correct) | best  |"
    )
    sep = (
        "|-----------|--------------------|----:|--------:|---------:|---------------------------:|----------:|-----------------------:|--------------:|------:|"
    )
    rows = [header, sep]
    rows_out = []
    for prob in PROBLEMS:
        for model in MODELS:
            s = stats[(prob, model)]
            rows.append(
                "| {prob:<9} | {model:<18} | {n:>3} | {nc:>7} | {pk:>8} | {ma:>26} | {sa:>9} | {mc:>22} | {sc:>13} | {best:>5} |".format(
                    prob=prob,
                    model=MODEL_LABEL[model],
                    n=s["n"],
                    nc=s["n_correct"],
                    pk=fmt(s["pass_at_k"]),
                    ma=fmt(s["mean_all"]),
                    sa=fmt(s["std_all"]),
                    mc=fmt(s["mean_corr"]),
                    sc=fmt(s["std_corr"]),
                    best=fmt(s["best"], 2),
                )
            )
            rows_out.append(
                {
                    "problem": prob,
                    "model": MODEL_LABEL[model],
                    **s,
                }
            )
    table_md = "\n".join(rows)
    (OUT_DIR / "table.md").write_text(table_md + "\n")
    with (OUT_DIR / "stats.json").open("w") as f:
        json.dump(rows_out, f, indent=2)
    print("\n" + table_md + "\n")
    print(f"wrote {OUT_DIR / 'table.md'}")
    print(f"wrote {OUT_DIR / 'stats.json'}")


if __name__ == "__main__":
    main()
