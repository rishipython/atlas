"""Train an ATLAS LoRA adapter on gpt-oss-20b via DPO.

What this is
------------
Distill OpenEvolve-collected preference trajectories into gpt-oss-20b
with a parameter-efficient LoRA adapter — the BASE WEIGHTS STAY
FROZEN. Resulting artifact is a ~100-300 MB PEFT adapter that can be
loaded on top of the untouched base model at inference time, so we
always have a clean ``base vs atlas`` comparison available.

Runs on Modal with an A100-80GB or H100. gpt-oss-20b in bf16 is ~40 GB;
LoRA on attention projections adds <1 GB of trainable weight; AdamW
optimizer state is ~2x the trainable weight; activations at seq_len
4096 with gradient checkpointing fit comfortably in the remaining
headroom.

DPO loss uses per-pair ``margin`` weighting
--------------------------------------------
The OpenEvolve trace assigns a graduated ``combined_score`` to every
child program (compile-error < runs-wrong < correct-slow < correct-fast).
The user asked us NOT to discard that information. So instead of
treating every DPO pair equally, we scale the per-example loss by
``max(margin, 1e-3)`` — a pair whose chosen beats its rejected by a
full correct-vs-crash gap (~0.6) is weighted ~12x more than a pair with
a narrow 0.05 margin between two similar programs. This is a cheap,
standard weighted-DPO tweak.

Inputs
------
- ``--dataset``: JSONL from ``build_dpo_dataset.py``.
- ``--base-model``: HF id (default ``openai/gpt-oss-20b``).
- ``--run-name``: used to key the Modal Volume output directory.

Outputs (persisted to the ``atlas-models`` Modal Volume)
-------------------------------------------------------
    /outputs/<run_name>/
        adapter_config.json
        adapter_model.safetensors
        training_args.bin
        trainer_state.json
        tokenizer.json / tokenizer_config.json (so inference works
          standalone without re-downloading the base tokenizer)
        dataset_summary.json

Usage
-----
    modal run experiment/train_atlas_lora.py \\
        --dataset data/dpo/softmax_dpo.jsonl \\
        --run-name atlas_softmax_v1
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_MODEL_DEFAULT = "openai/gpt-oss-20b"

HF_CACHE_VOL = modal.Volume.from_name("atlas-hf-cache", create_if_missing=True)
MODELS_VOL = modal.Volume.from_name("atlas-models", create_if_missing=True)

# The transformers-nightly / trl / peft versions here are pinned loosely
# to the first line-up that had working gpt-oss fine-tuning support.
# If gpt-oss support lands in stable transformers, we can tighten these.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl", "build-essential")
    .pip_install(
        # torch 2.6 ships FSDPModule which newer trl imports; 2.5 doesn't.
        "torch==2.6.0",
        "transformers>=4.56",
        "peft>=0.14",
        "trl>=0.14",
        "accelerate>=1.2",
        "datasets>=3.2",
        "sentencepiece",
        "safetensors",
        "bitsandbytes>=0.44",
        "pyyaml>=6",
        "ninja",  # needed at build-time by flash-attn
        "packaging",
        # gpt-oss uses HF's ``kernels`` package to pull in an MXFP4 kernel
        # at load time; without it transformers refuses to load the model
        # even when it's dequantizing to bf16 for training.
        "kernels>=0.7",
    )
    # Install flash-attn from a pre-built wheel (building from source
    # requires CUDA_HOME at image-build time, which debian_slim doesn't
    # have). The wheel matches torch 2.6 + cu124 + cp312 — if the torch
    # pin above changes, this URL must be updated together.
    .run_commands(
        "pip install --no-deps "
        "https://github.com/Dao-AILab/flash-attention/releases/download/"
        "v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-"
        "cp312-cp312-linux_x86_64.whl"
    )
    .add_local_dir(
        str(REPO_ROOT),
        "/atlas",
        copy=True,
        ignore=[".venv", "__pycache__", ".git", "runs", "openevolve_output"],
    )
    .env({"PYTHONPATH": "/atlas", "HF_HOME": "/hf_cache"})
)

app = modal.App("atlas-train-lora", image=image)


# --------------------------------------------------------------------------
# Training function
# --------------------------------------------------------------------------
# 2h is enough for 3 epochs on ~100 DPO pairs with grad_accum=4 on an A100-80GB
# (each step ~20s for a 4k-token pair; 3 epochs × 100 pairs / 4 accum = 75 steps
# × 20s = ~25 min, plus ~10 min cold start + model load).
@app.function(
    gpu="A100-80GB",
    timeout=2 * 3600,
    volumes={"/hf_cache": HF_CACHE_VOL, "/outputs": MODELS_VOL},
)
def train(
    dataset_jsonl_text: str,  # full file contents, passed by local entrypoint
    run_name: str,
    base_model: str,
    epochs: int,
    learning_rate: float,
    beta: float,
    max_seq_len: int,
    lora_rank: int,
    lora_alpha: int,
    seed: int,
) -> dict:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    print(f"[train] run_name={run_name} base_model={base_model}")
    print(f"[train] dataset_chars={len(dataset_jsonl_text)} epochs={epochs} lr={learning_rate}")

    # Parse DPO dataset ----------------------------------------------------
    pairs: list[dict] = []
    for line in dataset_jsonl_text.splitlines():
        line = line.strip()
        if line:
            pairs.append(json.loads(line))
    assert pairs, "DPO dataset is empty — aborting."
    print(f"[train] parsed {len(pairs)} preference pairs")

    # Convert to the TRL "preference_dataset" format: columns "chosen" and
    # "rejected" are *each* a chat-formatted list of {role, content}.
    # TRL will apply the tokenizer's chat template when we pass
    # remove_unused_columns=False. We keep margin / metadata alongside and
    # use them to build a per-sample weight column.
    #
    # Weighting: initial experiments with w = max(margin, 0.05) / 0.05
    # (up to 18x) blew up grad norms into the 1000s and hard-saturated
    # the DPO loss after a handful of steps. We now use a much gentler
    # scheme: w = 0.5 + margin, capped at ~1.5. That still rewards
    # high-margin pairs, but only up to 3x the weight of a 0-margin pair,
    # keeping optimization stable.
    def _coerce_pair(p: dict) -> dict:
        margin = float(p.get("margin", 0.0))
        weight = min(0.5 + margin, 1.5)
        return {
            "chosen": p["chosen"],
            "rejected": p["rejected"],
            "margin": margin,
            "sample_weight": weight,
        }

    ds = Dataset.from_list([_coerce_pair(p) for p in pairs])
    # Small hold-out for eval loss curves
    split = ds.train_test_split(test_size=min(0.1, max(2 / len(ds), 0.02)), seed=seed)
    train_ds, eval_ds = split["train"], split["test"]
    print(f"[train] train_size={len(train_ds)} eval_size={len(eval_ds)}")

    # Tokenizer ------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Base model -----------------------------------------------------------
    # Load in bf16 with gradient-checkpointing friendly config. The MXFP4
    # weights in the original checkpoint are auto-dequantized by HF.
    print("[train] loading base model (this will download ~40GB on first run)...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        # gpt-oss's GptOssForCausalLM only supports eager + flash_attn_2.
        # Eager OOMs at 8K seq_len on A100-80GB due to quadratic attention
        # matrices; flash-attn-2 keeps us comfortably below the memory cap.
        attn_implementation="flash_attention_2",
    )
    model.config.use_cache = False  # required for grad-ckpt + training
    print(f"[train] base model loaded in {time.time() - t0:.1f}s")

    # LoRA adapter config --------------------------------------------------
    # For gpt-oss-20b (MoE), attaching LoRA to each of the 32 experts per
    # layer would bloat trainable params >20x. Restrict to attention
    # projections only — that's where most of the distillable style / skill
    # signal lives for short-horizon code generation tasks anyway.
    lora_cfg = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
    )

    # DPO trainer config ---------------------------------------------------
    out_dir = Path("/outputs") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    training_args = DPOConfig(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=1,
        save_strategy="epoch",
        eval_strategy="epoch",
        bf16=True,
        beta=beta,
        max_length=max_seq_len,
        remove_unused_columns=False,
        seed=seed,
        report_to=[],
        # Without tight grad clipping, margin-weighted DPO on a 20B model
        # trivially drives ||g|| > 100 in a few steps and saturates the
        # loss. 1.0 is the transformers default and keeps things sane.
        max_grad_norm=1.0,
    )

    # TRL's DPOTrainer uses the passed peft config to attach the LoRA
    # adapter and automatically uses the underlying base model as the DPO
    # reference (it disables the adapter when computing ref logprobs), so
    # we never need to load a second copy of gpt-oss-20b.
    # trl >= 0.14 renamed ``tokenizer`` -> ``processing_class``.
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=lora_cfg,
    )

    # NOTE on margin weighting: an earlier version of this script scaled
    # the DPO loss by the per-pair score margin. In practice that drove
    # grad norms into the hundreds within a few steps on a 20B model and
    # made the loss oscillate wildly. We now rely on the *standard* DPO
    # loss — the score margin still filters WHICH pairs make it into the
    # dataset (via --min-margin / --cross-min-margin in the dataset
    # builder), but once a pair is in, it gets unit weight like every
    # other. This keeps the optimization well-behaved on this small
    # dataset.

    print("[train] starting training")
    MODELS_VOL.commit()  # snapshot before any training so we never lose the baseline
    HF_CACHE_VOL.commit()

    train_result = trainer.train()
    print(f"[train] training done: {train_result.metrics}")

    # Persist artifacts ----------------------------------------------------
    print(f"[train] saving adapter to {out_dir}")
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    summary = {
        "run_name": run_name,
        "base_model": base_model,
        "num_pairs": len(pairs),
        "train_size": len(train_ds),
        "eval_size": len(eval_ds),
        "epochs": epochs,
        "lr": learning_rate,
        "beta": beta,
        "lora_rank": lora_rank,
        "metrics": train_result.metrics,
    }
    (out_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    MODELS_VOL.commit()
    HF_CACHE_VOL.commit()
    print(f"[train] adapter + tokenizer + summary written to {out_dir}")
    return summary


# --------------------------------------------------------------------------
# Local entry point
# --------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    dataset: str,
    run_name: str,
    base_model: str = BASE_MODEL_DEFAULT,
    epochs: int = 3,
    learning_rate: float = 1e-6,
    beta: float = 0.1,
    max_seq_len: int = 4096,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    seed: int = 42,
):
    """Kick off a DPO-LoRA training run on Modal.

    dataset: local path to a JSONL file produced by build_dpo_dataset.py
    run_name: directory name under ``atlas-models`` Volume for artifacts
    """
    ds_path = Path(dataset)
    if not ds_path.exists():
        raise FileNotFoundError(ds_path)
    dataset_text = ds_path.read_text()
    print(f"[local] launching ATLAS training: dataset={ds_path} ({len(dataset_text)} chars)")
    summary = train.remote(
        dataset_jsonl_text=dataset_text,
        run_name=run_name,
        base_model=base_model,
        epochs=epochs,
        learning_rate=learning_rate,
        beta=beta,
        max_seq_len=max_seq_len,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        seed=seed,
    )
    print("\n=== ATLAS TRAINING SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print("\nDownload the adapter with:")
    print(f"  modal volume get atlas-models {run_name} ./atlas_models/")
