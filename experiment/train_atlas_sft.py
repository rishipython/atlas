"""Train an ATLAS LoRA adapter on gpt-oss-20b via reward-weighted SFT.

This supersedes ``train_atlas_lora.py`` (DPO-based, which the user
vetoed after a head-to-head run showed ATLAS regressing on its own
training distribution).  The new plan trains in TWO PHASES:

* **Phase 1** — SFT on (OE-style system + OE-style user prompt) →
  full Triton program.  Teaches the model to produce good full
  programs given the dense-context OpenEvolve prompt.
* **Phase 2** — SFT on (base system + base user prompt) → same
  full Triton program, continuing from the Phase-1 LoRA.  Bridges the
  model from "good under OE context" to "good under the short
  user-facing prompt we actually use at inference/eval time."  A
  configurable fraction of Phase-1 examples is mixed back in as
  replay so we don't fully overwrite Phase-1 behavior.

Both phases use **reward-weighted NLL**: per-example loss is averaged
over the assistant's tokens, multiplied by ``sample_weight`` (the
combined_score — the user asked us to use the reward info rather than
discarding it), then averaged across the batch.  This is a RAFT-style
distillation objective: "minimize NLL of the good programs, more
heavily for the best ones."

Usage (from a local ``modal`` conda env):

    # Phase 1 (from scratch LoRA)
    modal run experiment/train_atlas_sft.py \\
        --dataset data/sft/softmax_phase1.jsonl \\
        --phase phase1 --run-name atlas_softmax_phase1_v1

    # Phase 2 (continue training, mix in replay from Phase 1)
    modal run experiment/train_atlas_sft.py \\
        --dataset data/sft/softmax_phase2.jsonl \\
        --phase phase2 --run-name atlas_softmax_phase2_v1 \\
        --resume-from atlas_softmax_phase1_v1 \\
        --replay-dataset data/sft/softmax_phase1.jsonl \\
        --replay-fraction 0.3

Each phase saves a standalone LoRA adapter under its own ``run_name``
on the ``atlas-models`` Modal Volume, so the Phase-1 adapter remains
available for inspection or alternate continuation.
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_MODEL_DEFAULT = "openai/gpt-oss-20b"

HF_CACHE_VOL = modal.Volume.from_name("atlas-hf-cache", create_if_missing=True)
MODELS_VOL = modal.Volume.from_name("atlas-models", create_if_missing=True)

# Same image recipe as ``train_atlas_lora.py``: torch 2.6 + flash-attn 2
# pre-built wheel (building from source requires CUDA_HOME at image-build
# time which debian_slim doesn't have), transformers / peft / kernels
# versions that support gpt-oss's MXFP4 load path.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl", "build-essential")
    .pip_install(
        "torch==2.6.0",
        "transformers>=4.56",
        "peft>=0.14",
        "trl>=0.14",
        "accelerate>=1.2",
        "datasets>=3.2",
        "sentencepiece",
        "safetensors",
        "pyyaml>=6",
        "kernels>=0.7",
        "ninja",
        "packaging",
    )
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

app = modal.App("atlas-train-sft", image=image)


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------
@app.function(
    gpu="A100-80GB",
    timeout=2 * 3600,
    volumes={"/hf_cache": HF_CACHE_VOL, "/outputs": MODELS_VOL},
)
def train(
    dataset_jsonl_text: str,
    replay_jsonl_text: str,
    replay_fraction: float,
    phase: str,
    run_name: str,
    base_model: str,
    resume_from: str | None,
    epochs: int,
    learning_rate: float,
    max_seq_len: int,
    lora_rank: int,
    lora_alpha: int,
    seed: int,
    gradient_accumulation_steps: int = 4,
    weight_clip: float | None = None,
) -> dict:
    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch
    import torch.nn as nn
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    print(f"[train] phase={phase} run_name={run_name} base_model={base_model}")
    print(f"[train] dataset_chars={len(dataset_jsonl_text)} replay_chars={len(replay_jsonl_text)}")
    print(f"[train] resume_from={resume_from} epochs={epochs} lr={learning_rate} seq={max_seq_len}")

    # --- Parse dataset(s) -------------------------------------------------
    def _parse_jsonl(text: str) -> list[dict]:
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    records = _parse_jsonl(dataset_jsonl_text)
    assert records, "Primary dataset is empty — aborting."
    print(f"[train] parsed {len(records)} primary records")

    # Mix replay (Phase-1 examples) into Phase-2 training so the bridge
    # step doesn't wipe the "OE-context behavior" we just learned.
    if replay_jsonl_text and replay_fraction > 0:
        replay = _parse_jsonl(replay_jsonl_text)
        rng = random.Random(seed)
        # Sample `replay_fraction * len(primary)` replay records (with
        # replacement if we want more than exist).
        n_replay = max(1, int(round(replay_fraction * len(records))))
        if n_replay >= len(replay):
            replay_sampled = list(replay)
        else:
            replay_sampled = rng.sample(replay, n_replay)
        # Downweight replay a bit so primary dataset dominates.  The
        # stored sample_weight is already the combined_score, so we
        # multiply replay weights by 0.5 to keep them in the loss but
        # let the primary distribution dominate.
        for r in replay_sampled:
            r["sample_weight"] = float(r.get("sample_weight", 1.0)) * 0.5
            r["_is_replay"] = True
        records.extend(replay_sampled)
        print(f"[train] added {len(replay_sampled)} replay records (fraction={replay_fraction})")
    print(f"[train] total training records: {len(records)}")

    # --- Tokenizer + base model ------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[train] loading base model (bf16, flash-attn-2)...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        # gpt-oss's GptOssForCausalLM only supports eager + flash_attn_2;
        # eager OOMs at 8K seq_len on A100-80GB, flash-attn-2 fits cleanly.
        attn_implementation="flash_attention_2",
    )
    model.config.use_cache = False  # required for grad-ckpt + training
    print(f"[train] base loaded in {time.time() - t0:.1f}s")

    # --- LoRA attach / resume --------------------------------------------
    lora_cfg = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        # Attention-proj-only: MoE experts are too numerous to LoRA-fy
        # cheaply on gpt-oss-20b and most of the distillable skill signal
        # lives on the attention path anyway (same choice as the DPO run).
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    if resume_from:
        resume_path = Path("/outputs") / resume_from
        assert resume_path.exists(), f"resume_from adapter {resume_path} not found on atlas-models volume"
        print(f"[train] resuming LoRA from {resume_path}")
        # is_trainable=True keeps LoRA params as requires_grad for further
        # optimization. The base stays frozen regardless.
        model = PeftModel.from_pretrained(model, str(resume_path), is_trainable=True)
    else:
        model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Enable gradient checkpointing for the LoRA-wrapped model (saves a
    # ton of activation memory — essential at seq_len=8k+ on 80GB).
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    # PEFT requires input-gradients enabled for ckpt to pass gradients
    # through the frozen base into the LoRA params.
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    # --- Tokenization: text-level prompt+assistant concat --------------
    # Using ``apply_chat_template(messages, tokenize=True)`` with the
    # full dialog (assistant included) is unreliable for gpt-oss's
    # Harmony template: the channel markers ``<|channel|>final<|message|>``
    # may or may not be emitted depending on whether the message has a
    # ``thinking`` field, and the tokens don't line up cleanly with the
    # ``add_generation_prompt=True`` rendering.  Instead we render the
    # prompt as TEXT (with gen-prompt), manually construct the Harmony
    # assistant turn (analysis channel if thinking present, final
    # channel for content, terminated by ``<|return|>``), then tokenize.
    # The first ``len(prompt_ids)`` tokens of ``full_ids`` are a prefix
    # of ``prompt_ids`` (up to at most 1-2 BPE drift tokens, which we
    # tolerate below), so the label mask is essentially exact.
    def _render_assistant_turn(content: str, thinking: str | None) -> str:
        # gpt-oss Harmony format.  When gen_prompt=True the prompt text
        # already ends with ``<|start|>assistant``; we append the
        # per-channel portions from there.
        if thinking:
            # analysis channel → end-of-analysis → new assistant start
            # for the final channel → content → return.
            return (
                "<|channel|>analysis<|message|>"
                + thinking
                + "<|end|><|start|>assistant<|channel|>final<|message|>"
                + content
                + "<|return|>"
            )
        return "<|channel|>final<|message|>" + content + "<|return|>"

    _encode_diag_printed = {"done": False}

    def _encode(rec: dict) -> dict:
        messages = rec["messages"]
        assert messages[-1]["role"] == "assistant", "expected assistant turn last"
        prompt_text = tokenizer.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True
        )
        assistant_content = messages[-1]["content"]
        assistant_thinking = messages[-1].get("thinking")
        assistant_turn = _render_assistant_turn(assistant_content, assistant_thinking)
        full_text = prompt_text + assistant_turn
        if not _encode_diag_printed["done"]:
            # One-time diagnostic: show the last 250 chars of the prompt
            # (so we can confirm it ends where we expect) and the first
            # 250 chars of the manually-rendered assistant turn (so we
            # can confirm the channel markers look right).
            print("[train] prompt_text tail (250 chars):", repr(prompt_text[-250:]))
            print("[train] assistant_turn head (250 chars):", repr(assistant_turn[:250]))
            _encode_diag_printed["done"] = True

        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

        # Sanity: prompt_ids must be a prefix of full_ids.  A one-token
        # drift is possible in pathological BPE cases; we tolerate up to
        # 2 tokens of mismatch by trimming prompt_ids accordingly (labels
        # for those tokens will still be masked since the assistant's
        # meaningful content starts well after).
        common = 0
        for pa, pb in zip(prompt_ids, full_ids):
            if pa != pb:
                break
            common += 1
        effective_prompt_len = common

        # Left-truncate to fit max_seq_len, keeping the assistant tail.
        if len(full_ids) > max_seq_len:
            overflow = len(full_ids) - max_seq_len
            full_ids = full_ids[overflow:]
            effective_prompt_len = max(0, effective_prompt_len - overflow)

        labels = list(full_ids)
        for i in range(min(effective_prompt_len, len(labels))):
            labels[i] = -100
        return {
            "input_ids": full_ids,
            "labels": labels,
            "sample_weight": float(rec.get("sample_weight", 1.0)),
            "debug_prompt_len": effective_prompt_len,
            "debug_full_len": len(full_ids),
        }

    print("[train] tokenizing dataset...")
    ds = Dataset.from_list(records)
    tok_ds = ds.map(_encode, remove_columns=ds.column_names)
    # Diagnostics: first 3 records' lengths.
    for i in range(min(3, len(tok_ds))):
        r = tok_ds[i]
        n_assist = sum(1 for lbl in r["labels"] if lbl != -100)
        print(
            f"[train] rec {i}: full_len={r['debug_full_len']} "
            f"prompt_len={r['debug_prompt_len']} assistant_tokens={n_assist}"
        )

    def _has_assistant_tokens(r):
        return any(lbl != -100 for lbl in r["labels"])
    before = len(tok_ds)
    tok_ds = tok_ds.filter(_has_assistant_tokens)
    print(f"[train] tokenized: {len(tok_ds)} / {before} records have non-empty assistant span")
    assert len(tok_ds) > 0, (
        "All records had empty assistant spans after tokenization — "
        "check tokenizer chat template behavior for gpt-oss Harmony."
    )

    # Hold out a tiny eval slice for loss curves (same split each run),
    # but only if we have enough records to spare two for eval.
    if len(tok_ds) >= 5:
        split = tok_ds.train_test_split(test_size=0.1, seed=seed)
        train_ds, eval_ds = split["train"], split["test"]
    else:
        train_ds, eval_ds = tok_ds, tok_ds.select([])
    print(f"[train] train_size={len(train_ds)} eval_size={len(eval_ds)}")
    # Log input-length histogram so we can see if we're close to the cap.
    lens = sorted(len(r["input_ids"]) for r in train_ds)
    print(f"[train] input lengths: min={lens[0]} p50={lens[len(lens)//2]} max={lens[-1]} cap={max_seq_len}")

    # --- Data collator ---------------------------------------------------
    pad_id = tokenizer.pad_token_id

    def _collate(batch: list[dict]) -> dict:
        max_len = max(len(b["input_ids"]) for b in batch)
        input_ids = []
        attention_mask = []
        labels_ = []
        weights = []
        for b in batch:
            ids = b["input_ids"]
            lbl = b["labels"]
            pad_n = max_len - len(ids)
            input_ids.append(ids + [pad_id] * pad_n)
            attention_mask.append([1] * len(ids) + [0] * pad_n)
            labels_.append(lbl + [-100] * pad_n)
            weights.append(b["sample_weight"])
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels_, dtype=torch.long),
            "sample_weight": torch.tensor(weights, dtype=torch.float32),
        }

    # --- Reward-weighted loss -------------------------------------------
    class RewardWeightedTrainer(Trainer):
        """HF Trainer with per-example NLL × sample_weight.

        Standard causal LM loss averages CE across all non-masked label
        tokens of the batch.  Here we first compute per-example mean NLL
        (averaging over that example's assistant tokens only), multiply
        by the example's ``sample_weight``, then average across the
        batch.  This matches a reward-weighted imitation objective
        (RAFT-flavored) where better programs contribute proportionally
        more gradient.
        """

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            weights = inputs.pop("sample_weight")  # (B,)
            labels = inputs["labels"]
            outputs = model(**inputs)
            logits = outputs.logits  # (B, T, V)
            B, T, V = logits.shape
            shift_labels = labels[:, 1:]  # (B, T-1), view into labels
            # Chunked CE along seq dim — avoids a second full-logits
            # allocation plus a full-size softmax scratch buffer.  With
            # V ≈ 2e5 and bf16, each 2k-token chunk is ~0.8 GB of
            # intermediate, vs. 5+ GB for the whole sequence.
            ce = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
            CE_CHUNK = 512
            per_tok_parts = []
            for start in range(0, T - 1, CE_CHUNK):
                end = min(start + CE_CHUNK, T - 1)
                sl = logits[:, start:end, :]  # slice view
                tgt = shift_labels[:, start:end]
                pt = ce(sl.reshape(-1, V), tgt.reshape(-1)).view(B, end - start)
                per_tok_parts.append(pt)
            per_tok = torch.cat(per_tok_parts, dim=1)
            mask = (shift_labels != -100).to(per_tok.dtype)
            tokens_per_ex = mask.sum(dim=-1).clamp(min=1.0)
            per_ex_nll = (per_tok * mask).sum(dim=-1) / tokens_per_ex  # (B,)
            w = weights.to(per_ex_nll.device).to(per_ex_nll.dtype)
            # Clip per-example weights to the magnitude cap set on the
            # trainer.  This bounds the gradient contribution of any
            # single example, which is essential when weights can be
            # negative (advantage-weighted mode): otherwise a single
            # very bad trajectory with a large negative advantage can
            # blow up the loss by pushing NLL → ∞ during optimization.
            clip = getattr(self, "_atlas_weight_clip", None)
            if clip is not None:
                w = torch.clamp(w, min=-clip, max=clip)
            # Normalize by mean |w| so the loss scale stays comparable
            # to unweighted NLL regardless of how positive- or
            # negative-heavy the batch is.  Using mean(w) (as in the
            # original reward-weighted variant) is unsafe here because
            # signed advantages can produce mean(w) ≈ 0, which would
            # blow up the divisor.
            abs_w_mean = w.abs().mean().clamp(min=1e-6)
            loss = (per_ex_nll * w / abs_w_mean).mean()
            return (loss, outputs) if return_outputs else loss

    # --- TrainingArguments ----------------------------------------------
    out_dir = Path("/outputs") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        # Slightly higher LR than DPO's 1e-6: SFT is a less slippery
        # objective than pairwise DPO so a standard LoRA LR (~1e-4) is
        # fine, but we keep it conservative at 5e-5 given the tiny
        # dataset.
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=1,
        eval_strategy="epoch" if len(eval_ds) >= 1 else "no",
        bf16=True,
        seed=seed,
        report_to=[],
        # Keep grad norms sane.  Small dataset + bf16 + flash-attn gets
        # occasional spikes; 1.0 is the standard safety valve.
        max_grad_norm=1.0,
        remove_unused_columns=False,
    )

    trainer = RewardWeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds if len(eval_ds) >= 1 else None,
        data_collator=_collate,
    )
    # Attach the weight-clip scalar after construction so the override
    # in ``compute_loss`` can find it.  ``None`` means "no clip"
    # (preserves previous behavior for reward-weighted runs).
    trainer._atlas_weight_clip = weight_clip

    print("[train] starting training")
    MODELS_VOL.commit()
    HF_CACHE_VOL.commit()

    train_result = trainer.train()
    print(f"[train] done: {train_result.metrics}")

    print(f"[train] saving adapter to {out_dir}")
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    summary = {
        "phase": phase,
        "run_name": run_name,
        "resume_from": resume_from,
        "base_model": base_model,
        "num_records_primary": len(records) - (len(records) - before if False else 0),
        "num_records_total_after_tok_filter": len(tok_ds),
        "train_size": len(train_ds),
        "eval_size": len(eval_ds),
        "epochs": epochs,
        "lr": learning_rate,
        "max_seq_len": max_seq_len,
        "lora_rank": lora_rank,
        "metrics": train_result.metrics,
    }
    (out_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    MODELS_VOL.commit()
    HF_CACHE_VOL.commit()
    print(f"[train] saved adapter + tokenizer + summary → {out_dir}")
    return summary


# ---------------------------------------------------------------------------
# Local entry point
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    dataset: str,
    phase: str,
    run_name: str,
    base_model: str = BASE_MODEL_DEFAULT,
    resume_from: str | None = None,
    replay_dataset: str | None = None,
    replay_fraction: float = 0.3,
    epochs: int = 3,
    learning_rate: float = 5e-5,
    max_seq_len: int = 16384,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    seed: int = 42,
    gradient_accumulation_steps: int = 4,
    weight_clip: float | None = None,
):
    """Kick off reward-weighted SFT on Modal.

    Parameters
    ----------
    dataset         : local JSONL path from ``build_sft_dataset.py``
                      (either the Phase-1 or Phase-2 output).
    phase           : ``"phase1"`` or ``"phase2"`` — bookkeeping label.
    run_name        : unique subdir under the ``atlas-models`` Volume
                      for the resulting LoRA adapter.
    resume_from     : (Phase 2 only) adapter dir-name to initialize the
                      LoRA weights from.  None = fresh LoRA.
    replay_dataset  : (Phase 2 only) a second JSONL (typically the
                      Phase-1 dataset) to mix in as replay.
    replay_fraction : Phase-2 replay sample count = this × primary size.
    max_seq_len     : 12k by default — fits even the longest OE prompt
                      (~11k tokens) + the full assistant target.  Phase 2
                      prompts are much shorter (~2.5k), so this is only
                      heavy on Phase 1.  If OOM, drop to 8192 for Phase 1.
    """
    assert phase in ("phase1", "phase2"), f"phase must be 'phase1' or 'phase2', got {phase!r}"
    ds_path = Path(dataset)
    if not ds_path.exists():
        raise FileNotFoundError(ds_path)
    dataset_text = ds_path.read_text()

    replay_text = ""
    if replay_dataset:
        rp = Path(replay_dataset)
        if not rp.exists():
            raise FileNotFoundError(rp)
        replay_text = rp.read_text()

    print(
        f"[local] launching SFT: phase={phase} dataset={ds_path} "
        f"({len(dataset_text)} chars) replay={replay_dataset} resume_from={resume_from}"
    )
    summary = train.remote(
        dataset_jsonl_text=dataset_text,
        replay_jsonl_text=replay_text,
        replay_fraction=replay_fraction,
        phase=phase,
        run_name=run_name,
        base_model=base_model,
        resume_from=resume_from,
        epochs=epochs,
        learning_rate=learning_rate,
        max_seq_len=max_seq_len,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        seed=seed,
        gradient_accumulation_steps=gradient_accumulation_steps,
        weight_clip=weight_clip,
    )
    print("\n=== TRAINING SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nAdapter saved to atlas-models Volume under: {run_name}")
    print(f"Download with: modal volume get atlas-models {run_name} ./atlas_models/")
