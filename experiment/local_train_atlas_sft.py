"""Local one-GPU reward-weighted SFT training for Colab / workstations."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

BASE_MODEL_DEFAULT = "openai/gpt-oss-20b"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--phase", required=True, choices=["phase1", "phase2"])
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--base-model", default=BASE_MODEL_DEFAULT)
    parser.add_argument("--resume-from", default=None, help="Local adapter directory.")
    parser.add_argument("--replay-dataset", default=None)
    parser.add_argument("--replay-fraction", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-seq-len", type=int, default=16384)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--weight-clip", type=float, default=None)
    parser.add_argument("--output-root", default="atlas_models")
    args = parser.parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import torch
    import torch.nn as nn
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    def _parse_jsonl(path: str) -> list[dict]:
        return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]

    records = _parse_jsonl(args.dataset)
    assert records, "Primary dataset is empty — aborting."
    primary_count = len(records)

    if args.replay_dataset and args.replay_fraction > 0:
        replay = _parse_jsonl(args.replay_dataset)
        rng = random.Random(args.seed)
        n_replay = max(1, int(round(args.replay_fraction * len(records))))
        replay_sampled = list(replay) if n_replay >= len(replay) else rng.sample(replay, n_replay)
        for r in replay_sampled:
            r["sample_weight"] = float(r.get("sample_weight", 1.0)) * 0.5
            r["_is_replay"] = True
        records.extend(replay_sampled)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[train] loading base model {args.base_model} ...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    model.config.use_cache = False
    print(f"[train] base loaded in {time.time() - t0:.1f}s")

    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    if args.resume_from:
        resume_path = Path(args.resume_from)
        if not resume_path.exists():
            resume_path = Path(args.output_root) / args.resume_from
        assert resume_path.exists(), f"resume_from adapter {resume_path} not found"
        model = PeftModel.from_pretrained(model, str(resume_path), is_trainable=True)
    else:
        model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    def _render_assistant_turn(content: str, thinking: str | None) -> str:
        if thinking:
            return (
                "<|channel|>analysis<|message|>"
                + thinking
                + "<|end|><|start|>assistant<|channel|>final<|message|>"
                + content
                + "<|return|>"
            )
        return "<|channel|>final<|message|>" + content + "<|return|>"

    def _encode(rec: dict) -> dict:
        messages = rec["messages"]
        prompt_text = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
        assistant_turn = _render_assistant_turn(messages[-1]["content"], messages[-1].get("thinking"))
        full_text = prompt_text + assistant_turn
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        common = 0
        for pa, pb in zip(prompt_ids, full_ids):
            if pa != pb:
                break
            common += 1
        effective_prompt_len = common
        if len(full_ids) > args.max_seq_len:
            overflow = len(full_ids) - args.max_seq_len
            full_ids = full_ids[overflow:]
            effective_prompt_len = max(0, effective_prompt_len - overflow)
        labels = list(full_ids)
        for i in range(min(effective_prompt_len, len(labels))):
            labels[i] = -100
        return {
            "input_ids": full_ids,
            "labels": labels,
            "sample_weight": float(rec.get("sample_weight", 1.0)),
        }

    ds = Dataset.from_list(records)
    tok_ds = ds.map(_encode, remove_columns=ds.column_names)
    tok_ds = tok_ds.filter(lambda r: any(lbl != -100 for lbl in r["labels"]))
    assert len(tok_ds) > 0, "All records had empty assistant spans after tokenization."
    if len(tok_ds) >= 5:
        split = tok_ds.train_test_split(test_size=0.1, seed=args.seed)
        train_ds, eval_ds = split["train"], split["test"]
    else:
        train_ds, eval_ds = tok_ds, tok_ds.select([])

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

    class RewardWeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            weights = inputs.pop("sample_weight")
            labels = inputs["labels"]
            outputs = model(**inputs)
            logits = outputs.logits
            B, T, V = logits.shape
            shift_labels = labels[:, 1:]
            ce = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
            parts = []
            for start in range(0, T - 1, 512):
                end = min(start + 512, T - 1)
                sl = logits[:, start:end, :]
                tgt = shift_labels[:, start:end]
                pt = ce(sl.reshape(-1, V), tgt.reshape(-1)).view(B, end - start)
                parts.append(pt)
            per_tok = torch.cat(parts, dim=1)
            mask = (shift_labels != -100).to(per_tok.dtype)
            per_ex_nll = (per_tok * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1.0)
            w = weights.to(per_ex_nll.device).to(per_ex_nll.dtype)
            if args.weight_clip is not None:
                w = torch.clamp(w, min=-args.weight_clip, max=args.weight_clip)
            loss = (per_ex_nll * w / w.abs().mean().clamp(min=1e-6)).mean()
            return (loss, outputs) if return_outputs else loss

    out_dir = Path(args.output_root) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=1,
        eval_strategy="epoch" if len(eval_ds) >= 1 else "no",
        bf16=True,
        seed=args.seed,
        report_to=[],
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
    result = trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    summary = {
        "phase": args.phase,
        "run_name": args.run_name,
        "resume_from": args.resume_from,
        "base_model": args.base_model,
        "num_records_primary": primary_count,
        "num_records_total": len(records),
        "num_records_total_after_tok_filter": len(tok_ds),
        "train_size": len(train_ds),
        "eval_size": len(eval_ds),
        "epochs": args.epochs,
        "lr": args.learning_rate,
        "max_seq_len": args.max_seq_len,
        "lora_rank": args.lora_rank,
        "metrics": result.metrics,
    }
    (out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
