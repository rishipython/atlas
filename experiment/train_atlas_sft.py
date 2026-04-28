"""Train a reward-aware AlgoTune LoRA adapter on `gpt-oss-20b`."""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_MODEL_DEFAULT = "openai/gpt-oss-20b"

HF_CACHE_VOL = modal.Volume.from_name("atlas-hf-cache", create_if_missing=True)
MODELS_VOL = modal.Volume.from_name("atlas-models", create_if_missing=True)

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


@app.function(
    gpu="A100-80GB",
    timeout=2 * 3600,
    volumes={"/hf_cache": HF_CACHE_VOL, "/outputs": MODELS_VOL},
)
def train(
    dataset_jsonl_text: str,
    run_name: str,
    base_model: str,
    resume_from: str | None,
    epochs: int,
    learning_rate: float,
    max_seq_len: int,
    lora_rank: int,
    lora_alpha: int,
    seed: int,
    gradient_accumulation_steps: int,
    weight_clip: float | None,
) -> dict:
    import os

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import torch
    import torch.nn as nn
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    def parse_jsonl(text: str) -> list[dict]:
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    records = parse_jsonl(dataset_jsonl_text)
    if not records:
        raise RuntimeError("Training dataset is empty.")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    model.config.use_cache = False
    print(f"[train] base model loaded in {time.time() - t0:.1f}s")

    lora_cfg = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    if resume_from:
        resume_path = Path("/outputs") / resume_from
        if not resume_path.exists():
            raise FileNotFoundError(resume_path)
        model = PeftModel.from_pretrained(model, str(resume_path), is_trainable=True)
    else:
        model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    def render_assistant_turn(content: str, thinking: str | None) -> str:
        if thinking:
            return (
                "<|channel|>analysis<|message|>"
                + thinking
                + "<|end|><|start|>assistant<|channel|>final<|message|>"
                + content
                + "<|return|>"
            )
        return "<|channel|>final<|message|>" + content + "<|return|>"

    def encode(record: dict) -> dict:
        messages = record["messages"]
        prompt_text = tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True,
        )
        assistant = messages[-1]
        full_text = prompt_text + render_assistant_turn(
            assistant["content"],
            assistant.get("thinking"),
        )
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_seq_len,
        )["input_ids"]

        prompt_len = min(len(prompt_ids), len(full_ids))
        labels = [-100] * prompt_len + full_ids[prompt_len:]
        labels = labels[: len(full_ids)]

        weight = float(record.get("sample_weight", 1.0))
        if weight_clip is not None:
            weight = min(weight, weight_clip)

        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
            "sample_weight": weight,
        }

    dataset = Dataset.from_list([encode(record) for record in records])

    def collate(features: list[dict]) -> dict:
        max_len = max(len(feature["input_ids"]) for feature in features)

        def pad(values: list[int], fill: int) -> list[int]:
            return values + [fill] * (max_len - len(values))

        return {
            "input_ids": torch.tensor([pad(f["input_ids"], tokenizer.pad_token_id) for f in features]),
            "attention_mask": torch.tensor([pad(f["attention_mask"], 0) for f in features]),
            "labels": torch.tensor([pad(f["labels"], -100) for f in features]),
            "sample_weight": torch.tensor([float(f["sample_weight"]) for f in features], dtype=torch.float32),
        }

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            sample_weight = inputs.pop("sample_weight")
            labels = inputs["labels"]
            outputs = model(**inputs)
            logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            active = shift_labels.ne(-100)

            loss_fct = nn.CrossEntropyLoss(reduction="none")
            per_token = loss_fct(
                logits.view(-1, logits.size(-1)),
                shift_labels.view(-1),
            ).view(shift_labels.size())

            denom = active.sum(dim=1).clamp(min=1)
            per_example = (per_token * active).sum(dim=1) / denom
            loss = (per_example * sample_weight.to(per_example.device)).mean()
            return (loss, outputs) if return_outputs else loss

    out_dir = Path("/outputs") / run_name
    args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        logging_steps=1,
        save_strategy="no",
        bf16=True,
        report_to=[],
        remove_unused_columns=False,
        seed=seed,
    )

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=collate,
    )
    trainer.train()

    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    MODELS_VOL.commit()

    summary = {
        "run_name": run_name,
        "base_model": base_model,
        "resume_from": resume_from,
        "num_records": len(records),
        "epochs": epochs,
        "learning_rate": learning_rate,
        "max_seq_len": max_seq_len,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    MODELS_VOL.commit()
    return summary


@app.local_entrypoint()
def main(
    dataset: str,
    run_name: str,
    base_model: str = BASE_MODEL_DEFAULT,
    resume_from: str | None = None,
    epochs: int = 1,
    learning_rate: float = 1e-4,
    max_seq_len: int = 8192,
    lora_rank: int = 32,
    lora_alpha: int = 64,
    seed: int = 42,
    gradient_accumulation_steps: int = 4,
    weight_clip: float | None = None,
) -> None:
    dataset_text = Path(dataset).read_text()
    summary = train.remote(
        dataset_jsonl_text=dataset_text,
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
    print(json.dumps(summary, indent=2))
