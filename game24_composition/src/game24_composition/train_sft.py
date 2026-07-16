"""Supervised fine-tuning for separate A-only and B-only data."""

from __future__ import annotations

import argparse
from pathlib import Path

from .formatters import supervised_text, user_prompt
from .utils import read_jsonl, set_seed, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--fallback_model_name", default=None)
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=768)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_total_limit", type=int, default=None)
    parser.add_argument("--allow_ab", action="store_true")
    args = parser.parse_args()

    train(args)


def train(args):
    import torch
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        model_name_loaded = args.model_name
        tokenizer, model, is_peft = load_tokenizer_and_model(
            AutoTokenizer, AutoModelForCausalLM, args.model_name, torch
        )
    except Exception:
        if not args.fallback_model_name:
            raise
        model_name_loaded = args.fallback_model_name
        tokenizer, model, is_peft = load_tokenizer_and_model(
            AutoTokenizer, AutoModelForCausalLM, model_name_loaded, torch
        )

    if args.lora_r > 0 and not is_peft:
        config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, config)
        model.print_trainable_parameters()

    model.config.use_cache = False
    records = read_jsonl(args.train_file)
    if not args.allow_ab and any(row.get("task_type") == "AB_symbolic_game24" for row in records):
        raise ValueError("M_sep training data must not contain AB composition examples.")

    class SFTDataset(Dataset):
        def __init__(self, rows):
            self.items = [encode_example(tokenizer, row, args.max_seq_len) for row in rows]

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            return self.items[idx]

    def collate(batch):
        max_len = max(len(item["input_ids"]) for item in batch)
        input_ids, attention_mask, labels = [], [], []
        for item in batch:
            pad = max_len - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [tokenizer.pad_token_id] * pad)
            attention_mask.append([1] * len(item["input_ids"]) + [0] * pad)
            labels.append(item["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    train_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        save_total_limit=args.save_total_limit,
        bf16=bf16,
        fp16=torch.cuda.is_available() and not bf16,
        report_to=[],
        seed=args.seed,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=SFTDataset(records),
        data_collator=collate,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    write_json(
        output_dir / "training_config.json",
        {
            **vars(args),
            "model_name_loaded": model_name_loaded,
            "num_train_examples": len(records),
            "task_types": sorted({row.get("task_type") for row in records}),
        },
    )


def load_tokenizer_and_model(tokenizer_cls, model_cls, model_name, torch):
    tokenizer = tokenizer_cls.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        from peft import AutoPeftModelForCausalLM

        return tokenizer, load_causal_model(AutoPeftModelForCausalLM, model_name, torch, is_trainable=True), True
    except Exception:
        return tokenizer, load_causal_model(model_cls, model_name, torch), False


def load_causal_model(model_cls, model_name, torch, **extra_kwargs):
    kwargs = {"trust_remote_code": True, **extra_kwargs}
    if torch.cuda.is_available():
        kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return model_cls.from_pretrained(model_name, **kwargs)


def encode_example(tokenizer, row, max_seq_len):
    prompt = user_prompt(tokenizer, row["question"])
    full = supervised_text(tokenizer, row["question"], row["answer"])
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    full_ids = tokenizer(full, add_special_tokens=False).input_ids
    if full_ids[: len(prompt_ids)] == prompt_ids:
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    else:
        labels = full_ids.copy()

    if len(full_ids) > max_seq_len:
        full_ids = full_ids[:max_seq_len]
        labels = labels[:max_seq_len]
    return {"input_ids": full_ids, "labels": labels}


if __name__ == "__main__":
    main()
