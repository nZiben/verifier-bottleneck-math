"""Evaluate decode, numeric Game24, symbolic Game24, and AB pass@k."""

from __future__ import annotations

import argparse
from pathlib import Path

from .checker import check_game24
from .formatters import user_prompt
from .utils import (
    extract_ints,
    generation_path_for,
    parse_pass_k,
    read_jsonl,
    set_seed,
    write_json,
    write_jsonl,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--splits", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--pass_k", default=None)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run(args)


def run(args):
    set_seed(args.seed)
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    pass_ks = parse_pass_k(args.pass_k) if args.pass_k else None
    all_generation_rows = []

    if pass_ks:
        if len(args.splits) != 1:
            raise ValueError("pass@k evaluation expects exactly one split.")
        split_path = args.splits[0]
        examples = read_jsonl(split_path)
        summary, rows = evaluate_passk(
            model,
            tokenizer,
            examples,
            Path(split_path).stem,
            pass_ks,
            args.num_samples,
            args.temperature,
            args.top_p,
            args.max_new_tokens,
        )
        summary.update(
            {
                "model_path": args.model_path,
                "split_file": split_path,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "num_samples": args.num_samples,
                "max_new_tokens": args.max_new_tokens,
            }
        )
        all_generation_rows.extend(rows)
    else:
        summaries = {}
        for split_path in args.splits:
            examples = read_jsonl(split_path)
            split_name = Path(split_path).stem
            summary, rows = evaluate_split_once(
                model,
                tokenizer,
                examples,
                split_name,
                args.temperature,
                args.top_p,
                args.max_new_tokens,
            )
            summaries[split_name] = summary
            all_generation_rows.extend(rows)
        summary = {
            "model_path": args.model_path,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            "splits": summaries,
        }

    write_json(args.out, summary)
    write_jsonl(generation_path_for(args.out), all_generation_rows)
    print(f"Wrote {args.out}")
    print(f"Wrote {generation_path_for(args.out)}")


def load_model_and_tokenizer(model_path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {"trust_remote_code": True}
    if torch.cuda.is_available():
        kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        kwargs["device_map"] = "auto"

    try:
        from peft import AutoPeftModelForCausalLM

        model = AutoPeftModelForCausalLM.from_pretrained(model_path, **kwargs)
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    if not torch.cuda.is_available():
        model.to("cpu")
    model.eval()
    return model, tokenizer


def evaluate_split_once(model, tokenizer, examples, split_name, temperature, top_p, max_new_tokens):
    rows = []
    correct = 0
    for idx, example in enumerate(examples):
        generation = generate_completions(
            model, tokenizer, example["question"], 1, temperature, top_p, max_new_tokens
        )[0]
        score = score_generation(example, generation)
        correct += int(score["is_correct"])
        rows.append(
            {
                "split": split_name,
                "id": example["id"],
                "task_type": example["task_type"],
                "sample_index": 0,
                "generation": generation,
                **score,
            }
        )
        if (idx + 1) % 25 == 0:
            print(f"{split_name}: evaluated {idx + 1}/{len(examples)}")

    metric_name = metric_for_task(examples[0]["task_type"] if examples else split_name)
    return {
        "split": split_name,
        "task_type": examples[0]["task_type"] if examples else None,
        "num_examples": len(examples),
        metric_name: correct / len(examples) if examples else 0.0,
        "num_correct": correct,
    }, rows


def evaluate_passk(model, tokenizer, examples, split_name, pass_ks, num_samples, temperature, top_p, max_new_tokens):
    rows = []
    solved_counts = {k: 0 for k in pass_ks}
    solved_at_max = 0
    for idx, example in enumerate(examples):
        generations = generate_completions(
            model, tokenizer, example["question"], num_samples, temperature, top_p, max_new_tokens
        )
        correctness = []
        for sample_index, generation in enumerate(generations):
            score = score_generation(example, generation)
            correctness.append(bool(score["is_correct"]))
            rows.append(
                {
                    "split": split_name,
                    "id": example["id"],
                    "task_type": example["task_type"],
                    "sample_index": sample_index,
                    "generation": generation,
                    **score,
                }
            )
        for k in pass_ks:
            if any(correctness[: min(k, len(correctness))]):
                solved_counts[k] += 1
        if any(correctness):
            solved_at_max += 1
        if (idx + 1) % 10 == 0:
            print(f"{split_name}: sampled {idx + 1}/{len(examples)}")

    return {
        "split": split_name,
        "num_examples": len(examples),
        "pass_at_k": {str(k): solved_counts[k] / len(examples) if examples else 0.0 for k in pass_ks},
        "num_never_solved": len(examples) - solved_at_max,
        "num_solved_at_256": solved_at_max if num_samples >= 256 else None,
    }, rows


def score_generation(example, generation):
    metadata = example.get("metadata", {})
    task_type = example.get("task_type")
    if task_type == "A_decode":
        predicted = extract_ints(generation)
        gold = [int(number) for number in metadata["numbers"]]
        return {
            "is_correct": predicted == gold,
            "reason": "correct" if predicted == gold else "decoded sequence mismatch",
            "predicted_numbers": predicted,
            "gold_numbers": gold,
            "checker": None,
        }

    if task_type == "B_game24_numeric":
        checker = check_game24(generation, numbers=metadata["numbers"])
    elif task_type == "AB_symbolic_game24":
        checker = check_game24(generation, numbers=metadata["numbers"], symbols=metadata["symbols"])
    else:
        raise ValueError(f"Unknown task_type: {task_type}")

    return {
        "is_correct": checker["is_correct"],
        "reason": checker["reason"],
        "checker": checker,
    }


def metric_for_task(task_type):
    if task_type == "A_decode":
        return "decode_accuracy"
    if task_type == "B_game24_numeric":
        return "numeric_game24_pass@1"
    if task_type == "AB_symbolic_game24":
        return "symbolic_game24_pass@1"
    return "accuracy"


def generate_completions(model, tokenizer, question, num_samples, temperature, top_p, max_new_tokens):
    import torch

    prompt = user_prompt(tokenizer, question)
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    do_sample = temperature > 0
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "num_return_sequences": num_samples if do_sample else 1,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    with torch.inference_mode():
        output = model.generate(**inputs, **gen_kwargs)

    input_len = inputs["input_ids"].shape[1]
    completions = [
        tokenizer.decode(sequence[input_len:], skip_special_tokens=True).strip() for sequence in output
    ]
    if not do_sample and num_samples > 1:
        completions = completions * num_samples
    return completions


if __name__ == "__main__":
    main()
