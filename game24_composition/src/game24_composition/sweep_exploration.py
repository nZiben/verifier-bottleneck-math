"""Exploration sweep on AB tasks with the perfect checker."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .evaluate import generate_completions, load_model_and_tokenizer, score_generation
from .plots import plot_passk_curve
from .symbols import SYMBOL_TO_NUMBER
from .utils import count_jsonl, ensure_dir, read_json, read_jsonl, set_seed, write_json, write_jsonl

SETTINGS = [
    {"temperature": 0.2, "top_p": 0.9, "num_samples": 16},
    {"temperature": 0.7, "top_p": 0.95, "num_samples": 64},
    {"temperature": 1.0, "top_p": 0.95, "num_samples": 128},
    {"temperature": 1.2, "top_p": 0.98, "num_samples": 256},
]
PASS_KS = [1, 4, 16, 64, 128, 256]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--test_file", default="data/synthetic/test_AB.jsonl")
    parser.add_argument("--out_dir", default="outputs/exploration_sweep")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run(args)


def run(args):
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    examples = read_jsonl(args.test_file)
    model, tokenizer = load_model_and_tokenizer(args.model_path)

    results = []
    generation_rows = []
    for setting_index, setting in enumerate(SETTINGS, start=1):
        print(f"Setting {setting_index}: {setting}")
        result, rows = run_setting(
            model,
            tokenizer,
            examples,
            setting_index,
            setting,
            args.max_new_tokens,
        )
        results.append(result)
        generation_rows.extend(rows)

    write_results_csv(out_dir / "results.csv", results)
    write_json(out_dir / "results.json", {"settings": results})
    write_jsonl(out_dir / "generations.jsonl", generation_rows)
    plot_passk_curve(results, out_dir / "passk_curve.png")
    write_first5_summary(Path(args.model_path), Path(args.test_file), out_dir.parent, results)

    print(f"Wrote exploration sweep to {out_dir}")
    print(f"Wrote {out_dir.parent / 'first5_summary.md'}")


def run_setting(model, tokenizer, examples, setting_index, setting, max_new_tokens):
    pass_ks = [k for k in PASS_KS if k <= setting["num_samples"]]
    solved_counts = {k: 0 for k in pass_ks}
    accepted_correct_samples = 0
    unique_correct = set()
    tasks_solved = 0
    rows = []

    for idx, example in enumerate(examples):
        generations = generate_completions(
            model,
            tokenizer,
            example["question"],
            setting["num_samples"],
            setting["temperature"],
            setting["top_p"],
            max_new_tokens,
        )
        correctness = []
        for sample_index, generation in enumerate(generations):
            score = score_generation(example, generation)
            checker = score.get("checker")
            is_correct = bool(score["is_correct"])
            correctness.append(is_correct)
            if is_correct:
                accepted_correct_samples += 1
                if checker and checker.get("extracted_expression"):
                    unique_correct.add(checker["extracted_expression"])
            rows.append(
                {
                    "setting_id": setting_index,
                    "temperature": setting["temperature"],
                    "top_p": setting["top_p"],
                    "num_samples": setting["num_samples"],
                    "id": example["id"],
                    "sample_index": sample_index,
                    "generation": generation,
                    **score,
                }
            )

        for k in pass_ks:
            if any(correctness[:k]):
                solved_counts[k] += 1
        if any(correctness):
            tasks_solved += 1
        if (idx + 1) % 10 == 0:
            print(f"  sampled {idx + 1}/{len(examples)}")

    result = {
        "setting_id": setting_index,
        **setting,
        "pass_at_k": {str(k): solved_counts[k] / len(examples) if examples else 0.0 for k in pass_ks},
        "accepted_correct_samples": accepted_correct_samples,
        "unique_correct_expressions": len(unique_correct),
        "tasks_solved_at_least_once": tasks_solved,
        "tasks_never_solved": len(examples) - tasks_solved,
    }
    return result, rows


def write_results_csv(path, results):
    fieldnames = [
        "setting_id",
        "temperature",
        "top_p",
        "num_samples",
        "pass@1",
        "pass@4",
        "pass@16",
        "pass@64",
        "pass@128",
        "pass@256",
        "accepted_correct_samples",
        "unique_correct_expressions",
        "tasks_solved_at_least_once",
        "tasks_never_solved",
    ]
    ensure_dir(Path(path).parent)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = {
                "setting_id": result["setting_id"],
                "temperature": result["temperature"],
                "top_p": result["top_p"],
                "num_samples": result["num_samples"],
                "accepted_correct_samples": result["accepted_correct_samples"],
                "unique_correct_expressions": result["unique_correct_expressions"],
                "tasks_solved_at_least_once": result["tasks_solved_at_least_once"],
                "tasks_never_solved": result["tasks_never_solved"],
            }
            for k in PASS_KS:
                row[f"pass@{k}"] = result["pass_at_k"].get(str(k), "")
            writer.writerow(row)


def write_first5_summary(model_path, test_file, outputs_dir, sweep_results):
    data_dir = test_file.parent
    composition = read_json(outputs_dir / "composition_gap_results.json", default={})
    baseline = read_json(outputs_dir / "baseline_passk.json", default={})
    training = read_json(model_path / "training_config.json", default={})
    manifest = read_json(data_dir / "manifest.json", default={})

    dataset_sizes = manifest.get("dataset_sizes") or {
        name: count_if_exists(data_dir / f"{name}.jsonl")
        for name in ["train_A", "eval_A", "train_B", "eval_B", "test_AB", "train_sep"]
    }
    split_metrics = composition.get("splits", {})
    eval_a = split_metrics.get("eval_A", {}).get("decode_accuracy")
    eval_b = split_metrics.get("eval_B", {}).get("numeric_game24_pass@1")
    ab = split_metrics.get("test_AB", {}).get("symbolic_game24_pass@1")

    lines = [
        "# First 5 Experiment Summary",
        "",
        "## Dataset Sizes",
        *[f"- {name}: {size}" for name, size in dataset_sizes.items()],
        "",
        "## Symbol Mapping",
        ", ".join(f"{symbol}={number}" for symbol, number in SYMBOL_TO_NUMBER.items()),
        "",
        "## Model Used",
        str(training.get("model_name_loaded") or training.get("model_name") or model_path),
        "",
        "## Training Setup",
        (
            f"epochs={training.get('epochs')}, lr={training.get('lr')}, "
            f"batch_size={training.get('batch_size')}, grad_accum={training.get('grad_accum')}, "
            f"max_seq_len={training.get('max_seq_len')}, lora_r={training.get('lora_r')}, "
            f"lora_alpha={training.get('lora_alpha')}"
        ),
        "",
        "## Greedy Evaluation",
        f"- A-only decode accuracy: {fmt(eval_a)}",
        f"- B-only numeric Game24 pass@1: {fmt(eval_b)}",
        f"- AB symbolic Game24 pass@1: {fmt(ab)}",
        "",
        "## AB Baseline pass@k",
        fmt_passk(baseline.get("pass_at_k", {})),
        "",
        "## Exploration Sweep",
        sweep_table(sweep_results),
        "",
        "## Composition Gap",
        composition_gap_sentence(eval_a, eval_b, ab),
        "",
        "## Exploration Effect",
        exploration_sentence(sweep_results),
        "",
        "## Explicit Exclusions",
        "No noisy checker, checker noise sweep, self-training, fine-tuning on accepted AB samples, RL, GSM8K, MATH, tree search, majority vote checker, or step-by-step verifier has been implemented.",
        "",
    ]
    (outputs_dir / "first5_summary.md").write_text("\n".join(lines), encoding="utf-8")


def count_if_exists(path):
    return count_jsonl(path) if path.exists() else "not found"


def fmt(value):
    return "not run" if value is None else f"{value:.4f}"


def fmt_passk(pass_at_k):
    if not pass_at_k:
        return "not run"
    return "\n".join(f"- pass@{k}: {value:.4f}" for k, value in pass_at_k.items())


def sweep_table(results):
    rows = ["| setting | temp | top_p | n | pass@1 | pass@16 | pass@64 | pass@256 | solved | correct samples |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for result in results:
        passk = result["pass_at_k"]
        rows.append(
            "| {setting_id} | {temperature} | {top_p} | {num_samples} | {p1} | {p16} | {p64} | {p256} | {solved} | {correct} |".format(
                setting_id=result["setting_id"],
                temperature=result["temperature"],
                top_p=result["top_p"],
                num_samples=result["num_samples"],
                p1=fmt(passk.get("1")),
                p16=fmt(passk.get("16")),
                p64=fmt(passk.get("64")),
                p256=fmt(passk.get("256")),
                solved=result["tasks_solved_at_least_once"],
                correct=result["accepted_correct_samples"],
            )
        )
    return "\n".join(rows)


def composition_gap_sentence(eval_a, eval_b, ab):
    if eval_a is None or eval_b is None or ab is None:
        return "Composition gap status: not fully measured yet."
    if ab < eval_a and ab < eval_b:
        return "Composition gap observed: AB pass@1 is lower than both A-only and B-only metrics."
    return "Composition gap not clearly observed from these metrics."


def exploration_sentence(results):
    if len(results) < 2:
        return "Exploration effect status: not enough settings."
    low = results[0]["tasks_solved_at_least_once"]
    high = results[-1]["tasks_solved_at_least_once"]
    if high > low:
        return f"Higher exploration solved more AB tasks at least once ({high} vs {low})."
    return f"Higher exploration did not solve more AB tasks at least once ({high} vs {low})."


if __name__ == "__main__":
    main()
