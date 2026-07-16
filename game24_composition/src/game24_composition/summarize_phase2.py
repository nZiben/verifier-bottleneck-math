"""Write a compact markdown summary for phase 2 experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from .utils import read_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--outputs_dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    outputs = Path(args.outputs_dir)
    manifest = read_json(Path(args.data_dir) / "manifest.json", default={})
    accepted = read_json(outputs / "accepted_ab_train.stats.json", default={})
    noisy = read_json(outputs / "noisy_checker_eval.json", default={})

    lines = [
        "# Phase 2 Experiment Summary",
        "",
        "## Dataset",
        bullet_sizes(manifest.get("dataset_sizes", {})),
        "",
        "## Base Model Before SFT",
        split_metrics(read_json(outputs / "base_model_results.json", default={})),
        passk(read_json(outputs / "base_model_passk.json", default={})),
        "",
        "## Strong B-only Model",
        split_metrics(read_json(outputs / "b_only_results.json", default={})),
        "",
        "## Strong A+B Model",
        split_metrics(read_json(outputs / "strong_sep_results.json", default={})),
        passk(read_json(outputs / "strong_sep_passk.json", default={})),
        "",
        "## Perfect-checker Self-training",
        f"- accepted AB training examples: {accepted.get('accepted_training_examples', 'not run')}",
        f"- tasks with accepted examples: {accepted.get('tasks_with_accepted_examples', 'not run')}",
        split_metrics(read_json(outputs / "self_train_results.json", default={})),
        passk(read_json(outputs / "self_train_passk.json", default={})),
        "",
        "## Noisy Checker",
        noisy_table(noisy.get("results", [])),
        "",
    ]
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.out}")


def bullet_sizes(sizes):
    return "\n".join(f"- {key}: {value}" for key, value in sizes.items()) or "- not run"


def split_metrics(summary):
    splits = summary.get("splits", {})
    if not splits:
        return "- not run"
    lines = []
    for name, row in splits.items():
        metric = row.get("decode_accuracy")
        metric = row.get("numeric_game24_pass@1", metric)
        metric = row.get("symbolic_game24_pass@1", metric)
        lines.append(f"- {name}: {fmt(metric)} ({row.get('num_correct')}/{row.get('num_examples')})")
    return "\n".join(lines)


def passk(summary):
    values = summary.get("pass_at_k", {})
    if not values:
        return "- pass@k: not run"
    return "\n".join(f"- pass@{key}: {fmt(value)}" for key, value in sorted(values.items(), key=lambda item: int(item[0])))


def noisy_table(results):
    if not results:
        return "- not run"
    lines = [
        "| alpha | beta | accepted | wrong | precision | recall | wrong-task accepts |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        lines.append(
            "| {alpha_false_accept:.2f} | {beta_false_reject:.2f} | {accepted} | {accepted_wrong} | {precision} | {recall} | {tasks_with_wrong_accept} |".format(
                precision=fmt(row["accept_precision"]),
                recall=fmt(row["correct_recall"]),
                **row,
            )
        )
    return "\n".join(lines)


def fmt(value):
    return "not run" if value is None else f"{value:.4f}"


if __name__ == "__main__":
    main()
