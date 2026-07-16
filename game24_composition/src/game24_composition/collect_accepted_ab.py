"""Build AB self-training data from generations accepted by the perfect checker."""

from __future__ import annotations

import argparse
from pathlib import Path

from .utils import read_jsonl, write_json, write_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", required=True)
    parser.add_argument("--generations", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--stats_out", default=None)
    parser.add_argument("--max_per_task", type=int, default=4)
    args = parser.parse_args()

    kept, stats = collect_accepted_ab(
        read_jsonl(args.examples),
        read_jsonl(args.generations),
        args.max_per_task,
        args.generations,
        args.examples,
    )
    write_jsonl(args.out, kept)
    write_json(args.stats_out or str(Path(args.out).with_suffix(".stats.json")), stats)
    print(f"Wrote {len(kept)} accepted AB examples to {args.out}")


def collect_accepted_ab(examples_rows, generation_rows, max_per_task, generations_path=None, examples_path=None):
    examples = {row["id"]: row for row in examples_rows}
    kept = []
    per_task = {}
    correct_rows = 0
    for row in generation_rows:
        if not row.get("is_correct"):
            continue
        correct_rows += 1
        example = examples.get(row["id"])
        if not example:
            continue
        count = per_task.get(row["id"], 0)
        if count >= max_per_task:
            continue
        checker = row.get("checker") or {}
        expression = checker.get("extracted_expression")
        if not expression:
            continue
        kept.append(
            {
                **example,
                "id": f"{example['id']}_accepted_{count}",
                "answer": f"<answer>{expression}</answer>",
                "metadata": {
                    **example.get("metadata", {}),
                    "teacher": "perfect_checker",
                    "source_sample_index": row.get("sample_index"),
                },
            }
        )
        per_task[row["id"]] = count + 1

    return kept, {
        "source_generations": generations_path,
        "source_examples": examples_path,
        "correct_generation_rows": correct_rows,
        "accepted_training_examples": len(kept),
        "tasks_with_accepted_examples": len(per_task),
        "max_per_task": max_per_task,
    }


if __name__ == "__main__":
    main()
