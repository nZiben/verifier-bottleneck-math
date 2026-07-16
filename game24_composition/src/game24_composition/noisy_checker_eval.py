"""Simulate noisy verifier acceptance on already-scored generations."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from .utils import ensure_dir, read_jsonl, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--alphas", default="0,0.01,0.05,0.10")
    parser.add_argument("--betas", default="0,0.05,0.10,0.20")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = read_jsonl(args.generations)
    results = [
        score_noise(rows, alpha, beta, args.seed)
        for alpha in parse_floats(args.alphas)
        for beta in parse_floats(args.betas)
    ]
    write_json(args.out_json, {"generations": args.generations, "results": results})
    write_csv(args.out_csv, results)
    print(f"Wrote noisy checker sweep to {args.out_json} and {args.out_csv}")


def parse_floats(value):
    return [float(part) for part in str(value).split(",") if part.strip()]


def score_noise(rows, alpha, beta, seed):
    rng = random.Random(f"{seed}:{alpha}:{beta}")
    accepted = accepted_correct = accepted_wrong = false_rejects = 0
    tasks_with_accept = set()
    tasks_with_wrong_accept = set()
    correct_total = sum(1 for row in rows if row.get("is_correct"))

    for row in rows:
        is_correct = bool(row.get("is_correct"))
        accept = rng.random() >= beta if is_correct else rng.random() < alpha
        if not accept:
            false_rejects += int(is_correct)
            continue
        accepted += 1
        tasks_with_accept.add(row.get("id"))
        if is_correct:
            accepted_correct += 1
        else:
            accepted_wrong += 1
            tasks_with_wrong_accept.add(row.get("id"))

    return {
        "alpha_false_accept": alpha,
        "beta_false_reject": beta,
        "accepted": accepted,
        "accepted_correct": accepted_correct,
        "accepted_wrong": accepted_wrong,
        "accept_precision": accepted_correct / accepted if accepted else 0.0,
        "correct_recall": accepted_correct / correct_total if correct_total else 0.0,
        "false_rejects": false_rejects,
        "tasks_with_accept": len(tasks_with_accept),
        "tasks_with_wrong_accept": len(tasks_with_wrong_accept),
    }


def write_csv(path, results):
    ensure_dir(Path(path).parent)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
