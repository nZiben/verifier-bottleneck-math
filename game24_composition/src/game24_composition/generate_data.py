"""Generate A, B, and AB synthetic JSONL data."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .solver import enumerate_solvable_tuples, solve_game24
from .symbols import NUMBER_TO_SYMBOL, SYMBOL_TO_NUMBER, SYMBOLS, symbols_to_numbers
from .utils import canonical_tuple, ensure_dir, write_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="data/synthetic")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_train_A", type=int, default=3000)
    parser.add_argument("--n_eval_A", type=int, default=300)
    parser.add_argument("--n_train_B", type=int, default=3000)
    parser.add_argument("--n_eval_B", type=int, default=300)
    parser.add_argument("--n_test_AB", type=int, default=300)
    parser.add_argument("--value_min", type=int, default=1)
    parser.add_argument("--value_max", type=int, default=10)
    args = parser.parse_args()

    generate(args)


def generate(args):
    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    solvable = enumerate_solvable_tuples(args.value_min, args.value_max)
    rng.shuffle(solvable)
    train_pool, eval_pool, test_pool = split_tuple_pools(solvable)

    train_A = [make_decode_example("train_A", idx, rng) for idx in range(args.n_train_A)]
    eval_A = [make_decode_example("eval_A", idx, rng) for idx in range(args.n_eval_A)]
    train_B = [make_numeric_example("train_B", idx, rng, train_pool) for idx in range(args.n_train_B)]
    eval_B = [make_numeric_example("eval_B", idx, rng, eval_pool) for idx in range(args.n_eval_B)]
    test_AB = [make_symbolic_game24_example("test_AB", idx, rng, test_pool) for idx in range(args.n_test_AB)]
    train_sep = train_A + train_B

    train_canon = {tuple(row["metadata"]["canonical_tuple"]) for row in train_B}
    test_canon = {tuple(row["metadata"]["canonical_tuple"]) for row in test_AB}
    assert train_canon.isdisjoint(test_canon)

    write_jsonl(out_dir / "train_A.jsonl", train_A)
    write_jsonl(out_dir / "eval_A.jsonl", eval_A)
    write_jsonl(out_dir / "train_B.jsonl", train_B)
    write_jsonl(out_dir / "eval_B.jsonl", eval_B)
    write_jsonl(out_dir / "test_AB.jsonl", test_AB)
    write_jsonl(out_dir / "train_sep.jsonl", train_sep)

    manifest = {
        "seed": args.seed,
        "value_range": [args.value_min, args.value_max],
        "symbol_mapping": SYMBOL_TO_NUMBER,
        "num_solvable_canonical_tuples": len(solvable),
        "tuple_pool_sizes": {
            "train_B": len(train_pool),
            "eval_B": len(eval_pool),
            "test_AB": len(test_pool),
        },
        "dataset_sizes": {
            "train_A": len(train_A),
            "eval_A": len(eval_A),
            "train_B": len(train_B),
            "eval_B": len(eval_B),
            "test_AB": len(test_AB),
            "train_sep": len(train_sep),
        },
        "train_B_test_AB_disjoint": True,
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"Wrote synthetic data to {out_dir}")
    print(f"Solvable canonical tuples: {len(solvable)}")
    print(f"Pool sizes: train={len(train_pool)} eval={len(eval_pool)} test={len(test_pool)}")


def split_tuple_pools(solvable):
    if len(solvable) < 3:
        raise ValueError("Need at least three solvable canonical tuples to split train/eval/test pools.")
    n_total = len(solvable)
    n_eval = max(1, n_total // 5)
    n_test = max(1, n_total // 5)
    eval_pool = solvable[:n_eval]
    test_pool = solvable[n_eval : n_eval + n_test]
    train_pool = solvable[n_eval + n_test :]
    if not train_pool:
        raise ValueError("No tuples left for training pool.")
    return train_pool, eval_pool, test_pool


def make_decode_example(split, idx, rng):
    symbols = [rng.choice(SYMBOLS) for _ in range(4)]
    numbers = symbols_to_numbers(symbols)
    return {
        "id": f"{split}_{idx:06d}",
        "task_type": "A_decode",
        "question": "Decode the symbols into numbers.\nSymbols: " + " ".join(symbols),
        "answer": " ".join(str(number) for number in numbers),
        "metadata": {"symbols": symbols, "numbers": numbers},
    }


def make_numeric_example(split, idx, rng, pool):
    numbers = list(rng.choice(pool))
    rng.shuffle(numbers)
    solution = solve_game24(numbers)
    return {
        "id": f"{split}_{idx:06d}",
        "task_type": "B_game24_numeric",
        "question": (
            "Use each number exactly once with +, -, *, / and parentheses to make 24.\n"
            "Numbers: "
            + " ".join(str(number) for number in numbers)
            + "\nReturn only the final expression inside <answer>...</answer>."
        ),
        "answer": f"<answer>{solution}</answer>",
        "metadata": {
            "numbers": numbers,
            "solution": solution,
            "canonical_tuple": list(canonical_tuple(numbers)),
        },
    }


def make_symbolic_game24_example(split, idx, rng, pool):
    numbers = list(rng.choice(pool))
    rng.shuffle(numbers)
    symbols = [NUMBER_TO_SYMBOL[number] for number in numbers]
    solution = solve_game24(numbers)
    return {
        "id": f"{split}_{idx:06d}",
        "task_type": "AB_symbolic_game24",
        "question": (
            "Use each symbol exactly once. First decode the symbols into numbers, then make 24 "
            "using +, -, *, / and parentheses.\nSymbols: "
            + " ".join(symbols)
            + "\nReturn only the final expression inside <answer>...</answer>."
        ),
        "answer": f"<answer>{solution}</answer>",
        "metadata": {
            "symbols": symbols,
            "numbers": numbers,
            "solution": solution,
            "canonical_tuple": list(canonical_tuple(numbers)),
        },
    }


if __name__ == "__main__":
    main()
