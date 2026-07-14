"""Small shared utilities."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def write_json(path, payload):
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def read_json(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        return default


def count_jsonl(path):
    with open(path, "r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def canonical_tuple(numbers):
    return tuple(sorted(int(number) for number in numbers))


def parse_pass_k(value):
    return [int(part) for part in str(value).split(",") if part.strip()]


def extract_ints(text):
    return [int(match) for match in re.findall(r"\b\d+\b", text or "")]


def generation_path_for(out_path):
    path = Path(out_path)
    stem = path.stem
    if stem.endswith("_results"):
        stem = stem[: -len("_results")] + "_generations"
    else:
        stem = stem + "_generations"
    return path.with_name(stem + ".jsonl")


def set_seed(seed):
    random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
