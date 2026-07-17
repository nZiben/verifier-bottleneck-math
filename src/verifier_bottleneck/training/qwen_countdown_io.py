"""Qwen prompt and token encoding for the Countdown composition protocol."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch  # type: ignore[import-not-found]

from verifier_bottleneck.data.countdown_composition import ArithmeticPuzzle

ACTION_SYMBOLS = {
    "P0": "0",
    "P1": "1",
    "P2": "2",
    "P3": "3",
    "P4": "4",
    "P5": "5",
    "+": "+",
    "-": "-",
    "*": "*",
    "/": "/",
}


def _single_token_id(tokenizer: Any, symbol: str, used_ids: set[int]) -> int:
    for representation in (f" {symbol}", symbol):
        token_ids = tokenizer.encode(representation, add_special_tokens=False)
        if len(token_ids) == 1 and int(token_ids[0]) not in used_ids:
            return int(token_ids[0])
    raise RuntimeError(f"no single Qwen token represents {symbol!r}")


def action_token_ids(tokenizer: Any) -> dict[str, int]:
    """Map every constrained postfix action to one distinct Qwen token."""
    if tokenizer.eos_token_id is None:
        raise RuntimeError("Qwen tokenizer does not define EOS")
    used_ids = {int(tokenizer.eos_token_id)}
    result: dict[str, int] = {}
    for action, symbol in ACTION_SYMBOLS.items():
        token_id = _single_token_id(tokenizer, symbol, used_ids)
        result[action] = token_id
        used_ids.add(token_id)
    result["<EOS>"] = int(tokenizer.eos_token_id)
    return result


def decode_prompt(symbol: str) -> str:
    """Build one isolated Skill-A prompt."""
    return f"Decode the arbitrary symbol {symbol}. Numeric value:"


def countdown_prompt(puzzle: ArithmeticPuzzle, *, value_to_symbol: Mapping[int, str] | None) -> str:
    """Build a numeric-B or symbolic-A+B Countdown prompt."""
    values = (
        [str(number) for number in puzzle.numbers]
        if value_to_symbol is None
        else [value_to_symbol[number] for number in puzzle.numbers]
    )
    indexed = ", ".join(f"{index}={value}" for index, value in enumerate(values))
    return (
        f"Solve Countdown. Indexed inputs: {indexed}. Target: {puzzle.target}. "
        "Return only a postfix expression using index digits and operators + - * /. "
        "Postfix:"
    )


def encoded_decode_rows(
    decode_examples: Sequence[Any],
    tokenizer: Any,
    numeric_sequences: Mapping[int, Sequence[int]],
) -> list[tuple[list[int], int]]:
    """Encode supervised isolated Skill-A rows."""
    rows = []
    eos_id = int(tokenizer.eos_token_id)
    for example in decode_examples:
        prompt_ids = [
            int(token_id)
            for token_id in tokenizer.encode(
                decode_prompt(str(example.symbol)), add_special_tokens=True
            )
        ]
        answer_ids = [*numeric_sequences[int(example.value)], eos_id]
        rows.append((prompt_ids + answer_ids, len(prompt_ids)))
    return rows


def encoded_countdown_rows(
    puzzles: Sequence[ArithmeticPuzzle],
    *,
    tokenizer: Any,
    action_ids: Mapping[str, int],
    value_to_symbol: Mapping[int, str] | None,
) -> list[tuple[list[int], int]]:
    """Encode supervised numeric-B or diagnostic symbolic-A+B rows."""
    rows = []
    for puzzle in puzzles:
        prompt_ids = [
            int(token_id)
            for token_id in tokenizer.encode(
                countdown_prompt(puzzle, value_to_symbol=value_to_symbol),
                add_special_tokens=True,
            )
        ]
        answer_ids = [action_ids[action] for action in puzzle.solution]
        answer_ids.append(action_ids["<EOS>"])
        rows.append((prompt_ids + answer_ids, len(prompt_ids)))
    return rows


def padded_prompts(
    puzzles: Sequence[ArithmeticPuzzle],
    *,
    tokenizer: Any,
    value_to_symbol: Mapping[int, str] | None,
    device: Any,
) -> tuple[Any, Any]:
    """Left-pad prompts for batched cached autoregressive decoding."""
    prompts = [
        [
            int(token_id)
            for token_id in tokenizer.encode(
                countdown_prompt(puzzle, value_to_symbol=value_to_symbol),
                add_special_tokens=True,
            )
        ]
        for puzzle in puzzles
    ]
    maximum_length = max(len(prompt) for prompt in prompts)
    pad_id = int(tokenizer.pad_token_id)
    sequences = torch.full((len(prompts), maximum_length), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(sequences)
    for row_index, prompt in enumerate(prompts):
        sequences[row_index, maximum_length - len(prompt) :] = torch.tensor(
            prompt, dtype=torch.long, device=device
        )
        attention_mask[row_index, maximum_length - len(prompt) :] = 1
    return sequences, attention_mask
