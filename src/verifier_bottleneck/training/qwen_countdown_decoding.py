"""Qwen-specific inference adapters for the shared Countdown evaluation."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any

import torch  # type: ignore[import-not-found]

from verifier_bottleneck.data.countdown_composition import ArithmeticPuzzle, verify_postfix
from verifier_bottleneck.evaluation.countdown import PostfixState, sample_index_at_temperature
from verifier_bottleneck.training.qwen_countdown_io import decode_prompt, padded_prompts


def decode_exact_accuracy(
    model: Any,
    *,
    decode_examples: Sequence[Any],
    tokenizer: Any,
    numeric_sequences: Mapping[int, Sequence[int]],
    device: Any,
) -> tuple[int, int]:
    """Measure exact isolated Skill-A decoding with constrained numeric answers."""
    eos_id = int(tokenizer.eos_token_id)
    candidates = {
        value: (*tuple(int(token_id) for token_id in token_ids), eos_id)
        for value, token_ids in numeric_sequences.items()
    }
    sequence_to_value = {sequence: value for value, sequence in candidates.items()}
    maximum_answer_length = max(len(sequence) for sequence in candidates.values())
    solved = 0
    model.eval()
    with torch.no_grad():
        for example in decode_examples:
            prompt = [
                int(token_id)
                for token_id in tokenizer.encode(
                    decode_prompt(str(example.symbol)), add_special_tokens=True
                )
            ]
            generated: list[int] = []
            active_candidates = dict(candidates)
            for _ in range(maximum_answer_length):
                prefix_length = len(generated)
                legal_ids = sorted(
                    {
                        sequence[prefix_length]
                        for sequence in active_candidates.values()
                        if len(sequence) > prefix_length
                    }
                )
                input_ids = torch.tensor([prompt + generated], dtype=torch.long, device=device)
                attention_mask = torch.ones_like(input_ids)
                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                ).logits[0, -1, :]
                chosen_id = legal_ids[int(logits[legal_ids].argmax().item())]
                generated.append(chosen_id)
                generated_prefix = tuple(generated)
                active_candidates = {
                    value: sequence
                    for value, sequence in active_candidates.items()
                    if sequence[: len(generated_prefix)] == generated_prefix
                }
                if chosen_id == eos_id:
                    break
            predicted_value = sequence_to_value.get(tuple(generated))
            solved += int(predicted_value == int(example.value))
    return solved, len(decode_examples)


def generate_batch(
    model: Any,
    puzzles: Sequence[ArithmeticPuzzle],
    *,
    tokenizer: Any,
    action_ids: Mapping[str, int],
    value_to_symbol: Mapping[int, str] | None,
    temperature: float,
    seed: int,
    device: Any,
) -> list[bool]:
    """Generate and verify one batch of grammar-constrained postfix proposals."""
    sequences, attention_mask = padded_prompts(
        puzzles,
        tokenizer=tokenizer,
        value_to_symbol=value_to_symbol,
        device=device,
    )
    states = [PostfixState() for _ in puzzles]
    random_sources = [
        random.Random(seed + row_index * 1_000_000_007) for row_index in range(len(puzzles))
    ]
    maximum_tokens = 2 * max(len(puzzle.numbers) for puzzle in puzzles)
    pad_id = int(tokenizer.pad_token_id)
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=sequences, attention_mask=attention_mask, use_cache=True)
        next_logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values
        for _ in range(maximum_tokens):
            next_ids = [pad_id] * len(puzzles)
            active = [0] * len(puzzles)
            for row_index, puzzle in enumerate(puzzles):
                state = states[row_index]
                if state.finished:
                    continue
                legal_actions = state.legal_actions(len(puzzle.numbers))
                legal_ids = [action_ids[action] for action in legal_actions]
                random_source = random_sources[row_index]
                random_value = random_source.random() if temperature > 0.0 else 0.0
                chosen_action = legal_actions[
                    sample_index_at_temperature(
                        next_logits[row_index, legal_ids].float().tolist(),
                        temperature=temperature,
                        random_value=random_value,
                    )
                ]
                next_ids[row_index] = action_ids[chosen_action]
                active[row_index] = 1
                state.apply(chosen_action)
            attention_mask = torch.cat(
                (
                    attention_mask,
                    torch.tensor(active, dtype=torch.long, device=device).unsqueeze(1),
                ),
                dim=1,
            )
            if all(state.finished for state in states):
                break
            outputs = model(
                input_ids=torch.tensor(next_ids, dtype=torch.long, device=device).unsqueeze(1),
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            next_logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values
    return [
        state.finished and verify_postfix(puzzle, tuple(state.tokens))
        for puzzle, state in zip(puzzles, states, strict=True)
    ]
