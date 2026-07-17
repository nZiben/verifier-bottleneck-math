"""Grammar-constrained decoding and evaluation for Countdown composition."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from verifier_bottleneck.data.countdown_composition import (
    NUMERIC_B,
    OPERATORS,
    SYMBOLIC_AB,
    ArithmeticPuzzle,
)
from verifier_bottleneck.live_logging import live_log


class EvaluationSettings(Protocol):
    """Minimal settings required by the temperature sweep."""

    @property
    def temperatures(self) -> tuple[float, ...]: ...

    @property
    def pass_k(self) -> tuple[int, ...]: ...

    @property
    def proposal_batch_size(self) -> int: ...

    @property
    def seed(self) -> int: ...


class ScalarWriter(Protocol):
    """TensorBoard-compatible scalar writer."""

    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> object: ...

    def flush(self) -> object: ...


BatchGenerator = Callable[
    [Sequence[ArithmeticPuzzle], Mapping[int, str] | None, float, int], list[bool]
]


@dataclass
class PostfixState:
    """Mutable grammar state for one constrained postfix proposal."""

    used_mask: int = 0
    stack_depth: int = 0
    finished: bool = False
    tokens: list[str] = field(default_factory=list)

    def legal_actions(self, number_count: int) -> list[str]:
        """Return legal pointer, operator, and termination actions."""
        if self.finished:
            return []
        actions = [
            f"P{index}" for index in range(number_count) if not self.used_mask & (1 << index)
        ]
        if self.stack_depth >= 2:
            actions.extend(OPERATORS)
        if self.stack_depth == 1 and self.used_mask.bit_count() >= 2:
            actions.append("<EOS>")
        return actions

    def apply(self, action: str) -> None:
        """Advance the grammar state after one legal action."""
        if self.finished:
            raise ValueError("cannot apply an action after EOS")
        if action == "<EOS>":
            if self.stack_depth != 1 or self.used_mask.bit_count() < 2:
                raise ValueError("EOS requires one result built from at least two inputs")
            self.finished = True
        elif action.startswith("P"):
            try:
                pointer = int(action[1:])
            except ValueError as error:
                raise ValueError(f"invalid pointer action: {action!r}") from error
            if pointer < 0 or self.used_mask & (1 << pointer):
                raise ValueError(f"pointer is invalid or already used: {action!r}")
            self.used_mask |= 1 << pointer
            self.stack_depth += 1
            self.tokens.append(action)
        elif action in OPERATORS:
            if self.stack_depth < 2:
                raise ValueError("an operator requires two stack values")
            self.stack_depth -= 1
            self.tokens.append(action)
        else:
            raise ValueError(f"unknown postfix action: {action!r}")


def sample_index_at_temperature(
    logits: Sequence[float], *, temperature: float, random_value: float
) -> int:
    """Select from softmax(logits / temperature), or greedily at T=0."""
    if not logits:
        raise ValueError("logits must be non-empty")
    if temperature < 0.0:
        raise ValueError("temperature must be non-negative")
    if temperature == 0.0:
        return max(range(len(logits)), key=logits.__getitem__)
    if not 0.0 <= random_value < 1.0:
        raise ValueError("random_value must be in [0, 1)")

    scaled = [float(logit) / temperature for logit in logits]
    maximum = max(scaled)
    weights = [math.exp(value - maximum) for value in scaled]
    threshold = random_value * sum(weights)
    cumulative = 0.0
    for index, weight in enumerate(weights):
        cumulative += weight
        if threshold < cumulative:
            return index
    return len(weights) - 1


def run_temperature_sweep(
    puzzles: Sequence[ArithmeticPuzzle],
    *,
    settings: EvaluationSettings,
    value_to_symbol: Mapping[int, str],
    generate_batch: BatchGenerator,
    writer: ScalarWriter,
    proposal_outcomes_path: Path,
    log_prefix: str = "composition",
) -> list[dict[str, object]]:
    """Evaluate paired numeric-B and symbolic-A+B tasks at every temperature."""
    maximum_k = settings.pass_k[-1]
    measurements: list[dict[str, object]] = []
    outcomes: list[dict[str, object]] = []
    representations: tuple[tuple[str, Mapping[int, str] | None], ...] = (
        (NUMERIC_B, None),
        (SYMBOLIC_AB, value_to_symbol),
    )
    for representation_index, (benchmark, symbols) in enumerate(representations):
        for temperature_index, temperature in enumerate(settings.temperatures):
            expanded = [puzzle for puzzle in puzzles for _ in range(maximum_k)]
            correctness: list[bool] = []
            total_batches = math.ceil(len(expanded) / settings.proposal_batch_size)
            progress_interval = max(1, total_batches // 10)
            live_log(
                f"{log_prefix} generation started benchmark={benchmark} "
                f"temperature={temperature:.4f} tasks={len(puzzles)} "
                f"proposals={len(expanded)} batches={total_batches}"
            )
            for batch_index, start in enumerate(
                range(0, len(expanded), settings.proposal_batch_size)
            ):
                seed = settings.seed + temperature_index * 1_000_003 + batch_index
                correctness.extend(
                    generate_batch(
                        expanded[start : start + settings.proposal_batch_size],
                        symbols,
                        temperature,
                        seed,
                    )
                )
                completed = batch_index + 1
                if completed % progress_interval == 0 or completed == total_batches:
                    live_log(
                        f"{log_prefix} generation progress benchmark={benchmark} "
                        f"temperature={temperature:.4f} batches={completed}/{total_batches}"
                    )

            solved_at_k = {
                f"pass@{k}": sum(
                    any(correctness[index * maximum_k : index * maximum_k + k])
                    for index in range(len(puzzles))
                )
                for k in settings.pass_k
            }
            pass_at_k = {key: solved / len(puzzles) for key, solved in solved_at_k.items()}
            measurements.append(
                {
                    "benchmark": benchmark,
                    "temperature": temperature,
                    "tasks": len(puzzles),
                    "proposals_per_task": maximum_k,
                    "valid_proposals": sum(correctness),
                    "total_proposals": len(correctness),
                    "solved_at_k": solved_at_k,
                    "pass_at_k": pass_at_k,
                }
            )
            for task_index, puzzle in enumerate(puzzles):
                task_results = correctness[task_index * maximum_k : (task_index + 1) * maximum_k]
                outcomes.append(
                    {
                        "benchmark": benchmark,
                        "temperature": temperature,
                        "task_index": task_index,
                        "numbers": list(puzzle.numbers),
                        "symbols": [value_to_symbol[number] for number in puzzle.numbers],
                        "target": puzzle.target,
                        "successful_proposal_indices": [
                            index for index, correct in enumerate(task_results) if correct
                        ],
                    }
                )

            measurement_step = representation_index * len(settings.temperatures) + temperature_index
            for key, rate in pass_at_k.items():
                writer.add_scalar(f"temperature/{benchmark}/{key}", rate, measurement_step)
            writer.flush()
            live_log(
                f"{log_prefix} benchmark={benchmark} temperature={temperature:.4f} "
                f"pass@{maximum_k}={pass_at_k[f'pass@{maximum_k}']:.4f}"
            )

    proposal_outcomes_path.parent.mkdir(parents=True, exist_ok=True)
    proposal_outcomes_path.write_text(
        "".join(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n" for row in outcomes),
        encoding="utf-8",
    )
    return measurements
