"""Shared configuration and curriculum protocol for Countdown A+B experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from verifier_bottleneck.configuration import (
    as_float,
    as_int,
    as_mapping,
    require_exact_keys,
)
from verifier_bottleneck.data.countdown_composition import (
    CountdownCompositionSplit,
    make_countdown_composition_split,
)

TASK_FIELDS = {
    "countdown_train_examples",
    "countdown_validation_examples",
    "countdown_test_examples",
    "countdown_minimum_target",
    "countdown_maximum_target",
    "symbol_seed",
}
EVALUATION_FIELDS = {"temperatures", "pass_k", "proposal_batch_size", "seed"}


@dataclass(frozen=True)
class CompositionTaskConfig:
    """Countdown split and arbitrary-symbol construction settings."""

    countdown_train_examples: int
    countdown_validation_examples: int
    countdown_test_examples: int
    countdown_minimum_target: int
    countdown_maximum_target: int
    symbol_seed: int

    def validate(self) -> None:
        """Validate split sizes and the target interval."""
        if (
            min(
                self.countdown_train_examples,
                self.countdown_validation_examples,
                self.countdown_test_examples,
            )
            < 1
        ):
            raise ValueError("Countdown split counts must be positive")
        if not 1 <= self.countdown_minimum_target <= self.countdown_maximum_target <= 999:
            raise ValueError("Countdown target range must be within [1, 999]")


@dataclass(frozen=True)
class EvaluationConfig:
    """Grammar-constrained temperature-sampling settings."""

    temperatures: tuple[float, ...]
    pass_k: tuple[int, ...]
    proposal_batch_size: int
    seed: int

    def validate(self) -> None:
        """Validate bounded temperatures and proposal counts."""
        if not self.temperatures or any(
            not 0.0 <= temperature <= 5.0 for temperature in self.temperatures
        ):
            raise ValueError("evaluation temperatures must be non-empty and in [0, 5]")
        if len(set(self.temperatures)) != len(self.temperatures):
            raise ValueError("evaluation temperatures must be unique")
        if (
            not self.pass_k
            or tuple(sorted(set(self.pass_k))) != self.pass_k
            or self.pass_k[0] < 1
            or self.pass_k[-1] > 4096
        ):
            raise ValueError("evaluation.pass_k must be increasing and in [1, 4096]")
        if self.proposal_batch_size < 1:
            raise ValueError("proposal_batch_size must be positive")


class ProtocolConfig(Protocol):
    """Common fields shared by scratch and pretrained configurations."""

    @property
    def seed(self) -> int: ...

    @property
    def task(self) -> CompositionTaskConfig: ...

    @property
    def evaluation(self) -> EvaluationConfig: ...


def validate_experiment_metadata(
    *, experiment_name: str, description: str, tags: tuple[str, ...], device: str
) -> None:
    """Validate model-independent experiment metadata."""
    if not experiment_name.strip() or not description.strip():
        raise ValueError("experiment_name and description must not be empty")
    if not tags or any(not tag.strip() for tag in tags):
        raise ValueError("tags must contain non-empty values")
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")


def parse_task_config(value: object) -> CompositionTaskConfig:
    """Parse the shared ``task`` YAML section."""
    raw = as_mapping(value, field="task")
    require_exact_keys(raw, field="task", required=TASK_FIELDS)
    config = CompositionTaskConfig(
        **{field: as_int(raw[field], field=f"task.{field}") for field in TASK_FIELDS}
    )
    config.validate()
    return config


def parse_evaluation_config(value: object) -> EvaluationConfig:
    """Parse the shared ``evaluation`` YAML section."""
    raw = as_mapping(value, field="evaluation")
    require_exact_keys(raw, field="evaluation", required=EVALUATION_FIELDS)
    temperatures = raw["temperatures"]
    pass_k = raw["pass_k"]
    if not isinstance(temperatures, list) or not isinstance(pass_k, list):
        raise ValueError("evaluation temperatures and pass_k must be lists")
    config = EvaluationConfig(
        temperatures=tuple(
            as_float(value, field="evaluation.temperatures[]") for value in temperatures
        ),
        pass_k=tuple(as_int(value, field="evaluation.pass_k[]") for value in pass_k),
        proposal_batch_size=as_int(
            raw["proposal_batch_size"], field="evaluation.proposal_batch_size"
        ),
        seed=as_int(raw["seed"], field="evaluation.seed"),
    )
    config.validate()
    return config


def build_composition_dataset(config: ProtocolConfig) -> CountdownCompositionSplit:
    """Build the deterministic A/B training and paired B/A+B test split."""
    task = config.task
    return make_countdown_composition_split(
        train_examples=task.countdown_train_examples,
        validation_examples=task.countdown_validation_examples,
        test_examples=task.countdown_test_examples,
        minimum_target=task.countdown_minimum_target,
        maximum_target=task.countdown_maximum_target,
        seed=config.seed,
        symbol_seed=task.symbol_seed,
    )


def proposal_count(config: ProtocolConfig) -> int:
    """Return the maximum paired B/A+B proposals generated by a run."""
    return (
        2
        * config.task.countdown_test_examples
        * len(config.evaluation.temperatures)
        * config.evaluation.pass_k[-1]
    )


def replay_example_count(total_examples: int, replay_fraction: float) -> int:
    """Return the nearest deterministic replay count for one effective batch."""
    if total_examples < 2 or not 0.0 < replay_fraction < 1.0:
        raise ValueError("replay allocation requires at least two examples and fraction in (0, 1)")
    count = round(total_examples * replay_fraction)
    if not 1 <= count < total_examples:
        raise ValueError("replay fraction must allocate at least one example to A and B")
    return count


def distribute_replay_examples(
    *, micro_batch_size: int, accumulation_steps: int, replay_fraction: float
) -> tuple[int, ...]:
    """Distribute effective A replay across accumulated microbatches."""
    effective_batch_size = micro_batch_size * accumulation_steps
    replay_total = replay_example_count(effective_batch_size, replay_fraction)
    counts = tuple(
        round((index + 1) * replay_total / accumulation_steps)
        - round(index * replay_total / accumulation_steps)
        for index in range(accumulation_steps)
    )
    if any(not 0 <= count <= micro_batch_size for count in counts):
        raise ValueError("replay allocation does not fit within the microbatch size")
    return counts
