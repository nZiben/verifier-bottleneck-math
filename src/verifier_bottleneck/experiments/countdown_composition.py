"""Scratch-transformer A+B composition experiment for symbolic Countdown."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from verifier_bottleneck.configuration import (
    as_float,
    as_int,
    as_mapping,
    as_str,
    as_str_tuple,
    load_yaml,
    require_exact_keys,
)
from verifier_bottleneck.data.countdown_composition import (
    COUNTDOWN_VALUES,
    ArithmeticPuzzle,
    CountdownCompositionSplit,
    SymbolDecodingExample,
)
from verifier_bottleneck.experiments.countdown_protocol import (
    CompositionTaskConfig,
    EvaluationConfig,
    build_composition_dataset,
    parse_evaluation_config,
    parse_task_config,
    proposal_count,
    validate_experiment_metadata,
)
from verifier_bottleneck.experiments.countdown_runtime import (
    ArtifactSpec,
    run_countdown_experiment,
)

COMPOSITION_SPECIAL_TOKENS = (
    "<PAD>",
    "<BOS>",
    "<DECODE>",
    "<COUNTDOWN>",
    "<SEP>",
    "<EOS>",
    "+",
    "-",
    "*",
    "/",
    "P0",
    "P1",
    "P2",
    "P3",
    "P4",
    "P5",
)
SYMBOL_TOKENS = tuple(f"S{index}" for index in range(len(COUNTDOWN_VALUES)))


@dataclass(frozen=True)
class ModelConfig:
    """Scratch transformer shape and context length."""

    layers: int
    d_model: int
    heads: int
    d_ff: int
    block_size: int
    dropout: float


@dataclass(frozen=True)
class TrainingConfig:
    """Scratch supervised-training settings."""

    skill_a_pretraining_max_steps: int
    skill_a_pretraining_evaluation_interval: int
    skill_a_replay_fraction: float
    steps: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    log_interval: int
    evaluation_interval: int
    early_stopping_patience: int
    early_stopping_min_delta: float


@dataclass(frozen=True)
class CountdownCompositionConfig:
    """Complete scratch A/B-to-A+B composition configuration."""

    experiment_name: str
    description: str
    tags: tuple[str, ...]
    seed: int
    device: str
    output: Path
    task: CompositionTaskConfig
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig

    def validate(self) -> None:
        """Reject invalid or accidentally unbounded settings."""
        validate_experiment_metadata(
            experiment_name=self.experiment_name,
            description=self.description,
            tags=self.tags,
            device=self.device,
        )
        self.task.validate()
        if not 1 <= self.model.layers <= 8:
            raise ValueError("model.layers must be in [1, 8]")
        if self.model.d_model < 16 or self.model.d_model % self.model.heads:
            raise ValueError("model.d_model must be at least 16 and divisible by heads")
        if self.model.d_ff < self.model.d_model:
            raise ValueError("model.d_ff must be at least model.d_model")
        if not 22 <= self.model.block_size <= 512:
            raise ValueError("model.block_size must be in [22, 512]")
        if not 0.0 <= self.model.dropout < 1.0:
            raise ValueError("model.dropout must be in [0, 1)")
        if self.training.steps < 1 or self.training.batch_size < 2:
            raise ValueError("training steps must be positive and batch size at least two")
        if (
            min(
                self.training.skill_a_pretraining_max_steps,
                self.training.skill_a_pretraining_evaluation_interval,
            )
            < 1
        ):
            raise ValueError("Skill A pretraining counts must be positive")
        if not 0.0 < self.training.skill_a_replay_fraction < 1.0:
            raise ValueError("skill_a_replay_fraction must be in (0, 1)")
        replay_examples = round(self.training.batch_size * self.training.skill_a_replay_fraction)
        if not 1 <= replay_examples < self.training.batch_size:
            raise ValueError("skill_a_replay_fraction must select both A and B examples")
        if self.training.learning_rate <= 0.0 or self.training.weight_decay < 0.0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if not 0 <= self.training.warmup_steps <= self.training.steps:
            raise ValueError("warmup_steps must be between zero and training.steps")
        if (
            min(
                self.training.log_interval,
                self.training.evaluation_interval,
                self.training.early_stopping_patience,
            )
            < 1
        ):
            raise ValueError("logging, evaluation, and patience values must be positive")
        if self.training.early_stopping_min_delta < 0.0:
            raise ValueError("early_stopping_min_delta must be non-negative")
        self.evaluation.validate()

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible configuration."""
        value = asdict(self)
        value["output"] = self.output.as_posix()
        return cast(dict[str, object], value)


@dataclass(frozen=True)
class CompositionVocabulary:
    """Atomic scratch tokens for isolated decoding and Countdown solving."""

    tokens: tuple[str, ...]

    @classmethod
    def build(cls) -> CompositionVocabulary:
        number_tokens = tuple(f"N{number}" for number in range(1000))
        return cls(COMPOSITION_SPECIAL_TOKENS + SYMBOL_TOKENS + number_tokens)

    @property
    def token_to_id(self) -> dict[str, int]:
        """Map token strings to IDs."""
        return {token: index for index, token in enumerate(self.tokens)}

    def encode_decode(self, example: SymbolDecodingExample) -> tuple[list[int], int]:
        """Encode an isolated A mapping example."""
        prompt = ("<BOS>", "<DECODE>", example.symbol, "<SEP>")
        answer = (f"N{example.value}", "<EOS>")
        mapping = self.token_to_id
        return [mapping[token] for token in prompt + answer], len(prompt)

    def encode_countdown(
        self,
        example: ArithmeticPuzzle,
        *,
        value_to_symbol: Mapping[int, str] | None,
    ) -> tuple[list[int], int]:
        """Encode numeric B or symbolic unseen A+B using the same task framing."""
        number_tokens = (
            [f"N{number}" for number in example.numbers]
            if value_to_symbol is None
            else [value_to_symbol[number] for number in example.numbers]
        )
        prompt = ["<BOS>", "<COUNTDOWN>", *number_tokens, f"N{example.target}", "<SEP>"]
        answer = [*example.solution, "<EOS>"]
        mapping = self.token_to_id
        return [mapping[token] for token in prompt + answer], len(prompt)


def parse_countdown_composition_config(raw: object) -> CountdownCompositionConfig:
    """Parse a strict YAML-compatible scratch composition config."""
    root = as_mapping(raw, field="config")
    require_exact_keys(
        root,
        field="config",
        required={
            "experiment_name",
            "description",
            "tags",
            "seed",
            "device",
            "output",
            "task",
            "model",
            "training",
            "evaluation",
        },
    )
    model = as_mapping(root["model"], field="model")
    require_exact_keys(
        model,
        field="model",
        required={"layers", "d_model", "heads", "d_ff", "block_size", "dropout"},
    )
    training = as_mapping(root["training"], field="training")
    training_fields = {
        "skill_a_pretraining_max_steps",
        "skill_a_pretraining_evaluation_interval",
        "skill_a_replay_fraction",
        "steps",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "warmup_steps",
        "log_interval",
        "evaluation_interval",
        "early_stopping_patience",
        "early_stopping_min_delta",
    }
    require_exact_keys(training, field="training", required=training_fields)
    config = CountdownCompositionConfig(
        experiment_name=as_str(root["experiment_name"], field="experiment_name"),
        description=as_str(root["description"], field="description"),
        tags=as_str_tuple(root["tags"], field="tags"),
        seed=as_int(root["seed"], field="seed"),
        device=as_str(root["device"], field="device"),
        output=Path(as_str(root["output"], field="output")),
        task=parse_task_config(root["task"]),
        model=ModelConfig(
            layers=as_int(model["layers"], field="model.layers"),
            d_model=as_int(model["d_model"], field="model.d_model"),
            heads=as_int(model["heads"], field="model.heads"),
            d_ff=as_int(model["d_ff"], field="model.d_ff"),
            block_size=as_int(model["block_size"], field="model.block_size"),
            dropout=as_float(model["dropout"], field="model.dropout"),
        ),
        training=TrainingConfig(
            skill_a_pretraining_max_steps=as_int(
                training["skill_a_pretraining_max_steps"],
                field="training.skill_a_pretraining_max_steps",
            ),
            skill_a_pretraining_evaluation_interval=as_int(
                training["skill_a_pretraining_evaluation_interval"],
                field="training.skill_a_pretraining_evaluation_interval",
            ),
            skill_a_replay_fraction=as_float(
                training["skill_a_replay_fraction"],
                field="training.skill_a_replay_fraction",
            ),
            steps=as_int(training["steps"], field="training.steps"),
            batch_size=as_int(training["batch_size"], field="training.batch_size"),
            learning_rate=as_float(training["learning_rate"], field="training.learning_rate"),
            weight_decay=as_float(training["weight_decay"], field="training.weight_decay"),
            warmup_steps=as_int(training["warmup_steps"], field="training.warmup_steps"),
            log_interval=as_int(training["log_interval"], field="training.log_interval"),
            evaluation_interval=as_int(
                training["evaluation_interval"], field="training.evaluation_interval"
            ),
            early_stopping_patience=as_int(
                training["early_stopping_patience"],
                field="training.early_stopping_patience",
            ),
            early_stopping_min_delta=as_float(
                training["early_stopping_min_delta"],
                field="training.early_stopping_min_delta",
            ),
        ),
        evaluation=parse_evaluation_config(root["evaluation"]),
    )
    config.validate()
    return config


def load_countdown_composition_config(path: Path) -> CountdownCompositionConfig:
    """Load a scratch composition YAML config."""
    return parse_countdown_composition_config(load_yaml(path))


def build_countdown_composition_dataset(
    config: CountdownCompositionConfig,
) -> CountdownCompositionSplit:
    """Build the exact A/B/AB split."""
    return build_composition_dataset(config)


def estimate_composition_parameter_count(config: CountdownCompositionConfig) -> int:
    """Estimate the implemented tied-embedding scratch model size."""
    vocabulary_size = len(CompositionVocabulary.build().tokens)
    width = config.model.d_model
    feed_forward = config.model.d_ff
    embeddings = vocabulary_size * width + config.model.block_size * width
    attention = 4 * width * width + 4 * width
    mlp = 2 * width * feed_forward + feed_forward + width
    norms = 4 * width
    return embeddings + config.model.layers * (attention + mlp + norms) + 2 * width


def describe_countdown_composition(config: CountdownCompositionConfig) -> dict[str, object]:
    """Describe the workload without importing PyTorch."""
    config.validate()
    return {
        "experiment_name": config.experiment_name,
        "training_skills": ["A_symbol_decoding", "B_numeric_countdown"],
        "unseen_test": "A+B_symbolic_countdown",
        "decode_mappings": len(COUNTDOWN_VALUES),
        "numeric_train_examples": config.task.countdown_train_examples,
        "numeric_validation_examples": config.task.countdown_validation_examples,
        "paired_numeric_and_symbolic_test_examples": config.task.countdown_test_examples,
        "skill_a_pretraining_max_steps": config.training.skill_a_pretraining_max_steps,
        "skill_a_replay_fraction": config.training.skill_a_replay_fraction,
        "training_steps": config.training.steps,
        "skill_b_phase_steps": config.training.steps,
        "maximum_total_training_steps": (
            config.training.skill_a_pretraining_max_steps + config.training.steps
        ),
        "temperatures": list(config.evaluation.temperatures),
        "pass_k": list(config.evaluation.pass_k),
        "maximum_generated_proposals": proposal_count(config),
        "estimated_parameter_count": estimate_composition_parameter_count(config),
        "downloads_models": False,
        "downloads_datasets": False,
        "launches_cloud_job": False,
        "tensorboard_enabled": True,
    }


def run_countdown_composition_from_path(
    config_path: Path, *, output_path: Path | None = None, device: str | None = None
) -> dict[str, object]:
    """Train separate A/B skills and test unseen symbolic A+B."""
    resolved_config_path = config_path.resolve()
    config = load_countdown_composition_config(resolved_config_path)
    try:
        from verifier_bottleneck.training.countdown_composition_transformer import (
            train_countdown_composition_transformer,
        )
    except ModuleNotFoundError as error:
        if error.name in {"torch", "tensorboard"}:
            raise RuntimeError("PyTorch and TensorBoard are required") from error
        raise

    def trainer(run_directory: Path, selected_device: str) -> dict[str, object]:
        return train_countdown_composition_transformer(
            config,
            device=selected_device,
            tensorboard_directory=run_directory / "tensorboard",
            checkpoint_path=run_directory / "best-model.pt",
            proposal_outcomes_path=run_directory / "proposal-outcomes.jsonl",
            codebook_path=run_directory / "symbol-codebook.json",
        )

    return run_countdown_experiment(
        config=config,
        config_path=resolved_config_path,
        experiment_type="countdown_symbolic_composition",
        trainer=trainer,
        artifacts=(
            ArtifactSpec("best-model.pt", "checkpoint", "Best validation checkpoint."),
            ArtifactSpec(
                "proposal-outcomes.jsonl",
                "evaluation",
                "Paired numeric-B and symbolic-A+B per-task proposal outcomes.",
            ),
            ArtifactSpec(
                "symbol-codebook.json",
                "metadata",
                "Seeded arbitrary symbol-to-number codebook learned as skill A.",
            ),
        ),
        tensorboard_description="TensorBoard event stream for composition training.",
        output_path=output_path,
        device=device,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train separate symbol decoding and Countdown, then test composition."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = _build_parser().parse_args(argv)
    try:
        config = load_countdown_composition_config(args.config.resolve())
        if args.dry_run:
            print(json.dumps(describe_countdown_composition(config), indent=2, sort_keys=True))
            return 0
        result = run_countdown_composition_from_path(
            args.config, output_path=args.output, device=args.device
        )
    except (RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "run_id": result["run_id"],
                "record": result["record_path"],
                "summary": result["summary_path"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
