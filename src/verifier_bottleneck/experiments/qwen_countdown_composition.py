"""Qwen LoRA A+B composition experiment for symbolic Countdown."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from verifier_bottleneck.configuration import (
    as_float,
    as_int,
    as_mapping,
    as_str,
    as_str_tuple,
    load_yaml,
    require_exact_keys,
)
from verifier_bottleneck.data.countdown_composition import CountdownCompositionSplit
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


def numeric_answer_token_sequences(
    tokenizer: Any, values: Sequence[int]
) -> dict[int, tuple[int, ...]]:
    """Encode numeric Skill-A answers without assuming one token per value."""
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise RuntimeError("Qwen tokenizer does not define EOS")
    result: dict[int, tuple[int, ...]] = {}
    sequence_to_value: dict[tuple[int, ...], int] = {}
    for value in dict.fromkeys(values):
        token_ids = tuple(
            int(token_id) for token_id in tokenizer.encode(f" {value}", add_special_tokens=False)
        )
        if not token_ids:
            raise RuntimeError(f"Qwen tokenizer produced no tokens for {value!r}")
        if int(eos_token_id) in token_ids:
            raise RuntimeError(f"numeric answer {value!r} unexpectedly contains EOS")
        previous_value = sequence_to_value.get(token_ids)
        if previous_value is not None and previous_value != value:
            raise RuntimeError(
                "Qwen tokenizer maps distinct numeric answers "
                f"{previous_value!r} and {value!r} to the same token sequence"
            )
        result[value] = token_ids
        sequence_to_value[token_ids] = value
    return result


@dataclass(frozen=True)
class QwenModelConfig:
    """Pinned Qwen and LoRA adapter settings."""

    pretrained_model: str
    revision: str
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    target_modules: tuple[str, ...]


@dataclass(frozen=True)
class QwenTrainingConfig:
    """Memory-bounded Qwen supervised-training settings."""

    skill_a_pretraining_max_steps: int
    skill_a_pretraining_evaluation_interval: int
    skill_a_replay_fraction: float
    steps: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    maximum_gradient_norm: float
    log_interval: int
    evaluation_interval: int
    early_stopping_patience: int
    early_stopping_min_delta: float


@dataclass(frozen=True)
class QwenCountdownCompositionConfig:
    """Complete Qwen composition configuration."""

    experiment_name: str
    description: str
    tags: tuple[str, ...]
    seed: int
    device: str
    output: Path
    task: CompositionTaskConfig
    model: QwenModelConfig
    training: QwenTrainingConfig
    evaluation: EvaluationConfig

    def validate(self) -> None:
        """Reject invalid settings before any model download."""
        validate_experiment_metadata(
            experiment_name=self.experiment_name,
            description=self.description,
            tags=self.tags,
            device=self.device,
        )
        self.task.validate()
        if self.model.pretrained_model != "Qwen/Qwen2.5-0.5B-Instruct":
            raise ValueError("this job is restricted to Qwen/Qwen2.5-0.5B-Instruct")
        if not self.model.revision.strip() or not self.model.target_modules:
            raise ValueError("Qwen revision and LoRA target modules must not be empty")
        if self.model.lora_rank < 1 or self.model.lora_alpha < 1:
            raise ValueError("LoRA rank and alpha must be positive")
        if not 0.0 <= self.model.lora_dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0, 1)")
        if (
            min(
                self.training.steps,
                self.training.skill_a_pretraining_max_steps,
                self.training.skill_a_pretraining_evaluation_interval,
                self.training.micro_batch_size,
                self.training.gradient_accumulation_steps,
                self.training.log_interval,
                self.training.evaluation_interval,
                self.training.early_stopping_patience,
            )
            < 1
        ):
            raise ValueError("Qwen training counts must be positive")
        if not 0.0 < self.training.skill_a_replay_fraction < 1.0:
            raise ValueError("skill_a_replay_fraction must be in (0, 1)")
        effective_batch_size = (
            self.training.micro_batch_size * self.training.gradient_accumulation_steps
        )
        replay_examples = round(effective_batch_size * self.training.skill_a_replay_fraction)
        if not 1 <= replay_examples < effective_batch_size:
            raise ValueError("skill_a_replay_fraction must select both A and B examples")
        if self.training.learning_rate <= 0.0 or self.training.weight_decay < 0.0:
            raise ValueError("learning rate must be positive and weight decay non-negative")
        if not 0 <= self.training.warmup_steps <= self.training.steps:
            raise ValueError("warmup_steps must be between zero and training.steps")
        if self.training.maximum_gradient_norm <= 0.0:
            raise ValueError("maximum_gradient_norm must be positive")
        if self.training.early_stopping_min_delta < 0.0:
            raise ValueError("early_stopping_min_delta must be non-negative")
        self.evaluation.validate()

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible configuration."""
        value = asdict(self)
        value["output"] = self.output.as_posix()
        return cast(dict[str, object], value)


def parse_qwen_countdown_composition_config(
    raw: object,
) -> QwenCountdownCompositionConfig:
    """Parse a strict Qwen composition config."""
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
        required={
            "pretrained_model",
            "revision",
            "lora_rank",
            "lora_alpha",
            "lora_dropout",
            "target_modules",
        },
    )
    training = as_mapping(root["training"], field="training")
    training_fields = {
        "skill_a_pretraining_max_steps",
        "skill_a_pretraining_evaluation_interval",
        "skill_a_replay_fraction",
        "steps",
        "micro_batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "weight_decay",
        "warmup_steps",
        "maximum_gradient_norm",
        "log_interval",
        "evaluation_interval",
        "early_stopping_patience",
        "early_stopping_min_delta",
    }
    require_exact_keys(training, field="training", required=training_fields)
    config = QwenCountdownCompositionConfig(
        experiment_name=as_str(root["experiment_name"], field="experiment_name"),
        description=as_str(root["description"], field="description"),
        tags=as_str_tuple(root["tags"], field="tags"),
        seed=as_int(root["seed"], field="seed"),
        device=as_str(root["device"], field="device"),
        output=Path(as_str(root["output"], field="output")),
        task=parse_task_config(root["task"]),
        model=QwenModelConfig(
            pretrained_model=as_str(model["pretrained_model"], field="model.pretrained_model"),
            revision=as_str(model["revision"], field="model.revision"),
            lora_rank=as_int(model["lora_rank"], field="model.lora_rank"),
            lora_alpha=as_int(model["lora_alpha"], field="model.lora_alpha"),
            lora_dropout=as_float(model["lora_dropout"], field="model.lora_dropout"),
            target_modules=as_str_tuple(model["target_modules"], field="model.target_modules"),
        ),
        training=QwenTrainingConfig(
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
            micro_batch_size=as_int(
                training["micro_batch_size"], field="training.micro_batch_size"
            ),
            gradient_accumulation_steps=as_int(
                training["gradient_accumulation_steps"],
                field="training.gradient_accumulation_steps",
            ),
            learning_rate=as_float(training["learning_rate"], field="training.learning_rate"),
            weight_decay=as_float(training["weight_decay"], field="training.weight_decay"),
            warmup_steps=as_int(training["warmup_steps"], field="training.warmup_steps"),
            maximum_gradient_norm=as_float(
                training["maximum_gradient_norm"],
                field="training.maximum_gradient_norm",
            ),
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


def load_qwen_countdown_composition_config(
    path: Path,
) -> QwenCountdownCompositionConfig:
    """Load a Qwen composition YAML config."""
    return parse_qwen_countdown_composition_config(load_yaml(path))


def build_qwen_countdown_composition_dataset(
    config: QwenCountdownCompositionConfig,
) -> CountdownCompositionSplit:
    """Build the identical split used by the scratch composition experiment."""
    return build_composition_dataset(config)


def describe_qwen_countdown_composition(
    config: QwenCountdownCompositionConfig,
) -> dict[str, object]:
    """Describe workload without importing or downloading Qwen."""
    config.validate()
    return {
        "experiment_name": config.experiment_name,
        "pretrained_model": config.model.pretrained_model,
        "revision": config.model.revision,
        "fine_tuning": "LoRA supervised fine-tuning on separate A and B",
        "training_skills": ["A_symbol_decoding", "B_numeric_countdown"],
        "unseen_test": "A+B_symbolic_countdown",
        "numeric_train_examples": config.task.countdown_train_examples,
        "numeric_validation_examples": config.task.countdown_validation_examples,
        "paired_numeric_and_symbolic_test_examples": config.task.countdown_test_examples,
        "skill_a_pretraining_max_steps": config.training.skill_a_pretraining_max_steps,
        "skill_a_replay_fraction": config.training.skill_a_replay_fraction,
        "optimizer_steps": config.training.steps,
        "skill_b_phase_optimizer_steps": config.training.steps,
        "maximum_total_optimizer_steps": (
            config.training.skill_a_pretraining_max_steps + config.training.steps
        ),
        "effective_batch_size": (
            config.training.micro_batch_size * config.training.gradient_accumulation_steps
        ),
        "temperatures": list(config.evaluation.temperatures),
        "pass_k": list(config.evaluation.pass_k),
        "maximum_generated_proposals": proposal_count(config),
        "downloads_models": True,
        "downloads_datasets": False,
        "launches_cloud_job": False,
        "tensorboard_enabled": True,
    }


def run_qwen_countdown_composition_from_path(
    config_path: Path, *, output_path: Path | None = None, device: str | None = None
) -> dict[str, object]:
    """Fine-tune separate A/B skills and evaluate unseen symbolic A+B."""
    resolved_config_path = config_path.resolve()
    config = load_qwen_countdown_composition_config(resolved_config_path)
    try:
        from verifier_bottleneck.training.qwen_countdown_composition import (
            train_qwen_countdown_composition,
        )
    except ModuleNotFoundError as error:
        if error.name in {"torch", "tensorboard", "transformers", "peft"}:
            raise RuntimeError(
                "PyTorch, TensorBoard, Transformers, and PEFT are required"
            ) from error
        raise

    def trainer(run_directory: Path, selected_device: str) -> dict[str, object]:
        return train_qwen_countdown_composition(
            config,
            device=selected_device,
            tensorboard_directory=run_directory / "tensorboard",
            checkpoint_path=run_directory / "best-qwen-adapter.pt",
            proposal_outcomes_path=run_directory / "proposal-outcomes.jsonl",
            codebook_path=run_directory / "symbol-codebook.json",
            token_map_path=run_directory / "qwen-token-map.json",
        )

    return run_countdown_experiment(
        config=config,
        config_path=resolved_config_path,
        experiment_type="qwen_countdown_symbolic_composition",
        trainer=trainer,
        artifacts=tuple(
            ArtifactSpec(name, kind, f"Qwen composition artifact: {name}.")
            for name, kind in (
                ("best-qwen-adapter.pt", "checkpoint"),
                ("proposal-outcomes.jsonl", "evaluation"),
                ("symbol-codebook.json", "metadata"),
                ("qwen-token-map.json", "metadata"),
            )
        ),
        tensorboard_description="TensorBoard events for Qwen composition training.",
        output_path=output_path,
        device=device,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Qwen isolated A/B training and unseen Countdown A+B evaluation."
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
        config = load_qwen_countdown_composition_config(args.config.resolve())
        if args.dry_run:
            print(
                json.dumps(
                    describe_qwen_countdown_composition(config),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        result = run_qwen_countdown_composition_from_path(
            args.config, output_path=args.output, device=args.device
        )
    except (RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
