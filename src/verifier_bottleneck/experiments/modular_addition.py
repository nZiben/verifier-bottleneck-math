"""Configuration and entry point for the modular-addition sandbox experiment."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from verifier_bottleneck.data.modular_addition import make_modular_addition_split
from verifier_bottleneck.experiment_tracking import ExperimentRecorder
from verifier_bottleneck.jupyter_setup import find_repository_root
from verifier_bottleneck.verifiers.noisy import VerifierParameters

SEQUENCE_LENGTH = 4


@dataclass(frozen=True)
class TaskConfig:
    """Synthetic modular-addition dataset settings."""

    modulus: int
    train_fraction: float


@dataclass(frozen=True)
class ModelConfig:
    """Tiny causal-transformer settings."""

    layers: int
    d_model: int
    heads: int
    d_ff: int
    dropout: float


@dataclass(frozen=True)
class TrainingConfig:
    """Optimizer and evaluation settings."""

    steps: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    evaluation_interval: int


@dataclass(frozen=True)
class VerifierConfig:
    """Monte Carlo settings and verifier operating points."""

    repetitions: int
    seed: int
    operating_points: tuple[VerifierParameters, ...]


@dataclass(frozen=True)
class ExperimentConfig:
    """Complete modular-addition experiment configuration."""

    experiment_name: str
    description: str
    tags: tuple[str, ...]
    seed: int
    device: str
    output: Path
    task: TaskConfig
    model: ModelConfig
    training: TrainingConfig
    verifier: VerifierConfig

    def validate(self) -> None:
        """Reject configurations that are invalid or unexpectedly expensive."""
        if not self.experiment_name.strip():
            raise ValueError("experiment_name must not be empty")
        if not self.description.strip():
            raise ValueError("description must not be empty")
        if not self.tags or any(not tag.strip() for tag in self.tags):
            raise ValueError("tags must contain at least one non-empty tag")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be one of: auto, cpu, cuda")
        if self.model.layers not in {1, 2}:
            raise ValueError("model.layers must be 1 or 2 for this sandbox")
        if self.model.d_model < 4:
            raise ValueError("model.d_model must be at least 4")
        if self.model.heads < 1 or self.model.d_model % self.model.heads != 0:
            raise ValueError("model.heads must divide model.d_model exactly")
        if self.model.d_ff < self.model.d_model:
            raise ValueError("model.d_ff must be at least model.d_model")
        if not 0.0 <= self.model.dropout < 1.0:
            raise ValueError("model.dropout must be in [0, 1)")
        if self.training.steps < 1:
            raise ValueError("training.steps must be at least 1")
        if self.training.batch_size < 1:
            raise ValueError("training.batch_size must be at least 1")
        if self.training.learning_rate <= 0.0:
            raise ValueError("training.learning_rate must be positive")
        if self.training.weight_decay < 0.0:
            raise ValueError("training.weight_decay must be non-negative")
        if not 0 <= self.training.warmup_steps <= self.training.steps:
            raise ValueError("training.warmup_steps must be between 0 and training.steps")
        if self.training.evaluation_interval < 1:
            raise ValueError("training.evaluation_interval must be at least 1")
        if self.verifier.repetitions < 1:
            raise ValueError("verifier.repetitions must be at least 1")
        if not self.verifier.operating_points:
            raise ValueError("verifier.operating_points must not be empty")
        make_modular_addition_split(
            modulus=self.task.modulus,
            train_fraction=self.task.train_fraction,
            seed=self.seed,
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        result = asdict(self)
        result["output"] = self.output.as_posix()
        return cast(dict[str, object], result)


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return cast(Mapping[str, object], value)


def _check_keys(
    value: Mapping[str, object],
    *,
    field: str,
    required: set[str],
) -> None:
    missing = required - value.keys()
    unknown = value.keys() - required
    if missing:
        raise ValueError(f"{field} is missing required fields: {sorted(missing)}")
    if unknown:
        raise ValueError(f"{field} has unsupported fields: {sorted(unknown)}")


def _integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be a number")
    return float(value)


def _string(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return value


def parse_experiment_config(raw: object) -> ExperimentConfig:
    """Parse a strict mapping into an experiment configuration."""
    root = _mapping(raw, field="config")
    root_fields = {
        "experiment_name",
        "description",
        "tags",
        "seed",
        "device",
        "output",
        "task",
        "model",
        "training",
        "verifier",
    }
    _check_keys(root, field="config", required=root_fields)
    tags_raw = root["tags"]
    if not isinstance(tags_raw, list) or not all(
        isinstance(tag, str) for tag in tags_raw
    ):
        raise ValueError("tags must be a list of strings")

    task = _mapping(root["task"], field="task")
    _check_keys(task, field="task", required={"modulus", "train_fraction"})

    model = _mapping(root["model"], field="model")
    _check_keys(
        model,
        field="model",
        required={"layers", "d_model", "heads", "d_ff", "dropout"},
    )

    training = _mapping(root["training"], field="training")
    _check_keys(
        training,
        field="training",
        required={
            "steps",
            "batch_size",
            "learning_rate",
            "weight_decay",
            "warmup_steps",
            "evaluation_interval",
        },
    )

    verifier = _mapping(root["verifier"], field="verifier")
    _check_keys(
        verifier,
        field="verifier",
        required={"repetitions", "seed", "operating_points"},
    )
    operating_points_raw = verifier["operating_points"]
    if not isinstance(operating_points_raw, list):
        raise ValueError("verifier.operating_points must be a list")
    operating_points: list[VerifierParameters] = []
    for index, raw_point in enumerate(operating_points_raw):
        point = _mapping(raw_point, field=f"verifier.operating_points[{index}]")
        _check_keys(
            point,
            field=f"verifier.operating_points[{index}]",
            required={"alpha", "beta"},
        )
        operating_points.append(
            VerifierParameters(
                alpha=_number(
                    point["alpha"],
                    field=f"verifier.operating_points[{index}].alpha",
                ),
                beta=_number(
                    point["beta"],
                    field=f"verifier.operating_points[{index}].beta",
                ),
            )
        )

    config = ExperimentConfig(
        experiment_name=_string(root["experiment_name"], field="experiment_name"),
        description=_string(root["description"], field="description"),
        tags=tuple(cast(list[str], tags_raw)),
        seed=_integer(root["seed"], field="seed"),
        device=_string(root["device"], field="device"),
        output=Path(_string(root["output"], field="output")),
        task=TaskConfig(
            modulus=_integer(task["modulus"], field="task.modulus"),
            train_fraction=_number(
                task["train_fraction"], field="task.train_fraction"
            ),
        ),
        model=ModelConfig(
            layers=_integer(model["layers"], field="model.layers"),
            d_model=_integer(model["d_model"], field="model.d_model"),
            heads=_integer(model["heads"], field="model.heads"),
            d_ff=_integer(model["d_ff"], field="model.d_ff"),
            dropout=_number(model["dropout"], field="model.dropout"),
        ),
        training=TrainingConfig(
            steps=_integer(training["steps"], field="training.steps"),
            batch_size=_integer(
                training["batch_size"], field="training.batch_size"
            ),
            learning_rate=_number(
                training["learning_rate"], field="training.learning_rate"
            ),
            weight_decay=_number(
                training["weight_decay"], field="training.weight_decay"
            ),
            warmup_steps=_integer(
                training["warmup_steps"], field="training.warmup_steps"
            ),
            evaluation_interval=_integer(
                training["evaluation_interval"],
                field="training.evaluation_interval",
            ),
        ),
        verifier=VerifierConfig(
            repetitions=_integer(
                verifier["repetitions"], field="verifier.repetitions"
            ),
            seed=_integer(verifier["seed"], field="verifier.seed"),
            operating_points=tuple(operating_points),
        ),
    )
    config.validate()
    return config


def load_experiment_config(path: Path) -> ExperimentConfig:
    """Load and validate a YAML experiment configuration."""
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as error:
        raise RuntimeError(
            "PyYAML is required to load experiment configs. "
            'Install the lightweight extra with: python -m pip install -e ".[sandbox]"'
        ) from error

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"config file does not exist: {path}") from error
    except OSError as error:
        raise ValueError(f"could not read config file {path}: {error}") from error
    except yaml.YAMLError as error:
        raise ValueError(f"invalid YAML in {path}: {error}") from error
    return parse_experiment_config(raw)


def estimate_parameter_count(config: ExperimentConfig) -> int:
    """Calculate the parameter count of the implemented PyTorch model."""
    d_model = config.model.d_model
    d_ff = config.model.d_ff
    modulus = config.task.modulus
    embeddings = (modulus + 2) * d_model + SEQUENCE_LENGTH * d_model
    attention = 4 * d_model * d_model + 4 * d_model
    feed_forward = 2 * d_model * d_ff + d_ff + d_model
    layer_norms = 4 * d_model
    encoder_layers = config.model.layers * (
        attention + feed_forward + layer_norms
    )
    final_norm = 2 * d_model
    output_head = modulus * d_model + modulus
    return embeddings + encoder_layers + final_norm + output_head


def describe_experiment(config: ExperimentConfig) -> dict[str, object]:
    """Describe dataset and model size without importing PyTorch or training."""
    split = make_modular_addition_split(
        modulus=config.task.modulus,
        train_fraction=config.task.train_fraction,
        seed=config.seed,
    )
    return {
        "experiment_name": config.experiment_name,
        "description": config.description,
        "tags": list(config.tags),
        "device": config.device,
        "output_root": config.output.as_posix(),
        "modulus": split.modulus,
        "table_size": split.modulus**2,
        "train_examples": len(split.train),
        "test_examples": len(split.test),
        "training_steps": config.training.steps,
        "estimated_parameter_count": estimate_parameter_count(config),
        "downloads_models": False,
        "launches_cloud_job": False,
    }


def run_from_path(
    config_path: Path,
    *,
    output_path: Path | None = None,
    device: str | None = None,
) -> dict[str, object]:
    """Run an experiment directly in the current Python process."""
    resolved_config_path = config_path.resolve()
    config = load_experiment_config(resolved_config_path)
    try:
        repository_root = find_repository_root(resolved_config_path.parent)
    except ValueError:
        repository_root = find_repository_root(Path.cwd())
    try:
        from verifier_bottleneck.training.tiny_transformer import (
            train_modular_addition,
        )
    except ModuleNotFoundError as error:
        if error.name == "torch":
            raise RuntimeError(
                "PyTorch is required to train this sandbox. Use a Colab or "
                "DataSphere runtime that already provides PyTorch; the sandbox "
                "extra intentionally does not download the large torch wheel."
            ) from error
        raise

    selected_device = device if device is not None else config.device
    if selected_device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device override must be one of: auto, cpu, cuda")
    selected_output = output_path if output_path is not None else config.output
    if not selected_output.is_absolute():
        selected_output = repository_root / selected_output
    recorder = ExperimentRecorder.start(
        repository_root=repository_root,
        output_root=selected_output,
        experiment_name=config.experiment_name,
        experiment_type="modular_addition",
        description=config.description,
        tags=config.tags,
        seed=config.seed,
        config=config.to_dict(),
        config_path=resolved_config_path,
        monotonic=time.perf_counter,
    )
    try:
        payload = train_modular_addition(config, device=selected_device)
        paths = recorder.complete(
            dataset=cast(dict[str, object], payload["dataset"]),
            model=cast(dict[str, object], payload["model"]),
            optimization=cast(dict[str, object], payload["optimization"]),
            definitions=cast(dict[str, object], payload["definitions"]),
            trajectory=cast(list[dict[str, object]], payload["trajectory"]),
            results=cast(dict[str, object], payload["results"]),
            environment_updates=cast(
                dict[str, object], payload["environment_updates"]
            ),
            monotonic=time.perf_counter,
        )
    except BaseException as error:
        recorder.fail(error, monotonic=time.perf_counter)
        raise
    return {**recorder.record, **paths}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="verifier-bottleneck-mod-add",
        description=(
            "Train a tiny transformer from scratch on modular addition and "
            "measure noisy-verifier filtered accuracy."
        ),
    )
    parser.add_argument("--config", required=True, type=Path, help="YAML config path.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Override the root directory where a unique run directory is created.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        help="Override the configured training device.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and describe the experiment without importing PyTorch.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for the modular-addition sandbox."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_experiment_config(args.config.resolve())
        if args.dry_run:
            print(json.dumps(describe_experiment(config), indent=2, sort_keys=True))
            return 0
        result = run_from_path(
            args.config,
            output_path=args.output,
            device=args.device,
        )
    except (RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    results = cast(dict[str, object], result["results"])
    final = cast(dict[str, object], results["final_metrics"])
    print(
        json.dumps(
            {
                "run_id": result["run_id"],
                "record": result["record_path"],
                "summary": result["summary_path"],
                "step": final["step"],
                "train_accuracy": final["train_accuracy"],
                "test_accuracy": final["test_accuracy"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
