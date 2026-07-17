"""Shared experiment lifecycle for Countdown model backends."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from verifier_bottleneck.analysis.countdown_artifacts import (
    write_temperature_sweep_artifacts,
)
from verifier_bottleneck.experiment_tracking import ExperimentRecorder, register_artifact
from verifier_bottleneck.jupyter_setup import find_repository_root


class RuntimeConfig(Protocol):
    """Configuration fields needed by the shared run lifecycle."""

    @property
    def experiment_name(self) -> str: ...

    @property
    def description(self) -> str: ...

    @property
    def tags(self) -> tuple[str, ...]: ...

    @property
    def seed(self) -> int: ...

    @property
    def device(self) -> str: ...

    @property
    def output(self) -> Path: ...

    def to_dict(self) -> dict[str, object]: ...


Trainer = Callable[[Path, str], dict[str, object]]


@dataclass(frozen=True)
class ArtifactSpec:
    """One file produced by a model-specific trainer."""

    relative_path: str
    kind: str
    description: str


def run_countdown_experiment(
    *,
    config: RuntimeConfig,
    config_path: Path,
    experiment_type: str,
    trainer: Trainer,
    artifacts: tuple[ArtifactSpec, ...],
    tensorboard_description: str,
    output_path: Path | None,
    device: str | None,
) -> dict[str, object]:
    """Record, execute, finalize, and register one composition experiment."""
    resolved_config_path = config_path.resolve()
    try:
        repository_root = find_repository_root(resolved_config_path.parent)
    except ValueError:
        repository_root = find_repository_root(Path.cwd())
    selected_output = output_path if output_path is not None else config.output
    if not selected_output.is_absolute():
        selected_output = repository_root / selected_output

    recorder = ExperimentRecorder.start(
        repository_root=repository_root,
        output_root=selected_output,
        experiment_name=config.experiment_name,
        experiment_type=experiment_type,
        description=config.description,
        tags=config.tags,
        seed=config.seed,
        config=config.to_dict(),
        config_path=resolved_config_path,
        monotonic=time.perf_counter,
    )
    try:
        payload = trainer(recorder.run_directory, device or config.device)
        paths = recorder.complete(
            dataset=cast(dict[str, object], payload["dataset"]),
            model=cast(dict[str, object], payload["model"]),
            optimization=cast(dict[str, object], payload["optimization"]),
            definitions=cast(dict[str, object], payload["definitions"]),
            trajectory=cast(list[dict[str, object]], payload["trajectory"]),
            results=cast(dict[str, object], payload["results"]),
            environment_updates=cast(dict[str, object], payload["environment_updates"]),
            monotonic=time.perf_counter,
        )
        record_path = Path(str(paths["record_path"]))
        results = cast(dict[str, object], payload["results"])
        measurements = cast(list[dict[str, object]], results["temperature_measurements"])
        write_temperature_sweep_artifacts(record_path, measurements)
        for tensorboard_path in sorted((recorder.run_directory / "tensorboard").glob("*")):
            if tensorboard_path.is_file():
                register_artifact(
                    record_path,
                    tensorboard_path,
                    kind="telemetry",
                    description=tensorboard_description,
                )
        for artifact in artifacts:
            register_artifact(
                record_path,
                recorder.run_directory / artifact.relative_path,
                kind=artifact.kind,
                description=artifact.description,
            )
    except BaseException as error:
        recorder.fail(error, monotonic=time.perf_counter)
        raise
    return {**recorder.record, **paths}
