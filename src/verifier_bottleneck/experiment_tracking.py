"""Shared experiment provenance, result persistence, and artifact registration."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import shutil
import sys
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from verifier_bottleneck.cli import collect_system_report

SCHEMA_VERSION = "1.0"
SOURCE_PATTERNS = (
    "src/**/*.py",
    # DataSphere stages env.python.local-paths without the repository's src/
    # parent, so hash the deployed package shape as well.
    "verifier_bottleneck/**/*.py",
    "configs/**/*.yaml",
    "configs/**/*.yml",
    "jobs/**/*.yaml",
    "jobs/**/*.yml",
    "scripts/**/*.py",
    "notebooks/**/*.ipynb",
    "tests/**/*.py",
    "docs/**/*.md",
    "pyproject.toml",
    "requirements*.txt",
    "README.md",
)

Clock = Callable[[], datetime]


def utc_now() -> datetime:
    """Return the current timezone-aware UTC time."""
    return datetime.now(timezone.utc)


def canonical_json(value: object) -> str:
    """Serialize a value deterministically for hashing and storage."""
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def sha256_bytes(value: bytes) -> str:
    """Return the SHA-256 digest of bytes."""
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: object) -> str:
    """Return a stable digest for a JSON-serializable object."""
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def sha256_file(path: Path) -> str:
    """Hash a file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_git_commit(repository_root: Path) -> str | None:
    """Read the current commit without invoking Git or a subprocess."""
    git_directory = repository_root / ".git"
    head_path = git_directory / "HEAD"
    if not head_path.is_file():
        return None
    head = head_path.read_text(encoding="utf-8").strip()
    if not head.startswith("ref: "):
        return head or None

    reference = head.removeprefix("ref: ")
    reference_path = git_directory / reference
    if reference_path.is_file():
        return reference_path.read_text(encoding="utf-8").strip() or None

    packed_refs = git_directory / "packed-refs"
    if packed_refs.is_file():
        for line in packed_refs.read_text(encoding="utf-8").splitlines():
            if line.startswith(("#", "^")):
                continue
            commit, _, name = line.partition(" ")
            if name == reference:
                return commit
    return None


def collect_source_manifest(repository_root: Path) -> dict[str, object]:
    """Hash reproducibility-relevant repository files."""
    relative_paths: set[Path] = set()
    for pattern in SOURCE_PATTERNS:
        for path in repository_root.glob(pattern):
            if path.is_file():
                relative_paths.add(path.relative_to(repository_root))

    file_hashes = {
        relative_path.as_posix(): sha256_file(repository_root / relative_path)
        for relative_path in sorted(relative_paths)
    }
    return {
        "algorithm": "sha256",
        "source_sha256": sha256_json(file_hashes),
        "file_count": len(file_hashes),
        "files": file_hashes,
    }


def collect_package_versions() -> dict[str, str]:
    """Return installed Python distribution versions without network access."""
    packages: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        try:
            name = distribution.metadata["Name"]
        except KeyError:
            continue
        packages[str(name).lower()] = distribution.version
    return dict(sorted(packages.items()))


def collect_execution_context() -> dict[str, object]:
    """Collect a small allowlist of execution context without recording secrets."""
    context: dict[str, object] = {
        "python_executable": sys.executable,
        "process_id": os.getpid(),
    }
    ds_project_home = os.environ.get("DS_PROJECT_HOME")
    if ds_project_home:
        context["platform"] = "yandex_datasphere"
        context["ds_project_home"] = ds_project_home
    elif os.environ.get("COLAB_RELEASE_TAG"):
        context["platform"] = "google_colab"
        context["colab_release_tag"] = os.environ["COLAB_RELEASE_TAG"]
    else:
        context["platform"] = "local_or_other"
    return context


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _write_json_lines(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    rendered = "".join(canonical_json(dict(row)) + "\n" for row in rows)
    path.write_text(rendered, encoding="utf-8")


def _unique_run_directory(output_root: Path, proposed_run_id: str) -> tuple[str, Path]:
    run_id = proposed_run_id
    run_directory = output_root / run_id
    suffix = 2
    while run_directory.exists():
        run_id = f"{proposed_run_id}-{suffix:02d}"
        run_directory = output_root / run_id
        suffix += 1
    run_directory.mkdir(parents=True)
    return run_id, run_directory


def _artifact_entry(
    *,
    run_directory: Path,
    artifact_path: Path,
    kind: str,
    description: str,
) -> dict[str, object]:
    resolved_artifact = artifact_path.resolve()
    resolved_run_directory = run_directory.resolve()
    try:
        relative_path = resolved_artifact.relative_to(resolved_run_directory)
    except ValueError as error:
        raise ValueError(
            f"artifact must be inside the run directory: {resolved_artifact}"
        ) from error
    if not resolved_artifact.is_file():
        raise ValueError(f"artifact does not exist or is not a file: {resolved_artifact}")
    return {
        "path": relative_path.as_posix(),
        "kind": kind,
        "description": description,
        "size_bytes": resolved_artifact.stat().st_size,
        "sha256": sha256_file(resolved_artifact),
    }


@dataclass
class ExperimentRecorder:
    """Lifecycle manager for one experiment run."""

    repository_root: Path
    output_root: Path
    run_directory: Path
    record_path: Path
    summary_path: Path
    metrics_path: Path
    source_manifest_path: Path
    config_snapshot_path: Path
    record: dict[str, object]
    started_monotonic: float
    clock: Clock

    @classmethod
    def start(
        cls,
        *,
        repository_root: Path,
        output_root: Path,
        experiment_name: str,
        experiment_type: str,
        description: str,
        tags: Sequence[str],
        seed: int,
        config: Mapping[str, object],
        config_path: Path,
        torch_module: Any | None = None,
        clock: Clock = utc_now,
        monotonic: Callable[[], float],
    ) -> ExperimentRecorder:
        """Create a run directory and persist a running record immediately."""
        if not experiment_name.strip():
            raise ValueError("experiment_name must not be empty")
        started_monotonic = monotonic()
        started_at = clock()
        if started_at.tzinfo is None:
            raise ValueError("experiment clock must return a timezone-aware datetime")

        resolved_repository_root = repository_root.resolve()
        resolved_output_root = output_root.resolve()
        resolved_output_root.mkdir(parents=True, exist_ok=True)
        config_dict = dict(config)
        config_sha256 = sha256_json(config_dict)
        timestamp = started_at.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        proposed_run_id = f"{timestamp}_{config_sha256[:10]}"
        run_id, run_directory = _unique_run_directory(
            resolved_output_root, proposed_run_id
        )

        source_manifest = collect_source_manifest(resolved_repository_root)
        source_manifest_path = run_directory / "source-manifest.json"
        _write_json(source_manifest_path, source_manifest)

        config_snapshot_path = run_directory / "config.yaml"
        shutil.copyfile(config_path.resolve(), config_snapshot_path)

        record_path = run_directory / "record.json"
        summary_path = run_directory / "summary.json"
        metrics_path = run_directory / "metrics.jsonl"
        artifacts = [
            _artifact_entry(
                run_directory=run_directory,
                artifact_path=config_snapshot_path,
                kind="configuration",
                description="Exact YAML configuration used for this run.",
            ),
            _artifact_entry(
                run_directory=run_directory,
                artifact_path=source_manifest_path,
                kind="provenance",
                description="SHA-256 manifest of reproducibility-relevant repository files.",
            ),
        ]
        record: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "run": {
                "run_id": run_id,
                "experiment_name": experiment_name,
                "experiment_type": experiment_type,
                "description": description,
                "tags": list(tags),
                "status": "running",
                "started_at_utc": started_at.astimezone(timezone.utc).isoformat(),
                "completed_at_utc": None,
                "duration_seconds": None,
            },
            "reproducibility": {
                "primary_seed": seed,
                "config_sha256": config_sha256,
                "source_sha256": source_manifest["source_sha256"],
                "git_commit": read_git_commit(resolved_repository_root),
                "deterministic_algorithms_requested": True,
            },
            "config": config_dict,
            "environment": {
                "system": collect_system_report(torch_module),
                "execution_context": collect_execution_context(),
                "python_packages": collect_package_versions(),
            },
            "dataset": {},
            "model": {},
            "optimization": {},
            "definitions": {},
            "metrics": {"trajectory": []},
            "results": {},
            "artifacts": artifacts,
            "error": None,
        }
        _write_json(record_path, record)
        return cls(
            repository_root=resolved_repository_root,
            output_root=resolved_output_root,
            run_directory=run_directory,
            record_path=record_path,
            summary_path=summary_path,
            metrics_path=metrics_path,
            source_manifest_path=source_manifest_path,
            config_snapshot_path=config_snapshot_path,
            record=record,
            started_monotonic=started_monotonic,
            clock=clock,
        )

    @property
    def run_id(self) -> str:
        """Return this run's identifier."""
        run = cast(dict[str, object], self.record["run"])
        return cast(str, run["run_id"])

    def complete(
        self,
        *,
        dataset: Mapping[str, object],
        model: Mapping[str, object],
        optimization: Mapping[str, object],
        definitions: Mapping[str, object],
        trajectory: Sequence[Mapping[str, object]],
        results: Mapping[str, object],
        environment_updates: Mapping[str, object],
        monotonic: Callable[[], float],
    ) -> dict[str, object]:
        """Finalize a successful run and write full and compact outputs."""
        completed_at = self.clock().astimezone(timezone.utc)
        duration_seconds = monotonic() - self.started_monotonic
        run = cast(dict[str, object], self.record["run"])
        run.update(
            {
                "status": "completed",
                "completed_at_utc": completed_at.isoformat(),
                "duration_seconds": duration_seconds,
            }
        )
        environment = cast(dict[str, object], self.record["environment"])
        environment.update(environment_updates)
        self.record["dataset"] = dict(dataset)
        self.record["model"] = dict(model)
        self.record["optimization"] = dict(optimization)
        self.record["definitions"] = dict(definitions)
        self.record["metrics"] = {"trajectory": [dict(row) for row in trajectory]}
        self.record["results"] = dict(results)

        _write_json_lines(self.metrics_path, trajectory)
        artifacts = cast(list[dict[str, object]], self.record["artifacts"])
        artifacts.append(
            _artifact_entry(
                run_directory=self.run_directory,
                artifact_path=self.metrics_path,
                kind="metrics",
                description="One JSON object per evaluation checkpoint.",
            )
        )
        summary = self._build_summary()
        _write_json(self.summary_path, summary)
        artifacts.append(
            _artifact_entry(
                run_directory=self.run_directory,
                artifact_path=self.summary_path,
                kind="summary",
                description="Compact comparison-ready run summary.",
            )
        )
        _write_json(self.record_path, self.record)
        return self.paths()

    def fail(
        self,
        error: BaseException,
        *,
        monotonic: Callable[[], float],
    ) -> None:
        """Persist a failed run before the exception is re-raised."""
        completed_at = self.clock().astimezone(timezone.utc)
        run = cast(dict[str, object], self.record["run"])
        run.update(
            {
                "status": "failed",
                "completed_at_utc": completed_at.isoformat(),
                "duration_seconds": monotonic() - self.started_monotonic,
            }
        )
        self.record["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": "".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            ),
        }
        _write_json(self.summary_path, self._build_summary())
        _write_json(self.record_path, self.record)

    def _build_summary(self) -> dict[str, object]:
        run = cast(dict[str, object], self.record["run"])
        reproducibility = cast(dict[str, object], self.record["reproducibility"])
        environment = cast(dict[str, object], self.record["environment"])
        system = cast(dict[str, object], environment["system"])
        model = cast(dict[str, object], self.record["model"])
        dataset = cast(dict[str, object], self.record["dataset"])
        results = cast(dict[str, object], self.record["results"])
        return {
            "schema_version": self.record["schema_version"],
            "run_id": run["run_id"],
            "experiment_name": run["experiment_name"],
            "experiment_type": run["experiment_type"],
            "description": run["description"],
            "tags": run["tags"],
            "status": run["status"],
            "started_at_utc": run["started_at_utc"],
            "completed_at_utc": run["completed_at_utc"],
            "duration_seconds": run["duration_seconds"],
            "seed": reproducibility["primary_seed"],
            "config_sha256": reproducibility["config_sha256"],
            "source_sha256": reproducibility["source_sha256"],
            "git_commit": reproducibility["git_commit"],
            "device": environment.get("device"),
            "gpu_names": system.get("gpu_names", []),
            "pytorch_version": system.get("pytorch_version"),
            "cuda_runtime_version": system.get("cuda_runtime_version"),
            "parameter_count": model.get("parameter_count"),
            "trainable_parameter_count": model.get("trainable_parameter_count"),
            "dataset_name": dataset.get("name"),
            "dataset_fingerprint": dataset.get("fingerprint"),
            "train_examples": dataset.get("train_examples"),
            "test_examples": dataset.get("test_examples"),
            "peak_gpu_memory_bytes": environment.get("peak_gpu_memory_bytes"),
            "final_metrics": results.get("final_metrics", {}),
            "best_metrics": results.get("best_metrics", {}),
            "record_path": "record.json",
            "error": self.record["error"],
        }

    def paths(self) -> dict[str, object]:
        """Return absolute output paths for notebook and CLI callers."""
        return {
            "run_id": self.run_id,
            "run_directory": str(self.run_directory),
            "record_path": str(self.record_path),
            "summary_path": str(self.summary_path),
            "metrics_path": str(self.metrics_path),
            "output_path": str(self.record_path),
        }


def register_artifact(
    record_path: Path,
    artifact_path: Path,
    *,
    kind: str,
    description: str,
) -> dict[str, object]:
    """Register an artifact created after experiment completion, such as a plot."""
    resolved_record_path = record_path.resolve()
    record = cast(
        dict[str, object],
        json.loads(resolved_record_path.read_text(encoding="utf-8")),
    )
    run_directory = resolved_record_path.parent
    entry = _artifact_entry(
        run_directory=run_directory,
        artifact_path=artifact_path,
        kind=kind,
        description=description,
    )
    artifacts = cast(list[dict[str, object]], record["artifacts"])
    existing_paths = {cast(str, artifact["path"]) for artifact in artifacts}
    if entry["path"] in existing_paths:
        artifacts[:] = [
            artifact for artifact in artifacts if artifact["path"] != entry["path"]
        ]
    artifacts.append(entry)
    _write_json(resolved_record_path, record)
    return entry
