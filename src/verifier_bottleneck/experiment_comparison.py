"""Create paper-analysis tables from saved experiment summaries."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast


def discover_records(root: Path) -> list[dict[str, object]]:
    """Load all completed and failed experiment records under a directory."""
    records: list[dict[str, object]] = []
    for record_path in sorted(root.resolve().glob("**/record.json")):
        raw = json.loads(record_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or "schema_version" not in raw:
            continue
        record = cast(dict[str, object], raw)
        record["_record_path"] = str(record_path)
        records.append(record)
    return records


def _mapping(value: object) -> Mapping[str, object]:
    return cast(Mapping[str, object], value) if isinstance(value, Mapping) else {}


def build_run_rows(records: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Flatten one comparison row per run."""
    rows: list[dict[str, object]] = []
    for record in records:
        run = _mapping(record.get("run"))
        reproducibility = _mapping(record.get("reproducibility"))
        environment = _mapping(record.get("environment"))
        system = _mapping(environment.get("system"))
        dataset = _mapping(record.get("dataset"))
        model = _mapping(record.get("model"))
        results = _mapping(record.get("results"))
        final_metrics = _mapping(results.get("final_metrics"))
        best_metrics = _mapping(results.get("best_metrics"))
        rows.append(
            {
                "run_id": run.get("run_id"),
                "experiment_name": run.get("experiment_name"),
                "experiment_type": run.get("experiment_type"),
                "status": run.get("status"),
                "tags": ",".join(cast(list[str], run.get("tags", []))),
                "started_at_utc": run.get("started_at_utc"),
                "duration_seconds": run.get("duration_seconds"),
                "seed": reproducibility.get("primary_seed"),
                "config_sha256": reproducibility.get("config_sha256"),
                "source_sha256": reproducibility.get("source_sha256"),
                "git_commit": reproducibility.get("git_commit"),
                "device": environment.get("device"),
                "gpu_names": ",".join(cast(list[str], system.get("gpu_names", []))),
                "pytorch_version": system.get("pytorch_version"),
                "parameter_count": model.get("parameter_count"),
                "dataset_name": dataset.get("name"),
                "dataset_fingerprint": dataset.get("fingerprint"),
                "train_examples": dataset.get("train_examples"),
                "test_examples": dataset.get("test_examples"),
                "final_step": final_metrics.get("step"),
                "final_train_loss": final_metrics.get("train_loss"),
                "final_train_accuracy": final_metrics.get("train_accuracy"),
                "final_test_loss": final_metrics.get("test_loss"),
                "final_test_accuracy": final_metrics.get("test_accuracy"),
                "best_test_accuracy": best_metrics.get("test_accuracy"),
                "peak_gpu_memory_bytes": environment.get("peak_gpu_memory_bytes"),
                "record_path": record.get("_record_path"),
            }
        )
    return rows


def build_verifier_rows(
    records: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Flatten one comparison row per run and final verifier operating point."""
    rows: list[dict[str, object]] = []
    for record in records:
        run = _mapping(record.get("run"))
        results = _mapping(record.get("results"))
        final_metrics = _mapping(results.get("final_metrics"))
        measurements = final_metrics.get("verifier_measurements", [])
        if not isinstance(measurements, list):
            continue
        for measurement_raw in measurements:
            measurement = _mapping(measurement_raw)
            rows.append(
                {
                    "run_id": run.get("run_id"),
                    "experiment_name": run.get("experiment_name"),
                    "step": final_metrics.get("step"),
                    "test_accuracy": final_metrics.get("test_accuracy"),
                    "alpha": measurement.get("alpha"),
                    "beta": measurement.get("beta"),
                    "base_accuracy": measurement.get("base_accuracy"),
                    "predicted_accepted_accuracy": measurement.get(
                        "predicted_accepted_accuracy"
                    ),
                    "empirical_accepted_accuracy": measurement.get(
                        "empirical_accepted_accuracy"
                    ),
                    "predicted_acceptance_rate": measurement.get(
                        "predicted_acceptance_rate"
                    ),
                    "empirical_acceptance_rate": measurement.get(
                        "empirical_acceptance_rate"
                    ),
                    "accepted_count": measurement.get("accepted_count"),
                    "proposal_count": measurement.get("proposal_count"),
                    "record_path": record.get("_record_path"),
                }
            )
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_comparison(root: Path, output_directory: Path) -> dict[str, object]:
    """Write JSON and CSV tables suitable for analysis and paper figures."""
    records = discover_records(root)
    run_rows = build_run_rows(records)
    verifier_rows = build_verifier_rows(records)
    output_directory.mkdir(parents=True, exist_ok=True)
    runs_path = output_directory / "runs.csv"
    verifier_path = output_directory / "verifier-measurements.csv"
    comparison_path = output_directory / "comparison.json"
    _write_csv(runs_path, run_rows)
    _write_csv(verifier_path, verifier_rows)
    comparison = {
        "run_count": len(run_rows),
        "verifier_measurement_count": len(verifier_rows),
        "runs": run_rows,
        "verifier_measurements": verifier_rows,
    }
    comparison_path.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "run_count": len(run_rows),
        "verifier_measurement_count": len(verifier_rows),
        "runs_csv": str(runs_path),
        "verifier_measurements_csv": str(verifier_path),
        "comparison_json": str(comparison_path),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="verifier-bottleneck-compare",
        description="Build comparison-ready JSON and CSV tables from experiment records.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs"),
        help="Root directory containing experiment run directories.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("outputs") / "comparison",
        help="Directory for comparison.json and CSV tables.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for comparison exports."""
    args = _build_parser().parse_args(argv)
    if not args.root.is_dir():
        print(f"error: experiment root does not exist: {args.root}", file=sys.stderr)
        return 2
    report = export_comparison(args.root, args.output_directory)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
