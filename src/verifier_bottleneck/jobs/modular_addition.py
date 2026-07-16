"""Run modular addition and export its complete record as Job output files."""

from __future__ import annotations

import argparse
import shutil
import zipfile
from collections.abc import Sequence
from pathlib import Path

from verifier_bottleneck.experiments.modular_addition import run_from_path

DEFAULT_JOB_RUN_ROOT = Path(".job-runs") / "modular-addition"


def find_latest_run_directory(output_root: Path) -> Path:
    """Return the newest run directory containing a canonical record."""
    record_paths = list(output_root.glob("*/record.json"))
    if not record_paths:
        raise RuntimeError(f"no experiment record was created under {output_root}")
    latest_record = max(
        record_paths,
        key=lambda path: (path.stat().st_mtime_ns, path.parent.name),
    )
    return latest_record.parent


def export_run(
    run_directory: Path,
    *,
    archive_path: Path,
    record_path: Path,
    summary_path: Path,
) -> None:
    """Copy paper-facing JSON files and archive the complete run directory."""
    source_record = run_directory / "record.json"
    source_summary = run_directory / "summary.json"
    for source, destination in (
        (source_record, record_path),
        (source_summary, summary_path),
    ):
        if not source.is_file():
            raise RuntimeError(f"expected experiment output does not exist: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        archive_path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for source in sorted(run_directory.rglob("*")):
            if source.is_file():
                archive.write(source, source.relative_to(run_directory).as_posix())


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the modular-addition sandbox and export its complete experiment "
            "record for DataSphere Jobs."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--record", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=DEFAULT_JOB_RUN_ROOT,
        help="Internal directory for the versioned experiment run.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="cpu",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the experiment and materialize all declared Job output files."""
    args = _build_parser().parse_args(argv)
    try:
        result = run_from_path(
            args.config,
            output_path=args.run_root,
            device=args.device,
        )
        run_directory = Path(str(result["run_directory"]))
    except BaseException:
        if args.run_root.is_dir():
            run_directory = find_latest_run_directory(args.run_root)
            export_run(
                run_directory,
                archive_path=args.archive,
                record_path=args.record,
                summary_path=args.summary,
            )
        raise

    export_run(
        run_directory,
        archive_path=args.archive,
        record_path=args.record,
        summary_path=args.summary,
    )
    print(f"Run archive: {args.archive}")
    print(f"Canonical record: {args.record}")
    print(f"Comparison summary: {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
