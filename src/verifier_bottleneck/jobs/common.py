"""Reusable DataSphere Job command-line lifecycle."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path

from verifier_bottleneck.jobs.modular_addition import export_run, find_latest_run_directory
from verifier_bottleneck.live_logging import live_log

RunFromPath = Callable[..., dict[str, object]]


def build_experiment_job_parser(
    *, description: str, default_run_root: Path
) -> argparse.ArgumentParser:
    """Create the standard parser used by experiment-export Jobs."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--record", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--run-root", type=Path, default=default_run_root)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    return parser


def run_export_job(
    argv: Sequence[str] | None,
    *,
    description: str,
    default_run_root: Path,
    log_name: str,
    run_from_path: RunFromPath,
) -> int:
    """Run an experiment and always export the latest canonical run."""
    args = build_experiment_job_parser(
        description=description, default_run_root=default_run_root
    ).parse_args(argv)
    live_log(f"{log_name} started config={args.config}")
    try:
        result = run_from_path(args.config, output_path=args.run_root, device=args.device)
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
    live_log(f"Run archive: {args.archive}")
    return 0
