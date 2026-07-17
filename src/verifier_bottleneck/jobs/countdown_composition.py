"""DataSphere entry point for the scratch Countdown composition experiment."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from verifier_bottleneck.experiments.countdown_composition import (
    run_countdown_composition_from_path,
)
from verifier_bottleneck.jobs.common import run_export_job

DEFAULT_JOB_RUN_ROOT = Path(".job-runs") / "countdown-symbolic-composition-scratch"


def main(argv: Sequence[str] | None = None) -> int:
    """Run once and materialize all declared DataSphere outputs."""
    return run_export_job(
        argv,
        description="Run scratch Countdown A/B-to-A+B composition and export it.",
        default_run_root=DEFAULT_JOB_RUN_ROOT,
        log_name="Countdown composition Job",
        run_from_path=run_countdown_composition_from_path,
    )


if __name__ == "__main__":
    raise SystemExit(main())
