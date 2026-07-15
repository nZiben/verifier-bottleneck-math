"""Explicitly submit one DataSphere job configuration."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

PROJECT_ID = "bt1k0f436ik4qgb96r7s"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Submit an explicitly selected config to Yandex DataSphere."
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to the DataSphere job YAML configuration.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually launch the paid cloud job; otherwise only print the command.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = args.config
    if not config.is_file():
        raise SystemExit(f"Job config does not exist: {config}")
    if config.suffix.lower() not in {".yaml", ".yml"}:
        raise SystemExit(f"Job config must be YAML: {config}")

    command = [
        "datasphere",
        "project",
        "job",
        "execute",
        "-p",
        PROJECT_ID,
        "-c",
        str(config),
    ]
    if not args.execute:
        print("Dry run; no DataSphere job was launched.")
        print("Command:", " ".join(command))
        print("Re-run with --execute only after budget approval.")
        return 0

    if shutil.which("datasphere") is None:
        raise SystemExit(
            "DataSphere CLI was not found on PATH. Install and authenticate it, "
            "then retry after budget approval."
        )

    completed = subprocess.run(command, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
