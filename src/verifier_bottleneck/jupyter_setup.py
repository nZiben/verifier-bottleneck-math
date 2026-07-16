"""Install and validate this repository inside a Jupyter kernel environment."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

PYTORCH_INDEX_URL = "https://download.pytorch.org/whl/cu124"
PYPI_INDEX_URL = "https://pypi.org/simple"
GPU_REQUIREMENTS = "requirements-rl-compositionality-gpu.txt"

CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


def validate_python_version(major: int, minor: int) -> None:
    """Require the project Python version used by DataSphere."""
    if (major, minor) != (3, 10):
        raise ValueError(
            f"Python 3.10 is required, but the current interpreter is {major}.{minor}."
        )


def find_repository_root(start: Path) -> Path:
    """Return the nearest parent containing the project metadata."""
    resolved_start = start.resolve()
    candidates = (resolved_start, *resolved_start.parents)
    for candidate in candidates:
        if (candidate / "pyproject.toml").is_file() and (candidate / "src").is_dir():
            return candidate
    raise ValueError(
        f"Could not find the repository root from {resolved_start}. "
        "Open this notebook from inside the cloned repository."
    )


def build_install_commands(
    repository_root: Path,
    *,
    python_executable: str,
    include_dev: bool,
    include_gpu: bool,
) -> list[list[str]]:
    """Build deterministic pip commands for the selected environment layers."""
    editable_target = str(repository_root)
    if include_dev:
        editable_target = f"{editable_target}[dev]"

    commands = [
        [
            python_executable,
            "-m",
            "pip",
            "install",
            "--editable",
            editable_target,
        ]
    ]
    if include_gpu:
        requirements_path = repository_root / GPU_REQUIREMENTS
        if not requirements_path.is_file():
            raise ValueError(f"GPU requirements file does not exist: {requirements_path}")
        commands.append(
            [
                python_executable,
                "-m",
                "pip",
                "install",
                "--index-url",
                PYTORCH_INDEX_URL,
                "--extra-index-url",
                PYPI_INDEX_URL,
                "--requirement",
                str(requirements_path),
            ]
        )
    return commands


def format_command(command: Sequence[str]) -> str:
    """Render a command for display without changing how it will execute."""
    if os.name == "nt":
        return subprocess.list2cmdline(command)
    return shlex.join(command)


def run_install_commands(
    commands: Sequence[Sequence[str]],
    *,
    dry_run: bool,
    runner: CommandRunner = subprocess.run,
) -> None:
    """Run pip commands sequentially, or only display them in dry-run mode."""
    for command in commands:
        print(format_command(command))
        if not dry_run:
            runner(list(command), check=True, text=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Install verifier-bottleneck into the current Python interpreter. "
            "Designed for execution with the Jupyter kernel's sys.executable."
        )
    )
    parser.add_argument(
        "--repository-root",
        type=Path,
        help="Repository root. By default it is discovered from the current directory.",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Also install pytest, Ruff, mypy, and PyYAML.",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Install the full GPU requirements file. This can take several minutes.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print pip commands without installing anything.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Install the selected Jupyter environment layers."""
    args = _build_parser().parse_args(argv)
    try:
        validate_python_version(sys.version_info.major, sys.version_info.minor)
        repository_root = (
            args.repository_root.resolve()
            if args.repository_root is not None
            else find_repository_root(Path.cwd())
        )
        commands = build_install_commands(
            repository_root,
            python_executable=sys.executable,
            include_dev=args.dev,
            include_gpu=args.gpu,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    print(f"Repository: {repository_root}")
    print(f"Python: {sys.executable}")
    if args.gpu:
        print("GPU layer selected; no models will be downloaded by this script.")
    run_install_commands(commands, dry_run=args.dry_run)
    if not args.dry_run:
        print("Installation complete. Restart the Jupyter kernel before continuing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
