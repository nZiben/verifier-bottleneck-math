"""Generate a JSON report describing the local execution environment."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any


def _load_torch() -> ModuleType | None:
    """Import PyTorch when it is available without making it a dependency."""
    try:
        return importlib.import_module("torch")
    except (ImportError, OSError):
        return None


def collect_system_report(torch_module: Any | None = None) -> dict[str, object]:
    """Collect platform, disk, and optional PyTorch/CUDA information."""
    cwd = Path.cwd()
    report: dict[str, object] = {
        "python_version": platform.python_version(),
        "operating_system": platform.platform(),
        "current_working_directory": str(cwd),
        "free_disk_space_bytes": shutil.disk_usage(cwd).free,
        "cuda_available": False,
        "gpu_count": 0,
        "gpu_names": [],
        "gpu_memory_bytes": [],
    }

    ds_project_home = os.environ.get("DS_PROJECT_HOME")
    if ds_project_home is not None:
        report["ds_project_home"] = ds_project_home

    torch = _load_torch() if torch_module is None else torch_module
    if torch is None:
        return report

    report["pytorch_version"] = str(torch.__version__)
    cuda_runtime_version = torch.version.cuda
    if cuda_runtime_version is not None:
        report["cuda_runtime_version"] = str(cuda_runtime_version)

    cuda_available = bool(torch.cuda.is_available())
    report["cuda_available"] = cuda_available
    if not cuda_available:
        return report

    gpu_count = int(torch.cuda.device_count())
    gpu_names: list[str] = []
    gpu_memory_bytes: list[int] = []
    for device_index in range(gpu_count):
        gpu_names.append(str(torch.cuda.get_device_name(device_index)))
        properties = torch.cuda.get_device_properties(device_index)
        gpu_memory_bytes.append(int(properties.total_memory))

    report["gpu_count"] = gpu_count
    report["gpu_names"] = gpu_names
    report["gpu_memory_bytes"] = gpu_memory_bytes
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="verifier-bottleneck-report",
        description="Write a JSON report for the current Python, OS, disk, and CUDA environment.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write JSON to this path instead of standard output.",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Return a non-zero exit status when CUDA is unavailable.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the system-report command."""
    args = _build_parser().parse_args(argv)
    report = collect_system_report()
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"

    if args.output is None:
        sys.stdout.write(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")

    if args.require_cuda and not report["cuda_available"]:
        print("CUDA is required but unavailable.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
