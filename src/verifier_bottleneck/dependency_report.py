"""Measure import compatibility and startup time for the GPU dependency stack."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import platform
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any


@dataclass(frozen=True)
class DependencySpec:
    """Map a distribution name to the module imported for its smoke test."""

    distribution: str
    module: str


DEPENDENCIES = (
    DependencySpec("accelerate", "accelerate"),
    DependencySpec("codetiming", "codetiming"),
    DependencySpec("datasets", "datasets"),
    DependencySpec("dill", "dill"),
    DependencySpec("hydra-core", "hydra"),
    DependencySpec("liger-kernel", "liger_kernel"),
    DependencySpec("numpy", "numpy"),
    DependencySpec("pandas", "pandas"),
    DependencySpec("peft", "peft"),
    DependencySpec("pyarrow", "pyarrow"),
    DependencySpec("pybind11", "pybind11"),
    DependencySpec("pylatexenc", "pylatexenc"),
    DependencySpec("ray", "ray"),
    DependencySpec("reasoning-gym", "reasoning_gym"),
    DependencySpec("tabulate", "tabulate"),
    DependencySpec("tensordict", "tensordict"),
    DependencySpec("torch", "torch"),
    DependencySpec("torchdata", "torchdata"),
    DependencySpec("transformers", "transformers"),
    DependencySpec("vllm", "vllm"),
    DependencySpec("wandb", "wandb"),
)


def measure_dependency_imports(
    dependencies: Sequence[DependencySpec] = DEPENDENCIES,
    *,
    importer: Callable[[str], ModuleType] = importlib.import_module,
    version_getter: Callable[[str], str] = importlib.metadata.version,
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[list[dict[str, object]], float]:
    """Import each dependency and return per-import and total elapsed seconds."""
    total_started = clock()
    results: list[dict[str, object]] = []
    for dependency in dependencies:
        import_started = clock()
        result: dict[str, object] = {
            "distribution": dependency.distribution,
            "module": dependency.module,
        }
        try:
            importer(dependency.module)
            result["version"] = version_getter(dependency.distribution)
            result["status"] = "ok"
        except Exception as error:
            result["status"] = "error"
            result["error"] = f"{type(error).__name__}: {error}"
        result["import_seconds"] = clock() - import_started
        results.append(result)
    return results, clock() - total_started


def _cuda_report(torch: Any | None) -> dict[str, object]:
    report: dict[str, object] = {
        "cuda_available": False,
        "gpu_count": 0,
        "gpu_names": [],
        "gpu_memory_bytes": [],
    }
    if torch is None:
        return report

    cuda_available = bool(torch.cuda.is_available())
    report["cuda_available"] = cuda_available
    report["cuda_runtime_version"] = torch.version.cuda
    if not cuda_available:
        return report

    gpu_count = int(torch.cuda.device_count())
    report["gpu_count"] = gpu_count
    report["gpu_names"] = [
        str(torch.cuda.get_device_name(index)) for index in range(gpu_count)
    ]
    report["gpu_memory_bytes"] = [
        int(torch.cuda.get_device_properties(index).total_memory)
        for index in range(gpu_count)
    ]
    return report


def collect_dependency_report() -> dict[str, object]:
    """Collect dependency import timings and CUDA information."""
    imports, total_import_seconds = measure_dependency_imports()
    report: dict[str, object] = {
        "python_version": platform.python_version(),
        "operating_system": platform.platform(),
        "dependencies": imports,
        "total_import_seconds": total_import_seconds,
        "all_imports_succeeded": all(item["status"] == "ok" for item in imports),
    }
    report.update(_cuda_report(sys.modules.get("torch")))
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure imports for the RL-Compositionality-derived GPU environment."
    )
    parser.add_argument("--output", type=Path, required=True, help="JSON report path.")
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail when PyTorch cannot access CUDA.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Write the dependency timing report and return its compatibility status."""
    args = _build_parser().parse_args(argv)
    report = collect_dependency_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    if not report["all_imports_succeeded"]:
        print("One or more dependency imports failed; inspect the JSON report.", file=sys.stderr)
        return 1
    if args.require_cuda and not report["cuda_available"]:
        print("CUDA is required but unavailable.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
