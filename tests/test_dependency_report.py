from __future__ import annotations

from types import ModuleType, SimpleNamespace

from verifier_bottleneck.dependency_report import (
    DependencySpec,
    _cuda_report,
    measure_dependency_imports,
)


def test_measure_dependency_imports_is_offline_and_deterministic() -> None:
    ticks = iter([0.0, 1.0, 2.5, 4.0])

    results, total_seconds = measure_dependency_imports(
        [DependencySpec("example-dist", "example_module")],
        importer=lambda name: ModuleType(name),
        version_getter=lambda name: "1.2.3",
        clock=lambda: next(ticks),
    )

    assert results == [
        {
            "distribution": "example-dist",
            "module": "example_module",
            "version": "1.2.3",
            "status": "ok",
            "import_seconds": 1.5,
        }
    ]
    assert total_seconds == 4.0


def test_measure_dependency_imports_records_errors() -> None:
    def fail_import(name: str) -> ModuleType:
        raise ImportError(f"cannot import {name}")

    results, _ = measure_dependency_imports(
        [DependencySpec("missing-dist", "missing_module")],
        importer=fail_import,
    )

    assert results[0]["status"] == "error"
    assert results[0]["error"] == "ImportError: cannot import missing_module"


def test_cuda_report_uses_mocked_gpu() -> None:
    cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        get_device_name=lambda index: "Mock GPU",
        get_device_properties=lambda index: SimpleNamespace(total_memory=16 * 1024**3),
    )
    torch = SimpleNamespace(cuda=cuda, version=SimpleNamespace(cuda="12.4"))

    report = _cuda_report(torch)

    assert report["cuda_available"] is True
    assert report["gpu_count"] == 1
    assert report["gpu_names"] == ["Mock GPU"]
    assert report["gpu_memory_bytes"] == [16 * 1024**3]
