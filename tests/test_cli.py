from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from verifier_bottleneck import cli


class FakeCuda:
    def is_available(self) -> bool:
        return True

    def device_count(self) -> int:
        return 2

    def get_device_name(self, device_index: int) -> str:
        return f"Mock GPU {device_index}"

    def get_device_properties(self, device_index: int) -> SimpleNamespace:
        return SimpleNamespace(total_memory=(device_index + 1) * 1024)


def test_report_without_torch_or_datasphere(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DS_PROJECT_HOME", raising=False)
    monkeypatch.setattr(cli, "_load_torch", lambda: None)

    report = cli.collect_system_report()

    assert report["cuda_available"] is False
    assert report["gpu_count"] == 0
    assert report["gpu_names"] == []
    assert report["gpu_memory_bytes"] == []
    assert "ds_project_home" not in report
    assert "pytorch_version" not in report
    assert isinstance(report["free_disk_space_bytes"], int)


def test_report_with_mocked_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DS_PROJECT_HOME", "/project")
    torch = SimpleNamespace(
        __version__="2.6.0",
        version=SimpleNamespace(cuda="12.4"),
        cuda=FakeCuda(),
    )

    report = cli.collect_system_report(torch)

    assert report["ds_project_home"] == "/project"
    assert report["pytorch_version"] == "2.6.0"
    assert report["cuda_runtime_version"] == "12.4"
    assert report["cuda_available"] is True
    assert report["gpu_count"] == 2
    assert report["gpu_names"] == ["Mock GPU 0", "Mock GPU 1"]
    assert report["gpu_memory_bytes"] == [1024, 2048]


def test_cli_writes_json_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "_load_torch", lambda: None)
    output = tmp_path / "nested" / "report.json"

    exit_code = cli.main(["--output", str(output)])

    assert exit_code == 0
    assert json.loads(output.read_text(encoding="utf-8"))["cuda_available"] is False


def test_cli_requires_cuda(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "_load_torch", lambda: None)
    output = tmp_path / "report.json"

    exit_code = cli.main(["--output", str(output), "--require-cuda"])

    assert exit_code == 1
    assert output.exists()


def test_cli_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--help"])

    output = capsys.readouterr().out
    assert exc_info.value.code == 0
    assert "--output" in output
    assert "--require-cuda" in output
