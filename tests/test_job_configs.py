from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).parent.parent


@pytest.mark.parametrize(
    ("config_name", "instance_type", "requires_cuda"),
    [
        ("cpu-smoke.yaml", "c1.8", False),
        ("gpu-smoke.yaml", "gt4.1", True),
    ],
)
def test_job_config(
    config_name: str, instance_type: str, requires_cuda: bool
) -> None:
    config_path = PROJECT_ROOT / "jobs" / config_name
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["cloud-instance-types"] == [instance_type]
    assert config["env"] == {"python": "auto"}
    assert config["outputs"]
    assert "src/verifier_bottleneck/cli.py" in config["cmd"]
    assert ("--require-cuda" in config["cmd"]) is requires_cuda


@pytest.mark.parametrize(
    ("config_name", "instance_type"),
    [
        ("gpu-dependencies-t4-smoke.yaml", "gt4.1"),
        ("gpu-dependencies-v100-smoke.yaml", "g1.1"),
        ("gpu-dependencies-a100-smoke.yaml", "g2.1"),
    ],
)
def test_gpu_dependency_job_uses_explicit_manual_environment(
    config_name: str, instance_type: str
) -> None:
    config_path = PROJECT_ROOT / "jobs" / config_name
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    python_environment = config["env"]["python"]

    assert config["cloud-instance-types"] == [instance_type]
    assert python_environment["type"] == "manual"
    assert python_environment["version"] == "3.10"
    assert python_environment["requirements-file"] == (
        "requirements-rl-compositionality-gpu.txt"
    )
    local_paths = python_environment["local-paths"]
    assert local_paths == ["src/verifier_bottleneck"]
    assert all((PROJECT_ROOT / path).is_dir() for path in local_paths)
    assert "-m verifier_bottleneck.dependency_report" in config["cmd"]
    assert "--require-cuda" in config["cmd"]


def test_datasphere_requirements_file_contains_only_requirement_specifiers() -> None:
    requirements_path = PROJECT_ROOT / "requirements-rl-compositionality-gpu.txt"
    lines = requirements_path.read_text(encoding="utf-8").splitlines()

    assert lines
    assert all(line == line.strip() and line and "#" not in line for line in lines)
