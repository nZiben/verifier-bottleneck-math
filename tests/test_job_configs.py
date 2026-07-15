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
