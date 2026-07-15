from pathlib import Path

import pytest

from scripts import submit_job


def test_submission_helper_is_dry_run_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "job.yaml"
    config.write_text("name: test\n", encoding="utf-8")

    def unexpected_run(*args: object, **kwargs: object) -> None:
        raise AssertionError("subprocess.run must not be called during a dry run")

    monkeypatch.setattr(submit_job.subprocess, "run", unexpected_run)

    assert submit_job.main([str(config)]) == 0


def test_submission_helper_requires_datasphere_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "job.yaml"
    config.write_text("name: test\n", encoding="utf-8")
    monkeypatch.setattr(submit_job.shutil, "which", lambda executable: None)

    with pytest.raises(SystemExit, match="DataSphere CLI was not found"):
        submit_job.main([str(config), "--execute"])
