from pathlib import Path

import pytest

from verifier_bottleneck.jupyter_setup import (
    GPU_REQUIREMENTS,
    build_install_commands,
    find_repository_root,
    run_install_commands,
    validate_python_version,
)


def test_find_repository_root_from_nested_directory(tmp_path: Path) -> None:
    repository_root = tmp_path / "repository"
    nested_directory = repository_root / "notebooks" / "experiments"
    nested_directory.mkdir(parents=True)
    (repository_root / "pyproject.toml").write_text("", encoding="utf-8")
    (repository_root / "src").mkdir()

    assert find_repository_root(nested_directory) == repository_root


def test_find_repository_root_has_clear_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Could not find the repository root"):
        find_repository_root(tmp_path)


def test_python_version_validation_is_explicit() -> None:
    validate_python_version(3, 10)

    with pytest.raises(ValueError, match="Python 3.10 is required"):
        validate_python_version(3, 11)


def test_build_install_commands_keeps_gpu_layer_explicit(tmp_path: Path) -> None:
    (tmp_path / GPU_REQUIREMENTS).write_text("torch==2.6.0\n", encoding="utf-8")

    commands = build_install_commands(
        tmp_path,
        python_executable="/kernel/python",
        include_dev=True,
        include_gpu=True,
    )

    assert commands[0] == [
        "/kernel/python",
        "-m",
        "pip",
        "install",
        "--editable",
        f"{tmp_path}[dev]",
    ]
    assert commands[1][-2:] == ["--requirement", str(tmp_path / GPU_REQUIREMENTS)]
    assert "--index-url" in commands[1]
    assert "--extra-index-url" in commands[1]


def test_build_install_commands_omits_gpu_by_default(tmp_path: Path) -> None:
    commands = build_install_commands(
        tmp_path,
        python_executable="/kernel/python",
        include_dev=False,
        include_gpu=False,
    )

    assert len(commands) == 1
    assert commands[0][-1] == str(tmp_path)


def test_dry_run_never_executes_pip() -> None:
    def unexpected_runner(*args: object, **kwargs: object) -> None:
        raise AssertionError("pip must not execute during a dry run")

    run_install_commands(
        [["python", "-m", "pip", "install", "--editable", "."]],
        dry_run=True,
        runner=unexpected_runner,
    )
