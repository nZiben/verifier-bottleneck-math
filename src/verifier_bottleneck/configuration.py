"""Strict, reusable helpers for YAML-backed experiment configuration."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast


def as_mapping(value: object, *, field: str) -> Mapping[str, object]:
    """Return ``value`` as a mapping or raise a field-specific error."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return cast(Mapping[str, object], value)


def require_exact_keys(value: Mapping[str, object], *, field: str, required: set[str]) -> None:
    """Reject missing and unknown configuration keys."""
    missing = required - value.keys()
    unknown = value.keys() - required
    if missing:
        raise ValueError(f"{field} is missing required fields: {sorted(missing)}")
    if unknown:
        raise ValueError(f"{field} has unsupported fields: {sorted(unknown)}")


def as_int(value: object, *, field: str) -> int:
    """Parse a strict integer (booleans are not accepted)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def as_float(value: object, *, field: str) -> float:
    """Parse a strict finite-style numeric YAML value."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be a number")
    return float(value)


def as_str(value: object, *, field: str) -> str:
    """Parse a string value."""
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return value


def as_str_tuple(value: object, *, field: str) -> tuple[str, ...]:
    """Parse a YAML list of strings as an immutable tuple."""
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field} must be a list of strings")
    return tuple(cast(list[str], value))


def load_yaml(path: Path) -> object:
    """Load one YAML document with consistent dependency and I/O errors."""
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as error:
        raise RuntimeError("PyYAML is required to load experiment configs") from error
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"could not load config {path}: {error}") from error
