"""Fault-tolerant, immediately flushed console logging for DataSphere Jobs."""

from __future__ import annotations

from datetime import datetime, timezone

_console_logging_enabled = True


def live_log(message: str) -> None:
    """Emit a timestamped message without allowing a closed pipe to fail a run."""
    global _console_logging_enabled
    if not _console_logging_enabled:
        return
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        print(f"[{timestamp}] {message}", flush=True)
    except (BrokenPipeError, OSError):
        _console_logging_enabled = False
