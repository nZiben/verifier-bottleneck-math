"""Repository entrypoint for setting up a DataSphere Jupyter kernel."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))


if __name__ == "__main__":
    main = importlib.import_module("verifier_bottleneck.jupyter_setup").main
    raise SystemExit(main(["--repository-root", str(REPOSITORY_ROOT), *sys.argv[1:]]))
