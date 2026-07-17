"""Generate plots, tables, and conclusions from an arithmetic Job archive."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from verifier_bottleneck.analysis.arithmetic_results import analyze_archive


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze the ZIP returned by a scratch or Qwen Countdown composition Job."
        )
    )
    parser.add_argument("archive", type=Path, help="ZIP returned by DataSphere")
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Directory in which to materialize the analyzed run",
    )
    parser.add_argument(
        "--analyzed-archive",
        type=Path,
        help="Destination for the augmented ZIP containing the analysis",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Analyze one completed Job archive."""
    args = _build_parser().parse_args(argv)
    archive = args.archive.resolve()
    if not archive.is_file():
        raise SystemExit(f"Job archive does not exist: {archive}")
    if archive.suffix.lower() != ".zip":
        raise SystemExit(f"Job archive must be a ZIP file: {archive}")
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else (Path.cwd() / "outputs" / f"{archive.stem}-analysis").resolve()
    )
    analyzed_archive = (
        args.analyzed_archive.resolve()
        if args.analyzed_archive is not None
        else archive.with_name(f"{archive.stem}-analysis.zip")
    )
    paths = analyze_archive(
        archive,
        output_root=output_root,
        analyzed_archive_path=analyzed_archive,
    )
    print(json.dumps(paths, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
