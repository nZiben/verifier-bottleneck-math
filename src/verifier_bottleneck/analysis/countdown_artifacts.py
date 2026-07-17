"""Small run-time artifacts shared by Countdown experiment backends."""

from __future__ import annotations

import csv
import html
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

from verifier_bottleneck.experiment_tracking import register_artifact


def write_temperature_sweep_artifacts(
    record_path: Path, measurements: Sequence[Mapping[str, object]]
) -> None:
    """Write the standard long-form CSV and compact preview plot."""
    run_directory = record_path.parent
    csv_path = run_directory / "temperature-sweep.csv"
    fields = [
        "benchmark",
        "temperature",
        "tasks",
        "proposals_per_task",
        "pass_at_k",
        "solved",
        "rate",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for measurement in measurements:
            rates = cast(Mapping[str, object], measurement["pass_at_k"])
            solved = cast(Mapping[str, object], measurement["solved_at_k"])
            for key in sorted(rates, key=lambda value: int(value.removeprefix("pass@"))):
                writer.writerow(
                    {
                        "benchmark": measurement["benchmark"],
                        "temperature": measurement["temperature"],
                        "tasks": measurement["tasks"],
                        "proposals_per_task": measurement["proposals_per_task"],
                        "pass_at_k": key,
                        "solved": solved[key],
                        "rate": rates[key],
                    }
                )

    maximum_k = max(
        int(key.removeprefix("pass@"))
        for key in cast(Mapping[str, object], measurements[0]["pass_at_k"])
    )
    width, height = 760, 430
    left, top, plot_width, plot_height = 70, 45, 640, 310
    temperatures = sorted(
        {float(cast(float, measurement["temperature"])) for measurement in measurements}
    )
    colors = {"countdown_numeric_b": "#4c78a8", "countdown_symbolic_ab": "#e45756"}
    lines = []
    for benchmark, color in colors.items():
        rows = sorted(
            (measurement for measurement in measurements if measurement["benchmark"] == benchmark),
            key=lambda row: float(cast(float, row["temperature"])),
        )
        points = []
        for row in rows:
            temperature = float(cast(float, row["temperature"]))
            rate = float(
                cast(
                    float,
                    cast(Mapping[str, object], row["pass_at_k"])[f"pass@{maximum_k}"],
                )
            )
            x = left + temperature / max(temperatures[-1], 1e-9) * plot_width
            y = top + (1.0 - rate) * plot_height
            points.append(f"{x:.1f},{y:.1f}")
        lines.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{" ".join(points)}"/>'
        )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        '<rect width="100%" height="100%" fill="white"/>'
        '<g font-family="sans-serif" fill="#222">'
        f'<text x="{width / 2}" y="24" text-anchor="middle" font-size="18">'
        f"Numeric B versus unseen symbolic A+B pass@{maximum_k}</text>"
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#222"/>'
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" '
        f'y2="{top + plot_height}" stroke="#222"/>'
        f"{''.join(lines)}"
        f'<text x="{left}" y="{height - 20}" fill="#4c78a8">numeric B</text>'
        f'<text x="{left + 100}" y="{height - 20}" fill="#e45756">symbolic A+B</text>'
        f'<text x="{width / 2}" y="{height - 3}" text-anchor="middle">temperature</text>'
        "</g></svg>"
    )
    svg_path = run_directory / "temperature-sweep.svg"
    svg_path.write_text(html.unescape(svg), encoding="utf-8")
    register_artifact(
        record_path,
        csv_path,
        kind="table",
        description="Numeric-B and symbolic-A+B temperature measurements.",
    )
    register_artifact(
        record_path,
        svg_path,
        kind="plot",
        description=f"Numeric B versus symbolic A+B pass@{maximum_k}.",
    )
