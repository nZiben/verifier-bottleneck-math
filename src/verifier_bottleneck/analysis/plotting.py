"""Small dependency-free SVG primitives shared by experiment analyses."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast
from xml.sax.saxutils import escape

COLORS = (
    "#4c78a8",
    "#f58518",
    "#54a24b",
    "#e45756",
    "#72b7b2",
    "#b279a2",
    "#ff9da6",
    "#9d755d",
)


def _scale(value: float, minimum: float, maximum: float, start: float, size: float) -> float:
    if maximum == minimum:
        return start + size / 2.0
    return start + (value - minimum) / (maximum - minimum) * size


def line_chart_svg(
    *,
    title: str,
    panels: Sequence[Mapping[str, object]],
    x_label: str,
) -> str:
    """Render a multi-panel line chart as a standalone SVG document."""
    panel_count = len(panels)
    if panel_count < 1:
        raise ValueError("line chart requires at least one panel")
    maximum_series = max(
        len(cast(Sequence[Mapping[str, object]], panel["series"])) for panel in panels
    )
    width = max(900, 120 + panel_count * 260 + (panel_count - 1) * 70)
    height = 560
    outer_left = 72
    outer_right = 25
    gap = 70
    top = 90 + maximum_series * 16
    bottom = 78
    panel_width = (width - outer_left - outer_right - gap * (len(panels) - 1)) / len(panels)
    panel_height = height - top - bottom
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" ',
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="Arial, sans-serif" fill="#222">',
        f'<text x="{width / 2}" y="30" text-anchor="middle" font-size="20" ',
        f'font-weight="bold">{escape(title)}</text>',
    ]
    for panel_index, panel in enumerate(panels):
        left = outer_left + panel_index * (panel_width + gap)
        series = cast(Sequence[Mapping[str, object]], panel["series"])
        all_points = [
            point
            for item in series
            for point in cast(Sequence[tuple[float, float]], item["points"])
        ]
        x_min = float(cast(float, panel.get("x_min", min(point[0] for point in all_points))))
        x_max = float(cast(float, panel.get("x_max", max(point[0] for point in all_points))))
        y_min = float(cast(float, panel.get("y_min", 0.0)))
        y_max = float(cast(float, panel["y_max"]))
        y_tick_format = str(panel.get("y_tick_format", "decimal"))
        elements.extend(
            [
                f'<text x="{left + panel_width / 2}" y="58" text-anchor="middle" ',
                f'font-size="15" font-weight="bold">{escape(str(panel["title"]))}</text>',
                f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + panel_height}" ',
                'stroke="#333"/>',
                f'<line x1="{left}" y1="{top + panel_height}" ',
                f'x2="{left + panel_width}" y2="{top + panel_height}" stroke="#333"/>',
            ]
        )
        for tick in range(6):
            value = y_min + (y_max - y_min) * tick / 5
            y = top + panel_height - panel_height * tick / 5
            tick_label = f"{100 * value:.0f}%" if y_tick_format == "percent" else f"{value:.2f}"
            elements.extend(
                [
                    f'<line x1="{left}" y1="{y:.2f}" x2="{left + panel_width}" ',
                    f'y2="{y:.2f}" stroke="#e5e5e5"/>',
                    f'<text x="{left - 9}" y="{y + 4:.2f}" text-anchor="end" ',
                    f'font-size="11">{tick_label}</text>',
                ]
            )
        x_ticks = sorted({point[0] for point in all_points})
        for value in x_ticks:
            x = _scale(value, x_min, x_max, left, panel_width)
            elements.append(
                f'<text x="{x:.2f}" y="{top + panel_height + 20}" '
                f'text-anchor="middle" font-size="11">{value:g}</text>'
            )
        best_x = panel.get("best_x")
        if isinstance(best_x, int | float):
            x = _scale(float(best_x), x_min, x_max, left, panel_width)
            best_label = escape(str(panel.get("best_label", "best evaluation")))
            elements.extend(
                [
                    f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" ',
                    f'y2="{top + panel_height}" stroke="#777" stroke-dasharray="5,4"/>',
                    f'<text x="{x + 5:.2f}" y="{top + 14}" font-size="10">{best_label}</text>',
                ]
            )
        for series_index, item in enumerate(series):
            color = str(item.get("color", COLORS[series_index % len(COLORS)]))
            points = cast(Sequence[tuple[float, float]], item["points"])
            coordinates = " ".join(
                f"{_scale(x, x_min, x_max, left, panel_width):.2f},"
                f"{top + panel_height - _scale(y, y_min, y_max, 0, panel_height):.2f}"
                for x, y in points
            )
            elements.append(
                f'<polyline points="{coordinates}" fill="none" stroke="{color}" '
                'stroke-width="2.5"/>'
            )
            for x_value, y_value in points:
                x = _scale(x_value, x_min, x_max, left, panel_width)
                y = top + panel_height - _scale(y_value, y_min, y_max, 0, panel_height)
                elements.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.5" fill="{color}"/>')
            legend_x = left + 6
            legend_y = 78 + series_index * 16
            elements.extend(
                [
                    f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 20}" ',
                    f'y2="{legend_y}" stroke="{color}" stroke-width="3"/>',
                    f'<text x="{legend_x + 25}" y="{legend_y + 4}" font-size="11">',
                    f"{escape(str(item['label']))}</text>",
                ]
            )
        elements.extend(
            [
                f'<text transform="translate({left - 48} {top + panel_height / 2}) ',
                'rotate(-90)" text-anchor="middle" font-size="13">',
                f"{escape(str(panel['y_label']))}</text>",
            ]
        )
    elements.append(
        f'<text x="{width / 2}" y="{height - 18}" text-anchor="middle" '
        f'font-size="13">{escape(x_label)}</text>'
    )
    elements.append("</g></svg>")
    return "".join(elements)
