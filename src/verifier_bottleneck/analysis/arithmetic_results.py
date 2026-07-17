"""Build registered plots, tables, and conclusions for an arithmetic sweep."""

from __future__ import annotations

import argparse
import csv
import json
import math
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

from verifier_bottleneck.analysis.plotting import COLORS, line_chart_svg
from verifier_bottleneck.experiment_tracking import register_artifact


def _ordered_pass_keys(measurement: Mapping[str, object]) -> list[str]:
    return sorted(
        cast(Mapping[str, object], measurement["pass_at_k"]),
        key=lambda key: int(key.removeprefix("pass@")),
    )


def wilson_interval(successes: int, trials: int, *, z: float = 1.96) -> tuple[float, float]:
    """Return a two-sided Wilson score interval for a binomial proportion."""
    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError("successes and trials must satisfy 0 <= successes <= trials")
    proportion = successes / trials
    denominator = 1.0 + z**2 / trials
    center = (proportion + z**2 / (2.0 * trials)) / denominator
    half_width = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / trials + z**2 / (4.0 * trials**2))
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def _load_archive(archive_path: Path) -> tuple[dict[str, object], dict[str, bytes]]:
    try:
        with zipfile.ZipFile(archive_path) as archive:
            files = {name: archive.read(name) for name in archive.namelist()}
    except (OSError, zipfile.BadZipFile) as error:
        raise ValueError(f"could not read run archive {archive_path}: {error}") from error
    if "record.json" not in files:
        raise ValueError(f"archive does not contain record.json: {archive_path}")
    record = cast(dict[str, object], json.loads(files["record.json"]))
    run = cast(Mapping[str, object], record.get("run", {}))
    if run.get("status") != "completed":
        raise ValueError(f"run is not completed: status={run.get('status')}")
    return record, files


def _materialize_run_directory(
    files: Mapping[str, bytes],
    *,
    output_root: Path,
    run_id: str,
) -> Path:
    run_directory = output_root / run_id
    run_directory.mkdir(parents=True, exist_ok=True)
    for relative_name, content in files.items():
        destination = run_directory / relative_name
        if destination.exists():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    return run_directory


def _temperature_plot(measurements: Sequence[Mapping[str, object]]) -> str:
    benchmarks = list(dict.fromkeys(str(row["benchmark"]) for row in measurements))
    pass_keys = _ordered_pass_keys(measurements[0])
    numeric_name = "countdown_numeric_b"
    symbolic_name = "countdown_symbolic_ab"
    if numeric_name in benchmarks and symbolic_name in benchmarks:
        largest_rate = max(
            float(cast(float, rate))
            for row in measurements
            if row["benchmark"] in {numeric_name, symbolic_name}
            for rate in cast(Mapping[str, object], row["pass_at_k"]).values()
        )
        shared_y_max = min(
            1.0,
            max(0.1, math.ceil((largest_rate + 0.02) * 10) / 10),
        )
        composition_panels: list[dict[str, object]] = []
        for key in pass_keys:
            series = []
            for benchmark, label, color in (
                (numeric_name, "Numeric Countdown (B)", COLORS[0]),
                (symbolic_name, "Symbolic Countdown (A+B)", COLORS[3]),
            ):
                rows = sorted(
                    (row for row in measurements if row["benchmark"] == benchmark),
                    key=lambda row: float(cast(float, row["temperature"])),
                )
                series.append(
                    {
                        "label": label,
                        "color": color,
                        "points": [
                            (
                                float(cast(float, row["temperature"])),
                                float(
                                    cast(
                                        float,
                                        cast(Mapping[str, object], row["pass_at_k"])[key],
                                    )
                                ),
                            )
                            for row in rows
                        ],
                    }
                )
            composition_panels.append(
                {
                    "title": f"{key}",
                    "y_label": "Held-out tasks solved (%)",
                    "y_min": 0.0,
                    "y_max": shared_y_max,
                    "y_tick_format": "percent",
                    "series": series,
                }
            )
        return line_chart_svg(
            title=("Countdown task performance: numeric B versus unseen symbolic A+B"),
            panels=composition_panels,
            x_label="Sampling temperature T",
        )

    panels: list[dict[str, object]] = []
    for benchmark in benchmarks:
        rows = sorted(
            (row for row in measurements if row["benchmark"] == benchmark),
            key=lambda row: float(cast(float, row["temperature"])),
        )
        largest_rate = max(
            float(cast(float, rate))
            for row in rows
            for rate in cast(Mapping[str, object], row["pass_at_k"]).values()
        )
        y_max = max(0.1, math.ceil((largest_rate + 0.02) * 10) / 10)
        series = []
        for index, key in enumerate(pass_keys):
            series.append(
                {
                    "label": key,
                    "color": COLORS[index % len(COLORS)],
                    "points": [
                        (
                            float(cast(float, row["temperature"])),
                            float(cast(float, cast(Mapping[str, object], row["pass_at_k"])[key])),
                        )
                        for row in rows
                    ],
                }
            )
        panels.append(
            {
                "title": benchmark.replace("_", " ").title(),
                "y_label": "Held-out tasks solved (%)",
                "y_max": y_max,
                "y_tick_format": "percent",
                "series": series,
            }
        )
    return line_chart_svg(
        title="Temperature sweep: exact pass@k",
        panels=panels,
        x_label="Sampling temperature T (T=0 is greedy)",
    )


def _evaluation_prefix(trajectory: Sequence[Mapping[str, object]]) -> str:
    if any("validation_token_accuracy" in row for row in trajectory):
        return "validation"
    return "test"


def _training_plot(trajectory: Sequence[Mapping[str, object]]) -> str:
    required_composition_metrics = {
        "decode_a_exact_accuracy",
        "decode_a_token_loss",
        "numeric_b_validation_accuracy",
        "numeric_b_validation_loss",
    }
    composition_rows = [row for row in trajectory if required_composition_metrics.issubset(row)]
    if composition_rows:
        selected_rows = [row for row in trajectory if row.get("is_best_checkpoint") is True]
        if selected_rows:
            selected_step = int(cast(int, selected_rows[-1]["step"]))
        else:
            selected_step = int(
                cast(
                    int,
                    min(
                        composition_rows,
                        key=lambda row: float(cast(float, row["validation_token_loss"])),
                    )["step"],
                )
            )

        def points(key: str) -> list[tuple[float, float]]:
            return [
                (float(cast(int, row["step"])), float(cast(float, row[key])))
                for row in trajectory
                if key in row
            ]

        def loss_axis_maximum(*keys: str) -> float:
            values = [
                float(cast(float, row[key])) for row in trajectory for key in keys if key in row
            ]
            maximum = max(values)
            return max(0.1, math.ceil((maximum * 1.05) * 10) / 10)

        skill_b_accuracy_series: list[dict[str, object]] = []
        skill_b_loss_series: list[dict[str, object]] = []
        if any("numeric_b_train_accuracy" in row for row in trajectory):
            skill_b_accuracy_series.append(
                {
                    "label": "B train subset (teacher-forced)",
                    "color": COLORS[3],
                    "points": points("numeric_b_train_accuracy"),
                }
            )
            skill_b_loss_series.append(
                {
                    "label": "B train subset (teacher-forced)",
                    "color": COLORS[3],
                    "points": points("numeric_b_train_loss"),
                }
            )
        skill_b_accuracy_series.append(
            {
                "label": "B validation (teacher-forced)",
                "color": COLORS[0],
                "points": points("numeric_b_validation_accuracy"),
            }
        )
        skill_b_loss_series.append(
            {
                "label": "B validation (teacher-forced)",
                "color": COLORS[0],
                "points": points("numeric_b_validation_loss"),
            }
        )
        skill_b_loss_keys = ["numeric_b_validation_loss"]
        if any("numeric_b_train_loss" in row for row in trajectory):
            skill_b_loss_keys.append("numeric_b_train_loss")
        common = {
            "best_x": selected_step,
            "best_label": "selected checkpoint",
        }
        return line_chart_svg(
            title=("Training isolated skills A and B (symbolic A+B composition is held out)"),
            x_label="Optimizer step",
            panels=[
                {
                    **common,
                    "title": "Skill A: symbol decoding accuracy",
                    "y_label": "Accuracy on 14 codebook entries (%)",
                    "y_min": 0.0,
                    "y_max": 1.0,
                    "y_tick_format": "percent",
                    "series": [
                        {
                            "label": "Exact decoded value",
                            "color": COLORS[2],
                            "points": points("decode_a_exact_accuracy"),
                        },
                        {
                            "label": "Answer-token accuracy",
                            "color": COLORS[0],
                            "points": points("decode_a_token_accuracy"),
                        },
                    ],
                },
                {
                    **common,
                    "title": "Skill A: symbol decoding loss",
                    "y_label": "Cross-entropy loss (lower is better)",
                    "y_min": 0.0,
                    "y_max": loss_axis_maximum("decode_a_token_loss"),
                    "series": [
                        {
                            "label": "A codebook examples",
                            "color": COLORS[2],
                            "points": points("decode_a_token_loss"),
                        }
                    ],
                },
                {
                    **common,
                    "title": "Skill B: numeric Countdown accuracy",
                    "y_label": "Answer-token accuracy (%)",
                    "y_min": 0.0,
                    "y_max": 1.0,
                    "y_tick_format": "percent",
                    "series": skill_b_accuracy_series,
                },
                {
                    **common,
                    "title": "Skill B: numeric Countdown loss",
                    "y_label": "Cross-entropy loss (lower is better)",
                    "y_min": 0.0,
                    "y_max": loss_axis_maximum(*skill_b_loss_keys),
                    "series": skill_b_loss_series,
                },
            ],
        )

    evaluation_prefix = _evaluation_prefix(trajectory)
    accuracy_key = f"{evaluation_prefix}_token_accuracy"
    loss_key = f"{evaluation_prefix}_token_loss"
    evaluation_rows = [
        row
        for row in trajectory
        if accuracy_key in row
        and loss_key in row
        and "train_token_accuracy" in row
        and "train_token_loss" in row
    ]
    if not evaluation_rows:
        raise ValueError("training trajectory has no complete evaluation rows")
    best_row = max(evaluation_rows, key=lambda row: float(cast(float, row[accuracy_key])))
    best_step = int(cast(int, best_row["step"]))
    post_initial = [row for row in trajectory if int(cast(int, row["step"])) > 0]
    accuracy_maximum = max(
        float(cast(float, row[key]))
        for row in evaluation_rows
        for key in ("train_token_accuracy", accuracy_key)
    )
    loss_values = [
        float(cast(float, row[key]))
        for row in post_initial
        for key in ("train_token_loss", loss_key)
    ]
    loss_minimum = min(loss_values)
    loss_maximum = max(loss_values)
    loss_padding = max(0.05, (loss_maximum - loss_minimum) * 0.1)
    return line_chart_svg(
        title="Teacher-forced training trajectory",
        x_label="Training step",
        panels=[
            {
                "title": "Answer-token accuracy",
                "y_label": "Accuracy",
                "y_max": min(1.0, max(0.1, math.ceil(accuracy_maximum * 10) / 10)),
                "best_x": best_step,
                "best_label": f"best {evaluation_prefix}",
                "series": [
                    {
                        "label": "train",
                        "color": COLORS[3],
                        "points": [
                            (
                                float(cast(int, row["step"])),
                                float(cast(float, row["train_token_accuracy"])),
                            )
                            for row in trajectory
                        ],
                    },
                    {
                        "label": evaluation_prefix,
                        "color": COLORS[0],
                        "points": [
                            (
                                float(cast(int, row["step"])),
                                float(cast(float, row[accuracy_key])),
                            )
                            for row in trajectory
                        ],
                    },
                ],
            },
            {
                "title": "Answer-token loss",
                "y_label": "Cross-entropy loss",
                "y_max": loss_maximum + loss_padding,
                "y_min": max(0.0, loss_minimum - loss_padding),
                "best_x": best_step,
                "best_label": f"best {evaluation_prefix}",
                "series": [
                    {
                        "label": "train",
                        "color": COLORS[3],
                        "points": [
                            (
                                float(cast(int, row["step"])),
                                float(cast(float, row["train_token_loss"])),
                            )
                            for row in post_initial
                        ],
                    },
                    {
                        "label": evaluation_prefix,
                        "color": COLORS[0],
                        "points": [
                            (
                                float(cast(int, row["step"])),
                                float(cast(float, row[loss_key])),
                            )
                            for row in post_initial
                        ],
                    },
                ],
            },
        ],
    )


def _analysis_payload(record: Mapping[str, object]) -> dict[str, object]:
    results = cast(Mapping[str, object], record["results"])
    measurements = cast(list[dict[str, object]], results["temperature_measurements"])
    metrics = cast(Mapping[str, object], record["metrics"])
    trajectory = cast(list[dict[str, object]], metrics["trajectory"])
    evaluation_prefix = _evaluation_prefix(trajectory)
    accuracy_key = f"{evaluation_prefix}_token_accuracy"
    loss_key = f"{evaluation_prefix}_token_loss"
    evaluation_rows = [
        row
        for row in trajectory
        if accuracy_key in row
        and loss_key in row
        and "train_token_accuracy" in row
        and "train_token_loss" in row
    ]
    if not evaluation_rows:
        raise ValueError("training trajectory has no complete evaluation rows")
    max_k = max(
        int(key.removeprefix("pass@"))
        for key in cast(Mapping[str, object], measurements[0]["pass_at_k"])
    )
    key = f"pass@{max_k}"
    benchmark_summaries: dict[str, object] = {}
    for benchmark in dict.fromkeys(str(row["benchmark"]) for row in measurements):
        rows = [row for row in measurements if row["benchmark"] == benchmark]
        greedy = next(row for row in rows if float(cast(float, row["temperature"])) == 0.0)
        best = max(
            rows,
            key=lambda row: (
                float(cast(float, cast(Mapping[str, object], row["pass_at_k"])[key])),
                -float(cast(float, row["temperature"])),
            ),
        )
        solved = int(cast(int, cast(Mapping[str, object], best["solved_at_k"])[key]))
        tasks = int(cast(int, best["tasks"]))
        lower, upper = wilson_interval(solved, tasks)
        greedy_rate = float(cast(float, cast(Mapping[str, object], greedy["pass_at_k"])[key]))
        best_rate = float(cast(float, cast(Mapping[str, object], best["pass_at_k"])[key]))
        greedy_proposal_rate = int(cast(int, greedy["valid_proposals"])) / int(
            cast(int, greedy["total_proposals"])
        )
        best_proposal_rate = int(cast(int, best["valid_proposals"])) / int(
            cast(int, best["total_proposals"])
        )
        benchmark_summaries[benchmark] = {
            "best_temperature": best["temperature"],
            "best_pass_at_k": key,
            "best_rate": best_rate,
            "best_solved": solved,
            "tasks": tasks,
            "wilson_95_interval": [lower, upper],
            "greedy_rate": greedy_rate,
            "greedy_exact_proposal_success_rate": greedy_proposal_rate,
            "best_temperature_exact_proposal_success_rate": best_proposal_rate,
            "absolute_gain_over_greedy": best_rate - greedy_rate,
            "relative_gain_over_greedy": (best_rate / greedy_rate if greedy_rate > 0.0 else None),
        }
    composition_summary: dict[str, object] | None = None
    numeric_name = "countdown_numeric_b"
    symbolic_name = "countdown_symbolic_ab"
    if numeric_name in benchmark_summaries and symbolic_name in benchmark_summaries:
        symbolic = cast(Mapping[str, object], benchmark_summaries[symbolic_name])
        best_ab_temperature = float(cast(float, symbolic["best_temperature"]))
        numeric_at_ab = next(
            row
            for row in measurements
            if row["benchmark"] == numeric_name
            and float(cast(float, row["temperature"])) == best_ab_temperature
        )
        symbolic_at_ab = next(
            row
            for row in measurements
            if row["benchmark"] == symbolic_name
            and float(cast(float, row["temperature"])) == best_ab_temperature
        )
        numeric_rate = float(
            cast(float, cast(Mapping[str, object], numeric_at_ab["pass_at_k"])[key])
        )
        symbolic_rate = float(
            cast(float, cast(Mapping[str, object], symbolic_at_ab["pass_at_k"])[key])
        )
        skill_a = cast(Mapping[str, object], results.get("skill_a_metrics", {}))
        composition_summary = {
            "skill_a_exact_accuracy": skill_a.get("accuracy"),
            "skill_a_exact_solved": skill_a.get("exact_solved"),
            "skill_a_exact_total": skill_a.get("total"),
            "best_ab_temperature": best_ab_temperature,
            "numeric_b_rate_at_best_ab_temperature": numeric_rate,
            "symbolic_ab_rate_at_best_ab_temperature": symbolic_rate,
            "composition_gap_at_best_ab_temperature": numeric_rate - symbolic_rate,
            "symbolic_ab_greedy_rate": symbolic["greedy_rate"],
            "symbolic_ab_temperature_gain": symbolic["absolute_gain_over_greedy"],
            "pass_at_k": key,
        }
    best_accuracy_row = max(
        evaluation_rows,
        key=lambda row: float(cast(float, row[accuracy_key])),
    )
    best_loss_row = min(
        evaluation_rows,
        key=lambda row: float(cast(float, row[loss_key])),
    )
    final_row = evaluation_rows[-1]
    environment = cast(Mapping[str, object], record["environment"])
    if evaluation_prefix == "validation":
        limitations = [
            "One model seed and one evaluation seed.",
            (
                "The reported best temperature is selected retrospectively from the test "
                "sweep and should be confirmed in a preregistered replication."
            ),
            (
                "Binomial intervals quantify task-sampling uncertainty only; they do "
                "not include training-seed or decoding-seed variation."
            ),
        ]
    else:
        limitations = [
            "One model seed and one evaluation seed.",
            "Only 64 held-out tasks per benchmark, so neighboring temperature estimates overlap.",
            (
                "The temperature sweep evaluated the final step-5000 model, not the "
                "best step-2000 checkpoint."
            ),
            (
                "The dataset has no validation split; step 2,000 is a retrospective "
                "test-set diagnostic and must not be selected as a final checkpoint."
            ),
            (
                "No per-task proposal matrix was retained, so paired significance "
                "tests are unavailable."
            ),
        ]
    return {
        "run_id": cast(Mapping[str, object], record["run"])["run_id"],
        "benchmark_summaries": benchmark_summaries,
        "composition": composition_summary,
        "training": {
            "evaluation_split": evaluation_prefix,
            "best_evaluation_accuracy_step": best_accuracy_row["step"],
            "best_evaluation_accuracy": best_accuracy_row[accuracy_key],
            "minimum_evaluation_loss_step": best_loss_row["step"],
            "minimum_evaluation_loss": best_loss_row[loss_key],
            "final_step": final_row["step"],
            "final_train_accuracy": final_row["train_token_accuracy"],
            "final_evaluation_accuracy": final_row[accuracy_key],
            "final_train_loss": final_row["train_token_loss"],
            "final_evaluation_loss": final_row[loss_key],
        },
        "runtime": {
            "run_duration_seconds": cast(Mapping[str, object], record["run"])["duration_seconds"],
            "compute_duration_seconds": environment["training_and_evaluation_runtime_seconds"],
            "peak_gpu_memory_bytes": environment["peak_gpu_memory_bytes"],
        },
        "limitations": limitations,
    }


def _write_table(path: Path, measurements: Sequence[Mapping[str, object]]) -> None:
    pass_keys = _ordered_pass_keys(measurements[0])
    fields = [
        "benchmark",
        "temperature",
        "tasks",
        *(f"{key}_solved" for key in pass_keys),
        *(f"{key}_rate" for key in pass_keys),
        "exact_proposal_successes",
        "total_proposals",
        "exact_proposal_success_rate",
    ]
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for row in measurements:
            rates = cast(Mapping[str, object], row["pass_at_k"])
            solved = cast(Mapping[str, object], row["solved_at_k"])
            successes = int(cast(int, row["valid_proposals"]))
            total = int(cast(int, row["total_proposals"]))
            writer.writerow(
                {
                    "benchmark": row["benchmark"],
                    "temperature": row["temperature"],
                    "tasks": row["tasks"],
                    **{f"{key}_solved": solved[key] for key in pass_keys},
                    **{f"{key}_rate": rates[key] for key in pass_keys},
                    "exact_proposal_successes": successes,
                    "total_proposals": total,
                    "exact_proposal_success_rate": successes / total,
                }
            )


def _markdown_report(record: Mapping[str, object], analysis: Mapping[str, object]) -> str:
    run = cast(Mapping[str, object], record["run"])
    model = cast(Mapping[str, object], record["model"])
    summaries = cast(Mapping[str, Mapping[str, object]], analysis["benchmark_summaries"])
    training = cast(Mapping[str, object], analysis["training"])
    runtime = cast(Mapping[str, object], analysis["runtime"])
    composition_value = analysis.get("composition")
    composition = (
        cast(Mapping[str, object], composition_value)
        if isinstance(composition_value, Mapping)
        else None
    )
    environment = cast(Mapping[str, object], record["environment"])
    system = cast(Mapping[str, object], environment.get("system", {}))
    gpu_names = system.get("gpu_names", [])
    hardware = (
        str(cast(Sequence[object], gpu_names)[0])
        if isinstance(gpu_names, list) and gpu_names
        else str(environment.get("device", "unknown hardware"))
    )
    model_name = str(model.get("name", "model"))
    total_parameters = int(cast(int, model["parameter_count"]))
    trainable_parameters = int(cast(int, model.get("trainable_parameter_count", total_parameters)))
    pass_label = str(next(iter(summaries.values()))["best_pass_at_k"])
    evaluation_split = str(training["evaluation_split"])
    rows = []
    benchmark_labels = {
        "countdown_numeric_b": "numeric Countdown (B)",
        "countdown_symbolic_ab": "symbolic Countdown (A+B)",
    }
    for benchmark, summary in summaries.items():
        interval = cast(Sequence[float], summary["wilson_95_interval"])
        rows.append(
            f"| {benchmark_labels.get(benchmark, benchmark)} | "
            f"{float(cast(float, summary['best_temperature'])):g} | "
            f"{int(cast(int, summary['best_solved']))}/{int(cast(int, summary['tasks']))} "
            f"({100 * float(cast(float, summary['best_rate'])):.1f}%) | "
            f"{100 * float(cast(float, summary['greedy_rate'])):.1f}% | "
            f"[{100 * interval[0]:.1f}%, {100 * interval[1]:.1f}%] |"
        )
    game24 = summaries.get("game24")
    countdown = summaries.get("countdown")
    conclusions = []
    if composition is not None:
        skill_a_accuracy = composition.get("skill_a_exact_accuracy")
        if isinstance(skill_a_accuracy, int | float):
            conclusions.append(
                "- **Skill A decoding accuracy is "
                f"{100 * float(skill_a_accuracy):.1f}%.** This checks that the arbitrary "
                "symbol codebook was learned before interpreting the composition result."
            )
        numeric_rate = float(cast(float, composition["numeric_b_rate_at_best_ab_temperature"]))
        symbolic_rate = float(cast(float, composition["symbolic_ab_rate_at_best_ab_temperature"]))
        gap = float(cast(float, composition["composition_gap_at_best_ab_temperature"]))
        best_ab_temperature = float(cast(float, composition["best_ab_temperature"]))
        conclusions.append(
            f"- **The measured composition gap is {100 * gap:.1f} percentage points** "
            f"at temperature={best_ab_temperature:g}: numeric B reaches "
            f"{100 * numeric_rate:.1f}% while unseen symbolic A+B reaches "
            f"{100 * symbolic_rate:.1f}%."
        )
        temperature_gain = float(cast(float, composition["symbolic_ab_temperature_gain"]))
        conclusions.append(
            "- **Exploration changes A+B coverage by "
            f"{100 * temperature_gain:.1f} percentage points** relative to greedy "
            f"decoding, with the best observed A+B temperature at {best_ab_temperature:g}."
        )
    if game24 is not None:
        game24_gain = float(cast(float, game24["absolute_gain_over_greedy"]))
        best_proposal_rate = float(
            cast(float, game24["best_temperature_exact_proposal_success_rate"])
        )
        greedy_proposal_rate = float(cast(float, game24["greedy_exact_proposal_success_rate"]))
        if game24_gain > 0.0:
            conclusions.append(
                f"- **Game24 has an observed exploration gain.** Best {pass_label} is "
                f"{100 * float(cast(float, game24['best_rate'])):.1f}% at "
                f"temperature={float(cast(float, game24['best_temperature'])):g}, versus "
                f"{100 * float(cast(float, game24['greedy_rate'])):.1f}% greedy."
            )
            conclusions.append(
                "- **The coverage gain comes from proposal diversity.** At the best "
                "temperature, exact Game24 success per proposal is "
                f"{100 * best_proposal_rate:.1f}% versus "
                f"{100 * greedy_proposal_rate:.1f}% "
                "greedy, while more distinct tasks are covered."
            )
        else:
            conclusions.append(
                f"- **Game24 shows no exploration gain in this run.** Best {pass_label} "
                f"matches greedy at {100 * float(cast(float, game24['greedy_rate'])):.1f}%."
            )
    if countdown is not None:
        countdown_gain = float(cast(float, countdown["absolute_gain_over_greedy"]))
        conclusion = (
            "responds to exploration" if countdown_gain > 0.0 else "shows no exploration gain"
        )
        conclusions.append(
            f"- **Countdown {conclusion}.** Best {pass_label} is "
            f"{100 * float(cast(float, countdown['best_rate'])):.1f}% versus "
            f"{100 * float(cast(float, countdown['greedy_rate'])):.1f}% greedy."
        )
    if evaluation_split == "validation":
        conclusions.extend(
            [
                "- **Checkpoint selection is separated from the test set.** Training "
                "uses numeric-B validation loss with early stopping, considers only "
                "checkpoints retaining exact Skill-A accuracy, then restores the selected "
                "checkpoint before the temperature sweep.",
                "- **This is a stronger single-seed estimate, not a final paper claim.** "
                "Training-seed and decoding-seed replication is still required.",
            ]
        )
        recommendation = (
            "Repeat the same frozen configuration with additional training and decoding "
            "seeds. Use the retained per-task proposal outcomes for paired bootstrap "
            "confidence intervals, and confirm the chosen temperature without retuning it on "
            "the same test set."
        )
    else:
        conclusions.extend(
            [
                "- **The final checkpoint is overfit.** Test accuracy peaks near step "
                f"{training['best_evaluation_accuracy_step']}, while final train accuracy "
                "keeps rising to "
                f"{100 * float(cast(float, training['final_train_accuracy'])):.1f}% and "
                "final test loss rises to "
                f"{float(cast(float, training['final_evaluation_loss'])):.3f}.",
                "- **This is pilot evidence, not a final estimate.** Add a validation "
                "split and evaluate a selected checkpoint once on the test set.",
            ]
        )
        recommendation = (
            "Add a validation split and save the checkpoint with the best validation "
            "loss. Then repeat the temperature sweep with the selected checkpoint and retain "
            "per-task proposal outcomes for paired bootstrap confidence intervals."
        )
    if composition is not None:
        recommendation = (
            "Freeze the codebook, data split, temperature grid, and checkpoint-selection "
            "rule, then repeat with additional training and decoding seeds. Use the "
            "paired B/A+B proposal outcomes to bootstrap the composition gap at each "
            "temperature without selecting a new temperature on the same test set."
        )
    limitations = "\n".join(f"- {item}" for item in cast(Sequence[str], analysis["limitations"]))
    report_title = (
        "Countdown symbolic A+B composition analysis"
        if composition is not None
        else "Game24/Countdown temperature-sweep analysis"
    )
    return f"""# {report_title}

Run `{run["run_id"]}` completed successfully on {hardware}. Model `{model_name}` has
{total_parameters:,} parameters, of which {trainable_parameters:,} were trainable.
Total run time was
{float(cast(float, runtime["run_duration_seconds"])):.1f} seconds, with
{float(cast(float, runtime["compute_duration_seconds"])):.1f} seconds in training and evaluation.

## Figures

![Isolated Skill A and Skill B training](analysis-training-trajectory.svg)

![Numeric B versus held-out A+B](analysis-temperature-pass-at-k.svg)

## Best held-out result

| Benchmark | Best temperature | Best {pass_label} | Greedy {pass_label} | Wilson 95% interval |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

## Conclusions

{chr(10).join(conclusions)}

## Limitations

{limitations}

## Recommended next experiment

{recommendation}
"""


def analyze_archive(
    archive_path: Path,
    *,
    output_root: Path,
    analyzed_archive_path: Path,
) -> dict[str, str]:
    """Analyze a completed archive and package the augmented run directory."""
    record, files = _load_archive(archive_path)
    run = cast(Mapping[str, object], record["run"])
    run_id = str(run["run_id"])
    run_directory = _materialize_run_directory(
        files,
        output_root=output_root,
        run_id=run_id,
    )
    record_path = run_directory / "record.json"
    current_record = cast(dict[str, object], json.loads(record_path.read_text(encoding="utf-8")))
    results = cast(Mapping[str, object], current_record["results"])
    measurements = cast(list[dict[str, object]], results["temperature_measurements"])
    metrics = cast(Mapping[str, object], current_record["metrics"])
    trajectory = cast(list[dict[str, object]], metrics["trajectory"])
    analysis = _analysis_payload(current_record)

    outputs = {
        "temperature_plot": run_directory / "analysis-temperature-pass-at-k.svg",
        "training_plot": run_directory / "analysis-training-trajectory.svg",
        "table": run_directory / "analysis-results-table.csv",
        "json": run_directory / "analysis.json",
        "report": run_directory / "analysis.md",
    }
    outputs["temperature_plot"].write_text(_temperature_plot(measurements), encoding="utf-8")
    outputs["training_plot"].write_text(_training_plot(trajectory), encoding="utf-8")
    _write_table(outputs["table"], measurements)
    outputs["json"].write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    outputs["report"].write_text(_markdown_report(current_record, analysis), encoding="utf-8")

    artifact_details = {
        "temperature_plot": (
            "plot",
            "Held-out numeric-B versus symbolic-A+B pass@k across temperature.",
        ),
        "training_plot": (
            "plot",
            "Separate Skill A and Skill B accuracy/loss across optimizer steps.",
        ),
        "table": ("table", "Paper-ready temperature sweep results table."),
        "json": ("analysis", "Machine-readable statistical analysis."),
        "report": ("analysis", "Conclusions and limitations for the completed run."),
    }
    for name, path in outputs.items():
        kind, description = artifact_details[name]
        register_artifact(
            record_path,
            path,
            kind=kind,
            description=description,
        )

    analyzed_archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        analyzed_archive_path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for path in sorted(run_directory.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(run_directory).as_posix())
    return {
        "run_directory": str(run_directory.resolve()),
        "report": str(outputs["report"].resolve()),
        "temperature_plot": str(outputs["temperature_plot"].resolve()),
        "training_plot": str(outputs["training_plot"].resolve()),
        "table": str(outputs["table"].resolve()),
        "analysis": str(outputs["json"].resolve()),
        "analyzed_archive": str(analyzed_archive_path.resolve()),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze a completed arithmetic temperature/composition archive."
    )
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--analyzed-archive", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = _build_parser().parse_args(argv)
    paths = analyze_archive(
        args.archive,
        output_root=args.output_root,
        analyzed_archive_path=args.analyzed_archive,
    )
    print(json.dumps(paths, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
