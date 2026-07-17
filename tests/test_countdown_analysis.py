from __future__ import annotations

import xml.etree.ElementTree as ET

from verifier_bottleneck.analysis.arithmetic_results import (
    _analysis_payload,
    _markdown_report,
    _temperature_plot,
    _training_plot,
    wilson_interval,
)


def test_wilson_interval_contains_observed_rate() -> None:
    lower, upper = wilson_interval(79, 128)
    assert lower < 79 / 128 < upper


def test_composition_plot_supports_all_five_pass_curves() -> None:
    rates = {
        "pass@1": 0.01,
        "pass@128": 0.2,
        "pass@16": 0.05,
        "pass@256": 0.3,
        "pass@64": 0.1,
    }
    measurements = [
        {"benchmark": benchmark, "temperature": temperature, "pass_at_k": rates}
        for benchmark in ("countdown_numeric_b", "countdown_symbolic_ab")
        for temperature in (0.0, 1.0)
    ]

    svg = _temperature_plot(measurements)

    positions = [svg.index(key) for key in ("pass@1", "pass@16", "pass@64", "pass@128", "pass@256")]
    assert positions == sorted(positions)
    assert "Numeric Countdown (B)" in svg
    assert "Symbolic Countdown (A+B)" in svg
    assert "Held-out tasks solved (%)" in svg
    assert "unseen symbolic A+B" in svg
    ET.fromstring(svg)


def test_composition_training_plot_separates_skill_a_and_skill_b() -> None:
    phase_a = {
        "step": 0,
        "phase": "skill_a_pretraining",
        "decode_a_token_loss": 2.5,
        "decode_a_token_accuracy": 0.1,
        "decode_a_exact_accuracy": 0.0,
        "is_best_checkpoint": False,
    }
    phase_b = [
        {
            "step": step,
            "phase": "skill_b_with_a_replay",
            "train_token_loss": 2.0 - step / 1000,
            "train_token_accuracy": step / 1000,
            "validation_token_loss": 2.1 - step / 1000,
            "validation_token_accuracy": step / 1200,
            "decode_a_token_loss": 1.5 - step / 1000,
            "decode_a_token_accuracy": step / 1000,
            "decode_a_exact_accuracy": step / 1000,
            "numeric_b_train_loss": 2.0 - step / 1000,
            "numeric_b_train_accuracy": step / 1200,
            "numeric_b_validation_loss": 2.1 - step / 1000,
            "numeric_b_validation_accuracy": step / 1400,
            "is_best_checkpoint": step == 500,
        }
        for step in (100, 350, 600)
    ]
    trajectory = [phase_a, *phase_b]

    svg = _training_plot(trajectory)

    assert "Skill A: symbol decoding accuracy" in svg
    assert "Skill B: numeric Countdown accuracy" in svg
    assert "Optimizer step" in svg
    assert "selected checkpoint" in svg
    assert "symbolic A+B composition is held out" in svg
    ET.fromstring(svg)


def test_composition_report_calculates_paired_gap() -> None:
    def measurement(benchmark: str, temperature: float, solved: int) -> dict[str, object]:
        return {
            "benchmark": benchmark,
            "temperature": temperature,
            "tasks": 10,
            "valid_proposals": solved,
            "total_proposals": 10,
            "solved_at_k": {"pass@1": solved},
            "pass_at_k": {"pass@1": solved / 10},
        }

    record = {
        "run": {"run_id": "test", "duration_seconds": 2.0},
        "model": {
            "name": "nano_countdown_composition_transformer",
            "parameter_count": 100,
            "trainable_parameter_count": 100,
        },
        "environment": {
            "device": "cpu",
            "training_and_evaluation_runtime_seconds": 1.0,
            "peak_gpu_memory_bytes": 0,
            "system": {"gpu_names": []},
        },
        "metrics": {
            "trajectory": [
                {
                    "step": 0,
                    "phase": "skill_a_pretraining",
                    "decode_a_token_loss": 3.0,
                    "decode_a_token_accuracy": 0.0,
                    "decode_a_exact_accuracy": 0.0,
                    "is_best_checkpoint": False,
                },
                {
                    "step": 50,
                    "phase": "skill_b_with_a_replay",
                    "train_token_loss": 2.0,
                    "train_token_accuracy": 0.1,
                    "validation_token_loss": 2.1,
                    "validation_token_accuracy": 0.1,
                },
                {
                    "step": 250,
                    "phase": "skill_b_with_a_replay",
                    "train_token_loss": 1.0,
                    "train_token_accuracy": 0.5,
                    "validation_token_loss": 1.1,
                    "validation_token_accuracy": 0.4,
                },
            ]
        },
        "results": {
            "skill_a_metrics": {"exact_solved": 14, "total": 14, "accuracy": 1.0},
            "temperature_measurements": [
                measurement("countdown_numeric_b", 0.0, 5),
                measurement("countdown_numeric_b", 0.3, 8),
                measurement("countdown_symbolic_ab", 0.0, 1),
                measurement("countdown_symbolic_ab", 0.3, 3),
            ],
        },
    }

    analysis = _analysis_payload(record)
    composition = analysis["composition"]
    assert isinstance(composition, dict)
    assert composition["composition_gap_at_best_ab_temperature"] == 0.5
    report = _markdown_report(record, analysis)
    assert "Countdown symbolic A+B composition analysis" in report
    assert "50.0 percentage points" in report
