"""Canonical result payload shared by Countdown training backends."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

from verifier_bottleneck.data.countdown_composition import CountdownCompositionSplit


def build_countdown_training_payload(
    *,
    split: CountdownCompositionSplit,
    environment_updates: Mapping[str, object],
    model: Mapping[str, object],
    optimization: Mapping[str, object],
    trajectory: Sequence[dict[str, object]],
    selected_metrics: Mapping[str, object],
    measurements: Sequence[dict[str, object]],
    stopped_early: bool,
) -> dict[str, object]:
    """Assemble the stable recorder payload for any Countdown model backend."""
    final_trajectory = trajectory[-1]
    decode_solved = int(cast(int, selected_metrics["decode_a_exact_solved"]))
    decode_total = int(cast(int, selected_metrics["decode_a_exact_total"]))
    return {
        "environment_updates": dict(environment_updates),
        "dataset": {
            "name": "countdown_symbol_decoding_composition",
            "train_examples": len(split.decode_examples) + len(split.numeric_train),
            "validation_examples": len(split.decode_examples) + len(split.numeric_validation),
            "test_examples": len(split.numeric_test) + len(split.symbolic_test),
            "decode_a_examples": len(split.decode_examples),
            "numeric_b_train_examples": len(split.numeric_train),
            "numeric_b_validation_examples": len(split.numeric_validation),
            "numeric_b_test_examples": len(split.numeric_test),
            "symbolic_ab_test_examples": len(split.symbolic_test),
            "paired_b_ab_test_tasks": True,
            "symbolic_ab_seen_during_training": False,
            **split.fingerprints(),
            "downloads": False,
        },
        "model": dict(model),
        "optimization": dict(optimization),
        "definitions": {
            "skill_a": "Decode a seeded arbitrary S-token into its numeric value.",
            "skill_b": "Solve numeric Countdown with postfix pointer actions.",
            "composition_ab": (
                "Solve symbolic Countdown by combining separately trained decoding and "
                "numeric-solving skills; symbolic Countdown is absent from training and "
                "checkpoint selection."
            ),
            "temperature": (
                "At T=0 choose the highest-logit grammar-legal action. At T>0 sample "
                "grammar-legal actions from softmax(legal logits / T)."
            ),
            "pass_at_k": "Fraction of paired held-out tasks solved among k proposals.",
            "paired_randomness": (
                "Numeric B and symbolic A+B use identical sampling uniforms for each "
                "temperature and paired task/proposal position."
            ),
        },
        "trajectory": list(trajectory),
        "results": {
            "final_metrics": {
                **selected_metrics,
                "last_training_step": final_trajectory["step"],
                "stopped_early": stopped_early,
                "temperature_measurements": list(measurements),
            },
            "best_metrics": dict(selected_metrics),
            "skill_a_metrics": {
                "exact_solved": decode_solved,
                "total": decode_total,
                "accuracy": decode_solved / decode_total,
            },
            "temperature_measurements": list(measurements),
            "selected_checkpoint_metrics": dict(selected_metrics),
            "last_training_metrics": final_trajectory,
            "evaluation_checkpoint_count": len(trajectory),
        },
    }
