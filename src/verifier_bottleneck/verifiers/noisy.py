"""A binary verifier with configurable true- and false-positive rates."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class VerifierParameters:
    """Verifier operating point.

    ``alpha`` is ``P(V=1 | c=1)`` and ``beta`` is ``P(V=1 | c=0)``.
    """

    alpha: float
    beta: float

    def __post_init__(self) -> None:
        for name, value in (("alpha", self.alpha), ("beta", self.beta)):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1; received {value}")


@dataclass(frozen=True)
class VerifierMeasurement:
    """Theoretical and empirical verifier-filtered accuracy."""

    alpha: float
    beta: float
    base_accuracy: float
    predicted_accepted_accuracy: float | None
    empirical_accepted_accuracy: float | None
    predicted_acceptance_rate: float
    empirical_acceptance_rate: float
    accepted_count: int
    proposal_count: int

    def to_dict(self) -> dict[str, float | int | None]:
        """Return a JSON-serializable representation."""
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "base_accuracy": self.base_accuracy,
            "predicted_accepted_accuracy": self.predicted_accepted_accuracy,
            "empirical_accepted_accuracy": self.empirical_accepted_accuracy,
            "predicted_acceptance_rate": self.predicted_acceptance_rate,
            "empirical_acceptance_rate": self.empirical_acceptance_rate,
            "accepted_count": self.accepted_count,
            "proposal_count": self.proposal_count,
        }


def predicted_accepted_accuracy(
    base_accuracy: float,
    parameters: VerifierParameters,
) -> float | None:
    """Compute ``P(c=1 | V=1)`` from ``a``, ``alpha``, and ``beta``.

    The denominator is zero when the verifier never accepts at the supplied
    base accuracy. In that case accepted accuracy is undefined and ``None`` is
    returned.
    """
    if not 0.0 <= base_accuracy <= 1.0:
        raise ValueError(
            f"base_accuracy must be between 0 and 1; received {base_accuracy}"
        )
    denominator = (
        parameters.alpha * base_accuracy
        + parameters.beta * (1.0 - base_accuracy)
    )
    if denominator == 0.0:
        return None
    return parameters.alpha * base_accuracy / denominator


def measure_noisy_verifier(
    correctness: Sequence[bool],
    parameters: VerifierParameters,
    *,
    repetitions: int,
    seed: int,
) -> VerifierMeasurement:
    """Apply the noisy verifier repeatedly using an explicit deterministic seed."""
    if not correctness:
        raise ValueError("correctness must contain at least one proposal")
    if repetitions < 1:
        raise ValueError(f"repetitions must be at least 1; received {repetitions}")

    base_accuracy = sum(correctness) / len(correctness)
    proposal_count = len(correctness) * repetitions
    accepted_count = 0
    correct_accepted_count = 0
    generator = random.Random(seed)

    for _ in range(repetitions):
        for is_correct in correctness:
            acceptance_probability = (
                parameters.alpha if is_correct else parameters.beta
            )
            if generator.random() < acceptance_probability:
                accepted_count += 1
                if is_correct:
                    correct_accepted_count += 1

    predicted_acceptance_rate = (
        parameters.alpha * base_accuracy
        + parameters.beta * (1.0 - base_accuracy)
    )
    empirical_accepted_accuracy = (
        correct_accepted_count / accepted_count if accepted_count else None
    )
    return VerifierMeasurement(
        alpha=parameters.alpha,
        beta=parameters.beta,
        base_accuracy=base_accuracy,
        predicted_accepted_accuracy=predicted_accepted_accuracy(
            base_accuracy, parameters
        ),
        empirical_accepted_accuracy=empirical_accepted_accuracy,
        predicted_acceptance_rate=predicted_acceptance_rate,
        empirical_acceptance_rate=accepted_count / proposal_count,
        accepted_count=accepted_count,
        proposal_count=proposal_count,
    )

