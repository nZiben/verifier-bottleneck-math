"""Deterministic modular-addition tables with abstract-symbol tokenization."""

from __future__ import annotations

import random
from dataclasses import dataclass
from hashlib import sha256


@dataclass(frozen=True)
class ModularAdditionExample:
    """One equation ``left + right = target (mod p)``."""

    left: int
    right: int
    target: int

    def input_tokens(self, modulus: int) -> tuple[int, int, int, int]:
        """Encode operands with dedicated ``+`` and ``=`` tokens."""
        return (self.left, modulus, self.right, modulus + 1)


@dataclass(frozen=True)
class ModularAdditionSplit:
    """A fixed train/test partition of the complete operation table."""

    modulus: int
    train: tuple[ModularAdditionExample, ...]
    test: tuple[ModularAdditionExample, ...]

    def fingerprints(self) -> dict[str, str]:
        """Return stable hashes for the full split and each partition."""
        train_fingerprint = _fingerprint_examples(self.train)
        test_fingerprint = _fingerprint_examples(self.test)
        combined = sha256(
            f"{self.modulus}:{train_fingerprint}:{test_fingerprint}".encode()
        ).hexdigest()
        return {
            "algorithm": "sha256",
            "fingerprint": combined,
            "train_fingerprint": train_fingerprint,
            "test_fingerprint": test_fingerprint,
        }


def _fingerprint_examples(examples: tuple[ModularAdditionExample, ...]) -> str:
    digest = sha256()
    for example in examples:
        digest.update(
            f"{example.left},{example.right},{example.target}\n".encode("ascii")
        )
    return digest.hexdigest()


def is_prime(value: int) -> bool:
    """Return whether ``value`` is prime."""
    if value < 2:
        return False
    if value == 2:
        return True
    if value % 2 == 0:
        return False
    divisor = 3
    while divisor * divisor <= value:
        if value % divisor == 0:
            return False
        divisor += 2
    return True


def make_modular_addition_split(
    *,
    modulus: int,
    train_fraction: float,
    seed: int,
) -> ModularAdditionSplit:
    """Create and deterministically split the complete addition table modulo ``p``."""
    if not is_prime(modulus):
        raise ValueError(f"modulus must be prime and at least 2; received {modulus}")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(
            f"train_fraction must be strictly between 0 and 1; received {train_fraction}"
        )

    examples = [
        ModularAdditionExample(left, right, (left + right) % modulus)
        for left in range(modulus)
        for right in range(modulus)
    ]
    random.Random(seed).shuffle(examples)
    train_size = int(len(examples) * train_fraction)
    if train_size == 0 or train_size == len(examples):
        raise ValueError(
            "train_fraction leaves an empty train or test split for "
            f"modulus={modulus}; received {train_fraction}"
        )
    return ModularAdditionSplit(
        modulus=modulus,
        train=tuple(examples[:train_size]),
        test=tuple(examples[train_size:]),
    )
