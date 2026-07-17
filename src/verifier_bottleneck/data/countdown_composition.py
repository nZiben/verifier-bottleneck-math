"""Deterministic symbol-decoding plus Countdown composition data."""

from __future__ import annotations

import random
from dataclasses import dataclass
from fractions import Fraction

from verifier_bottleneck.experiment_tracking import sha256_json

COUNTDOWN = "countdown"
OPERATORS = ("+", "-", "*", "/")
COUNTDOWN_POOL = tuple(range(1, 11)) * 2 + (25, 50, 75, 100)
COUNTDOWN_VALUES = tuple(sorted(set(COUNTDOWN_POOL)))
NUMERIC_B = "countdown_numeric_b"
SYMBOLIC_AB = "countdown_symbolic_ab"


@dataclass(frozen=True)
class ArithmeticPuzzle:
    """One guaranteed-solvable Countdown puzzle."""

    numbers: tuple[int, ...]
    target: int
    solution: tuple[str, ...]
    benchmark: str = COUNTDOWN

    def key(self) -> tuple[object, ...]:
        """Return the task identity without its reference solution."""
        return (self.numbers, self.target)

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-compatible row."""
        return {
            "numbers": list(self.numbers),
            "target": self.target,
            "solution": list(self.solution),
        }


@dataclass(frozen=True)
class _Expression:
    value: Fraction
    postfix: tuple[str, ...]


def _countdown_candidate(
    random_source: random.Random,
    *,
    minimum_target: int,
    maximum_target: int,
) -> ArithmeticPuzzle | None:
    sampled = random_source.sample(range(len(COUNTDOWN_POOL)), 6)
    numbers = tuple(sorted(COUNTDOWN_POOL[index] for index in sampled))
    expressions = [
        _Expression(Fraction(number), (f"P{index}",))
        for index, number in enumerate(numbers)
    ]
    while len(expressions) > 1:
        left_index, right_index = sorted(
            random_source.sample(range(len(expressions)), 2)
        )
        right = expressions.pop(right_index)
        left = expressions.pop(left_index)
        candidates = [
            _Expression(
                left.value + right.value,
                left.postfix + right.postfix + ("+",),
            ),
            _Expression(
                left.value * right.value,
                left.postfix + right.postfix + ("*",),
            ),
        ]
        if left.value > right.value:
            candidates.append(
                _Expression(
                    left.value - right.value,
                    left.postfix + right.postfix + ("-",),
                )
            )
        elif right.value > left.value:
            candidates.append(
                _Expression(
                    right.value - left.value,
                    right.postfix + left.postfix + ("-",),
                )
            )
        if right.value and left.value % right.value == 0:
            candidates.append(
                _Expression(
                    left.value / right.value,
                    left.postfix + right.postfix + ("/",),
                )
            )
        if left.value and right.value % left.value == 0:
            candidates.append(
                _Expression(
                    right.value / left.value,
                    right.postfix + left.postfix + ("/",),
                )
            )
        bounded = [candidate for candidate in candidates if candidate.value <= 100_000]
        expressions.append(random_source.choice(bounded))
    result = expressions[0]
    if result.value.denominator != 1:
        return None
    target = int(result.value)
    if not minimum_target <= target <= maximum_target:
        return None
    return ArithmeticPuzzle(numbers=numbers, target=target, solution=result.postfix)


def make_countdown_examples(
    *,
    train_examples: int,
    test_examples: int,
    minimum_target: int,
    maximum_target: int,
    seed: int,
) -> tuple[tuple[ArithmeticPuzzle, ...], tuple[ArithmeticPuzzle, ...]]:
    """Generate unique, guaranteed-solvable Countdown tasks locally."""
    if train_examples < 1 or test_examples < 1:
        raise ValueError("Countdown train and test counts must be positive")
    if minimum_target < 1 or maximum_target < minimum_target:
        raise ValueError("Countdown target range is invalid")
    requested = train_examples + test_examples
    random_source = random.Random(seed)
    examples: list[ArithmeticPuzzle] = []
    keys: set[tuple[object, ...]] = set()
    maximum_attempts = max(10_000, requested * 500)
    for _ in range(maximum_attempts):
        if len(examples) >= requested:
            break
        candidate = _countdown_candidate(
            random_source,
            minimum_target=minimum_target,
            maximum_target=maximum_target,
        )
        if candidate is None or candidate.key() in keys:
            continue
        keys.add(candidate.key())
        examples.append(candidate)
    if len(examples) < requested:
        raise ValueError(
            f"generated only {len(examples)} unique Countdown examples; "
            "reduce the requested size or widen the target range"
        )
    random_source.shuffle(examples)
    return tuple(examples[:train_examples]), tuple(examples[train_examples:])


def verify_postfix(puzzle: ArithmeticPuzzle, tokens: tuple[str, ...]) -> bool:
    """Verify a Countdown postfix proposal with exact arithmetic."""
    stack: list[Fraction] = []
    used: set[int] = set()
    for token in tokens:
        if token.startswith("P") and token[1:].isdigit():
            index = int(token[1:])
            if index >= len(puzzle.numbers) or index in used:
                return False
            used.add(index)
            stack.append(Fraction(puzzle.numbers[index]))
            continue
        if token not in OPERATORS or len(stack) < 2:
            return False
        right = stack.pop()
        left = stack.pop()
        if token == "+":
            result = left + right
        elif token == "-":
            result = left - right
        elif token == "*":
            result = left * right
        else:
            if right == 0:
                return False
            result = left / right
        if result <= 0 or result.denominator != 1:
            return False
        stack.append(result)
    return len(stack) == 1 and len(used) >= 2 and stack[0] == puzzle.target


@dataclass(frozen=True)
class SymbolDecodingExample:
    """One isolated A example mapping an arbitrary symbol to its number."""

    symbol: str
    value: int

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-compatible row."""
        return {"symbol": self.symbol, "value": self.value}


@dataclass(frozen=True)
class CountdownCompositionSplit:
    """Separate A and B training data plus paired B and unseen A+B tests."""

    decode_examples: tuple[SymbolDecodingExample, ...]
    numeric_train: tuple[ArithmeticPuzzle, ...]
    numeric_validation: tuple[ArithmeticPuzzle, ...]
    numeric_test: tuple[ArithmeticPuzzle, ...]
    symbolic_test: tuple[ArithmeticPuzzle, ...]

    @property
    def symbol_to_value(self) -> dict[str, int]:
        """Return the learned codebook."""
        return {example.symbol: example.value for example in self.decode_examples}

    @property
    def value_to_symbol(self) -> dict[int, str]:
        """Return the inverse codebook used to render unseen A+B prompts."""
        return {example.value: example.symbol for example in self.decode_examples}

    def fingerprints(self) -> dict[str, str]:
        """Fingerprint every logical split and the paired test identities."""
        decode_rows = [example.to_dict() for example in self.decode_examples]
        train_rows = [example.to_dict() for example in self.numeric_train]
        validation_rows = [example.to_dict() for example in self.numeric_validation]
        test_rows = [example.to_dict() for example in self.numeric_test]
        symbolic_rows = [
            {
                "symbols": [self.value_to_symbol[number] for number in example.numbers],
                "target": example.target,
                "solution": list(example.solution),
            }
            for example in self.symbolic_test
        ]
        return {
            "fingerprint": sha256_json(
                {
                    "decode_a": decode_rows,
                    "numeric_b_train": train_rows,
                    "numeric_b_validation": validation_rows,
                    "numeric_b_test": test_rows,
                    "symbolic_ab_test": symbolic_rows,
                }
            ),
            "decode_a_fingerprint": sha256_json(decode_rows),
            "numeric_b_train_fingerprint": sha256_json(train_rows),
            "numeric_b_validation_fingerprint": sha256_json(validation_rows),
            "numeric_b_test_fingerprint": sha256_json(test_rows),
            "symbolic_ab_test_fingerprint": sha256_json(symbolic_rows),
        }


def make_symbol_codebook(*, seed: int) -> tuple[SymbolDecodingExample, ...]:
    """Create a seeded, non-monotonic bijection over the Countdown number pool."""
    values = list(COUNTDOWN_VALUES)
    random.Random(seed).shuffle(values)
    return tuple(
        SymbolDecodingExample(symbol=f"S{index}", value=value)
        for index, value in enumerate(values)
    )


def make_countdown_composition_split(
    *,
    train_examples: int,
    validation_examples: int,
    test_examples: int,
    minimum_target: int,
    maximum_target: int,
    seed: int,
    symbol_seed: int,
) -> CountdownCompositionSplit:
    """Build isolated A/B training and a paired, never-trained A+B test."""
    train_and_validation, numeric_test = make_countdown_examples(
        train_examples=train_examples + validation_examples,
        test_examples=test_examples,
        minimum_target=minimum_target,
        maximum_target=maximum_target,
        seed=seed,
    )
    numeric_train = train_and_validation[:train_examples]
    numeric_validation = train_and_validation[train_examples:]
    decode_examples = make_symbol_codebook(seed=symbol_seed)
    return CountdownCompositionSplit(
        decode_examples=decode_examples,
        numeric_train=numeric_train,
        numeric_validation=numeric_validation,
        numeric_test=numeric_test,
        symbolic_test=numeric_test,
    )
