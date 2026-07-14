"""Exact Game24 solver using Fraction arithmetic."""

from __future__ import annotations

from functools import lru_cache
from fractions import Fraction
from itertools import combinations_with_replacement

TARGET = Fraction(24, 1)


def solve_game24(numbers):
    """Return one exact expression for numbers, or None if unsolvable."""
    canonical = tuple(sorted(int(number) for number in numbers))
    return _solve_canonical(canonical)


def is_solvable(numbers):
    return solve_game24(numbers) is not None


def enumerate_solvable_tuples(value_min=1, value_max=10):
    return [
        tuple(numbers)
        for numbers in combinations_with_replacement(range(value_min, value_max + 1), 4)
        if solve_game24(numbers)
    ]


@lru_cache(maxsize=None)
def _solve_canonical(numbers):
    items = tuple((Fraction(number), str(number)) for number in numbers)
    return _search(items)


def _search(items):
    if len(items) == 1:
        return items[0][1] if items[0][0] == TARGET else None

    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            rest = tuple(item for idx, item in enumerate(items) if idx not in (i, j))
            for value, expr in _combine(items[i], items[j]):
                found = _search(rest + ((value, expr),))
                if found:
                    return found
    return None


def _combine(left, right):
    a, a_expr = left
    b, b_expr = right

    yield a + b, f"({a_expr} + {b_expr})"
    yield a * b, f"({a_expr} * {b_expr})"
    yield a - b, f"({a_expr} - {b_expr})"
    yield b - a, f"({b_expr} - {a_expr})"
    if b != 0:
        yield a / b, f"({a_expr} / {b_expr})"
    if a != 0:
        yield b / a, f"({b_expr} / {a_expr})"
