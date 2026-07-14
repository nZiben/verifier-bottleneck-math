"""Perfect checker for numeric and symbolic Game24 expressions."""

from __future__ import annotations

import ast
import re
from collections import Counter
from fractions import Fraction

from .symbols import SYMBOL_TO_NUMBER, symbols_to_numbers

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
SPAN_RE = re.compile(r"[A-Za-z0-9_()+\-*/\s]+")


class CheckError(ValueError):
    pass


def check_game24(text, numbers=None, symbols=None, symbol_mapping=None):
    mapping = symbol_mapping or SYMBOL_TO_NUMBER
    target_numbers = [int(number) for number in (numbers or symbols_to_numbers(symbols, mapping))]
    target_symbols = list(symbols or [])

    for expression in candidate_expressions(text):
        parsed = _try_parse(expression)
        if parsed is None:
            continue
        try:
            value, number_leaves, symbol_leaves = _eval_node(parsed.body, mapping)
            usage_error = _usage_error(number_leaves, symbol_leaves, target_numbers, target_symbols, mapping)
            if usage_error:
                return _result(False, usage_error, expression, value)
            if value != Fraction(24, 1):
                return _result(False, "expression does not evaluate to 24", expression, value)
            return _result(True, "correct", expression, value)
        except CheckError as exc:
            return _result(False, str(exc), expression, None)

    return _result(False, "no parseable arithmetic expression found", None, None)


def candidate_expressions(text):
    text = text or ""
    tagged = ANSWER_RE.findall(text)
    for expression in reversed(tagged):
        yield expression.strip()

    for line in reversed(text.splitlines() or [text]):
        line = re.sub(r"^\s*(answer|expression)\s*:\s*", "", line.strip(), flags=re.IGNORECASE)
        for match in reversed(list(SPAN_RE.finditer(line))):
            candidate = match.group(0).strip()
            if _looks_like_expression(candidate):
                yield candidate


def _looks_like_expression(candidate):
    return bool(candidate and re.search(r"[A-Za-z0-9_]", candidate) and re.search(r"[+\-*/]", candidate))


def _try_parse(expression):
    try:
        return ast.parse(expression, mode="eval")
    except SyntaxError:
        return None


def _eval_node(node, mapping):
    if isinstance(node, ast.BinOp):
        left, left_numbers, left_symbols = _eval_node(node.left, mapping)
        right, right_numbers, right_symbols = _eval_node(node.right, mapping)
        if isinstance(node.op, ast.Add):
            value = left + right
        elif isinstance(node.op, ast.Sub):
            value = left - right
        elif isinstance(node.op, ast.Mult):
            value = left * right
        elif isinstance(node.op, ast.Div):
            if right == 0:
                raise CheckError("division by zero")
            value = left / right
        else:
            raise CheckError("unsupported operator")
        return value, left_numbers + right_numbers, left_symbols + right_symbols

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int):
            raise CheckError("only integer literals are allowed")
        return Fraction(node.value, 1), [int(node.value)], []

    if isinstance(node, ast.Name):
        if node.id not in mapping:
            raise CheckError(f"unknown symbol: {node.id}")
        return Fraction(mapping[node.id], 1), [], [node.id]

    raise CheckError("unsupported expression syntax")


def _usage_error(number_leaves, symbol_leaves, target_numbers, target_symbols, mapping):
    if symbol_leaves and not target_symbols:
        return "symbols are not allowed for this numeric task"

    if target_symbols and symbol_leaves:
        extra = Counter(symbol_leaves) - Counter(target_symbols)
        if extra:
            return "expression uses symbols outside the prompt"

    decoded_symbol_values = [mapping[symbol] for symbol in symbol_leaves]
    used_numbers = number_leaves + decoded_symbol_values
    if Counter(used_numbers) != Counter(target_numbers):
        return "expression does not use exactly the given values once each"
    return None


def _result(is_correct, reason, expression, value):
    return {
        "is_correct": bool(is_correct),
        "reason": reason,
        "extracted_expression": expression,
        "value": str(value) if value is not None else None,
    }
