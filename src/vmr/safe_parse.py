"""Whitelisted parser for model-written answers.

Model output is untrusted, and `sympy.sympify` / `parse_expr` call `eval`. This module walks the
Python AST instead and builds SymPy objects only from an explicit list of names and operators, so
an answer such as `__import__('os').system(...)` is a parse failure rather than code execution.
"""

from __future__ import annotations

import ast

import sympy as sp

X = sp.Symbol("x", positive=True)  # Every problem states its domain, and all domains are in x > 0.

MAX_CHARS = 20000
MAX_NODES = 20000
MAX_INT_EXPONENT = 64

FUNCTIONS = {
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "cot": sp.cot, "sec": sp.sec, "csc": sp.csc,
    "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
    "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
    "asinh": sp.asinh, "acosh": sp.acosh, "atanh": sp.atanh,
    "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt, "Abs": sp.Abs,
}
CONSTANTS = {"pi": sp.pi, "E": sp.E}


class ParseFailure(ValueError):
    """The answer is not in the required grammar. This is a format failure, never a wrong answer."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _parse_tree(text: str, trusted: bool = False) -> ast.expr:
    text = text.strip()
    if not text:
        raise ParseFailure("empty_answer")
    if len(text) > MAX_CHARS and not trusted:
        raise ParseFailure("answer_too_long")
    try:
        tree = ast.parse(text, mode="eval")
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        raise ParseFailure("syntax_error") from None
    if sum(1 for _ in ast.walk(tree)) > MAX_NODES and not trusted:
        raise ParseFailure("answer_too_long")
    return tree.body


def _number(node: ast.Constant) -> sp.Expr:
    value = node.value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ParseFailure("non_numeric_literal")
    if isinstance(value, int):
        return sp.Integer(value)
    # A decimal is read as the exact rational it spells, so 0.5 is 1/2 and 0.333 is not 1/3.
    return sp.Rational(repr(value))


def _guard_power(base: sp.Expr, exponent: sp.Expr) -> None:
    if exponent.is_number and abs(exponent) > MAX_INT_EXPONENT:
        raise ParseFailure("exponent_too_large")
    if base.is_number and base.is_Integer and abs(base) > 10**6 and exponent.is_number and exponent > 8:
        raise ParseFailure("exponent_too_large")


def _build(node: ast.expr) -> sp.Expr:
    if isinstance(node, ast.Constant):
        return _number(node)
    if isinstance(node, ast.Name):
        if node.id == "x":
            return X
        if node.id in CONSTANTS:
            return CONSTANTS[node.id]
        raise ParseFailure(f"unknown_name:{node.id}")
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _build(node.operand)
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp):
        left, right = _build(node.left), _build(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise ParseFailure("division_by_zero")
            return left / right
        if isinstance(node.op, ast.Pow):
            _guard_power(left, right)
            return left**right
        raise ParseFailure(f"operator_not_allowed:{type(node.op).__name__}")
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS:
            name = node.func.id if isinstance(node.func, ast.Name) else "attribute"
            raise ParseFailure(f"function_not_allowed:{name}")
        if len(node.args) != 1 or node.keywords:
            raise ParseFailure(f"bad_arity:{node.func.id}")
        return FUNCTIONS[node.func.id](_build(node.args[0]))
    raise ParseFailure(f"node_not_allowed:{type(node).__name__}")


def parse_math(text: str, trusted: bool = False) -> sp.Expr:
    """Parse a closed-form expression in x. Raises ParseFailure if it is outside the grammar.

    `trusted` lifts only the size limits, for ground truth this repo generated itself.
    """
    tree = _parse_tree(text, trusted)
    try:
        expr = _build(tree)
    except ParseFailure:
        raise
    except (RecursionError, ZeroDivisionError, OverflowError, ValueError, TypeError, AttributeError):
        raise ParseFailure("could_not_build_expression") from None
    if expr.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
        raise ParseFailure("not_finite")
    return expr


def _rational_exponent(node: ast.expr) -> sp.Rational:
    if isinstance(node, ast.Constant):
        return sp.Rational(_number(node))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _rational_exponent(node.operand)
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        denominator = _rational_exponent(node.right)
        if denominator == 0:
            raise ParseFailure("division_by_zero")
        return _rational_exponent(node.left) / denominator
    raise ParseFailure("exponent_not_rational")


def _accumulate(node: ast.expr, power: sp.Rational, names: frozenset[str], out: dict) -> None:
    if isinstance(node, ast.Name):
        if node.id not in names:
            raise ParseFailure(f"unknown_name:{node.id}")
        out[node.id] = out.get(node.id, sp.Integer(0)) + power
    elif isinstance(node, ast.Constant):
        _number(node)  # A pure number has no dimension. The value only has to be a real literal.
    elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        _accumulate(node.left, power, names, out)
        _accumulate(node.right, power, names, out)
    elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        _accumulate(node.left, power, names, out)
        _accumulate(node.right, -power, names, out)
    elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
        _accumulate(node.left, power * _rational_exponent(node.right), names, out)
    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "sqrt":
        if len(node.args) != 1 or node.keywords:
            raise ParseFailure("bad_arity:sqrt")
        _accumulate(node.args[0], power / 2, names, out)
    else:
        raise ParseFailure("not_a_monomial")


def parse_monomial(text: str, names: frozenset[str]) -> dict[str, sp.Rational]:
    """Parse a product of powers of the allowed names, returning {name: total exponent}."""
    out: dict[str, sp.Rational] = {}
    try:
        _accumulate(_parse_tree(text), sp.Integer(1), names, out)
    except (RecursionError, OverflowError, ZeroDivisionError, TypeError, ValueError) as error:
        if isinstance(error, ParseFailure):
            raise
        raise ParseFailure("could_not_build_expression") from None
    return {name: power for name, power in out.items() if power != 0}
