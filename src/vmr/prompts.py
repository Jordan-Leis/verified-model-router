"""The one prompt template. Every model sees the same text for a given problem."""

from __future__ import annotations

COMMON = (
    "Solve the problem below. You may show your working. Finish with exactly one final line in this format, "
    "and write nothing after it:\n\nANSWER: <answer>\n\n"
    "Write that line as plain text, with no markdown, bold, backticks or LaTeX.\n"
)
MATH_RULES = (
    "Rules for <answer>: use Python syntax, with ** for powers and * for every multiplication. "
    "The only allowed names are x, pi, E (Euler's number), and the functions sin, cos, tan, cot, sec, csc, asin, "
    "acos, atan, sinh, cosh, tanh, asinh, acosh, atanh, exp, log (natural logarithm, one argument), sqrt and Abs. "
    "Give an exact closed form, with fractions rather than decimals. It does not need to be simplified.\n"
)
ANTIDERIVATIVE_RULE = "Do not write a constant of integration.\n"
VALUE_RULE = "<answer> must be a number and must not contain x.\n"
UNIT_RULES = (
    "Rules for <answer>: write a product of powers using only kg, m, s, A and K, in Python syntax, with ** for "
    "powers, * for multiplication and / for division, for example kg*m**2/s**3 or kg**(1/2)*m**(-3). "
    "If the quantity is dimensionless, write 1.\n"
)


def monomial_rules(names: list[str]) -> str:
    return (
        f"Rules for <answer>: write a product of powers using only {', '.join(names)}, in Python syntax, with ** "
        "for powers and * for multiplication. Put fractional exponents in parentheses, for example hbar**(1/2)*c**(-3/2).\n"
    )


def build_prompt(problem: dict) -> str:
    kind = problem["answer_type"]
    if kind == "dimension":
        rules = UNIT_RULES
    elif kind == "monomial":
        rules = monomial_rules(problem["allowed_names"])
    else:
        rules = MATH_RULES + {"antiderivative": ANTIDERIVATIVE_RULE, "value": VALUE_RULE, "expr": ""}[kind]
    return f"{COMMON}{rules}\nProblem: {problem['statement']}"
