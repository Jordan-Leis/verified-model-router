import sympy as sp

from vmr.safe_parse import X, ParseFailure, parse_math, parse_monomial
from vmr.verifier import (
    CORRECT, INCONCLUSIVE, NO_ANSWER, TRUNCATED, UNPARSEABLE, WRONG,
    dimension_vector, extract_answer, monomial_dimension, verify_answer, verify_response,
    PHYSICAL_CONSTANTS,
)
import pytest


def expr_problem(truth, answer_type="expr", domain=("1/2", "3")):
    return {"id": "t", "answer_type": answer_type, "truth": truth, "domain": list(domain)}


def dim_problem(truth_dimension, names, answer_type="dimension"):
    return {"id": "d", "answer_type": answer_type, "truth_dimension": truth_dimension, "allowed_names": names}


@pytest.mark.parametrize("attack", [
    "__import__('os').system('echo hi')", "x.__class__", "(lambda: 1)()", "[1,2]", "open('f')",
    "sympify('1')", "x if x else 1", "exec('1')",
])
def test_parser_refuses_code(attack):
    with pytest.raises(ParseFailure):
        parse_math(attack)


@pytest.mark.parametrize("text", ["9**9**9", "2**100000", "x**1000", "1/0", "x^2", "ln(x)", "2x", "\\frac{1}{2}", "x + C", "e**x", ""])
def test_parser_rejects_outside_grammar(text):
    with pytest.raises(ParseFailure):
        parse_math(text)


def test_parser_reads_decimals_exactly():
    assert parse_math("0.5*x") == X / 2
    assert parse_math("0.333*x") != X / 3


def test_extract_answer_is_strict():
    assert extract_answer("work\nANSWER: x**2\n\n") == "x**2"
    assert extract_answer("ANSWER: x\nmore text") is None
    assert extract_answer("**ANSWER:** x") is None
    assert extract_answer("answer: x") is None
    assert extract_answer("ANSWER:") is None
    assert extract_answer(None) is None


def test_equivalent_forms_accepted():
    p = expr_problem("sin(x)**2")
    assert verify_answer("1 - cos(x)**2", p).status == CORRECT
    assert verify_answer("(1 - cos(2*x))/2", p).status == CORRECT


def test_positive_domain_identities():
    assert verify_answer("sqrt(x**2)", expr_problem("x")).status == CORRECT
    assert verify_answer("log(x**2)", expr_problem("2*log(x)")).status == CORRECT


def test_wrong_answers_rejected_including_tiny_error():
    p = expr_problem("x**2 + 1")
    assert verify_answer("x**2 + 2", p).status == WRONG
    assert verify_answer("x**2 + 1 + x/1000000000000", p).status == WRONG


def test_antiderivative_ignores_constant_only():
    p = expr_problem("2*x", "antiderivative")
    assert verify_answer("x**2", p).status == CORRECT
    assert verify_answer("x**2 + 7", p).status == CORRECT
    assert verify_answer("x**2 + x", p).status == WRONG


def test_value_must_not_contain_x():
    assert verify_answer("x", expr_problem("2", "value")).status == WRONG
    assert verify_answer("4/2", expr_problem("2", "value")).status == CORRECT


def test_format_failure_is_not_wrong():
    p = expr_problem("x")
    assert verify_response("ANSWER: x^2", "stop", p).status == UNPARSEABLE
    assert verify_response("the answer is x", "stop", p).status == NO_ANSWER
    assert verify_response("so we continue to", "length", p).status == TRUNCATED
    assert verify_response(None, "stop", p).status == NO_ANSWER


def test_dimension_vectors_match_known_values():
    assert dimension_vector(PHYSICAL_CONSTANTS["hbar"]) == (1, 2, -1, 0, 0)
    assert dimension_vector(PHYSICAL_CONSTANTS["G"]) == (-1, 3, -2, 0, 0)
    planck_mass = {"hbar": sp.Rational(1, 2), "c": sp.Rational(1, 2), "G": sp.Rational(-1, 2)}
    assert monomial_dimension(planck_mass) == (1, 0, 0, 0, 0)


def test_monomial_forms():
    names = ["kg", "m", "s", "A", "K"]
    p = dim_problem(["1", "2", "-2", "0", "0"], names)
    for text in ["kg*m**2/s**2", "kg*m**2*s**-2", "kg*(m/s)**2", "m**2*kg/s/s", "2*kg*m**2/s**2"]:
        assert verify_answer(text, p).status == CORRECT, text
    assert verify_answer("kg*m/s**2", p).status == WRONG
    assert verify_answer("kg*m**2 + s", p).status == UNPARSEABLE
    assert verify_answer("J", p).status == UNPARSEABLE


def test_monomial_rational_exponents():
    p = dim_problem(["1", "0", "0", "0", "0"], ["hbar", "c", "G"], "monomial")
    assert verify_answer("sqrt(hbar*c/G)", p).status == CORRECT
    assert verify_answer("hbar**(1/2)*c**(1/2)*G**(-1/2)", p).status == CORRECT
    assert verify_answer("hbar*c/G", p).status == WRONG
    assert verify_answer("kg", p).status == UNPARSEABLE


def test_undecidable_is_inconclusive_not_wrong(monkeypatch):
    # With no numeric decision and no symbolic proof, the verifier must abstain rather than guess.
    monkeypatch.setattr("vmr.verifier.compare_numeric", lambda *a: "undecided")
    monkeypatch.setattr("vmr.verifier._proved_zero", lambda *a: False)
    assert verify_answer("x + 1", expr_problem("x")).status == INCONCLUSIVE
