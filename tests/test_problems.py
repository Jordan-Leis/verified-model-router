import sympy as sp

from vmr.problems import MAIN_SEED, derivative_at_zero, generate, self_check, x
from vmr.prompts import build_prompt
from vmr.verifier import CORRECT, verify_answer

SMALL = {"derivative": (3, 4, 2), "integral": (3, 3, 4), "dimension": (2, 2, 2)}


def test_taylor_shortcut_matches_direct_differentiation():
    factors = [sp.exp(2 * x), sp.cos(3 * x), 1 / (1 + 2 * x**2)]
    direct = sp.diff(sp.Mul(*factors), x, 4).subs(x, 0)
    assert derivative_at_zero(factors, 4) == direct


def test_generation_is_deterministic_and_unique():
    first, second = generate(MAIN_SEED, SMALL), generate(MAIN_SEED, SMALL)
    assert first == second
    assert len({p["statement"] for p in first}) == len(first)


def test_stored_truths_pass_their_own_verifier():
    problems = generate(MAIN_SEED, SMALL)
    self_check(problems)
    for p in problems:
        kind = p["answer_type"]
        if kind == "antiderivative":
            answer = p["reference"]
        elif kind == "dimension":
            answer = "*".join(f"{u}**({e})" for u, e in zip(("kg", "m", "s", "A", "K"), p["truth_dimension"]))
        elif kind == "monomial":
            answer = "*".join(f"{k}**({v})" for k, v in p["reference_exponents"].items())
        else:
            answer = p["truth"]
        if kind in ("expr", "value") and len(answer) > 20000:
            continue
        assert verify_answer(answer, p).status == CORRECT, p["id"]
        assert p["statement"] in build_prompt(p)
