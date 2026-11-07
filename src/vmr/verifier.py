"""SymPy verifier for model answers.

The verifier returns one of six statuses, and only `correct` and `wrong` say anything about the
mathematics or physics of an answer:

    correct       parsed, and equal to the truth
    wrong         parsed, and provably or numerically different from the truth
    unparseable   an ANSWER line exists but is outside the required grammar
    no_answer     the response has no valid ANSWER line as its final line
    truncated     the response hit the output-token limit before it could finish
    inconclusive  parsed, but the checks could not decide

Math problems are decided by high-precision evaluation at random points in the stated domain, plus
a time-limited symbolic proof that is recorded when it succeeds. A numeric match therefore means
"equal at 8 random points to 60 digits", not a proof. The `method` field says which one applied.
Dimension problems compare exact rational exponent vectors from `sympy.physics.units`, so they check
dimensions only: an answer can pass with a wrong numeric prefactor.
"""

from __future__ import annotations

import hashlib
import random
import re
import signal
from contextlib import contextmanager
from dataclasses import asdict, dataclass

import sympy as sp
from sympy.physics import units as u
from sympy.physics.units.systems.si import SI, dimsys_SI

from vmr.safe_parse import X, ParseFailure, parse_math, parse_monomial

CORRECT, WRONG = "correct", "wrong"
UNPARSEABLE, NO_ANSWER, TRUNCATED, INCONCLUSIVE = "unparseable", "no_answer", "truncated", "inconclusive"
STATUSES = (CORRECT, WRONG, UNPARSEABLE, NO_ANSWER, TRUNCATED, INCONCLUSIVE)

SPOT_CHECKS = 8
DIGITS = 60
TOLERANCE = sp.Float("1e-30", DIGITS)
SYMBOLIC_TIMEOUT_S = 4.0
NUMERIC_TIMEOUT_S = 10.0

ANSWER_LINE = re.compile(r"^ANSWER: (?P<answer>\S.*)$")

BASE_NAMES = ("mass", "length", "time", "current", "temperature")
SI_UNITS = {"kg": u.kilogram, "m": u.meter, "s": u.second, "A": u.ampere, "K": u.kelvin}
PHYSICAL_CONSTANTS = {
    "hbar": u.hbar, "c": u.speed_of_light, "G": u.gravitational_constant,
    "k_B": u.boltzmann_constant, "e_charge": u.elementary_charge,
    "eps0": u.vacuum_permittivity, "mu0": u.vacuum_permeability,
}


@dataclass(frozen=True)
class Verdict:
    status: str
    reason: str = ""
    method: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class _Timeout(Exception):
    pass


@contextmanager
def time_limit(seconds: float):
    """Interrupt pure-Python SymPy work. It relies on SIGALRM, so it only works on the main thread."""

    def on_alarm(*_):
        raise _Timeout

    previous = signal.signal(signal.SIGALRM, on_alarm)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def extract_answer(text: str | None) -> str | None:
    """Return the answer on the final non-empty line, or None. The format is deliberately strict."""
    lines = [line.rstrip() for line in (text or "").splitlines() if line.strip()]
    if not lines:
        return None
    match = ANSWER_LINE.match(lines[-1])
    return match.group("answer").strip() if match else None


def dimension_vector(quantity) -> tuple:
    """Exponents of (mass, length, time, current, temperature) for a sympy.physics.units Quantity."""
    deps = dimsys_SI.get_dimensional_dependencies(SI.get_quantity_dimension(quantity))
    by_name = {str(dim.name): sp.Rational(power) for dim, power in deps.items()}
    unknown = set(by_name) - set(BASE_NAMES)
    if unknown:
        raise ValueError(f"unexpected base dimensions: {unknown}")
    return tuple(by_name.get(name, sp.Integer(0)) for name in BASE_NAMES)


def monomial_dimension(powers: dict[str, sp.Rational], table: dict | None = None) -> tuple:
    """Total dimension of a monomial. `table` gives per-problem names as stored exponent vectors."""
    units = {**SI_UNITS, **PHYSICAL_CONSTANTS}
    total = [sp.Integer(0)] * len(BASE_NAMES)
    for name, power in powers.items():
        vector = [sp.Rational(v) for v in table[name]] if table else dimension_vector(units[name])
        for i, exponent in enumerate(vector):
            total[i] += power * exponent
    return tuple(total)


def sample_points(problem_id: str, domain: tuple[str, str], count: int = SPOT_CHECKS) -> list[sp.Rational]:
    """Reproducible rational points in the open domain. The seed does not use Python's salted hash."""
    seed = int.from_bytes(hashlib.sha256(problem_id.encode()).digest()[:8], "big")
    rng = random.Random(seed)
    lo, hi = sp.Rational(domain[0]), sp.Rational(domain[1])
    return [lo + (hi - lo) * sp.Rational(rng.randint(1, 9998), 9999) for _ in range(count)]


def _evaluate(expr: sp.Expr, point: sp.Rational) -> sp.Expr | None:
    value = expr.subs(X, point).evalf(DIGITS)
    if value.has(sp.zoo, sp.nan, sp.oo, -sp.oo) or value.free_symbols or not value.is_number:
        return None
    return value


def compare_numeric(left: sp.Expr, right: sp.Expr, points: list[sp.Rational]) -> str:
    """Return 'equal', 'different', or 'undecided'. A difference at one point is enough to reject."""
    decided = 0
    for point in points:
        a, b = _evaluate(left, point), _evaluate(right, point)
        if a is None or b is None:
            continue
        decided += 1
        if abs(a - b) > TOLERANCE * max(sp.Integer(1), abs(a), abs(b)):
            return "different"
    return "equal" if decided >= max(3, len(points) // 2) else "undecided"


def _proved_zero(difference: sp.Expr) -> bool:
    try:
        with time_limit(SYMBOLIC_TIMEOUT_S):
            return difference == 0 or sp.simplify(difference) == 0
    except _Timeout:
        return False
    except Exception:  # SymPy raises many internal types on odd input. Failing to prove is not failure.
        return False


def _check_equal(candidate: sp.Expr, truth: sp.Expr, problem: dict) -> Verdict:
    points = sample_points(problem["id"], tuple(problem["domain"]))
    try:
        with time_limit(NUMERIC_TIMEOUT_S):
            outcome = compare_numeric(candidate, truth, points)
    except _Timeout:
        outcome = "undecided"
    except Exception:
        outcome = "undecided"
    if outcome == "different":
        return Verdict(WRONG, "differs_at_spot_check", "numeric")
    proved = _proved_zero(candidate - truth)
    if proved:
        return Verdict(CORRECT, "", "symbolic")
    if outcome == "equal":
        return Verdict(CORRECT, "", "numeric")
    return Verdict(INCONCLUSIVE, "numeric_undecided_and_no_proof", "")


def _verify_math(answer: str, problem: dict) -> Verdict:
    try:
        candidate = parse_math(answer)
    except ParseFailure as failure:
        return Verdict(UNPARSEABLE, failure.reason)
    truth = parse_math(problem["truth"], trusted=True)
    kind = problem["answer_type"]
    if kind == "value" and candidate.has(X):
        return Verdict(WRONG, "value_depends_on_x", "structural")
    if kind == "antiderivative":
        # Differentiating the answer makes any constant of integration irrelevant, and needs no
        # reference antiderivative: `truth` here is the integrand.
        try:
            with time_limit(NUMERIC_TIMEOUT_S):
                candidate = sp.diff(candidate, X)
        except _Timeout:
            return Verdict(INCONCLUSIVE, "derivative_timed_out")
    return _check_equal(candidate, truth, problem)


def _verify_dimension(answer: str, problem: dict) -> Verdict:
    names = frozenset(problem["allowed_names"])
    try:
        powers = parse_monomial(answer, names)
    except ParseFailure as failure:
        return Verdict(UNPARSEABLE, failure.reason)
    target = tuple(sp.Rational(value) for value in problem["truth_dimension"])
    if monomial_dimension(powers, problem.get("name_dimensions")) == target:
        return Verdict(CORRECT, "", "dimensional")
    return Verdict(WRONG, "dimension_mismatch", "dimensional")


def verify_answer(answer: str, problem: dict) -> Verdict:
    """Check an already-extracted answer string."""
    if problem["answer_type"] in ("dimension", "monomial"):
        return _verify_dimension(answer, problem)
    return _verify_math(answer, problem)


def verify_response(text: str | None, finish_reason: str | None, problem: dict) -> Verdict:
    """Check a full model response, keeping format failures apart from wrong answers."""
    answer = extract_answer(text)
    if answer is None:
        if finish_reason == "length":
            return Verdict(TRUNCATED, "hit_output_token_limit")
        return Verdict(NO_ANSWER, "final_line_is_not_an_answer_line")
    return verify_answer(answer, problem)
