"""Seeded problem generator. Run `python -m vmr.problems` to write data/problems.jsonl.

Each family has three tiers. Templates are used round-robin inside a tier, with seeded random
parameters. docs/calibration.md records how the tiers were set and what was held fixed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import sympy as sp

from vmr.safe_parse import X as x
from vmr.safe_parse import parse_math
from vmr.verifier import (
    PHYSICAL_CONSTANTS, SI_UNITS, _Timeout, dimension_vector, monomial_dimension, time_limit,
)

MAIN_SEED = 20261107
PILOT_SEED = 8  # Pilot 1 used seed 7 with the earlier tiers; its problems are kept in data/pilot1/.
TIER_COUNTS = {"derivative": (10, 15, 15), "integral": (10, 15, 15), "dimension": (10, 15, 15)}
PILOT_COUNTS = {"derivative": (3, 3, 4), "integral": (3, 3, 4), "dimension": (3, 3, 4)}
DEFAULT_DOMAIN = ("1/2", "3")

R = sp.Rational


def _nonzero(rng: random.Random, lo: int, hi: int) -> int:
    return rng.choice([v for v in range(lo, hi + 1) if v != 0])


# --- Derivatives -----------------------------------------------------------------------------

def _d1_power_trig(r):
    return r.randint(2, 6) * x ** r.randint(2, 5) * r.choice([sp.sin, sp.cos])(r.randint(2, 5) * x), 1


def _d1_chain_power(r):
    return (x ** 2 + r.randint(1, 6)) ** r.randint(3, 7), 1


def _d1_exp_log(r):
    return sp.exp(r.randint(2, 4) * x) * sp.log(x + r.randint(1, 5)), 1


def _d2_quotient(r):
    return r.choice([sp.sin, sp.cos])(r.randint(2, 4) * x) / (x ** 2 + r.randint(1, 5)), 2


def _d2_nested(r):
    return sp.sin(sp.log(x ** 2 + r.randint(1, 5))) * sp.exp(x / r.randint(2, 3)), 1


def _d2_root_product(r):
    a = r.randint(1, 4)
    return sp.sqrt(x + a) * sp.atan(r.randint(2, 3) * x) * sp.exp(-x), 2


def _d3_nested_third(r):
    return sp.log(1 + sp.sin(x) ** 2 + r.randint(1, 5) * x ** 2) * sp.exp(r.randint(1, 3) * x), 3


def _d3_tower_third(r):
    return sp.atan(sp.exp(r.randint(1, 3) * x) / (x + r.randint(1, 5))), 3


def _series_a(r):
    return [sp.exp(r.randint(1, 3) * x), sp.cos(r.randint(1, 3) * x), 1 / (1 + r.randint(1, 3) * x ** 2)]


def _series_b(r):
    return [sp.log(1 + r.randint(1, 3) * x), sp.sin(r.randint(1, 3) * x), sp.exp(_nonzero(r, -2, 2) * x)]


def _series_c(r):
    return [sp.atan(r.randint(1, 3) * x), sp.sqrt(1 + 4 * r.randint(1, 3) * x), 1 / (1 - r.randint(1, 2) * x)]


def taylor_coefficients(factor: sp.Expr, order: int) -> list[sp.Rational]:
    coefficients, derivative = [], factor
    for k in range(order + 1):
        coefficients.append(derivative.subs(x, 0) / sp.factorial(k))
        derivative = sp.diff(derivative, x)
    return coefficients


def derivative_at_zero(factors: list[sp.Expr], order: int) -> sp.Rational:
    """Exact order-th derivative of a product at 0, using truncated Taylor series.

    Differentiating the whole product eight times takes minutes because the expression swells, while
    each single factor is cheap and the coefficients are exact rationals.
    """
    product = [sp.Integer(1)] + [sp.Integer(0)] * order
    for factor in factors:
        series = taylor_coefficients(factor, order)
        product = [sum(product[i] * series[k - i] for i in range(k + 1)) for k in range(order + 1)]
    return product[order] * sp.factorial(order)


SIMPLE_FACTORS = [
    lambda r: sp.exp(_nonzero(r, -3, 3) * x), lambda r: sp.cos(r.randint(1, 3) * x),
    lambda r: sp.log(1 + r.randint(1, 3) * x), lambda r: 1 / (1 + r.randint(1, 3) * x ** 2),
    lambda r: 1 / (1 - r.randint(1, 3) * x), lambda r: sp.sqrt(1 + 4 * r.randint(1, 3) * x),
    lambda r: sp.atan(r.randint(1, 3) * x),
]
COMPOSITE_FACTORS = [
    lambda r: sp.exp(sp.sin(r.randint(1, 3) * x)), lambda r: sp.log(1 + sp.sin(r.randint(1, 2) * x)),
    lambda r: sp.atan(sp.exp(r.randint(1, 2) * x) - 1), lambda r: sp.tan(r.randint(1, 3) * x),
    lambda r: sp.exp(sp.atan(r.randint(1, 2) * x)), lambda r: sp.cos(sp.sin(r.randint(1, 2) * x)),
]


def _series_hard(r):
    return [r.choice(COMPOSITE_FACTORS)(r)] + [make(r) for make in r.sample(SIMPLE_FACTORS, 2)]


def _value_problem(factors, order):
    f = sp.Mul(*factors)
    value = derivative_at_zero(factors, order)
    statement = f"Let f(x) = {f}. Compute the exact value of the derivative of order {order} of f at x = 0."
    return statement, "value", str(value), {"domain": ["1", "2"]}


def _series_hardest(r):
    return [r.choice(COMPOSITE_FACTORS)(r)] + [make(r) for make in r.sample(SIMPLE_FACTORS, 3)]


def _derivative_problem(rng, tier, slot):
    """Pilot 2 tiers: T1 = third derivatives and order 6-8 series, T2 = order 9-11, T3 = order 12-13."""
    if tier == 3:
        return _value_problem(_series_hardest(rng), rng.randint(12, 13))
    if tier == 2:
        return _value_problem(_series_hard(rng), rng.randint(9, 11))
    if slot % 2 == 0:
        return _value_problem([_series_a, _series_b, _series_c][(slot // 2) % 3](rng), rng.randint(6, 8))
    templates = [_d3_nested_third, _d3_tower_third]
    f, order = templates[(slot // 2) % len(templates)](rng)
    truth = sp.diff(f, x, order)
    statement = f"Let f(x) = {f} for x in (1/2, 3). Compute the third derivative of f with respect to x, as a function of x."
    return statement, "expr", str(truth), {"domain": list(DEFAULT_DOMAIN)}


# --- Integrals -------------------------------------------------------------------------------

def _i1_poly_exp(r):
    return x ** r.randint(2, 3) * sp.exp(r.randint(2, 4) * x)


def _i1_x_trig(r):
    return x * r.choice([sp.sin, sp.cos])(r.randint(2, 5) * x)


def _i1_log_substitution(r):
    return x / (x ** 2 + r.randint(1, 9))


def _i2_exp_trig(r):
    return sp.exp(r.randint(1, 3) * x) * r.choice([sp.sin, sp.cos])(r.randint(2, 4) * x)


def _i2_partial_fractions(r):
    a, b = r.sample(range(1, 7), 2)
    return (r.randint(1, 5) * x + r.randint(1, 7)) / ((x + a) * (x + b) ** 2)


def _i2_poly_log(r):
    return x ** r.randint(2, 4) * sp.log(r.randint(1, 4) * x) ** 2


def _i2_complete_square(r):
    a = r.randint(1, 4)
    return 1 / (x ** 2 + 2 * a * x + a ** 2 + r.randint(1, 9))


def _i3_poly_exp_trig(r):
    return x ** 2 * sp.exp(r.randint(1, 2) * x) * r.choice([sp.sin, sp.cos])(r.randint(2, 3) * x)


def _i3_mixed_fractions(r):
    return (x ** 2 + r.randint(1, 5) * x + r.randint(1, 5)) / ((x + r.randint(1, 4)) * (x ** 2 + r.randint(1, 6)))


def _i3_repeated_quadratic(r):
    return (r.randint(1, 4) * x + r.randint(1, 4)) / (x ** 2 + r.randint(1, 6)) ** 2


def _i3_x_atan(r):
    return x ** r.randint(2, 3) * sp.atan(r.randint(1, 3) * x)


def _i3_root_quadratic(r):
    return x ** 2 * sp.sqrt(x ** 2 + r.randint(1, 6))


def _h_poly_exp_trig(r):
    return x ** 4 * sp.exp(r.randint(1, 3) * x) * r.choice([sp.sin, sp.cos])(r.randint(2, 3) * x)


def _h_quartic(r):
    return r.randint(1, 3) / (x ** 4 + r.choice([1, 2, 3, 4, 5, 9, 16]))


def _h_cubic(r):
    return (x + r.randint(1, 3)) / (x ** 3 + r.randint(1, 5) ** 3)


def _h_repeated_mixed(r):
    return (x ** 3 + r.randint(1, 3) * x + r.randint(1, 3)) / ((x + r.randint(1, 3)) ** 2 * (x ** 2 + r.randint(1, 3)) ** 2)


def _h_log_cubed(r):
    return x ** r.randint(2, 4) * sp.log(r.randint(2, 3) * x) ** 3


def _h_trig_product(r):
    return sp.sin(r.randint(2, 3) * x) ** 3 * sp.cos(r.randint(1, 4) * x) ** 2


def _h_exp_two_trig(r):
    return x * sp.exp(r.randint(1, 2) * x) * sp.sin(r.randint(2, 3) * x) * sp.cos(r.randint(1, 4) * x)


def _h_root_poly(r):
    a = r.randint(1, 3)
    return x ** 3 * sp.sqrt(x ** 2 + 2 * a * x + a ** 2 + r.randint(1, 4))


def _h_cubed_quadratic(r):
    return x ** r.choice([0, 2, 5]) / (x ** 2 + r.randint(1, 6)) ** 3


def _x_poly5_exp_trig(r):
    return x ** 5 * sp.exp(r.randint(1, 2) * x) * r.choice([sp.sin, sp.cos])(r.randint(2, 3) * x)


def _x_quartic_ratio(r):
    return (x ** 2 + r.randint(1, 4)) / (x ** 4 + r.choice([2, 3, 5, 6, 7]))


def _x_trig_powers(r):
    a = r.randint(1, 3)
    return sp.sin(a * x) ** 5 * sp.cos(a * x) ** 4 + sp.cos(r.randint(2, 4) * x) ** 3 * sp.sin(a * x)


def _x_cubed_repeated(r):
    return (x ** 3 + r.randint(1, 3) * x ** 2 + r.randint(1, 3)) / ((x + r.randint(1, 3)) ** 3 * (x ** 2 + r.randint(1, 3)) ** 2)


def _x_log_quartic(r):
    return x ** r.randint(3, 5) * sp.log(r.randint(2, 3) * x) ** 4


def _x_exp_three_trig(r):
    return x ** 2 * sp.exp(r.randint(1, 2) * x) * sp.sin(r.randint(2, 3) * x) * sp.cos(r.randint(1, 4) * x)


def _integral_problem(rng, tier, slot):
    """Pilot 2 tiers: each earlier tier moved down one level, with a new, harder tier 3."""
    templates = {
        1: [_i3_poly_exp_trig, _i3_mixed_fractions, _i3_repeated_quadratic, _i3_x_atan, _i3_root_quadratic],
        2: [_h_poly_exp_trig, _h_quartic, _h_cubic, _h_repeated_mixed, _h_log_cubed, _h_trig_product,
            _h_exp_two_trig, _h_root_poly, _h_cubed_quadratic],
        3: [_x_poly5_exp_trig, _x_quartic_ratio, _x_trig_powers, _x_cubed_repeated, _x_log_quartic,
            _x_exp_three_trig],
    }[tier]
    make = templates[slot % len(templates)]
    for _ in range(5):
        f = make(rng)
        reference = _antiderivative(f)
        if reference is not None:
            break
    else:
        raise RuntimeError(f"{make.__name__}: SymPy found no usable antiderivative")
    statement = (
        f"Find an antiderivative with respect to x of f(x) = {f} for x in (1/2, 3). "
        "Give any one antiderivative, in closed form, and leave out the constant of integration."
    )
    return statement, "antiderivative", str(f), {"domain": list(DEFAULT_DOMAIN), "reference": str(reference)}


def _antiderivative(f):
    """Return a closed form whose derivative matches f at high precision, or None."""
    try:
        with time_limit(20):
            result = sp.integrate(f, x)
    except _Timeout:
        return None
    if result.has(sp.Integral, sp.Piecewise, sp.I):
        return None
    for point in (R(7, 10), R(13, 10), R(5, 2)):
        delta = (sp.diff(result, x) - f).subs(x, point).evalf(50)
        if abs(delta) > sp.Float("1e-30", 50):
            raise AssertionError(f"SymPy antiderivative of {f} failed the differentiation check")
    return result


# --- Dimensional analysis --------------------------------------------------------------------

# Each named quantity is given as a monomial in SI units, then reduced by sympy.physics.units.
from sympy.physics import units as u  # noqa: E402

QUANTITIES = {
    "force": u.newton, "energy": u.joule, "power": u.watt, "pressure": u.pascal, "electric charge": u.coulomb,
    "voltage": u.volt, "electrical resistance": u.ohm, "capacitance": u.farad, "inductance": u.henry,
    "magnetic flux": u.weber, "magnetic flux density": u.tesla,
}
DERIVED = {
    "dynamic viscosity": {"pascal": 1, "second": 1},
    "thermal conductivity": {"watt": 1, "meter": -1, "kelvin": -1},
    "specific heat capacity": {"joule": 1, "kilogram": -1, "kelvin": -1},
    "entropy": {"joule": 1, "kelvin": -1},
    "surface tension": {"newton": 1, "meter": -1},
    "electric field strength": {"volt": 1, "meter": -1},
    "electrical resistivity": {"ohm": 1, "meter": 1},
    "electrical conductivity": {"ohm": -1, "meter": -1},
    "mass density": {"kilogram": 1, "meter": -3},
    "angular momentum": {"joule": 1, "second": 1},
    "kinematic viscosity": {"meter": 2, "second": -1},
    "heat flux": {"watt": 1, "meter": -2},
}
BASE = {"mass": u.kilogram, "length": u.meter, "time": u.second, "electric current": u.ampere, "temperature": u.kelvin}


def quantity_dimension(name: str) -> tuple:
    if name in QUANTITIES:
        return dimension_vector(QUANTITIES[name])
    if name in BASE:
        return dimension_vector(BASE[name])
    total = [sp.Integer(0)] * 5
    for unit, power in DERIVED[name].items():
        for i, exponent in enumerate(dimension_vector(getattr(u, unit))):
            total[i] += power * exponent
    return tuple(total)


CONSTANT_LABELS = {
    "hbar": "the reduced Planck constant", "c": "the speed of light", "G": "Newton's gravitational constant",
    "k_B": "the Boltzmann constant", "e_charge": "the elementary charge",
    "eps0": "the vacuum permittivity", "mu0": "the vacuum permeability",
}
CONSTANT_SETS = [
    ("hbar", "c", "G"), ("hbar", "c", "k_B", "G"), ("hbar", "c", "G", "eps0"), ("hbar", "c", "G", "k_B", "eps0"),
    ("hbar", "c", "G", "k_B", "mu0"), ("hbar", "G", "e_charge", "k_B", "eps0"),
]


def _format_power(power: int) -> str:
    return "" if power == 1 else f"^{power}" if power > 0 else f"^({power})"


def _compound_problem(rng):
    names = rng.sample(sorted(QUANTITIES) + sorted(DERIVED), rng.randint(3, 4))
    powers = [_nonzero(rng, -2, 2) for _ in names]
    total = [sp.Integer(0)] * 5
    for name, power in zip(names, powers):
        for i, exponent in enumerate(quantity_dimension(name)):
            total[i] += power * exponent
    formula = " * ".join(f"({name}){_format_power(power)}" for name, power in zip(names, powers))
    statement = (
        f"A quantity Q has the same dimensions as {formula}. "
        "Express the dimensions of Q as a product of powers of the SI base units kg, m, s, A and K."
    )
    return statement, "dimension", [str(v) for v in total], {"allowed_names": sorted(SI_UNITS)}


def _solve_exponents(vectors: list[tuple], target: tuple) -> list | None:
    """Solve for the exponents that give `target`. None means no unique rational solution exists."""
    matrix = sp.Matrix([[v[i] for v in vectors] for i in range(5)])
    if matrix.rank() != len(vectors):
        return None
    symbols = sp.symbols(f"a0:{len(vectors)}")
    solution = sp.linsolve((matrix, sp.Matrix(target)), *symbols)
    if not solution:
        return None
    values = list(next(iter(solution)))
    return values if all(v.is_Rational for v in values) else None


def _monomial_problem(rng):
    while True:
        constants = rng.choice(CONSTANT_SETS)
        target_name = rng.choice(sorted(QUANTITIES) + sorted(DERIVED) + sorted(BASE))
        target = quantity_dimension(target_name)
        solution = _solve_exponents([dimension_vector(PHYSICAL_CONSTANTS[c]) for c in constants], target)
        if solution is not None:
            break
    exponents = dict(zip(constants, solution))
    assert monomial_dimension(exponents) == target
    listed = ", ".join(f"{CONSTANT_LABELS[c]} ({c})" for c in constants)
    statement = (
        f"Find the unique product of powers of {listed} that has the dimensions of {target_name}. "
        "Numerical prefactors are ignored, so give the monomial with coefficient 1."
    )
    extras = {"allowed_names": sorted(constants), "reference_exponents": {k: str(v) for k, v in exponents.items()}}
    return statement, "monomial", [str(v) for v in target], extras


def _basis_problem(rng, compound_target: bool = False):
    """Express a target in an arbitrary basis of five named quantities: a 5x5 exact linear solve.

    With `compound_target`, the target is itself a product of powers of two named quantities, and the
    solution must have five nonzero exponents and a denominator of at least 3.
    """
    pool = sorted(QUANTITIES) + sorted(DERIVED)
    while True:
        chosen = rng.sample(pool, 7)
        basis = chosen[:5]
        if compound_target:
            p, q = _nonzero(rng, -2, 2), _nonzero(rng, -2, 2)
            target = tuple(p * a + q * b for a, b in zip(quantity_dimension(chosen[5]), quantity_dimension(chosen[6])))
            target_name = f"({chosen[5]}){_format_power(p)} * ({chosen[6]}){_format_power(q)}"
        else:
            target_name = chosen[5]
            target = quantity_dimension(target_name)
        solution = _solve_exponents([quantity_dimension(n) for n in basis], target)
        if solution is None:
            continue
        if compound_target:
            if all(v != 0 for v in solution) and any(sp.Rational(v).q >= 3 for v in solution):
                break
        elif sum(1 for v in solution if v != 0) >= 4 and any(not v.is_Integer for v in solution):
            break
    names = [f"Q{i}" for i in range(1, 6)]
    table = {name: [str(v) for v in quantity_dimension(q)] for name, q in zip(names, basis)}
    exponents = dict(zip(names, solution))
    assert monomial_dimension(exponents, table) == target
    listed = ", ".join(f"{name} = {q}" for name, q in zip(names, basis))
    statement = (
        f"Let {listed}. Find the unique product of powers of Q1, Q2, Q3, Q4 and Q5 that has the dimensions of "
        f"{target_name}. Numerical prefactors are ignored, so give the monomial with coefficient 1."
    )
    extras = {"allowed_names": names, "name_dimensions": table, "reference_exponents": {k: str(v) for k, v in exponents.items()}}
    return statement, "monomial", [str(v) for v in target], extras


def _dimension_problem(rng, tier, slot):
    """Pilot 2 tiers: T1 = constant monomials, T2 = basis solves, T3 = basis solves for compound targets."""
    if tier == 1:
        return _monomial_problem(rng)
    return _basis_problem(rng, compound_target=tier == 3)


GENERATORS = {"derivative": _derivative_problem, "integral": _integral_problem, "dimension": _dimension_problem}


def generate(seed: int, counts: dict[str, tuple[int, int, int]]) -> list[dict]:
    rng = random.Random(seed)
    problems, seen = [], set()
    for family, tiers in counts.items():
        serial = 0
        for tier, count in enumerate(tiers, start=1):
            slot = repeats = 0
            while slot < count:
                statement, kind, truth, extras = GENERATORS[family](rng, tier, slot)
                if statement in seen:
                    repeats += 1
                    if repeats > 200:
                        raise RuntimeError(f"{family} tier {tier} slot {slot}: template has run out of new problems")
                    continue
                repeats = 0
                seen.add(statement)
                serial += 1
                slot += 1
                record = {"id": f"{family[:5]}-{serial:03d}", "family": family, "tier": tier,
                          "answer_type": kind, "statement": statement, **extras}
                record["truth_dimension" if kind in ("dimension", "monomial") else "truth"] = truth
                problems.append(record)
    return problems


def self_check(problems: list[dict]) -> None:
    """Every stored truth must parse under the same grammar the models are held to."""
    for p in problems:
        if "truth" in p:
            parse_math(p["truth"], trusted=True)
        if "reference" in p:
            parse_math(p["reference"], trusted=True)


def load(path: str | Path = "data/problems.jsonl") -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail if regenerating differs from the committed set")
    args = parser.parse_args()
    for name, seed, counts in (("problems", MAIN_SEED, TIER_COUNTS), ("pilot_problems", PILOT_SEED, PILOT_COUNTS)):
        problems = generate(seed, counts)
        self_check(problems)
        path = Path("data") / f"{name}.jsonl"
        text = "".join(json.dumps(p, sort_keys=True) + "\n" for p in problems)
        if args.check:
            if path.read_text() != text:
                raise SystemExit(f"{path} does not match what the generator produces for seed {seed}")
        else:
            path.write_text(text)
        print(f"{path}: {len(problems)} problems, sha256 {hashlib.sha256(text.encode()).hexdigest()[:12]}")


if __name__ == "__main__":
    main()
