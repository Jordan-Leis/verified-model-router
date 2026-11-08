"""Audit the verifier itself.

* False accepts: deliberately wrong answers (mutations of the truth) that the verifier calls correct.
* False rejects: equivalent rewrites of the truth that the verifier does not call correct.
Each injected answer is confirmed wrong (or equivalent) by an independent check (mpmath through
`lambdify`, at points not used by the verifier) before it is counted, so a mutation that happens to
be equivalent is dropped, not scored. For dimension problems the basis has full rank, so any change
of exponents changes the dimension, and mutants are wrong by construction.

It also writes a stratified sample of real model verdicts for a person to check by hand.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import mpmath
import sympy as sp

from vmr.analysis import wilson
from vmr.matrix import read_rows
from vmr.problems import load
from vmr.safe_parse import X, ParseFailure, parse_math
from vmr.verifier import CORRECT, verify_answer

SEED = 1107
INDEPENDENT_POINTS = [mpmath.mpf("0.61"), mpmath.mpf("1.37"), mpmath.mpf("2.03"), mpmath.mpf("2.71"), mpmath.mpf("1.88")]


def _independently_equal(a: sp.Expr, b: sp.Expr, derivative: bool) -> bool | None:
    if derivative:
        a, b = sp.diff(a, X), sp.diff(b, X)
    mpmath.mp.dps = 40
    try:
        fa, fb = sp.lambdify(X, a, "mpmath"), sp.lambdify(X, b, "mpmath")
        diffs = [abs(fa(p) - fb(p)) / max(1, abs(fb(p))) for p in INDEPENDENT_POINTS]
    except Exception:
        return None
    return all(d < mpmath.mpf("1e-25") for d in diffs)


def _target(problem: dict) -> sp.Expr:
    return parse_math(problem["reference"] if problem["answer_type"] == "antiderivative" else problem["truth"], trusted=True)


def math_mutants(problem: dict) -> dict[str, sp.Expr]:
    t = _target(problem)
    kind = problem["answer_type"]
    out = {"scale_by_2": 2 * t if t != 0 else t + 1,
           "shift": t + (X if kind == "antiderivative" else 1),
           # Relative for values: some exact answers are near 1e20, where an absolute 1e-8 is invisible.
           "small_perturbation": t * (1 + sp.Rational(1, 10**8)) if kind == "value" else t + X / 10**8}
    if isinstance(t, sp.Add):
        terms = list(t.args)
        out["flip_sign_of_one_term"] = sp.Add(-terms[0], *terms[1:])
        out["drop_one_term"] = sp.Add(*terms[:-1])
    else:
        out["flip_sign"] = -t
    swapped = t.replace(sp.sin, lambda arg: sp.Function("tmp")(arg)).replace(sp.cos, sp.sin).replace(sp.Function("tmp"), sp.cos)
    if swapped != t:
        out["swap_sin_cos"] = swapped
    if kind != "value":
        out["wrong_inner_scale"] = t.subs(X, sp.Rational(11, 10) * X)
    return out


def math_equivalents(problem: dict) -> dict[str, str]:
    t = _target(problem)
    s = str(t)
    out = {"as_generated": s, "expanded": str(sp.expand(t)), "together": str(sp.together(t)),
           "identity_inserted": f"({s}) + sin(x)**2 + cos(x)**2 - 1", "exp_log_factor": f"({s})*exp(log(x))/x"}
    if problem["answer_type"] == "antiderivative":
        out["plus_constant"] = f"{s} + 7"
        out["plus_hidden_constant"] = f"{s} + atan(x) + atan(1/x)"
    if problem["answer_type"] == "value":
        out = {"as_generated": s, "split_fraction": f"({sp.fraction(t)[0]})/({sp.fraction(t)[1]})", "times_one": f"2*({s})/2"}
    return out


def monomial_forms(problem: dict) -> tuple[dict[str, str], dict[str, str]]:
    ref = {k: sp.Rational(v) for k, v in problem["reference_exponents"].items()}
    nonzero = [k for k, v in ref.items() if v != 0]

    def render(exps, sep="*"):
        return sep.join(f"{k}**({v})" for k, v in exps.items() if v != 0) or "1"

    pos = "*".join(f"{k}**({v})" for k, v in ref.items() if v > 0) or "1"
    neg = "*".join(f"{k}**({-v})" for k, v in ref.items() if v < 0)
    equivalents = {"explicit": render(ref), "reversed_order": render(dict(reversed(list(ref.items())))),
                   "with_numeric_prefactor": f"2*{render(ref)}", "as_fraction": f"({pos})/({neg})" if neg else pos,
                   "square_root_of_square": f"sqrt(({render(ref)})**2)"}
    first = nonzero[0]
    mutants = {"one_exponent_plus_1": render({**ref, first: ref[first] + 1}),
               "one_exponent_minus_half": render({**ref, first: ref[first] - sp.Rational(1, 2)}),
               "drop_one_factor": render({k: v for k, v in ref.items() if k != first}),
               "all_exponents_negated": render({k: -v for k, v in ref.items()})}
    if len(nonzero) >= 2 and ref[nonzero[0]] != ref[nonzero[1]]:
        a, b = nonzero[:2]
        mutants["two_exponents_swapped"] = render({**ref, a: ref[b], b: ref[a]})
    return equivalents, mutants


def injected_audit(problems: list[dict]) -> dict:
    tally = {"false_accept": Counter(), "false_reject": Counter()}
    trials = {"false_accept": Counter(), "false_reject": Counter()}
    dropped = Counter()
    reject_statuses = Counter()
    examples = []
    for p in problems:
        family = p["family"]
        if p["answer_type"] == "monomial":
            equivalents, mutants = monomial_forms(p)
            mutant_texts = mutants
        else:
            truth = _target(p)
            derivative = p["answer_type"] == "antiderivative"
            mutant_texts = {}
            for name, expr in math_mutants(p).items():
                if _independently_equal(expr, truth, derivative) is not False:
                    dropped["mutant_not_confirmed_wrong"] += 1
                    continue
                mutant_texts[name] = str(expr)
            equivalents = {}
            for name, text in math_equivalents(p).items():
                try:
                    expr = parse_math(text, trusted=True)
                except ParseFailure:
                    dropped["rewrite_outside_grammar"] += 1
                    continue
                if _independently_equal(expr, truth, derivative) is not True:
                    dropped["rewrite_not_confirmed_equal"] += 1
                    continue
                equivalents[name] = text
        for name, text in mutant_texts.items():
            status = verify_answer(text, p).status
            trials["false_accept"][family] += 1
            if status == CORRECT:
                tally["false_accept"][family] += 1
                examples.append({"kind": "false_accept", "problem": p["id"], "mutation": name, "answer": text[:300]})
        for name, text in equivalents.items():
            status = verify_answer(text, p).status
            trials["false_reject"][family] += 1
            if status != CORRECT:
                tally["false_reject"][family] += 1
                reject_statuses[status] += 1
                examples.append({"kind": "false_reject", "problem": p["id"], "rewrite": name, "status": status, "answer": text[:300]})
    summary = {}
    for kind in ("false_accept", "false_reject"):
        per = {f: {"errors": tally[kind][f], "trials": trials[kind][f], "rate": round(tally[kind][f] / trials[kind][f], 4),
                   "wilson95": wilson(tally[kind][f], trials[kind][f])} for f in sorted(trials[kind])}
        k, n = sum(tally[kind].values()), sum(trials[kind].values())
        per["all"] = {"errors": k, "trials": n, "rate": round(k / n, 4), "wilson95": wilson(k, n)}
        summary[kind] = per
    return {"summary": summary, "false_reject_statuses": dict(reject_statuses), "dropped_before_scoring": dict(dropped),
            "examples": examples[:40],
            "known_blind_spots": [
                "Dimension checks ignore numeric prefactors by design, so 2*pi*hbar passes where hbar is wanted.",
                "Math checks are numeric at 8 points to a relative 1e-30, plus an optional symbolic proof: a difference smaller than that is accepted.",
                "Every domain is in x > 0, so identities such as sqrt(x**2) = x are accepted; they are true there.",
            ]}


def verdict_sample(rows: list[dict], problems: dict, size: int = 20) -> list[dict]:
    rng = random.Random(SEED)
    quotas = [("wrong", 7), ("correct", 7), ("other", 6)]
    pools = {"wrong": [r for r in rows if r["status"] == "wrong"],
             "correct": [r for r in rows if r["status"] == "correct"],
             "other": [r for r in rows if r["status"] not in ("correct", "wrong")]}
    picked = []
    for name, quota in quotas:
        pool = sorted(pools[name], key=lambda r: (r["problem_id"], r["model"]))
        picked += rng.sample(pool, min(quota, len(pool)))
    short = len(picked) < size
    if short:
        rest = [r for r in rows if r not in picked]
        picked += rng.sample(rest, size - len(picked))
    sample = []
    for r in picked:
        p = problems[r["problem_id"]]
        truth = p.get("reference") or p.get("truth") or p.get("reference_exponents")
        sample.append({"problem_id": r["problem_id"], "model": r["model"], "statement": p["statement"],
                       "truth_or_reference": truth, "model_answer": (r["answer"] or "")[:600],
                       "verdict": r["status"], "reason": r["reason"], "method": r["method"]})
    return sample


def sample_markdown(sample: list[dict]) -> str:
    lines = ["# Verifier audit: 20 real verdicts to check by hand", "",
             "Status: **NOT YET CHECKED.** Jordan has not reviewed these. Tick a box only after checking it yourself.", "",
             "For each one, decide whether the verdict is right. For math problems, paste the answer and the "
             "reference into SymPy or WolframAlpha and compare (for antiderivatives, differentiate the answer). "
             "For `monomial` problems, multiply out the dimensions by hand. `no_answer`, `truncated` and `unparseable` "
             "are format verdicts: check that the response really lacks a valid final `ANSWER:` line, not whether the math is right. "
             "Full responses are in `data/raw/<model>.jsonl`; search for the problem id.", ""]
    for i, s in enumerate(sample, 1):
        lines += [f"## {i}. {s['problem_id']} / {s['model']} -> **{s['verdict']}** {('(' + s['reason'] + ')') if s['reason'] else ''} {('[' + s['method'] + ']') if s['method'] else ''}",
                  "", f"Problem: {s['statement']}", "", f"Reference: `{s['truth_or_reference']}`", "",
                  f"Model answer: `{s['model_answer'] or '(none)'}`", "", "- [ ] verdict is right   - [ ] verdict is wrong   Notes:", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problems", default="data/problems.jsonl")
    parser.add_argument("--verdicts", default="results/verdicts.jsonl")
    parser.add_argument("--out", default="results/verifier_audit.json")
    parser.add_argument("--sample-markdown", default=None, help="also write the hand-check sheet here")
    args = parser.parse_args()
    problems = load(args.problems)
    audit = injected_audit(problems)
    sample = verdict_sample(read_rows(Path(args.verdicts)), {p["id"]: p for p in problems})
    audit["hand_check_sample"] = sample
    Path(args.out).write_text(json.dumps(audit, indent=1) + "\n")
    print(json.dumps(audit["summary"]["false_accept"]["all"]), json.dumps(audit["summary"]["false_reject"]["all"]))
    if args.sample_markdown:
        Path(args.sample_markdown).write_text(sample_markdown(sample))


if __name__ == "__main__":
    main()
