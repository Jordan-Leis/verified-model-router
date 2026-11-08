"""All headline numbers: per-model accuracy, routing policies, and the specialization test.

Pre-registered decision rule (committed before the main run's responses existed):
the models count as *specialized at the family level* only if all three hold:
  1. in at least one family, the best model differs from the best model overall;
  2. a permutation test of the specialization headroom H gives p < 0.05, where
     H = (accuracy of routing each family to its best model) - (accuracy of the best single model);
  3. the mean split-half gain is at least 2 percentage points. Pick the per-family best and the overall
     best on one half of the problems, then score both on the other half. This removes the
     hindsight bias that makes H > 0 even when nothing is specialized.
If any condition fails, the README reports that the models are not specialized on these problems.

Two kinds of interval are reported and kept apart:
  * order interval: spread over shuffled visiting orders and Thompson randomness, with the problems fixed.
    It measures how much the router's luck matters, not how general the result is.
  * problem interval: a bootstrap over problems, resampled within each family. It is the honest
    uncertainty for "would this hold on another draw of similar problems".
Neither captures model sampling noise, because every model answered every problem exactly once.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from vmr.bandit import bootstrap_orders, final_third_share, shuffled_orders, thompson, uniform_random
from vmr.matrix import Matrix, read_rows, to_matrix

SEED = 20261107
RUNS = 2000
PERMUTATIONS = 5000
VALUES_PER_CORRECT = [0.001, 0.003, 0.01, 0.03, 0.1]
NOISE_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4]


def r4(value) -> float:
    return round(float(value), 4)


def wilson(k: int, n: int, z: float = 1.959964) -> list[float]:
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [r4(max(0.0, centre - half)), r4(min(1.0, centre + half))]


def interval(samples: np.ndarray) -> list[float]:
    return [r4(np.percentile(samples, 2.5)), r4(np.percentile(samples, 97.5))]


def exact_mcnemar(a: np.ndarray, b: np.ndarray) -> dict:
    only_a, only_b = int((a & ~b).sum()), int((b & ~a).sum())
    n = only_a + only_b
    tail = sum(math.comb(n, k) for k in range(min(only_a, only_b) + 1)) / 2**n if n else 1.0
    return {"only_first_correct": only_a, "only_second_correct": only_b, "p_two_sided": r4(min(1.0, 2 * tail))}


# --- Hindsight and oracle baselines, computed on any set of problem indices ---------------------

def best_arm(correct: np.ndarray, cost: np.ndarray) -> int:
    """Highest accuracy, with ties broken by lower mean cost."""
    acc, spend = correct.mean(axis=0), cost.mean(axis=0)
    return int(np.lexsort((spend, -acc))[0])


def family_best_arms(m: Matrix, rows: np.ndarray) -> np.ndarray:
    return np.array([best_arm(m.correct[rows][m.families[rows] == f], m.cost[rows][m.families[rows] == f])
                     if (m.families[rows] == f).any() else 0 for f in range(len(m.family_names))])


def cascade_order(m: Matrix) -> np.ndarray:
    return np.argsort(m.cost.mean(axis=0))


def pilot_family_arms(m: Matrix, pilots: Path = Path("results/pilots.json")) -> np.ndarray | None:
    """Per-family best model on pilot 2, chosen before the main run existed: a baseline without hindsight.

    Ties go to the model listed first in data/models.json, which is the cheapest.
    """
    if not pilots.exists():
        return None
    scores = json.loads(pilots.read_text())["pilot2"]
    arms = []
    for family in m.family_names:
        accuracy = [scores[f"{family}|{model}"]["accuracy"] for model in m.models]
        arms.append(int(np.argmax(accuracy)))
    return np.array(arms)


def static_policies(m: Matrix, rows: np.ndarray, pilot_arms: np.ndarray | None = None) -> dict[str, tuple[float, float]]:
    """(accuracy, mean cost per problem) for the deterministic policies on the given problems."""
    correct, cost, fam = m.correct[rows], m.cost[rows], m.families[rows]
    out = {f"only {name}": (correct[:, k].mean(), cost[:, k].mean()) for k, name in enumerate(m.models)}
    single = best_arm(correct, cost)
    out["best single model (hindsight)"] = (correct[:, single].mean(), cost[:, single].mean())
    per_family = family_best_arms(m, rows)[fam]
    pick = np.arange(rows.size)
    out["best model per family (hindsight)"] = (correct[pick, per_family].mean(), cost[pick, per_family].mean())
    if pilot_arms is not None:
        chosen = pilot_arms[fam]
        out["best model per family, chosen on pilot 2 (no hindsight)"] = (correct[pick, chosen].mean(), cost[pick, chosen].mean())
    any_correct = correct.any(axis=1)
    cheapest_correct = np.where(correct, cost, np.inf).min(axis=1)
    out["per-problem oracle"] = (any_correct.mean(), np.where(any_correct, cheapest_correct, cost.min(axis=1)).mean())
    order = cascade_order(m)
    spent = np.zeros(rows.size)
    solved = np.zeros(rows.size, dtype=bool)
    for k in order:
        spent += np.where(solved, 0.0, cost[:, k])
        solved |= correct[:, k]
    out["verify-and-escalate cascade (cheapest first)"] = (solved.mean(), spent.mean())
    return out


def headroom(m: Matrix, rows: np.ndarray, families: np.ndarray) -> float:
    correct, cost = m.correct[rows], m.cost[rows]
    total = 0
    for f in np.unique(families):
        mask = families == f
        total += correct[mask][:, best_arm(correct[mask], cost[mask])].sum()
    return total / rows.size - correct[:, best_arm(correct, cost)].mean()


def split_half_gains(m: Matrix, rng: np.random.Generator, splits: int = RUNS) -> np.ndarray:
    gains = np.zeros(splits)
    for s in range(splits):
        train, test = [], []
        for f in range(len(m.family_names)):
            members = rng.permutation(np.flatnonzero(m.families == f))
            half = members.size // 2
            train += list(members[:half])
            test += list(members[half:])
        train, test = np.array(train), np.array(test)
        single = best_arm(m.correct[train], m.cost[train])
        per_family = family_best_arms(m, train)[m.families[test]]
        gains[s] = m.correct[test, per_family].mean() - m.correct[test, single].mean()
    return gains


# --- Main ---------------------------------------------------------------------------------------

def model_table(rows: list[dict], m: Matrix) -> dict:
    out = {}
    for model in m.models:
        mine = [r for r in rows if r["model"] == model]
        counts = Counter(r["status"] for r in mine)
        decided = counts["correct"] + counts["wrong"]
        out[model] = {
            "n": len(mine), "status_counts": dict(sorted(counts.items())),
            "strict_accuracy": r4(counts["correct"] / len(mine)), "strict_accuracy_ci95": wilson(counts["correct"], len(mine)),
            "accuracy_when_parsed": r4(counts["correct"] / decided) if decided else None,
            "lenient_accuracy": r4(sum(r["lenient_status"] == "correct" for r in mine) / len(mine)),
            "total_cost_usd": r4(sum(r["cost"] for r in mine)), "mean_cost_usd": round(sum(r["cost"] for r in mine) / len(mine), 6),
            "mean_completion_tokens": round(sum(r["completion_tokens"] for r in mine) / len(mine)),
            "verified_by": dict(Counter(r["method"] for r in mine if r["status"] == "correct")),
        }
    return out


def cell_table(rows: list[dict], key: str) -> dict:
    out, spend = {}, Counter()
    for r in rows:
        name = f"{r[key]}|{r['model']}"
        out.setdefault(name, Counter())[r["status"]] += 1
        spend[name] += r["cost"]
    table = {}
    for name, counts in sorted(out.items()):
        n = sum(counts.values())
        table[name] = {"correct": counts["correct"], "n": n, "accuracy": r4(counts["correct"] / n),
                       "ci95": wilson(counts["correct"], n), "mean_cost_usd": round(spend[name] / n, 6),
                       "status_counts": dict(sorted(counts.items()))}
    return table


def run(verdicts: Path, out_dir: Path, source: str) -> dict:
    rows = read_rows(verdicts)
    m = to_matrix(rows)
    rng = np.random.default_rng(SEED)
    everything = np.arange(len(m.problem_ids))
    shuffles = shuffled_orders(everything.size, RUNS, rng)
    boots = bootstrap_orders(m.families, RUNS, rng)

    policies = {}
    pilot_arms = pilot_family_arms(m)
    static = static_policies(m, everything, pilot_arms)
    boot_static = [static_policies(m, b, pilot_arms) for b in boots]
    for name, (acc, cost) in static.items():
        policies[name] = {"accuracy": r4(acc), "mean_cost_usd": round(float(cost), 6),
                          "accuracy_problem_interval": interval(np.array([b[name][0] for b in boot_static]))}

    dynamic = {
        "Thompson sampling per family": dict(contextual=True),
        "Thompson sampling, no family context": dict(contextual=False),
    }
    replays = {}
    for name, kwargs in dynamic.items():
        replay = thompson(m, shuffles, rng, **kwargs)
        boot = thompson(m, boots, rng, **kwargs)
        replays[name] = (replay, boot)
        policies[name] = {"accuracy": r4(replay.accuracy.mean()), "mean_cost_usd": round(float(replay.mean_cost.mean()), 6),
                          "accuracy_order_interval": interval(replay.accuracy),
                          "accuracy_problem_interval": interval(boot.accuracy)}
    random_replay = uniform_random(m, shuffles, rng)
    policies["uniform random"] = {"accuracy": r4(random_replay.accuracy.mean()),
                                  "mean_cost_usd": round(float(random_replay.mean_cost.mean()), 6),
                                  "accuracy_order_interval": interval(random_replay.accuracy),
                                  "accuracy_problem_interval": interval(uniform_random(m, boots, rng).accuracy)}

    ts, ts_boot = replays["Thompson sampling per family"]
    flat, flat_boot = replays["Thompson sampling, no family context"]
    single_boot = np.array([b["best single model (hindsight)"][0] for b in boot_static])
    paired = {
        "Thompson per family minus best single (hindsight)": interval(ts_boot.accuracy - single_boot),
        "Thompson per family minus no-context Thompson": interval(ts_boot.accuracy - flat_boot.accuracy),
        "Thompson per family minus uniform random": interval(ts_boot.accuracy - uniform_random(m, boots, rng).accuracy),
    }
    paired_means = {
        "Thompson per family minus best single (hindsight)": r4(ts.accuracy.mean() - static["best single model (hindsight)"][0]),
        "Thompson per family minus no-context Thompson": r4(ts.accuracy.mean() - flat.accuracy.mean()),
    }

    cost_aware = []
    for value in VALUES_PER_CORRECT:
        replay = thompson(m, shuffles, rng, value_per_correct=value)
        cost_aware.append({"value_per_correct_usd": value, "accuracy": r4(replay.accuracy.mean()),
                           "accuracy_order_interval": interval(replay.accuracy),
                           "mean_cost_usd": round(float(replay.mean_cost.mean()), 6)})

    noise = []
    for level in NOISE_LEVELS:
        replay = thompson(m, shuffles, rng, false_accept=level, false_reject=level)
        noise.append({"false_accept_and_false_reject_rate": level, "true_accuracy": r4(replay.accuracy.mean()),
                      "accuracy_order_interval": interval(replay.accuracy)})

    # Specialization.
    overall_best = best_arm(m.correct, m.cost)
    fam_best = family_best_arms(m, everything)
    h = headroom(m, everything, m.families)
    null = np.array([headroom(m, everything, rng.permutation(m.families)) for _ in range(PERMUTATIONS)])
    p_value = (1 + (null >= h - 1e-12).sum()) / (1 + PERMUTATIONS)
    gains = split_half_gains(m, rng)
    per_family = {}
    for f, name in enumerate(m.family_names):
        mask = m.families == f
        per_family[name] = {
            "best_model": m.models[fam_best[f]],
            "accuracy_by_model": {model: r4(m.correct[mask, k].mean()) for k, model in enumerate(m.models)},
            "best_vs_overall_best_mcnemar": exact_mcnemar(m.correct[mask, fam_best[f]], m.correct[mask, overall_best]),
        }
    rule = {
        "1_some_family_best_differs": bool((fam_best != overall_best).any()),
        "2_permutation_p_below_0.05": bool(p_value < 0.05),
        "3_split_half_gain_at_least_0.02": bool(gains.mean() >= 0.02),
    }
    specialization = {
        "overall_best_model": m.models[overall_best], "per_family": per_family,
        "headroom_H": r4(h), "headroom_permutation_p": r4(p_value), "permutations": PERMUTATIONS,
        "split_half_gain_mean": r4(gains.mean()), "split_half_gain_2.5_97.5_percentiles": interval(gains),
        "rule": rule, "specialized": all(rule.values()),
        "pilot2_family_choices": {f: m.models[a] for f, a in zip(m.family_names, pilot_arms)} if pilot_arms is not None else None,
    }

    solved_by = Counter(int(v) for v in m.correct.sum(axis=1))
    overlap = {"problems_solved_by_k_models": {str(k): solved_by.get(k, 0) for k in range(len(m.models) + 1)},
               "pairwise_both_wrong": {f"{m.models[a]} & {m.models[b]}": int((~m.correct[:, a] & ~m.correct[:, b]).sum())
                                       for a in range(len(m.models)) for b in range(a + 1, len(m.models))}}

    trust = final_third_share(m, shuffles, ts)
    results = {
        "source": source, "n_problems": len(m.problem_ids), "models": m.models, "families": m.family_names,
        "runs": RUNS, "seed": SEED,
        "total_cost_usd": r4(m.cost.sum()),
        "models_overall": model_table(rows, m),
        "family_by_model": cell_table(rows, "family"),
        "tier_by_model": cell_table([{**r, "tier": f"{r['family']} T{r['tier']}"} for r in rows], "tier"),
        "policies": policies, "paired_problem_intervals": paired, "paired_means": paired_means,
        "cost_aware_thompson": cost_aware, "reward_noise_simulation": noise,
        "specialization": specialization, "error_overlap": overlap,
        "router_final_third_pick_share": {name: {model: r4(trust[f, k]) for k, model in enumerate(m.models)}
                                          for f, name in enumerate(m.family_names)},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=1, sort_keys=False) + "\n")
    return results


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--verdicts", default="results/verdicts.jsonl")
    parser.add_argument("--out", default="results")
    parser.add_argument("--source", default="openrouter")
    args = parser.parse_args()
    results = run(Path(args.verdicts), Path(args.out), args.source)
    print(json.dumps({k: results["policies"][k]["accuracy"] for k in results["policies"]}, indent=1))
    print("specialized:", results["specialization"]["specialized"], results["specialization"]["rule"])


if __name__ == "__main__":
    main()
