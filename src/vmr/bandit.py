"""Thompson sampling replayed over a complete response matrix.

Every model answered every problem, so the replay needs no off-policy correction: the router sees
only the reward of the arm it chose, and the arms it did not choose are simply never revealed to it.

`order` is a (runs, steps) array of problem indices. A permutation of all problems gives the
"shuffled order" replay, and sampling with replacement inside each family gives the problem bootstrap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vmr.matrix import Matrix


@dataclass
class Replay:
    picks: np.ndarray  # (runs, steps) arm index
    correct: np.ndarray  # (runs, steps) bool, the true verifier outcome of each pick
    cost: np.ndarray  # (runs, steps)
    alpha: np.ndarray  # (runs, contexts, arms) final posterior
    beta: np.ndarray

    @property
    def accuracy(self) -> np.ndarray:
        return self.correct.mean(axis=1)

    @property
    def mean_cost(self) -> np.ndarray:
        return self.cost.mean(axis=1)


def shuffled_orders(problem_count: int, runs: int, rng: np.random.Generator) -> np.ndarray:
    return rng.permuted(np.tile(np.arange(problem_count), (runs, 1)), axis=1)


def bootstrap_orders(families: np.ndarray, runs: int, rng: np.random.Generator) -> np.ndarray:
    """Resample problems with replacement inside each family, then shuffle the visiting order."""
    columns = []
    for family in np.unique(families):
        members = np.flatnonzero(families == family)
        columns.append(rng.choice(members, size=(runs, members.size), replace=True))
    return rng.permuted(np.concatenate(columns, axis=1), axis=1)


def thompson(
    matrix: Matrix,
    order: np.ndarray,
    rng: np.random.Generator,
    contextual: bool = True,
    value_per_correct: float | None = None,
    false_accept: float = 0.0,
    false_reject: float = 0.0,
) -> Replay:
    """Run one Beta-Bernoulli Thompson sampler per order, with Beta(1, 1) priors.

    With `value_per_correct` set, the router maximises sampled accuracy minus running mean cost
    divided by that dollar value, so V is what a correct answer is worth. `false_accept` and
    `false_reject` corrupt the reward the router sees, but never the accuracy it is scored on.
    """
    runs, steps = order.shape
    arms = len(matrix.models)
    contexts = len(matrix.family_names) if contextual else 1
    alpha = np.ones((runs, contexts, arms))
    beta = np.ones((runs, contexts, arms))
    spend = np.zeros((runs, contexts, arms))
    pulls = np.zeros((runs, contexts, arms))
    picks = np.zeros((runs, steps), dtype=np.int64)
    correct = np.zeros((runs, steps), dtype=bool)
    cost = np.zeros((runs, steps))
    row = np.arange(runs)
    for t in range(steps):
        problem = order[:, t]
        context = matrix.families[problem] if contextual else np.zeros(runs, dtype=np.int64)
        score = rng.beta(alpha[row, context], beta[row, context])
        if value_per_correct is not None:
            seen = pulls[row, context]
            average = np.divide(spend[row, context], seen, out=np.zeros((runs, arms)), where=seen > 0)
            score = score - average / value_per_correct
        arm = score.argmax(axis=1)
        truth = matrix.correct[problem, arm]
        paid = matrix.cost[problem, arm]
        reward = truth.copy()
        if false_accept or false_reject:
            noise = rng.random(runs)
            reward = np.where(truth, noise >= false_reject, noise < false_accept)
        alpha[row, context, arm] += reward
        beta[row, context, arm] += ~reward
        spend[row, context, arm] += paid
        pulls[row, context, arm] += 1
        picks[:, t], correct[:, t], cost[:, t] = arm, truth, paid
    return Replay(picks, correct, cost, alpha, beta)


def uniform_random(matrix: Matrix, order: np.ndarray, rng: np.random.Generator) -> Replay:
    runs, steps = order.shape
    arm = rng.integers(0, len(matrix.models), size=(runs, steps))
    empty = np.zeros((runs, 1, len(matrix.models)))
    return Replay(arm, matrix.correct[order, arm], matrix.cost[order, arm], empty, empty)


def final_third_share(matrix: Matrix, order: np.ndarray, replay: Replay) -> np.ndarray:
    """Share of picks per (family, model) over the last third of each family's visits in each run."""
    shares = np.zeros((len(matrix.family_names), len(matrix.models)))
    families = matrix.families[order]
    for f in range(len(matrix.family_names)):
        mask = families == f
        position = mask.cumsum(axis=1)
        total = mask.sum(axis=1, keepdims=True)
        late = mask & (position > (2 * total) // 3)
        chosen = replay.picks[late]
        shares[f] = np.bincount(chosen, minlength=len(matrix.models)) / chosen.size
    return shares
