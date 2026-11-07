import numpy as np

from vmr.bandit import bootstrap_orders, final_third_share, shuffled_orders, thompson, uniform_random
from vmr.matrix import Matrix


def fixture(p_by_family, n=60, seed=0):
    rng = np.random.default_rng(seed)
    p = np.array(p_by_family, dtype=float)
    families = np.repeat(np.arange(p.shape[0]), n)
    correct = rng.random((families.size, p.shape[1])) < p[families]
    return Matrix([str(i) for i in range(families.size)], families, [f"f{i}" for i in range(p.shape[0])],
                  np.ones(families.size, dtype=int), [f"m{i}" for i in range(p.shape[1])], correct,
                  np.tile(np.array([0.001, 0.01, 0.1])[: p.shape[1]], (families.size, 1)),
                  np.where(correct, "correct", "wrong"))


def test_orders_are_permutations_and_bootstrap_keeps_family_sizes():
    m = fixture([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
    rng = np.random.default_rng(1)
    order = shuffled_orders(len(m.problem_ids), 10, rng)
    assert all(sorted(row) == list(range(len(m.problem_ids))) for row in order)
    sample = bootstrap_orders(m.families, 10, rng)
    assert all((m.families[row] == 0).sum() == 60 for row in sample)


def test_learns_specialists_on_synthetic_matrix():
    m = fixture([[0.95, 0.1, 0.1], [0.1, 0.95, 0.1], [0.1, 0.1, 0.95]])
    rng = np.random.default_rng(2)
    order = shuffled_orders(len(m.problem_ids), 300, rng)
    replay = thompson(m, order, rng)
    shares = final_third_share(m, order, replay)
    assert (shares.argmax(axis=1) == np.arange(3)).all()
    assert shares.diagonal().min() > 0.9
    assert replay.accuracy.mean() > uniform_random(m, order, rng).accuracy.mean() + 0.4


def test_context_free_router_cannot_exploit_specialists():
    m = fixture([[0.95, 0.1, 0.1], [0.1, 0.95, 0.1], [0.1, 0.1, 0.95]])
    rng = np.random.default_rng(3)
    order = shuffled_orders(len(m.problem_ids), 300, rng)
    assert thompson(m, order, rng).accuracy.mean() > thompson(m, order, rng, contextual=False).accuracy.mean() + 0.2


def test_cost_aware_router_prefers_cheap_arm_when_quality_ties():
    m = fixture([[0.8, 0.8, 0.8]], n=150)
    rng = np.random.default_rng(4)
    order = shuffled_orders(len(m.problem_ids), 200, rng)
    assert thompson(m, order, rng, value_per_correct=0.02).mean_cost.mean() < thompson(m, order, rng).mean_cost.mean() / 3


def test_reward_noise_never_changes_scored_truth():
    m = fixture([[0.9, 0.2, 0.2]], n=100)
    rng = np.random.default_rng(5)
    order = shuffled_orders(len(m.problem_ids), 50, rng)
    noisy = thompson(m, order, rng, false_accept=0.5, false_reject=0.5)
    assert (noisy.correct == m.correct[order, noisy.picks]).all()
    assert noisy.accuracy.mean() < thompson(m, order, rng).accuracy.mean()
