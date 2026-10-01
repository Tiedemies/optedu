# tests/test_metaheuristics.py
# Population-based and stochastic methods: genetic algorithm (§6.2.1), particle swarm, simulated annealing.
# All take an rng; with the same seed a run must be exactly reproducible.
import numpy as np
import pytest

from helpers import Counter, check_result, sphere
from optedu.algorithms.genetic import (
    _clip_to_bounds, _gaussian_mutation, _tournament_select, _uniform_crossover, genetic_minimize,
)
from optedu.algorithms.pso import pso_minimize
from optedu.algorithms.sa import simulated_annealing

BOUNDS = [(-5.0, 5.0)] * 5


def run_ga(seed=42, f=sphere, **kw):
    return genetic_minimize(f=f, bounds=BOUNDS, pop_size=30, generations=60, rng=np.random.default_rng(seed), **kw)

def run_pso(seed=123, f=sphere, **kw):
    return pso_minimize(f, bounds=BOUNDS, n_particles=25, iters=80, rng=np.random.default_rng(seed), **kw)

def run_sa(seed=999, f=sphere, **kw):
    kw.setdefault("iters", 500)
    return simulated_annealing(f, np.ones(5) * 4.0, rng=np.random.default_rng(seed), **kw)

RUNNERS = {"ga": run_ga, "pso": run_pso, "sa": run_sa}


# ---------------- shared behaviour ----------------

@pytest.mark.parametrize("name", RUNNERS)
def test_finds_near_optimal_point_on_sphere(name):
    out = RUNNERS[name]()
    check_result(out)
    assert out["status"] == "maxit"            # fixed budget methods
    assert out["f"] < 1.0
    assert out["f"] == pytest.approx(sphere(out["x"]))


@pytest.mark.parametrize("name", RUNNERS)
def test_same_seed_same_result(name):
    a, b = RUNNERS[name](seed=7), RUNNERS[name](seed=7)
    assert np.array_equal(a["x"], b["x"]) and a["f"] == b["f"]
    c = RUNNERS[name](seed=8)
    assert not np.array_equal(a["x"], c["x"])


@pytest.mark.parametrize("name", RUNNERS)
def test_counts_match_actual_calls(name):
    f = Counter(sphere)
    out = RUNNERS[name](f=f)
    assert out["counts"]["nfev"] == f.calls


@pytest.mark.parametrize("name", ["ga", "pso"])
def test_best_so_far_never_gets_worse(name):
    vals = np.asarray(RUNNERS[name]()["history"]["f"])
    assert np.all(vals[1:] <= vals[:-1])


@pytest.mark.parametrize("name", ["ga", "pso"])
def test_points_stay_inside_the_box(name):
    for x in RUNNERS[name]()["history"]["x"]:
        assert np.all(x >= -5.0) and np.all(x <= 5.0)


# ---------------- genetic algorithm ----------------

def test_ga_history_length_and_evaluations():
    out = run_ga()
    assert len(out["history"]["f"]) == 60 + 1            # generation 0 + 60 generations
    assert out["counts"] == {"nit": 60, "nfev": 30 * 61}


@pytest.mark.parametrize("bounds", [[1.0, 2.0], [(1.0, 0.0)]])
def test_ga_rejects_bad_bounds(bounds):
    with pytest.raises(ValueError):
        genetic_minimize(f=sphere, bounds=bounds)


def test_tournament_selection_picks_the_fitter_individual():
    rng = np.random.default_rng(0)
    fitness = np.array([5.0, 1.0, 3.0])
    picks = [_tournament_select(rng, fitness, k=3) for _ in range(50)]
    # with k = 3 the best of the three sampled indices wins; index 0 (worst) can only win if sampled alone
    assert picks.count(1) > picks.count(0)


def test_uniform_crossover_takes_each_gene_from_a_parent():
    rng = np.random.default_rng(0)
    a, b = np.zeros(10), np.ones(10)
    child = _uniform_crossover(rng, a, b)
    assert set(child) <= {0.0, 1.0} and 0 < child.sum() < 10


def test_gaussian_mutation_probability():
    rng = np.random.default_rng(0)
    x = np.zeros(20)
    assert np.array_equal(_gaussian_mutation(rng, x, 1.0, p_mut=0.0), x)      # never mutate
    assert np.all(_gaussian_mutation(rng, x, 1.0, p_mut=1.0) != 0.0)          # always mutate


def test_clip_to_bounds():
    assert np.array_equal(_clip_to_bounds(np.array([-9.0, 0.5, 9.0]), -np.ones(3), np.ones(3)), [-1.0, 0.5, 1.0])


# ---------------- particle swarm ----------------

def test_pso_history_length():
    out = run_pso()
    assert len(out["history"]["x"]) == len(out["history"]["f"]) == 80 + 1
    assert out["counts"]["nit"] == 80


# ---------------- simulated annealing ----------------

def test_sa_history():
    out = run_sa(iters=50)
    hist = out["history"]
    assert len(hist["x"]) == len(hist["f"]) == 50 + 1       # current point per iteration
    best = [v for v, _ in hist["meta"]["best"]]
    assert np.all(np.diff(best) <= 0)                        # best-so-far never gets worse
    assert out["f"] == best[-1]


def test_sa_at_zero_temperature_only_accepts_improvements():
    # T0 -> 0: exp(-delta / T) = 0 for any worse point, so SA becomes a greedy random search.
    vals = np.asarray(run_sa(T0=1e-300, iters=200)["history"]["f"])
    assert np.all(np.diff(vals) <= 0)


def test_sa_at_high_temperature_accepts_worse_points():
    vals = np.asarray(run_sa(T0=1e6, alpha=1.0, iters=200)["history"]["f"])
    assert np.any(np.diff(vals) > 0)


def test_sa_respects_bounds():
    out = simulated_annealing(sphere, np.ones(2) * 0.9, bounds=[(0.5, 1.0)] * 2, iters=200,
                              rng=np.random.default_rng(0))
    for x in out["history"]["x"]:
        assert np.all(x >= 0.5) and np.all(x <= 1.0)
    assert np.allclose(out["x"], [0.5, 0.5], atol=0.05)     # the minimizer of ||x||^2 on the box
