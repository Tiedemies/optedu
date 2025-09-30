# tests/test_metaheuristics_smoke.py
import numpy as np
import pytest

# Expect GA / PSO / SA in these modules per README section on DFO/metaheuristics
from optedu.algorithms.genetic import genetic_minimize  # type: ignore
from optedu.algorithms.pso import pso_minimize  # type: ignore
from optedu.algorithms.sa import simulated_annealing  # type: ignore

def sphere(x):
    x = np.asarray(x, float)
    return float(np.dot(x, x))

def test_ga_smoke_seeded_reduces_value():
    rng = np.random.default_rng(42)
    out = genetic_minimize(
        f=sphere, bounds=[(-5, 5)]*5, pop_size=30, generations=60, p_mut=0.1, elitism=True, rng=rng
    )
    assert np.isfinite(out["best_value"])
    assert out["best_value"] < 1.0  # should get near zero

def test_pso_smoke_seeded_reduces_value():
    rng = np.random.default_rng(123)
    out = pso_minimize(
        f=sphere, bounds=[(-5, 5)]*5, n_particles=25, iters=80, w=0.7, c1=1.4, c2=1.4, rng=rng
    )
    assert np.isfinite(out["best_value"])
    assert out["best_value"] < 1.0

def test_sa_smoke_seeded_reduces_value():
    rng = np.random.default_rng(999)
    out = simulated_annealing(
        f=sphere, x0=np.ones(5)*4, T0=1.0, alpha=0.95, iters=500, step_scale=0.5, rng=rng
    )
    assert np.isfinite(out["best_value"])
    assert out["best_value"] < 1.0
