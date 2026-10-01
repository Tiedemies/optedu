# tests/test_dfo.py
# Derivative-free local methods: Nelder–Mead (§6.1.1) and Hooke–Jeeves (§6.1.2).
import numpy as np
import pytest

from helpers import Counter, check_result
from optedu.algorithms.hooke_jeeves import hooke_jeeves
from optedu.algorithms.nelder_mead import _diameter, _init_simplex, nelder_mead
from optedu.problems.himmelblau import Himmelblau
from optedu.problems.rosenbrock import Rosenbrock

NM_OPS = {"reflect", "expand", "outside_contract", "inside_contract", "shrink"}
HJ_OPS = {"init", "explore", "pattern", "reduce"}


# ---------------- Nelder–Mead ----------------

def test_nelder_mead_on_convex_quadratic():
    a = np.array([1.0, -2.0])
    f = lambda x: float(np.sum((np.asarray(x) - a) ** 2))      # f(x) = ||x - a||^2
    out = nelder_mead(f=f, x0=np.array([5.0, 5.0]), maxit=5000, tol=1e-8)
    check_result(out)
    assert out["status"] == "converged"
    assert np.allclose(out["x"], a, atol=1e-4)


def test_nelder_mead_on_rosenbrock():
    prob = Rosenbrock(n=2)
    out = nelder_mead(f=prob.f, x0=np.array([-1.2, 1.0]), maxit=2000)
    assert out["status"] == "converged" and np.allclose(out["x"], [1.0, 1.0], atol=1e-4)


def test_nelder_mead_history():
    prob = Himmelblau()
    out = nelder_mead(f=prob.f, x0=np.array([4.5, 4.5]), initial_step=0.5)
    hist = out["history"]
    vals = np.asarray(hist["f"])
    assert np.all(vals[1:] <= vals[:-1])                        # best-so-far value never increases
    assert len(hist["step"]) == len(hist["f"])                  # step = simplex diameter per iteration
    assert hist["step"][-1] <= 1e-8
    assert len(hist["meta"]["op"]) == out["counts"]["nit"]      # one operation per iteration
    assert set(hist["meta"]["op"]) <= NM_OPS


def test_nelder_mead_shrinks_and_stalls_on_a_staircase():
    # f is piecewise constant (a staircase). On a flat step no contraction improves, so the simplex
    # shrinks until it collapses: NM "converges" without reaching the minimum f = 0.
    f = lambda x: float(np.floor(4 * (x[0] ** 2 + x[1] ** 2)))
    out = nelder_mead(f=f, x0=np.array([1.0, 0.3]), maxit=500)
    assert "shrink" in out["history"]["meta"]["op"]
    assert out["status"] == "converged" and out["f"] > 0


def test_nelder_mead_shrinks_on_a_nonsmooth_ridge():
    # Along the kink x1 = x2^2 of this function even the outside contraction can fail, which also ends in a shrink.
    f = lambda x: float(abs(x[0] - x[1] ** 2) + 0.01 * x[0] ** 2)
    out = nelder_mead(f=f, x0=np.array([0.1, 2.0]), maxit=300)
    assert "shrink" in out["history"]["meta"]["op"]
    assert out["f"] < f(np.array([0.1, 2.0]))


def test_nelder_mead_tiny_initial_simplex_stops_immediately():
    out = nelder_mead(f=Himmelblau().f, x0=np.array([3.0, 2.0]), initial_step=1e-10)
    assert out["status"] == "converged" and out["counts"]["nit"] == 0


def test_initial_simplex():
    s = _init_simplex(np.array([2.0, 0.0]))
    assert s.shape == (3, 2)                                    # n + 1 vertices
    assert np.allclose(s[1] - s[0], [0.1, 0.0])                 # 5% of |x0_1| = 0.1
    assert np.allclose(s[2] - s[0], [0.0, 0.05])                # x0_2 = 0 -> 0.05
    s = _init_simplex(np.array([2.0, 0.0]), initial_step=0.5)
    assert np.allclose(s[1:] - s[0], 0.5 * np.eye(2))
    assert _diameter(s) == pytest.approx(0.5 * np.sqrt(2))


def test_nelder_mead_counts_match_actual_calls():
    f = Counter(Rosenbrock().f)
    out = nelder_mead(f=f, x0=np.array([-1.2, 1.0]), maxit=100)
    assert out["counts"]["nfev"] == f.calls


# ---------------- Hooke–Jeeves ----------------

def test_hooke_jeeves_on_himmelblau():
    prob = Himmelblau()
    out = hooke_jeeves(f=prob.f, x0=np.zeros(2))
    check_result(out)
    assert out["status"] == "converged"
    assert np.allclose(out["x"], [3.0, 2.0], atol=1e-5)


def test_hooke_jeeves_on_rosenbrock():
    prob = Rosenbrock(n=2)
    out = hooke_jeeves(f=prob.f, x0=np.array([-1.2, 1.0]), maxit=5000, tol=1e-8)
    assert out["status"] == "converged" and np.allclose(out["x"], [1.0, 1.0], atol=1e-4)


def test_hooke_jeeves_history():
    prob = Rosenbrock(n=2)
    out = hooke_jeeves(f=prob.f, x0=np.array([-1.2, 1.0]), maxit=200)
    hist = out["history"]
    ops = hist["meta"]["op"]
    assert ops[0] == "init" and set(ops) == HJ_OPS
    assert len(ops) == len(hist["f"]) == len(hist["step"])      # one entry per logged move
    assert np.all(np.diff(hist["f"]) <= 0)                      # the base point never gets worse
    # the step length only changes on 'reduce', and then by the factor theta = 0.5
    for k in range(1, len(ops)):
        expected = hist["step"][k - 1] * (0.5 if ops[k] == "reduce" else 1.0)
        assert hist["step"][k] == pytest.approx(expected)


def test_hooke_jeeves_stops_when_steps_are_small():
    out = hooke_jeeves(f=Himmelblau().f, x0=np.zeros(2), delta0=0.5, theta=0.5, tol=1e-3)
    assert out["history"]["step"][-1] <= 1e-3


def test_hooke_jeeves_counts_match_actual_calls():
    f = Counter(Rosenbrock().f)
    out = hooke_jeeves(f=f, x0=np.array([-1.2, 1.0]), maxit=100)
    assert out["counts"]["nfev"] == f.calls
