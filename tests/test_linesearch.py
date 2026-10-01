# tests/test_linesearch.py
# Step-size rules: Armijo backtracking, golden-section search, exact (numeric) line search.
import numpy as np
import pytest

from helpers import Counter, quadratic
from optedu.algorithms.linesearch import (
    _golden_section, backtracking_armijo, exact_line_search, exact_quadratic_step,
)
from optedu.problems.rosenbrock import Rosenbrock


# ---------------- Armijo backtracking ----------------

def test_armijo_step_satisfies_sufficient_decrease():
    prob = Rosenbrock()
    x = np.array([-1.2, 1.0]); d = -prob.grad(x)
    c1 = 1e-4
    t, _ = backtracking_armijo(prob.f, prob.grad, x, d, c1=c1)
    # Armijo condition: f(x + t d) <= f(x) + c1 t grad(x)^T d
    assert prob.f(x + t * d) <= prob.f(x) + c1 * t * prob.grad(x) @ d
    # ... and t is the first step of the sequence 1, 1/2, 1/4, ... that satisfies it
    assert prob.f(x + 2 * t * d) > prob.f(x) + c1 * 2 * t * prob.grad(x) @ d


def test_armijo_accepts_full_step_when_it_is_good_enough():
    f, grad, _, _ = quadratic(n=2)
    x = np.zeros(2); d = -grad(x) / 5.0          # short descent step: t = 1 already works
    t, _ = backtracking_armijo(f, grad, x, d)
    assert t == 1.0


def test_armijo_reports_its_function_evaluations():
    prob = Rosenbrock(); f = Counter(prob.f)
    x = np.array([-1.2, 1.0])
    _, neval = backtracking_armijo(f, prob.grad, x, -prob.grad(x))
    assert neval == f.calls


def test_armijo_rejects_ascent_direction():
    f, grad, _, _ = quadratic(n=2)
    x = np.zeros(2)
    with pytest.raises(ValueError):
        backtracking_armijo(f, grad, x, grad(x))   # +grad is an ascent direction


def test_armijo_stops_after_max_backtracks():
    # f returns NaN away from x, so the Armijo test never passes: we must not loop forever.
    f = lambda x: 0.0 if np.allclose(x, 0.0) else np.nan
    grad = lambda x: np.array([1.0])
    t, neval = backtracking_armijo(f, grad, np.zeros(1), np.array([-1.0]), max_backtracks=10)
    assert neval == 11 and t == pytest.approx(0.5 ** 10)


# ---------------- golden-section search ----------------

def test_golden_section_finds_minimum_of_unimodal_function():
    phi = Counter(lambda t: (t - 0.3) ** 2 + 1.0)
    t_star, f_star, neval = _golden_section(phi, 0.0, 0.5, 1.0, tol=1e-10)
    # Near t* the function changes like (t - t*)^2, so with ~1e-16 relative precision in phi the
    # minimizer can only be located to about sqrt(1e-16) = 1e-8: we ask for 1e-6.
    assert t_star == pytest.approx(0.3, abs=1e-6)
    assert f_star == pytest.approx(1.0)
    assert neval == phi.calls


# ---------------- exact line search ----------------

def test_exact_line_search_on_quadratic_matches_formula():
    # For a quadratic the exact step has a closed form: t* = -grad^T d / (d^T Q d)
    f, grad, hess, _ = quadratic(n=3)
    x = np.array([3.0, -1.0, 2.0]); d = -grad(x)
    t, _ = exact_line_search(f, x, d)
    assert t == pytest.approx(exact_quadratic_step(hess(x), grad(x), d), rel=1e-6)


def test_exact_line_search_step_is_a_minimum_along_the_line():
    prob = Rosenbrock()
    x = np.array([-1.2, 1.0]); d = -prob.grad(x)
    t, _ = exact_line_search(prob.f, x, d)
    phi = lambda s: prob.f(x + s * d)
    assert phi(t) <= phi(0.0)
    assert phi(t) <= min(phi(0.99 * t), phi(1.01 * t)) + 1e-12


def test_exact_line_search_reports_its_function_evaluations():
    prob = Rosenbrock(); f = Counter(prob.f)
    x = np.array([-1.2, 1.0])
    _, neval = exact_line_search(f, x, -prob.grad(x))
    assert neval == f.calls


def test_exact_line_search_returns_zero_without_decrease():
    # Along an ascent direction no positive step decreases f: t = 0 tells the caller to fall back.
    f, grad, _, _ = quadratic(n=2)
    x = np.zeros(2)
    t, neval = exact_line_search(f, x, grad(x))
    assert t == 0.0 and neval > 0


def test_exact_quadratic_step_negative_curvature():
    # d^T Q d <= 0: not bounded below along d, the helper returns t = 1
    Q = np.diag([1.0, -1.0])
    assert exact_quadratic_step(Q, np.array([0.0, 1.0]), np.array([0.0, -1.0])) == 1.0
