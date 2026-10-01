# tests/test_unconstrained.py
# Gradient-based methods: gradient descent, Newton, BFGS.
import numpy as np
import pytest

from helpers import Counter, check_result, quadratic
from optedu.algorithms.bfgs import bfgs
from optedu.algorithms.gradient_descent import gradient_descent
from optedu.algorithms.newton import newton, newton_method
from optedu.problems.himmelblau import Himmelblau
from optedu.problems.rosenbrock import Rosenbrock


# ---------------- gradient descent ----------------

@pytest.mark.parametrize("step", ["exact", "armijo"])
def test_gradient_descent_on_quadratic(step):
    f, grad, _, x_star = quadratic(n=4)
    out = gradient_descent(f=f, grad=grad, x0=np.ones(4) * 3.0, maxit=2000, tol=1e-6, step=step)
    check_result(out)
    assert out["status"] == "converged"
    # ||x - x*|| <= ||grad f(x)|| / lambda_min  with lambda_min = 1
    assert np.linalg.norm(out["x"] - x_star) < 1e-5
    # descent method: f never increases
    vals = np.asarray(out["history"]["f"])
    assert np.all(vals[1:] <= vals[:-1] + 1e-12)


def test_one_exact_step_decreases_f_at_least_as_much_as_armijo():
    # The exact step minimizes f along the line, so after ONE step it is never worse than Armijo.
    # (Over many steps this need not hold: exact steps zigzag, and on this problem Armijo needs fewer iterations.)
    f, grad, _, _ = quadratic(n=4)
    x0 = np.ones(4) * 3.0
    exact = gradient_descent(f=f, grad=grad, x0=x0, maxit=1, step="exact")
    armijo = gradient_descent(f=f, grad=grad, x0=x0, maxit=1, step="armijo")
    assert exact["f"] <= armijo["f"]


def test_gradient_descent_zigzags_with_exact_steps():
    # With exact line search consecutive steepest-descent directions are orthogonal: g_{k+1}^T g_k = 0
    f, grad, _, _ = quadratic(n=2)
    out = gradient_descent(f=f, grad=grad, x0=np.array([3.0, 3.0]), maxit=5, step="exact")
    xs = out["history"]["x"]
    for k in range(3):
        assert abs(grad(xs[k + 1]) @ grad(xs[k])) < 1e-6 * np.linalg.norm(grad(xs[k])) ** 2


def test_gradient_descent_on_rosenbrock_makes_progress():
    prob = Rosenbrock(n=2)
    out = gradient_descent(f=prob.f, grad=prob.grad, x0=np.array([-1.2, 1.0]), maxit=2000, tol=1e-6)
    vals = np.asarray(out["history"]["f"])
    assert vals[-1] < 1e-2 * vals[0]
    assert out["history"]["step"][0] is None and all(s > 0 for s in out["history"]["step"][1:])


def test_gradient_descent_starting_at_minimizer_stops_immediately():
    f, grad, _, x_star = quadratic(n=3)
    out = gradient_descent(f=f, grad=grad, x0=x_star)
    assert out["status"] == "converged" and out["counts"]["nit"] == 0


def test_gradient_descent_maxit_status():
    prob = Rosenbrock(n=2)
    out = gradient_descent(f=prob.f, grad=prob.grad, x0=np.array([-1.2, 1.0]), maxit=3)
    assert out["status"] == "maxit" and out["counts"]["nit"] == 3
    assert len(out["history"]["f"]) == 4                       # start + 3 iterations


def test_gradient_descent_rejects_unknown_step():
    f, grad, _, _ = quadratic(n=2)
    with pytest.raises(ValueError):
        gradient_descent(f=f, grad=grad, x0=np.ones(2), step="exact_numeric")


# ---------------- Newton ----------------

def test_newton_solves_quadratic_in_one_step():
    f, grad, hess, x_star = quadratic(n=3)
    out = newton(f=f, grad=grad, hess=hess, x0=np.zeros(3), tol=1e-12)
    check_result(out)
    assert out["status"] == "converged" and out["counts"]["nit"] == 1
    assert np.allclose(out["x"], x_star)


def test_newton_converges_quadratically_on_rosenbrock():
    prob = Rosenbrock(n=2)
    out = newton(f=prob.f, grad=prob.grad, hess=prob.hess, x0=np.array([1.2, 1.2]), tol=1e-12)
    assert out["status"] == "converged" and np.allclose(out["x"], [1.0, 1.0])
    # quadratic convergence: the error roughly squares in each of the last steps
    errs = [np.linalg.norm(x - prob.x_star) for x in out["history"]["x"]]
    errs = [e for e in errs if e > 1e-12]
    assert errs[-1] <= 10 * errs[-2] ** 2


def test_plain_newton_can_converge_to_a_maximum():
    # Newton looks for grad f = 0, not for a minimum: from (0, 0) on Himmelblau it finds the local maximum.
    prob = Himmelblau()
    out = newton(f=prob.f, grad=prob.grad, hess=prob.hess, x0=np.zeros(2))
    assert out["status"] == "converged"
    assert np.all(np.linalg.eigvalsh(prob.hess(out["x"])) < 0)         # negative definite: a maximum


def test_modified_damped_newton_finds_a_minimum_from_the_same_start():
    prob = Himmelblau()
    out = newton(f=prob.f, grad=prob.grad, hess=prob.hess, x0=np.zeros(2), safeguard=True, damped=True)
    assert out["status"] == "converged" and out["f"] < 1e-12
    vals = np.asarray(out["history"]["f"])
    assert np.all(vals[1:] <= vals[:-1] + 1e-12)                       # damping makes it a descent method


def test_newton_singular_hessian_falls_back_to_steepest_descent():
    # f(x) = x1^4 + x2^2 has a singular Hessian whenever x1 = 0
    f = lambda x: x[0] ** 4 + x[1] ** 2
    grad = lambda x: np.array([4 * x[0] ** 3, 2 * x[1]])
    hess = lambda x: np.array([[12 * x[0] ** 2, 0.0], [0.0, 2.0]])
    out = newton(f=f, grad=grad, hess=hess, x0=np.array([0.0, 1.0]), maxit=3)
    assert out["history"]["meta"]["steepest_descent_fallback"] == [0, 1, 2]


@pytest.mark.parametrize("run", [
    lambda f, g, h, x0: newton(f=f, grad=g, hess=h, x0=x0),
    lambda f, g, h, x0: bfgs(f=f, grad=g, x0=x0),
])
def test_starting_at_minimizer_stops_immediately(run):
    f, grad, hess, x_star = quadratic(n=3)
    out = run(f, grad, hess, x_star)
    assert out["status"] == "converged" and out["counts"]["nit"] == 0


@pytest.mark.parametrize("method", [gradient_descent, bfgs])
def test_keeps_running_at_round_off_level(method):
    # tol = 1e-12 cannot be reached (round-off in f). The exact line search then finds no decrease
    # and returns t = 0, the method falls back to Armijo, and the run ends with "maxit" at the minimum.
    f, grad, _, x_star = quadratic(n=2)
    out = method(f=f, grad=grad, x0=np.array([3.0, 3.0]), tol=1e-12, maxit=100)
    assert out["status"] == "maxit"
    assert out["f"] == pytest.approx(f(x_star), abs=1e-12)


def test_newton_method_alias():
    assert newton_method is newton


# ---------------- BFGS ----------------

@pytest.mark.parametrize("step", ["exact", "armijo"])
def test_bfgs_on_rosenbrock(step):
    prob = Rosenbrock(n=2)
    out = bfgs(f=prob.f, grad=prob.grad, x0=np.array([-1.2, 1.0]), tol=1e-6, step=step)
    check_result(out)
    assert out["status"] == "converged"
    assert np.allclose(out["x"], [1.0, 1.0], atol=1e-4)


def test_bfgs_finds_quadratic_minimizer_in_at_most_n_exact_steps():
    # With exact line search, BFGS on an n-dimensional quadratic terminates in at most n steps.
    f, grad, _, x_star = quadratic(n=4)
    out = bfgs(f=f, grad=grad, x0=np.ones(4) * 3.0, tol=1e-6, step="exact")
    assert out["counts"]["nit"] <= 4 + 1          # +1: the numeric line search is not perfectly exact
    assert np.allclose(out["x"], x_star, atol=1e-5)


def test_bfgs_with_safeguard():
    prob = Rosenbrock(n=2)
    out = bfgs(f=prob.f, grad=prob.grad, x0=np.array([-1.2, 1.0]), tol=1e-6, safeguard=True)
    assert out["status"] == "converged"


def test_bfgs_rejects_unknown_step():
    f, grad, _, _ = quadratic(n=2)
    with pytest.raises(ValueError):
        bfgs(f=f, grad=grad, x0=np.ones(2), step="wolfe")


# ---------------- evaluation counts ----------------

@pytest.mark.parametrize("name, run", [
    ("gd_exact",      lambda f, g, h, x0: gradient_descent(f=f, grad=g, x0=x0, maxit=30)),
    ("gd_armijo",     lambda f, g, h, x0: gradient_descent(f=f, grad=g, x0=x0, maxit=30, step="armijo")),
    ("newton",        lambda f, g, h, x0: newton(f=f, grad=g, hess=h, x0=x0)),
    ("newton_damped", lambda f, g, h, x0: newton(f=f, grad=g, hess=h, x0=x0, damped=True)),
    ("bfgs_exact",    lambda f, g, h, x0: bfgs(f=f, grad=g, x0=x0, maxit=30)),
    ("bfgs_armijo",   lambda f, g, h, x0: bfgs(f=f, grad=g, x0=x0, maxit=30, step="armijo")),
])
def test_reported_counts_match_actual_calls(name, run):
    prob = Rosenbrock(n=2)
    f, g, h = Counter(prob.f), Counter(prob.grad), Counter(prob.hess)
    counts = run(f, g, h, np.array([-1.2, 1.0]))["counts"]
    assert counts["nfev"] == f.calls
    assert counts["njev"] == g.calls
    assert counts.get("nhev", 0) == h.calls
