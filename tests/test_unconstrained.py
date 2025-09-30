# tests/test_unconstrained.py
import numpy as np

from optedu.algorithms.gradient_descent import gradient_descent  # type: ignore
from optedu.algorithms.newton import newton_method  # type: ignore

def quad_data(n=3):
    # Strongly convex quadratic: f(x) = 0.5 x^T Q x - b^T x
    # with Q SPD, minimizer x* = Q^{-1} b
    U = np.eye(n)
    vals = np.linspace(1.0, 5.0, n)
    Q = U @ np.diag(vals) @ U.T
    b = np.arange(1, n+1, dtype=float)
    x_star = np.linalg.solve(Q, b)
    def f(x):
        x = np.asarray(x, float)
        return 0.5 * x @ (Q @ x) - b @ x
    def grad(x):
        x = np.asarray(x, float)
        return Q @ x - b
    def hess(x):
        return Q
    return f, grad, hess, x_star

def test_gradient_descent_linear_rate_on_spd_quadratic():
    f, grad, _, x_star = quad_data(n=4)
    x0 = np.ones(4) * 3.0
    # Use "exact" step (1/L) or internal line search, as implemented
    out = gradient_descent(f=f, grad=grad, x0=x0, maxit=500, tol=1e-10)
    assert out["status"] == "converged"
    assert np.linalg.norm(out["x"] - x_star) < 1e-6
    # Monotone decrease in f:
    vals = np.asarray(out.get("f_hist", []), float)
    if len(vals) > 1:
        assert np.all(vals[1:] <= vals[:-1] + 1e-12)

def test_newton_quadratic_converges_quickly():
    f, grad, hess, x_star = quad_data(n=3)
    x0 = np.zeros(3)
    out = newton_method(f=f, grad=grad, hess=hess, x0=x0, maxit=20, tol=1e-12)
    assert out["status"] == "converged"
    assert np.linalg.norm(out["x"] - x_star) < 1e-10
