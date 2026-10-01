# tests/helpers.py
# Small helpers shared by the tests (import with: from helpers import ...).
from typing import get_args

import numpy as np

from optedu.utils.types import Status

STATUSES = set(get_args(Status))     # {"converged", "maxit", "infeasible", "unbounded", "failed"}


class Counter:
    """Wrap a function and count how often it is called (to check the reported nfev / njev / nhev)."""
    def __init__(self, fn):
        self.fn = fn
        self.calls = 0

    def __call__(self, x):
        self.calls += 1
        return self.fn(x)


def quadratic(n=3):
    """
    Strongly convex quadratic f(x) = 0.5 x^T Q x - b^T x with Q = diag(1, ..., 5).
    Returns f, grad, hess and the minimizer x* = Q^{-1} b.
    """
    Q = np.diag(np.linspace(1.0, 5.0, n))
    b = np.arange(1, n + 1, dtype=float)

    def f(x):
        x = np.asarray(x, float)
        return 0.5 * float(x @ Q @ x) - float(b @ x)

    def grad(x):
        return Q @ np.asarray(x, float) - b

    def hess(x):
        return Q

    return f, grad, hess, np.linalg.solve(Q, b)


def sphere(x):
    """f(x) = ||x||^2, minimum 0 at x = 0."""
    x = np.asarray(x, float)
    return float(x @ x)


def numerical_gradient(f, x, h=1e-6):
    """Central differences: g_i ~ (f(x + h e_i) - f(x - h e_i)) / (2h)."""
    x = np.asarray(x, float)
    g = np.zeros_like(x)
    for i in range(x.size):
        e = np.zeros_like(x)
        e[i] = h
        g[i] = (f(x + e) - f(x - e)) / (2 * h)
    return g


def check_result(result):
    """The unified result contract (see optedu/utils/types.py) that every algorithm follows."""
    assert result["status"] in STATUSES
    history = result["history"]
    assert isinstance(history, dict) and "f" in history
    if "x" in history:
        assert len(history["x"]) == len(history["f"])     # one x per f, entry 0 = start
    assert isinstance(result["counts"]["nit"], int)
