# tests/test_dfo.py
import numpy as np
import pytest

# Expect Nelder–Mead under optedu.algorithms.nelder_mead:nelder_mead
from optedu.algorithms.nelder_mead import nelder_mead  # type: ignore

def test_nelder_mead_on_convex_quadratic():
    # f(x) = ||x - a||^2
    a = np.array([1.0, -2.0])
    def f(x):
        x = np.asarray(x, float)
        return float(np.sum((x - a)**2))
    x0 = np.array([5.0, 5.0])
    out = nelder_mead(f=f, x0=x0, maxit=5000, tol=1e-8)
    assert out["status"] in {"converged", "maxit"}
    assert np.linalg.norm(out["x"] - a) < 1e-3
