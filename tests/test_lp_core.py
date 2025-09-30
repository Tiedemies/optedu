# tests/test_lp_core.py
import numpy as np
import pytest

# Unified return type; LP extras under result["lp"]["..."]

from optedu.problems.lp import LP
from optedu.algorithms.lp_two_phase import two_phase_simplex
from optedu.algorithms.lp_simplex import simplex

def _solve_two_phase(A, b, c):
    prob = LP(A=np.asarray(A, float), b=np.asarray(b, float), c=np.asarray(c, float))
    # unified: status, x, f, lp.basis, lp.direction (if unbounded)
    return two_phase_simplex(A=prob.A, b=prob.b, c=prob.c)

def test_lp_optimal_basic():
    # A small feasible LP with unique optimum
    # min  -x1 - 2x2  s.t.  x1 + x2 <= 4,  x1 + 3x2 <= 6,  x >= 0
    A = [[1, 1],
         [1, 3]]
    b = [4, 6]
    c = [-1, -2]   # minimize c^T x
    res = _solve_two_phase(A, b, c)
    assert res["status"] == "optimal"
    x = res["x"]
    # Known optimum at x = (3, 1) with f = -5
    assert np.allclose(x, [3.0, 1.0], atol=1e-10)
    assert np.isclose(res["f"], -5.0, atol=1e-10)

def test_lp_infeasible_small():
    # Infeasible: x1 >= 2 and x1 <= 1 simultaneously (after standardization) yields no feasible x
    A = [[1], [-1]]
    b = [1, -2]   # Encodes x1 <= 1 and x1 >= 2 contradiction
    c = [0.0]
    res = _solve_two_phase(A, b, c)
    assert res["status"] == "infeasible"

def test_lp_unbounded_with_direction():
    # min  -x1  s.t.  x1 - x2 >= 0  (i.e., -x1 + x2 <= 0),  x >= 0
    # This is unbounded to -inf by increasing x1, x2 = x1.
    A = [[-1, 1]]
    b = [0]
    c = [-1, 0]
    res = _solve_two_phase(A, b, c)
    assert res["status"] == "unbounded"
    # unified LP witness lives under res["lp"]["direction"]
    d = res.get("lp", {}).get("direction", None)
    assert d is not None
    d = np.asarray(d, float)
    # Check recession properties Ad = 0 and c^T d < 0
    A_arr = np.asarray(A, float)
    assert np.allclose(A_arr @ d, [0.0], atol=1e-10)
    assert float(np.dot(c, d)) < 0.0
