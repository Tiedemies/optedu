# tests/test_lp_core.py
import numpy as np
import pytest

# Assumptions from README:
# - LP container at optedu.problems.lp:LP
# - Two-phase and simplex available under optedu.algorithms.lp_two_phase / lp_simplex
# - Unbounded reporting via return dict: {"status": "...", "direction": d} when unbounded

from optedu.problems.lp import LP  # type: ignore
from optedu.algorithms.lp_two_phase import two_phase_simplex  # type: ignore
from optedu.algorithms.lp_simplex import simplex  # type: ignore

def _solve_two_phase(A, b, c):
    prob = LP(A=np.asarray(A, float), b=np.asarray(b, float), c=np.asarray(c, float))
    # two_phase_simplex expected to return dict with keys: status, x, obj (and maybe basis)
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
    # Known optimum at x = (3, 1) with obj = -5
    assert np.allclose(x, [3.0, 1.0], atol=1e-8)
    assert abs(res["obj"] - (-5.0)) < 1e-8

def test_lp_infeasible():
    # Infeasible: x1 + x2 <= 1 and x1 + x2 >= 3 (encoded as -x1 - x2 <= -3)
    A = [[ 1,  1],
         [-1, -1]]
    b = [1, -3]
    c = [1, 1]
    res = _solve_two_phase(A, b, c)
    assert res["status"] in {"infeasible", "infeasible_phase_I_failed"}

def test_lp_unbounded_with_direction():
    # min  -x1  s.t.  x1 - x2 >= 0  (i.e., -x1 + x2 <= 0),  x >= 0
    # This is unbounded to -inf by increasing x1, x2 = x1.
    A = [[-1, 1]]
    b = [0]
    c = [-1, 0]
    res = _solve_two_phase(A, b, c)
    assert res["status"] == "unbounded"
    d = res.get("direction", None)
    assert d is not None
    d = np.asarray(d, float)
    # Check recession properties Ad = 0 (<= 0 satisfied tightly) and c^T d < 0
    assert np.allclose(A @ d, [0.0], atol=1e-10)
    assert float(np.dot(c, d)) < 0.0
