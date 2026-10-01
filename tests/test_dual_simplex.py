# tests/test_dual_simplex.py
# Dual simplex (page 71) for standard form  min c^T x  s.t.  A x = b,  x >= 0.
import numpy as np
import pytest

from helpers import check_result
from optedu.algorithms.dual_simplex import _find_dual_feasible_basis, _is_dual_feasible, dual_simplex_standard
from optedu.algorithms.lp_two_phase import solve_two_phase

# 3x1 - 2x2 - s1 = 5,  x1 + 2x2 - s2 = 4,  min x1 + x2   ->  x = (2.25, 0.875), f = 3.125
A_DUAL = [[3, -2, -1, 0], [1, 2, 0, -1]]
B_DUAL = [5, 4]
C_DUAL = [1, 1, 0, 0]


def covering_lp(rng, m, n):
    """
    min c^T x  s.t.  G x >= h,  x >= 0  with c > 0, written as [G | -I] (x, s) = h.
    The surplus basis (the -I columns) is dual feasible (reduced costs = c > 0) but primal
    infeasible (s = -h < 0): exactly the situation the dual simplex is made for.
    """
    G = rng.uniform(0.1, 1.0, size=(m, n))
    h = rng.uniform(1.0, 2.0, size=m)
    c = np.append(rng.uniform(0.5, 2.0, size=n), np.zeros(m))
    return np.hstack([G, -np.eye(m)]), h, c, list(range(n, n + m))


@pytest.mark.parametrize("basis", [[2, 3], None])
def test_dual_simplex_optimal(basis):
    res = dual_simplex_standard(A_DUAL, B_DUAL, C_DUAL, basis=basis)
    check_result(res)
    assert res["status"] == "converged"
    assert np.allclose(res["x"], [2.25, 0.875, 0.0, 0.0])
    assert np.isclose(res["f"], 3.125)


def test_dual_simplex_strong_duality():
    # At the optimum the dual solution pi gives the same objective: b^T pi = c^T x
    res = dual_simplex_standard(A_DUAL, B_DUAL, C_DUAL, basis=[2, 3])
    assert np.dot(B_DUAL, res["lp"]["dual"]) == pytest.approx(res["f"])


def test_dual_objective_never_decreases():
    # The dual simplex keeps dual feasibility and moves to better DUAL solutions: c_B^T x_B goes up.
    rng = np.random.default_rng(0)
    A, b, c, basis = covering_lp(rng, m=4, n=5)
    res = dual_simplex_standard(A, b, c, basis=basis)
    assert np.all(np.diff(res["history"]["f"]) >= -1e-12)
    assert res["counts"]["nit"] == len(res["history"]["meta"]["enter_leave"])


@pytest.mark.parametrize("seed", range(10))
def test_dual_and_primal_simplex_agree_on_random_lps(seed):
    A, b, c, basis = covering_lp(np.random.default_rng(seed), m=3, n=4)
    res_dual = dual_simplex_standard(A, b, c, basis=basis)
    res_primal = solve_two_phase(A, b, c)
    assert res_dual["status"] == res_primal["status"] == "converged"
    assert res_dual["f"] == pytest.approx(res_primal["f"], abs=1e-9)
    assert np.allclose(A @ res_dual["x"], b) and np.all(res_dual["x"] >= -1e-9)


def test_dual_simplex_infeasible():
    # -x1 - x2 - x3 = 1 has no solution with x >= 0; basis [2] is dual-feasible.
    res = dual_simplex_standard([[-1, -1, -1]], [1], [1, 1, 0], basis=[2])
    assert res["status"] == "infeasible"


def test_dual_simplex_rejects_non_dual_feasible_basis():
    res = dual_simplex_standard(A_DUAL, B_DUAL, [1, 1, -5, -5], basis=[0, 1])
    assert res["status"] == "failed"
    assert "dual-feasible" in res["message"]


def test_dual_simplex_maxit():
    A, b, c, basis = covering_lp(np.random.default_rng(1), m=4, n=5)
    res = dual_simplex_standard(A, b, c, basis=basis, maxit=0)
    assert res["status"] == "maxit"


def test_is_dual_feasible():
    A = np.array(A_DUAL, float); c = np.array(C_DUAL, float)
    assert _is_dual_feasible(A, c, np.array([2, 3]), tol=1e-9)           # reduced costs = c_N >= 0
    assert not _is_dual_feasible(A, -c, np.array([2, 3]), tol=1e-9)


def test_find_dual_feasible_basis():
    A = np.array(A_DUAL, float); c = np.array(C_DUAL, float)
    B = _find_dual_feasible_basis(A, c)
    assert B is not None and _is_dual_feasible(A, c, B, tol=1e-9)
    # min -x1 - x2 over x1 + x2 + s = 1: the slack basis [2] is NOT dual feasible (reduced costs -1, -1),
    # so the search has to swap; B = [0] gives reduced costs (0, 1) for (x2, s), which is dual feasible.
    A = np.array([[1.0, 1.0, 1.0]]); c = np.array([-1.0, -1.0, 0.0])
    assert not _is_dual_feasible(A, c, np.array([2]), tol=1e-9)
    B = _find_dual_feasible_basis(A, c)
    assert B is not None and _is_dual_feasible(A, c, B, tol=1e-9)


def test_find_dual_feasible_basis_needs_swaps():
    # The first independent column (cost 0) gives reduced costs -1, -1: the search must swap it out.
    A = np.array([[1.0, 1.0, 1.0]]); c = np.array([0.0, -1.0, -1.0])
    assert not _is_dual_feasible(A, c, np.array([0]), tol=1e-9)
    B = _find_dual_feasible_basis(A, c)
    assert B is not None and _is_dual_feasible(A, c, B, tol=1e-9)


def test_no_dual_feasible_basis():
    # min -x1 - x2 over x1 - x2 = 0 is unbounded, so the dual is infeasible: no dual-feasible basis exists.
    A = np.array([[1.0, -1.0]]); c = np.array([-1.0, -1.0])
    assert _find_dual_feasible_basis(A, c) is None
    res = dual_simplex_standard(A, [0.0], c)
    assert res["status"] == "failed" and "No dual-feasible basis" in res["message"]


def test_rank_deficient_matrix_has_no_basis():
    assert _find_dual_feasible_basis(np.array([[1.0, 1.0], [2.0, 2.0]]), np.array([1.0, 1.0])) is None


def test_is_dual_feasible_edge_cases():
    assert not _is_dual_feasible(np.array([[1.0, 1.0], [1.0, 1.0]]), np.ones(2), np.array([0, 1]), tol=1e-9)  # singular
    assert _is_dual_feasible(np.eye(2), np.ones(2), np.array([0, 1]), tol=1e-9)    # no nonbasic columns


def test_dual_simplex_dimension_check():
    with pytest.raises(ValueError):
        dual_simplex_standard(A_DUAL, [5, 4, 3], C_DUAL)
