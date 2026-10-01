# tests/test_lp_core.py
# Primal simplex (page 43) and the two-phase method (§3.3.2) for standard form
#     min c^T x  s.t.  A x = b,  x >= 0.
import itertools

import numpy as np
import pytest

from helpers import check_result
from optedu.algorithms.lp_simplex import _find_identity_basis, simplex_standard
from optedu.algorithms.lp_two_phase import solve_two_phase


def brute_force_lp(A, b, c):
    """
    Independent check: an optimum of a bounded, feasible LP is attained at a basic feasible solution.
    Try every choice of m columns, keep the feasible ones, return the best objective value (None if infeasible).
    """
    A, b, c = (np.asarray(v, float) for v in (A, b, c))
    m, n = A.shape
    best = None
    for B in itertools.combinations(range(n), m):
        A_B = A[:, B]
        if abs(np.linalg.det(A_B)) < 1e-10:
            continue
        x_B = np.linalg.solve(A_B, b)
        if np.all(x_B >= -1e-9):
            value = float(c[list(B)] @ x_B)
            best = value if best is None else min(best, value)
    return best


def random_bounded_lp(rng, m, n):
    """Random feasible LP; the last row sum(x) + s = 10 keeps the feasible set bounded."""
    A = rng.normal(size=(m, n))
    x_feasible = rng.uniform(0.0, 1.0, size=n)
    b = A @ x_feasible
    A = np.vstack([np.hstack([A, np.zeros((m, 1))]), np.ones((1, n + 1))])   # extra column = slack s
    b = np.append(b, 10.0)
    c = rng.normal(size=n + 1)
    return A, b, c


def check_optimality(A, b, c, res, tol=1e-8):
    """KKT conditions at the returned basis: feasibility, dual feasibility (r_N >= 0), complementarity."""
    A, b, c = (np.asarray(v, float) for v in (A, b, c))
    x, B = res["x"], np.asarray(res["lp"]["basis"])
    assert np.allclose(A @ x, b) and np.all(x >= -tol)
    y = np.linalg.solve(A[:, B].T, c[B])          # duals: A_B^T y = c_B
    r = c - A.T @ y                               # reduced costs (zero on the basis)
    assert np.all(r >= -tol)
    assert abs(r @ x) < 1e-7                      # complementary slackness: x_j r_j = 0


# ---------------- primal simplex ----------------

A_SLACK = [[1, 1, 1, 0],          # x1 +  x2 + s1      = 4
           [1, 3, 0, 1]]          # x1 + 3x2      + s2 = 6
B_SLACK = [4, 6]
C_SLACK = [-1, -2, 0, 0]          # min -x1 - 2x2  ->  x = (3, 1), f = -5


def test_find_identity_basis():
    assert _find_identity_basis(np.array(A_SLACK, float), np.array(B_SLACK, float)) == [2, 3]
    assert _find_identity_basis(np.array(A_SLACK, float), np.array([4.0, -6.0])) is None   # b < 0
    assert _find_identity_basis(np.array([[1.0, 2.0]]), np.array([1.0])) == [0]
    assert _find_identity_basis(np.array([[2.0, 2.0]]), np.array([1.0])) is None


def test_simplex_from_slack_basis():
    res = simplex_standard(A_SLACK, B_SLACK, C_SLACK)
    check_result(res)
    assert res["status"] == "converged"
    assert np.allclose(res["x"], [3, 1, 0, 0]) and res["f"] == pytest.approx(-5.0)
    check_optimality(A_SLACK, B_SLACK, C_SLACK, res)


def test_simplex_history():
    res = simplex_standard(A_SLACK, B_SLACK, C_SLACK)
    hist = res["history"]
    assert np.all(np.diff(hist["f"]) <= 1e-12)                       # objective never increases (min)
    assert len(hist["meta"]["basis"]) == len(hist["f"])               # one basis per vertex visited
    assert len(hist["meta"]["enter_leave"]) == len(hist["f"]) - 1     # one pivot between vertices
    assert res["counts"]["nit"] == len(hist["meta"]["enter_leave"])


def test_simplex_with_given_basis():
    res = simplex_standard(A_SLACK, B_SLACK, C_SLACK, basis=[0, 3])   # x1 = 4, s2 = 2: feasible
    assert res["status"] == "converged" and res["f"] == pytest.approx(-5.0)


def test_simplex_without_identity_basis_fails_cleanly():
    res = simplex_standard([[2.0, 2.0]], [1.0], [1.0, 1.0])
    assert res["status"] == "failed" and "two-phase" in res["message"]


@pytest.mark.parametrize("basis, message", [([0, 1], "infeasible"),   # x_B = (-1, ...) : not a feasible basis
                                            ([0, 0], "Singular")])
def test_simplex_rejects_bad_basis(basis, message):
    A = [[1, 1, 1, 0], [1, -1, 0, 1]]
    with pytest.raises(RuntimeError, match=message):
        simplex_standard(A, [1.0, 3.0], [1, 1, 0, 0], basis=basis)


def test_simplex_unbounded_returns_vertex_and_ray():
    # min -x1  s.t.  -x1 + x2 + s = 0 : x1 = x2 can grow forever
    A, b, c = [[-1, 1, 1]], [0], [-1, 0, 0]
    res = simplex_standard(A, b, c)
    assert res["status"] == "unbounded"
    x, d = res["x"], res["lp"]["direction"]
    A = np.asarray(A, float)
    assert np.allclose(A @ d, 0) and np.all(d >= 0) and np.dot(c, d) < 0
    for t in [1.0, 10.0, 100.0]:                  # x + t d stays feasible, objective decreases without bound
        assert np.allclose(A @ (x + t * d), b) and np.all(x + t * d >= 0)


def test_simplex_maxit():
    res = simplex_standard(A_SLACK, B_SLACK, C_SLACK, maxit=0)
    assert res["status"] == "maxit"


def test_bland_rule_prevents_cycling_on_beales_example():
    # Beale's classic degenerate LP cycles with the "most negative reduced cost" rule; Bland's rule terminates.
    A = [[0.25, -8, -1, 9, 1, 0, 0],
         [0.5, -12, -0.5, 3, 0, 1, 0],
         [0, 0, 1, 0, 0, 0, 1]]
    b = [0, 0, 1]
    c = [-0.75, 20, -0.5, 6, 0, 0, 0]
    res = simplex_standard(A, b, c, maxit=100)
    assert res["status"] == "converged"
    assert res["f"] == pytest.approx(-1.25) == pytest.approx(brute_force_lp(A, b, c))


@pytest.mark.parametrize("seed", range(10))
def test_simplex_matches_brute_force_on_random_lps(seed):
    # [A | I] x = b with b >= 0 has the slack identity basis, so the plain simplex can start
    rng = np.random.default_rng(seed)
    m, n = 3, 4
    A = np.hstack([rng.uniform(0.1, 1.0, size=(m, n)), np.eye(m)])
    b = rng.uniform(1.0, 2.0, size=m)
    c = np.append(rng.normal(size=n), np.zeros(m))
    res = simplex_standard(A, b, c)
    assert res["status"] == "converged"
    assert res["f"] == pytest.approx(brute_force_lp(A, b, c), abs=1e-9)
    check_optimality(A, b, c, res)


# ---------------- two-phase ----------------

def test_two_phase_optimal():
    # min -x1 - 2x2  s.t.  x1 + x2 = 4,  x1 + 3x2 = 6,  x >= 0  ->  x = (3, 1), f = -5
    res = solve_two_phase(A=[[1, 1], [1, 3]], b=[4, 6], c=[-1, -2])
    check_result(res)
    assert res["status"] == "converged"
    assert np.allclose(res["x"], [3.0, 1.0], atol=1e-10) and np.isclose(res["f"], -5.0)


def test_two_phase_infeasible():
    res = solve_two_phase(A=[[1], [-1]], b=[1, -2], c=[0.0])      # x1 = 1 and x1 = 2
    assert res["status"] == "infeasible"
    assert "Phase I" in res["message"]


def test_two_phase_unbounded_with_direction():
    # min -x1  s.t.  -x1 + x2 = 0,  x >= 0
    A, c = [[-1, 1]], [-1, 0]
    res = solve_two_phase(A=A, b=[0], c=c)
    assert res["status"] == "unbounded"
    d = np.asarray(res["lp"]["direction"], float)
    assert np.allclose(np.asarray(A, float) @ d, [0.0]) and float(np.dot(c, d)) < 0.0


def test_two_phase_negative_right_hand_side():
    # Row signs are normalized so that the artificial basis a = b is feasible: -x1 - x2 = -2
    res = solve_two_phase(A=[[-1, -1]], b=[-2], c=[1, 2])
    assert np.allclose(res["x"], [2.0, 0.0])


def test_two_phase_redundant_constraint():
    # The second row is twice the first: an artificial stays basic after Phase I and the row is dropped.
    res = solve_two_phase(A=[[1, 1], [2, 2]], b=[2, 4], c=[1, 2])
    assert res["status"] == "converged"
    assert np.allclose(res["x"], [2.0, 0.0]) and np.isclose(res["f"], 2.0)


@pytest.mark.parametrize("seed", range(10))
def test_two_phase_matches_brute_force_on_random_lps(seed):
    A, b, c = random_bounded_lp(np.random.default_rng(seed), m=2, n=4)
    res = solve_two_phase(A, b, c)
    assert res["status"] == "converged"
    assert res["f"] == pytest.approx(brute_force_lp(A, b, c), abs=1e-8)
    check_optimality(A, b, c, res)


def test_drive_out_artificials_degenerate_pivot():
    # Phase I ended with the artificial of row 1 (column 3) basic at value 0. Row 1 of the tableau has
    # a nonzero entry for x2, so x2 replaces the artificial (a degenerate pivot) and no row is dropped.
    from optedu.algorithms.lp_two_phase import _drive_out_artificials
    A = np.array([[1.0, 1.0], [1.0, -1.0]])
    A1 = np.hstack([A, np.eye(2)])
    basis, keep = _drive_out_artificials(A1, [0, 3], n=2, tol=1e-9)
    assert basis == [0, 1] and keep == [0, 1]


def test_two_phase_degenerate_zero_right_hand_side():
    # b = 0: the only feasible point is x = 0 and every basis is degenerate
    res = solve_two_phase(A=[[1, 1], [1, -1]], b=[0, 0], c=[1, 1])
    assert res["status"] == "converged" and np.allclose(res["x"], 0.0)


def test_simplex_dimension_check():
    with pytest.raises(ValueError):
        simplex_standard(A_SLACK, [4, 6, 8], C_SLACK)
