# tests/test_lp_standardize.py
# Converting a general LP (<=, =, >= rows; lower/upper bounds; free variables; min or max)
# to standard form, and solving general LPs with solve_two_phase_generic.
import itertools

import numpy as np
import pytest

from optedu.algorithms.lp_two_phase import solve_two_phase_generic
from optedu.problems.lp_standardize import to_standard_form

INF = np.inf


# ---------------- the conversion itself ----------------

def test_slacks_for_le_and_ge_rows():
    A = [[1, 2], [3, 4], [5, 6]]
    b = [7, 8, 9]
    A_std, b_std, c_std, info = to_standard_form([1, 1], A, b, ["le", "ge", "eq"], lb=[0, 0])
    #   x1 + 2x2 + s1      = 7      (le: add a slack)
    # -3x1 - 4x2      + s2 = -8     (ge: multiply by -1, then add a slack)
    #  5x1 + 6x2           = 9      (eq: unchanged)
    assert np.allclose(A_std, [[1, 2, 1, 0], [-3, -4, 0, 1], [5, 6, 0, 0]])
    assert np.allclose(b_std, [7, -8, 9])
    assert np.allclose(c_std, [1, 1, 0, 0])
    assert info["objective_offset"] == 0.0


def test_max_becomes_min_of_negated_objective():
    _, _, c_std, _ = to_standard_form([1, -2], [[1, 1]], [1], ["eq"], objective="max", lb=[0, 0])
    assert np.allclose(c_std, [-1, 2])


def test_free_variable_is_split():
    # x = x+ - x-, both >= 0
    A_std, _, c_std, info = to_standard_form([3], [[2]], [1], ["eq"])     # lb, ub default to -inf, +inf
    assert np.allclose(A_std, [[2, -2]]) and np.allclose(c_std, [3, -3])
    assert info["reconstruct"](np.array([5.0, 2.0]))[0] == pytest.approx(3.0)


def test_lower_bound_is_shifted():
    # x = y + 2, y >= 0:  3x = 9  ->  3y = 3;  objective 4x = 4y + 8
    A_std, b_std, c_std, info = to_standard_form([4], [[3]], [9], ["eq"], lb=[2])
    assert np.allclose(A_std, [[3]]) and np.allclose(b_std, [3])
    assert info["objective_offset"] == pytest.approx(8.0)
    assert info["reconstruct"](np.array([1.0]))[0] == pytest.approx(3.0)


def test_upper_bound_only_is_flipped():
    # x <= 5 with no lower bound:  x = 5 - y, y >= 0
    A_std, b_std, c_std, info = to_standard_form([1], [[1]], [2], ["eq"], lb=[-INF], ub=[5])
    assert np.allclose(A_std, [[-1]]) and np.allclose(b_std, [-3]) and np.allclose(c_std, [-1])
    assert info["objective_offset"] == pytest.approx(5.0)
    assert info["reconstruct"](np.array([3.0]))[0] == pytest.approx(2.0)


def test_both_bounds_add_a_row():
    # 1 <= x <= 4:  x = y + 1  and an extra row  y + s = 3
    A_std, b_std, _, _ = to_standard_form([1], [[1]], [2], ["eq"], lb=[1], ub=[4])
    assert np.allclose(A_std, [[1, 0], [1, 1]]) and np.allclose(b_std, [1, 3])


def test_bad_senses_raise():
    with pytest.raises(ValueError):
        to_standard_form([1], [[1]], [1], ["le", "ge"])
    with pytest.raises(ValueError):
        to_standard_form([1], [[1]], [1], ["<="])


def standard_point(x, A_std, b_std, info):
    """Map an original point x to the standard-form vector z (variables, then slacks)."""
    k = len(info["col_meta"])
    z = np.zeros(A_std.shape[1])
    for col, meta in enumerate(info["col_meta"]):
        j = meta["orig"]
        if meta["type"] == "shifted":
            z[col] = x[j] - meta["L"]
        elif meta["type"] == "flipped":
            z[col] = meta["U"] - x[j]
        elif meta["type"] == "free_plus":
            z[col] = max(x[j], 0.0)
        elif meta["type"] == "free_minus":
            z[col] = max(-x[j], 0.0)
    residual = b_std - A_std[:, :k] @ z[:k]
    for col in range(k, A_std.shape[1]):                 # each slack column has a single 1 in its row
        row = int(np.argmax(A_std[:, col]))
        z[col] = residual[row]
    return z


@pytest.mark.parametrize("seed", range(10))
def test_round_trip_feasible_point(seed):
    # A point x that is feasible for the original LP maps to a feasible z with the same objective value.
    rng = np.random.default_rng(seed)
    m, n = 3, 4
    A = rng.normal(size=(m, n)); c = rng.normal(size=n)
    x = rng.normal(size=n)
    senses = list(rng.choice(["le", "ge", "eq"], size=m))
    gap = rng.uniform(0.0, 1.0, size=m)
    b = A @ x + np.where(np.array(senses) == "le", gap, np.where(np.array(senses) == "ge", -gap, 0.0))
    lb = np.where(rng.random(n) < 0.5, x - rng.uniform(0, 1, n), -INF)
    ub = np.where(rng.random(n) < 0.5, x + rng.uniform(0, 1, n), INF)
    objective = rng.choice(["min", "max"])

    A_std, b_std, c_std, info = to_standard_form(c, A, b, senses, objective=objective, lb=lb, ub=ub)
    z = standard_point(x, A_std, b_std, info)
    assert np.allclose(A_std @ z, b_std) and np.all(z >= -1e-12)
    assert np.allclose(info["reconstruct"](z), x)
    sign = -1.0 if objective == "max" else 1.0
    assert sign * (c_std @ z + info["objective_offset"]) == pytest.approx(c @ x)


# ---------------- solving general LPs ----------------

def is_feasible(x, A, b, senses, lb, ub, tol=1e-9):
    """Does x satisfy every row (le / ge / eq) and the bounds lb <= x <= ub?"""
    for a, bi, s in zip(A, b, senses):
        v = np.dot(a, x)
        if (s == "le" and v > bi + tol) or (s == "ge" and v < bi - tol) or (s == "eq" and abs(v - bi) > tol):
            return False
    return bool(np.all(x >= np.asarray(lb) - tol) and np.all(x <= np.asarray(ub) + tol))


def brute_force_2d(A, b, senses, c, lb, ub, objective):
    """
    Independent check for 2-variable LPs: the optimum is at a vertex, i.e. where two of the
    constraint / bound lines meet. Try all pairs and keep the best feasible intersection.
    """
    lines = [(np.asarray(a, float), bi) for a, bi in zip(A, b)]
    for j in range(2):
        e = np.eye(2)[j]
        if np.isfinite(lb[j]): lines.append((e, lb[j]))
        if np.isfinite(ub[j]): lines.append((e, ub[j]))

    sign = -1.0 if objective == "max" else 1.0
    best = None
    for (a1, b1), (a2, b2) in itertools.combinations(lines, 2):
        M = np.vstack([a1, a2])
        if abs(np.linalg.det(M)) < 1e-10:
            continue
        x = np.linalg.solve(M, [b1, b2])
        if is_feasible(x, A, b, senses, lb, ub) and (best is None or sign * (c @ x) < sign * best):
            best = float(c @ x)
    return best


@pytest.mark.parametrize("seed", range(15))
def test_generic_matches_brute_force_on_random_2d_lps(seed):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(3, 2)); c = rng.normal(size=2)
    x_feasible = rng.uniform(-2, 2, size=2)
    senses = list(rng.choice(["le", "ge"], size=3))
    gap = rng.uniform(0.1, 1.0, size=3)
    b = A @ x_feasible + np.where(np.array(senses) == "le", gap, -gap)
    lb = np.where(rng.random(2) < 0.7, x_feasible - rng.uniform(0.5, 2, 2), -INF)
    ub = np.where(rng.random(2) < 0.7, x_feasible + rng.uniform(0.5, 2, 2), INF)
    objective = str(rng.choice(["min", "max"]))

    expected = brute_force_2d(A, b, senses, c, lb, ub, objective)
    res = solve_two_phase_generic(A, b, c, senses=senses, objective=objective, lb=lb, ub=ub)
    if res["status"] == "unbounded":
        # no vertex is optimal then: x + t d must stay feasible for large t and improve the objective
        x, d = res["x"], res["lp"]["direction"]
        sign = -1.0 if objective == "max" else 1.0
        assert sign * (c @ d) < 0
        assert is_feasible(x + 1e3 * d, A, b, senses, lb, ub, tol=1e-6)
        return
    assert res["status"] == "converged"
    assert res["f"] == pytest.approx(expected, abs=1e-8)
    x = res["x"]
    assert is_feasible(x, A, b, senses, lb, ub)
    assert res["f"] == pytest.approx(c @ x)


def test_generic_max_and_min_agree():
    # max x1 + 2 x2  s.t.  x1 + x2 <= 4,  x1 + 3 x2 <= 6,  x >= 0   ->  x = (3, 1), f = 5
    A, b, senses = [[1, 1], [1, 3]], [4, 6], ["le", "le"]
    res_max = solve_two_phase_generic(A, b, [1, 2], senses=senses, objective="max")
    res_min = solve_two_phase_generic(A, b, [-1, -2], senses=senses, objective="min")
    assert np.allclose(res_max["x"], [3.0, 1.0]) and np.isclose(res_max["f"], 5.0)
    assert np.allclose(res_min["x"], [3.0, 1.0]) and np.isclose(res_min["f"], -5.0)
    assert len(res_max["extra"]["standard_x"]) == 4              # x1, x2 and two slacks


def test_generic_default_is_nonnegative_variables():
    # min x1 + x2  s.t.  x1 + x2 >= -5 : with x >= 0 the optimum is 0 (not unbounded)
    res = solve_two_phase_generic([[1, 1]], [-5], [1, 1], senses=["ge"])
    assert res["status"] == "converged" and res["f"] == pytest.approx(0.0)


def test_generic_unbounded_ray_in_original_variables():
    # min -x1  s.t.  x1 - x2 <= 1,  x >= 0   ->  unbounded along d = (1, 1)
    res = solve_two_phase_generic([[1, -1]], [1], [-1, 0], senses=["le"])
    assert res["status"] == "unbounded"
    d = np.asarray(res["lp"]["direction"])
    assert d.shape == (2,)                                       # original variables, no slacks
    assert np.all(d >= 0) and d @ [1, -1] <= 1e-12 and d @ [-1, 0] < 0


def test_generic_infeasible():
    res = solve_two_phase_generic([[1, 1], [1, 1]], [1, 3], [1, 1], senses=["le", "ge"])
    assert res["status"] == "infeasible"
