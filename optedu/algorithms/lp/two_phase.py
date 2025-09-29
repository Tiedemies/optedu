# optedu/algorithms/lp/two_phase.py
# -------------------------------------------------------------------
# Two-phase orchestration that comports with §3.3.2 in the material:
#   Phase I: build auxiliary problem min 1^T a  s.t. A x + I a = b (after row sign normalization),
#            warm-start with artificial basis, and run the *same* page-43 simplex.
#   Phase II: restore original c and run the same simplex again from the feasible basis.
#
# Pedagogical notes:
#   • We *only* use the page-43 simplex as a black box.
#   • If Phase II is unbounded, we catch UnboundedSimplex and return a witness ray d with A d = 0,
#     d feasible (nonnegativity preserved along the ray), and c^T d < 0 (for MIN).
# -------------------------------------------------------------------

from __future__ import annotations
import numpy as np
from typing import Callable, Dict, Any, List, Tuple

from .simplex_standard import UnboundedSimplex  # typed signal from the simplex

SimplexFn = Callable[..., tuple]  # (x_star, info)


# (§3.3.2) Row sign normalization so that we can take a=b as feasible for artificials
def _normalize_rows(A: np.ndarray, b: np.ndarray, tol: float) -> tuple[np.ndarray, np.ndarray]:
    A2 = np.asarray(A, dtype=float).copy()
    b2 = np.asarray(b, dtype=float).copy()
    for i in range(A2.shape[0]):
        if b2[i] < -tol:
            A2[i, :] *= -1.0
            b2[i]    *= -1.0
    return A2, b2


# (§3.3.2) Build the auxiliary Phase-I problem: min 1^T a  s.t. [A | I][x;a] = b
def _build_phase1(A: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, List[int], List[int]]:
    m, n = A.shape
    A1 = np.hstack([A, np.eye(m)])  # columns: x(0..n-1) | a(n..n+m-1)
    b1 = b.copy()
    c1 = np.zeros(n + m)
    c1[n:] = 1.0
    x_cols = list(range(n))
    a_cols = list(range(n, n + m))
    return A1, b1, c1, x_cols, a_cols


def _extract_final_basis(info: Dict[str, Any]) -> List[int]:
    bases = info.get("basis", None)
    if bases is None or len(bases) == 0:
        raise RuntimeError("simplex_standard must return info['basis'] with at least one entry.")
    return list(map(int, list(bases[-1])))


# Minimal cleanup: pivot out zero-valued artificials if we can (keeps feasibility)
def _pivot_out_zero_artificials(A: np.ndarray, b: np.ndarray, basis: List[int], a_cols: List[int], tol: float) -> List[int]:
    m, n = A.shape
    B = np.array(basis, dtype=int)
    # Current basic values for this basis:
    A_B = A[:, B]
    try:
        x_B = np.linalg.solve(A_B, b)
    except np.linalg.LinAlgError:
        x_B = np.zeros(m)

    for i in range(m):
        if B[i] in a_cols and abs(x_B[i]) <= tol:
            # Try bring in any original nonbasic column with nonzero in this row
            nonbasic = [j for j in range(n) if j not in set(B)]
            for j in nonbasic:
                if abs(A[i, j]) > tol:
                    # page-43 single step: direction & ratio test
                    try:
                        d_B = -np.linalg.solve(A_B, A[:, j])
                    except np.linalg.LinAlgError:
                        continue
                    mask = d_B < -tol
                    if not np.any(mask):
                        continue
                    ratios = np.full(m, np.inf)
                    ratios[mask] = x_B[mask] / (-d_B[mask])
                    k = int(np.argmin(ratios))
                    if not np.isfinite(ratios[k]):
                        continue
                    theta = ratios[k]
                    x_B = x_B + theta * d_B
                    x_B[k] = theta
                    B[k] = j
                    A_B = A[:, B]  # refresh
                    break
    return list(map(int, list(B)))


def solve_two_phase(
    A: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    *,
    simplex: SimplexFn,
    tol: float = 1e-9,
    maxit: int = 10000
) -> tuple[np.ndarray, Dict[str, Any]]:
    """
    Solve min c^T x s.t. A x = b, x >= 0 via the two-phase method (per §3.3.2),
    using the same page-43 simplex in both phases. Returns either an optimal
    solution or (if unbounded) a feasible point with a recession direction.
    """
    A = np.asarray(A, dtype=float); b = np.asarray(b, dtype=float); c = np.asarray(c, dtype=float)
    m, n = A.shape

    # ----- Phase I: Build & solve the auxiliary problem (per §3.3.2) -----
    A2, b2 = _normalize_rows(A, b, tol)         # row sign normalization
    A1, b1, c1, x_cols, a_cols = _build_phase1(A2, b2)
    basis1_init = a_cols.copy()                  # artificial identity basis (feasible)

    # print("--- Phase I: Auxiliary Problem ---")
    x1, info1 = simplex(A1, b1, c1, basis=basis1_init, tol=tol, maxit=maxit)
    phase1_value = float(np.sum(x1[n:]))        # sum of artificials at optimum

    if phase1_value > max(tol, 1e-8):
        # Infeasible original LP
        return x1[:n], {
            "phase1_feasible": False,
            "phase1_value": phase1_value,
            "phase1_basis": _extract_final_basis(info1),
            "status": "infeasible",
            "hist_phase2": {}
        }

    # ----- Phase I → Phase II: get a feasible basis for Ax=b, x>=0 -----
    basis1 = _extract_final_basis(info1)        # indices over 0..n+m-1
    basis2 = _pivot_out_zero_artificials(A2, b2, basis1, a_cols, tol)
    # Prefer original columns in the Phase II warm start
    basis2_orig = [j for j in basis2 if j < n]
    if len(basis2_orig) < m:
        need = m - len(basis2_orig)
        extras = [j for j in basis2 if j not in basis2_orig]
        basis2_orig += extras[:need]
    # print(f"Phase II basis (indices): {basis2_orig}")
    # ----- Phase II: run the same simplex on the original objective -----
    try:
        x2, info2 = simplex(A2, b2, c, basis=basis2_orig, tol=tol, maxit=maxit)
        # print("--- Phase II: Optimal Solution Found ---")
        return x2, {
            "phase1_feasible": True,
            "phase1_value": phase1_value,
            "phase1_basis": basis1,
            "phase2_basis": basis2_orig,
            "status": "optimal",
            "hist_phase2": info2
        }

    except UnboundedSimplex as ub:
        B = np.array(basis2_orig, dtype=int)
        A_B = A2[:, B]
        try:
            x_B = np.linalg.solve(A_B, b2)
        except np.linalg.LinAlgError:
            x_B = np.zeros(m)
        x_feas = np.zeros(n)
        x_feas[B] = x_B

        d = ub.ray[:n].copy()                # recession direction in the (standard-form) x-space
        dir_obj_slope = float(c @ d)         # c^T d  (negative for MIN)

        info2 = {"f": [], "basis": [B.copy()], "enter_leave": []}
        return x_feas, {
            "phase1_feasible": True,
            "phase1_value": phase1_value,
            "phase1_basis": basis1,
            "phase2_basis": basis2_orig,
            "status": "unbounded",
            "reason": "ratio test failed (no leaving variable).",
            "ray": d,                               # << witness direction
            "ray_objective_slope": dir_obj_slope,   # << c^T d (should be < 0)
            "entering": ub.entering,
            "hist_phase2": info2
        }
