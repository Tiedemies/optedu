# optedu/algorithms/lp_two_phase.py
# -------------------------------------------------------------------
# Two-phase orchestration that comports with §3.3.2 in the material:
#   Phase I: build auxiliary problem min 1^T a  s.t. A x + I a = b (after row sign normalization),
#            warm-start with artificial basis, and run the *same* page-43 simplex.
#   Phase II: restore original c and run the same simplex again from the feasible basis.
#
# Pedagogical notes:
#   • We *only* use the page-43 simplex as a black box.
#   • If Phase II is unbounded, the simplex returns status "unbounded" with a witness ray
#     result["lp"]["direction"] = d: A d = 0, d >= 0 (feasible along the ray) and c^T d < 0 (for MIN).
# -------------------------------------------------------------------

from __future__ import annotations
import numpy as np
from typing import Any, List
from ..utils.types import AlgoResult
from .lp_simplex import simplex_standard
from ..problems.lp_standardize import to_standard_form


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


# (§3.3.2) After Phase I some artificials may still be basic, at value zero (degenerate case).
# For each such row i, look at row i of the tableau B^{-1} A. Any original column with a
# nonzero entry there can replace the artificial (a degenerate pivot: x does not change).
# If there is no such column, constraint i is a combination of the others (redundant) and is dropped.
def _drive_out_artificials(A1: np.ndarray, basis: List[int], n: int, tol: float) -> tuple[List[int], List[int]]:
    """Return (basis with only original columns, indices of the constraint rows to keep)."""
    m = A1.shape[0]
    B = [int(j) for j in basis]
    keep = []
    for i in range(m):
        if B[i] < n:                       # an original column is basic in row i: nothing to do
            keep.append(i)
            continue
        # Row i of the tableau: e_i^T A_B^{-1} A  (only the original columns matter)
        row_i = np.linalg.solve(A1[:, B].T, np.eye(m)[i]) @ A1[:, :n]
        candidates = [j for j in range(n) if j not in B and abs(row_i[j]) > tol]
        if candidates:
            B[i] = candidates[0]           # degenerate pivot: artificial leaves, column j enters
            keep.append(i)
        # else: redundant constraint -> row i is not kept
    return [B[i] for i in keep], keep

def solve_two_phase_generic(
    A: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    *,
    senses: List[str],
    objective: str = "min",
    lb: Any = None,
    ub: Any = None,
    tol: float = 1e-9,
    maxit: int = 10000,
) -> AlgoResult:
    """
    Solve a general LP via the two-phase method (per §3.3.2),
    using the same page-43 simplex in both phases. Returns either an optimal
    solution or (if unbounded) a feasible point with a recession direction.

    Parameters
    ----------
    A, b, c : original LP data, with n variables and m constraints
    senses  : list of length m with entries in {"le","eq","ge"}
    objective : "min" (default) or "max"
    lb, ub : arrays of length n with lower/upper bounds. Default lb = 0 (x >= 0, the course
             convention) and ub = +inf. Use:
             - lb[i] = -np.inf to indicate no lower bound (free variable if also ub[i] = +inf)
             - ub[i] = +np.inf to indicate no upper bound
    tol : tolerance for feasibility and optimality
    maxit : maximum number of simplex iterations

    Returns
    -------
    AlgoResult with status "converged", "infeasible", "unbounded" or "maxit".
    x, f and (when unbounded) the ray result["lp"]["direction"] are in the ORIGINAL variables
    and objective sense; the standard-form solution z is kept in result["extra"]["standard_x"].
    """
    if lb is None:
        lb = np.zeros(len(c))                  # course convention: x >= 0
    A_std, b_std, c_std, info_std = to_standard_form(A=A, b=b, c=c, senses=senses,
                                                    objective=objective,
                                                    lb=lb, ub=ub
                                                    )
    result = solve_two_phase(A_std, b_std, c_std, tol=tol, maxit=maxit)

    # Map the standard-form solution z back to the original problem.
    reconstruct = info_std["reconstruct"]
    if result.get("x") is not None:
        z = result["x"]
        sign = -1.0 if objective.lower().startswith("max") else 1.0   # max problems were solved as min of -c
        result["x"] = reconstruct(z)
        result["f"] = sign * (float(c_std @ z) + info_std["objective_offset"])
        result["extra"] = {"standard_x": z}
    if result["status"] == "unbounded":
        # reconstruct is affine (x = M z + shift), so a ray d in z maps to M d = reconstruct(d) - reconstruct(0)
        d = result["lp"]["direction"]
        result["lp"]["direction"] = reconstruct(d) - reconstruct(np.zeros_like(d))
    return result


def solve_two_phase(
    A: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    *,
    tol: float = 1e-9,
    maxit: int = 10000,
) -> AlgoResult:
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

    first_result = simplex_standard(A1, b1, c1, basis=basis1_init, tol=tol, maxit=maxit)
    x1 = first_result["x"]
    phase1_value = float(np.sum(x1[n:]))        # sum of artificials at optimum

    if phase1_value > max(tol, 1e-8):
        # Infeasible original LP
        return AlgoResult(status="infeasible", x=None, f=np.inf, history=first_result["history"],
                          counts=first_result["counts"],
                          message=f"Phase I optimum {phase1_value:.3g} > 0: the LP has no feasible point.")

    # ----- Phase I → Phase II: get a feasible basis for Ax=b, x>=0 -----
    basis1 = first_result["lp"]["basis"]
    basis2, keep = _drive_out_artificials(A1, basis1, n, tol)
    A2, b2 = A2[keep, :], b2[keep]               # drop redundant constraints (if any)

    # ----- Phase II: run the same simplex on the original objective -----
    return simplex_standard(A2, b2, c, basis=basis2, tol=tol, maxit=maxit)
    