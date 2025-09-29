# optedu/algorithms/lp/simplex_standard.py
# -------------------------------------------------------------------
# Primal simplex for STANDARD FORM:
#     min c^T x   s.t.  A x = b,  x >= 0
#
# This implementation is written to comport with the material:
# - "Page 43" Simplex loop steps are annotated explicitly.
# - We work with a general feasible basis B (A_B invertible, x_B = A_B^{-1} b >= 0).
# - When the ratio test fails (direction has no negative basic component), we
#   raise a typed UnboundedSimplex that contains a *recession direction* d
#   satisfying A d = 0 and c^T d < 0 (for MIN), constructed from the entering
#   column j and d_B = -A_B^{-1} a_j, with d_j = 1 and d_N\{j} = 0.
# -------------------------------------------------------------------

from __future__ import annotations
import numpy as np
from typing import Dict, Any, List, Optional, Tuple


class UnboundedSimplex(RuntimeError):
    """Typed signal for unboundedness carrying a feasible recession direction."""
    def __init__(self, message: str, *, ray: np.ndarray, entering: int,
                 basis: np.ndarray, x_B: np.ndarray, reduced_cost: float):
        super().__init__(message)
        self.ray = ray                  # full n-vector d with A d = 0
        self.entering = entering        # entering column index j
        self.basis = basis              # current basis indices B
        self.x_B = x_B                  # current basic values (feasible)
        self.reduced_cost = reduced_cost  # r_j < 0 for MIN


def _find_identity_basis(A: np.ndarray, b: np.ndarray, tol: float = 1e-10) -> Optional[List[int]]:
    m, n = A.shape
    if np.any(b < -tol):
        return None
    basis = [-1] * m
    used = set()
    for i in range(m):
        found = False
        for j in range(n):
            if j in used:
                continue
            col = A[:, j]
            if abs(col[i] - 1.0) <= 1e-10 and np.all(np.abs(col[np.arange(m) != i]) <= 1e-10):
                basis[i] = j
                used.add(j)
                found = True
                break
        if not found:
            return None
    return basis


def simplex_standard(
    A: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    *,
    basis: Optional[List[int]] = None,
    tol: float = 1e-9,
    maxit: int = 10000
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Page-43 primal simplex for:  min c^T x  s.t. A x = b, x >= 0

    Steps (matching the material):
      [P43:1] Choose feasible base
      [P43:2] Solve basic solution (A_B x_B = b; A_B^T y = c_B)
      [P43:3] Reduced costs r_N = c_N - A_N^T y
      [P43:4] Optimality test (MIN): if all r_N >= -tol -> optimal
      [P43:5] Choose entering index (Bland: smallest with r_N < -tol)
      [P43:6] Direction/unbounded: d_B = -A_B^{-1} a_j
      [P43:7] Ratio test
      [P43:8] Pivot (update basis/values)
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    m, n = A.shape
    if b.shape != (m,) or c.shape != (n,):
        raise ValueError("Dimension mismatch in simplex_standard.")

    # [P43:1] Choose feasible base unless it has been provided. 
    if basis is None:
        basis = _find_identity_basis(A, b, tol=1e-10)
        if basis is None:
            raise RuntimeError("Provide a feasible basis or run Phase I first.")
    B = np.array(basis, dtype=int)
    N = np.array([j for j in range(n) if j not in set(B)], dtype=int)

    hist_f: List[float] = []
    hist_basis: List[np.ndarray] = []
    hist_pivots: List[Tuple[int, int]] = []

    iters = 0
    while True:
        # print(f"--- Iteration {iters} ---")
        # print(f"Basis indices: {B}")
        # print(f"Nonbasis indices: {N}")
        # [P43:2] Solve basic solution
        A_B = A[:, B]                                   # (m,m)
        try:
            x_B = np.linalg.solve(A_B, b)               # basic values
        except np.linalg.LinAlgError as e:
            raise RuntimeError("Singular basis matrix in simplex_standard.") from e
        if np.any(x_B < -1e-8):
            raise RuntimeError("Provided basis is infeasible (x_B has negative components).")


        # Current x and objective
        x = np.zeros(n, dtype=float)
        x[B] = x_B
        f_val = float(c @ x)
        hist_f.append(f_val)
        hist_basis.append(B.copy())
        # print(f"Current basic solution x_B: {x_B}")
        # print(f"Current objective value: {f_val}")

        # [P43:3] Reduced costs
        A_N = A[:, N]
        y = np.linalg.solve(A_B.T, c[B])    # dual variables
        r_N = c[N] - A_N.T @ y ## This is the transpose of the notes' version
        #print(f"Reduced costs (nonbasic): {r_N}")
        # [P43:4] Optimality (MIN): All reduced costs >= -tol. We don't compare to 0
        # directly to avoid numerical noise issues.
        if np.all(r_N >= -tol):
            break

        # [P43:5] Find an index with r_j < -tol (first such index)
        neg_idx = np.where(r_N < -tol)[0]
        jN_rel = int(neg_idx.min())
        j_enter = int(N[jN_rel])
        a_j = A[:, j_enter]
        rj = float(r_N[jN_rel])
        #print(f"Entering index: {j_enter} with reduced cost {rj}")

        # [P43:6] Direction/unbounded  (construct d_B = -A_B^{-1} a_j) d_B is the a* from the notes
        try:
            d_B = -np.linalg.solve(A_B, a_j)            # direction in basic variables
        except np.linalg.LinAlgError as e:
            raise RuntimeError("Singular basis while computing direction.") from e

        # If all d_B >= -tol → no leaving variable exists → UNBOUNDED.
        mask = d_B < -tol
        if not np.any(mask):
            # Build full recession direction d:
            #   d_j = 1 for entering variable
            #   d_B = -A_B^{-1} a_j
            #   d_other_N = 0
            d = np.zeros(n, dtype=float)
            d[j_enter] = 1.0
            d[B] = d_B
            # Sanity: A d ≈ 0; objective slope = c^T d = r_j < 0 (for MIN)
            # Raise typed exception with the witness ray
            # print("Unbounded direction found (no leaving variable).")
            raise UnboundedSimplex(
                "Unbounded: no leaving variable (direction nonnegative in all basic components).",
                ray=d, entering=j_enter, basis=B.copy(), x_B=x_B.copy(), reduced_cost=rj
            )

        # [P43:7] Ratio test, to find the index in B to leave the basis. 
        ratios = np.full(m, np.inf, dtype=float)
        ratios[mask] = x_B[mask] / (-d_B[mask])
        i_leave = int(np.argmin(ratios))
        theta = ratios[i_leave]
        if not np.isfinite(theta):
            # This path should be unreachable given the mask test, but keep a guard.
            d = np.zeros(n, dtype=float)
            d[j_enter] = 1.0
            d[B] = d_B
            raise UnboundedSimplex(
                "Unbounded (no finite ratio).",
                ray=d, entering=j_enter, basis=B.copy(), x_B=x_B.copy(), reduced_cost=rj
            )
        # print(f"Leaving index: {B[i_leave]} with ratio {theta}")
        # Pivot updates (values and basis indices) stored in history (not in the notes)
        x_B = x_B + theta * d_B
        x_B[i_leave] = theta
        leaving_var = int(B[i_leave])
        B[i_leave] = j_enter
        hist_pivots.append((j_enter, leaving_var))

        # Refresh N
        N = np.array([j for j in range(n) if j not in set(B)], dtype=int)

        iters += 1
        if iters > maxit:
            raise RuntimeError("Maximum iterations exceeded in simplex_standard.")

    x_star = np.zeros(n, dtype=float)
    x_star[B] = x_B
    info: Dict[str, Any] = {
        "f": hist_f,
        "basis": hist_basis,
        "enter_leave": hist_pivots
    }
    return x_star, info
