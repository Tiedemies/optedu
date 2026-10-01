# optedu/algorithms/lp_simplex.py
# -------------------------------------------------------------------
# Primal simplex for STANDARD FORM:
#     min c^T x   s.t.  A x = b,  x >= 0

from __future__ import annotations
import numpy as np
from typing import List, Optional, Tuple
from ..utils.types import History, LPExtras, AlgoResult


## Find identity
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

## simplex_standard takes A, b, c and an optional starting basis, and returns an AlgoResult (LP data under result["lp"])
def simplex_standard(
    A: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    *,
    basis: Optional[List[int]] = None,
    tol: float = 1e-9,
    maxit: int = 10000
) -> AlgoResult:
    """
    Page-43 primal simplex for:  min c^T x  s.t. A x = b, x >= 0

    Steps (matching the material):
      [P43:1] Choose feasible base
      [P43:2] Solve basic solution (A_B x_B = b; A_B^T y = c_B)
      [P43:3] Reduced costs r_N = c_N - A_N^T y
      [P43:4] Optimality test (MIN): if all r_N >= -tol -> optimal
      [P43:5] Choose entering index (Bland: smallest index with r_N < -tol)
      [P43:6] Direction/unbounded: d_B = -A_B^{-1} a_j
      [P43:7] Ratio test (Bland: among ties, the smallest basic index leaves)
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
            return AlgoResult(status="failed", history=History(),
                              message="No identity basis found in A; pass a feasible basis or use two-phase.")
    B = np.array(basis, dtype=int)
    N = np.array([j for j in range(n) if j not in set(B)], dtype=int)

    hist_f: List[float] = []
    hist_basis: List[np.ndarray] = []
    hist_pivots: List[Tuple[int, int]] = []
    # One history for every outcome. LP iterates are vertices, so the per-iteration trace is the
    # objective value f; the visited bases and the (entering, leaving) pivots go in meta.
    history = History(f=hist_f, meta={"basis": hist_basis, "enter_leave": hist_pivots})

    iters = 0
    while True:
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

        # [P43:3] Reduced costs
        A_N = A[:, N]
        y = np.linalg.solve(A_B.T, c[B])    # dual variables
        r_N = c[N] - A_N.T @ y ## This is the transpose of the notes' version
        # [P43:4] Optimality (MIN): All reduced costs >= -tol. We don't compare to 0
        # directly to avoid numerical noise issues.
        if np.all(r_N >= -tol):
            result = AlgoResult(
                status="converged",
                x=x,
                f=f_val,
                lp=LPExtras(basis=B.copy()),
                history=history,
                counts={"nit": iters}
            )            
            break

        # [P43:5] Find an index with r_j < -tol (first such index)
        neg_idx = np.where(r_N < -tol)[0]
        jN_rel = int(neg_idx.min())
        j_enter = int(N[jN_rel])
        a_j = A[:, j_enter]

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
            # Return the current vertex x together with the witness ray d: x + t d is feasible for all t >= 0.
            result = AlgoResult(
                status="unbounded",
                x=x,
                f=f_val,
                lp=LPExtras(basis=B.copy(), direction=d),
                history=history,
                counts={"nit": iters}
            )
            break

        # [P43:7] Ratio test, to find the index in B to leave the basis. 
        ratios = np.full(m, np.inf, dtype=float)
        ratios[mask] = x_B[mask] / (-d_B[mask])
        theta = float(np.min(ratios))
        # Bland's rule for the leaving variable: among tied ratios, the smallest variable index leaves.
        ties = np.where(ratios <= theta + tol)[0]
        i_leave = int(ties[np.argmin(B[ties])])
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
            result = AlgoResult(
                status="maxit",
                x=x,
                f=f_val,
                lp=LPExtras(basis=B.copy()),
                history=history,
                counts={"nit": iters}
            )
            break

    # Finalize result
    return result 