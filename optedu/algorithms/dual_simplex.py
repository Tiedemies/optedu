# -------------------------------------------------------------------
# Dual Simplex for STANDARD FORM:
#     min c^T x   s.t.  A x = b,  x >= 0
#
# Assumes/constructs a DUAL-FEASIBLE basis (reduced costs >= -tol),
# then iterates until primal feasibility (x_B >= -tol) is reached.
#
# References to the course material:
#   [P71:1-7] Dual Simplex algorithm steps (Chapter 3).
# Notation aligned with Chapter 3 but ascii:
#   x_B : basic primal values
#   pi_B: simplex multipliers (dual basic solution)
#   c_hat_N : reduced costs of nonbasic variables
#   a_hat_r : r-th row of B^{-1} N
# -------------------------------------------------------------------


from __future__ import annotations
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from ..utils.types import History, LPExtras, AlgoResult, Status




def _is_dual_feasible(A: np.ndarray, c: np.ndarray, B_idx: np.ndarray, tol: float) -> bool:
    """Return True if basis B is dual-feasible (minimization): c_hat_N = c_N - A_N^T pi_B >= -tol."""
    m, n = A.shape
    N_idx = np.array([j for j in range(n) if j not in set(B_idx)], dtype=int)
    A_B = A[:, B_idx]
    try:
        pi_B = np.linalg.solve(A_B.T, c[B_idx])  # pi_B solves A_B^T pi_B = c_B
    except np.linalg.LinAlgError:
        return False
    if N_idx.size == 0:
        return True
    A_N = A[:, N_idx]
    c_hat_N = c[N_idx] - A_N.T @ pi_B
    return np.all(c_hat_N >= -tol)

def _find_dual_feasible_basis(
    A: np.ndarray,
    c: np.ndarray,
    tol: float = 1e-10,
    max_swaps: int = 5_000
) -> Optional[np.ndarray]:
    """
    Heuristic search for a dual-feasible basis:
      1) Try identity/slack basis (if present).
      2) Greedy single-column swaps that strictly reduce the number/size of negative reduced costs.
    Returns np.ndarray of length m with column indices of a dual-feasible basis, or None.
    """
    m, n = A.shape

    # Initialize with first m linearly independent columns (fallback).
    # Build via a thin QR to pick a nonsingular set if needed.
    # (Robustness: in degenerate cases this may fail.)
    try:
        import numpy.linalg as nla
        # Greedy linearly independent set using QR with pivoting
        Q, R, piv = nla.qr(A, mode='reduced', pivoting=True)  # type: ignore
        B_try = np.array(piv[:m], dtype=int)
        # Ensure nonsingular A_B
        A_B = A[:, B_try]
        if abs(np.linalg.det(A_B)) < 1e-14:
            # Fall back to a simple scan
            raise Exception("QR picked nearly singular basis")
    except Exception:
        # Very basic fallback: scan columns to pick m with growing rank
        B_list = []
        used = set()
        for j in range(n):
            if len(B_list) == m:
                break
            A_cols = A[:, B_list + [j]]
            if np.linalg.matrix_rank(A_cols) == len(B_list) + 1:
                B_list.append(j)
                used.add(j)
        if len(B_list) < m:
            return None
        B_try = np.array(B_list, dtype=int)

    def neg_reduced_costs_count(A, c, B_idx):
        n_all = A.shape[1]
        N_idx = np.array([j for j in range(n_all) if j not in set(B_idx)], dtype=int)
        A_B = A[:, B_idx]
        try:
            pi_B = np.linalg.solve(A_B.T, c[B_idx])
        except np.linalg.LinAlgError:
            return np.inf, None, None, None
        if N_idx.size == 0:
            return 0, pi_B, N_idx, np.empty((0,))
        c_hat_N = c[N_idx] - A[:, N_idx].T @ pi_B
        count = int(np.sum(c_hat_N < -tol))
        total_violation = float(np.sum(np.minimum(c_hat_N, 0.0)))
        # Prefer fewer negatives; break ties by less total violation
        score = (count, total_violation)
        return score, pi_B, N_idx, c_hat_N

    # If already dual-feasible, done.
    if _is_dual_feasible(A, c, B_try, tol):
        return B_try

    # Greedy single swaps
    swaps = 0
    best_score, _, _, _ = neg_reduced_costs_count(A, c, B_try)
    improved = True
    while swaps < max_swaps and improved:
        improved = False
        base_score = best_score
        N_idx_full = np.array([j for j in range(n) if j not in set(B_try)], dtype=int)
        for i in range(m):
            for j in N_idx_full:
                B_new = B_try.copy()
                B_new[i] = j
                # Must stay nonsingular
                try:
                    if np.linalg.matrix_rank(A[:, B_new]) < m:
                        continue
                except np.linalg.LinAlgError:
                    continue
                new_score, _, _, _ = neg_reduced_costs_count(A, c, B_new)
                if new_score[0] < base_score[0] or (
                    new_score[0] == base_score[0] and new_score[1] > base_score[1]
                ):
                    B_try = B_new
                    best_score = new_score
                    improved = True
                    swaps += 1
                    if _is_dual_feasible(A, c, B_try, tol):
                        return B_try
                    break
            if improved:
                break
    # Final check
    if _is_dual_feasible(A, c, B_try, tol):
        return B_try
    return None

def dual_simplex_standard(
    A: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    *,
    basis: Optional[List[int]] = None,
    tol: float = 1e-9,
    maxit: int = 10000
):
    print("Using dual_simplex_standard")
    """
    Page-71 dual simplex for:  min c^T x  s.t. A x = b, x >= 0

    Steps (matching the material):
      [P71:1] Choose a dual feasible basis B0.
      [P71:2] Solve x_B from A_B x_B = b. If x_B >= 0, stop: (x_B, pi_B) optimal.
      [P71:3] Choose an index r with x_B[r] < 0 (e.g., most negative component).
      [P71:4] Let a_hat_r be the r-th row of B^{-1} N.
      [P71:5] If a_hat_r >= 0, stop: primal infeasible.
      [P71:6] Compute reduced costs c_hat_N, choose s minimizing (-c_hat_N[j]/a_hat_r[j]) over a_hat_r[j] < 0.
      [P71:7] Pivot: replace B[:, r] with N[:, s], and go to [P71:2].
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    m, n = A.shape
    if b.shape != (m,) or c.shape != (n,):
        raise ValueError("Dimension mismatch in dual_simplex_standard.")

    # [P71:1] Choose a dual-feasible basis (if not provided).
    if basis is None:
        B = _find_dual_feasible_basis(A, c, tol=max(tol, 1e-12))
        print("No basis. Found dual-feasible basis:", B)
        if B is None:
            return AlgoResult(status="failed", extra={"message": "No dual-feasible basis found."})
        B = np.array(B, dtype=int)
    else:
        print("Using provided basis:", basis)
        B = np.array(basis, dtype=int)
        if not _is_dual_feasible(A, c, B, tol):
            # Try to repair if the provided basis is not dual-feasible.
            B_repaired = _find_dual_feasible_basis(A, c, tol=max(tol, 1e-12))
            if B_repaired is None:
                return AlgoResult(status="failed", extra={"message": "Provided basis is not dual-feasible and repair failed."})
            B = B_repaired

    N = np.array([j for j in range(n) if j not in set(B)], dtype=int)

    hist_f: List[float] = []
    hist_basis: List[np.ndarray] = []
    hist_pivots: List[Tuple[int, int]] = []

    iters = 0
    while True:
        # Solve for the current basic values and multipliers.
        A_B = A[:, B]  # (m, m)
        print(B)
        try:
            x_B = np.linalg.solve(A_B, b)
        except np.linalg.LinAlgError as e:
            raise RuntimeError("Singular basis matrix in dual_simplex_standard.") from e

        # Build full x and objective value.
        x = np.zeros(n, dtype=float)
        x[B] = x_B
        f_val = float(c @ x)
        hist_f.append(f_val)
        hist_basis.append(B.copy())

        # Compute multipliers and reduced costs c_hat_N.
        A_N = A[:, N]
        pi_B = np.linalg.solve(A_B.T, c[B])          # [P71:2] dual basic solution
        c_hat_N = c[N] - A_N.T @ pi_B                # reduced costs (dual feasibility check already held at start)
        print(f"Iteration {iters}: f = {f_val}, x_B = {x_B}, c_hat_N = {c_hat_N}")
        # [P71:2] Optimality test: if x_B >= -tol (primal feasible) and we maintained dual feasibility ⇒ optimal.
        if np.all(x_B >= -tol):
            result = AlgoResult(
                status="converged",
                x=x,
                f=f_val,
                lp=LPExtras(basis=B.copy(), y=pi_B),
                history=History(f=hist_f, x=hist_basis, meta={"enter_leave": hist_pivots})
            )
            break

        # [P71:3] Choose r with x_B[r] < 0 (use most negative).
        r = int(np.argmin(x_B))  # most negative component
        if x_B[r] >= -tol:
            # Numerical guard: if nothing is negative by tolerance, treat as feasible.
            result = AlgoResult(
                status="converged",
                x=x,
                f=f_val,
                lp=LPExtras(basis=B.copy(), y=pi_B),
                history=History(f=hist_f, x=hist_basis, meta={"enter_leave": hist_pivots})
            )
            break

        # [P71:4] Compute a_hat_r = e_r^T B^{-1} N  (r-th row of B^{-1} N).
        try:
            E_r = np.zeros((m,))
            E_r[r] = 1.0
            # Solve B^T w = e_r  ⇒ w^T = e_r^T B^{-1}; then a_hat_r = w^T N = (B^{-1} N)_r
            w = np.linalg.solve(A_B.T, E_r)
            a_hat_r = w @ A_N  # shape (|N|,)
        except np.linalg.LinAlgError as e:
            raise RuntimeError("Singular basis while computing a_hat_r in dual_simplex_standard.") from e

        # [P71:5] If a_hat_r >= 0 ⇒ infeasible.
        if np.all(a_hat_r >= -tol):
            print(a_hat_r)
            print("Primal infeasible: no eligible entering variable.")
            result = AlgoResult(
                status="infeasible",
                lp=LPExtras(basis=B.copy()),
                history=History(f=hist_f, x=hist_basis, meta={"enter_leave": hist_pivots})
            )
            break

        # [P71:6] Choose s minimizing (-c_hat_N[j]/a_hat_r[j]) over a_hat_r[j] < 0.
        mask = a_hat_r < -tol
        ratios = np.full(a_hat_r.shape, np.inf, dtype=float)
        ratios[mask] = (-c_hat_N[mask]) / a_hat_r[mask]  # both numerator, denominator should make sense
        j_rel = int(np.argmin(ratios))
        if not np.isfinite(ratios[j_rel]):
            print("No eligible entering variable found; primal infeasible.")
            # No eligible entering variable -> infeasible (per step 5 outcome).
            result = AlgoResult(
                status="infeasible",
                lp=LPExtras(basis=B.copy()),
                history=History(f=hist_f, x=hist_basis, meta={"enter_leave": hist_pivots})
            )
            break
        s = int(N[j_rel])  # entering column index (nonbasic)

        # [P71:7] Pivot: replace r-th column of B with N[:, s].
        leaving_var = int(B[r])
        B[r] = s
        hist_pivots.append((s, leaving_var))

        # Refresh N
        N = np.array([j for j in range(n) if j not in set(B)], dtype=int)

        iters += 1
        if iters > maxit:
            result = AlgoResult(
                status="maxit",
                x=x,
                f=f_val,
                lp=LPExtras(basis=B.copy(), y=pi_B),
                history=History(f=hist_f, x=hist_basis, meta={"enter_leave": hist_pivots})
            )
            break

    return result
