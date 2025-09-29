# optedu/algorithms/solve_lp.py
from __future__ import annotations
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

from optedu.problems.lp_standardize import to_standard_form
from optedu.algorithms.lp.simplex_standard import simplex_standard
from optedu.algorithms.lp.two_phase import solve_two_phase

def solve_lp(
    c: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    *,
    senses: Optional[List[str]] = None,   # e.g. ["le","eq","ge"]; if None, treated as all "eq"
    objective: str = "min",               # "min" or "max"
    lb: Optional[np.ndarray] = None,      # lower bounds (len n) or None
    ub: Optional[np.ndarray] = None,      # upper bounds (len n) or None
    tol: float = 1e-9,
    maxit: int = 10000,
    return_standard: bool = False
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Convenience solver:
      1) Convert to standard form  min c_std^T z s.t. A_std z = b_std, z >= 0
      2) Solve via two-phase using your page-43 simplex
      3) Map z* back to original variables x* and compute original objective

    Returns:
      x_star : solution in ORIGINAL variables
      info   : dict with fields:
               - 'original_value' : c^T x_star (for the given 'objective')
               - 'phase1_feasible', 'phase1_value'
               - 'phase2_basis', 'hist_phase2'
               - (optional) 'standard' : { 'c_std','A_std','b_std','z_star' } if return_standard=True
    """
    c = np.asarray(c, dtype=float)
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    n = A.shape[1]

    # default senses: all equalities if not provided
    if senses is None:
        senses = ["eq"] * A.shape[0]

    # 1) Standardize (handles objective flip for "max", bounds, frees, shifts)
    c_std, A_std, b_std, info_std = to_standard_form(
        c, A, b, senses, objective=objective, lb=lb, ub=ub
    )

    # 2) Two-phase + page-43 simplex (as black box)
    z_star, two_phase_info = solve_two_phase(
        A_std, b_std, c_std, simplex=simplex_standard, tol=tol, maxit=maxit
    )

    # 3) Map back to original variables & compute original objective value
    x_star = info_std["reconstruct"](z_star)
    # original objective value (do NOT use c_std here)
    original_value = float(c @ x_star)

    # If original 'objective' was "max", caller can interpret the value accordingly; we just compute c^T x.
    out: Dict[str, Any] = {
        "original_value": original_value,
        **two_phase_info
    }
    if return_standard:
        out["standard"] = {
            "c_std": c_std,
            "A_std": A_std,
            "b_std": b_std,
            "z_star": z_star,
        }
    return x_star, out
