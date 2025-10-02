# optedu/algorithms/gradient_descent.py
# -------------------------------------------------------------------
# Gradient Descent (steepest descent) for unconstrained minimization
# [S4] comments map to Chapter 4 generic schema stages.
# -------------------------------------------------------------------

from __future__ import annotations
from typing import Callable, Dict, Any, Literal
import numpy as np

from ..utils.types import History, ensure_array, AlgoResult  # History is the dict-like recorder used across the course
from .linesearch import backtracking_armijo, exact_line_search

Array = np.ndarray
Objective = Callable[[Array], float]
Gradient  = Callable[[Array], Array]
StepPolicy = Literal["exact", "armijo"]


def gradient_descent(
    *,
    f: Objective,
    grad: Gradient,
    x0,
    # ------------------------ [S4] Inputs / Hyperparameters ------------------------
    maxit: int = 2000,              # hard iteration cap (safety net)
    tol: float = 1e-8,              # gradient-norm tolerance for convergence
    step_policy: StepPolicy = "exact",  # default: attempt exact line search first
    # Armijo parameters (used if step_policy='armijo' or exact step is unusable)
    c1: float = 1e-4,
    rho: float = 0.5,
    t0: float = 1.0,
) -> Dict[str, Any]:
    """
    Gradient Descent with exact line search by default (via linesearch.exact_line_search),
    falling back to Armijo backtracking (linesearch.backtracking_armijo) if the exact step
    is not available or not usable (non-finite or non-positive).

    Returns (unified schema; no duplication):
        {
          "status": "converged" | "maxit" | "failed",
          "x": ndarray,           # final iterate (approximate minimizer)
          "f": float,             # objective value at x (under the problem's own sense)
          "history": History,     # the single history container (keys: f, x, grad_norm, step, meta)
          "counts": {"nit": int}  # iterations performed (successful updates)
        }
    """

    # -------------------------------- [S4] Initialization --------------------------------
    x: Array = ensure_array(x0)     # current iterate x_k
    history = History()             # unified recorder: stores series f, x, grad_norm, step, meta
    nit = 0                         # iteration counter (number of successful updates)

    # Evaluate once at the starting point and log it into history.
    #   fx : scalar f(x_k)
    #   g  : gradient ∇f(x_k)
    #   ng : gradient norm ||∇f(x_k)|| (first-order stationarity proxy)
    fx = float(f(x))
    g  = grad(x)
    nfev = 1; njev = 1   # oracle call counts
    ng = float(np.linalg.norm(g))
    history.append(x=x.copy(), f=fx, grad_norm=ng)   # no step recorded for the initial state

    status = "maxit"   # default; will switch to "converged" if the tolerance is met
    # message optional; omitted to keep the return minimal per unified contract

    # --------------------------------- Main iteration loop ---------------------------------
    while nit < maxit:
        # ------------------------------- [S4] Stopping (pre-check) -------------------------------
        # If we are already stationary at the current x, we declare convergence.
        if ng <= tol:
            status = "converged"
            break

        # ------------------------------------ [S4] Direction ------------------------------------
        # Steepest descent direction: d_k = -∇f(x_k)
        d = -g

        # ------------------------------------ [S4] Step-size -----------------------------------
        # Default policy tries an exact line search; if unusable, we fall back to Armijo.
        # exact_line_search is assumed to have signature exact_line_search(f, x, d).
        t = None
        if step_policy == "exact":
            try:
                t_candidate = float(exact_line_search(f, x, d))
                if np.isfinite(t_candidate) and t_candidate > 0.0:
                    t = t_candidate
            except Exception:
                t = None

        if t is None or step_policy == "armijo":
            # Armijo backtracking uses (f, grad, x, d, c1, rho, t0).
            # We rely on the linesearch helper; we do not try to replicate its internal logic here.
            t = backtracking_armijo(f, grad, x, d, c1=c1, rho=rho, t0=t0)

        # ------------------------------------ [S4] Update --------------------------------------
        # Apply the step: x_{k+1} = x_k + t_k d_k
        x = x + t * d

        # Recompute oracle at the new point and log aligned entries into the single history.
        fx = float(f(x)); nfev += 1
        g  = grad(x); njev += 1
        ng = float(np.linalg.norm(g))
        history.append(x=x.copy(), f=fx, grad_norm=ng, step=float(t))

        # ------------------------------- [S4] Stopping (post-update) ----------------------------
        if ng <= tol:
            status = "converged"
            nit += 1   # count this successful update
            break

        nit += 1

    # ------------------------------------ [S4] Outputs ----------------------------------------
    # We return exactly the unified structure. History is the single source of truth for traces.
    result = AlgoResult()
    result.status = status
    result.x = x
    result.f = fx
    result.history = history
    result.counts = {"nit": nit, "nfev": nfev, "njev": njev}
    return result
