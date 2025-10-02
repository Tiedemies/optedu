# optedu/algorithms/bfgs.py
# -------------------------------------------------------------------
# Quasi-Newton BFGS (inverse-Hessian form) for unconstrained minimization
# Unified return type; History is the single source of truth for traces.
# [S4] comments map to Chapter 4 generic schema stages.
# -------------------------------------------------------------------

from __future__ import annotations
from typing import Callable, Dict, Any, Literal
import numpy as np

from ..utils.types import History, ensure_array          # History: dict-like recorder with .append(...)
from .linesearch import backtracking_armijo, exact_line_search
from ..utils.types import AlgoResult

Array = np.ndarray
Objective = Callable[[Array], float]
Gradient  = Callable[[Array], Array]
StepPolicy = Literal["exact", "armijo"]


def bfgs(
    *,
    f: Objective,
    grad: Gradient,
    x0,
    # ------------------------ [S4] Inputs / Hyperparameters ------------------------
    maxit: int = 1000,            # iteration cap
    tol: float = 1e-8,            # ||∇f|| stopping tolerance
    safeguard: bool = False,       # if True, safeguard direction to ensure descent
    step_policy: StepPolicy = "exact",   # default: exact line search if usable, else Armijo
    c1: float = 1e-4,             # Armijo parameter
    rho: float = 0.5,             # backtracking contraction
    t0: float = 1.0,              # initial trial step
) -> Dict[str, Any]:
    """
    BFGS with inverse-Hessian approximation H_k:
        p_k = - H_k ∇f(x_k),    
        H_{k+1} = (I - rho_k s_k y_k^T) H_k (I - rho_k y_k s_k^T) + rho_k s_k s_k^T,
    where   
        s_k = x_{k+1}-x_k,  y_k = ∇f(x_{k+1})-∇f(x_k),  
        rho_k = 1 / (y_k^T s_k).

    - If the search direction is not a descent direction (g^T p >= 0), 
      we may **safeguard** by resetting H_k := I and using steepest 
      descent p := -g for this step.
    - Line search policy: exact_line_search by default; Armijo fallback as needed.

    Returns (unified; no duplication):
        {
          "status": "converged" | "maxit" | "failed",
          "x": ndarray,
          "f": float,
          "history": History,    # keys: f, x, grad_norm, step, meta (meta unused here)
          "counts": {"nit": int, "nfev": int, "njev": int}
        }
    """

    # -------------------------------- [S4] Initialization --------------------------------
    x: Array = ensure_array(x0)        # current iterate x_k
    n = x.size

    history = History()                # single history container
    H = np.eye(n, dtype=float)         # inverse-Hessian approximation H_0 = I

    fx = float(f(x)); nfev = 1         # objective at x_0
    g  = grad(x);      njev = 1        # gradient at x_0
    ng = float(np.linalg.norm(g))      # gradient norm

    result = AlgoResult()               # final output dict

    history.append(x=x.copy(), f=fx, grad_norm=ng)  # initial log (no step yet)
    result.status = "maxit"
    nit = 0 # Iteration counter
    njev = 1 # gradient evaluations
    nfev = 1 # function evaluations


    # --------------------------------- Main iteration loop ---------------------------------
    while nit < maxit:
        # ------------------------------- [S4] Stopping (pre-check) -------------------------------
        if ng <= tol:
            result.status = "converged"
            break

        # ------------------------------------ [S4] Direction ------------------------------------
        p = -H @ g                            # quasi-Newton direction
        if safeguard and float(np.dot(g, p)) >= 0.0:        # safeguard: if not descent, reset and use -g
            H[:] = np.eye(n, dtype=float)
            p = -g

        # ------------------------------------ [S4] Step-size -----------------------------------
        # Try exact line search; if invalid or step_policy='armijo', use Armijo.
        t = None
        if step_policy == "exact":
            try:
                t_candidate = float(exact_line_search(f, x, p))
                if np.isfinite(t_candidate) and t_candidate > 0.0:
                    t = t_candidate
            except Exception:
                t = None
        if t is None or step_policy == "armijo":
            t = backtracking_armijo(f, grad, x, p, c1=c1, rho=rho, t0=t0)

        # ------------------------------------ [S4] Update --------------------------------------
        x_new = x + t * p
        fx_new = float(f(x_new)); nfev += 1
        g_new  = grad(x_new);      njev += 1
        ng_new = float(np.linalg.norm(g_new))

        # BFGS curvature pair
        s = x_new - x
        y = g_new - g
        ys = float(np.dot(y, s))      # denominator for ρ_k

        # Update H if curvature condition holds (y^T s > 0); else skip (keeps H PSD)
        if ys > 1e-12:
            rho_k = 1.0 / ys
            # Rank-2 update: (I - ρ s y^T) H (I - ρ y s^T) + ρ s s^T
            I = np.eye(n, dtype=float)
            V = I - rho_k * np.outer(s, y)
            H = V @ H @ V.T + rho_k * np.outer(s, s)
        # else: skip update to avoid violating curvature; next iteration proceeds with current H

        # Advance state and log
        x, fx, g, ng = x_new, fx_new, g_new, ng_new
        history.append(x=x.copy(), f=fx, grad_norm=ng, step=float(t))

        # ------------------------------- [S4] Stopping (post-update) ----------------------------
        if ng <= tol:
            result.status = "converged"
            nit += 1
            break

        nit += 1

    # ------------------------------------ [S4] Outputs ----------------------------------------
    result.status = result.status if result.status == "converged" else "maxit"
    result.x = x
    result.f = fx
    result.history = history
    result.counts = {"nit": nit, "nfev": nfev, "njev": njev}
    return result