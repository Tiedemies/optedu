# optedu/algorithms/newton.py
# -------------------------------------------------------------------
# Newton's method for unconstrained smooth minimization
# Unified return type; history is the single source of truth (History).
# [S4] comments map directly to Chapter 4 generic schema stages.
# -------------------------------------------------------------------

from __future__ import annotations
from typing import Callable, Dict, Any
import numpy as np

from ..utils.types import History, ensure_array, AlgoResult  # History: dict-like recorder with .append(...)
from .linesearch import backtracking_armijo

Array = np.ndarray
Objective = Callable[[Array], float]
Gradient  = Callable[[Array], Array]
Hessian   = Callable[[Array], Array]


def _pd_safeguard(H: Array, eps: float = 1e-10) -> Array:
    """
    Make a symmetric matrix positive definite by flooring its eigenvalues at 'eps'.
    Used in *modified Newton* to guarantee a descent direction.
    """
    w, V = np.linalg.eigh(0.5 * (H + H.T))   # symmetrize defensively
    w_floor = np.maximum(w, eps)
    return (V * w_floor) @ V.T


def newton(
    *,
    f: Objective,
    grad: Gradient,
    hess: Hessian,
    x0,
    # ------------------------ [S4] Inputs / Hyperparameters ------------------------
    maxit: int = 200,
    tol: float = 1e-8,         # convergence tolerance on ||∇f(x_k)||
    damped: bool = False,       # if True, use Armijo backtracking; else full Newton step
    safeguard: bool = False,   # if True, apply eigenvalue flooring (modified Newton)
    c1: float = 1e-4,          # Armijo sufficient-decrease parameter
    rho: float = 0.5,          # backtracking contraction factor
    t0: float = 1.0,           # initial step size for backtracking
) -> Dict[str, Any]:
    """
    Newton's method for minimizing f.

    - By default, uses the *raw Hessian*. This is the classical Newton method,
      which may fail if the Hessian is indefinite.
    - If safeguard=True, applies eigenvalue flooring to ensure a descent
      direction (this is *modified Newton*).

    Returns AlgoResult with fields:
        {
          "status": "converged" | "maxit" | "failed",
          "x": ndarray,             # final iterate
          "f": float,               # f(x)
          "history": History,       # keys: f, x, grad_norm, step
          "counts": {"nit": int, "nfev": int, "njev": int, "nhev": int}
        }
    """

    # -------------------------------- [S4] Initialization --------------------------------
    x: Array = ensure_array(x0)
    history = History()           # single history container

    fx = float(f(x)); nfev = 1
    g  = grad(x);      njev = 1
    ng = float(np.linalg.norm(g))

    history.append(x=x.copy(), f=fx, grad_norm=ng)
    status = "maxit"

    nit = 0
    nhev = 0

    # --------------------------------- Main iteration loop ---------------------------------
    while nit < maxit:
        # ------------------------------- [S4] Stopping (pre-check) -------------------------------
        if ng <= tol:
            status = "converged"
            break

        # ------------------------------------ [S4] Direction ------------------------------------
        H = hess(x); nhev += 1
        if safeguard:
            # Modified Newton: safeguard by flooring eigenvalues
            Huse = _pd_safeguard(H)
        else:
            # Standard Newton: use raw Hessian (may fail if not PD)
            Huse = H

        try:
            d = -np.linalg.solve(Huse, g)
        except np.linalg.LinAlgError:
            # Fall back: if Hessian is singular, use steepest descent direction
            d = -g

        # ------------------------------------ [S4] Step-size -----------------------------------
        if damped:
            # Damped Newton: Armijo backtracking to guarantee sufficient decrease
            t = backtracking_armijo(f, grad, x, d, c1=c1, rho=rho, t0=t0)
        else:
            # Undamped Newton: full step
            t = 1.0

        # ------------------------------------ [S4] Update --------------------------------------
        x = x + t * d

        fx = float(f(x)); nfev += 1
        g  = grad(x);      njev += 1
        ng = float(np.linalg.norm(g))

        history.append(x=x.copy(), f=fx, grad_norm=ng, step=float(t))

        # ------------------------------- [S4] Stopping (post-update) ----------------------------
        if ng <= tol:
            status = "converged"
            nit += 1
            break

        nit += 1

    # ------------------------------------ [S4] Outputs ----------------------------------------
    result = AlgoResult()
    result.status = status
    result.x = x
    result.f = fx
    result.history = history
    result.counts = {"nit": nit, "nfev": nfev, "njev": njev, "nhev": nhev}
    return result

# Compatibility alias for existing imports in tests/notebooks
newton_method = newton
