# optedu/algorithms/nelder_mead.py
# -------------------------------------------------------------------
# Nelder–Mead simplex method for unconstrained minimization (6.1.1)
# Unified return type; History is the single source of truth for traces.
# History entries (per iteration):
#   - x: best-so-far point
#   - f: best-so-far objective value
#   - step: simplex diameter (progress metric for stopping/plots)
#   - meta: {"op": one of {"reflect","expand","outside_contract","inside_contract","shrink"}}
# Comments map directly to Chapter 6.1.1 nelder mead schema.
# -------------------------------------------------------------------

from __future__ import annotations
from typing import Callable, Dict, Any
import numpy as np

from ..utils.types import History, ensure_array

Array = np.ndarray
Objective = Callable[[Array], float]


def _init_simplex(x0: Array) -> Array:
    """
    Construct an initial (n+1)-vertex simplex around x0 using a simple,
    scale-aware rule. Unconstrained, so no clipping.
    """
    x0 = ensure_array(x0)
    n = x0.size
    simplex = np.tile(x0, (n + 1, 1))
    # Classic practical rule: perturb each coordinate i to form vertex i+1
    for i in range(n):
        step = 0.05 * (1.0 if x0[i] == 0.0 else abs(x0[i]))
        if step == 0.0:
            step = 0.00025  # tiny fallback
        simplex[i + 1, i] += step
    return simplex


def _simplex_order(simplex: Array, fvals: Array) -> tuple[Array, Array]:
    """Sort simplex vertices and their objective values by ascending f."""
    idx = np.argsort(fvals)
    return simplex[idx], fvals[idx]


def _centroid(simplex: Array) -> Array:
    """Centroid of the best n vertices (exclude the worst)."""
    return np.mean(simplex[:-1, :], axis=0)


def _diameter(simplex: Array) -> float:
    """Simplex diameter = max pairwise Euclidean distance among vertices."""
    m, n = simplex.shape
    dmax = 0.0
    for i in range(m):
        for j in range(i + 1, m):
            d = float(np.linalg.norm(simplex[i] - simplex[j]))
            if d > dmax:
                dmax = d
    return dmax


def nelder_mead(
    *,
    f: Objective,
    x0,
    # ------------------------  Inputs / Hyperparameters ------------------------
    maxit: int = 1000,
    tol: float = 1e-8,        # stopping tolerance (diameter and f-spread)
    alpha: float = 1.0,       # reflection coefficient
    gamma: float = 2.0,       # expansion coefficient
    rho: float = 0.5,         # contraction coefficient
    sigma: float = 0.5,       # shrink coefficient
) -> Dict[str, Any]:
    """
    Nelder–Mead (reflection / expansion / contraction / shrink).
    Follows the standard 6.1.1 decision tree.

    Returns (unified; no duplication):
        {
          "status": "converged" | "maxit",
          "x": ndarray,             # best-so-far point at termination
          "f": float,               # best-so-far objective value
          "history": History,       # per-iter best x,f; 'step'=simplex diameter; 'meta' notes op
          "counts": {"nit": int, "nfev": int}
        }
    """

    # --------------------------------  Initialization --------------------------------
    x0 = ensure_array(x0)
    simplex = _init_simplex(x0)
    m, n = simplex.shape

    # Evaluate all vertices
    fvals = np.array([f(v) for v in simplex], dtype=float)
    nfev = int(m)

    # Sort simplex by f
    simplex, fvals = _simplex_order(simplex, fvals)

    # Prepare history and log initial best-so-far
    history = History()
    best_x = simplex[0].copy()
    best_f = float(fvals[0])
    # step: diameter at initial simplex (for students/plots)
    diam = _diameter(simplex)
    history.append(x=best_x.copy(), f=best_f, step=diam, meta={"op": "init"})

    nit = 0
    status = "maxit"

    # --------------------------------- Main iteration loop ---------------------------------
    while nit < maxit:
        # ------------------------------- Stopping (pre-check) -------------------------------
        diam = _diameter(simplex)
        fspread = float(fvals[-1] - fvals[0])
        if diam <= tol and abs(fspread) <= tol:
            status = "converged"
            break

        # Current best, second worst, worst
        x_best, f_best = simplex[0], fvals[0]
        x_worst, f_worst = simplex[-1], fvals[-1]

        # Algorithm Step 2: NM uses simplex geometry: compute centroid of best n vertices
        xc = _centroid(simplex)

        # Step3: Reflection
        xr = xc + alpha * (xc - x_worst)
        fr = float(f(xr)); nfev += 1

        op = "reflect"

        if f_best <= fr < fvals[-2]:
            # Accept reflection (replace worst)
            simplex[-1] = xr
            fvals[-1] = fr
            op = "reflect"

        elif fr < f_best:
            # step 4: Expansion or reflection? 
            xe = xc + gamma * (xr - xc)
            fe = float(f(xe)); nfev += 1
            if fe < fr:
                simplex[-1] = xe
                fvals[-1] = fe
                op = "expand"
            else:
                simplex[-1] = xr
                fvals[-1] = fr
                op = "reflect"

        else:
            # step 5: Contraction
            if fr < f_worst:
                # Outside contraction
                xoc = xc + rho * (xr - xc)
                foc = float(f(xoc)); nfev += 1
                if foc <= fr:
                    simplex[-1] = xoc
                    fvals[-1] = foc
                    op = "outside_contract"
                else:
                    # step 6: Shrink
                    for i in range(1, m):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = float(f(simplex[i])); nfev += 1
                    op = "shrink"
            else:
                # step 5 alternative: Inside contraction
                xic = xc - rho * (xc - x_worst)
                fic = float(f(xic)); nfev += 1
                if fic < f_worst:
                    simplex[-1] = xic
                    fvals[-1] = fic
                    op = "inside_contract"
                else:
                    # step 6: Shrink
                    for i in range(1, m):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = float(f(simplex[i])); nfev += 1
                    op = "shrink"

        # Step 1: Reorder simplex after the update
        simplex, fvals = _simplex_order(simplex, fvals)

        # -----------------------------  Logging / History ----------------------------------
        best_x = simplex[0].copy()
        best_f = float(fvals[0])
        diam = _diameter(simplex)
        history.append(x=best_x.copy(), f=best_f, step=diam, meta={"op": op})

        # ------------------------------- Stopping rules (post-update) ----------------------------
        if diam <= tol and float(fvals[-1] - fvals[0]) <= tol:
            status = "converged"
            nit += 1
            break

        nit += 1

    # ------------------------------------ [S4] Outputs ----------------------------------------
    return {
        "status": status,
        "x": best_x,
        "f": best_f,
        "history": history,            # single source of truth; includes step (diam) and meta (op)
        "counts": {"nit": nit, "nfev": nfev},
    }
