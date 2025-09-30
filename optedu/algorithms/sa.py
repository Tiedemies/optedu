# optedu/algorithms/sa.py
from __future__ import annotations
import numpy as np
from typing import Callable, Optional, Sequence, Tuple, Dict, Any, List

Array = np.ndarray
Bounds = Optional[Sequence[Tuple[float, float]]]

def _ensure_rng(rng: Optional[np.random.Generator]) -> np.random.Generator:
    return rng if isinstance(rng, np.random.Generator) else np.random.default_rng()

def _proposal_gaussian(x: Array, step_scale: float, rng: np.random.Generator) -> Array:
    return x + rng.normal(0.0, step_scale, size=x.shape)

def _project_bounds(x: Array, bounds: Sequence[Tuple[float, float]]) -> Array:
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    return np.minimum(hi, np.maximum(lo, x))

def simulated_annealing(
    f: Callable[[Array], float],
    x0: Array,
    T0: float = 1.0,
    alpha: float = 0.95,
    iters: int = 1000,
    step_scale: float = 0.5,
    bounds: Bounds = None,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """
    Simulated Annealing (Metropolis) for continuous minimization.

    Parameters
    ----------
    f : callable
        Objective f(x) -> scalar.
    x0 : array_like
        Initial point.
    T0 : float
        Initial temperature (> 0).
    alpha : float
        Cooling factor in (0, 1). Temperature follows T_k = T0 * alpha^k.
    iters : int
        Number of iterations.
    step_scale : float
        Std of Gaussian proposal.
    bounds : sequence of (low, high) or None
        If provided, proposals are clipped to these bounds.
    rng : numpy.random.Generator
        Optional RNG for determinism.

    Returns
    -------
    dict with keys: best_x, best_value, nit, status, history
    """
    rng = _ensure_rng(rng)
    x = np.asarray(x0, dtype=float).copy()
    fx = float(f(x))
    best_x = x.copy()
    best_val = fx

    T = max(T0, 1e-12)
    history: List[float] = [fx]

    for k in range(iters):
        y = _proposal_gaussian(x, step_scale, rng)
        if bounds is not None:
            y = _project_bounds(y, bounds)

        fy = float(f(y))
        delta = fy - fx
        if delta <= 0.0:
            # Accept improvement
            x, fx = y, fy
        else:
            # Metropolis acceptance
            if rng.random() < np.exp(-delta / max(T, 1e-12)):
                x, fx = y, fy

        if fx < best_val:
            best_val = fx
            best_x = x.copy()

        history.append(best_val)
        T *= alpha  # geometric cooling

    return {
        "best_x": best_x,
        "best_value": best_val,
        "nit": iters,
        "status": "converged" if iters > 0 else "maxit",
        "history": history,
    }
