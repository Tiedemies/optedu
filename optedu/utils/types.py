from __future__ import annotations
from typing import TypedDict, Dict, List, Any, Optional, Literal
import numpy as np

Status = Literal["converged", "maxit", "infeasible", "unbounded", "failed"]

class History(TypedDict, total=False):
    # Iteration-aligned time series (same length if present)
    f: List[float]                 # objective values over iterations (best-so-far or per-iterate, doc per algo)
    x: List[np.ndarray]            # iterates (or best-so-far positions); large objects ok if pedagogy needs it
    grad_norm: List[float]         # ||∇f|| when applicable
    step: List[float]              # step sizes (line search / trust region radii etc.), if applicable
    meta: Dict[str, Any]           # algorithm-specific trace; e.g., LP: pivots, basis sequence; SA: T; PSO: velocities

class Counts(TypedDict, total=False):
    nit: int                       # iterations performed
    nfev: int                      # objective evaluations
    njev: int                      # gradient evaluations
    nhev: int                      # Hessian evaluations

class LPExtras(TypedDict, total=False):
    basis: List[int]               # final/basic index set (LP)
    direction: np.ndarray          # feasible recession ray when status=="unbounded"
    dual: np.ndarray               # dual solution if computed
    reduced_costs: np.ndarray      # r_N at termination if useful

class AlgoResult(TypedDict, total=False):
    status: Status                 # REQUIRED
    x: np.ndarray                  # final (or best-so-far) point if meaningful
    f: float                       # achieved objective value at x (under the problem's own sense)
    history: History               # REQUIRED: present but may be partially populated
    counts: Counts                 # iteration/evaluation counts
    message: str                   # human-readable termination note
    lp: LPExtras                   # LP-specific optional attachments
    extra: Dict[str, Any]          # free-form extensibility


## Function to ensure x is an ndarray
def ensure_array(x) -> np.ndarray:
    """
    Ensure that the input is a NumPy array of dtype float.

    - If x is already an ndarray, it is cast/copied to float if needed.
    - If x is a list, tuple, or scalar, it is converted to a 1-D float array.
    """
    return np.asarray(x, dtype=float)
