# optedu/algorithms/hooke_jeeves.py
# -------------------------------------------------------------------
# Hooke–Jeeves (pattern search) for unconstrained minimization
# Unified return type; History is the single source of truth for traces.
#
# Commentary emphasizes how the algorithm comports with section 6.1.2
# -------------------------------------------------------------------

from __future__ import annotations
from typing import Callable, Dict, Any
import numpy as np

from ..utils.types import History, ensure_array, AlgoResult  # History: a plain dict of lists used across the course

Array = np.ndarray
Objective = Callable[[Array], float]


def _explore(f: Objective, x: Array, delta: Array) -> tuple[Array, float, int]:
    """
    Perform an exploratory move around x with coordinate-wise probes of size delta.
    For each coordinate i, try +delta_i and -delta_i (accept first improvement).
    Returns:
        (x_new, f_new, nfev_increment)
    """
    nfev = 0
    fx = float(f(x)); nfev += 1
    x_new = x.copy()

    for i in range(x.size):
        # try +delta_i
        trial = x_new.copy()
        trial[i] += delta[i]
        f_trial = float(f(trial)); nfev += 1
        if f_trial < fx:
            x_new, fx = trial, f_trial
        else:
            # try -delta_i
            trial = x_new.copy()
            trial[i] -= delta[i]
            f_trial = float(f(trial)); nfev += 1
            if f_trial < fx:
                x_new, fx = trial, f_trial
        # move to next coordinate using the best found so far (greedy coordinate exploration)

    return x_new, fx, nfev


def hooke_jeeves(
    *,
    f: Objective,
    x0,
    # ------------------------ Inputs / Hyperparameters ------------------------
    maxit: int = 1000,          # maximum number of *outer* cycles (explore/pattern/reduce)
    tol: float = 1e-6,          # terminate when all step lengths are <= tol
    delta0: float = 0.5,        # initial step length for all coordinates
    theta: float = 0.5,         # reduction factor for step length (0 < theta < 1)
) -> Dict[str, Any]:
    """
    Hooke–Jeeves pattern search (derivative-free).

    Returns (unified; no duplication):
        {
          "status": "converged" | "maxit",
          "x": ndarray,             # best point at termination
          "f": float,               # f(x)
          "history": History,       # logs best-so-far (x,f); 'step' = max delta; 'meta' op: 'explore'|'pattern'|'reduce'
          "counts": {"nit": int, "nfev": int}
        }

    Notes for students:
      • Exploratory move: probe ±delta_i along each axis; accept greedy improvements.
      • Pattern move: if exploration improves from base x_b to x_e, try x_p = x_e + (x_e - x_b).
      • Step-length reduction: if exploration fails to improve, delta ← theta·delta.
    """
    x0 = ensure_array(x0)
    n = x0.size

    # Step lengths per coordinate (vector), all start at delta0
    delta = np.full(n, float(delta0))

    # History lists (one entry per logged event); packed into a History at the end
    x_history = []
    f_history = []
    step_history = []
    op_history = []

    def log(x, fx, op):
        x_history.append(x.copy())
        f_history.append(float(fx))
        step_history.append(float(np.max(delta)))
        op_history.append(op)

    # Evaluate start
    x_base = x0.copy()
    f_base = float(f(x_base)); nfev = 1
    log(x_base, f_base, "init")

    nit = 0
    status = "maxit"

    while nit < maxit:
        # Terminate if all probe lengths are small
        if float(np.max(delta)) <= tol:
            status = "converged"
            break

        # -------------------------------------------------
        # Exploratory move at current base and step lengths
        # -------------------------------------------------
        x_explore, f_explore, inc = _explore(f, x_base, delta)
        nfev += inc

        if f_explore < f_base:
            # Improvement found by exploration
            log(x_explore, f_explore, "explore")

            # -----------------------
            # Pattern move (accel.)
            # -----------------------
            # Direction of improvement: d = x_explore - x_base
            d = x_explore - x_base
            x_pattern = x_explore + d
            f_pattern = float(f(x_pattern)); nfev += 1

            if f_pattern < f_explore:
                # Accept pattern move: accelerated progress along d
                x_base, f_base = x_pattern, f_pattern
                log(x_base, f_base, "pattern")
            else:
                # No gain from pattern; accept exploration result as new base
                x_base, f_base = x_explore, f_explore
                # (No extra history entry; exploration already logged)

        else:
            # --------------------------------------------------
            # No exploratory improvement: reduce step length(s)
            # --------------------------------------------------
            delta *= float(theta)
            log(x_base, f_base, "reduce")

        nit += 1

    return AlgoResult(
        status=status,
        x=x_base,
        f=f_base,
        history=History(f=f_history, x=x_history, step=step_history, meta={"op": op_history}),
        counts={"nit": nit, "nfev": nfev}
    )