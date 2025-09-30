# optedu/algorithms/genetic.py
# -------------------------------------------------------------------
# Genetic Algorithm (GA) for unconstrained minimization on a box domain
# The implementation follows Chapter 6 §6.2.1 exactly, annotated:
#     [Step 1] Initialization
#     [Step 2] Evaluation
#     [Step 3] Selection
#     [Step 4] Crossover
#     [Step 5] Mutation
#     [Step 6] Replacement
#     [Step 7] Termination
# -------------------------------------------------------------------

from __future__ import annotations
from typing import Callable, Dict, Any, Sequence, Tuple, Optional
import numpy as np

from ..utils.types import History  # dict-like recorder with .append(...)

Array   = np.ndarray
Bounds  = Sequence[Tuple[float, float]]
Objective = Callable[[Array], float]


# ---------- Utilities (simple, didactic) ----------
def _clip_to_bounds(x: Array, lo: Array, hi: Array) -> Array:
    """Clamp a vector elementwise to the box [lo, hi]."""
    return np.minimum(np.maximum(x, lo), hi)

def _init_population(rng: np.random.Generator, pop_size: int, lo: Array, hi: Array) -> Array:
    """Uniformly sample 'pop_size' individuals in the box [lo, hi]."""
    return rng.random((pop_size, lo.size)) * (hi - lo) + lo

def _tournament_select(rng: np.random.Generator, fitness: Array, k: int = 2) -> int:
    """
    Tournament selection (size k). Lower fitness is better (minimization).
    Returns the index of the selected parent.
    """
    idx = rng.integers(0, fitness.size, size=k)
    return int(idx[np.argmin(fitness[idx])])

def _uniform_crossover(rng: np.random.Generator, a: Array, b: Array) -> Array:
    """Gene-wise uniform crossover between parents a and b."""
    mask = rng.random(a.size) < 0.5
    return np.where(mask, a, b)

def _gaussian_mutation(rng: np.random.Generator, x: Array, scale: Array, p_mut: float) -> Array:
    """
    Independent Gaussian mutation per gene with probability p_mut.
    The scale is typically a fraction of the box width.
    """
    mask = rng.random(x.size) < p_mut
    noise = rng.normal(loc=0.0, scale=scale, size=x.size)
    y = x.copy()
    y[mask] += noise[mask]
    return y


def genetic_minimize(
    *,
    f: Objective,
    bounds: Bounds,
    pop_size: int = 30,
    generations: int = 100,
    p_mut: float = 0.1,
    elitism: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """
    Genetic Algorithm (GA) following Chapter 6 §6.2.1 (7-step scheme).

    Parameters
    ----------
    f : callable
        Objective function f(x) -> scalar, to be minimized.
    bounds : sequence of (lo, hi)
        Closed interval per coordinate (box domain).
    pop_size : int
        Population size (number of individuals per generation).
    generations : int
        Number of generational updates.
    p_mut : float
        Per-gene mutation probability.
    elitism : bool
        If True, carry the current best individual unchanged to the next generation.
    rng : np.random.Generator or None
        Random generator for reproducibility (default: np.random.default_rng()).

    Returns (unified; no duplication)
    ---------------------------------
    {
      "status": "maxit",              # GA runs for a fixed budget by default
      "x": ndarray,                   # best-so-far individual at termination
      "f": float,                     # best-so-far objective value
      "history": History,             # keys used here: f, x (no gradients or steps for GA)
      "counts": {"nit": generations, "nfev": <total evaluations>}
    }
    """
    rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng()

    # Parse and validate bounds
    bounds_arr = np.asarray(bounds, dtype=float)
    if bounds_arr.ndim != 2 or bounds_arr.shape[1] != 2:
        raise ValueError("bounds must be a sequence of (lo, hi) pairs.")
    lo = bounds_arr[:, 0]
    hi = bounds_arr[:, 1]
    if not np.all(hi >= lo):
        raise ValueError("Each bound must satisfy hi >= lo.")
    dim = lo.size

    # Mutation scale: fraction of box width (simple, effective default)
    mut_scale = 0.1 * (hi - lo)

    # ----------------------------- [Step 1] Initialization -----------------------------
    # Generate initial population uniformly in the box.
    X = _init_population(rng, pop_size, lo, hi)

    # ------------------------------- [Step 2] Evaluation -------------------------------
    # Evaluate fitness for the initial population.
    fitness = np.array([f(ind) for ind in X], dtype=float)
    nfev = int(pop_size)

    # Prepare unified history (single source of truth).
    history = History()
    best_idx = int(np.argmin(fitness))
    best_x = X[best_idx].copy()
    best_f = float(fitness[best_idx])
    # Log initial best-so-far snapshot (generation 0)
    history.append(x=best_x.copy(), f=best_f)

    nit = 0
    status = "maxit"   # GA typically uses a fixed budget; we report "maxit" at completion.

    # =============================== Generational loop ================================
    while nit < generations:
        # -------------------------------- [Step 3] Selection --------------------------------
        # We will build the next generation by repeatedly selecting parents
        # using tournament selection biased toward lower (better) fitness.

        # -------------------------------- [Step 4] Crossover --------------------------------
        # Gene-wise uniform crossover forms offspring from parent pairs.

        # -------------------------------- [Step 5] Mutation ---------------------------------
        # Apply independent Gaussian mutations with probability p_mut per gene.

        # Create the next generation container
        X_next = np.empty_like(X)

        # Optional elitism: carry current best to next generation unchanged
        start = 0
        if elitism:
            X_next[0] = best_x
            start = 1

        for i in range(start, pop_size):
            # Parent selection (tournament of size 2)
            i1 = _tournament_select(rng, fitness, k=2)
            i2 = _tournament_select(rng, fitness, k=2)
            p1, p2 = X[i1], X[i2]

            # Crossover
            child = _uniform_crossover(rng, p1, p2)

            # Mutation
            child = _gaussian_mutation(rng, child, mut_scale, p_mut)

            # Enforce bounds
            child = _clip_to_bounds(child, lo, hi)

            X_next[i] = child

        # -------------------------------- [Step 6] Replacement -------------------------------
        # Replace current population with offspring (plus elite if enabled).
        X = X_next

        # Evaluate the new population
        fitness = np.array([f(ind) for ind in X], dtype=float)
        nfev += int(pop_size)

        # Track best-so-far across generations (monotone in f)
        gen_best_idx = int(np.argmin(fitness))
        gen_best_x = X[gen_best_idx].copy()
        gen_best_f = float(fitness[gen_best_idx])
        if gen_best_f < best_f:
            best_f = gen_best_f
            best_x = gen_best_x

        # Log one entry per generation: best-so-far (x, f)
        history.append(x=best_x.copy(), f=best_f)

        nit += 1

        # -------------------------------- [Step 7] Termination -------------------------------
        # For this basic GA, we stop after 'generations'. Alternative rules
        # (e.g., no-improvement window, target fitness) can be added later.

    # ------------------------------------ Outputs ------------------------------------
    return {
        "status": status,     # finished because the generation budget was reached
        "x": best_x,
        "f": best_f,
        "history": history,   # History only: contains 'f' and 'x' series (no grad/step in GA)
        "counts": {"nit": nit, "nfev": nfev},
    }
