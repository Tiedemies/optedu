# optedu/algorithms/genetic.py
from __future__ import annotations
import numpy as np
from typing import Callable, Iterable, List, Sequence, Tuple, Optional, Dict, Any

Array = np.ndarray
Bounds = Sequence[Tuple[float, float]]

def _ensure_rng(rng: Optional[np.random.Generator]) -> np.random.Generator:
    return rng if isinstance(rng, np.random.Generator) else np.random.default_rng()

def _clip_bounds(x: Array, bounds: Bounds) -> Array:
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    return np.clip(x, lo, hi)

def _init_population(pop_size: int, bounds: Bounds, rng: np.random.Generator) -> Array:
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    return rng.random((pop_size, len(bounds))) * (hi - lo) + lo

def _tournament_select(fvals: Array, k: int, rng: np.random.Generator) -> int:
    """Return index of best among k random contestants (minimization)."""
    idxs = rng.integers(0, fvals.size, size=k)
    return idxs[np.argmin(fvals[idxs])]

def _blend_crossover(p1: Array, p2: Array, alpha: float, rng: np.random.Generator) -> Tuple[Array, Array]:
    """
    BLX-α crossover for real-coded GA.
    For each gene j, sample from U(min-α*range, max+α*range).
    """
    lo = np.minimum(p1, p2)
    hi = np.maximum(p1, p2)
    span = hi - lo
    c_lo = lo - alpha * span
    c_hi = hi + alpha * span
    c1 = rng.random(p1.shape) * (c_hi - c_lo) + c_lo
    c2 = rng.random(p1.shape) * (c_hi - c_lo) + c_lo
    return c1, c2

def _gaussian_mutation(x: Array, sigma: float, rng: np.random.Generator) -> Array:
    return x + rng.normal(0.0, sigma, size=x.shape)

def genetic_minimize(
    f: Callable[[Array], float],
    bounds: Bounds,
    pop_size: int = 40,
    generations: int = 100,
    tournament_k: int = 3,
    p_crossover: float = 0.9,
    p_mut: float = 0.1,
    sigma_mut: float = 0.1,
    elitism: bool = True,
    alpha_blx: float = 0.3,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """
    Real-coded Genetic Algorithm for continuous minimization.

    Parameters
    ----------
    f : callable
        Objective f(x) -> scalar.
    bounds : sequence of (low, high)
        Variable bounds for each dimension.
    pop_size : int
        Population size (>= 4).
    generations : int
        Number of generations.
    tournament_k : int
        Tournament size for parent selection.
    p_crossover : float
        Probability of crossover for a mating pair.
    p_mut : float
        Per-individual mutation probability.
    sigma_mut : float
        Mutation noise std (absolute, in same units as x).
    elitism : bool
        Keep the best individual unmodified each generation.
    alpha_blx : float
        BLX-α crossover expansion parameter (0.0–0.5 typical).
    rng : numpy.random.Generator
        Optional RNG for determinism.

    Returns
    -------
    dict with keys: best_x, best_value, nit, status, history
    """
    rng = _ensure_rng(rng)
    assert pop_size >= 4, "pop_size must be at least 4"
    n = len(bounds)

    pop = _init_population(pop_size, bounds, rng)
    fvals = np.array([f(ind) for ind in pop], dtype=float)

    best_idx = int(np.argmin(fvals))
    best_x = pop[best_idx].copy()
    best_val = float(fvals[best_idx])

    history: List[float] = [best_val]

    for g in range(generations):
        new_pop: List[Array] = []

        # Elitism: carry best forward
        if elitism:
            new_pop.append(best_x.copy())

        # Fill the rest
        while len(new_pop) < pop_size:
            # Select parents via tournament
            i1 = _tournament_select(fvals, tournament_k, rng)
            i2 = _tournament_select(fvals, tournament_k, rng)
            p1, p2 = pop[i1], pop[i2]

            c1, c2 = p1.copy(), p2.copy()
            if rng.random() < p_crossover:
                c1, c2 = _blend_crossover(p1, p2, alpha_blx, rng)

            # Mutation (per individual)
            if rng.random() < p_mut:
                c1 = _gaussian_mutation(c1, sigma_mut, rng)
            if rng.random() < p_mut:
                c2 = _gaussian_mutation(c2, sigma_mut, rng)

            # Respect bounds
            c1 = _clip_bounds(c1, bounds)
            c2 = _clip_bounds(c2, bounds)

            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)

        pop = np.vstack(new_pop)
        fvals = np.array([f(ind) for ind in pop], dtype=float)

        # Update best
        idx = int(np.argmin(fvals))
        if fvals[idx] < best_val:
            best_val = float(fvals[idx])
            best_x = pop[idx].copy()

        history.append(best_val)

    status = "converged" if generations > 0 else "maxit"
    return {
        "best_x": best_x,
        "best_value": best_val,
        "nit": generations,
        "status": status,
        "history": history,
    }
