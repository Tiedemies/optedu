# Particle Swarm Optimization (PSO) implementation
from __future__ import annotations
import numpy as np
from typing import Sequence, Tuple, Dict, Any, Callable
from ..utils.types import History, AlgoResult

# The flow of the algorithm is as follows:
# 1. Initialize a swarm of particles with random positions and velocities within the given bounds.
# 2. Evaluate the objective function at each particle's position to determine their fitness.
# 3. Update each particle's best-known position (pbest) and 
# 4. the global best position (gbest) based on their fitness.
# 5. Update each particle's velocity and position based on its own experience and that of its neighbors.
# 6. Position update based on the new velocity.
# 7. Check for termination criteria (e.g., maximum iterations), othwerwise repeat from step 2.

Bounds = Sequence[Tuple[float, float]]

def pso_minimize(f: Callable[[np.ndarray], float], *, bounds: Bounds, n_particles: int = 30,
                 iters: int = 100, w: float = 0.7, c1: float = 1.4, c2: float = 1.4,
                 rng: np.random.Generator | None = None) -> Dict[str, Any]:
    rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng()
    dim = len(bounds)
    lo = np.array([b[0] for b in bounds], float)
    hi = np.array([b[1] for b in bounds], float)
    ## 1: Initialize particles' positions and velocities inside the bounds
    X = rng.random((n_particles, dim)) * (hi - lo) + lo
    ## Speeds are normized to 0.1 times the range of each dimension
    V = rng.normal(scale=0.1, size=(n_particles, dim))

    pbest = X.copy()
    ## Evaluate initial personal bests for each particle
    pbest_val = np.array([f(x) for x in X], float)
    g_idx = int(np.argmin(pbest_val))
    ## Initialize global best
    gbest = pbest[g_idx].copy()
    gbest_val = float(pbest_val[g_idx])
    hist_x = []
    hist_f = [gbest_val]
    # We run for a fixed number of iterations
    for _ in range(iters):
        ## 5: Update velocities and positions; r1 and r2 are uniform random numbers in [0,1]
        r1 = rng.random((n_particles, dim))
        r2 = rng.random((n_particles, dim))
        V = w*V + c1*r1*(pbest - X) + c2*r2*(gbest - X)

        ## 6: Position update based on the new velocity, truncate to the bounds
        X = np.minimum(np.maximum(X + V, lo), hi)
        ## Evaluate new positions
        vals = np.array([f(x) for x in X], float)

        improve = vals < pbest_val
        pbest[improve] = X[improve]
        pbest_val[improve] = vals[improve]
        ## We add the best point in this iteration to the history
        hist_x.append(gbest.copy())
        g_idx = int(np.argmin(pbest_val))
        if pbest_val[g_idx] < gbest_val:
            gbest_val = float(pbest_val[g_idx])
            gbest = pbest[g_idx].copy()
        hist_f.append(gbest_val)
    ### Return

    return AlgoResult(
        status="maxit",
        x=gbest,
        f=gbest_val,
        history=History(f=hist_f, x=hist_x),
        counts={"nit": len(hist_f)-1, "nfev": n_particles*(len(hist_f)-1) + n_particles}
    )
