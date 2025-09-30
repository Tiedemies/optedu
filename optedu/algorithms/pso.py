from __future__ import annotations
import numpy as np
from typing import Sequence, Tuple, Dict, Any, Callable
from ..utils.types import pack_result, _ensure_history_dict

Bounds = Sequence[Tuple[float, float]]

def pso_minimize(f: Callable[[np.ndarray], float], *, bounds: Bounds, n_particles: int = 30,
                 iters: int = 100, w: float = 0.7, c1: float = 1.4, c2: float = 1.4,
                 rng: np.random.Generator | None = None) -> Dict[str, Any]:
    rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng()
    dim = len(bounds)
    lo = np.array([b[0] for b in bounds], float)
    hi = np.array([b[1] for b in bounds], float)

    X = rng.random((n_particles, dim)) * (hi - lo) + lo
    V = rng.normal(scale=0.1, size=(n_particles, dim))

    pbest = X.copy()
    pbest_val = np.array([f(x) for x in X], float)
    g_idx = int(np.argmin(pbest_val))
    gbest = pbest[g_idx].copy()
    gbest_val = float(pbest_val[g_idx])

    hist_f = [gbest_val]

    for _ in range(iters):
        r1 = rng.random((n_particles, dim))
        r2 = rng.random((n_particles, dim))
        V = w*V + c1*r1*(pbest - X) + c2*r2*(gbest - X)
        X = np.minimum(np.maximum(X + V, lo), hi)
        vals = np.array([f(x) for x in X], float)

        improve = vals < pbest_val
        pbest[improve] = X[improve]
        pbest_val[improve] = vals[improve]

        g_idx = int(np.argmin(pbest_val))
        if pbest_val[g_idx] < gbest_val:
            gbest_val = float(pbest_val[g_idx])
            gbest = pbest[g_idx].copy()

        hist_f.append(gbest_val)

    hist = _ensure_history_dict(f=hist_f)
    return pack_result(status="converged", x=gbest, f=gbest_val, nit=iters, history=hist)
