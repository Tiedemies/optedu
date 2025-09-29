import numpy as np
from typing import Callable, Tuple
from ..utils.types import History, ensure_array

def nelder_mead(f: Callable[[np.ndarray], float],
                x0, step=0.1, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5,
                maxit: int = 500, tol: float = 1e-8) -> Tuple[np.ndarray, History]:
    x0 = ensure_array(x0); n = x0.size
    simplex = np.vstack([x0, x0 + step*np.eye(n)])
    hist = History()
    for k in range(maxit):
        vals = np.array([f(x) for x in simplex])
        idx = np.argsort(vals); simplex = simplex[idx]; vals = vals[idx]
        hist.append(x=simplex[0].copy(), f=float(vals[0]))
        if np.std(vals) < tol: break
        x_low, x_high = simplex[0], simplex[-1]
        centroid = simplex[:-1].mean(axis=0)
        xr = centroid + alpha*(centroid - x_high); fr = f(xr)
        if fr < vals[0]:
            xe = centroid + gamma*(xr - centroid); fe = f(xe)
            simplex[-1] = xe if fe < fr else xr
        elif fr < vals[-2]:
            simplex[-1] = xr
        else:
            if fr < vals[-1]:
                xc = centroid + rho*(xr - centroid); fc = f(xc)   # outside
            else:
                xc = centroid - rho*(xr - centroid); fc = f(xc)   # inside
            if fc < vals[-1]:
                simplex[-1] = xc
            else:
                for i in range(1, n+1):
                    simplex[i] = simplex[0] + sigma*(simplex[i] - simplex[0])
    return simplex[0], hist
