import numpy as np
from typing import Callable, Tuple
from ..utils.types import History, ensure_array
from .linesearch import backtracking_armijo

def bfgs(f: Callable[[np.ndarray], float],
         grad: Callable[[np.ndarray], np.ndarray],
         x0, maxit: int = 500, tol: float = 1e-8,
         c1: float = 1e-4, rho: float = 0.5, t0: float = 1.0) -> Tuple[np.ndarray, History]:
    x = ensure_array(x0); n = x.size
    H = np.eye(n); hist = History(); g = grad(x)
    for k in range(maxit):
        fval = float(f(x)); ng = float(np.linalg.norm(g))
        hist.append(x=x.copy(), f=fval, grad_norm=ng)
        if ng < tol: break
        d = -H @ g
        t = backtracking_armijo(f, grad, x, d, c1=c1, rho=rho, t0=t0)
        s = t * d; x_new = x + s; g_new = grad(x_new); y = g_new - g
        ys = float(y @ s)
        if ys > 1e-12:
            rho_k = 1.0 / ys; I = np.eye(n)
            H = (I - rho_k*np.outer(s, y)) @ H @ (I - rho_k*np.outer(y, s)) + rho_k*np.outer(s, s)
        x, g = x_new, g_new
    return x, hist
