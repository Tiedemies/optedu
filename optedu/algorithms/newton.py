import numpy as np
from typing import Callable, Tuple
from ..utils.types import History, ensure_array
from .linesearch import backtracking_armijo

def newton(f: Callable[[np.ndarray], float],
           grad: Callable[[np.ndarray], np.ndarray],
           hess: Callable[[np.ndarray], np.ndarray],
           x0, damped: bool = True, c1: float = 1e-4, rho: float = 0.5,
           maxit: int = 200, tol: float = 1e-8, t0: float = 1.0) -> Tuple[np.ndarray, History]:
    x = ensure_array(x0)
    hist = History()
    for k in range(maxit):
        g = grad(x); H = hess(x)
        try:
            w, V = np.linalg.eigh(H)
            w = np.maximum(w, 1e-10)
            Hpd = (V * w) @ V.T
            d = -np.linalg.solve(Hpd, g)
        except np.linalg.LinAlgError:
            d = -g
        fval = float(f(x)); ng = float(np.linalg.norm(g))
        hist.append(x=x.copy(), f=fval, grad_norm=ng)
        if ng < tol: break
        t = backtracking_armijo(f, grad, x, d, c1=c1, rho=rho, t0=t0) if damped else 1.0
        x = x + t*d
    return x, hist
