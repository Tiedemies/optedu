import numpy as np
from typing import Callable, Tuple, Optional
from ..utils.types import History, ensure_array
from .linesearch import backtracking_armijo, exact_quadratic_step, exact_line_search

def gradient_descent(f: Callable[[np.ndarray], float],
                     grad: Callable[[np.ndarray], np.ndarray],
                     x0, step: str = "backtracking",
                     c1: float = 1e-4, rho: float = 0.5, t0: float = 1.0,
                     maxit: int = 500, tol: float = 1e-8,
                     callback=None, Q_for_exact: Optional[np.ndarray] = None) -> Tuple[np.ndarray, History]:
    x = ensure_array(x0)
    hist = History()
    for k in range(maxit):
        g = grad(x)
        ng = np.linalg.norm(g)
        fval = float(f(x))
        # (optional) store gradient vector if you want to check orthogonality
        # hist.append(x=x.copy(), f=fval, grad_norm=ng, grad=g.copy())
        hist.append(x=x.copy(), f=fval, grad_norm=ng)
        if callback is not None:
            callback(x, fval, k, hist)
        if ng < tol:
            break
        d = -g
        if step == "backtracking":
            t = backtracking_armijo(f, grad, x, d, c1=c1, rho=rho, t0=t0)
        elif step == "constant":
            t = float(t0)
        elif step == "exact_quadratic":
            if Q_for_exact is None:
                raise ValueError("Provide Q_for_exact for exact quadratic step.")
            t = exact_quadratic_step(Q_for_exact, g, d)
        elif step == "exact_numeric":
            t = exact_line_search(f, x, d, t_init=1.0, shrink=0.5, grow=2.0, tol=1e-8, maxit=200)
        else:
            raise ValueError("Unknown step rule.")
        x = x + t * d
    return x, hist
