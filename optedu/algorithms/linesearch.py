import numpy as np
from typing import Callable
from ..utils.types import ensure_array

def backtracking_armijo(f: Callable, grad: Callable, x: np.ndarray, d: np.ndarray,
                        c1: float = 1e-4, rho: float = 0.5, t0: float = 1.0) -> float:
    x = ensure_array(x); d = ensure_array(d)
    t = float(t0)
    fx = float(f(x))
    gTd = float(grad(x).dot(d))
    if gTd >= 0:
        raise AssertionError("Direction must be a descent direction for Armijo backtracking.")
    while True:
        xn = x + t*d
        if f(xn) <= fx + c1*t*gTd:
            return t
        t *= rho

def exact_quadratic_step(Q: np.ndarray, gradx: np.ndarray, d: np.ndarray) -> float:
    num = gradx.dot(d); den = d.dot(Q @ d)
    if den <= 0: return 1.0
    return - num / den

# --- Robust exact numeric line search (shrink-then-expand + golden section) ---

def _golden_section(phi, a, b, c, tol=1e-8, maxit=200):
    """
    Golden-section search on [a,c]. 'b' is a point inside (not necessarily the minimizer).
    Returns t*, phi(t*). Assumes phi is unimodal on [a,c].
    """
    a = float(a); c = float(c)
    invphi  = (np.sqrt(5.0) - 1.0) / 2.0    # 1/phi
    invphi2 = (3.0 - np.sqrt(5.0)) / 2.0    # 1/phi^2

    # interior points
    x1 = a + invphi2 * (c - a)
    x2 = a + invphi  * (c - a)
    f1 = float(phi(x1))
    f2 = float(phi(x2))

    for _ in range(maxit):
        if abs(c - a) <= tol * (abs(a) + abs(c) + 1.0):
            break
        if f1 < f2:
            c, f2 = x2, f2
            x2 = x1
            f2 = f1
            x1 = a + invphi2 * (c - a)
            f1 = float(phi(x1))
        else:
            a, f1 = x1, f1
            x1 = x2
            f1 = f2
            x2 = a + invphi * (c - a)
            f2 = float(phi(x2))

    # pick best among sampled endpoints
    candidates = [(a, float(phi(a))), (x1, f1), (x2, f2), (c, float(phi(c)))]
    t_star, f_star = min(candidates, key=lambda z: z[1])
    return float(t_star), float(f_star)


def exact_line_search(
    f, x, d, *,
    t_init=1.0,           # initial try
    shrink=0.5,           # shrink factor for finding initial decrease
    grow=2.0,             # expansion factor to bracket the minimum
    min_step=1e-12,       # stop shrinking under this
    tol=1e-8,
    maxit=200
):
    """
    Minimize phi(t) = f(x + t d) with t >= 0 using a robust bracket + golden-section.
    1) Shrink t until phi(t) < phi(0). 2) Expand forward to bracket minimum. 3) Golden-section on [a,c].
    Returns t* (float).
    """
    x = ensure_array(x); d = ensure_array(d)

    def phi(t):
        # enforce t >= 0
        t = max(0.0, float(t))
        return float(f(x + t * d))

    fa = phi(0.0)

    # --- Phase 1: find an initial decrease from 0 by shrinking t ---
    t = float(abs(t_init))
    fb = phi(t)
    n_shrink = 0
    while fb >= fa and t > min_step and n_shrink < maxit:
        t *= shrink
        fb = phi(t)
        n_shrink += 1

    if fb >= fa:
        # Could not find a decrease: as a safety, return zero step (caller can fall back to Armijo)
        return 0.0

    # Now we have 0 = a < b = t with phi(b) < phi(a)
    a = 0.0; fa = fa
    b = t;    fb = fb

    # --- Phase 2: expand forward to find c with phi(c) > phi(b) ---
    step = t
    c = b + step
    fc = phi(c)
    n_expand = 0
    while fc < fb and n_expand < maxit:
        a, fa = b, fb
        b, fb = c, fc
        step *= grow
        c = b + step
        fc = phi(c)
        n_expand += 1

    # We have a bracket [a, c] with a < b < c and phi(b) <= phi(a), phi(b) <= phi(c)
    t_star, _ = _golden_section(phi, a, b, c, tol=tol, maxit=maxit)
    return float(t_star)

