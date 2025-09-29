import numpy as np
from typing import Callable, Tuple
from ..utils.types import History, ensure_array

def hooke_jeeves(f: Callable[[np.ndarray], float],
                 x0, step=0.5, shrink=0.5, tol=1e-6, maxit=1000) -> Tuple[np.ndarray, History]:
    x = ensure_array(x0); n = x.size; hist = History(); k = 0
    while step > tol and k < maxit:
        f0 = float(f(x)); hist.append(x=x.copy(), f=f0)
        x_new = x.copy()
        for i in range(n):
            for delta in [step, -step]:
                trial = x_new.copy(); trial[i] += delta
                if f(trial) < f(x_new): x_new = trial
        if f(x_new) < f0:
            d = x_new - x; x = x_new + d
        else:
            step *= shrink
        k += 1
    return x, hist
