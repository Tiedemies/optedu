from .types import History, ensure_array
import numpy as np

def default_callback(xk: np.ndarray, fk: float, k: int, hist: History) -> None:
    hist.append(x=xk.copy(), f=fk, k=k)

def wrap_objective(f):
    count = {'n': 0, 'last': None}
    def fw(x):
        val = f(x)
        count['n'] += 1
        count['last'] = float(val)
        return val
    fw.nfev = lambda: count['n']
    fw.last = lambda: count['last']
    return fw
