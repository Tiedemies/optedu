from typing import Any
import numpy as np

ArrayLike = Any

class History(dict):
    def append(self, **kwargs):
        for k, v in kwargs.items():
            self.setdefault(k, []).append(v)

def ensure_array(x) -> np.ndarray:
    return np.asarray(x, dtype=float).copy()
