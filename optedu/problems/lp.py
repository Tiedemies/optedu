# optedu/problems/lp.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

_ALLOWED_SENSES = {"le", "eq", "ge"}

@dataclass
class LP:
    """
    Lightweight LP container for the JSON template:

      min or max  c^T x
      s.t.        A_i x (<=, =, >=) b_i,  for i=1..m
                  x >= 0  (after standardization)

    Fields
    ------
    A : (m,n) array-like
    b : (m,)   array-like
    c : (n,)   array-like
    sense  : "min" or "max"   (global objective sense)
    senses : Optional[List[str]] per-row in {"le","eq","ge"} (default: all "eq")
    """
    A: np.ndarray
    b: np.ndarray
    c: np.ndarray
    sense: str = "min"
    senses: Optional[List[str]] = None

    def __post_init__(self):
        # Coerce to numpy arrays
        self.A = np.asarray(self.A, dtype=float)
        self.b = np.asarray(self.b, dtype=float)
        self.c = np.asarray(self.c, dtype=float)

        if self.A.ndim != 2:
            raise ValueError("A must be 2D.")
        m, n = self.A.shape
        if self.b.shape != (m,):
            raise ValueError(f"b must have shape ({m},), got {self.b.shape}.")
        if self.c.shape != (n,):
            raise ValueError(f"c must have shape ({n},), got {self.c.shape}.")

        # Normalize objective sense
        self.sense = str(self.sense).lower()
        if self.sense not in {"min", "max"}:
            raise ValueError("sense must be 'min' or 'max'.")

        # Normalize per-row senses
        if self.senses is None:
            self.senses = ["eq"] * m
        else:
            if len(self.senses) != m:
                raise ValueError(f"senses must have length m={m}.")
            self.senses = [str(s).lower() for s in self.senses]
            bad = [s for s in self.senses if s not in _ALLOWED_SENSES]
            if bad:
                raise ValueError(f"Invalid entries in senses: {bad}; allowed: {_ALLOWED_SENSES}.")

    def __repr__(self):
        return f"LP(m={self.A.shape[0]}, n={self.A.shape[1]}, sense='{self.sense}', senses={self.senses})"
