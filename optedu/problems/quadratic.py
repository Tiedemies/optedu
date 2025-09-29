import numpy as np

class Quadratic:
    """f(x) = 0.5 x^T Q x - c^T x (convex if Q PSD)."""
    def __init__(self, Q=None, c=None):
        if Q is None: Q = np.array([[2.0, 0.0],[0.0, 10.0]])
        if c is None: c = np.array([-2.0, -8.0])
        self.Q = Q.astype(float); self.c = c.astype(float)
    def f(self, x):
        x = np.asarray(x, dtype=float)
        return 0.5 * x.dot(self.Q @ x) - self.c.dot(x)
    def grad(self, x):
        x = np.asarray(x, dtype=float)
        return self.Q @ x - self.c
    def hess(self, x):
        return self.Q
