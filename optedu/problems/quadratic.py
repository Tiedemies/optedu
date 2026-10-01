import numpy as np

class Quadratic:
    """f(x) = 0.5 x^T Q x - c^T x (convex if Q PSD)."""
    def __init__(self, Q=None, c=None):
        if Q is None: Q = np.array([[2.0, 0.0],[0.0, 10.0]])
        if c is None: c = np.array([-2.0, -8.0])
        self.Q = np.asarray(Q, dtype=float); self.c = np.asarray(c, dtype=float)   # lists from JSON are fine
        # Known minimizer (if Q is positive definite): grad f = Q x - c = 0  ->  x* = Q^{-1} c
        self.x_star = np.linalg.solve(self.Q, self.c)
        self.f_star = float(self.f(self.x_star))
    def f(self, x):
        x = np.asarray(x, dtype=float)
        return 0.5 * x.dot(self.Q @ x) - self.c.dot(x)
    def grad(self, x):
        x = np.asarray(x, dtype=float)
        return self.Q @ x - self.c
    def hess(self, x):
        return self.Q
