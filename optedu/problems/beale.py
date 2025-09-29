import numpy as np

class Beale:
    def f(self, x):
        x1, x2 = float(x[0]), float(x[1])
        return (1.5 - x1 + x1*x2)**2 + (2.25 - x1 + x1*x2**2)**2 + (2.625 - x1 + x1*x2**3)**2
    def grad(self, x):
        x1, x2 = float(x[0]), float(x[1])
        t1 = 1.5 - x1 + x1*x2
        t2 = 2.25 - x1 + x1*x2**2
        t3 = 2.625 - x1 + x1*x2**3
        df_dx1 = 2*t1*(-1 + x2) + 2*t2*(-1 + x2**2) + 2*t3*(-1 + x2**3)
        df_dx2 = 2*t1*(x1) + 2*t2*(2*x1*x2) + 2*t3*(3*x1*x2**2)
        return np.array([df_dx1, df_dx2], dtype=float)
    def hess(self, x):
        # finite-diff Hessian for pedagogy
        x = np.asarray(x, dtype=float); eps = 1e-6
        e1 = np.array([1,0], float); e2 = np.array([0,1], float)
        g = self.grad(x)
        H = np.zeros((2,2))
        H[:,0] = (self.grad(x+eps*e1)-g)/eps
        H[:,1] = (self.grad(x+eps*e2)-g)/eps
        return H
