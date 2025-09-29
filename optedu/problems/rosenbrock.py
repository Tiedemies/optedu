import numpy as np

class Rosenbrock:
    def __init__(self, a=1.0, b=100.0, n=2):
        self.a = float(a); self.b = float(b); self.n = int(n)
        assert n >= 2
    def f(self, x):
        x = np.asarray(x, dtype=float); s = 0.0
        for i in range(self.n-1):
            s += self.b*(x[i+1]-x[i]**2)**2 + (self.a - x[i])**2
        return s
    def grad(self, x):
        x = np.asarray(x, dtype=float); g = np.zeros_like(x)
        for i in range(self.n-1):
            g[i] += -4*self.b*(x[i+1]-x[i]**2)*x[i] + 2*(x[i]-self.a)
            g[i+1] += 2*self.b*(x[i+1]-x[i]**2)
        return g
    def hess(self, x):
        x = np.asarray(x, dtype=float); n = x.size
        H = np.zeros((n,n))
        for i in range(n-1):
            H[i,i] += 12*self.b*x[i]**2 - 4*self.b*x[i+1] + 2
            H[i,i+1] += -4*self.b*x[i]
            H[i+1,i] += -4*self.b*x[i]
            H[i+1,i+1] += 2*self.b
        return H
