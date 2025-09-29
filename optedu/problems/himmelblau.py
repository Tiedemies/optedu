import numpy as np

class Himmelblau:
    def f(self, x):
        x = np.asarray(x, dtype=float); x1, x2 = x[0], x[1]
        return (x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2
    def grad(self, x):
        x = np.asarray(x, dtype=float); x1, x2 = x[0], x[1]
        df_dx1 = 4*x1*(x1**2 + x2 - 11) + 2*(x1 + x2**2 - 7)
        df_dx2 = 2*(x1**2 + x2 - 11) + 4*x2*(x1 + x2**2 - 7)
        return np.array([df_dx1, df_dx2], dtype=float)
    def hess(self, x):
        x = np.asarray(x, dtype=float); x1, x2 = x[0], x[1]
        h11 = 12*x1**2 + 4*x2 - 42
        h22 = 12*x2**2 + 4*x1 - 26
        h12 = 4*(x1 + x2)
        return np.array([[h11, h12],[h12, h22]], dtype=float)
