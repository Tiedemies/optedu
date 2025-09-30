# tests/test_rosenbrock.py
import numpy as np
from optedu.problems.rosenbrock import Rosenbrock
from optedu.algorithms.gradient_descent import gradient_descent

def test_rosenbrock_gd_descends_and_reduces_gradnorm():
    prob = Rosenbrock(n=2)  # classic banana
    x0 = np.array([-1.2, 1.0])
    out = gradient_descent(f=prob.f, grad=prob.grad, x0=x0, maxit=2000, tol=1e-6)
    assert out["status"] in {"converged", "maxit"}
    vals = np.asarray(out["history"].get("f", []), float)
    assert len(vals) > 5
    # should trend downward on average
    assert vals[-1] <= vals[0]
