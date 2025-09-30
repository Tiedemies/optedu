# tests/test_rosenbrock.py
import numpy as np

from optedu.problems.rosenbrock import Rosenbrock  # type: ignore
from optedu.algorithms.gradient_descent import gradient_descent  # type: ignore

def test_rosenbrock_gd_descends_and_reduces_gradnorm():
    prob = Rosenbrock(n=2)  # classic banana
    x0 = np.array([-1.2, 1.0])
    out = gradient_descent(f=prob.f, grad=prob.grad, x0=x0, maxit=2000, tol=1e-6)
    assert out["status"] in {"converged", "maxit"}
    # Function decreases (usually strictly for GD + backtracking)
    fh = np.asarray(out.get("f_hist", []), float)
    if len(fh) > 3:
        assert fh[-1] <= fh[0]
    # Gradient norm should drop substantially
    gnh = np.asarray(out.get("gnorm_hist", []), float)
    if len(gnh) > 3:
        assert gnh[-1] <= 0.1 * gnh[0]
