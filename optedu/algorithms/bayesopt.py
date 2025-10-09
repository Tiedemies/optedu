from __future__ import annotations
import math
from typing import Callable, Dict, Any, Optional, Tuple, List
import numpy as np

# ---- Typed result compatibility (no hard import on your types) ----
from ..utils.types import History, AlgoResult  # dict-like recorder with .append(...)

# ===========================
#   Gaussian Process (RBF)
# ===========================
def _square_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # ||a-b||^2 for all rows
    # a: (n, d), b: (m, d)
    a2 = np.sum(a*a, axis=1, keepdims=True)
    b2 = np.sum(b*b, axis=1)[None, :]
    return a2 + b2 - 2.0*np.dot(a, b.T)

def _rbf_kernel(X: np.ndarray, Y: np.ndarray, lengthscale: float, variance: float, ard: bool=False) -> np.ndarray:
    if ard:
        # lengthscale is a vector of shape (d,)
        ls = np.asarray(lengthscale, dtype=float).reshape(1, -1)
        Xs = X / ls
        Ys = Y / ls
        dist2 = _square_dist(Xs, Ys)
    else:
        dist2 = _square_dist(X / lengthscale, Y / lengthscale)
    return variance * np.exp(-0.5 * dist2)

def _stable_cholesky(K: np.ndarray, jitter: float=1e-10, max_tries: int=5) -> np.ndarray:
    # Try increasing jitter if needed
    I = np.eye(K.shape[0], dtype=K.dtype)
    jj = jitter
    for _ in range(max_tries):
        try:
            L = np.linalg.cholesky(K + jj * I)
            return L
        except np.linalg.LinAlgError:
            jj *= 10.0
    # Final attempt raises
    return np.linalg.cholesky(K + jj * I)

class GP:
    """
    Simple zero-mean GP with RBF kernel and Gaussian noise.
    Minimization convention: we model f directly.
    """
    def __init__(self,
                 lengthscale: float | np.ndarray = 0.5,
                 variance: float = 1.0,
                 noise: float = 1e-6,
                 ard: bool = False):
        self.lengthscale = lengthscale
        self.variance = variance
        self.noise = noise
        self.ard = ard
        self._fitted = False
        self.X = None
        self.y = None
        self.L = None       # Cholesky of K + noise*I
        self.alpha = None   # (K + noise I)^(-1) y

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = np.ascontiguousarray(X, dtype=float)
        self.y = np.asarray(y, dtype=float).reshape(-1, 1)
        K = _rbf_kernel(self.X, self.X, self.lengthscale, self.variance, self.ard)
        K[np.diag_indices_from(K)] += self.noise
        L = _stable_cholesky(K)
        # Solve for alpha via two triangular solves
        # L L^T alpha = y -> first solve L v = y ; then L^T alpha = v
        v = np.linalg.solve(L, self.y)
        alpha = np.linalg.solve(L.T, v)
        self.L = L
        self.alpha = alpha
        self._fitted = True

    def predict(self, Xstar: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")
        Xs = np.ascontiguousarray(Xstar, dtype=float)
        Kxs = _rbf_kernel(Xs, self.X, self.lengthscale, self.variance, self.ard)  # (m, n)
        # mean = Kxs @ alpha
        mu = Kxs @ self.alpha
        # var = k(x*,x*) - v^T v with v = solve(L, Kxs^T)
        v = np.linalg.solve(self.L, Kxs.T)
        kxx = self.variance * np.ones((Xs.shape[0], 1), dtype=float)  # RBF k(x,x)=variance
        var = kxx - np.sum(v*v, axis=0, keepdims=True).T
        # Clamp numerical negatives
        var = np.maximum(var, 1e-18)
        return mu.ravel(), var.ravel()

# ===========================
#   Acquisition functions
# ===========================
def _phi(x):
    return (1.0 / math.sqrt(2.0*math.pi)) * np.exp(-0.5 * x*x)

def _Phi(x):
    # Standard normal CDF (erf-based)
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def _expected_improvement(mu: np.ndarray, var: np.ndarray, f_best: float, xi: float=0.0) -> np.ndarray:
    # For minimization: improvement = f_best - mu - xi
    sigma = np.sqrt(var, dtype=float)
    imp = (f_best - mu - xi)
    with np.errstate(divide='ignore', invalid='ignore'):
        Z = imp / sigma
        # Where sigma == 0, EI = max(0, imp)
        ei = np.where(sigma > 0,
                      imp * np.vectorize(_Phi)(Z) + sigma * np.vectorize(_phi)(Z),
                      np.maximum(0.0, imp))
    return np.maximum(0.0, ei)

def _probability_improvement(mu: np.ndarray, var: np.ndarray, f_best: float, xi: float=0.0) -> np.ndarray:
    sigma = np.sqrt(var, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        Z = (f_best - mu - xi) / sigma
        pi = np.where(sigma > 0, np.vectorize(_Phi)(Z), (mu < f_best - xi).astype(float))
    return pi

def _ucb_score(mu: np.ndarray, var: np.ndarray, kappa: float=2.0) -> np.ndarray:
    # Minimization UCB: choose x that minimizes mu - kappa*sigma
    sigma = np.sqrt(var, dtype=float)
    return -(mu - kappa * sigma)  # negate so we can argmax

def _acq_values(acq: str, mu: np.ndarray, var: np.ndarray, f_best: float,
                xi: float, kappa: float) -> np.ndarray:
    if acq == "ei":
        return _expected_improvement(mu, var, f_best, xi=xi)
    elif acq == "pi":
        return _probability_improvement(mu, var, f_best, xi=xi)
    elif acq == "ucb":
        return _ucb_score(mu, var, kappa=kappa)
    else:
        raise ValueError(f"Unknown acquisition '{acq}'")

# ===========================
#   Utilities
# ===========================
def _sample_uniform(bounds: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    lo = bounds[:, 0]
    hi = bounds[:, 1]
    return rng.random((n, bounds.shape[0])) * (hi - lo) + lo

def _ensure_bounds(bounds) -> np.ndarray:
    B = np.asarray(bounds, dtype=float)
    if B.ndim != 2 or B.shape[1] != 2:
        raise ValueError("bounds must be array-like of shape (d, 2)")
    if np.any(B[:, 0] >= B[:, 1]):
        raise ValueError("Each bound must satisfy lower < upper.")
    return B

# ===========================
#   Main BO routine
# ===========================
def bayes_optimize(
    f: Callable[[np.ndarray], float],
    bounds: List[List[float]] | np.ndarray,
    *,
    iters: int = 40,
    n_init: int = 5,
    cand_points: int = 2048,
    acq: str = "ei",          # "ei" | "pi" | "ucb"
    xi: float = 0.0,          # exploration for EI/PI
    kappa: float = 2.0,       # exploration for UCB
    lengthscale: float | np.ndarray = 0.5,
    variance: float = 1.0,
    noise: float = 1e-6,      # observational noise; keep tiny for deterministic f
    ard: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> AlgoResult:
    """
    Pure NumPy Bayesian Optimization (minimization) with a GP surrogate (RBF).
    - bounds: (d,2) box
    - Starts with n_init random evaluations, then iteratively:
      fit GP -> maximize acquisition over a uniform candidate set -> sample f -> update.

    Returns:
        dict with keys: status, x, f, history, counts, message, extra
    """
    rng = rng or np.random.default_rng()
    B = _ensure_bounds(bounds)
    d = B.shape[0]

    # History containers (compatible with your visuals)
    hist: History = {
        "x": [],               # evaluated points
        "f": [],               # observed values
        "best_f": [],          # best-so-far value
        "best_x": [],          # best-so-far point
        "acq_best": [],        # acquisition value at the chosen next point
        "acq_name": acq,
        "meta": {
            "kernel": "RBF-ARD" if ard else "RBF",
            "lengthscale": np.asarray(lengthscale).tolist() if np.ndim(lengthscale) else float(lengthscale),
            "variance": float(variance),
            "noise": float(noise),
            "xi": float(xi),
            "kappa": float(kappa),
        },
    }

    # Initial design
    X = _sample_uniform(B, n_init, rng)
    y = np.zeros(n_init, dtype=float)
    for i in range(n_init):
        yi = float(f(X[i]))
        y[i] = yi
        hist["x"].append(X[i].tolist())
        hist["f"].append(yi)
        if i == 0 or yi < hist["best_f"][-1]:
            hist["best_f"].append(yi)
            hist["best_x"].append(X[i].tolist())
        else:
            hist["best_f"].append(hist["best_f"][-1])
            hist["best_x"].append(hist["best_x"][-1])

    # Main loop
    gp = GP(lengthscale=lengthscale, variance=variance, noise=noise, ard=ard)

    evals = n_init
    for t in range(iters):
        # Fit GP on current data
        gp.fit(X, y)

        # Candidate set (uniform; robust and dependency-free)
        C = _sample_uniform(B, cand_points, rng)

        # Predict mu, var; score acquisition
        mu, var = gp.predict(C)
        f_best = float(np.min(y))
        A = _acq_values(acq, mu, var, f_best=f_best, xi=xi, kappa=kappa)

        # Argmax acquisition (ties broken arbitrarily)
        j = int(np.argmax(A))
        x_next = C[j]
        a_best = float(A[j])

        # Evaluate objective
        y_next = float(f(x_next))
        # Update data
        X = np.vstack([X, x_next[None, :]])
        y = np.concatenate([y, [y_next]])
        evals += 1

        # Record history
        hist["x"].append(x_next.tolist())
        hist["f"].append(y_next)
        hist["acq_best"].append(a_best)
        if y_next < hist["best_f"][-1]:
            hist["best_f"].append(y_next)
            hist["best_x"].append(x_next.tolist())
        else:
            hist["best_f"].append(hist["best_f"][-1])
            hist["best_x"].append(hist["best_x"][-1])

    # Final result
    idx = int(np.argmin(y))
    x_star = X[idx]
    f_star = float(y[idx])

    result = AlgoResult(
        status = "maxit",
        x = x_star,
        f = f_star,
        history = hist,
        counts = {"evals": int(evals), "iters": int(iters)},
        message = f"Bayesian optimization finished with {evals} evaluations.",
        extra = {
            "final_lengthscale": gp.lengthscale if isinstance(gp.lengthscale, float) else np.asarray(gp.lengthscale).tolist(),
            "final_variance": gp.variance,
            "final_noise": gp.noise
        }
    )
    return result
