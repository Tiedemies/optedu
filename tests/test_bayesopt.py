# tests/test_bayesopt.py
# Bayesian optimization: the Gaussian-process surrogate, the acquisition functions, and the main loop.
import numpy as np
import pytest

from helpers import Counter, check_result, sphere
from optedu.algorithms.bayesopt import (
    GP, _acq_values, _ensure_bounds, _expected_improvement, _probability_improvement,
    _rbf_kernel, _stable_cholesky, _ucb_score, bayes_optimize,
)

PHI0 = 1.0 / np.sqrt(2.0 * np.pi)       # standard normal density at 0


# ---------------- kernel and GP ----------------

def test_rbf_kernel_values():
    X = np.array([[0.0], [1.0]])
    K = _rbf_kernel(X, X, lengthscale=1.0, variance=2.0)
    # k(x, x') = variance * exp(-|x - x'|^2 / (2 l^2))
    assert np.allclose(K, [[2.0, 2.0 * np.exp(-0.5)], [2.0 * np.exp(-0.5), 2.0]])


def test_rbf_kernel_with_one_lengthscale_per_dimension():
    X = np.array([[0.0, 0.0]]); Y = np.array([[1.0, 1.0]])
    k = _rbf_kernel(X, Y, lengthscale=[1.0, 1e6], variance=1.0, ard=True)
    assert k[0, 0] == pytest.approx(np.exp(-0.5))      # the second coordinate is ignored


def test_stable_cholesky_handles_singular_matrix():
    K = np.ones((3, 3))                                  # rank 1 (e.g. three identical points)
    L = _stable_cholesky(K)
    assert np.allclose(L @ L.T, K, atol=1e-6)


def test_gp_interpolates_its_data():
    X = np.array([[0.0], [1.0], [2.5]]); y = np.array([1.0, -1.0, 0.5])
    gp = GP(lengthscale=1.0, variance=1.0, noise=1e-10)
    gp.fit(X, y)
    mu, var = gp.predict(X)
    assert np.allclose(mu, y, atol=1e-6)                 # mean passes through the observations
    assert np.all(var < 1e-6)                            # and is (almost) certain there


def test_gp_far_from_data_returns_to_the_prior():
    gp = GP(lengthscale=0.5, variance=3.0, noise=1e-8)
    gp.fit(np.array([[0.0]]), np.array([2.0]))
    mu, var = gp.predict(np.array([[100.0]]))
    assert mu[0] == pytest.approx(0.0) and var[0] == pytest.approx(3.0)   # zero mean, prior variance


def test_gp_one_point_posterior_formula():
    # One observation y at x: mu(x*) = k(x*, x) y / (k(x, x) + noise)
    gp = GP(lengthscale=1.0, variance=1.0, noise=0.1)
    gp.fit(np.array([[0.0]]), np.array([2.0]))
    mu, var = gp.predict(np.array([[1.0]]))
    k = np.exp(-0.5)
    assert mu[0] == pytest.approx(k * 2.0 / 1.1)
    assert var[0] == pytest.approx(1.0 - k * k / 1.1)


def test_gp_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        GP().predict(np.zeros((1, 1)))


# ---------------- acquisition functions (minimization) ----------------

def test_expected_improvement_values():
    # mu = f_best, sigma = 1: EI = sigma * phi(0)
    assert _expected_improvement(np.array([0.0]), np.array([1.0]), f_best=0.0)[0] == pytest.approx(PHI0)
    # sigma = 0: EI = max(0, f_best - mu)
    ei = _expected_improvement(np.array([-1.0, 1.0]), np.array([0.0, 0.0]), f_best=0.0)
    assert np.allclose(ei, [1.0, 0.0])


def test_expected_improvement_prefers_low_mean_and_high_uncertainty():
    ei = _expected_improvement(np.array([0.0, -1.0, 0.0]), np.array([1.0, 1.0, 4.0]), f_best=0.0)
    assert np.all(ei >= 0)
    assert ei[1] > ei[0] and ei[2] > ei[0]


def test_probability_of_improvement_values():
    pi = _probability_improvement(np.array([0.0, -10.0, 10.0]), np.array([1.0, 1.0, 1.0]), f_best=0.0)
    assert pi[0] == pytest.approx(0.5)
    assert pi[1] == pytest.approx(1.0) and pi[2] == pytest.approx(0.0, abs=1e-12)


def test_ucb_score():
    # minimization: we maximize -(mu - kappa * sigma)
    assert _ucb_score(np.array([1.0]), np.array([4.0]), kappa=2.0)[0] == pytest.approx(3.0)


def test_unknown_acquisition_raises():
    with pytest.raises(ValueError):
        _acq_values("magic", np.zeros(1), np.ones(1), 0.0, xi=0.0, kappa=2.0)


@pytest.mark.parametrize("bounds", [[1.0, 2.0], [[1.0, 0.0]], [[0.0, 0.0]]])
def test_bad_bounds_raise(bounds):
    with pytest.raises(ValueError):
        _ensure_bounds(bounds)


# ---------------- the BO loop ----------------

def run_bo(seed=0, f=sphere, **kw):
    return bayes_optimize(f, [[-2.0, 2.0], [-2.0, 2.0]], iters=25, n_init=5, cand_points=512,
                          rng=np.random.default_rng(seed), **kw)


@pytest.mark.parametrize("acq, target", [("ei", 0.01), ("pi", 0.01), ("ucb", 0.1)])
def test_bo_on_sphere(acq, target):
    out = run_bo(acq=acq)
    check_result(out)
    assert out["status"] == "maxit" and out["f"] < target


def test_bo_history_and_counts():
    f = Counter(sphere)
    out = run_bo(f=f)
    hist = out["history"]
    assert out["counts"] == {"nit": 25, "nfev": 30} and f.calls == 30
    assert len(hist["x"]) == len(hist["f"]) == len(hist["best_f"]) == 30
    assert len(hist["acq_best"]) == 25                    # one acquisition maximum per BO iteration
    assert np.allclose(hist["best_f"], np.minimum.accumulate(hist["f"]))
    assert out["f"] == min(hist["f"])
    for x in hist["x"]:
        assert np.all(np.abs(x) <= 2.0)


def test_bo_same_seed_same_result():
    a, b = run_bo(seed=3), run_bo(seed=3)
    assert np.array_equal(a["x"], b["x"])
