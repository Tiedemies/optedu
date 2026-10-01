# tests/test_problems.py
# The test problems: hand-written derivatives must agree with finite differences,
# and the advertised minimizers must really be minimizers.
import numpy as np
import pytest

from helpers import numerical_gradient
from optedu.problems.beale import Beale
from optedu.problems.himmelblau import Himmelblau
from optedu.problems.lp import LP
from optedu.problems.quadratic import Quadratic
from optedu.problems.rosenbrock import Rosenbrock

PROBLEMS = [Rosenbrock(n=2), Rosenbrock(n=4), Rosenbrock(a=2.0, b=10.0, n=2), Himmelblau(), Beale(), Quadratic()]
IDS = ["rosenbrock2", "rosenbrock4", "rosenbrock_a2", "himmelblau", "beale", "quadratic"]


def random_points(prob, k=5):
    rng = np.random.default_rng(0)
    n = prob.x_star.size
    return [rng.uniform(-1.5, 1.5, size=n) for _ in range(k)]


@pytest.mark.parametrize("prob", PROBLEMS, ids=IDS)
def test_gradient_matches_finite_differences(prob):
    for x in random_points(prob):
        assert np.allclose(prob.grad(x), numerical_gradient(prob.f, x), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("prob", PROBLEMS, ids=IDS)
def test_hessian_matches_finite_differences_of_gradient(prob):
    for x in random_points(prob):
        H_num = np.column_stack([numerical_gradient(lambda z: prob.grad(z)[i], x) for i in range(x.size)]).T
        assert np.allclose(prob.hess(x), H_num, rtol=1e-4, atol=1e-3)
        assert np.allclose(prob.hess(x), prob.hess(x).T)          # a Hessian is symmetric


@pytest.mark.parametrize("prob", PROBLEMS, ids=IDS)
def test_known_minimizer(prob):
    x = prob.x_star
    assert np.isclose(prob.f(x), prob.f_star)
    assert np.linalg.norm(prob.grad(x)) < 1e-10                  # first-order condition
    assert np.all(np.linalg.eigvalsh(prob.hess(x)) > -1e-6)      # second-order condition (PSD)


def test_himmelblau_has_four_global_minima():
    prob = Himmelblau()
    for m in prob.all_minima:
        assert prob.f(np.array(m)) < 1e-9


def test_rosenbrock_without_closed_form_minimizer():
    prob = Rosenbrock(a=2.0, n=3)
    assert prob.x_star is None and prob.f_star is None


def test_rosenbrock_dimension_comes_from_x():
    assert Rosenbrock(n=2).f([1.0, 1.0, 1.0, 1.0]) == 0.0


def test_quadratic_accepts_lists_from_json():
    q = Quadratic(Q=[[2, 0], [0, 10]], c=[-2, -8])
    assert np.allclose(q.x_star, [-1.0, -0.8])


# ---------------- LP container ----------------

def test_lp_defaults_and_normalization():
    lp = LP(A=[[1, 2]], b=[3], c=[1, 1], sense="MAX")
    assert lp.sense == "max"
    assert lp.senses == ["eq"]
    assert lp.A.dtype == float


@pytest.mark.parametrize("kwargs, message", [
    (dict(A=[1, 2], b=[1], c=[1, 1]), "2D"),
    (dict(A=[[1, 2]], b=[1, 2], c=[1, 1]), "b must have shape"),
    (dict(A=[[1, 2]], b=[1], c=[1]), "c must have shape"),
    (dict(A=[[1, 2]], b=[1], c=[1, 1], sense="minimize"), "sense"),
    (dict(A=[[1, 2]], b=[1], c=[1, 1], senses=["le", "ge"]), "length"),
    (dict(A=[[1, 2]], b=[1], c=[1, 1], senses=["<="]), "Invalid"),
])
def test_lp_rejects_bad_input(kwargs, message):
    with pytest.raises(ValueError, match=message):
        LP(**kwargs)


def test_lp_repr():
    assert repr(LP(A=[[1, 2]], b=[3], c=[1, 1], senses=["le"])) == "LP(m=1, n=2, sense='min', senses=['le'])"
