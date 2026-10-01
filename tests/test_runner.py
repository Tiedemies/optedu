# tests/test_runner.py
# The JSON runner optimize.py: loading targets, checking configs, assembling the algorithm call,
# and running end to end from the command line.
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import optimize
from optedu.algorithms.genetic import genetic_minimize
from optedu.algorithms.gradient_descent import gradient_descent
from optedu.algorithms.lp_two_phase import solve_two_phase_generic
from optedu.algorithms.sa import simulated_annealing
from optedu.problems.lp import LP
from optedu.problems.rosenbrock import Rosenbrock

ROOT = Path(__file__).resolve().parents[1]


# ---------------- loading ----------------

def test_load_symbol():
    assert optimize.load_symbol("optedu.problems.rosenbrock:Rosenbrock") is Rosenbrock
    with pytest.raises(ValueError):
        optimize.load_symbol("optedu.problems.rosenbrock.Rosenbrock")      # missing ':'
    with pytest.raises(ImportError):
        optimize.load_symbol("optedu.problems.rosenbrock:Banana")


def test_build_problem():
    obj, f, grad, hess = optimize.build_problem({"target": "optedu.problems.rosenbrock:Rosenbrock",
                                                 "kwargs": {"n": 3}})
    assert obj.n == 3 and f == obj.f and grad == obj.grad and hess == obj.hess
    obj, f, grad, hess = optimize.build_problem({"target": "optedu.problems.lp:LP",
                                                 "kwargs": {"A": [[1, 1]], "b": [1], "c": [1, 1]}})
    assert isinstance(obj, LP) and f is None and grad is None and hess is None


def test_build_algorithm_rejects_unknown_parameter():
    with pytest.raises(SystemExit, match="initial_step"):              # the message lists accepted names
        optimize.build_algorithm({"target": "optedu.algorithms.nelder_mead:nelder_mead",
                                  "kwargs": {"step": 0.5}})


def test_check_config():
    optimize.check_config({"title": "t", "problem": {}, "algorithm": {}, "x0": [], "visual": {}})
    with pytest.raises(SystemExit, match="visual"):
        optimize.check_config({"problem": {}, "algorithm": {}, "xlims": [0, 1]})


# ---------------- assembling the call ----------------

def test_assemble_call_for_smooth_problem():
    prob = Rosenbrock()
    pos, kw = optimize.assemble_call(gradient_descent, {"maxit": 5}, prob, prob.f, prob.grad, prob.hess, [0, 0])
    assert pos == []
    assert kw["f"] == prob.f and kw["grad"] == prob.grad and kw["x0"] == [0, 0] and kw["maxit"] == 5
    assert "hess" not in kw                                    # gradient descent does not ask for it


def test_assemble_call_user_kwargs_win():
    prob = Rosenbrock()
    _, kw = optimize.assemble_call(simulated_annealing, {"x0": [9, 9]}, prob, prob.f, None, None, [0, 0])
    assert kw["x0"] == [9, 9]


def test_assemble_call_for_lp():
    lp = LP(A=[[1, 1]], b=[4], c=[1, 2], sense="max", senses=["le"])
    _, kw = optimize.assemble_call(solve_two_phase_generic, {}, lp, None, None, None, None)
    assert kw["senses"] == ["le"] and kw["objective"] == "max"
    assert np.array_equal(kw["A"], lp.A)
    assert solve_two_phase_generic(**kw)["f"] == pytest.approx(8.0)   # max x1 + 2x2, x1 + x2 <= 4


def test_assemble_call_supplies_only_what_is_declared():
    prob = Rosenbrock()
    _, kw = optimize.assemble_call(genetic_minimize, {"bounds": [[-1, 1], [-1, 1]]}, prob, prob.f, prob.grad, None, [0, 0])
    assert set(kw) == {"f", "bounds"}


# ---------------- end to end ----------------

def run_cli(*args):
    env = dict(os.environ, MPLBACKEND="Agg")
    proc = subprocess.run([sys.executable, str(ROOT / "optimize.py"), *map(str, args)],
                          cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
    return proc


def write_config(tmp_path, cfg):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    return path


def test_cli_save_writes_result_json(tmp_path):
    out_dir = tmp_path / "out"
    proc = run_cli(ROOT / "configs" / "himmelblau_nm.json", "--save", out_dir)
    assert proc.returncode == 0, proc.stderr
    result = json.loads((out_dir / "result.json").read_text())
    assert result["status"] == "converged"
    assert np.allclose(result["x_star"], [3.0, 2.0], atol=1e-6)
    assert set(result) >= {"status", "x_star", "f_star", "iterations", "counts", "message"}


def test_cli_prints_ray_for_unbounded_lp():
    proc = run_cli(ROOT / "configs" / "lp_mixed_ineq.json")
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout.split("Result:")[1])
    assert result["status"] == "unbounded" and len(result["direction"]) == 4


def test_cli_seed_makes_random_runs_reproducible(tmp_path):
    cfg = {"problem": {"target": "optedu.problems.rosenbrock:Rosenbrock"},
           "algorithm": {"target": "optedu.algorithms.genetic:genetic_minimize",
                         "kwargs": {"bounds": [[-2, 2], [-2, 2]], "generations": 10}}}
    path = write_config(tmp_path, cfg)
    a, b, c = (run_cli(path, "--seed", s).stdout for s in (1, 1, 2))
    assert a == b and a != c


def test_cli_visualize_saves_figures(tmp_path):
    out_dir = tmp_path / "figs"
    proc = run_cli(ROOT / "configs" / "rosenbrock_gd.json", "-v", "--save", out_dir)
    assert proc.returncode == 0, proc.stderr
    assert (out_dir / "contour_path.png").exists() and (out_dir / "values.png").exists()


def test_cli_visualize_high_dimensional(tmp_path):
    cfg = {"problem": {"target": "optedu.problems.rosenbrock:Rosenbrock", "kwargs": {"n": 4}},
           "algorithm": {"target": "optedu.algorithms.bfgs:bfgs"},
           "x0": [-1.2, 1.0, -1.2, 1.0]}
    out_dir = tmp_path / "figs"
    proc = run_cli(write_config(tmp_path, cfg), "-v", "--save", out_dir)
    assert proc.returncode == 0, proc.stderr
    assert (out_dir / "trajectory_pca.png").exists()


def test_cli_bad_config_gives_message(tmp_path):
    cfg = {"problem": {"target": "optedu.problems.rosenbrock:Rosenbrock"},
           "algorithm": {"target": "optedu.algorithms.gradient_descent:gradient_descent"},
           "x0": [0, 0], "levels": 10}
    proc = run_cli(write_config(tmp_path, cfg))
    assert proc.returncode != 0 and "Unknown config key" in proc.stderr
