# optedu

Teaching-first optimization library for MSc/PhD coursework and live demos. It provides **unified visuals**, a **JSON runner** (`optimize.py`), and implementations that **mirror the course material** (e.g., page-43 simplex, §3.3.2 two-phase).

> Design goals: reproducible lecture demos, compact configs, clean APIs, and pedagogical comments that follow the lecture notes.

## Installation

You need Python 3.9 or newer. If you have never used a terminal for Python before, follow the steps exactly:

```bash
# 1. get the code
git clone https://github.com/Tiedemies/optedu.git
cd optedu

# 2. create and activate a virtual environment (keeps this course's packages separate)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. install optedu (editable, so your changes to the code take effect immediately)
pip install -e .

# optional: test tools
pip install -e ".[dev]"
```

Dependencies are lightweight: NumPy and Matplotlib.

## Quick start

### Rosenbrock + gradient descent (interactive, pan/zoom)

```bash
python optimize.py configs/rosenbrock_gd.json -v
```

### LP with mixed inequalities (two-phase + page-43 simplex)

```bash
python optimize.py configs/lp_mixed_ineq.json
```

* The runner prints `status`: `converged | maxit | unbounded | infeasible | failed`.
* For an **unbounded** LP it also prints a **recession direction** `direction` = d: moving from `x_star` along d stays feasible and makes the objective go to −∞ (for min).

## Algorithms

| Algorithm | Import target | Lecture material | Example config |
| --- | --- | --- | --- |
| Gradient descent | `optedu.algorithms.gradient_descent:gradient_descent` | Chapter 4 schema `[S4]` | `configs/rosenbrock_gd.json` |
| Newton (plain, damped, modified) | `optedu.algorithms.newton:newton` | Chapter 4 schema `[S4]` | — |
| BFGS | `optedu.algorithms.bfgs:bfgs` | Chapter 4 schema `[S4]` | — |
| Nelder–Mead | `optedu.algorithms.nelder_mead:nelder_mead` | §6.1.1 | `configs/himmelblau_nm.json` |
| Hooke–Jeeves | `optedu.algorithms.hooke_jeeves:hooke_jeeves` | §6.1.2 | — |
| Genetic algorithm | `optedu.algorithms.genetic:genetic_minimize` | §6.2.1 | — |
| Particle swarm | `optedu.algorithms.pso:pso_minimize` | — | — |
| Simulated annealing | `optedu.algorithms.sa:simulated_annealing` | — | — |
| Bayesian optimization | `optedu.algorithms.bayesopt:bayes_optimize` | — | `configs/rosenbrock_bayes.json` |
| Primal simplex (standard form) | `optedu.algorithms.lp_simplex:simplex_standard` | page 43 | — |
| Two-phase simplex | `optedu.algorithms.lp_two_phase:solve_two_phase` (standard form) or `:solve_two_phase_generic` (≤ / = / ≥ rows, bounds) | §3.3.2 | `configs/lp_mixed_ineq.json` |
| Dual simplex | `optedu.algorithms.dual_simplex:dual_simplex_standard` | page 71 | `configs/lp_dual.json` |

Every algorithm returns the same kind of result: a dict with `status`, `x`, `f`, `history` and `counts` (see `optedu/utils/types.py`). In `history`, entry 0 is the starting point and `x` and `f` have the same length.

### Parameter glossary

Parameter names follow the symbols of the lecture notes. Some symbols are used by more than one method with different meanings:

| Parameter | Meaning | Used in |
| --- | --- | --- |
| `maxit` | maximum number of iterations | GD, Newton, BFGS, Nelder–Mead, Hooke–Jeeves, LP solvers |
| `iters` | fixed number of iterations (budget) | PSO, SA, Bayesian optimization |
| `generations` | number of generations | GA |
| `tol` | stopping tolerance (e.g. on ‖∇f‖, simplex size, step length) | most methods |
| `step` | line search: `"exact"` or `"armijo"` | GD, BFGS |
| `c1` | Armijo sufficient-decrease constant | GD, Newton, BFGS |
| `c1`, `c2` | cognitive and social weights | PSO |
| `rho` | Armijo backtracking factor (t ← ρ t) | GD, Newton, BFGS |
| `rho` | contraction coefficient | Nelder–Mead |
| `t0` | first trial step of Armijo backtracking | GD, Newton, BFGS |
| `alpha`, `gamma`, `sigma` | reflection, expansion, shrink coefficients | Nelder–Mead |
| `initial_step` | edge length of the initial simplex | Nelder–Mead |
| `alpha` | cooling factor, T_k = T0 · alpha^k | SA |
| `T0` | initial temperature | SA |
| `delta0`, `theta` | initial step length, step reduction factor | Hooke–Jeeves |
| `w` | inertia weight | PSO |
| `xi`, `kappa` | exploration parameters for EI/PI and UCB | Bayesian optimization |
| `basis` | starting basis (column indices) | LP solvers |

## Configuration format

All experiments use the same JSON template:

```jsonc
{
  "title": "My experiment",                          // optional; default plot title
  "problem": { "target": "module.path:ClassOrFactory", "kwargs": { "param": 123 } },
  "algorithm": { "target": "module.path:function", "kwargs": { "tol": 1e-6 } },
  "x0": [ ... ],                                     // starting point (nonlinear problems only)
  "visual": {                                        // plot settings (used with -v)
    "interactive": true,                             // 2-D problems: live contour that recomputes on zoom
    "xlims": [-2, 2], "ylims": [-1, 3],              // limits of the plot box
    "levels": 50, "density": 320,                    // number of contour levels, grid resolution
    "style": { "axes.grid": true }                   // Matplotlib rcParams
  }
}
```

(JSON itself does not allow `//` comments; they are only explanations here.)

* `problem.target` and `algorithm.target` are **import strings** of the form `"package.module:Symbol"`.
* `optimize.py` inspects the **algorithm's signature** and supplies exactly the arguments it declares:
  * nonlinear problems may receive `f`, `grad`, `hess`, `x0`;
  * LP algorithms receive `A`, `b`, `c` (and `senses`, `sense` if they ask for them).
* Unknown top-level keys or unknown algorithm parameters stop the runner with a message listing what is allowed.
* Plots are shown only when you pass **`-v`**. `"interactive": true` then chooses the live (pan/zoom) contour instead of a static one.
* `--save DIR` writes `result.json` (and figures, with `-v`) to `DIR`; `--seed N` makes randomized algorithms reproducible.

## How to define a problem (for configs & live demos)

This project uses one simple pattern for **all** problems:

* A **problem** is constructed from `problem.target` with `problem.kwargs`.
* An **algorithm** is called from `algorithm.target` with `algorithm.kwargs`.
* The runner (`optimize.py`) inspects the algorithm's parameters and supplies what it needs automatically.

You do **not** hard-code call signatures in configs.

### A. Nonlinear (smooth) problems

#### 1) Minimal problem class

Place a class under `optedu/problems/<name>.py`. Implement at least `f(x)`; add `grad(x)` for first-order methods and `hess(x)` for Newton-type methods. Optionally give the known minimizer as `x_star` and `f_star` (used by tests).

```python
# optedu/problems/rosenbrock.py (abridged)
import numpy as np

class Rosenbrock:
    def __init__(self, a=1.0, b=100.0, n=2):
        self.a = float(a); self.b = float(b); self.n = int(n)

    def f(self, x):
        x = np.asarray(x, dtype=float); s = 0.0
        for i in range(x.size-1):
            s += self.b*(x[i+1]-x[i]**2)**2 + (self.a - x[i])**2
        return s

    def grad(self, x):
        x = np.asarray(x, dtype=float); g = np.zeros_like(x)
        for i in range(x.size-1):
            g[i] += -4*self.b*(x[i+1]-x[i]**2)*x[i] + 2*(x[i]-self.a)
            g[i+1] += 2*self.b*(x[i+1]-x[i]**2)
        return g

    # hess(x) is also defined; used by Newton-type methods
```

#### 2) Example config

```json
{
  "title": "GD on Rosenbrock (interactive)",
  "problem": { "target": "optedu.problems.rosenbrock:Rosenbrock", "kwargs": { "n": 2 } },
  "algorithm": { "target": "optedu.algorithms.gradient_descent:gradient_descent",
                 "kwargs": { "step": "exact", "maxit": 500 } },
  "x0": [-1.2, 1.0],
  "visual": {
    "xlims": [-2, 2],
    "ylims": [-1, 3],
    "levels": 50,
    "interactive": true,
    "density": 320,
    "style": { "axes.grid": true }
  }
}
```

* With `-v`, `interactive: true` and `n=2`, you get a live contour view (pan/zoom; the contour recomputes on zoom).

**Tips**

* Return NumPy arrays / floats; keep functions pure and deterministic (unless you pass a seed).
* For 2-D demos, include `visual.xlims/ylims/levels`; for >2-D, the runner shows a PCA trajectory.

### B. Linear programs (LPs)

For LPs, use the provided container and algorithms that mirror the course material (two-phase Phase I and the page-43 simplex steps).

#### 1) Problem container

`optedu.problems.lp:LP` holds `A`, `b`, `c`, the objective `sense` (`"min"` or `"max"`) and the per-row `senses` (`"le"`, `"eq"`, `"ge"`; default all `"eq"`). It checks the shapes when it is created.

#### 2) Example config (mixed ≤ / ≥)

```json
{
  "title": "Two-Phase on mixed-inequality LP",
  "problem": {
    "target": "optedu.problems.lp:LP",
    "kwargs": {
      "A": [[1,  2, -4,  0],
            [3, -1,  0, -2]],
      "b": [2, 1],
      "c": [-2, 1, -3, -1],
      "sense": "min",
      "senses": ["le", "ge"]
    }
  },
  "algorithm": {
    "target": "optedu.algorithms.lp_two_phase:solve_two_phase_generic",
    "kwargs": { "tol": 1e-9, "maxit": 10000 }
  }
}
```

* `solve_two_phase_generic` **standardizes** the LP (slacks for ≤ / ≥ rows; shifted, split or flipped variables for bounds; x ≥ 0 by default) and then calls the two-phase method.
* Phase I builds the auxiliary LP and calls the same page-43 simplex. Redundant constraints are detected and dropped. Phase II restores the original objective.
* `x`, `f` and the recession direction are reported in the **original** variables and objective sense; the standard-form solution is in `result["extra"]["standard_x"]`.
* Outcomes:
  * `status: "converged"` with the optimal solution,
  * `status: "infeasible"` (Phase I optimum > 0), or
  * `status: "unbounded"` with a vertex `x` and a **recession direction** `result["lp"]["direction"]`.

**Pedagogical note (reduced costs)**
For the **min** standard form min cᵀx s.t. Ax = b, x ≥ 0: solve Bᵀy = c_B, then r_N = c_N − Nᵀy.

## Visuals

* 2-D nonlinear problems: contour plot with the optimization path (interactive with `"interactive": true`) and a plot of f per iteration. Tune `xlims`, `ylims`, `levels`, `density`, `style` under `visual`.
* Higher-dimensional problems: the runner shows a PCA projection of the trajectory.
* LPs have no contours; the runner prints the status and numbers (and the ray when unbounded).

## Repository structure

```
optedu/
├─ optedu/
│  ├─ algorithms/   # one file per method (see the table above) + linesearch.py
│  ├─ problems/     # test functions, the LP container and LP standardization
│  ├─ utils/        # result/history types, plotting helpers
│  └─ visuals/      # static and interactive plots used by optimize.py
├─ configs/         # example JSON configs
├─ tests/
├─ optimize.py      # the JSON runner
├─ pyproject.toml
├─ requirements.txt
└─ README.md
```

## Testing

```bash
pip install -e ".[dev]"
pytest -q
```

The tests double as worked examples; each file covers one topic:

| File | What it checks |
| --- | --- |
| `test_problems.py` | gradients and Hessians against finite differences, known minimizers, the `LP` container |
| `test_linesearch.py` | Armijo condition, golden-section search, exact line search vs. the quadratic formula |
| `test_unconstrained.py` | gradient descent, Newton (plain, damped, modified), BFGS; reported evaluation counts |
| `test_dfo.py` | Nelder–Mead operations and simplex, Hooke–Jeeves moves and step lengths |
| `test_metaheuristics.py` | GA, PSO, SA: results, reproducibility with a seed, bounds, history |
| `test_bayesopt.py` | GP posterior formulas, EI / PI / UCB values, the BO loop |
| `test_lp_core.py` | primal simplex and two-phase (KKT conditions, Bland's rule, brute force on random LPs) |
| `test_lp_standardize.py` | standard-form conversion (round trip), general LPs vs. vertex enumeration in 2-D |
| `test_dual_simplex.py` | dual simplex vs. primal simplex, strong duality, infeasibility |
| `test_runner.py`, `test_configs.py` | `optimize.py` and every config in `configs/` |
| `test_visuals.py` | plots, including the interactive contour on zoom |

* Coverage report: `pip install pytest-cov` and run `pytest --cov=optedu`.
* Add your algorithm/problem tests under `tests/`; shared helpers are in `tests/helpers.py`.

## Contributing

* Keep public APIs stable (problem/algorithm call signatures).
* Add tests with each feature or bugfix.
* Prefer small, well-commented PRs. Readability for students comes before speed: plain loops and comments that follow the lecture notes are welcome. Please don't run auto-formatters over the algorithms.

## License

MIT (see `LICENSE`).
