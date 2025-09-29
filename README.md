````markdown
# optedu

Teaching-first optimization library for MSc/PhD coursework and live demos. It provides **unified visuals**, a **JSON-runner** (`optimize.py`), and implementations that **mirror the course material** (e.g., page-43 simplex, §3.3.2 two-phase).

> Design goals: reproducible lecture demos, compact configs, clean APIs, and pedagogical comments in the LP core.

## Installation

```bash
# clone your repository (example path)
git clone https://github.com/<you>/optedu.git
cd optedu

# dev install
pip install -e .

# optional dev tools
pip install -e .[dev]
````

Python 3.9+ recommended. Dependencies are lightweight: NumPy and Matplotlib.

## Quick start

### Rosenbrock + gradient descent (interactive, pan/zoom)

```bash
python optimize.py configs/rosenbrock_gd.json -v
```

### LP with mixed inequalities (two-phase + page-43 simplex)

```bash
python optimize.py configs/lp_mixed_ineq.json
```

* The solver prints `status: optimal | infeasible | unbounded`.
* For **unbounded**, it also prints a **recession direction** `d` with `A d = 0` and `c^T d < 0`.

## Configuration format

All experiments use the same JSON template:

```json
{
  "title": "My experiment",
  "problem": { "target": "module.path:ClassOrFactory", "kwargs": { "param": 123 } },
  "algorithm": { "target": "module.path:function", "kwargs": { "tol": 1e-6 } },
  "x0": [ ... ],                     // used by NL problems; ignored by LPs
  "interactive": true,               // for 2-D NL problems (live contour)
  "xlims": [-2, 2], "ylims": [-1, 3],
  "levels": 50, "density": 320,
  "style": { "axes.grid": true }
}
```

* `problem.target` and `algorithm.target` are **import strings** of the form `"package.module:Symbol"`.
* `optimize.py` inspects the **algorithm’s signature** and supplies exactly the arguments it declares.

  * NL problems may receive `f`, `grad`, `hess`, `x0`.
  * LP algorithms receive `A`, `b`, `c` (and, if needed, a `simplex` callback).
* Visuals automatically activate for 2-D NL problems when `"interactive": true` or when you pass `-v`.

## How to define a problem (for configs & live demos)

This project uses one simple pattern for **all** problems:

* A **problem** is constructed from `problem.target` with `problem.kwargs`.
* An **algorithm** is called from `algorithm.target` with `algorithm.kwargs`.
* The runner (`optimize.py`) inspects the algorithm’s parameters and supplies what it needs automatically:

  * Nonlinear problems: `f`, `grad`, `hess`, `x0` as requested by the algorithm’s signature.
  * Linear programs: `A`, `b`, `c` (and it will standardize if `senses` are provided).

You do **not** hard-code call signatures in configs.

### A. Nonlinear (smooth) problems

#### 1) Minimal problem class

Place a class under `optedu/problems/<name>.py`. Implement at least `f(x)`; add `grad(x)` for first-order methods and `hess(x)` for Newton-type methods.

```python
# optedu/problems/rosenbrock.py
import numpy as np

class Rosenbrock:
    def __init__(self, n: int = 2, a: float = 1.0, b: float = 100.0):
        self.n, self.a, self.b = n, a, b

    def f(self, x: np.ndarray) -> float:
        x = np.asarray(x, float)
        return np.sum((self.a - x[:-1])**2 + self.b*(x[1:] - x[:-1]**2)**2)

    def grad(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, float)
        g = np.zeros_like(x)
        g[:-1] = -2*(self.a - x[:-1]) - 4*self.b*(x[1:] - x[:-1]**2)*x[:-1]
        g[1:] += 2*self.b*(x[1:] - x[:-1]**2)
        return g

    # Optional; used by Newton-type methods if present
    def hess(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, float)
        n = self.n
        H = np.zeros((n, n))
        for i in range(n-1):
            H[i, i] += 2 + 12*self.b*x[i]**2 - 4*self.b*x[i+1]
            H[i, i+1] += -4*self.b*x[i]
            H[i+1, i] += -4*self.b*x[i]
            H[i+1, i+1] += 2*self.b
        return H
```

#### 2) Example config

```json
{
  "title": "GD on Rosenbrock (interactive)",
  "problem": { "target": "optedu.problems.rosenbrock:Rosenbrock", "kwargs": { "n": 2 } },
  "algorithm": { "target": "optedu.algorithms.gradient_descent:gradient_descent",
                 "kwargs": { "step": "exact_numeric", "maxit": 500 } },
  "x0": [-1.2, 1.0],
  "xlims": [-2, 2],
  "ylims": [-1, 3],
  "levels": 50,
  "interactive": true,
  "density": 320,
  "style": { "axes.grid": true }
}
```

* With `interactive: true` and `n=2`, you get a live contour view (pan/zoom; levels recompute on zoom).

**Tips**

* Return NumPy arrays / floats; keep functions pure and deterministic (unless you pass a seed).
* For 2-D demos, include `xlims/ylims/levels`; for >2-D, the runner shows PCA trajectory and a values plot.

### B. Linear programs (LPs)

For LPs, use the provided container and algorithms that mirror the course material (two-phase Phase I and the page-43 simplex steps).

#### 1) Problem container

```python
# optedu/problems/lp.py
import numpy as np
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class LP:
    A: np.ndarray
    b: np.ndarray
    c: np.ndarray
    sense: str = "min"                 # "min" or "max"
    senses: Optional[List[str]] = None # per-row: "le", "eq", "ge"

    def __post_init__(self):
        self.A = np.asarray(self.A, float)
        self.b = np.asarray(self.b, float)
        self.c = np.asarray(self.c, float)
        if self.senses is None:
            self.senses = ["eq"] * self.A.shape[0]
```

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
    "target": "optedu.algorithms.lp.two_phase:solve_two_phase",
    "kwargs": { "tol": 1e-9, "maxit": 10000 }
  },
  "x0": [0, 0, 0, 0],
  "style": { "axes.grid": true }
}
```

* The runner notices the algorithm expects `(A, b, c, ...)` and extracts them from the `LP` object.
* If `senses` includes `le/ge`, it **standardizes** to equalities and nonnegativity before calling **two-phase**.
* Phase I builds the auxiliary LP and calls the same page-43 simplex; Phase II restores the original objective.
* Outcomes are clearly reported:

  * `status: "optimal"` with the solution,
  * `status: "infeasible"` (Phase I optimum > 0), or
  * `status: "unbounded"` with a **recession direction** `ray` (and its objective slope).

**Pedagogical note (reduced costs)**
For the **min** standard form ( \min c^T x \ \text{s.t.}\ Ax=b,\ x\ge0 ): solve ( B^{\top} y = c_B ), then ( r_N = c_N - N^{\top} y ).

## Visuals

* For 2-D NL problems: set `"interactive": true` in the config (pan/zoom contours). You can tune:

  * `xlims`, `ylims`, `levels`, `density`, `style`.
* For higher-D NL problems: the runner shows a PCA trajectory and a values plot.
* LPs have no contours; the runner prints structured status + numbers (and rays when unbounded).

## Repository structure (incomplete, to be updated)

```
optedu/
├─ optedu/
│  ├─ algorithms/
│  │  ├─ solve_lp.py
│  │  └─ lp/
│  │     ├─ simplex_standard.py
│  │     └─ two_phase.py
│  ├─ problems/
│  │  ├─ lp.py
│  │  └─ lp_standardize.py
│  └─ visuals/
│     ├─ core.py
│     └─ interactive.py
├─ configs/
│  ├─ rosenbrock_gd.json
│  └─ lp_mixed_ineq.json
├─ tests/
│  └─ test_lp_unbounded.py
├─ optimize.py
├─ pyproject.toml
├─ requirements.txt
└─ README.md
```

## Testing (not currently functional) 

```bash
pytest -q
```

* Minimal LP tests confirm feasible/infeasible/unbounded outcomes.
* Add your algorithm/problem tests under `tests/`.

## Contributing

* Keep public APIs stable (problem/algorithm call signatures).
* Add tests with each feature or bugfix.
* Prefer small, well-commented PRs.

## License

MIT (see `LICENSE`).

````
