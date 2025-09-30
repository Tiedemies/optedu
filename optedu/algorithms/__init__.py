# Public API (stable surface)
from .lp_simplex import simplex
from .lp_two_phase import two_phase_simplex

from .gradient_descent import gradient_descent as gradient_descent_legacy
from .gradient_descent import gradient_descent_unified

from .newton import newton as newton_legacy
from .newton import newton_method

from .bfgs import bfgs_unified
from .nelder_mead import nelder_mead_unified
from .hooke_jeeves import hooke_jeeves_unified

from .genetic import genetic_minimize as genetic_minimize_legacy
from .genetic import genetic_minimize_unified

from .pso import pso_minimize
from .sa import simulated_annealing as simulated_annealing_legacy
from .sa import simulated_annealing_unified

__all__ = [
    # LP
    "simplex", "two_phase_simplex",
    # Smooth (legacy and unified)
    "gradient_descent_legacy", "gradient_descent_unified",
    "newton_legacy", "newton_method",
    "bfgs_unified", "nelder_mead_unified", "hooke_jeeves_unified",
    # Metaheuristics (legacy and unified)
    "genetic_minimize_legacy", "genetic_minimize_unified",
    "pso_minimize",
    "simulated_annealing_legacy", "simulated_annealing_unified",
]
