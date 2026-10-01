# optedu/problems/lp_standardize.py
# Converts a general LP to standard form:
#   min   c_std^T z
#   s.t.  A_std z = b_std,  z >= 0
#
# Handles:
#   - objective sense: "min" or "max"
#   - constraint senses per row: "le", "eq", "ge"
#   - variable lower/upper bounds (vector or None)
#   - free variables (lower=-inf and upper=+inf)
#   - returns mapping to reconstruct original variables and objective offset
#
# Output:
#   A_std, b_std, c_std, info
# where info has:
#   - 'reconstruct(x_std) -> x_orig'  : closure to map standard vars back
#   - 'objective_offset'              : constant added to the objective by shifts
#   - 'col_meta': list of dicts describing each standard var component
#
# All arrays are numpy float64.

from __future__ import annotations
import numpy as np
from typing import List, Optional, Tuple, Dict, Any

def to_standard_form(
    c: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    senses: List[str],
    *,
    objective: str = "min",            # "min" or "max"
    lb: Optional[np.ndarray] = None,   # lower bounds (len n) or None
    ub: Optional[np.ndarray] = None    # upper bounds (len n) or None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Convert a general LP to standard form: min c_std^T z s.t. A_std z = b_std, z >= 0.

    Parameters
    ----------
    c, A, b : original LP data, with n variables and m constraints
    senses  : list of length m with entries in {"le","eq","ge"}
    objective : "min" (default) or "max"
    lb, ub : arrays of length n with lower/upper bounds. Use:
             - lb[i] = -np.inf to indicate no lower bound
             - ub[i] = +np.inf to indicate no upper bound

    Returns
    -------
    A_std, b_std, c_std, info
    """
    c = np.asarray(c, dtype=float).copy()
    A = np.asarray(A, dtype=float).copy()
    b = np.asarray(b, dtype=float).copy()
    m, n = A.shape
    if len(senses) != m:
        raise ValueError("senses must have length m = A.shape[0]")

    if lb is None:
        lb = -np.inf * np.ones(n)
    else:
        lb = np.asarray(lb, dtype=float)

    if ub is None:
        ub =  np.inf * np.ones(n)
    else:
        ub = np.asarray(ub, dtype=float)

    # --- Step S0: normalize objective to MIN ---
    if objective.lower().startswith("max"):
        c = -c  # max c^T x  ==  min (-c)^T x

    # --- Step S1: variable preprocessing to ensure nonnegativity ---
    # We'll build a new variable vector z >= 0 from x using:
    #   (a) finite lower bound:   x_j = y_j + L_j, y_j >= 0
    #   (b) free variable:        x_j = y_j^+ - y_j^- with y^+, y^- >= 0
    #   (c) finite upper bound only: x_j = U_j - y_j, y_j >= 0   (flip the variable)
    #
    # Track how each original variable j maps to components in z.
    col_meta: List[Dict[str, Any]] = []  # one entry per column in A_std
    A_work = []
    c_work = []
    shift_const = 0.0  # objective constant term from x = y + L

    # Start with an empty design; for each x_j, add one or two columns to A_work, c_work.
    for j in range(n):
        Aj = A[:, j].copy()
        cj = c[j]
        Lj = lb[j]
        Uj = ub[j]

        if np.isfinite(Lj):  # shift to zero lower bound: x = y + L
            # Adjust objective constant: c^T x = c^T (y + L) = c^T y + c_j * L
            shift_const += cj * Lj
            # Constraint RHS b must account for the shift: A[:,j]*(y_j + L) -> A[:,j]*y_j and b -= A[:,j]*L
            b -= Aj * Lj
            # New variable y >= 0
            A_work.append(Aj)           # column for y
            c_work.append(cj)
            col_meta.append({"orig": j, "type": "shifted", "sign": +1.0, "L": Lj})
        else:
            # No finite lower bound given; check if free
            if not np.isfinite(Uj):
                # Free variable: x = y^+ - y^- (both >= 0)
                A_work.append(Aj);         c_work.append(cj);  col_meta.append({"orig": j, "type": "free_plus", "sign": +1.0})
                A_work.append(-Aj);        c_work.append(-cj); col_meta.append({"orig": j, "type": "free_minus","sign": -1.0})
                continue
            else:
                # lower bound -inf, upper finite: x = U - y with y >= 0
                # c_j x = c_j U - c_j y   and   A[:,j] x = A[:,j] U - A[:,j] y
                shift_const += cj * Uj
                b -= Aj * Uj
                A_work.append(-Aj);        c_work.append(-cj); col_meta.append({"orig": j, "type": "flipped", "sign": -1.0, "U": Uj})
        # For shifted variables, a finite upper bound U becomes an extra row y_j <= U - L in Step S2 below.

    # Build preliminary A_y and c_y
    A_y = np.column_stack(A_work) if A_work else np.zeros((m, 0))
    c_y = np.asarray(c_work, dtype=float)

    # --- Step S2: encode variable upper bounds as extra constraints (<=) on y ---
    # For each shifted variable x_j = y_k + L_j with a finite upper bound U_j, add row y_k <= U_j - L_j.
    extra_rows = []
    extra_rhs = []
    for k, meta in enumerate(col_meta):
        if meta["type"] == "shifted":
            j = meta["orig"]
            if np.isfinite(ub[j]):
                row = np.zeros(A_y.shape[1])
                row[k] = 1.0
                extra_rows.append(row)
                extra_rhs.append(ub[j] - lb[j])

    if extra_rows:
        A_y = np.vstack([A_y, np.zeros((len(extra_rows), A_y.shape[1]))])  # grow rows
        b = np.concatenate([b, np.asarray(extra_rhs, dtype=float)])
        senses = list(senses) + ["le"] * len(extra_rows)
        # put the extra_rows at the end rows
        A_y[-len(extra_rows):, :] = np.vstack(extra_rows)

    # --- Step S3: convert all constraints to equalities with slacks ---
    # We want: A_std z = b_std, z >= 0.
    #   "le":  a_i y <= b_i   ->   a_i y + s_i = b_i
    #   "ge":  a_i y >= b_i   ->  (-a_i) y + s_i = -b_i   (multiply by -1, then add a slack)
    #   "eq":  kept as is, no slack
    # Slack columns are placed after the y columns, one per inequality row.
    slack_cols = [1 if s in ("le", "ge") else 0 for s in senses]
    for s in senses:
        if s not in ("le", "eq", "ge"):
            raise ValueError("senses entries must be in {'le','eq','ge'}")
    total_slacks = sum(slack_cols)
    # Build equality rows:
    A_eq_rows = []
    b_eq = []
    slack_counter = 0
    for i, s in enumerate(senses):
        ai = A_y[i, :]
        bi = b[i]
        if s == "le":
            row = np.hstack([ai, np.zeros(total_slacks)])
            row[len(c_y) + slack_counter] = 1.0
            slack_counter += 1
            A_eq_rows.append(row); b_eq.append(bi)
        elif s == "ge":
            row = np.hstack([-ai, np.zeros(total_slacks)])
            row[len(c_y) + slack_counter] = 1.0
            slack_counter += 1
            A_eq_rows.append(row); b_eq.append(-bi)
        else:  # eq
            row = np.hstack([ai, np.zeros(total_slacks)])
            A_eq_rows.append(row); b_eq.append(bi)

    A_std = np.vstack(A_eq_rows) if A_eq_rows else np.zeros((0, len(c_y) + total_slacks))
    b_std = np.asarray(b_eq, dtype=float)
    # cost: slacks have zero cost
    c_std = np.concatenate([c_y, np.zeros(total_slacks)])

    # --- Step S4: package reconstruction mapping ---
    # Reconstruct x from standard z:
    #  - For each original j:
    #      * if free: x_j = z[k_plus] - z[k_minus]
    #      * if shifted: x_j = z[k] + L_j
    #      * if flipped: x_j = U_j - z[k]
    #  - Slacks ignored when reconstructing x (they correspond to constraints).
    def reconstruct(z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        x = np.zeros(n, dtype=float)
        # main part is first len(c_y) entries
        for k, meta in enumerate(col_meta):
            j = meta["orig"]
            t = meta["type"]
            if t == "free_plus":
                x[j] += z[k]
            elif t == "free_minus":
                x[j] -= z[k]
            elif t == "shifted":
                x[j] += z[k] + meta["L"]
            elif t == "flipped":
                x[j] += meta["U"] - z[k]
            else:
                raise RuntimeError("Unknown column meta type.")
        return x

    info: Dict[str, Any] = {
        "reconstruct": reconstruct,
        "objective_offset": float(shift_const),
        "col_meta": col_meta
    }
    return  A_std, b_std, c_std, info
