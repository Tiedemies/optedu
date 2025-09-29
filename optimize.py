#!/usr/bin/env python

# ============================================================
# Section: Imports & utilities
# ============================================================
import argparse, json, importlib, os, inspect
import numpy as np


def load_symbol(dotted: str):
    if ':' not in dotted:
        raise ValueError(f"Expected 'module.path:Symbol', got: {dotted}")
    mod_path, sym_name = dotted.split(':', 1)
    mod = importlib.import_module(mod_path)
    try:
        return getattr(mod, sym_name)
    except AttributeError:
        raise ImportError(f"Symbol {sym_name} not found in module {mod_path}")


# ============================================================
# Section: Problem / Algorithm builders
# ============================================================
def build_problem(spec: dict):
    """
    Instantiate the problem from JSON. We return the object plus any
    callable NL pieces (f, grad, hess) if they exist.

    Note: For LP problems we do NOT require f/grad/hess.
    """
    target = spec["target"]
    kwargs = spec.get("kwargs", {})
    sym = load_symbol(target)
    obj = sym(**kwargs) if callable(sym) else sym

    f = getattr(obj, 'f', None)
    grad = getattr(obj, 'grad', None)
    hess = getattr(obj, 'hess', None)

    # For NL problems, we require at least f(x)
    # For LP problems (with A,b,c), f can be None.
    if f is None and not (hasattr(obj, 'A') and hasattr(obj, 'b') and hasattr(obj, 'c')):
        raise ValueError("Problem object must expose method 'f(x)' or provide LP data (A,b,c).")

    return obj, f, grad, hess


def build_algorithm(spec: dict):
    target = spec["target"]
    kwargs = spec.get("kwargs", {})
    func = load_symbol(target)
    if not callable(func):
        raise ValueError("Algorithm target must be a callable.")
    return func, kwargs


# ============================================================
# Section: Argument assembly based on algorithm signature
# ============================================================
def assemble_call(algo_func, algo_kwargs, obj, f, grad, hess, x0):
    """
    Build positional/keyword arguments for algo_func based on its signature.

    - We expose a pool of 'supplyables' (f, grad, hess, x0, A, b, c, senses, sense, simplex)
    - We pass parameters in the order the function declares, respecting their 'kind':
        * POSITIONAL_ONLY / POSITIONAL_OR_KEYWORD: may be positional
        * KEYWORD_ONLY: must be keyword
    - JSON-provided algo_kwargs override defaults but do not shadow supplied core args by name.
    """
    sig = inspect.signature(algo_func)
    params = list(sig.parameters.values())

    # Supply pool from the problem object
    supply = {
        "f": f,
        "grad": grad,
        "hess": hess,
        "x0": x0,
        # LP data if present
        "A": getattr(obj, 'A', None),
        "b": getattr(obj, 'b', None),
        "c": getattr(obj, 'c', None),
        "senses": getattr(obj, 'senses', None),
        "sense": getattr(obj, 'sense', None),
    }

    # If the algorithm accepts a 'simplex' argument, provide our page-43 simplex.
    if any(p.name == "simplex" for p in params):
        from optedu.algorithms.lp.simplex_standard import simplex_standard
        supply["simplex"] = simplex_standard

    # If the algorithm expects (A,b,c) but problem included inequalities, standardize first.
    expects_Abc = any(p.name == "A" for p in params) and any(p.name == "b" for p in params) and any(p.name == "c" for p in params)
    if expects_Abc and supply["A"] is not None and getattr(obj, 'senses', None) is not None:
        from optedu.problems.lp_standardize import to_standard_form
        objective = supply.get("sense", "min") or "min"
        c_std, A_std, b_std, info_std = to_standard_form(
            np.asarray(supply["c"], dtype=float),
            np.asarray(supply["A"], dtype=float),
            np.asarray(supply["b"], dtype=float),
            list(supply["senses"]),
            objective=objective,
            lb=None, ub=None
        )
        # Replace A,b,c for the call; keep reconstruct for later
        supply["A"] = A_std
        supply["b"] = b_std
        supply["c"] = c_std
        supply["_std_reconstruct"] = info_std["reconstruct"]
        supply["_std_obj_offset"] = float(info_std.get("objective_offset", 0.0))
    else:
        supply["_std_reconstruct"] = None
        supply["_std_obj_offset"] = 0.0

    # Build call args respecting parameter kind
    call_pos = []
    call_kw = {}

    for p in params:
        name = p.name
        kind = p.kind  # POSITIONAL_ONLY, POSITIONAL_OR_KEYWORD, VAR_POSITIONAL, KEYWORD_ONLY, VAR_KEYWORD

        if kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue  # *args/**kwargs handled implicitly

        # If user provided explicitly in JSON → set as keyword (takes precedence)
        if name in algo_kwargs:
            call_kw[name] = algo_kwargs[name]
            continue

        # Otherwise, if we can supply it and it's not None, pass it
        if name in supply and supply[name] is not None:
            if kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
                if p.default is inspect._empty:
                    # required positional-or-keyword → pass positionally
                    call_pos.append(supply[name])
                else:
                    # has default → pass as keyword for clarity
                    call_kw[name] = supply[name]
            elif kind is inspect.Parameter.KEYWORD_ONLY:
                call_kw[name] = supply[name]

    # Any leftover user kwargs not already set
    for k, v in algo_kwargs.items():
        if k not in call_kw:
            call_kw[k] = v

    return call_pos, call_kw, supply["_std_reconstruct"], supply["_std_obj_offset"]


# ============================================================
# Section: Main (parse, run, visualize, save)
# ============================================================
def main():
    # ----- CLI -----
    p = argparse.ArgumentParser(description="Run an optimization experiment from JSON.")
    p.add_argument("config", help="Path to problem JSON config.")
    p.add_argument("-v", "--visualize", action="store_true", help="Show plots when applicable.")
    p.add_argument("--save", metavar="DIR", default=None, help="Directory to save plots/results.")
    p.add_argument("--seed", type=int, default=None, help="Random seed.")
    args = p.parse_args()

    # ----- Load config -----
    with open(args.config, "r") as f:
        cfg = json.load(f)

    if args.seed is not None:
        np.random.seed(args.seed)

    # ----- Build problem & algorithm -----
    problem_cfg = cfg["problem"]
    obj, f, grad, hess = build_problem(problem_cfg)

    x0 = np.array(cfg.get("x0", [0.0, 0.0]), dtype=float)  # Ignored by LP algorithms; fine for NL

    algo_cfg = cfg["algorithm"]
    algo_func, algo_kwargs = build_algorithm(algo_cfg)

    # ----- Assemble call & run -----
    call_pos, call_kw, reconstruct, obj_offset = assemble_call(algo_func, algo_kwargs, obj, f, grad, hess, x0)
    result = algo_func(*call_pos, **call_kw)
    x_star, hist = result

        # If we standardized an LP, map back to original variables and attach value.
    if reconstruct is not None:
        x_orig = reconstruct(np.asarray(x_star, dtype=float))
        hist.setdefault('standard', {})['z_star'] = x_star
        hist['original_value'] = float(np.asarray(getattr(obj, 'c'), dtype=float) @ x_orig) + obj_offset
        x_star = x_orig  # report original variables to user

        # --- NEW: if there's a recession ray, map it back too ---
        if isinstance(hist, dict) and ('ray' in hist):
            try:
                ray_orig = reconstruct(np.asarray(hist['ray'], dtype=float))
                hist['ray_original'] = ray_orig
                # Optional: objective slope in original vars (should match)
                c_orig = np.asarray(getattr(obj, 'c'), dtype=float)
                hist['ray_objective_slope_original'] = float(c_orig @ ray_orig)
            except Exception:
                # If reconstruction fails for the ray (shouldn't), keep the standard-form ray
                pass

    # ----- Print summary -----
        # ----- Print summary -----
    print("x* =", x_star)

    if isinstance(hist, dict) and hist.get('status') == 'unbounded':
        print("status:", "UNBOUNDED")
        if 'original_value' in hist:
            print("objective value at x* (c^T x):", hist['original_value'])
        # Prefer the original-variable ray if available
        if 'ray_original' in hist:
            print("recession direction d (original vars):", hist['ray_original'])
            if 'ray_objective_slope_original' in hist:
                print("objective slope along d (c^T d, original):", hist['ray_objective_slope_original'])
        elif 'ray' in hist:
            print("recession direction d (standard vars):", hist['ray'])
            if 'ray_objective_slope' in hist:
                print("objective slope along d (c^T d, standard):", hist['ray_objective_slope'])
        if hist.get('reason'):
            print("reason:", hist['reason'])

    elif isinstance(hist, dict) and hist.get('phase1_feasible') is False:
        print("status:", "INFEASIBLE (Phase I optimum > 0)")
        if 'phase1_value' in hist:
            print("phase I value (sum of artificials):", hist['phase1_value'])

    elif isinstance(hist, dict) and hist.get('f'):
        print("f(x*) =", hist['f'][-1])

    elif isinstance(hist, dict) and 'original_value' in hist:
        print("objective value (c^T x):", hist['original_value'])

  
    # ----- Visualization (only for NL problems where f exists) -----
    want_visuals = (args.visualize or cfg.get("visualize", False) or args.save)
    if want_visuals and callable(f):
        from optedu.visuals.core import apply_style
        from optedu.visuals.interactive import interactive_contour, interactive_values
        from optedu.visuals.core import visualize_highdim, visualize_2d, visualize_values as static_values

        apply_style(cfg.get("style"))

        # Determine dimension from history if available, else from x0/x*
        xs = hist.get('x', None) if isinstance(hist, dict) else None
        if xs is None:
            dim = len(x0) if x0 is not None else (len(x_star) if hasattr(x_star, '__len__') else 1)
            xs = [x0]
        else:
            dim = len(xs[0]) if len(xs) > 0 else len(x0)

        save_dir = args.save
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        interactive = bool(cfg.get("interactive", False))

        if dim == 2:
            xlims = tuple(cfg.get("xlims", [-2, 2]))
            ylims = tuple(cfg.get("ylims", [-1, 3]))
            levels = int(cfg.get("levels", 40))
            title = cfg.get("title", "Optimization path")
            if interactive:
                img = os.path.join(save_dir, "contour_path.png") if save_dir else None
                interactive_contour(f, hist, xlims=xlims, ylims=ylims, levels=levels,
                                    title=title, style=cfg.get("style"),
                                    density=int(cfg.get("density", 300)),
                                    annotate_every=max(1, len(hist.get('x', []))//10 if hist.get('x') else 1),
                                    show=args.visualize, save_path=img)
            else:
                img = os.path.join(save_dir, "contour_path.png") if save_dir else None
                visualize_2d(f, hist, xlims=xlims, ylims=ylims, levels=levels,
                             title=title, show=args.visualize, save_path=img)
                img2 = os.path.join(save_dir, "values.png") if save_dir else None
                static_values(hist, title="Function value", show=args.visualize, save_path=img2)
        else:
            title = cfg.get("title", "Trajectory (PCA)")
            img = os.path.join(save_dir, "trajectory_pca.png") if save_dir else None
            visualize_highdim(hist, title=title, show=args.visualize, save_path=img)
            img2 = os.path.join(save_dir, "values.png") if save_dir else None
            interactive_values(hist, title="Function value", show=args.visualize, save_path=img2)

    # ----- Save result JSON -----
    if args.save:
        out = {
            "x_star": x_star.tolist() if hasattr(x_star, 'tolist') else x_star,
            "f_star": (hist['f'][-1] if isinstance(hist, dict) and hist.get('f') else
                       (hist.get('original_value') if isinstance(hist, dict) else None)),
            "iterations": (len(hist.get('f', [])) if isinstance(hist, dict) else None)
        }
        with open(os.path.join(args.save, "result.json"), "w") as g:
            json.dump(out, g, indent=2)


# ============================================================
# Section: Entry point
# ============================================================
if __name__ == "__main__":
    main()
