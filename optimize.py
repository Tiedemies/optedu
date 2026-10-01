#!/usr/bin/env python3
"""
optimize.py — single entry-point runner for the optedu package.

- Reads a JSON config that specifies:
    • problem.target  = "module.path:ClassOrFactory"
    • problem.kwargs  = {...}
    • algorithm.target= "module.path:function"
    • algorithm.kwargs= {...}
    • x0              = list | ndarray (optional, if the algorithm needs it)
    • title           = text (optional; default plot title)
    • visual          = {interactive: bool, xlims, ylims, levels, title, style, density}
- Runs the algorithm and optionally visualises the recorded history (-v / --visualize).
- Saves artefacts under --save (result.json and any figures) if provided.

"""
from __future__ import annotations
import argparse, json, importlib, inspect, os, sys
from typing import Any, Tuple
import numpy as np

ALLOWED_KEYS = {"title", "problem", "algorithm", "x0", "visual"}

def check_config(cfg: dict) -> None:
    """Stop with a readable message if the config has keys the runner does not use."""
    unknown = set(cfg) - ALLOWED_KEYS
    if unknown:
        sys.exit(f"Unknown config key(s): {sorted(unknown)}. Allowed: {sorted(ALLOWED_KEYS)}.\n"
                 "Plot settings (xlims, ylims, levels, interactive, density, style) belong under \"visual\".")

# ---------- dynamic import helpers ----------
def load_symbol(dotted: str):
    """
    Import a symbol given 'package.module:Symbol' string.
    """
    if ':' not in dotted:
        raise ValueError(f"Expected 'module.path:Symbol', got: {dotted}")
    mod_path, sym_name = dotted.split(':', 1)
    mod = importlib.import_module(mod_path)
    try:
        return getattr(mod, sym_name)
    except AttributeError as e:
        raise ImportError(f"Symbol '{sym_name}' not found in module '{mod_path}'.") from e

def build_problem(spec: dict) -> Tuple[Any, Any, Any, Any]:
    """
    Instantiate the problem from config.
    Return (obj, f, grad, hess) where any of f/grad/hess may be None.
    """
    target = spec["target"]
    kwargs = spec.get("kwargs", {})
    sym = load_symbol(target)
    obj = sym(**kwargs) if callable(sym) else sym
    f    = getattr(obj, "f", None)
    grad = getattr(obj, "grad", None)
    hess = getattr(obj, "hess", None)
    return obj, f, grad, hess

def build_algorithm(spec: dict):
    func = load_symbol(spec["target"])
    if not callable(func):
        raise ValueError("Algorithm target must be a callable.")
    kwargs = spec.get("kwargs", {})
    # Stop early (with the list of accepted names) if the config passes a parameter the algorithm does not have.
    params = inspect.signature(func).parameters
    unknown = [k for k in kwargs if k not in params]
    if unknown:
        sys.exit(f"{spec['target']} has no parameter(s) {unknown}. It accepts: {list(params)}")
    return func, kwargs

# ---------- argument assembly (signature-aware) ----------
def assemble_call(algo_func, algo_kwargs, obj, f, grad, hess, x0):
    """
    Map available items (f, grad, hess, x0, and LP fields if present on obj)
    into the algorithm call according to its signature.
    """
    sig = inspect.signature(algo_func)
    params = list(sig.parameters.values())

    supply = {
        "f": f, "grad": grad, "hess": hess, "x0": x0,
        # Allow LP fields if the problem exposes them
        "A": getattr(obj, "A", None),
        "b": getattr(obj, "b", None),
        "c": getattr(obj, "c", None),
        "sense": getattr(obj, "sense", None),
        "objective": getattr(obj, "sense", None),   # same LP sense ("min"/"max"), under the name two-phase uses
        "senses": getattr(obj, "senses", None),
        "simplex": getattr(obj, "simplex", None),
        "bounds": getattr(obj, "bounds", None),
    }

    call_pos, call_kw = [], {}
    for p in params:
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        name = p.name
        # user-provided kwarg has priority
        if name in algo_kwargs:
            call_kw[name] = algo_kwargs[name]
            continue
        if name in supply and supply[name] is not None:
            if p.kind is inspect.Parameter.POSITIONAL_ONLY:
                call_pos.append(supply[name])
            else:
                call_kw[name] = supply[name]      # pass by name, so the order of arguments cannot get mixed up

    # pass through any remaining user kwargs
    for k, v in algo_kwargs.items():
        if k not in call_kw:
            call_kw[k] = v

    return call_pos, call_kw

# ---------- visualisation helpers ----------
def _maybe_visualize(args, cfg, f, hist, x0):
    from optedu.visuals.core import apply_style, visualize_2d, visualize_values as static_values, visualize_highdim
    from optedu.visuals.interactive import interactive_contour, interactive_values

    if not args.visualize:
        return
    print("Visualizing...")
    # try to infer dimension
    dim = None
    try:
        xs = hist.get("x", [])
        if xs and hasattr(xs[0], "__len__"):
            dim = len(xs[0])
    except Exception:
        pass
    if dim is None:
        # fallback to x0 length if available
        try:
            dim = len(x0) if x0 is not None else 1
        except Exception:
            dim = 1
    vis = cfg.get("visual", {}) if isinstance(cfg, dict) else {}
    apply_style(vis.get("style"))

    save_dir = args.save
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    interactive = bool(vis.get("interactive", False))

    if dim == 2 and callable(f):
        xlims  = tuple(vis.get("xlims", [-2, 2]))
        ylims  = tuple(vis.get("ylims", [-1, 3]))
        levels = int(vis.get("levels", 40))
        title  = vis.get("title", cfg.get("title", "Optimization path"))
        if interactive:
            img = os.path.join(save_dir, "contour_path.png") if save_dir else None
            interactive_contour(f, hist, xlims=xlims, ylims=ylims, levels=levels,
                                title=title, style=vis.get("style"),
                                density=int(vis.get("density", 300)),
                                annotate_every=max(1, len(hist.get('x', []))//10 if hist.get('x') else 1),
                                show=True, save_path=img)
            img2 = os.path.join(save_dir, "values.png") if save_dir else None
            interactive_values(hist, title="Function value", show=True, save_path=img2)
        else:
            img = os.path.join(save_dir, "contour_path.png") if save_dir else None
            visualize_2d(f, hist, xlims=xlims, ylims=ylims, levels=levels, title=title, show=True, save_path=img)
            img2 = os.path.join(save_dir, "values.png") if save_dir else None
            static_values(hist, title="Function value", show=True, save_path=img2)
    else:
        title = vis.get("title", cfg.get("title", "Trajectory (PCA)"))
        img = os.path.join(save_dir, "trajectory_pca.png") if save_dir else None
        visualize_highdim(hist, title=title, show=True, save_path=img)

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Run optedu algorithm from a config.")
    ap.add_argument("config", help="Path to a JSON config file.")
    ap.add_argument("-v", "--visualize", action="store_true", help="Show visualisation of algorithm history.")
    ap.add_argument("--save", default=None, help="Directory to save results/figures.")
    ap.add_argument("--seed", type=int, default=None, help="Random seed for algorithms that accept rng.")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = json.load(f)
    check_config(cfg)

    problem_spec  = cfg["problem"]
    algo_spec     = cfg["algorithm"]
    x0            = np.array(cfg.get("x0"), float) if cfg.get("x0") is not None else None

    obj, f, grad, hess = build_problem(problem_spec)
    algo_func, algo_kwargs = build_algorithm(algo_spec)

    # thread through RNG if supported
    if args.seed is not None:
        rng = np.random.default_rng(args.seed)
        if "rng" in inspect.signature(algo_func).parameters:
            algo_kwargs["rng"] = rng

    call_pos, call_kw = assemble_call(algo_func, algo_kwargs, obj, f, grad, hess, x0)
    result = algo_func(*call_pos, **call_kw)

    # Expect unified result dict with at least: status, x, f, history
    hist = result.get("history", {})
    x_star = result.get("x")
    f_star = result.get("f")

    # Visualisation (if requested)
    _maybe_visualize(args, cfg, f, hist, x0)

    # Summary of the run (printed, and saved as result.json with --save)
    out = {
        "status": result.get("status"),
        "x_star": x_star.tolist() if hasattr(x_star, "tolist") else x_star,
        "f_star": f_star,
        "iterations": (len(hist.get("f", [])) if isinstance(hist, dict) else None),
        "counts": result.get("counts"),
        "message": result.get("message"),
    }
    # Unbounded LP: also show the recession direction d (x + t d stays feasible, objective -> -inf)
    direction = result.get("lp", {}).get("direction")
    if direction is not None:
        out["direction"] = direction.tolist()

    if args.save:
        os.makedirs(args.save, exist_ok=True)
        with open(os.path.join(args.save, "result.json"), "w") as g:
            json.dump(out, g, indent=2)
        print(f"Results saved to: {os.path.abspath(args.save)}")
    else:
        print("Result:")
        print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
