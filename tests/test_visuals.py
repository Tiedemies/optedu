# tests/test_visuals.py
# Plotting helpers and figures (rendered off-screen with the Agg backend).
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from optedu.algorithms.gradient_descent import gradient_descent
from optedu.problems.rosenbrock import Rosenbrock
from optedu.utils.plotting import contour_2d, pca_trajectory, plot_path, plot_values
from optedu.visuals import interactive
from optedu.visuals.core import DEFAULT_STYLE, apply_style, visualize_2d, visualize_highdim, visualize_values


@pytest.fixture
def history():
    prob = Rosenbrock()
    return gradient_descent(f=prob.f, grad=prob.grad, x0=np.array([-1.2, 1.0]), maxit=20)["history"]


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_contour_2d():
    fig, ax = plt.subplots()
    _, cs = contour_2d(Rosenbrock().f, xlims=(-2, 2), ylims=(-1, 3), levels=10, grid=30, ax=ax)
    assert ax.get_xlim() == (-2, 2) and ax.get_ylim() == (-1, 3)
    assert len(cs.levels) > 0


def test_plot_path_2d_with_labels(history):
    fig, ax = plt.subplots()
    plot_path(history, ax=ax, annotate_every=5)
    line = ax.lines[0]
    assert np.allclose(line.get_xdata(), [x[0] for x in history["x"]])
    assert len(ax.texts) == len(range(0, len(history["x"]), 5))         # labels 0, 5, 10, ...


def test_plot_path_1d_and_empty():
    fig, ax = plt.subplots()
    plot_path({"x": [np.array([1.0]), np.array([0.5])]}, ax=ax)
    assert ax.get_xlabel() == "iteration"
    fig, ax = plt.subplots()
    plot_path({}, ax=ax)
    assert len(ax.lines) == 0


def test_plot_values(history):
    fig, ax = plt.subplots()
    plot_values(history, ax=ax)
    assert np.allclose(ax.lines[0].get_ydata(), history["f"])


def test_pca_trajectory_of_a_line_is_a_line():
    # points on a straight line in 3-D: the projection uses one principal component only
    X = [np.array([t, 2 * t, -t]) for t in np.linspace(0, 1, 6)]
    fig, ax = plt.subplots()
    pca_trajectory({"x": X}, ax=ax)
    assert np.allclose(ax.lines[0].get_ydata(), 0.0, atol=1e-12)


def test_apply_style():
    apply_style({"axes.grid": False})
    assert plt.rcParams["axes.grid"] is False
    apply_style()
    assert plt.rcParams["lines.linewidth"] == DEFAULT_STYLE["lines.linewidth"]


@pytest.mark.parametrize("make", [
    lambda h, p: visualize_2d(Rosenbrock().f, h, levels=10, title="t", show=False, save_path=p),
    lambda h, p: visualize_values(h, title="t", show=False, save_path=p),
    lambda h, p: visualize_highdim(h, title="t", show=False, save_path=p),
    lambda h, p: interactive.interactive_values(h, title="t", show=False, save_path=p),
])
def test_figures_are_saved(make, history, tmp_path):
    path = tmp_path / "figure.png"
    make(history, str(path))
    assert path.exists() and path.stat().st_size > 0


def test_interactive_contour_redraws_on_zoom(history, monkeypatch):
    # Redraw immediately instead of after a timer, and keep the figure open so we can inspect it.
    class Immediate:
        def __init__(self, fig, interval_ms, callback): self.callback = callback
        def schedule(self): self.callback()
    monkeypatch.setattr(interactive, "_Debouncer", Immediate)
    monkeypatch.setattr(interactive.plt, "close", lambda *a, **k: None)

    interactive.interactive_contour(Rosenbrock().f, history, xlims=(-2, 2), ylims=(-1, 3), density=40,
                                    annotate_every=5, show=False)
    ax = plt.gcf().axes[0]
    n_lines, n_labels = len(ax.lines), len(ax.texts)
    ax.set_xlim(-1, 1); ax.set_ylim(0, 2)                 # zoom: two redraws
    assert len(ax.lines) == n_lines and len(ax.texts) == n_labels   # the path is not drawn again
    assert ax.get_xlim() == (-1, 1)


def test_auto_grid_follows_axes_shape():
    fig, ax = plt.subplots(figsize=(8, 4))
    nx, ny = interactive._auto_grid(ax, density=100)
    assert ny == 100 and nx > ny                           # 100 samples on the short side, more on the long one


def test_interactive_contour_options(history, tmp_path):
    path = tmp_path / "contour.png"
    interactive.interactive_contour(Rosenbrock().f, history, density=40, title="GD", style={"axes.grid": False},
                                    show=False, save_path=str(path))
    assert path.exists()


def test_debouncer_calls_back():
    calls = []
    fig = plt.figure()
    debouncer = interactive._Debouncer(fig, interval_ms=10, callback=lambda: calls.append(1))
    debouncer.schedule()          # (re)starts the timer; the timer itself only runs in a GUI event loop
    debouncer._fire()             # what the timer does when it fires
    assert calls == [1]
