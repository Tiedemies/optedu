# optedu/visuals/interactive.py
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from ..utils.plotting import plot_path, plot_values

class _Debouncer:
    def __init__(self, fig: Figure, interval_ms: int, callback):
        self._timer = fig.canvas.new_timer(interval=interval_ms)
        self._timer.single_shot = True
        self._timer.add_callback(self._fire)
        self._callback = callback
    def _fire(self):
        self._callback()
    def schedule(self):
        # restart timer
        try:
            self._timer.stop()
        except Exception:
            pass
        self._timer.start()

def _auto_grid(ax: Axes, density: int = 300) -> np.ndarray:
    """
    Choose grid size based on on-screen pixel span so we don't oversample.
    'density' ~ target samples along the shorter axis (typ. 200-600).
    """
    # axis bbox in pixels
    bb = ax.get_window_extent().transformed(ax.figure.dpi_scale_trans.inverted())
    px_w = max(1, int(bb.width * ax.figure.dpi))
    px_h = max(1, int(bb.height * ax.figure.dpi))
    short = min(px_w, px_h)
    scale = max(50, min(4 * density, int(short / (short / density))))  # clamp
    # keep aspect ratio roughly square
    nx = int(scale * (px_w / short))
    ny = int(scale * (px_h / short))
    return np.array([max(50, nx), max(50, ny)], dtype=int)

def interactive_contour(
    f,
    hist,
    xlims=(-2.0, 2.0),
    ylims=(-2.0, 2.0),
    levels: int = 40,
    title: Optional[str] = None,
    style: Optional[Dict[str, Any]] = None,
    density: int = 300,           # ~samples on the short axis
    annotate_every: int = 0,
    show: bool = True,
    save_path: Optional[str] = None,
):
    """
    Interactive contour with dynamic recomputation on pan/zoom.
    - Use toolbar pan/zoom normally; after motion stops, the contour recomputes.
    - 'density' controls resolution (higher = sharper, slower).
    """
    # style (use your unified apply_style from visuals.core if you prefer)
    if style:
        plt.rcParams.update(style)

    fig, ax = plt.subplots()
    ax.set_xlim(*xlims); ax.set_ylim(*ylims)
    if title:
        ax.set_title(title)

    # initial draw
    cs = [None]  # store artists in a list for reassignment

    def _compute_and_draw():
        # clear previous contour artists
        if cs[0] is not None:
            for c in cs[0].collections:
                c.remove()
            cs[0] = None

        (xl, xr) = ax.get_xbound()
        (yl, yr) = ax.get_ybound()
        nx, ny = _auto_grid(ax, density=density)
        xs = np.linspace(xl, xr, nx)
        ys = np.linspace(yl, yr, ny)
        X, Y = np.meshgrid(xs, ys)

        # Evaluate f on the current view
        Z = np.empty_like(X)
        # vectorized evaluation is not guaranteed here; do safe loops
        for i in range(ny):
            for j in range(nx):
                Z[i, j] = f(np.array([X[i, j], Y[i, j]]))

        cs[0] = ax.contour(X, Y, Z, levels=levels)
        # Re-plot path (lines auto-scale with limits)
        plot_path(hist, ax=ax, annotate_every=annotate_every)
        fig.canvas.draw_idle()

    # initial render
    _compute_and_draw()

    # debounce updates on axis limit change
    debouncer = _Debouncer(fig, interval_ms=120, callback=_compute_and_draw)
    ax.callbacks.connect("xlim_changed", lambda a: debouncer.schedule())
    ax.callbacks.connect("ylim_changed", lambda a: debouncer.schedule())

    # optional save
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig)

def interactive_values(hist, title=None, show=True, save_path=None):
    fig, ax = plt.subplots()
    plot_values(hist, ax=ax)
    if title:
        ax.set_title(title)
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)
