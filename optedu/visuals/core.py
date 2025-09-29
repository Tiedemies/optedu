import matplotlib.pyplot as plt
from typing import Dict, Any, Optional
from ..utils.plotting import contour_2d, plot_path, plot_values, pca_trajectory

DEFAULT_STYLE = {
    "figure.figsize": (6, 5),
    "axes.grid": True,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "lines.linewidth": 2.0,
    "lines.markersize": 4.0,
    "font.size": 10,
}

def apply_style(style: Optional[Dict[str, Any]] = None):
    conf = DEFAULT_STYLE.copy()
    if style: conf.update(style)
    plt.rcParams.update(conf)

def visualize_2d(f, hist, xlims=(-2,2), ylims=(-2,2), levels=40, title=None, show=True, save_path=None):
    fig, ax = plt.subplots()
    contour_2d(f, xlims=xlims, ylims=ylims, levels=levels, ax=ax)
    path_annot = max(1, len(hist.get('x', []))//10) if hist.get('x') else 0
    plot_path(hist, ax=ax, annotate_every=path_annot)
    if title: ax.set_title(title)
    if save_path: fig.savefig(save_path, bbox_inches="tight", dpi=150)
    if show: plt.show()
    plt.close(fig)

def visualize_values(hist, title=None, show=True, save_path=None):
    fig, ax = plt.subplots()
    plot_values(hist, ax=ax)
    if title: ax.set_title(title)
    if save_path: fig.savefig(save_path, bbox_inches="tight", dpi=150)
    if show: plt.show()
    plt.close(fig)

def visualize_highdim(hist, title=None, show=True, save_path=None):
    fig, ax = plt.subplots()
    pca_trajectory(hist, ax=ax, show=False)
    if title: ax.set_title(title)
    if save_path: fig.savefig(save_path, bbox_inches="tight", dpi=150)
    if show: plt.show()
    plt.close(fig)
