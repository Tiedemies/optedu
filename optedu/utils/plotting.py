import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
from .types import History

def contour_2d(f: Callable[[np.ndarray], float], xlims=(-2,2), ylims=(-2,2), levels=30, grid=400, ax=None):
    if ax is None: ax = plt.gca()
    xs = np.linspace(*xlims, grid); ys = np.linspace(*ylims, grid)
    X, Y = np.meshgrid(xs, ys); Z = np.empty_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = f(np.array([X[i, j], Y[i, j]]))
    cs = ax.contour(X, Y, Z, levels=levels)
    ax.set_xlim(xlims); ax.set_ylim(ylims)
    ax.set_xlabel('x1'); ax.set_ylabel('x2')
    return ax, cs

def plot_path(hist: History, ax=None, show=False, annotate_every=0):
    if ax is None: ax = plt.gca()
    X = np.array(hist.get('x', []))
    if X.size == 0: return ax
    if X.shape[1] >= 2:
        ax.plot(X[:,0], X[:,1], marker='o', linestyle='-')
        if annotate_every and annotate_every > 0:
            for i in range(0, X.shape[0], annotate_every):
                ax.annotate(str(i), (X[i,0], X[i,1]))
    else:
        ax.plot(np.arange(X.shape[0]), X[:,0], marker='o')
        ax.set_xlabel('iteration'); ax.set_ylabel('x')
    if show: plt.show()
    return ax

def plot_values(hist: History, ax=None, show=False):
    if ax is None: ax = plt.gca()
    fvals = np.array(hist.get('f', []))
    ax.plot(fvals, marker='o')
    ax.set_xlabel('iteration'); ax.set_ylabel('f(x)')
    if show: plt.show()
    return ax

def pca_trajectory(hist: History, ax=None, show=False):
    X = np.array(hist.get('x', []))
    if X.size == 0: return plt.gca() if ax is None else ax
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Y = Xc @ Vt[:2].T
    if ax is None: ax = plt.gca()
    ax.plot(Y[:,0], Y[:,1], marker='o')
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
    if show: plt.show()
    return ax
