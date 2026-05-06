"""Shared plotting helpers: figure layout, PDF saving, font scaling.

Sets a matplotlib rcParams baseline that approximates R's default graphics
device styling (white background, moderate grid, serif-adjacent fonts).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# --- Global style baseline (approximates R's default device) ---
_STYLE_APPLIED = False

def _apply_style():
    global _STYLE_APPLIED
    if _STYLE_APPLIED:
        return
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.linewidth": 0.6,
        "axes.grid": False,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "lines.linewidth": 1.5,
        "font.size": 9,
        "legend.fontsize": 7,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "pdf.fonttype": 42,       # TrueType in PDF (editable text)
    })
    _STYLE_APPLIED = True


def figure_grid(
    n_rows: int,
    n_cols: int,
    cell_w: float = 4.0,
    cell_h: float = 4.0,
) -> tuple[plt.Figure, np.ndarray]:
    """Create a figure with an *n_rows* x *n_cols* subplot grid.

    Returns ``(fig, axes)`` where *axes* is always 2-D.
    """
    _apply_style()
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(cell_w * n_cols, cell_h * n_rows),
        squeeze=False,
    )
    return fig, axes


def new_figure(*args, **kwargs) -> tuple[plt.Figure, plt.Axes | np.ndarray]:
    """Wrapper around plt.subplots that ensures style is applied."""
    _apply_style()
    return plt.subplots(*args, **kwargs)


def save_pdf(fig: plt.Figure, out_dir: str | Path, filename: str) -> Path:
    """Save *fig* as a PDF in *out_dir*, close it, and return the path."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / filename
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return path


def dynamic_fontsize(max_label_length: int, base: float = 10.0) -> float:
    """Compute a font size that shrinks for long labels.

    Matches R's ``cex.main = min(1, 4*10/max_title_length)`` scaled to
    matplotlib's point system where base=10 ~ R cex=1.
    """
    if max_label_length <= 0:
        return base
    return min(base, 4 * 10 / max_label_length)


def gene_label_fontsize(n_genes: int) -> float:
    """Adaptive gene-name label size matching R's formula in eval_expr_loc."""
    return max(min(5.0, (0.5 * 4 * 12) / (1.414 * max(n_genes, 1))), 3.0)
