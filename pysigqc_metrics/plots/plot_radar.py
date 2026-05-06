"""Radar (spider) chart matching R's fmsb::radarchart styling.

Produces ``sig_radarplot.pdf`` — overlaid polygons per (sig, dataset).
R parameters replicated:
  - maxmin=T, axistype=1
  - cglcol='grey', cglty=1, cglwd=1
  - caxislabels=seq(0,1,length.out=5)
  - vlcex=0.6, calcex=0.5
  - plwd=2, pcol=dataset_colors, plty=sig_index
  - title='Signature Summary'
  - Legend: sorted by area, outside right, bty='n'
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ._colors import dataset_colors, sig_linestyle
from ._style import save_pdf, dynamic_fontsize

# 14 metric labels matching R's vlabels
_METRIC_LABELS = [
    "Rel. med. SD",
    "Abs. skewness",
    "Top 10% CV",
    "Top 25% CV",
    "Top 50% CV",
    "CV ratio",
    "Med. prop.\nnon-NA",
    "Med. prop.\nexpr.",
    "Autocorrelation",
    r"$\rho$(Mean,Med)",
    r"$\rho$(PCA1,Med)",
    r"$\rho$(Mean,PCA1)",
    "Prop. var. PC1",
    "Std. robustness",
]


def plot_radar(
    radar_result: dict,
    names_sigs: list[str],
    names_datasets: list[str],
    out_dir: str | Path,
) -> Path:
    radar_plot_mat = np.asarray(radar_result["radar_plot_mat"], dtype=float)
    areas = radar_result.get("areas", [])
    legend_labels = radar_result.get("legend_labels", [])

    n_ds = len(names_datasets)
    n_sigs = len(names_sigs)
    n_metrics = radar_plot_mat.shape[1]
    ds_colors = dataset_colors(n_ds)

    max_label = max((len(l) for l in legend_labels), default=20)
    legend_cex = min(0.8, 3 * 10 / max_label)
    legend_font = legend_cex * 10

    # Angular positions
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)  # Start at top (12 o'clock)
    ax.set_theta_direction(1)  # COUNTER-CLOCKWISE to match R's fmsb::radarchart

    # Grid: R cglcol='grey', cglty=1 (solid), cglwd=1
    ax.set_ylim(0, 1)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1"],
                        fontsize=5, color="grey")  # R calcex=0.5
    ax.spines["polar"].set_color("grey")
    ax.spines["polar"].set_linewidth(1)
    for spine in ax.spines.values():
        spine.set_color("grey")
    ax.grid(color="grey", linewidth=0.8, linestyle="-")  # solid grid
    ax.set_facecolor("white")

    # Spoke labels (R vlcex=0.6)
    labels = _METRIC_LABELS[:n_metrics]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=6)

    # Plot each polygon (R: plwd=2, pcol=colours_array[i], plty=k)
    data_rows = radar_plot_mat[2:]
    row_idx = 0
    handles = []
    labs = []
    for ki, sig in enumerate(names_sigs):
        for di, ds in enumerate(names_datasets):
            if row_idx >= len(data_rows):
                break
            vals = np.abs(data_rows[row_idx])
            vals_closed = np.concatenate([vals, [vals[0]]])

            ls = sig_linestyle(ki)
            line, = ax.plot(angles, vals_closed, color=ds_colors[di],
                            linestyle=ls, lw=2, alpha=0.9,
                            marker='o', markersize=5)  # Add dots like R's pty=16
            ax.fill(angles, vals_closed, color=ds_colors[di], alpha=0.05)

            label = legend_labels[row_idx] if row_idx < len(legend_labels) else f"{ds} {sig}"
            handles.append(line)
            labs.append(label)
            row_idx += 1

    # Legend outside right (R: bty='n', sorted by area)
    if handles:
        ax.legend(handles, labs, loc="upper left", bbox_to_anchor=(1.12, 1.0),
                  fontsize=legend_font, frameon=False, title="Datasets",
                  title_fontsize=legend_font + 1)

    ax.set_title("Signature Summary", fontsize=12, pad=20)
    fig.tight_layout()
    return save_pdf(fig, out_dir, "sig_radarplot.pdf")
