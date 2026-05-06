"""Create the final summary radar chart from accumulated metrics.

Port of R_refactored/make_radar_chart_loc.R — compute_radar() only.
Assembles 14 metrics per sig-dataset pair into an output table + area ratios.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd


ALL_METRICS = [
    "sd_median_ratio", "abs_skewness_ratio", "prop_top_10_percent",
    "prop_top_25_percent", "prop_top_50_percent", "coeff_of_var_ratio",
    "med_prop_na", "med_prop_above_med", "autocor_median",
    "rho_mean_med", "rho_pca1_med", "rho_mean_pca1",
    "prop_pca1_var", "standardization_comp",
]


def compute_radar(
    radar_plot_values: dict,
    names_sigs: list[str],
    names_datasets: list[str],
) -> dict:
    """Assemble radar chart from accumulated metric values.

    Returns dict with keys:
        radar_plot_mat: full matrix including max/min rows (for radarchart rendering)
        output_table: DataFrame (n_sigs*n_datasets x 14) of abs metric values
        areas: array of area ratios for each sig-dataset combination
        legend_labels: list of formatted labels
        radarplot_rownames: list of row name strings
        elapsed_seconds: wall-clock time
    """
    _t0 = time.perf_counter()
    n_metrics = len(ALL_METRICS)
    rows = []
    row_names = []
    legend_parts = []

    for sig in names_sigs:
        for ds in names_datasets:
            vals = radar_plot_values.get(sig, {}).get(ds, {})
            # R fills missing/NA values with 0 (line 59-60 in make_radar_chart_loc.R)
            row = []
            for m in ALL_METRICS:
                v = vals.get(m, 0.0)
                # Convert NaN/None to 0 to match R behavior
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    v = 0.0
                row.append(abs(v))
            rows.append(row)
            row_names.append(f"{ds.replace(' ', '.')}_{sig.replace(' ', '.')}")

    data_mat = np.array(rows, dtype=float)

    # Compute area ratios (sum of consecutive-metric products / n_metrics)
    areas = np.zeros(len(rows))
    for i in range(len(rows)):
        row = data_mat[i]
        area = 0.0
        for j in range(n_metrics):
            next_j = (j + 1) % n_metrics
            area += row[j] * row[next_j]
        areas[i] = area / n_metrics

    # Build legend labels
    idx = 0
    for sig in names_sigs:
        for ds in names_datasets:
            label = f"{ds} {sig} ({areas[idx]:.2g})"
            legend_parts.append(label)
            idx += 1

    # Build full radar_plot_mat with max/min rows (for plotting compatibility)
    max_row = np.ones(n_metrics)
    min_row = np.zeros(n_metrics)
    radar_plot_mat = np.vstack([max_row, min_row, data_mat])

    output_table = pd.DataFrame(data_mat, index=row_names, columns=ALL_METRICS)

    return {
        "radar_plot_mat": radar_plot_mat,
        "output_table": output_table,
        "areas": areas,
        "legend_labels": legend_parts,
        "radarplot_rownames": row_names,
        "elapsed_seconds": time.perf_counter() - _t0,
    }
