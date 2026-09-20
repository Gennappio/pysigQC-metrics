"""Sequential radar-metrics pipeline orchestrator.

Runs the 5 compute modules in series and assembles the radar chart.
No parallelism (no joblib), no negative control, no eval_struct.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from .eval_var import compute_var
from .eval_expr import compute_expr
from .eval_compactness import compute_compactness
from .eval_stan import compute_stan
from .compare_metrics import compute_metrics
from .radar_chart import compute_radar


def run_pipeline(
    gene_sigs_list: dict[str, list[str]],
    names_sigs: list[str],
    mRNA_expr_matrix: dict[str, pd.DataFrame],
    names_datasets: list[str],
    out_dir: str | Path | None = None,
    thresholds: dict[str, float] | list[float] | None = None,
    verbose: bool = False,
) -> dict:
    """Run the sequential radar-metrics pipeline.

    Args:
        gene_sigs_list: dict of signature name -> gene list
        names_sigs: ordered list of signature names
        mRNA_expr_matrix: dict of dataset name -> DataFrame (genes x samples)
        names_datasets: ordered list of dataset names
        out_dir: if not None, write radarchart_table.txt under this directory
        thresholds: expression thresholds per dataset (dict or list, default: median)
        verbose: if True, print progress

    Returns dict with:
        var_result, expr_result, compact_result, stan_result, metrics_result:
            individual module results
        radar_result: assembled radar chart (radar_plot_mat, output_table,
            areas, legend_labels, radarplot_rownames)
        radar_values: merged per-sig per-dataset metric dicts
        elapsed_seconds: total wall-clock time
    """
    _t0 = time.perf_counter()

    if isinstance(thresholds, list):
        if len(thresholds) != len(names_datasets):
            raise ValueError(
                f"Number of thresholds ({len(thresholds)}) must match "
                f"number of datasets ({len(names_datasets)})"
            )
        thresholds = dict(zip(names_datasets, thresholds))

    if verbose:
        print("[pipeline] compute_var ...")
    var_r = compute_var(gene_sigs_list, names_sigs, mRNA_expr_matrix, names_datasets)

    if verbose:
        print("[pipeline] compute_expr ...")
    expr_r = compute_expr(gene_sigs_list, names_sigs, mRNA_expr_matrix,
                          names_datasets, thresholds=thresholds)

    if verbose:
        print("[pipeline] compute_compactness ...")
    compact_r = compute_compactness(gene_sigs_list, names_sigs,
                                    mRNA_expr_matrix, names_datasets)

    if verbose:
        print("[pipeline] compute_metrics ...")
    metrics_r = compute_metrics(gene_sigs_list, names_sigs,
                                mRNA_expr_matrix, names_datasets)

    if verbose:
        print("[pipeline] compute_stan ...")
    stan_r = compute_stan(gene_sigs_list, names_sigs,
                          mRNA_expr_matrix, names_datasets)

    if verbose:
        print("[pipeline] assembling radar ...")
    radar_values: dict = {}
    for sig in names_sigs:
        radar_values[sig] = {}
        for ds in names_datasets:
            vals: dict = {}
            vals.update(var_r["radar_values"][sig][ds])
            vals.update(expr_r["radar_values"][sig][ds])
            vals.update(compact_r["radar_values"][sig][ds])
            vals.update(metrics_r["radar_values"][sig][ds])
            vals.update(stan_r["radar_values"][sig][ds])
            radar_values[sig][ds] = vals

    radar_result = compute_radar(radar_values, names_sigs, names_datasets)

    if out_dir is not None:
        out_path = Path(out_dir)
        radar_dir = out_path / "radarchart_table"
        radar_dir.mkdir(parents=True, exist_ok=True)
        radar_result["output_table"].to_csv(
            radar_dir / "radarchart_table.txt", sep="\t"
        )

    return {
        "var_result": var_r,
        "expr_result": expr_r,
        "compact_result": compact_r,
        "stan_result": stan_r,
        "metrics_result": metrics_r,
        "radar_result": radar_result,
        "radar_values": radar_values,
        "elapsed_seconds": time.perf_counter() - _t0,
    }
