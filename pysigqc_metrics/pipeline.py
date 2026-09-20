"""Sequential radar-metrics pipeline orchestrator.

Runs the 5 compute modules in series and assembles the radar chart.
No parallelism (no joblib), no negative control, no eval_struct.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import pandas as pd

from .backends import DatasetStatsCache, as_backend
from .utils import signature_union_indices
from .eval_var import compute_var
from .eval_expr import compute_expr
from .eval_compactness import compute_compactness
from .eval_stan import compute_stan
from .compare_metrics import compute_metrics
from .radar_chart import compute_radar


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)) or "dataset"


def run_pipeline(
    gene_sigs_list: dict[str, list[str]],
    names_sigs: list[str],
    mRNA_expr_matrix: dict,
    names_datasets: list[str],
    out_dir: str | Path | None = None,
    thresholds: dict[str, float] | list[float] | None = None,
    verbose: bool = False,
    share_cache: bool = True,
    stats_cache: str | Path | None = None,
) -> dict:
    """Run the sequential radar-metrics pipeline.

    Args:
        gene_sigs_list: dict of signature name -> gene list
        names_sigs: ordered list of signature names
        mRNA_expr_matrix: dict of dataset name -> expression matrix. Each value
            is a DataFrame (genes x samples), an AnnData (cells x genes; in
            memory, backed="r" or lazy) or an ExpressionBackend — e.g.
            SparseBackend(scipy_matrix, gene_names, sample_names). Sparse
            inputs are never densified as a whole: only signature genes are.
        names_datasets: ordered list of dataset names
        out_dir: if not None, write radarchart_table.txt under this directory
        thresholds: expression thresholds per dataset (dict or list, default: median)
        verbose: if True, print progress
        share_cache: wrap every dataset in a DatasetStatsCache so that per-gene
            statistics, the expression threshold and the fetched signature
            genes are computed once and shared by the five modules (default).
            Results are unchanged; only repeated work is removed. False runs
            every module on its own, as the individual compute_* calls do.
        stats_cache: optional directory. Per-gene statistics depend on the
            matrix only, so they are saved there as
            ``<dataset>.gene_stats.npz`` and reused by later runs on the same
            matrix (any signatures), skipping the scans of the whole matrix.
            A file is ignored when its fingerprint (shape, dtype, gene names,
            digest of the first and last vectors) does not match the dataset;
            delete it if a matrix was modified in place. Implies share_cache.

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

    # Resolve every input to a backend once; modules accept backends as is.
    mRNA_expr_matrix = {ds: as_backend(mRNA_expr_matrix[ds]) for ds in names_datasets}
    if share_cache or stats_cache is not None:
        for ds in names_datasets:
            cache = mRNA_expr_matrix[ds]
            if not isinstance(cache, DatasetStatsCache):
                cache = DatasetStatsCache(cache)
            threshold = None if thresholds is None else thresholds[ds]
            stats_file = None
            if stats_cache is not None:
                stats_file = Path(stats_cache) / f"{_safe_name(ds)}.gene_stats.npz"
                cache.load_stats(stats_file)
            seeded = cache.has_stats(threshold)
            cache.prepare(signature_union_indices(cache, gene_sigs_list, names_sigs), threshold)
            if stats_file is not None and not seeded:
                cache.save_stats(stats_file)
            mRNA_expr_matrix[ds] = cache

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
