"""Evaluate expression level properties of gene signatures across datasets.

Port of R_refactored/eval_expr_loc.R — compute_expr() only.
Produces 2 radar metrics: med_prop_na, med_prop_above_med.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd


def compute_expr(
    gene_sigs_list: dict[str, list[str]],
    names_sigs: list[str],
    mRNA_expr_matrix: dict[str, pd.DataFrame],
    names_datasets: list[str],
    thresholds: dict[str, float] | None = None,
) -> dict:
    """Compute expression-level metrics for each signature-dataset pair.

    Returns dict with keys:
        radar_values: nested dict [sig][dataset] -> dict of 2 metrics
        na_proportions: nested dict [sig][dataset] -> Series of per-gene NA proportions
        expr_proportions: nested dict [sig][dataset] -> Series of per-gene expression proportions
        thresholds: dict of per-dataset expression thresholds
        elapsed_seconds: wall-clock time
    """
    _t0 = time.perf_counter()
    radar_values: dict = {sig: {} for sig in names_sigs}
    na_proportions: dict = {sig: {} for sig in names_sigs}
    expr_proportions: dict = {sig: {} for sig in names_sigs}

    # Per-gene NA / above-threshold proportions depend only on the dataset.
    # Compute them once per dataset against the whole expression matrix; the
    # per-(signature, dataset) work below becomes a vectorized indexed gather.
    if isinstance(thresholds, (list, np.ndarray)):
        thresholds = dict(zip(names_datasets, thresholds))
    compute_thresholds = thresholds is None
    if compute_thresholds:
        thresholds = {}

    ds_cache: dict = {}
    for ds in names_datasets:
        df = mRNA_expr_matrix[ds]
        arr = df.to_numpy(dtype=float)
        n_samples = arr.shape[1]
        nan_mask = np.isnan(arr)
        has_na_full = nan_mask.any(axis=1)
        na_props_full = nan_mask.sum(axis=1) / n_samples

        # R: median(unlist(na.omit(matrix))) — na.omit drops rows with any NA.
        if compute_thresholds:
            clean = arr[~has_na_full]
            thresh = float(np.median(clean)) if clean.size else float("nan")
            thresholds[ds] = thresh
        else:
            thresh = float(thresholds[ds])

        # NaN comparison is False in numpy; rows with any NaN get NaN propagated
        # afterwards to match R's rowSums(genes_expr < threshold) semantics.
        below_count = (arr < thresh).sum(axis=1)
        expr_props_full = 1.0 - below_count / n_samples
        expr_props_full = np.where(has_na_full, np.nan, expr_props_full)

        ds_cache[ds] = (df.index, na_props_full, expr_props_full)

    for sig in names_sigs:
        gene_sig = list(gene_sigs_list[sig])
        for ds in names_datasets:
            ds_index, na_props_full, expr_props_full = ds_cache[ds]

            # Vectorized signature -> dataset row lookup; -1 marks missing genes.
            idx = ds_index.get_indexer(gene_sig)
            present = idx >= 0
            present_idx = idx[present]

            # Missing genes default to NA-prop = 1.0 and expr-prop = 0.0 (matches R).
            na_vals = np.ones(len(gene_sig))
            na_vals[present] = na_props_full[present_idx]
            expr_vals = np.zeros(len(gene_sig))
            expr_vals[present] = expr_props_full[present_idx]

            # Radar metrics are order-independent; compute before sorting.
            radar_values[sig][ds] = {
                "med_prop_na": float(np.median(1.0 - na_vals)),
                "med_prop_above_med": float(np.nanmedian(expr_vals)),
            }

            # Sorted Series for downstream display, matching R's sort order.
            na_proportions[sig][ds] = pd.Series(na_vals, index=gene_sig).sort_values(ascending=False)
            expr_proportions[sig][ds] = pd.Series(expr_vals, index=gene_sig).sort_values(ascending=True)

    return {
        "radar_values": radar_values,
        "na_proportions": na_proportions,
        "expr_proportions": expr_proportions,
        "thresholds": thresholds,
        "elapsed_seconds": time.perf_counter() - _t0,
    }
