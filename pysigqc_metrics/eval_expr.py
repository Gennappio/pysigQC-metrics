"""Evaluate expression level properties of gene signatures across datasets.

Port of R_refactored/eval_expr_loc.R — compute_expr() only.
Produces 2 radar metrics: med_prop_na, med_prop_above_med.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .utils import gene_intersection


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
    radar_values: dict = {}
    na_proportions: dict = {}
    expr_proportions: dict = {}

    # --- NA proportion analysis ---
    for sig in names_sigs:
        gene_sig = gene_sigs_list[sig]
        radar_values[sig] = {}
        na_proportions[sig] = {}
        expr_proportions[sig] = {}

        for ds in names_datasets:
            data_matrix = mRNA_expr_matrix[ds]
            inter = gene_intersection(gene_sig, data_matrix)
            genes_expr = data_matrix.loc[inter]
            n_samples = genes_expr.shape[1]

            # Proportion of NA per gene
            gene_na_props = genes_expr.isna().sum(axis=1) / n_samples

            # Genes in signature but not in dataset get NA proportion = 1.0
            missing_genes = [g for g in gene_sig if g not in data_matrix.index]
            for g in missing_genes:
                gene_na_props[g] = 1.0

            # Sort descending (like R's -sort(-x))
            gene_na_props = gene_na_props.sort_values(ascending=False)
            na_proportions[sig][ds] = gene_na_props

            # Radar metric: median proportion of non-NA values
            radar_values[sig][ds] = {}
            radar_values[sig][ds]["med_prop_na"] = float(np.median(1 - gene_na_props.values))

    # --- Compute thresholds ---
    if thresholds is None:
        thresholds = {}
        for ds in names_datasets:
            # R: median(unlist(na.omit(matrix))) — na.omit drops entire rows with any NA
            clean_df = mRNA_expr_matrix[ds].dropna(axis=0, how="any")
            thresholds[ds] = float(np.median(clean_df.values.flatten()))
    elif isinstance(thresholds, (list, np.ndarray)):
        thresholds = dict(zip(names_datasets, thresholds))

    # --- Expression proportion analysis ---
    for sig in names_sigs:
        gene_sig = gene_sigs_list[sig]

        for ds in names_datasets:
            data_matrix = mRNA_expr_matrix[ds]
            inter = gene_intersection(gene_sig, data_matrix)
            genes_expr = data_matrix.loc[inter]
            n_samples = genes_expr.shape[1]
            thresh = thresholds[ds]

            # Proportion of samples above threshold per gene
            # R: rowSums(genes_expr < threshold) propagates NA — if any sample is NA,
            # the entire gene gets NA. Then sort() removes NAs (na.last=NA default).
            below_thresh = genes_expr < thresh  # NaN comparisons → False in pandas
            has_na = genes_expr.isna().any(axis=1)
            gene_expr_props = 1.0 - below_thresh.sum(axis=1) / n_samples
            gene_expr_props[has_na] = np.nan  # match R's NA propagation

            # Missing genes get proportion = 0.0
            missing_genes = [g for g in gene_sig if g not in data_matrix.index]
            for g in missing_genes:
                gene_expr_props[g] = 0.0

            gene_expr_props = gene_expr_props.sort_values(ascending=True)
            expr_proportions[sig][ds] = gene_expr_props

            # R's sort() removes NAs, then median is computed on non-NA values
            radar_values[sig][ds]["med_prop_above_med"] = float(np.nanmedian(gene_expr_props.values))

    return {
        "radar_values": radar_values,
        "na_proportions": na_proportions,
        "expr_proportions": expr_proportions,
        "thresholds": thresholds,
        "elapsed_seconds": time.perf_counter() - _t0,
    }
