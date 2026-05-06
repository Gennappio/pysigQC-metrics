"""Evaluate expression variability of signature genes relative to all genes.

Port of R_refactored/eval_var_loc.R — compute_var() only (no plotting).
Produces 6 radar metrics: sd_median_ratio, abs_skewness_ratio,
prop_top_10_percent, prop_top_25_percent, prop_top_50_percent, coeff_of_var_ratio.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from .utils import gene_intersection


def compute_var(
    gene_sigs_list: dict[str, list[str]],
    names_sigs: list[str],
    mRNA_expr_matrix: dict[str, pd.DataFrame],
    names_datasets: list[str],
) -> dict:
    """Compute variability metrics for each signature-dataset pair.

    Returns dict with keys:
        radar_values: nested dict [sig][dataset] -> dict of 6 metric values
        mean_sd_tables: nested dict [sig][dataset] -> DataFrame (genes x [Mean, SD])
        all_sd: nested dict [sig][dataset] -> Series of per-gene SD (all genes)
        all_mean: nested dict [sig][dataset] -> Series of per-gene mean (all genes)
        inter: nested dict [sig][dataset] -> list of intersected gene names
        elapsed_seconds: wall-clock time
    """
    _t0 = time.perf_counter()
    radar_values: dict = {}
    mean_sd_tables: dict = {}
    all_sd: dict = {}
    all_mean: dict = {}
    inter_genes: dict = {}

    for sig in names_sigs:
        gene_sig = gene_sigs_list[sig]
        radar_values[sig] = {}
        mean_sd_tables[sig] = {}
        all_sd[sig] = {}
        all_mean[sig] = {}
        inter_genes[sig] = {}

        for ds in names_datasets:
            data_matrix = mRNA_expr_matrix[ds]
            inter = gene_intersection(gene_sig, data_matrix)
            inter_genes[sig][ds] = inter

            # Per-gene SD and mean across all samples (ddof=1 to match R's sd())
            sd_genes = data_matrix.apply(lambda row: np.nanstd(row.values.astype(float), ddof=1), axis=1)
            mean_genes = data_matrix.apply(lambda row: np.nanmean(row.values.astype(float)), axis=1)

            all_sd[sig][ds] = sd_genes
            all_mean[sig][ds] = mean_genes

            sd_sig = sd_genes.loc[inter].dropna()
            sd_all = sd_genes.dropna()

            # Metric 1: sd_median_ratio = median(sig_sd) / (median(all_sd) + median(sig_sd))
            med_sig_sd = np.median(sd_sig.values)
            med_all_sd = np.median(sd_all.values)
            sd_median_ratio = med_sig_sd / (med_all_sd + med_sig_sd) if (med_all_sd + med_sig_sd) != 0 else 0.0

            # Metric 2: abs_skewness_ratio
            mean_sig = mean_genes.loc[inter].dropna().values
            mean_all = mean_genes.dropna().values
            skew_sig = abs(sp_stats.skew(mean_sig, bias=True))
            skew_all = abs(sp_stats.skew(mean_all, bias=True))
            abs_skewness_ratio = skew_sig / (skew_all + skew_sig) if (skew_all + skew_sig) != 0 else 0.0

            # Mean/SD table for this sig-dataset pair
            mean_sd_tables[sig][ds] = pd.DataFrame(
                {"Mean": mean_genes.loc[inter], "SD": sd_genes.loc[inter]},
                index=inter,
            )

            # Coefficient of variation for all genes
            coeff_of_var = sd_genes / mean_genes
            cv_sig = coeff_of_var.loc[inter].dropna().values
            cv_all = coeff_of_var.dropna().values

            # Metrics 3-5: proportion of sig genes in top quantiles of all-gene CV
            q90, q75, q50 = np.nanquantile(cv_all, [0.9, 0.75, 0.5])
            n_sig = len(cv_sig)
            prop_top_10 = np.sum(cv_sig >= q90) / n_sig if n_sig > 0 else 0.0
            prop_top_25 = np.sum(cv_sig >= q75) / n_sig if n_sig > 0 else 0.0
            prop_top_50 = np.sum(cv_sig >= q50) / n_sig if n_sig > 0 else 0.0

            # Metric 6: coeff_of_var_ratio
            med_cv_sig = abs(np.nanmedian(cv_sig))
            med_cv_all = abs(np.nanmedian(cv_all))
            coeff_of_var_ratio = med_cv_sig / (med_cv_all + med_cv_sig) if (med_cv_all + med_cv_sig) != 0 else 0.0

            radar_values[sig][ds] = {
                "sd_median_ratio": sd_median_ratio,
                "abs_skewness_ratio": abs_skewness_ratio,
                "prop_top_10_percent": prop_top_10,
                "prop_top_25_percent": prop_top_25,
                "prop_top_50_percent": prop_top_50,
                "coeff_of_var_ratio": coeff_of_var_ratio,
            }

    return {
        "radar_values": radar_values,
        "mean_sd_tables": mean_sd_tables,
        "all_sd": all_sd,
        "all_mean": all_mean,
        "inter": inter_genes,
        "elapsed_seconds": time.perf_counter() - _t0,
    }
