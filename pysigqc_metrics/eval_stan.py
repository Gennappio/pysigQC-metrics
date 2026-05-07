"""Evaluate the effect of z-score standardization on signature scoring.

Port of R_refactored/eval_stan_loc.R — compute_stan() only.
Produces 1 radar metric: standardization_comp (Spearman rho between
raw median scores and z-transformed median scores).
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from .utils import gene_intersection


def compute_stan(
    gene_sigs_list: dict[str, list[str]],
    names_sigs: list[str],
    mRNA_expr_matrix: dict[str, pd.DataFrame],
    names_datasets: list[str],
) -> dict:
    """Compute standardization comparison metrics.

    Returns dict with keys:
        radar_values: nested dict [sig][dataset] -> {"standardization_comp": rho}
        med_scores: nested dict [sig][dataset] -> array of raw median scores per sample
        z_transf_scores: nested dict [sig][dataset] -> array of z-transformed median scores
        elapsed_seconds: wall-clock time
    """
    _t0 = time.perf_counter()
    radar_values: dict = {}
    med_scores_all: dict = {}
    z_transf_scores_all: dict = {}

    for sig in names_sigs:
        gene_sig = gene_sigs_list[sig]
        radar_values[sig] = {}
        med_scores_all[sig] = {}
        z_transf_scores_all[sig] = {}

        for ds in names_datasets:
            data_matrix = mRNA_expr_matrix[ds]
            inter = gene_intersection(gene_sig, data_matrix)

            # Z-transform each gene row in one vectorized pass (with
            # zero-variance / all-NaN guard producing zeros, matching
            # utils.z_transform applied per-row).
            sig_data = data_matrix.loc[inter].astype(float)
            arr = sig_data.to_numpy()
            mu = np.nanmean(arr, axis=1, keepdims=True)
            sd = np.nanstd(arr, axis=1, ddof=1, keepdims=True)
            bad = (sd == 0) | np.isnan(sd)
            sd_safe = np.where(bad, 1.0, sd)
            z_arr = np.where(bad, 0.0, (arr - mu) / sd_safe)

            # Median across genes for each sample
            z_transf_scores = np.nanmedian(z_arr, axis=0)
            med_scores = np.nanmedian(arr, axis=0)

            # Spearman correlation between raw and z-transformed scores
            rho, _ = sp_stats.spearmanr(med_scores, z_transf_scores)

            radar_values[sig][ds] = {"standardization_comp": float(rho)}
            med_scores_all[sig][ds] = med_scores
            z_transf_scores_all[sig][ds] = z_transf_scores

    return {
        "radar_values": radar_values,
        "med_scores": med_scores_all,
        "z_transf_scores": z_transf_scores_all,
        "elapsed_seconds": time.perf_counter() - _t0,
    }
