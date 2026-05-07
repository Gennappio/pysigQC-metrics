"""Compare various signature summary scoring metrics.

Port of R_corrected/compare_metrics_loc_noplots() — radar metrics only.
Computes Mean, Median and PCA1 sample scores and their pairwise Spearman
correlations.

Produces 4 radar metrics: rho_mean_med, rho_pca1_med, rho_mean_pca1, prop_pca1_var.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.decomposition import PCA

from .utils import gene_intersection


def compute_metrics(
    gene_sigs_list: dict[str, list[str]],
    names_sigs: list[str],
    mRNA_expr_matrix: dict[str, pd.DataFrame],
    names_datasets: list[str],
) -> dict:
    """Compute scoring comparison metrics for each signature-dataset pair.

    Args:
        gene_sigs_list: dict of signature name -> gene list
        names_sigs: list of signature names
        mRNA_expr_matrix: dict of dataset name -> DataFrame (genes x samples)
        names_datasets: list of dataset names

    Returns dict with keys:
        radar_values: nested dict [sig][dataset] -> dict of 4 metrics
        scores: nested dict [sig][dataset] -> dict of score arrays
        pca_results: nested dict [sig][dataset] -> dict with pca_obj, props_of_variances
        score_cor_mats: dict of Mean/Median/PCA1 correlation matrices
        elapsed_seconds: wall-clock time
    """
    _t0 = time.perf_counter()
    radar_values: dict = {}
    scores_all: dict = {}
    pca_results: dict = {}
    score_cor_mats: dict = {}

    for sig in names_sigs:
        gene_sig = gene_sigs_list[sig]
        radar_values[sig] = {}
        scores_all[sig] = {}
        pca_results[sig] = {}

        for ds in names_datasets:
            data_matrix = mRNA_expr_matrix[ds].copy()
            # Replace non-finite values with NaN
            data_matrix = data_matrix.where(np.isfinite(data_matrix), other=np.nan)
            inter = gene_intersection(gene_sig, data_matrix)

            sig_data = data_matrix.loc[inter]

            # --- Compute summary scores ---
            med_scores = sig_data.apply(lambda col: np.nanmedian(col.values), axis=0).values
            mean_scores = sig_data.apply(lambda col: np.nanmean(col.values), axis=0).values
            sample_names = list(data_matrix.columns)

            # PCA on signature genes (samples as observations, genes as features)
            pca1_scores = None
            props_of_variances = None
            pca_obj = None
            try:
                # R: prcomp(t(na.omit(data.matrix[inter,])))
                sig_clean = sig_data.dropna(axis=0, how="any").T  # samples x genes
                if sig_clean.shape[1] >= 2 and sig_clean.shape[0] >= 2:
                    pca_obj = PCA()
                    pca_obj.fit(sig_clean.values)
                    pca1_scores = pca_obj.transform(sig_clean.values)[:, 0]
                    props_of_variances = pca_obj.explained_variance_ratio_
            except (np.linalg.LinAlgError, ValueError) as e:
                # Narrow to the exceptions R's prcomp equivalent can raise
                # (singular matrix, missing values, degenerate input). Anything
                # else propagates. Matches R_refactored/compare_metrics_loc.R:52.
                warnings.warn(
                    f"PCA failed for sig={sig!r} ds={ds!r}: {type(e).__name__}: {e}",
                    RuntimeWarning, stacklevel=2,
                )
                pca1_scores = None
                pca_obj = None

            pca_results[sig][ds] = {
                "pca_obj": pca_obj,
                "props_of_variances": props_of_variances,
            }

            # --- Compute correlations ---
            rho_mean_med = 0.0
            rho_mean_pca1 = 0.0
            rho_pca1_med = 0.0

            if len(med_scores) > 1 and len(mean_scores) > 1:
                rho_mean_med, _ = sp_stats.spearmanr(med_scores, mean_scores)

            if pca1_scores is not None and len(pca1_scores) > 1:
                rho_mean_pca1, _ = sp_stats.spearmanr(mean_scores, pca1_scores)
                rho_pca1_med, _ = sp_stats.spearmanr(pca1_scores, med_scores)

            prop_pca1_var = 0.0
            if props_of_variances is not None and len(props_of_variances) > 0:
                prop_pca1_var = float(props_of_variances[0])

            radar_values[sig][ds] = {
                "rho_mean_med": float(rho_mean_med),
                "rho_pca1_med": float(rho_pca1_med),
                "rho_mean_pca1": float(rho_mean_pca1),
                "prop_pca1_var": prop_pca1_var,
            }

            # --- Build scoring correlation matrix ---
            score_cols = {"Mean": mean_scores, "Median": med_scores}
            if pca1_scores is not None:
                score_cols["PCA1"] = pca1_scores

            if len(score_cols) >= 2:
                score_df = pd.DataFrame(score_cols)
                cor_mat = score_df.corr(method="spearman")
                score_cor_mats[f"{ds}_{sig}"] = cor_mat

            # --- Store scores ---
            scores_all[sig][ds] = {
                "med_scores": med_scores,
                "mean_scores": mean_scores,
                "pca1_scores": pca1_scores,
                "common_score_cols": sample_names,
                "props_of_variances": props_of_variances,
            }

    return {
        "radar_values": radar_values,
        "scores": scores_all,
        "pca_results": pca_results,
        "score_cor_mats": score_cor_mats,
        "elapsed_seconds": time.perf_counter() - _t0,
    }
