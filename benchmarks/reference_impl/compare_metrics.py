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

    # Per-dataset cache: convert to NumPy once and replace non-finite with NaN.
    ds_cache: dict = {}
    for ds in names_datasets:
        dm = mRNA_expr_matrix[ds]
        arr = dm.to_numpy(dtype=float, copy=True)
        np.putmask(arr, ~np.isfinite(arr), np.nan)
        ds_cache[ds] = {
            "arr": arr,
            "index": dm.index,
            "columns": list(dm.columns),
        }

    for sig in names_sigs:
        gene_sig = gene_sigs_list[sig]
        radar_values[sig] = {}
        scores_all[sig] = {}
        pca_results[sig] = {}

        for ds in names_datasets:
            cache = ds_cache[ds]
            arr = cache["arr"]
            ds_index = cache["index"]
            sample_names = cache["columns"]

            # Resolve signature genes against the dataset index in one shot.
            sig_idx = ds_index.get_indexer(gene_sig)
            present_idx = sig_idx[sig_idx >= 0]
            sig_arr = arr[present_idx] if present_idx.size else arr[:0]

            # --- Compute summary scores ---
            n_samples = arr.shape[1]
            if sig_arr.shape[0] > 0:
                med_scores = np.nanmedian(sig_arr, axis=0)
                mean_scores = np.nanmean(sig_arr, axis=0)
            else:
                med_scores = np.full(n_samples, np.nan)
                mean_scores = np.full(n_samples, np.nan)

            # PCA on signature genes (samples as observations, genes as features)
            pca1_scores = None
            props_of_variances = None
            pca_obj = None
            if sig_arr.shape[0] >= 2 and sig_arr.shape[1] >= 2:
                # R: prcomp(t(na.omit(data.matrix[inter,])))
                row_clean = ~np.isnan(sig_arr).any(axis=1)
                sig_clean = sig_arr[row_clean].T  # samples x genes
                if sig_clean.shape[0] >= 2 and sig_clean.shape[1] >= 2:
                    try:
                        pca_obj = PCA()
                        pca1_scores = pca_obj.fit_transform(sig_clean)[:, 0]
                        props_of_variances = pca_obj.explained_variance_ratio_
                    except (np.linalg.LinAlgError, ValueError) as e:
                        # Narrow to the exceptions R's prcomp equivalent can raise
                        # (singular matrix, missing values, degenerate input).
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

            # --- Compute correlations: single rankdata + corrcoef ---
            rho_mean_med = 0.0
            rho_mean_pca1 = 0.0
            rho_pca1_med = 0.0

            if pca1_scores is not None and len(pca1_scores) > 1:
                stacked = np.vstack([mean_scores, med_scores, pca1_scores])
                col_mask = ~np.isnan(stacked).any(axis=0)
                if col_mask.sum() > 1:
                    ranks = sp_stats.rankdata(stacked[:, col_mask], axis=1)
                    cor = np.corrcoef(ranks)
                    rho_mean_med = float(cor[0, 1])
                    rho_mean_pca1 = float(cor[0, 2])
                    rho_pca1_med = float(cor[2, 1])
                    score_cor_mats[f"{ds}_{sig}"] = pd.DataFrame(
                        cor, index=["Mean", "Median", "PCA1"],
                        columns=["Mean", "Median", "PCA1"],
                    )
            elif n_samples > 1:
                stacked = np.vstack([mean_scores, med_scores])
                col_mask = ~np.isnan(stacked).any(axis=0)
                if col_mask.sum() > 1:
                    ranks = sp_stats.rankdata(stacked[:, col_mask], axis=1)
                    cor = np.corrcoef(ranks)
                    rho_mean_med = float(cor[0, 1])
                    score_cor_mats[f"{ds}_{sig}"] = pd.DataFrame(
                        cor, index=["Mean", "Median"],
                        columns=["Mean", "Median"],
                    )

            prop_pca1_var = 0.0
            if props_of_variances is not None and len(props_of_variances) > 0:
                prop_pca1_var = float(props_of_variances[0])

            radar_values[sig][ds] = {
                "rho_mean_med": rho_mean_med,
                "rho_pca1_med": rho_pca1_med,
                "rho_mean_pca1": rho_mean_pca1,
                "prop_pca1_var": prop_pca1_var,
            }

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
