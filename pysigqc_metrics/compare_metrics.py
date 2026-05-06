"""Compare various signature summary scoring metrics.

Port of R_refactored/compare_metrics_loc.R — compute_metrics() only.
Computes Mean, Median, PCA1 scores and their Spearman correlations.
Also computes GSVA/ssGSEA/PLAGE enrichment scores (optional) and
Gaussian mixture models.

Produces 4 radar metrics: rho_mean_med, rho_pca1_med, rho_mean_pca1, prop_pca1_var.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from .utils import gene_intersection


# ============================================================================
# Enrichment score methods (ssGSEA, PLAGE)
# ============================================================================

def _ssgsea_score(expr_matrix: np.ndarray, gene_indices: list[int]) -> np.ndarray:
    """Compute ssGSEA scores for a gene set.

    Single-sample GSEA: for each sample, rank genes, compute enrichment score
    based on cumulative sum of ranks for genes in the set vs. genes outside.

    Args:
        expr_matrix: genes x samples array
        gene_indices: indices of genes in the signature

    Returns:
        Array of scores (one per sample)
    """
    n_genes, n_samples = expr_matrix.shape
    n_sig = len(gene_indices)

    if n_sig == 0 or n_sig >= n_genes:
        return np.full(n_samples, np.nan)

    scores = np.zeros(n_samples)
    sig_set = set(gene_indices)

    for j in range(n_samples):
        col = expr_matrix[:, j]
        # Rank genes by expression (descending order for GSEA)
        valid_mask = np.isfinite(col)
        if valid_mask.sum() < 2:
            scores[j] = np.nan
            continue

        # Get ranks (1 = highest expression)
        ranks = sp_stats.rankdata(-col[valid_mask], method='average')
        valid_indices = np.where(valid_mask)[0]

        # Weighted cumulative sum (power = 1 for ssGSEA)
        in_set = np.array([i in sig_set for i in valid_indices])
        n_in = in_set.sum()
        n_out = len(valid_indices) - n_in

        if n_in == 0 or n_out == 0:
            scores[j] = np.nan
            continue

        # Sort by rank to compute running sum
        sorted_idx = np.argsort(ranks)
        in_set_sorted = in_set[sorted_idx]

        # ES = max deviation from zero
        p_hit = np.cumsum(in_set_sorted / n_in)
        p_miss = np.cumsum(~in_set_sorted / n_out)
        running_es = p_hit - p_miss

        # Use sum of positive and negative deviations (normalized)
        scores[j] = np.sum(running_es) / len(running_es)

    return scores


def _plage_score(expr_matrix: np.ndarray, gene_indices: list[int]) -> np.ndarray:
    """Compute PLAGE scores for a gene set.

    Pathway Level Analysis of Gene Expression: SVD-based method that computes
    the first singular vector (metagene) of the gene set expression matrix.

    Args:
        expr_matrix: genes x samples array
        gene_indices: indices of genes in the signature

    Returns:
        Array of scores (one per sample)
    """
    if len(gene_indices) < 2:
        return np.full(expr_matrix.shape[1], np.nan)

    sig_expr = expr_matrix[gene_indices, :]

    # Remove rows with any NaN
    valid_rows = ~np.any(np.isnan(sig_expr), axis=1)
    sig_expr_clean = sig_expr[valid_rows]

    if sig_expr_clean.shape[0] < 2:
        return np.full(expr_matrix.shape[1], np.nan)

    # Z-score normalize per gene (row)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        row_means = np.nanmean(sig_expr_clean, axis=1, keepdims=True)
        row_stds = np.nanstd(sig_expr_clean, axis=1, keepdims=True)
        row_stds[row_stds == 0] = 1  # Avoid division by zero
        sig_z = (sig_expr_clean - row_means) / row_stds

    # SVD - first right singular vector is the metagene
    try:
        U, S, Vt = np.linalg.svd(sig_z, full_matrices=False)
        # V[:,0] = first metagene (sample loadings)
        return Vt[0, :]
    except np.linalg.LinAlgError:
        return np.full(expr_matrix.shape[1], np.nan)


def _gsva_score(expr_matrix: np.ndarray, gene_indices: list[int]) -> np.ndarray:
    """Compute GSVA-like scores for a gene set.

    Gene Set Variation Analysis: similar to ssGSEA but with kernel density
    estimation. This is a simplified version using empirical CDF.

    Args:
        expr_matrix: genes x samples array
        gene_indices: indices of genes in the signature

    Returns:
        Array of scores (one per sample)
    """
    n_genes, n_samples = expr_matrix.shape
    n_sig = len(gene_indices)

    if n_sig == 0 or n_sig >= n_genes:
        return np.full(n_samples, np.nan)

    scores = np.zeros(n_samples)
    sig_set = set(gene_indices)

    for j in range(n_samples):
        col = expr_matrix[:, j]
        valid_mask = np.isfinite(col)
        if valid_mask.sum() < 2:
            scores[j] = np.nan
            continue

        valid_col = col[valid_mask]
        valid_indices = np.where(valid_mask)[0]

        # Compute ECDF ranks
        ranks = sp_stats.rankdata(valid_col, method='average') / len(valid_col)

        # Separate in-set and out-of-set ranks
        in_set = np.array([i in sig_set for i in valid_indices])

        if in_set.sum() == 0 or (~in_set).sum() == 0:
            scores[j] = np.nan
            continue

        # Score = difference between mean ranks
        scores[j] = np.mean(ranks[in_set]) - np.mean(ranks[~in_set])

    return scores


def compute_metrics(
    gene_sigs_list: dict[str, list[str]],
    names_sigs: list[str],
    mRNA_expr_matrix: dict[str, pd.DataFrame],
    names_datasets: list[str],
    compute_enrichment: bool = True,
    fit_mixture: bool = True,
) -> dict:
    """Compute scoring comparison metrics for each signature-dataset pair.

    Args:
        gene_sigs_list: dict of signature name -> gene list
        names_sigs: list of signature names
        mRNA_expr_matrix: dict of dataset name -> DataFrame (genes x samples)
        names_datasets: list of dataset names
        compute_enrichment: if True, compute GSVA/ssGSEA/PLAGE scores
        fit_mixture: if True, fit Gaussian mixture models on scoring arrays

    Returns dict with keys:
        radar_values: nested dict [sig][dataset] -> dict of 4 metrics
        scores: nested dict [sig][dataset] -> dict of score arrays
        pca_results: nested dict [sig][dataset] -> dict with pca_obj, props_of_variances
        score_cor_mats: dict of scoring correlation matrices (including enrichment)
        mixture_models: nested dict [sig][dataset] -> dict with model results
        enrichment_scores: nested dict [sig][dataset] -> dict with gsva/ssgsea/plage arrays
        elapsed_seconds: wall-clock time
    """
    _t0 = time.perf_counter()
    radar_values: dict = {}
    scores_all: dict = {}
    pca_results: dict = {}
    score_cor_mats: dict = {}
    mixture_models: dict = {}
    enrichment_scores: dict = {}

    for sig in names_sigs:
        gene_sig = gene_sigs_list[sig]
        radar_values[sig] = {}
        scores_all[sig] = {}
        pca_results[sig] = {}
        mixture_models[sig] = {}
        enrichment_scores[sig] = {}

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

            # --- Compute enrichment scores (GSVA, ssGSEA, PLAGE) ---
            gsva_scores = ssgsea_scores = plage_scores = None
            if compute_enrichment and len(inter) >= 2:
                # Get gene indices for the full expression matrix
                expr_arr = data_matrix.values.astype(float)
                all_genes = list(data_matrix.index)
                sig_indices = [all_genes.index(g) for g in inter if g in all_genes]

                if len(sig_indices) >= 2:
                    gsva_scores = _gsva_score(expr_arr, sig_indices)
                    ssgsea_scores = _ssgsea_score(expr_arr, sig_indices)
                    plage_scores = _plage_score(expr_arr, sig_indices)

            enrichment_scores[sig][ds] = {
                "gsva": gsva_scores,
                "ssgsea": ssgsea_scores,
                "plage": plage_scores,
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

            # --- Mixture models (using sklearn instead of mclust) ---
            mm = {"median": None, "mean": None, "pca1": None}
            if fit_mixture:
                for score_name, score_arr in [("median", med_scores), ("mean", mean_scores),
                                               ("pca1", pca1_scores)]:
                    if score_arr is not None and len(score_arr) > 1:
                        clean = score_arr[np.isfinite(score_arr)]
                        if len(clean) >= 2:
                            max_k = min(len(clean) // 2, 10)
                            max_k = max(max_k, 1)
                            best_bic = np.inf
                            best_model = None
                            bic_values = []
                            for k in range(1, max_k + 1):
                                try:
                                    gm = GaussianMixture(n_components=k, random_state=42)
                                    gm.fit(clean.reshape(-1, 1))
                                    bic = gm.bic(clean.reshape(-1, 1))
                                    bic_values.append((k, bic))
                                    if bic < best_bic:
                                        best_bic = bic
                                        best_model = gm
                                except (ValueError, np.linalg.LinAlgError) as e:
                                    # BIC sweep: individual k-value fits can legitimately
                                    # fail on degenerate data; other k values may still
                                    # succeed. Only narrow numerical failures are skipped.
                                    warnings.warn(
                                        f"GaussianMixture(n_components={k}) failed for "
                                        f"sig={sig!r} ds={ds!r} score={score_name!r}: "
                                        f"{type(e).__name__}: {e}",
                                        RuntimeWarning, stacklevel=2,
                                    )
                            mm[score_name] = {
                                "best_model": best_model,
                                "bic_values": bic_values,
                                "best_k": best_model.n_components if best_model else None,
                            }
            mixture_models[sig][ds] = mm

            # --- Build scoring correlation matrix ---
            score_cols = {"Mean": mean_scores, "Median": med_scores}
            if pca1_scores is not None:
                score_cols["PCA1"] = pca1_scores
            # Add enrichment scores if available
            if gsva_scores is not None and np.isfinite(gsva_scores).any():
                score_cols["GSVA"] = gsva_scores
            if ssgsea_scores is not None and np.isfinite(ssgsea_scores).any():
                score_cols["ssGSEA"] = ssgsea_scores
            if plage_scores is not None and np.isfinite(plage_scores).any():
                score_cols["PLAGE"] = plage_scores

            if len(score_cols) >= 2:
                score_df = pd.DataFrame(score_cols)
                cor_mat = score_df.corr(method="spearman")
                score_cor_mats[f"{ds}_{sig}"] = cor_mat

            # --- Store scores ---
            scores_all[sig][ds] = {
                "med_scores": med_scores,
                "mean_scores": mean_scores,
                "pca1_scores": pca1_scores,
                "gsva_scores": gsva_scores,
                "ssgsea_scores": ssgsea_scores,
                "plage_scores": plage_scores,
                "common_score_cols": sample_names,
                "props_of_variances": props_of_variances,
            }

    return {
        "radar_values": radar_values,
        "scores": scores_all,
        "pca_results": pca_results,
        "score_cor_mats": score_cor_mats,
        "mixture_models": mixture_models,
        "enrichment_scores": enrichment_scores,
        "elapsed_seconds": time.perf_counter() - _t0,
    }
