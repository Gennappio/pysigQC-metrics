"""Evaluate compactness (internal coherence) of gene signatures via autocorrelation.

Port of R_refactored/eval_compactness_loc.R (eval_compactness_loc_noplots).
Produces 1 radar metric: autocor_median (median of Spearman gene-gene correlation matrix).
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


def compute_compactness(
    gene_sigs_list: dict[str, list[str]],
    names_sigs: list[str],
    mRNA_expr_matrix: dict[str, pd.DataFrame],
    names_datasets: list[str],
) -> dict:
    """Compute compactness metrics for each signature-dataset pair.

    Args:
        gene_sigs_list: dict of signature name -> gene list
        names_sigs: list of signature names
        mRNA_expr_matrix: dict of dataset name -> DataFrame (genes x samples)
        names_datasets: list of dataset names

    Returns dict with keys:
        radar_values: nested dict [sig][dataset] -> {"autocor_median": val}
        autocor_matrices: nested dict [sig][dataset] -> gene-gene Spearman correlation matrix
        elapsed_seconds: wall-clock time
    """
    _t0 = time.perf_counter()
    radar_values: dict = {sig: {} for sig in names_sigs}
    autocor_matrices: dict = {sig: {} for sig in names_sigs}

    # Per-dataset cache: drop rows with any NA once (matches R's na.omit on the
    # full-row selection) and pre-rank along samples once. The per-(sig, ds)
    # work below becomes a vectorized row gather + np.corrcoef on ranks.
    ds_cache: dict = {}
    for ds in names_datasets:
        df = mRNA_expr_matrix[ds]
        arr = df.to_numpy(dtype=float)
        keep = ~np.isnan(arr).any(axis=1)
        clean_arr = arr[keep]
        clean_index = df.index[keep]
        if clean_arr.shape[0] > 0:
            clean_ranks = sp_stats.rankdata(clean_arr, axis=1)
        else:
            clean_ranks = clean_arr
        ds_cache[ds] = (clean_index, clean_ranks)

    for sig in names_sigs:
        gene_sig = list(gene_sigs_list[sig])
        for ds in names_datasets:
            clean_index, clean_ranks = ds_cache[ds]

            # Vectorized lookup of signature genes among non-NA dataset rows;
            # genes missing or carrying any NA in the dataset are dropped, as
            # in R's intersect() + na.omit() pipeline.
            idx = clean_index.get_indexer(gene_sig)
            present_idx = idx[idx >= 0]
            n_genes = present_idx.size

            if n_genes > 1:
                sig_ranks = clean_ranks[present_idx]
                autocors = np.corrcoef(sig_ranks)
                np.fill_diagonal(autocors, 1.0)
                autocor_median = float(np.nanmedian(autocors))
            else:
                autocors = np.array([[1.0]])
                autocor_median = 0.0

            genes_present = clean_index[present_idx].tolist()
            autocor_matrices[sig][ds] = pd.DataFrame(
                autocors, index=genes_present, columns=genes_present
            )
            radar_values[sig][ds] = {"autocor_median": autocor_median}

    return {
        "radar_values": radar_values,
        "autocor_matrices": autocor_matrices,
        "elapsed_seconds": time.perf_counter() - _t0,
    }
