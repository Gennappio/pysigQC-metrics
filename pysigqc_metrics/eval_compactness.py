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
    # full-row selection). Ranking is done on-demand, only on the K signature
    # rows, with a lazy per-gene cache so that genes shared across signatures
    # are ranked at most once per dataset. This keeps memory at O(K * N) and
    # work at O(unique_sig_genes * N log N) instead of O(G * N log N), which
    # matters when G >> K (single-cell scale).
    ds_cache: dict = {}
    for ds in names_datasets:
        df = mRNA_expr_matrix[ds]
        arr = df.to_numpy(dtype=float)
        keep = ~np.isnan(arr).any(axis=1)
        ds_cache[ds] = {
            "index": df.index[keep],
            "arr": arr[keep],
            "rank_cache": {},
        }

    for sig in names_sigs:
        gene_sig = list(gene_sigs_list[sig])
        for ds in names_datasets:
            cache = ds_cache[ds]
            clean_index = cache["index"]
            clean_arr = cache["arr"]
            rank_cache = cache["rank_cache"]

            # Vectorized lookup of signature genes among non-NA dataset rows;
            # genes missing or carrying any NA in the dataset are dropped, as
            # in R's intersect() + na.omit() pipeline.
            idx = clean_index.get_indexer(gene_sig)
            present_idx = idx[idx >= 0]
            n_genes = present_idx.size

            if n_genes > 1:
                missing = [i for i in present_idx.tolist() if i not in rank_cache]
                if missing:
                    new_ranks = sp_stats.rankdata(clean_arr[missing], axis=1)
                    for j, gi in enumerate(missing):
                        rank_cache[gi] = new_ranks[j]
                sig_ranks = np.stack([rank_cache[i] for i in present_idx])
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
