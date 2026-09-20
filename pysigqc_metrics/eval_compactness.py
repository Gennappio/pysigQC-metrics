"""Evaluate compactness (internal coherence) of gene signatures via autocorrelation.

Port of R_refactored/eval_compactness_loc.R (eval_compactness_loc_noplots).
Produces 1 radar metric: autocor_median (median of Spearman gene-gene correlation matrix).
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from .backends import as_backend
from .utils import rank_rows, signature_union_indices


def compute_compactness(
    gene_sigs_list: dict[str, list[str]],
    names_sigs: list[str],
    mRNA_expr_matrix: dict,
    names_datasets: list[str],
) -> dict:
    """Compute compactness metrics for each signature-dataset pair.

    Args:
        gene_sigs_list: dict of signature name -> gene list
        names_sigs: list of signature names
        mRNA_expr_matrix: dict of dataset name -> DataFrame (genes x samples),
            ExpressionBackend or AnnData
        names_datasets: list of dataset names

    Returns dict with keys:
        radar_values: nested dict [sig][dataset] -> {"autocor_median": val}
        autocor_matrices: nested dict [sig][dataset] -> gene-gene Spearman correlation matrix
        elapsed_seconds: wall-clock time
    """
    _t0 = time.perf_counter()
    radar_values: dict = {sig: {} for sig in names_sigs}
    autocor_matrices: dict = {sig: {} for sig in names_sigs}


    # Only signature genes are ever densified: the union of all signatures is
    # fetched from the backend once per dataset (K_union x N at most, never
    # G x N). Ranking is done on-demand with a lazy per-gene cache so that
    # genes shared across signatures are ranked at most once per dataset.
    # Cached ranks are float32 whenever that is exact (see utils.rank_dtype);
    # np.corrcoef widens them to float64 before doing any arithmetic.
    # Genes carrying any NA are dropped, matching R's na.omit on the row
    # selection. Work is O(unique_sig_genes * N log N), memory O(K * N).
    ds_cache: dict = {}
    for ds in names_datasets:
        backend = as_backend(mRNA_expr_matrix[ds])
        _t = time.perf_counter()
        store = backend.load_genes(
            signature_union_indices(backend, gene_sigs_list, names_sigs))
        backend._tick("compactness_extract", _t)
        ds_cache[ds] = {"backend": backend, "store": store,
                        "rank_cache": {}, "has_na": {}}

    for sig in names_sigs:
        gene_sig = list(gene_sigs_list[sig])
        for ds in names_datasets:
            cache = ds_cache[ds]
            backend = cache["backend"]
            rank_cache = cache["rank_cache"]
            has_na = cache["has_na"]

            # Signature genes present in the dataset, in signature order.
            idx = backend.gene_indices(gene_sig)

            new = list(dict.fromkeys(i for i in idx.tolist() if i not in has_na))
            if new:
                _t = time.perf_counter()
                block = cache["store"].dense(np.asarray(new))
                na_rows = np.isnan(block).any(axis=1)
                backend._tick("compactness_extract", _t)
                _t = time.perf_counter()
                new_ranks = rank_rows(block)
                backend._tick("compactness_rank", _t)
                for j, gi in enumerate(new):
                    has_na[gi] = bool(na_rows[j])
                    if not na_rows[j]:
                        rank_cache[gi] = new_ranks[j]
                del block, new_ranks

            present_idx = np.array([i for i in idx.tolist() if not has_na[i]], dtype=np.intp)
            n_genes = present_idx.size

            if n_genes > 1:
                _t = time.perf_counter()
                sig_ranks = np.stack([rank_cache[i] for i in present_idx])
                autocors = np.corrcoef(sig_ranks)
                np.fill_diagonal(autocors, 1.0)
                autocor_median = float(np.nanmedian(autocors))
                backend._tick("compactness_corr", _t)
            else:
                autocors = np.array([[1.0]])
                autocor_median = 0.0

            genes_present = backend.gene_names[present_idx].tolist()
            autocor_matrices[sig][ds] = pd.DataFrame(
                autocors, index=genes_present, columns=genes_present
            )
            radar_values[sig][ds] = {"autocor_median": autocor_median}

    return {
        "radar_values": radar_values,
        "autocor_matrices": autocor_matrices,
        "elapsed_seconds": time.perf_counter() - _t0,
    }
