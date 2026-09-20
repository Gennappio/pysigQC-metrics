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

from .backends import as_backend
from .utils import nanmedian_over_genes, signature_union_indices


def compute_stan(
    gene_sigs_list: dict[str, list[str]],
    names_sigs: list[str],
    mRNA_expr_matrix: dict,
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

    # Fetch the union of all signature genes once per dataset; only the
    # current signature's genes are densified (K x N).
    ds_cache: dict = {}
    for ds in names_datasets:
        backend = as_backend(mRNA_expr_matrix[ds])
        _t = time.perf_counter()
        store = backend.load_genes(
            signature_union_indices(backend, gene_sigs_list, names_sigs))
        backend._tick("stan_extract", _t)
        ds_cache[ds] = (backend, store)

    for sig in names_sigs:
        gene_sig = gene_sigs_list[sig]
        radar_values[sig] = {}
        med_scores_all[sig] = {}
        z_transf_scores_all[sig] = {}

        for ds in names_datasets:
            backend, store = ds_cache[ds]
            _t = time.perf_counter()
            arr = store.dense(backend.gene_indices(gene_sig), order="K")
            backend._tick("stan_extract", _t)

            # Z-transform each gene row in one vectorized pass (with
            # zero-variance / all-NaN guard producing zeros, matching
            # utils.z_transform applied per-row).
            _t = time.perf_counter()
            mu = np.nanmean(arr, axis=1, keepdims=True)
            sd = np.nanstd(arr, axis=1, ddof=1, keepdims=True)
            bad = (sd == 0) | np.isnan(sd)
            sd_safe = np.where(bad, 1.0, sd)
            # Same values as np.where(bad, 0.0, (arr - mu) / sd_safe), built in
            # one K x N buffer instead of three.
            z_arr = arr - mu
            z_arr /= sd_safe
            z_arr[bad[:, 0]] = 0.0

            # Median across genes for each sample (both blocks are dead after
            # this, so they are partitioned in place).
            z_transf_scores = nanmedian_over_genes(z_arr, overwrite=True)
            del z_arr
            med_scores = nanmedian_over_genes(arr, overwrite=True)
            del arr
            backend._tick("stan_scores", _t)

            # Spearman correlation between raw and z-transformed scores
            _t = time.perf_counter()
            rho, _ = sp_stats.spearmanr(med_scores, z_transf_scores)
            backend._tick("stan_spearman", _t)

            radar_values[sig][ds] = {"standardization_comp": float(rho)}
            med_scores_all[sig][ds] = med_scores
            z_transf_scores_all[sig][ds] = z_transf_scores

    return {
        "radar_values": radar_values,
        "med_scores": med_scores_all,
        "z_transf_scores": z_transf_scores_all,
        "elapsed_seconds": time.perf_counter() - _t0,
    }
