"""Shared utilities for pysigqc modules."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

# Exact fast paths for the two NumPy/SciPy kernels that dominate run time on
# single-cell sized inputs (see SCALABILITY_REPORT.md). Both return results
# bit-identical to the plain call they replace; PYSIGQC_EXACT_FAST_PATHS=0
# switches them off (used by the benchmarks to measure their effect).
EXACT_FAST_PATHS = os.environ.get("PYSIGQC_EXACT_FAST_PATHS", "1") != "0"


def gene_intersection(signature: list[str], expression_matrix) -> list[str]:
    """Return the intersection of signature genes with matrix row names, preserving order.

    ``expression_matrix`` is a DataFrame (genes x samples) or an ExpressionBackend.
    """
    return [g for g in signature if g in expression_matrix.index]


def signature_union_indices(backend, gene_sigs_list, names_sigs) -> np.ndarray:
    """Dataset row positions of every gene used by at least one signature.

    Lets a module fetch all signature genes from the backend in one go, so
    that genes shared between signatures are read once.
    """
    genes = dict.fromkeys(g for sig in names_sigs for g in gene_sigs_list[sig])
    return backend.gene_indices(genes)


def z_transform(values: np.ndarray) -> np.ndarray:
    """Z-transform an array, returning zeros for zero-variance input."""
    sd = np.nanstd(values, ddof=1)
    if sd == 0 or np.isnan(sd):
        return np.zeros_like(values)
    return (values - np.nanmean(values)) / sd


def nanmedian_over_genes(block: np.ndarray, overwrite: bool = False) -> np.ndarray:
    """``np.nanmedian(block, axis=0)`` for a genes x samples block, faster.

    np.nanmedian goes through masked arrays for short axes; np.median is
    several times faster and gives the same value wherever a column has no
    NaN. Columns containing NaN (np.median returns NaN for them) are redone
    with np.nanmedian, so the result is bit-identical.

    ``overwrite=True`` lets the block be partitioned in place (its values end
    up reordered within each column) instead of working on a K x N copy. Use
    it when the block is not needed afterwards; the result does not change.
    """
    if not EXACT_FAST_PATHS or block.shape[0] == 0 or block.shape[1] == 0:
        return np.nanmedian(block, axis=0)
    med = np.median(block, axis=0, overwrite_input=overwrite)
    redo = np.isnan(med)
    if redo.any():
        med[redo] = np.nanmedian(block[:, redo], axis=0)
    return med


def nanmean_over_genes(block: np.ndarray) -> np.ndarray:
    """``np.nanmean(block, axis=0)`` without its unconditional K x N copy.

    np.nanmean always copies its input to zero the NaNs. Without NaN, np.mean
    runs the very same reduction on the same memory layout, so the values are
    bit-identical; as soon as the result shows a NaN the plain np.nanmean is
    used instead (a column subset would be summed in a different order).
    """
    if not EXACT_FAST_PATHS or block.shape[0] == 0 or block.shape[1] == 0:
        return np.nanmean(block, axis=0)
    mean = np.mean(block, axis=0)
    if np.isnan(mean).any():
        return np.nanmean(block, axis=0)
    return mean


# Average ranks are multiples of 0.5 bounded by the number of samples: float32
# holds them exactly up to 2**23 samples, at half the memory.
_FLOAT32_EXACT_RANKS = 1 << 23


def rank_dtype(n_samples: int):
    """Smallest float dtype that stores average ranks of n_samples exactly."""
    return np.float32 if EXACT_FAST_PATHS and n_samples <= _FLOAT32_EXACT_RANKS else np.float64


def rank_rows(block: np.ndarray) -> np.ndarray:
    """``scipy.stats.rankdata(block, axis=1)`` (average ranks), sparse-aware.

    In a row that is mostly zeros, all zeros form one tie group: only the
    non-zero entries need sorting. With n_neg negatives, n0 zeros and the
    within-non-zero average ranks r:
        negatives -> r,  zeros -> n_neg + (n0 + 1) / 2,  positives -> r + n0
    These are the exact average ranks (integers or half-integers), so the
    result equals rankdata on the full row. Rows containing NaN are all-NaN,
    as with rankdata's default nan_policy.

    Ranks are returned in :func:`rank_dtype` — float32 whenever that is exact —
    and must be widened to float64 before any arithmetic (np.corrcoef does).
    """
    if not EXACT_FAST_PATHS:
        return sp_stats.rankdata(block, axis=1)
    n = block.shape[1]
    out = np.empty(block.shape, dtype=rank_dtype(n))
    for k in range(block.shape[0]):
        row = block[k]
        nz = row != 0                      # NaN counts as non-zero here
        x = row[nz]
        if x.size * 2 > n:                 # dense row: nothing to gain
            out[k] = sp_stats.rankdata(row)
            continue
        if np.isnan(x).any():
            out[k] = np.nan
            continue
        n0 = n - x.size
        r = sp_stats.rankdata(x)
        n_neg = int(np.count_nonzero(x < 0))
        r[x > 0] += n0
        ranks = out[k]
        ranks[:] = n_neg + (n0 + 1) / 2.0
        ranks[nz] = r
    return out
