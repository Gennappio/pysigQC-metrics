"""Shared utilities for pysigqc modules."""

from __future__ import annotations

import numpy as np
import pandas as pd


def gene_intersection(
    signature: list[str], expression_matrix: pd.DataFrame
) -> list[str]:
    """Return the intersection of signature genes with matrix row names, preserving order."""
    return [g for g in signature if g in expression_matrix.index]


def z_transform(values: np.ndarray) -> np.ndarray:
    """Z-transform an array, returning zeros for zero-variance input."""
    sd = np.nanstd(values, ddof=1)
    if sd == 0 or np.isnan(sd):
        return np.zeros_like(values)
    return (values - np.nanmean(values)) / sd
