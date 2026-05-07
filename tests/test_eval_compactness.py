"""Tests for pysigqc_metrics.eval_compactness — port of R compute_compactness()."""

import numpy as np
import pandas as pd
import pytest

from pysigqc_metrics.eval_compactness import compute_compactness


def test_structure(signatures, datasets, names_sigs, names_datasets):
    result = compute_compactness(signatures, names_sigs, datasets, names_datasets)
    assert set(result.keys()) == {"radar_values", "autocor_matrices", "elapsed_seconds"}


def test_autocor_median_present(signatures, datasets, names_sigs, names_datasets):
    result = compute_compactness(signatures, names_sigs, datasets, names_datasets)
    for sig in names_sigs:
        for ds in names_datasets:
            assert "autocor_median" in result["radar_values"][sig][ds]


def test_compact_higher_than_diffuse(signatures, datasets, names_datasets):
    result = compute_compactness(signatures, ["compact_sig", "diffuse_sig"], datasets, names_datasets)
    for ds in names_datasets:
        c = result["radar_values"]["compact_sig"][ds]["autocor_median"]
        d = result["radar_values"]["diffuse_sig"][ds]["autocor_median"]
        assert c > d, f"compact={c} should be > diffuse={d} for {ds}"


def test_autocor_matrix_symmetric(signatures, datasets, names_sigs, names_datasets):
    result = compute_compactness(signatures, names_sigs, datasets, names_datasets)
    for sig in names_sigs:
        for ds in names_datasets:
            acm = result["autocor_matrices"][sig][ds].values
            np.testing.assert_allclose(acm, acm.T, atol=1e-10)
            np.testing.assert_allclose(np.diag(acm), 1.0, atol=1e-10)


def test_autocor_values_bounded(signatures, datasets, names_sigs, names_datasets):
    result = compute_compactness(signatures, names_sigs, datasets, names_datasets)
    for sig in names_sigs:
        for ds in names_datasets:
            acm = result["autocor_matrices"][sig][ds].values
            finite = acm[np.isfinite(acm)]
            assert np.all((finite >= -1) & (finite <= 1))


def test_cross_validation(signatures, datasets, names_sigs, names_datasets, ref_dir):
    result = compute_compactness(signatures, names_sigs, datasets, names_datasets)
    for sig in names_sigs:
        for ds in names_datasets:
            ref_file = ref_dir / f"compact_radar_{sig}_{ds}.csv"
            if not ref_file.exists():
                pytest.skip(f"No ref: {ref_file}")
            ref = pd.read_csv(ref_file)
            py = result["radar_values"][sig][ds]
            for col in ref.columns:
                np.testing.assert_allclose(py[col], ref[col].iloc[0], rtol=1e-4,
                                           err_msg=f"{col} {sig}/{ds}")
