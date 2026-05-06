"""Tests for pysigqc_metrics.eval_expr — port of R compute_expr()."""

import numpy as np
import pandas as pd
import pytest

from pysigqc_metrics.eval_expr import compute_expr


def test_structure(signatures, datasets, names_sigs, names_datasets):
    result = compute_expr(signatures, names_sigs, datasets, names_datasets)
    assert set(result.keys()) == {"radar_values", "na_proportions", "expr_proportions", "thresholds", "elapsed_seconds"}


def test_2_metrics(signatures, datasets, names_sigs, names_datasets):
    result = compute_expr(signatures, names_sigs, datasets, names_datasets)
    for sig in names_sigs:
        for ds in names_datasets:
            assert set(result["radar_values"][sig][ds].keys()) == {"med_prop_na", "med_prop_above_med"}


def test_values_in_0_1(signatures, datasets, names_sigs, names_datasets):
    result = compute_expr(signatures, names_sigs, datasets, names_datasets)
    for sig in names_sigs:
        for ds in names_datasets:
            for name, val in result["radar_values"][sig][ds].items():
                assert 0.0 <= val <= 1.0, f"{name}={val}"


def test_detects_na(signatures, datasets):
    result = compute_expr(signatures, ["compact_sig"], datasets, ["dataset_A"])
    na_props = result["na_proportions"]["compact_sig"]["dataset_A"]
    assert na_props["gene_missing"] == 1.0


def test_thresholds_positive(signatures, datasets, names_sigs, names_datasets):
    result = compute_expr(signatures, names_sigs, datasets, names_datasets)
    for ds in names_datasets:
        assert result["thresholds"][ds] > 0


def test_custom_thresholds(signatures, datasets, names_sigs, names_datasets):
    custom = {"dataset_A": 3.0, "dataset_B": 5.0}
    result = compute_expr(signatures, names_sigs, datasets, names_datasets, thresholds=custom)
    assert result["thresholds"] == custom


def test_cross_validation(signatures, datasets, names_sigs, names_datasets, ref_dir):
    result = compute_expr(signatures, names_sigs, datasets, names_datasets)
    for sig in names_sigs:
        for ds in names_datasets:
            ref_file = ref_dir / f"expr_radar_{sig}_{ds}.csv"
            if not ref_file.exists():
                pytest.skip(f"No ref: {ref_file}")
            ref = pd.read_csv(ref_file)
            py = result["radar_values"][sig][ds]
            for col in ref.columns:
                np.testing.assert_allclose(py[col], ref[col].iloc[0], rtol=1e-4,
                                           err_msg=f"{col} {sig}/{ds}")
