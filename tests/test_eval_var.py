"""Tests for pysigqc_metrics.eval_var — port of R compute_var()."""

import numpy as np
import pandas as pd
import pytest

from pysigqc_metrics.eval_var import compute_var


def test_compute_var_structure(signatures, datasets, names_sigs, names_datasets):
    result = compute_var(signatures, names_sigs, datasets, names_datasets)

    assert set(result.keys()) == {"radar_values", "mean_sd_tables", "all_sd", "all_mean", "inter", "elapsed_seconds"}
    for sig in names_sigs:
        assert sig in result["radar_values"]
        for ds in names_datasets:
            assert ds in result["radar_values"][sig]


def test_compute_var_6_metrics(signatures, datasets, names_sigs, names_datasets):
    result = compute_var(signatures, names_sigs, datasets, names_datasets)
    expected = {"sd_median_ratio", "abs_skewness_ratio", "prop_top_10_percent",
                "prop_top_25_percent", "prop_top_50_percent", "coeff_of_var_ratio"}
    for sig in names_sigs:
        for ds in names_datasets:
            assert set(result["radar_values"][sig][ds].keys()) == expected


def test_compute_var_values_in_0_1(signatures, datasets, names_sigs, names_datasets):
    result = compute_var(signatures, names_sigs, datasets, names_datasets)
    for sig in names_sigs:
        for ds in names_datasets:
            for name, val in result["radar_values"][sig][ds].items():
                if not np.isnan(val):
                    assert 0.0 <= val <= 1.0, f"{name}={val} not in [0,1] for {sig}/{ds}"


def test_compute_var_mean_sd_table_dims(signatures, datasets, names_sigs, names_datasets):
    result = compute_var(signatures, names_sigs, datasets, names_datasets)
    for sig in names_sigs:
        for ds in names_datasets:
            tbl = result["mean_sd_tables"][sig][ds]
            inter = result["inter"][sig][ds]
            assert tbl.shape == (len(inter), 2)
            assert list(tbl.columns) == ["Mean", "SD"]


def test_compute_var_missing_gene_excluded(signatures, datasets):
    result = compute_var(signatures, ["compact_sig"], datasets, ["dataset_A"])
    inter = result["inter"]["compact_sig"]["dataset_A"]
    assert "gene_missing" not in inter
    assert len(inter) == 4  # gene_1 through gene_4


def test_compute_var_deterministic(signatures, datasets, names_sigs, names_datasets):
    r1 = compute_var(signatures, names_sigs, datasets, names_datasets)
    r2 = compute_var(signatures, names_sigs, datasets, names_datasets)
    for sig in names_sigs:
        for ds in names_datasets:
            for m in r1["radar_values"][sig][ds]:
                assert r1["radar_values"][sig][ds][m] == r2["radar_values"][sig][ds][m]


def test_compute_var_cross_validation(signatures, datasets, names_sigs, names_datasets, ref_dir):
    """Compare Python output against R reference outputs."""
    result = compute_var(signatures, names_sigs, datasets, names_datasets)

    for sig in names_sigs:
        for ds in names_datasets:
            ref_file = ref_dir / f"var_radar_{sig}_{ds}.csv"
            if not ref_file.exists():
                pytest.skip(f"Reference file not found: {ref_file}")
            ref = pd.read_csv(ref_file)
            py_vals = result["radar_values"][sig][ds]
            for col in ref.columns:
                r_val = ref[col].iloc[0]
                py_val = py_vals[col]
                np.testing.assert_allclose(
                    py_val, r_val, rtol=1e-4, atol=1e-10,
                    err_msg=f"Mismatch for {col} in {sig}/{ds}: R={r_val}, Py={py_val}"
                )
