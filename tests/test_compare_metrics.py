"""Tests for pysigqc_metrics.compare_metrics — port of R compute_metrics()."""

import numpy as np
import pandas as pd
import pytest

from pysigqc_metrics.compare_metrics import compute_metrics


def test_structure(signatures, datasets, names_sigs, names_datasets):
    result = compute_metrics(signatures, names_sigs, datasets, names_datasets)
    assert set(result.keys()) == {"radar_values", "scores", "pca_results",
                                   "score_cor_mats", "mixture_models",
                                   "enrichment_scores", "elapsed_seconds"}


def test_4_radar_metrics(signatures, datasets, names_sigs, names_datasets):
    result = compute_metrics(signatures, names_sigs, datasets, names_datasets)
    expected = {"rho_mean_med", "rho_pca1_med", "rho_mean_pca1", "prop_pca1_var"}
    for sig in names_sigs:
        for ds in names_datasets:
            assert set(result["radar_values"][sig][ds].keys()) == expected


def test_correlation_values_bounded(signatures, datasets, names_sigs, names_datasets):
    result = compute_metrics(signatures, names_sigs, datasets, names_datasets)
    for sig in names_sigs:
        for ds in names_datasets:
            rv = result["radar_values"][sig][ds]
            for m in ["rho_mean_med", "rho_pca1_med", "rho_mean_pca1"]:
                val = rv[m]
                if not np.isnan(val):
                    assert -1.0 <= val <= 1.0, f"{m}={val}"


def test_prop_pca1_var_bounded(signatures, datasets, names_sigs, names_datasets):
    result = compute_metrics(signatures, names_sigs, datasets, names_datasets)
    for sig in names_sigs:
        for ds in names_datasets:
            val = result["radar_values"][sig][ds]["prop_pca1_var"]
            assert 0.0 <= val <= 1.0


def test_score_lengths(signatures, datasets, names_sigs, names_datasets):
    result = compute_metrics(signatures, names_sigs, datasets, names_datasets)
    for sig in names_sigs:
        for ds in names_datasets:
            sc = result["scores"][sig][ds]
            assert len(sc["med_scores"]) == 10
            assert len(sc["mean_scores"]) == 10


def test_pca_variance_sums_to_1(signatures, datasets, names_sigs, names_datasets):
    result = compute_metrics(signatures, names_sigs, datasets, names_datasets)
    for sig in names_sigs:
        for ds in names_datasets:
            pca = result["pca_results"][sig][ds]
            if pca["props_of_variances"] is not None:
                np.testing.assert_allclose(sum(pca["props_of_variances"]), 1.0, atol=1e-10)


def test_cross_validation(signatures, datasets, names_sigs, names_datasets, ref_dir):
    """Compare Python radar values against R reference outputs.

    Note: PCA sign ambiguity may flip rho_pca1_med and rho_mean_pca1,
    so we compare absolute values for those.
    """
    result = compute_metrics(signatures, names_sigs, datasets, names_datasets)
    for sig in names_sigs:
        for ds in names_datasets:
            ref_file = ref_dir / f"metrics_radar_{sig}_{ds}.csv"
            if not ref_file.exists():
                pytest.skip(f"No ref: {ref_file}")
            ref = pd.read_csv(ref_file)
            py = result["radar_values"][sig][ds]

            # rho_mean_med should match closely
            np.testing.assert_allclose(
                py["rho_mean_med"], ref["rho_mean_med"].iloc[0],
                rtol=1e-3, err_msg=f"rho_mean_med {sig}/{ds}"
            )
            # PCA sign ambiguity — compare absolute values
            np.testing.assert_allclose(
                abs(py["rho_pca1_med"]), abs(ref["rho_pca1_med"].iloc[0]),
                rtol=1e-3, err_msg=f"|rho_pca1_med| {sig}/{ds}"
            )
            np.testing.assert_allclose(
                abs(py["rho_mean_pca1"]), abs(ref["rho_mean_pca1"].iloc[0]),
                rtol=1e-3, err_msg=f"|rho_mean_pca1| {sig}/{ds}"
            )
            # prop_pca1_var should be close
            np.testing.assert_allclose(
                py["prop_pca1_var"], ref["prop_pca1_var"].iloc[0],
                rtol=0.05, err_msg=f"prop_pca1_var {sig}/{ds}"
            )
