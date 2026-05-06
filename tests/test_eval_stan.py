"""Tests for pysigqc_metrics.eval_stan — port of R compute_stan()."""

import numpy as np
import pandas as pd
import pytest

from pysigqc_metrics.eval_stan import compute_stan


def test_structure(signatures, datasets, names_sigs, names_datasets):
    result = compute_stan(signatures, names_sigs, datasets, names_datasets)
    assert set(result.keys()) == {"radar_values", "med_scores", "z_transf_scores", "elapsed_seconds"}


def test_metric_range(signatures, datasets, names_sigs, names_datasets):
    result = compute_stan(signatures, names_sigs, datasets, names_datasets)
    for sig in names_sigs:
        for ds in names_datasets:
            rho = result["radar_values"][sig][ds]["standardization_comp"]
            assert -1.0 <= rho <= 1.0


def test_score_lengths(signatures, datasets, names_sigs, names_datasets):
    result = compute_stan(signatures, names_sigs, datasets, names_datasets)
    for sig in names_sigs:
        for ds in names_datasets:
            assert len(result["med_scores"][sig][ds]) == 10
            assert len(result["z_transf_scores"][sig][ds]) == 10


def test_zero_variance_gene(signatures, datasets):
    """dataset_B has gene_5 with constant expression — should not produce NaN."""
    result = compute_stan(signatures, ["diffuse_sig"], datasets, ["dataset_B"])
    rho = result["radar_values"]["diffuse_sig"]["dataset_B"]["standardization_comp"]
    assert not np.isnan(rho)


def test_cross_validation(signatures, datasets, names_sigs, names_datasets, ref_dir):
    result = compute_stan(signatures, names_sigs, datasets, names_datasets)
    for sig in names_sigs:
        for ds in names_datasets:
            ref_file = ref_dir / f"stan_radar_{sig}_{ds}.csv"
            if not ref_file.exists():
                pytest.skip(f"No ref: {ref_file}")
            ref = pd.read_csv(ref_file)
            py = result["radar_values"][sig][ds]
            for col in ref.columns:
                np.testing.assert_allclose(py[col], ref[col].iloc[0], rtol=1e-4,
                                           err_msg=f"{col} {sig}/{ds}")
