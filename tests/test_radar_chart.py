"""Tests for pysigqc_metrics.radar_chart — port of R compute_radar()."""

import numpy as np
import pandas as pd
import pytest

from pysigqc_metrics.eval_var import compute_var
from pysigqc_metrics.eval_expr import compute_expr
from pysigqc_metrics.eval_compactness import compute_compactness
from pysigqc_metrics.eval_stan import compute_stan
from pysigqc_metrics.radar_chart import compute_radar


def _build_radar_values(signatures, datasets, names_sigs, names_datasets):
    var_r = compute_var(signatures, names_sigs, datasets, names_datasets)
    expr_r = compute_expr(signatures, names_sigs, datasets, names_datasets)
    compact_r = compute_compactness(signatures, names_sigs, datasets, names_datasets)
    stan_r = compute_stan(signatures, names_sigs, datasets, names_datasets)

    radar_values = {}
    for sig in names_sigs:
        radar_values[sig] = {}
        for ds in names_datasets:
            vals = {}
            vals.update(var_r["radar_values"][sig][ds])
            vals.update(expr_r["radar_values"][sig][ds])
            vals.update(compact_r["radar_values"][sig][ds])
            vals.update(stan_r["radar_values"][sig][ds])
            radar_values[sig][ds] = vals
    return radar_values


def test_structure(signatures, datasets, names_sigs, names_datasets):
    rv = _build_radar_values(signatures, datasets, names_sigs, names_datasets)
    result = compute_radar(rv, names_sigs, names_datasets)
    assert set(result.keys()) == {"radar_plot_mat", "output_table", "areas", "legend_labels", "radarplot_rownames", "elapsed_seconds"}


def test_output_dimensions(signatures, datasets, names_sigs, names_datasets):
    rv = _build_radar_values(signatures, datasets, names_sigs, names_datasets)
    result = compute_radar(rv, names_sigs, names_datasets)
    expected_rows = len(names_sigs) * len(names_datasets)
    assert result["output_table"].shape == (expected_rows, 14)


def test_non_negative(signatures, datasets, names_sigs, names_datasets):
    rv = _build_radar_values(signatures, datasets, names_sigs, names_datasets)
    result = compute_radar(rv, names_sigs, names_datasets)
    assert (result["output_table"].values >= 0).all()


def test_areas_positive(signatures, datasets, names_sigs, names_datasets):
    rv = _build_radar_values(signatures, datasets, names_sigs, names_datasets)
    result = compute_radar(rv, names_sigs, names_datasets)
    assert (result["areas"] >= 0).all()


def test_fills_missing_metrics():
    rv = {
        "compact_sig": {"dataset_A": {"sd_median_ratio": 0.5}},
        "diffuse_sig": {"dataset_A": {"sd_median_ratio": 0.3}},
    }
    result = compute_radar(rv, ["compact_sig", "diffuse_sig"], ["dataset_A"])
    assert result["output_table"].shape[1] == 14


def test_cross_validation(signatures, datasets, names_sigs, names_datasets, ref_dir):
    rv = _build_radar_values(signatures, datasets, names_sigs, names_datasets)
    result = compute_radar(rv, names_sigs, names_datasets)

    ref_file = ref_dir / "radar_output_table.csv"
    if not ref_file.exists():
        pytest.skip("No radar reference")
    ref = pd.read_csv(ref_file, index_col=0)

    # Compare output tables (note: Python uses 10 metrics here, R uses 14 with metrics module)
    # Only compare columns that are present in both (without compare_metrics, 4 will be zero)
    for col in ref.columns:
        if col in result["output_table"].columns:
            for idx in ref.index:
                if idx in result["output_table"].index:
                    r_val = ref.loc[idx, col]
                    py_val = result["output_table"].loc[idx, col]
                    # The 4 metrics from compare_metrics are 0 in our partial pipeline
                    if col in ("rho_mean_med", "rho_pca1_med", "rho_mean_pca1", "prop_pca1_var"):
                        continue
                    np.testing.assert_allclose(py_val, r_val, rtol=1e-4, atol=1e-10,
                                               err_msg=f"{col} at {idx}")
