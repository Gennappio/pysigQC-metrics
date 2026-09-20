"""Backend parity: every storage backend must reproduce the dense pipeline.

The same matrix is pushed through
    * the frozen pre-backend implementation (benchmarks/reference_impl),
    * the DataFrame path (DenseBackend),
    * SparseBackend over scipy CSR/CSC in both orientations,
    * AnnData in memory, backed H5AD, Zarr (sparse_dataset) and dask-lazy,
and all 14 radar metrics plus the intermediate per-gene outputs are compared
with a tight floating-point tolerance.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from pysigqc_metrics import ALL_METRICS, run_pipeline
from pysigqc_metrics.backends import (
    AnnDataBackend, DenseBackend, SparseBackend, as_backend,
)

RTOL = 1e-10
ATOL = 1e-12

pytestmark = pytest.mark.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Test matrices
# ---------------------------------------------------------------------------

def _base_matrix(seed=0, n_genes=60, n_cells=83, density=0.25, negative=False):
    rng = np.random.default_rng(seed)
    vals = np.log1p(rng.gamma(2.0, 1.5, (n_genes, n_cells))).astype(np.float32)
    if negative:
        vals -= np.float32(1.0)
    mask = rng.random((n_genes, n_cells)) < density
    arr = np.where(mask, vals, np.float32(0.0)).astype(np.float32)
    # Co-expressed block so that correlations / PCA are not pure noise.
    factor = rng.gamma(1.0, 1.0, n_cells).astype(np.float32)
    arr[:12] = arr[:12] * (factor > 0.7) + (0.3 * factor * (factor > 0.7)).astype(np.float32)
    genes = [f"g{i}" for i in range(n_genes)]
    cells = [f"c{j}" for j in range(n_cells)]
    return pd.DataFrame(arr, index=genes, columns=cells)


def _signatures(df):
    g = list(df.index)
    return {
        "coexpr": g[:12],
        "overlap": g[8:30],                       # overlaps 'coexpr'
        "with_missing": g[25:40] + ["NOT_A_GENE", "ALSO_MISSING"],
        "tail": g[40:],
    }


def _case_plain():
    return _base_matrix()


def _case_zeros_and_constant():
    df = _base_matrix(seed=1)
    df.iloc[3] = 0.0            # all-zero gene (zero variance), inside a signature
    df.iloc[9] = 2.5            # constant non-zero gene (zero variance, fully stored)
    df.iloc[45] = 0.0
    df.iloc[50] = 1.0
    return df


def _case_nan():
    df = _base_matrix(seed=2)
    df.iloc[5, [1, 7, 30]] = np.nan
    df.iloc[27, 4] = np.nan
    df.iloc[41, :] = np.nan     # all-NaN gene
    df.iloc[55, 10:20] = np.nan
    return df


def _case_inf():
    df = _base_matrix(seed=3)
    df.iloc[2, 5] = np.inf
    df.iloc[28, 9] = -np.inf
    df.iloc[44, 3] = np.inf
    df.iloc[6, 2] = np.nan
    return df


def _case_negative_values():
    return _base_matrix(seed=4, negative=True, density=0.3)


def _case_dense_nonzero_median():
    # > 50% stored positives: the global median is *not* zero, which forces
    # the exact-median partition path of the sparse backends.
    return _base_matrix(seed=5, density=0.8)


def _case_median_between_zero_and_positive():
    # Even number of values, exactly half zeros: median = (0 + min positive) / 2.
    rng = np.random.default_rng(6)
    n_genes, n_cells = 60, 50
    arr = np.zeros(n_genes * n_cells, dtype=np.float32)
    pos = rng.choice(arr.size, arr.size // 2, replace=False)
    arr[pos] = rng.gamma(2.0, 1.0, pos.size).astype(np.float32) + np.float32(0.5)
    df = pd.DataFrame(arr.reshape(n_genes, n_cells),
                      index=[f"g{i}" for i in range(n_genes)],
                      columns=[f"c{j}" for j in range(n_cells)])
    return df


CASES = {
    "plain": _case_plain,
    "zeros_constant": _case_zeros_and_constant,
    "nan": _case_nan,
    "inf": _case_inf,
    "negative": _case_negative_values,
    "nonzero_median": _case_dense_nonzero_median,
    "half_zero_median": _case_median_between_zero_and_positive,
}


# ---------------------------------------------------------------------------
# Backend builders (all from the same DataFrame, genes x samples)
# ---------------------------------------------------------------------------

def _cells_by_genes(df, fmt):
    m = sp.csr_matrix(df.to_numpy().T)   # NaN / Inf stay as explicit entries
    return m.tocsc() if fmt == "csc" else m


def _adata(df, fmt):
    ad = pytest.importorskip("anndata")
    return ad.AnnData(X=_cells_by_genes(df, fmt),
                      obs=pd.DataFrame(index=df.columns.astype(str)),
                      var=pd.DataFrame(index=df.index.astype(str)))


def build_backend(kind, df, tmp_path):
    chunk = 97  # tiny chunks: every scan is split into many pieces
    if kind.split("_")[0] in ("h5ad", "zarr", "lazy"):
        # On-disk backends need the anndata.io API (sparse_dataset, zarr v3).
        pytest.importorskip("anndata", minversion="0.11")
    if kind == "dataframe":
        return df
    if kind == "dense_backend":
        return DenseBackend(df)
    if kind in ("scipy_csr", "scipy_csc"):
        return SparseBackend(_cells_by_genes(df, kind[-3:]), df.index, df.columns,
                             gene_axis=1, chunk_nnz=chunk)
    if kind in ("scipy_csr_genes_first", "scipy_csc_genes_first"):
        m = sp.csr_matrix(df.to_numpy())
        m = m.tocsc() if "csc" in kind else m
        return SparseBackend(m, df.index, df.columns, gene_axis=0, chunk_nnz=chunk)
    if kind == "scipy_explicit_zeros":
        m = _cells_by_genes(df, "csr").copy()
        m.data[::7] = 0.0                 # stored zeros ...
        dense = pd.DataFrame(m.toarray().T, index=df.index, columns=df.columns)
        return SparseBackend(m, df.index, df.columns, chunk_nnz=chunk), dense
    if kind in ("anndata_csr", "anndata_csc"):
        return AnnDataBackend(_adata(df, kind[-3:]), chunk_nnz=chunk)
    if kind == "anndata_dense":
        ad = pytest.importorskip("anndata")
        a = ad.AnnData(X=np.ascontiguousarray(df.to_numpy().T),
                       obs=pd.DataFrame(index=df.columns), var=pd.DataFrame(index=df.index))
        return a
    if kind in ("h5ad_csr", "h5ad_csc"):
        ad = pytest.importorskip("anndata")
        path = tmp_path / f"{kind}.h5ad"
        _adata(df, kind[-3:]).write_h5ad(path)
        return AnnDataBackend(ad.read_h5ad(path, backed="r"), chunk_nnz=chunk)
    if kind in ("zarr_csr", "zarr_csc"):
        pytest.importorskip("anndata")
        zarr = pytest.importorskip("zarr")
        from anndata.io import read_elem, sparse_dataset
        path = tmp_path / f"{kind}.zarr"
        _adata(df, kind[-3:]).write_zarr(path)
        g = zarr.open_group(path, mode="r")
        return SparseBackend(sparse_dataset(g["X"]), read_elem(g["var"]).index,
                             read_elem(g["obs"]).index, chunk_nnz=chunk)
    if kind in ("lazy_csr", "lazy_csc"):
        ad = pytest.importorskip("anndata")
        pytest.importorskip("dask")
        pytest.importorskip("xarray")
        path = tmp_path / f"{kind}.zarr"
        _adata(df, kind[-3:]).write_zarr(path)
        return AnnDataBackend(ad.experimental.read_lazy(path))
    raise ValueError(kind)


SPARSE_KINDS = [
    "scipy_csr", "scipy_csc", "scipy_csr_genes_first", "scipy_csc_genes_first",
    "anndata_csr", "anndata_csc", "h5ad_csr", "h5ad_csc", "zarr_csr", "zarr_csc",
    "lazy_csr", "lazy_csc",
]


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------

def _run(data, sigs, thresholds=None, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return run_pipeline(sigs, list(sigs), {"ds": data}, ["ds"], thresholds=thresholds, **kw)


def _run_reference(df, sigs, thresholds=None):
    ref = pytest.importorskip("benchmarks.reference_impl")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ref.run_pipeline(sigs, list(sigs), {"ds": df}, ["ds"], thresholds=thresholds)


def _radar(out, sigs):
    return np.array([[out["radar_values"][s]["ds"][m] for m in ALL_METRICS] for s in sigs], dtype=float)


def assert_same_results(got, want, sigs):
    np.testing.assert_allclose(_radar(got, sigs), _radar(want, sigs),
                               rtol=RTOL, atol=ATOL, equal_nan=True)
    np.testing.assert_allclose(got["radar_result"]["output_table"].to_numpy(),
                               want["radar_result"]["output_table"].to_numpy(),
                               rtol=RTOL, atol=ATOL)
    t_got, t_want = got["expr_result"]["thresholds"]["ds"], want["expr_result"]["thresholds"]["ds"]
    np.testing.assert_allclose(t_got, t_want, rtol=0, atol=0, equal_nan=True)  # exact median
    for s in sigs:
        def cmp(a, b):
            np.testing.assert_allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float),
                                       rtol=RTOL, atol=ATOL, equal_nan=True)
        cmp(got["var_result"]["mean_sd_tables"][s]["ds"], want["var_result"]["mean_sd_tables"][s]["ds"])
        cmp(got["var_result"]["all_sd"][s]["ds"], want["var_result"]["all_sd"][s]["ds"])
        cmp(got["var_result"]["all_mean"][s]["ds"], want["var_result"]["all_mean"][s]["ds"])
        assert got["var_result"]["inter"][s]["ds"] == want["var_result"]["inter"][s]["ds"]
        for key in ("na_proportions", "expr_proportions"):
            a, b = got["expr_result"][key][s]["ds"], want["expr_result"][key][s]["ds"]
            cmp(a.sort_index(), b.sort_index())
        a, b = got["compact_result"]["autocor_matrices"][s]["ds"], want["compact_result"]["autocor_matrices"][s]["ds"]
        assert list(a.index) == list(b.index)
        cmp(a, b)
        for key in ("med_scores", "mean_scores"):
            cmp(got["metrics_result"]["scores"][s]["ds"][key], want["metrics_result"]["scores"][s]["ds"][key])
        pa, pb = (r["metrics_result"]["scores"][s]["ds"]["pca1_scores"] for r in (got, want))
        assert (pa is None) == (pb is None)
        if pa is not None:
            cmp(pa, pb)
        cmp(got["stan_result"]["med_scores"][s]["ds"], want["stan_result"]["med_scores"][s]["ds"])
        cmp(got["stan_result"]["z_transf_scores"][s]["ds"], want["stan_result"]["z_transf_scores"][s]["ds"])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", list(CASES))
def test_dataframe_path_is_bit_identical_to_reference(case):
    """The DataFrame path must not change at all for existing users."""
    df = CASES[case]()
    sigs = _signatures(df)
    got, want = _run(df, sigs), _run_reference(df, sigs)
    np.testing.assert_array_equal(_radar(got, sigs), _radar(want, sigs))


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("kind", SPARSE_KINDS)
def test_sparse_backends_match_dense(kind, case, tmp_path):
    df = CASES[case]()
    sigs = _signatures(df)
    want = _run_reference(df, sigs)
    got = _run(build_backend(kind, df, tmp_path), sigs)
    assert_same_results(got, want, sigs)


@pytest.mark.parametrize("threshold", [0.0, 0.75, -0.25, 1e9, -1e9])
@pytest.mark.parametrize("case", ["plain", "nan", "inf", "negative"])
@pytest.mark.parametrize("kind", ["scipy_csr", "scipy_csc", "h5ad_csr", "zarr_csc"])
def test_explicit_thresholds(kind, case, threshold, tmp_path):
    df = CASES[case]()
    sigs = _signatures(df)
    want = _run_reference(df, sigs, thresholds={"ds": threshold})
    got = _run(build_backend(kind, df, tmp_path), sigs, thresholds={"ds": threshold})
    assert_same_results(got, want, sigs)


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("kind", ["dataframe"] + SPARSE_KINDS)
def test_without_shared_cache(kind, case, tmp_path):
    """share_cache=True is the default (exercised by every other test); the
    per-module path must give the same results."""
    df = CASES[case]()
    sigs = _signatures(df)
    want = _run_reference(df, sigs)
    got = _run(build_backend(kind, df, tmp_path), sigs, share_cache=False)
    assert_same_results(got, want, sigs)


@pytest.mark.parametrize("kind", ["dataframe", "scipy_csr", "scipy_csc", "h5ad_csr"])
def test_modules_called_directly(kind, tmp_path):
    """compute_* called one by one (no pipeline, no cache) on any input type."""
    import pysigqc_metrics as pq
    df = CASES["nan"]()
    sigs = _signatures(df)
    names = list(sigs)
    want = _run_reference(df, sigs)
    data = as_backend(build_backend(kind, df, tmp_path))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parts = [f(sigs, names, {"ds": data}, ["ds"]) for f in
                 (pq.compute_var, pq.compute_expr, pq.compute_compactness,
                  pq.compute_metrics, pq.compute_stan)]
    got = np.array([[next(p["radar_values"][s]["ds"][m] for p in parts if m in p["radar_values"][s]["ds"])
                     for m in ALL_METRICS] for s in names], dtype=float)
    np.testing.assert_allclose(got, _radar(want, sigs), rtol=RTOL, atol=ATOL, equal_nan=True)


@pytest.mark.parametrize("threshold", [0.0, 0.75, -0.25])
@pytest.mark.parametrize("kind", ["scipy_csr", "scipy_csc", "h5ad_csc"])
def test_shared_cache_with_explicit_threshold(kind, threshold, tmp_path):
    df = CASES["negative"]()
    sigs = _signatures(df)
    want = _run_reference(df, sigs, thresholds=[threshold])
    for share in (True, False):
        got = _run(build_backend(kind, df, tmp_path), sigs, thresholds=[threshold], share_cache=share)
        assert_same_results(got, want, sigs)


def test_shared_cache_scans_the_matrix_twice():
    """The point of the cache: 2 scans instead of one or two per module."""
    from pysigqc_metrics.backends import DatasetStatsCache
    df = CASES["plain"]()
    sigs = _signatures(df)
    for fmt in ("csr", "csc"):
        counts = {}
        for share in (False, True):
            backend = SparseBackend(_cells_by_genes(df, fmt), df.index, df.columns)
            reads = {"n": 0}
            real = backend._reader.read

            def counting(a, b, real=real, reads=reads):
                reads["n"] += 1
                return real(a, b)

            backend._reader.read = counting
            _run(backend, sigs, share_cache=share)
            counts[share] = reads["n"]      # one chunk per scan at this size
        assert counts[True] == 2
        assert counts[False] > counts[True]


def test_explicit_stored_zeros(tmp_path):
    backend, dense = build_backend("scipy_explicit_zeros", CASES["plain"](), tmp_path)
    sigs = _signatures(dense)
    assert_same_results(_run(backend, sigs), _run_reference(dense, sigs), sigs)


def test_anndata_passed_directly(tmp_path):
    df = CASES["plain"]()
    sigs = _signatures(df)
    want = _run_reference(df, sigs)
    assert_same_results(_run(_adata(df, "csr"), sigs), want, sigs)
    assert_same_results(_run(build_backend("anndata_dense", df, tmp_path), sigs), want, sigs)


def test_anndata_layer():
    ad = pytest.importorskip("anndata")
    df = CASES["plain"]()
    sigs = _signatures(df)
    a = _adata(df, "csr")
    a.layers["lognorm"] = a.X.copy()
    a.X = a.X * 0
    backend = AnnDataBackend(a, layer="lognorm")
    assert_same_results(_run(backend, sigs), _run_reference(df, sigs), sigs)


def test_single_gene_signature():
    df = CASES["plain"]()
    sigs = _signatures(df)
    sigs["single"] = [df.index[0]]
    want = _run_reference(df, sigs)
    for fmt in ("csr", "csc"):
        backend = SparseBackend(_cells_by_genes(df, fmt), df.index, df.columns)
        assert_same_results(_run(backend, sigs), want, sigs)


def test_signature_with_no_gene_in_dataset_behaves_like_reference():
    """Pre-existing behaviour, deliberately preserved: a signature with no
    gene in the dataset makes compute_compactness raise. Backends must not
    diverge from the reference here either."""
    df = CASES["plain"]()
    sigs = {"absent": ["NOPE1", "NOPE2", "NOPE3"]}
    with pytest.raises(ValueError):
        _run_reference(df, sigs)
    for data in (df, SparseBackend(_cells_by_genes(df, "csr"), df.index, df.columns)):
        with pytest.raises(ValueError):
            _run(data, sigs)


def test_never_densifies_whole_matrix(monkeypatch):
    """Guard rail: the sparse path may only build K x N blocks."""
    df = _base_matrix(n_genes=300, n_cells=40)
    sigs = {"a": list(df.index[:10]), "b": list(df.index[5:25])}
    backend = SparseBackend(_cells_by_genes(df, "csr"), df.index, df.columns)
    biggest = {"n": 0}
    real_zeros = np.zeros

    def spy_zeros(shape, *a, **k):
        out = real_zeros(shape, *a, **k)
        if out.ndim == 2:
            biggest["n"] = max(biggest["n"], out.shape[0])
        return out

    def no_toarray(self, *a, **k):
        raise AssertionError("toarray() called on the expression matrix")

    monkeypatch.setattr(np, "zeros", spy_zeros)
    monkeypatch.setattr(sp.csr_matrix, "toarray", no_toarray)
    monkeypatch.setattr(sp.csc_matrix, "toarray", no_toarray)
    for share in (True, False):
        _run(backend, sigs, share_cache=share)
    assert 0 < biggest["n"] <= 25  # never more rows than signature genes


def test_backend_primitives_against_numpy():
    df = CASES["nan"]()
    arr = df.to_numpy(dtype=float)
    for fmt in ("csr", "csc"):
        b = SparseBackend(_cells_by_genes(df, fmt), df.index, df.columns, chunk_nnz=50)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, sd = b.gene_mean_sd()
            np.testing.assert_allclose(mean, np.nanmean(arr, axis=1), rtol=RTOL, atol=ATOL, equal_nan=True)
            np.testing.assert_allclose(sd, np.nanstd(arr, axis=1, ddof=1), rtol=RTOL, atol=ATOL, equal_nan=True)
        idx = np.array([7, 3, 3, 41])
        np.testing.assert_array_equal(b.get_genes(idx), arr[idx])
        blocks = np.vstack([blk for _, _, blk in b.iter_gene_blocks(17)])
        np.testing.assert_array_equal(blocks, arr)
        assert b.shape == arr.shape
        assert b.layout == ("gene-major" if fmt == "csc" else "cell-major")


def test_as_backend_rejects_raw_sparse():
    with pytest.raises(TypeError, match="SparseBackend"):
        as_backend(sp.csr_matrix(np.eye(3)))
    with pytest.raises(ValueError, match="does not match"):
        SparseBackend(sp.csr_matrix(np.eye(3)), ["a", "b"], ["x", "y", "z"])


# ---------------------------------------------------------------------------
# Exact fast paths (utils.nanmedian_over_genes / utils.rank_rows)
# ---------------------------------------------------------------------------

def _blocks():
    rng = np.random.default_rng(11)
    for k, n, density in [(1, 40, 0.1), (2, 41, 0.3), (7, 200, 0.05), (8, 200, 0.05),
                          (25, 333, 0.6), (30, 64, 0.0), (5, 50, 1.0)]:
        b = np.where(rng.random((k, n)) < density, rng.normal(0.5, 1.0, (k, n)), 0.0)
        b = np.round(b, 1)                      # plenty of ties, negatives and -0.0
        yield b
        with_special = b.copy()
        with_special[0, ::9] = np.nan
        with_special[-1, 3] = np.inf
        with_special[k // 2, 5] = -np.inf
        yield with_special
    yield np.full((4, 10), np.nan)
    yield np.zeros((0, 10))


@pytest.mark.parametrize("fast", [True, False])
def test_exact_fast_paths_are_bit_identical(fast, monkeypatch):
    from scipy import stats as sp_stats
    from pysigqc_metrics import utils
    monkeypatch.setattr(utils, "EXACT_FAST_PATHS", fast)
    for b in _blocks():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            np.testing.assert_array_equal(utils.nanmedian_over_genes(b), np.nanmedian(b, axis=0))
            if b.shape[0]:
                np.testing.assert_array_equal(utils.rank_rows(b), sp_stats.rankdata(b, axis=1))


@pytest.mark.parametrize("case", list(CASES))
def test_pipeline_identical_with_and_without_fast_paths(case, monkeypatch):
    from pysigqc_metrics import utils
    df = CASES[case]()
    sigs = _signatures(df)
    backend = lambda: SparseBackend(_cells_by_genes(df, "csc"), df.index, df.columns)
    fast = _run(backend(), sigs)
    monkeypatch.setattr(utils, "EXACT_FAST_PATHS", False)
    slow = _run(backend(), sigs)
    np.testing.assert_array_equal(_radar(fast, sigs), _radar(slow, sigs))


# ---------------------------------------------------------------------------
# Dense (DataFrame) path: row-blocked statistics must stay bit-identical
# ---------------------------------------------------------------------------

def _big_frame(dtype, order, n_genes=301, n_cells=97):
    rng = np.random.default_rng(21)
    arr = np.where(rng.random((n_genes, n_cells)) < 0.4, rng.gamma(2.0, 1.0, (n_genes, n_cells)), 0.0)
    if np.dtype(dtype).kind == "f":
        arr[5, 3] = np.nan
        arr[77, :] = np.nan
        arr[100, 10:50] = np.nan
        arr[8, 2] = np.inf
        arr[300, 7] = -np.inf
    else:
        arr = np.round(arr * 3)
    arr = np.asarray(arr, dtype=dtype, order=order)
    return pd.DataFrame(arr, index=[f"g{i}" for i in range(n_genes)],
                        columns=[f"c{j}" for j in range(n_cells)], copy=False)


def _mixed_frame():
    df = _big_frame(np.float32, "C").iloc[:, :40].copy()
    df["c3"] = df["c3"].astype(np.float64)
    df["c9"] = df["c9"].fillna(0).replace([np.inf, -np.inf], 0).round().astype(np.int64)
    return df


DENSE_FRAMES = {
    "f32_C": lambda: _big_frame(np.float32, "C"), "f32_F": lambda: _big_frame(np.float32, "F"),
    "f64_C": lambda: _big_frame(np.float64, "C"), "f64_F": lambda: _big_frame(np.float64, "F"),
    "int64": lambda: _big_frame(np.int64, "C"), "mixed": _mixed_frame,
}


# 301 genes: 97*2 -> 2-row blocks + a merged 3-row tail; 97*100 -> 100-row
# blocks whose 1-row remainder must be merged; 1<<25 -> a single block.
@pytest.mark.parametrize("block_elements", [97 * 2, 97 * 100, 97 * 150, 1 << 25])
@pytest.mark.parametrize("frame", list(DENSE_FRAMES))
def test_dense_blocks_are_bit_identical(frame, block_elements):
    df = DENSE_FRAMES[frame]()
    backend = DenseBackend(df, block_elements=block_elements)
    sizes = [b - a for a, b in backend._row_blocks(backend._native())]
    assert sum(sizes) == len(df) and min(sizes) >= 2

    full = df.to_numpy(dtype=float)                    # the historical computation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        want_sd, want_mean = np.nanstd(full, axis=1, ddof=1), np.nanmean(full, axis=1)
        nan_mask = np.isnan(full)
        clean = full[~nan_mask.any(axis=1)]
        want_thr = float(np.median(clean))
        mean, sd = backend.gene_mean_sd()
        f_mean, f_sd, f_stats, _ = backend.fused_stats(None)
    np.testing.assert_array_equal(mean, want_mean)
    np.testing.assert_array_equal(sd, want_sd)
    np.testing.assert_array_equal(f_mean, want_mean)
    np.testing.assert_array_equal(f_sd, want_sd)
    for thr in (None, 0.0, 0.7, -1.0):
        st = backend.expr_stats(thr)
        t = want_thr if thr is None else thr
        assert st.threshold == t
        np.testing.assert_array_equal(st.nan_counts, nan_mask.sum(axis=1))
        np.testing.assert_array_equal(st.below_counts, (full < t).sum(axis=1))
    assert f_stats.threshold == want_thr


@pytest.mark.parametrize("frame", list(DENSE_FRAMES))
def test_dense_pipeline_bit_identical_when_blocked(frame):
    df = DENSE_FRAMES[frame]()
    sigs = {"a": list(df.index[:15]), "b": list(df.index[10:60]), "c": list(df.index[70:110]) + ["NOPE"]}
    want = _run_reference(df, sigs)
    for share in (True, False):
        got = _run(DenseBackend(df, block_elements=df.shape[1] * 100), sigs, share_cache=share)
        np.testing.assert_array_equal(_radar(got, sigs), _radar(want, sigs))
        assert got["expr_result"]["thresholds"]["ds"] == want["expr_result"]["thresholds"]["ds"]
        for s_ in sigs:
            np.testing.assert_array_equal(got["var_result"]["all_sd"][s_]["ds"].to_numpy(),
                                          want["var_result"]["all_sd"][s_]["ds"].to_numpy())
            np.testing.assert_array_equal(got["var_result"]["all_mean"][s_]["ds"].to_numpy(),
                                          want["var_result"]["all_mean"][s_]["ds"].to_numpy())
            np.testing.assert_array_equal(got["compact_result"]["autocor_matrices"][s_]["ds"].to_numpy(),
                                          want["compact_result"]["autocor_matrices"][s_]["ds"].to_numpy())
            for key in ("med_scores", "mean_scores", "pca1_scores"):
                np.testing.assert_array_equal(got["metrics_result"]["scores"][s_]["ds"][key],
                                              want["metrics_result"]["scores"][s_]["ds"][key])
            for key in ("med_scores", "z_transf_scores"):
                np.testing.assert_array_equal(got["stan_result"][key][s_]["ds"], want["stan_result"][key][s_]["ds"])


def test_dense_median_of_even_and_odd_counts():
    for n_genes, n_cells in [(4, 5), (5, 5), (2, 2), (3, 1)]:
        rng = np.random.default_rng(n_genes * 10 + n_cells)
        df = pd.DataFrame(rng.normal(size=(n_genes, n_cells)).astype(np.float32))
        want = float(np.median(df.to_numpy(dtype=float)))
        assert DenseBackend(df, block_elements=2 * n_cells).expr_stats(None).threshold == want
    all_nan = pd.DataFrame(np.full((3, 4), np.nan))
    assert np.isnan(DenseBackend(all_nan).expr_stats(None).threshold)


# ---------------------------------------------------------------------------
# Memory-motivated exact kernels
# ---------------------------------------------------------------------------

def test_nanmean_and_inplace_median_are_bit_identical():
    from pysigqc_metrics import utils
    for b in _blocks():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            np.testing.assert_array_equal(utils.nanmean_over_genes(b), np.nanmean(b, axis=0))
            want = np.nanmedian(b, axis=0)
            scratch = b.copy()
            np.testing.assert_array_equal(utils.nanmedian_over_genes(scratch, overwrite=True), want)
            np.testing.assert_array_equal(utils.nanmedian_over_genes(b), want)   # b untouched


def test_ranks_are_float32_only_when_exact(monkeypatch):
    from scipy import stats as sp_stats
    from pysigqc_metrics import utils
    assert utils.rank_dtype(1 << 23) == np.float32
    assert utils.rank_dtype((1 << 23) + 1) == np.float64
    # every average rank up to the limit survives the float32 round trip
    top = np.arange((1 << 23) - 1000, (1 << 23) + 0.5, 0.5)
    np.testing.assert_array_equal(top.astype(np.float32).astype(np.float64), top)
    b = next(iter(_blocks()))
    ranks = utils.rank_rows(b)
    assert ranks.dtype == np.float32
    np.testing.assert_array_equal(ranks.astype(np.float64), sp_stats.rankdata(b, axis=1))
    np.testing.assert_array_equal(np.corrcoef(ranks), np.corrcoef(sp_stats.rankdata(b, axis=1)))
    monkeypatch.setattr(utils, "_FLOAT32_EXACT_RANKS", 10)
    assert utils.rank_rows(b).dtype == np.float64


@pytest.mark.parametrize("case", list(CASES))
def test_intermediate_results_identical_with_and_without_fast_paths(case, monkeypatch):
    from pysigqc_metrics import utils
    df = CASES[case]()
    sigs = _signatures(df)
    fast = _run(df, sigs)
    monkeypatch.setattr(utils, "EXACT_FAST_PATHS", False)
    slow = _run(df, sigs)
    for s_ in sigs:
        np.testing.assert_array_equal(fast["compact_result"]["autocor_matrices"][s_]["ds"].to_numpy(),
                                      slow["compact_result"]["autocor_matrices"][s_]["ds"].to_numpy())
        for key in ("med_scores", "mean_scores", "pca1_scores"):
            np.testing.assert_array_equal(fast["metrics_result"]["scores"][s_]["ds"][key],
                                          slow["metrics_result"]["scores"][s_]["ds"][key])
        for key in ("med_scores", "z_transf_scores"):
            np.testing.assert_array_equal(fast["stan_result"][key][s_]["ds"], slow["stan_result"][key][s_]["ds"])


def test_pca_scores_do_not_pin_the_score_matrix():
    """pca1_scores used to be a [:, 0] view keeping samples x genes alive."""
    df = CASES["plain"]()
    sigs = _signatures(df)
    out = _run(df, sigs)
    for s_ in sigs:
        pca1 = out["metrics_result"]["scores"][s_]["ds"]["pca1_scores"]
        assert pca1.base is None and pca1.flags.c_contiguous


# ---------------------------------------------------------------------------
# Persisted per-gene statistics (run_pipeline(stats_cache=...))
# ---------------------------------------------------------------------------

def _count_scans(backend):
    calls = {"n": 0}
    real = backend._scan_chunks

    def counting():
        calls["n"] += 1
        return real()

    backend._scan_chunks = counting
    return calls


@pytest.mark.parametrize("fmt", ["csr", "csc"])
def test_stats_cache_roundtrip_skips_the_scans(fmt, tmp_path):
    df = CASES["nan"]()
    sigs = _signatures(df)
    want = _run_reference(df, sigs)
    make = lambda: SparseBackend(_cells_by_genes(df, fmt), df.index, df.columns)

    first = make()
    scans = _count_scans(first)
    assert_same_results(_run(first, sigs, stats_cache=tmp_path), want, sigs)
    assert scans["n"] == 2 and (tmp_path / "ds.gene_stats.npz").exists()

    second = make()
    scans = _count_scans(second)
    other_sigs = {"x": list(df.index[3:20]), "y": list(df.index[15:45])}   # new signatures, same matrix
    got = _run(second, other_sigs, stats_cache=tmp_path)
    assert scans["n"] == 0
    assert_same_results(got, _run_reference(df, other_sigs), other_sigs)


def test_stats_cache_extends_to_new_thresholds(tmp_path):
    df = CASES["negative"]()
    sigs = _signatures(df)
    make = lambda: SparseBackend(_cells_by_genes(df, "csc"), df.index, df.columns)
    _run(make(), sigs, stats_cache=tmp_path)
    b = make()
    scans = _count_scans(b)
    got = _run(b, sigs, thresholds=[0.25], stats_cache=tmp_path)
    assert scans["n"] == 1                                     # only the new threshold's counts
    assert_same_results(got, _run_reference(df, sigs, thresholds=[0.25]), sigs)
    b = make()
    scans = _count_scans(b)
    for thr in (None, [0.25]):
        _run(b, sigs, thresholds=thr, stats_cache=tmp_path)
    assert scans["n"] == 0                                     # both are persisted now


def test_stale_or_broken_stats_cache_is_ignored(tmp_path):
    df = CASES["plain"]()
    sigs = _signatures(df)
    _run(df, sigs, stats_cache=tmp_path)
    stats_file = tmp_path / "ds.gene_stats.npz"

    values = df.to_numpy().copy()
    values[0, 0] += 1                                          # same shape, different content
    changed = pd.DataFrame(values, index=df.index, columns=df.columns)
    with pytest.warns(RuntimeWarning, match="different matrix"):
        got = run_pipeline(sigs, list(sigs), {"ds": changed}, ["ds"], stats_cache=tmp_path)
    assert_same_results(got, _run_reference(changed, sigs), sigs)

    renamed = df.rename(index={df.index[-1]: "renamed_gene"})
    sigs_r = {k: [g for g in v if g in renamed.index] for k, v in sigs.items()}
    with pytest.warns(RuntimeWarning, match="different matrix"):
        got = run_pipeline(sigs_r, list(sigs_r), {"ds": renamed}, ["ds"], stats_cache=tmp_path)
    assert_same_results(got, _run_reference(renamed, sigs_r), sigs_r)

    stats_file.write_bytes(b"not an npz file")
    with pytest.warns(RuntimeWarning, match="unreadable"):
        got = run_pipeline(sigs, list(sigs), {"ds": df}, ["ds"], stats_cache=tmp_path)
    assert_same_results(got, _run_reference(df, sigs), sigs)
    from pysigqc_metrics.backends import DatasetStatsCache
    assert DatasetStatsCache(df).load_stats(stats_file)        # rewritten, valid again


def test_stats_cache_file_per_dataset(tmp_path):
    a, b = CASES["plain"](), CASES["nan"]()
    sigs = _signatures(a)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_pipeline(sigs, list(sigs), {"set A": a, "set/B": b}, ["set A", "set/B"], stats_cache=tmp_path)
    assert sorted(f.name for f in tmp_path.iterdir()) == ["set_A.gene_stats.npz", "set_B.gene_stats.npz"]
