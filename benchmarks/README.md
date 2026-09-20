# pysigQC-metrics — scalability benchmarks

Reproducible measurements behind [`SCALABILITY_REPORT.md`](../SCALABILITY_REPORT.md):
can the radar pipeline run on 100k–1M+ cell datasets on top of existing
standards (scipy sparse, H5AD, Zarr), or is a specialised engine needed?

## Layout

| Path | What |
|---|---|
| `generate_singlecell.py` | Synthetic single-cell generator. Writes sparse chunks straight to H5AD — never builds the dense matrix — and derives the other layouts (H5AD-CSC, Zarr CSR/CSC, chunking / compression variants) from the same matrix. |
| `prepare_real.py` | Turns a public CELLxGENE H5AD into the same on-disk layout (streamed normalisation + log1p when X holds counts; curated signatures). |
| `benchmark_pipeline.py` | One run = one (dataset, backend) in a fresh subprocess. Records time and peak RSS per module, storage size, nnz, density, the 14 radar metrics. `--aggregate` builds `results.csv` and checks numerical parity. |
| `memory_probe.py` | Where the memory peak comes from: every module in its own process; peak RSS, real working set (`--no-malloc-cache`) and live-allocation peak (`--trace`). |
| `prototype_stats_sidecar.py` | The first prototype of persisted per-gene statistics (now `run_pipeline(stats_cache=...)`). |
| `run_grid.sh` | The full experiment grid (resumable: finished runs are cached as JSON). |
| `make_plots.py` | Figures in `plots/` from `results.csv`. |
| `reference_impl/` | **Frozen snapshot of the pipeline at commit `9b533b2`** (before any backend work). Experiment A always runs this code, and the parity tests compare against it. Do not edit. |
| `benchmark_results/*.json` | One file per run (raw record, including sub-timings and radar values). |
| `results.csv` | All runs, flattened. |
| `data/` | Generated datasets (git-ignored, ~250 GB for the full grid). |

## Reproducing

```bash
python -m venv .venv && .venv/bin/pip install -e ".[bench]"
PY=.venv/bin/python benchmarks/run_grid.sh            # data main density chunks cache fast real v2
PY=.venv/bin/python benchmarks/run_grid.sh main       # a single stage
.venv/bin/python benchmarks/make_plots.py
```

A single run:

```bash
python benchmarks/generate_singlecell.py --cells 100000 --genes 20000 --density 0.05 --derive
python benchmarks/benchmark_pipeline.py --dataset sc_100000x20000_d05 \
    --backends dense scipy_csr scipy_csc h5ad_csr h5ad_csc zarr_csr zarr_csc
```

## Experiments

| | Backend name | Input handed to the pipeline |
|---|---|---|
| A | `dense` | pandas DataFrame → dense NumPy (frozen reference implementation) |
| B | `scipy_csr` | `AnnData` in memory, `X` = scipy CSR (cells × genes) |
| C | `scipy_csc` | same, CSC |
| D | `h5ad_csr` | `anndata.read_h5ad(path, backed="r")`, CSR on disk |
| E | `h5ad_csc` | same, CSC on disk |
| F | `zarr_csr` | Zarr v3 store, `anndata.io.sparse_dataset` (lazy), CSR |
| G | `zarr_csc` | same, CSC |
| | `dataframe` | the same DataFrame as A through today's package (`DenseBackend`) |
| | `zarr_*+dask` | `anndata.experimental.read_lazy` (dask array of sparse chunks) |
| | `zarr_csr_c{1,10,50}k` | CSR, chunk length ≈ 1k / 10k / 50k cells × all genes |
| | `zarr_csc_g{10,100,1000}` | CSC, chunk length ≈ 10 / 100 / 1000 genes × all cells |
| | `*_raw`, `*_gzip` | no compression (Zarr) / gzip (H5AD) |

Zarr and HDF5 store a sparse matrix as three 1-D arrays (`data`, `indices`,
`indptr`), so "1k cells × all genes" is expressed as a chunk *length* of
`1000 × mean nnz per cell` elements.

## What is measured, and how

* **Wall time**: `time.perf_counter()` around each module call
  (`eval_var`, `eval_expr`, `eval_compactness`, `compare_metrics`, `eval_stan`)
  and around the whole sequence. Loading/opening the input is timed
  separately (`load_seconds`) and is *not* part of the pipeline time.
* **Peak RSS**: a psutil sampling thread (10 ms) gives the high-water mark per
  module; `ru_maxrss` gives the process-wide peak (`peak_rss`). `tracemalloc`
  is never the only measure: it sees NumPy buffers but not what C libraries
  (HDF5, BLAS, zstd) allocate on their own; `memory_probe.py --trace` reports
  it next to RSS. For in-memory backends the peak includes the loaded matrix
  (`rss_after_load` isolates it).
* **RSS on macOS overstates the requirement.** libmalloc keeps freed large
  blocks resident (its "large cache"): at 1M cells peak RSS reads 7.8 GB while
  the live-allocation peak is 3.2 GB. `--no-malloc-cache` runs the worker with
  `MallocLargeCache=0`, which makes peak RSS the real working set (3.9 GB), at
  no cost in time. Both figures are reported. On Linux/glibc large blocks go
  straight back to the OS.
* **Sub-timings** (`backend_*_seconds` columns): storage reads, sparse
  traversal, gene extraction, ranking, per-cell scores, PCA, correlation —
  accumulated by the backend while the modules run.
* **Parity**: the 14 radar metrics of every run are compared
  (`rtol=1e-9, atol=1e-12`) with the dense run of the same dataset; where
  dense does not fit in memory, with in-memory scipy CSR, which is itself
  verified against dense on every dataset where dense fits. Exhaustive
  edge-case parity (NaN, Inf, constant genes, thresholds, …) lives in
  `tests/test_backends.py`.
* **Run tags** keep the states apart so that every gain stays attributable:
  *(none)* sparse backends only; `pipeline` / `shared_cache` the shared cache;
  `fastpaths` / `fastpaths_cache` the exact fast paths; `v2` today's
  `run_pipeline` defaults after the memory work; `v2_ws` the same with the
  allocator cache off; `v2_stats_build` / `v2_stats_reuse` persisted statistics.
* **Memory guard**: the dense baseline is never allowed to swap. Its peak is
  predicted as `29 bytes × genes × cells` (float32 DataFrame 4 + float64 copy
  8 + NaN mask 1 + `na.omit` copy 8 + `np.median` working copy 8, all alive in
  `eval_expr`; measured: 28–32 B/element). Above `--mem-limit-gb` (default 40)
  the run is recorded as `OOM_EXPECTED` instead of being executed.

## Caveats

* Timings are **warm page cache**: files are read shortly after being written,
  and `purge` needs root on macOS. Cold-cache cost is bounded by file size /
  SSD throughput (a few seconds for the 12 GB 1M-cell file on this machine);
  on network or spinning storage the storage share would be larger.
* Single machine (Apple Silicon, 18 cores, 64 GB), single process; NumPy's
  BLAS threads are left at their defaults.
* Synthetic data has co-expressed programmes but no real biology; the two
  public datasets are there to check that conclusions carry over.
