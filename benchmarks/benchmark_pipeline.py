"""Reproducible benchmark of the pysigQC radar pipeline across storage backends.

One *run* = one (dataset, backend) pair, executed in a fresh subprocess so
that peak RSS is attributable to that run alone.

Backends
--------
    dense        A  current implementation: pandas DataFrame -> dense NumPy.
                    Runs the frozen snapshot in ``benchmarks/reference_impl``.
    dataframe       the same DataFrame through today's package (DenseBackend)
    scipy_csr    B  scipy CSR in memory (cells x genes)
    scipy_csc    C  scipy CSC in memory
    h5ad_csr     D  H5AD opened with backed="r", X = CSR
    h5ad_csc     E  H5AD opened with backed="r", X = CSC
    zarr_csr     F  Zarr, lazy (anndata sparse_dataset), X = CSR
    zarr_csc     G  Zarr, lazy, X = CSC
    Any other layout written by generate_singlecell.py (``zarr_csr_c10k``,
    ``h5ad_csr_gzip``, ...) can be used as a backend name directly. A
    ``+dask`` suffix (``zarr_csr+dask``) reads through
    ``anndata.experimental.read_lazy`` instead of ``sparse_dataset``.

Measured per run
----------------
wall time and peak RSS per module (eval_var, eval_expr, eval_compactness,
compare_metrics, eval_stan), total, load time, storage size, nnz, density,
and the 14 radar metrics (for parity checks). Peak RSS comes from a psutil
sampling thread (per module) and from ``ru_maxrss`` (whole process) — not
from tracemalloc alone, which misses what C libraries allocate on their own.

The dense baseline is never allowed to push the machine into swap: its peak
is predicted from the matrix size and the run is recorded as
``OOM_EXPECTED`` when the prediction exceeds ``--mem-limit-gb``.

Usage
-----
    python benchmarks/benchmark_pipeline.py --dataset sc_10000x20000_d05 \
        --backends dense scipy_csr scipy_csc h5ad_csr h5ad_csc zarr_csr zarr_csc
    python benchmarks/benchmark_pipeline.py --aggregate
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import threading
import time
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_DIR = HERE / "data"
RESULTS_DIR = HERE / "benchmark_results"
sys.path.insert(0, str(ROOT))

MODULES = ["eval_var", "eval_expr", "eval_compactness", "compare_metrics", "eval_stan"]

# Peak bytes per matrix element of the dense reference pipeline: float32
# DataFrame (4) + float64 copy (8) + NaN mask (1) + na.omit copy (8) +
# np.median working copy (8), all alive together in eval_expr. Confirmed by
# the measured runs (see README).
DENSE_PEAK_BYTES_PER_ELEMENT = 29
# Today's DataFrame path: float32 frame (4) + the median's value pool (4) +
# bounded row blocks. Measured 8.6-9.5 B/element; guard with some margin.
DATAFRAME_PEAK_BYTES_PER_ELEMENT = 11


class RssSampler:
    """Background thread tracking the process RSS high-water mark."""

    def __init__(self, interval: float = 0.01):
        import psutil
        self._proc = psutil.Process()
        self._interval = interval
        self._peak = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            self._peak = max(self._peak, self._proc.memory_info().rss)
            time.sleep(self._interval)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join()

    def rss(self) -> int:
        return self._proc.memory_info().rss

    def reset_peak(self) -> None:
        self._peak = self._proc.memory_info().rss

    def peak(self) -> int:
        self._peak = max(self._peak, self._proc.memory_info().rss)
        return self._peak


def max_rss_bytes() -> int:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(r) if sys.platform == "darwin" else int(r) * 1024


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------

def storage_path(ds_dir: Path, backend: str) -> Path:
    layout = backend.split("+")[0]
    layout = {"dense": "h5ad_csr", "dataframe": "h5ad_csr",
              "scipy_csr": "h5ad_csr", "scipy_csc": "h5ad_csc"}.get(layout, layout)
    return ds_dir / (f"{layout}.h5ad" if layout.startswith("h5ad") else f"{layout}.zarr")


def load_input(ds_dir: Path, backend: str, chunk_nnz: int | None):
    """Return (pipeline_input, keepalive). Loading is timed by the caller and
    is *not* part of the pipeline time."""
    import anndata as ad
    import pandas as pd

    path = storage_path(ds_dir, backend)
    if backend in ("dense", "dataframe"):
        adata = ad.read_h5ad(path)
        arr = adata.X.T.toarray()  # genes x cells, float32; sparse -> dense of the input only
        return pd.DataFrame(arr, index=adata.var_names, columns=adata.obs_names, copy=False), None

    from pysigqc_metrics.backends import AnnDataBackend, SparseBackend
    kw = {"chunk_nnz": chunk_nnz} if chunk_nnz else {}
    if backend in ("scipy_csr", "scipy_csc"):
        adata = ad.read_h5ad(path)
        return AnnDataBackend(adata, **kw), adata
    if backend.endswith("+dask"):
        adata = ad.experimental.read_lazy(path)
        return AnnDataBackend(adata, **kw), adata
    if backend.startswith("h5ad"):
        adata = ad.read_h5ad(path, backed="r")
        return AnnDataBackend(adata, **kw), adata
    if backend.startswith("zarr"):
        import zarr
        from anndata.io import read_elem, sparse_dataset
        g = zarr.open_group(path, mode="r")
        x = sparse_dataset(g["X"])
        genes = read_elem(g["var"]).index
        cells = read_elem(g["obs"]).index
        return SparseBackend(x, genes, cells, gene_axis=1, **kw), g
    raise ValueError(f"unknown backend {backend!r}")


# ---------------------------------------------------------------------------
# Worker: one run in this process
# ---------------------------------------------------------------------------

def run_worker(args) -> dict:
    ds_dir = DATA_DIR / args.dataset
    meta = json.loads((ds_dir / "meta.json").read_text())
    sigs = json.loads((ds_dir / "signatures.json").read_text())
    names_sigs = list(sigs)
    backend = args.backends[0]
    path = storage_path(ds_dir, backend)

    rec: dict = {
        "dataset": args.dataset, "backend": backend, "impl": args.impl,
        "mode": args.mode, "tag": args.tag,
        "cells": meta["n_cells"], "genes": meta["n_genes"], "nnz": meta["nnz"],
        "density": meta["density"],
        "input_matrix_bytes_dense_f64": meta["n_cells"] * meta["n_genes"] * 8,
        "input_matrix_bytes_sparse": meta["nnz"] * 8 + (meta["n_cells"] + 1) * 8,
        "storage_bytes": _size(path),
        "n_signatures": len(sigs),
        "n_signature_genes_union": len({g for v in sigs.values() for g in v}),
        "mem_limit_gb": args.mem_limit_gb,
        "chunk_nnz": args.chunk_nnz,
        "platform": platform.platform(), "python": platform.python_version(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    if backend in ("dense", "dataframe"):
        per_element = DENSE_PEAK_BYTES_PER_ELEMENT if backend == "dense" else DATAFRAME_PEAK_BYTES_PER_ELEMENT
        predicted = meta["n_cells"] * meta["n_genes"] * per_element
        rec["predicted_peak_bytes"] = predicted
        if predicted > args.mem_limit_gb * 1e9:
            rec["status"] = "OOM_EXPECTED"
            return rec

    warnings.simplefilter("ignore")
    np.seterr(all="ignore")
    if args.no_fast_paths:
        os.environ["PYSIGQC_EXACT_FAST_PATHS"] = "0"   # read at import time
    rec["fast_paths"] = not args.no_fast_paths and args.impl == "current"
    rec["share_cache"] = bool(args.share_cache)
    rec["malloc_large_cache"] = os.environ.get("MallocLargeCache", "1") != "0"
    sampler = RssSampler().start()

    t0 = time.perf_counter()
    data, _keepalive = load_input(ds_dir, backend, args.chunk_nnz)
    rec["load_seconds"] = time.perf_counter() - t0
    rec["rss_after_load"] = sampler.rss()

    if args.impl == "reference":
        from benchmarks import reference_impl as impl
    else:
        import pysigqc_metrics as impl

    mats = {"ds": data}
    names_ds = ["ds"]
    radar: dict = {s: {} for s in names_sigs}
    sub_timings: dict = {}

    t_total = time.perf_counter()
    if args.mode == "pipeline":
        kwargs = {}
        if args.impl == "current":      # the frozen reference has neither option
            kwargs["share_cache"] = bool(args.share_cache)
            if args.stats_cache:
                kwargs["stats_cache"] = args.stats_cache
        sampler.reset_peak()
        out = impl.run_pipeline(sigs, names_sigs, mats, names_ds, **kwargs)
        for mod, key in zip(MODULES, ["var_result", "expr_result", "compact_result",
                                      "metrics_result", "stan_result"]):
            rec[f"{mod}_seconds"] = out[key]["elapsed_seconds"]
            sub_timings[mod] = out[key].get("timings", {})
        for s in names_sigs:
            radar[s] = out["radar_values"][s]["ds"]
        sub_timings["cache"] = out.get("cache_timings", {})
    else:
        calls = {
            "eval_var": impl.compute_var, "eval_expr": impl.compute_expr,
            "eval_compactness": impl.compute_compactness,
            "compare_metrics": impl.compute_metrics, "eval_stan": impl.compute_stan,
        }
        for mod in MODULES:
            sampler.reset_peak()
            t0 = time.perf_counter()
            res = calls[mod](sigs, names_sigs, mats, names_ds)
            rec[f"{mod}_seconds"] = time.perf_counter() - t0
            rec[f"{mod}_peak_rss"] = sampler.peak()
            sub_timings[mod] = res.get("timings", {})
            for s in names_sigs:
                radar[s].update(res["radar_values"][s]["ds"])
            del res
    rec["total_seconds"] = time.perf_counter() - t_total
    sampler.stop()

    rec["peak_rss"] = max_rss_bytes()
    rec["backend_timings"] = dict(getattr(data, "timings", {}))
    rec["sub_timings"] = sub_timings
    rec["radar"] = {s: {k: _num(v) for k, v in radar[s].items()} for s in names_sigs}
    rec["status"] = "OK"
    return rec


def _num(v):
    v = float(v)
    return None if np.isnan(v) else v


def _size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def result_path(dataset: str, backend: str, tag: str) -> Path:
    suffix = f"__{tag}" if tag else ""
    return RESULTS_DIR / f"{dataset}__{backend.replace('+', '-')}{suffix}.json"


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_driver(args) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    for backend in args.backends:
        out = result_path(args.dataset, backend, args.tag)
        if out.exists() and not args.force:
            print(f"[bench] {out.name}: cached")
            continue
        impl = "reference" if backend == "dense" else args.impl
        cmd = [sys.executable, str(Path(__file__).resolve()), "--worker",
               "--dataset", args.dataset, "--backends", backend, "--impl", impl,
               "--mode", args.mode, "--tag", args.tag,
               "--mem-limit-gb", str(args.mem_limit_gb)]
        if args.chunk_nnz:
            cmd += ["--chunk-nnz", str(args.chunk_nnz)]
        if args.share_cache:
            cmd += ["--share-cache"]
        if args.no_fast_paths:
            cmd += ["--no-fast-paths"]
        if args.stats_cache:
            cmd += ["--stats-cache", args.stats_cache]
        env = dict(os.environ)
        if args.no_malloc_cache:
            env["MallocLargeCache"] = "0"
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=args.timeout, cwd=ROOT, env=env)
        if proc.returncode != 0:
            rec = {"dataset": args.dataset, "backend": backend, "tag": args.tag,
                   "status": "ERROR", "stderr": proc.stderr[-4000:]}
        else:
            rec = json.loads(proc.stdout.strip().splitlines()[-1])
        out.write_text(json.dumps(rec, indent=1))
        msg = rec["status"]
        if rec["status"] == "OK":
            msg += (f"  total={rec['total_seconds']:.1f}s  peak={rec['peak_rss'] / 1e9:.2f}GB  "
                    + " ".join(f"{m.split('_')[-1]}={rec[m + '_seconds']:.1f}" for m in MODULES))
        elif rec["status"] == "ERROR":
            msg += "\n" + rec["stderr"][-1500:]
        print(f"[bench] {args.dataset} {backend:<16} {msg}  (wall {time.perf_counter() - t0:.0f}s)",
              flush=True)


# ---------------------------------------------------------------------------
# Aggregation + parity
# ---------------------------------------------------------------------------

PARITY_RTOL = 1e-9
PARITY_ATOL = 1e-12


def _radar_vector(rec) -> np.ndarray:
    return np.array([[np.nan if v is None else v for v in sig.values()]
                     for sig in rec["radar"].values()], dtype=float)


def aggregate() -> None:
    import pandas as pd

    recs = [json.loads(p.read_text()) for p in sorted(RESULTS_DIR.glob("*.json"))]
    by_ds: dict = {}
    for r in recs:
        by_ds.setdefault(r["dataset"], []).append(r)

    rows = []
    for ds, group in by_ds.items():
        ok = {r["backend"]: r for r in group if r.get("status") == "OK" and not r.get("tag")}
        # Reference: the dense run when it exists, otherwise in-memory scipy
        # CSR (itself verified against dense on every dataset where dense fits).
        ref_name = "dense" if "dense" in ok else ("scipy_csr" if "scipy_csr" in ok else None)
        ref = _radar_vector(ok[ref_name]) if ref_name else None
        for r in group:
            row = {k: v for k, v in r.items()
                   if not isinstance(v, (dict, list)) and k != "stderr"}
            if r.get("status") == "OK" and ref is not None and "radar" in r:
                vec = _radar_vector(r)
                same = np.allclose(vec, ref, rtol=PARITY_RTOL, atol=PARITY_ATOL, equal_nan=True)
                row["parity_ref"] = ref_name
                row["parity_max_abs_diff"] = float(np.nanmax(np.abs(vec - ref))) if vec.size else 0.0
                row["numerical_parity"] = ("reference" if r["backend"] == ref_name and not r.get("tag")
                                           else "PASS" if same else "FAIL")
            for k, v in r.get("backend_timings", {}).items():
                row[f"backend_{k}_seconds"] = v
            for mod, d in r.get("sub_timings", {}).items():
                for k, v in d.items():
                    row[f"{mod}__{k}_seconds"] = v
            rows.append(row)

    df = pd.DataFrame(rows).sort_values(["genes", "cells", "density", "backend", "tag"])
    out = HERE / "results.csv"
    df.to_csv(out, index=False)
    print(f"[aggregate] {len(df)} runs -> {out}")
    show = ["dataset", "backend", "tag", "status", "total_seconds", "peak_rss", "numerical_parity"]
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        d = df[[c for c in show if c in df.columns]].copy()
        if "peak_rss" in d:
            d["peak_rss"] = (d["peak_rss"] / 1e9).round(2)
        print(d.to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset")
    ap.add_argument("--backends", nargs="+", default=["dense"])
    ap.add_argument("--impl", choices=["current", "reference"], default="current")
    ap.add_argument("--mode", choices=["modules", "pipeline"], default="modules",
                    help="modules: call the 5 modules one by one (per-module peak RSS); "
                         "pipeline: call run_pipeline()")
    ap.add_argument("--share-cache", action="store_true",
                    help="pipeline mode only: enable the shared DatasetStatsCache")
    ap.add_argument("--no-malloc-cache", action="store_true",
                    help="macOS: run the worker with MallocLargeCache=0. libmalloc keeps freed "
                         "large blocks resident, which inflates RSS by GBs; without that cache "
                         "peak RSS is the real working set")
    ap.add_argument("--stats-cache", default="",
                    help="pipeline mode only: directory for persisted per-gene statistics "
                         "(run_pipeline(stats_cache=...))")
    ap.add_argument("--no-fast-paths", action="store_true",
                    help="disable the exact fast median / sparse-aware ranking "
                         "(PYSIGQC_EXACT_FAST_PATHS=0), i.e. the state of the first benchmarks")
    ap.add_argument("--tag", default="", help="label for variant runs (kept apart in results)")
    ap.add_argument("--mem-limit-gb", type=float,
                    default=float(os.environ.get("PYSIGQC_BENCH_MEM_LIMIT_GB", 40)))
    ap.add_argument("--chunk-nnz", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=6 * 3600)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()

    if args.aggregate:
        aggregate()
    elif args.worker:
        print(json.dumps(run_worker(args)))
    else:
        if not args.dataset:
            ap.error("--dataset is required")
        run_driver(args)
        aggregate()


if __name__ == "__main__":
    main()
