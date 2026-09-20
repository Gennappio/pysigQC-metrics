"""Prototype: how fast is the pipeline when per-gene statistics are persisted?

The per-gene mean / SD / NaN count / below-threshold count are 5 vectors of
length n_genes that depend on the matrix only. This script computes them
once, stores them as an ``.npz`` sidecar next to the dataset, and then times
the pipeline with the sidecar loaded — i.e. what a "statistics index" inside
the existing H5AD/Zarr standards would buy, without any custom engine.

    python benchmarks/prototype_stats_sidecar.py sc_1000000x20000_d05 h5ad_csc h5ad_csr zarr_csc
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))


def worker(dataset: str, backend: str) -> dict:
    from benchmark_pipeline import DATA_DIR, RssSampler, load_input, max_rss_bytes
    from pysigqc_metrics import ALL_METRICS, DatasetStatsCache, run_pipeline

    warnings.simplefilter("ignore")
    np.seterr(all="ignore")
    ds_dir = DATA_DIR / dataset
    sigs = json.loads((ds_dir / "signatures.json").read_text())
    sidecar = ds_dir / "gene_stats_sidecar.npz"
    if not sidecar.exists():
        data, keep = load_input(ds_dir, backend, None)
        t0 = time.perf_counter()
        cache = DatasetStatsCache(data)
        cache.prepare(None, None)
        np.savez(sidecar, **cache.export_stats())
        print(f"[sidecar] built in {time.perf_counter() - t0:.1f}s "
              f"({sidecar.stat().st_size / 1e6:.2f} MB)", file=sys.stderr)
        del cache, data, keep

    sampler = RssSampler().start()
    data, keep = load_input(ds_dir, backend, None)
    t0 = time.perf_counter()
    cache = DatasetStatsCache(data)
    cache.seed_stats(dict(np.load(sidecar)))
    out = run_pipeline(sigs, list(sigs), {"ds": cache}, ["ds"])
    total = time.perf_counter() - t0
    sampler.stop()
    radar = [[out["radar_values"][s]["ds"][m] for m in ALL_METRICS] for s in sigs]
    return {"dataset": dataset, "backend": backend, "tag": "stats_sidecar", "status": "OK",
            "total_seconds": total, "peak_rss": max_rss_bytes(),
            "modules": {k: out[k]["elapsed_seconds"] for k in
                        ("var_result", "expr_result", "compact_result", "metrics_result", "stan_result")},
            "backend_timings": dict(data.timings), "radar": radar}


def main() -> None:
    if sys.argv[1] == "--worker":
        print(json.dumps(worker(sys.argv[2], sys.argv[3])))
        return
    dataset, backends = sys.argv[1], sys.argv[2:]
    ref = None
    ref_file = HERE / "benchmark_results" / f"{dataset}__scipy_csr.json"
    if ref_file.exists():
        r = json.loads(ref_file.read_text())
        ref = np.array([[np.nan if v is None else v for v in s.values()] for s in r["radar"].values()])
    for backend in backends:
        proc = subprocess.run([sys.executable, __file__, "--worker", dataset, backend],
                              capture_output=True, text=True, cwd=HERE.parent)
        if proc.returncode:
            print(proc.stderr[-2000:])
            continue
        rec = json.loads(proc.stdout.strip().splitlines()[-1])
        got = np.array(rec.pop("radar"), dtype=float)
        rec["numerical_parity"] = ("PASS" if ref is not None and np.allclose(
            got, ref, rtol=1e-9, atol=1e-12, equal_nan=True) else "n/a" if ref is None else "FAIL")
        out = HERE / "benchmark_results" / f"{dataset}__{backend}__stats_sidecar.json"
        rec.update(cells=None, mode="prototype")
        meta = json.loads((HERE / "data" / dataset / "meta.json").read_text())
        rec.update(cells=meta["n_cells"], genes=meta["n_genes"], nnz=meta["nnz"], density=meta["density"])
        out.write_text(json.dumps(rec, indent=1))
        print(f"[sidecar] {dataset} {backend:<10} total={rec['total_seconds']:.1f}s "
              f"peak={rec['peak_rss'] / 1e9:.2f}GB parity={rec['numerical_parity']} "
              + " ".join(f"{k.split('_')[0]}={v:.1f}" for k, v in rec["modules"].items()))


if __name__ == "__main__":
    main()
