"""Where does the memory peak come from? One module per fresh process.

``benchmark_pipeline.py`` reports per-module peaks inside one process, where a
module inherits whatever the allocator kept from the previous one. Here every
module runs alone in its own subprocess, so ``ru_maxrss`` is that module's own
requirement (input handle included).

    python benchmarks/memory_probe.py sc_1000000x20000_d05 h5ad_csr
    python benchmarks/memory_probe.py sc_50000x20000_d05 dense --tag after

Two numbers per step:
    peak RSS         what the OS reports. On macOS libmalloc keeps freed large
                     blocks resident (its "large cache"), which inflates RSS by
                     GBs; ``--no-malloc-cache`` reruns with MallocLargeCache=0.
    allocated peak   high-water mark of live allocations (tracemalloc, which
                     sees NumPy buffers) — the real requirement, input included.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

STEPS = ["eval_var", "eval_expr", "eval_compactness", "compare_metrics", "eval_stan", "pipeline"]


def worker(dataset: str, backend: str, step: str, trace: bool) -> dict:
    import pysigqc_metrics as pq
    from benchmark_pipeline import DATA_DIR, load_input, max_rss_bytes

    warnings.simplefilter("ignore")
    np.seterr(all="ignore")
    import tracemalloc

    ds_dir = DATA_DIR / dataset
    sigs = json.loads((ds_dir / "signatures.json").read_text())
    names = list(sigs)
    if trace:
        tracemalloc.start()
    data, _keep = load_input(ds_dir, backend, None)
    after_load = max_rss_bytes()
    calls = {
        "eval_var": pq.compute_var, "eval_expr": pq.compute_expr,
        "eval_compactness": pq.compute_compactness,
        "compare_metrics": pq.compute_metrics, "eval_stan": pq.compute_stan,
    }
    t0 = time.perf_counter()
    if step == "pipeline":
        pq.run_pipeline(sigs, names, {"ds": data}, ["ds"])
    else:
        calls[step](sigs, names, {"ds": data}, ["ds"])
    return {"step": step, "seconds": time.perf_counter() - t0,
            "rss_after_load": after_load, "peak_rss": max_rss_bytes(),
            "allocated_peak": tracemalloc.get_traced_memory()[1] if trace else None,
            "malloc_large_cache": os.environ.get("MallocLargeCache", "1") != "0"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("backend")
    ap.add_argument("--steps", nargs="+", default=STEPS)
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-malloc-cache", action="store_true", help="macOS: MallocLargeCache=0")
    ap.add_argument("--trace", action="store_true", help="also record the tracemalloc peak")
    ap.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.worker:
        print(json.dumps(worker(args.dataset, args.backend, args.worker, args.trace)))
        return

    rows = []
    for step in args.steps:
        env = dict(os.environ)
        if args.no_malloc_cache:
            env["MallocLargeCache"] = "0"
        cmd = [sys.executable, __file__, args.dataset, args.backend, "--worker", step]
        proc = subprocess.run(cmd + (["--trace"] if args.trace else []),
                              capture_output=True, text=True, cwd=HERE.parent, env=env)
        if proc.returncode:
            print(f"[probe] {step}: ERROR\n{proc.stderr[-1500:]}")
            continue
        r = json.loads(proc.stdout.strip().splitlines()[-1])
        rows.append(r)
        print(f"[probe] {args.dataset} {args.backend:<10} {step:<17} "
              f"peak RSS={r['peak_rss'] / 1e9:6.2f} GB  (after load {r['rss_after_load'] / 1e9:5.2f} GB)  "
              + (f"allocated peak={r['allocated_peak'] / 1e9:6.2f} GB  " if r.get("allocated_peak") else "")
              + f"{r['seconds']:6.1f} s", flush=True)
    out = HERE / "benchmark_results" / "memory_probe"
    out.mkdir(parents=True, exist_ok=True)
    suffix = f"__{args.tag}" if args.tag else ""
    (out / f"{args.dataset}__{args.backend}{suffix}.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
