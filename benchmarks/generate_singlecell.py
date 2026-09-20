"""Synthetic single-cell dataset generator for the scalability benchmarks.

The matrix is generated directly in sparse form, one chunk of cells at a
time, and appended to an on-disk H5AD (cells x genes, CSR). The full dense
matrix is never materialised: the largest temporary is one
``chunk_cells x n_genes`` float32 block.

Model (per chunk of cells):
    * every gene has a base detection probability ``p_g`` (heavy-tailed,
      rescaled so that the mean equals the requested density);
    * every cell has a size factor ``s_c`` (log-normal);
    * "program" signatures share a latent per-cell activity that modulates
      both detection probability and magnitude of their genes, so that
      signature genes are genuinely co-expressed;
    * expressed entries are ``log1p(Gamma)`` — strictly positive, right
      skewed, float32. Everything else is an implicit zero.

Outputs, under ``<out>/<name>/``:
    h5ad_csr.h5ad     canonical dataset (AnnData, X = CSR cells x genes)
    signatures.json   gene signatures (sizes 20/50/100, partial overlap)
    meta.json         shape, nnz, density, seed

Other layouts (H5AD-CSC, Zarr CSR/CSC, chunk variants) are derived from the
canonical file with ``--derive`` so that every backend sees the same matrix.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

DATA_DIR = Path(__file__).resolve().parent / "data"

SIG_SIZES = [20, 50, 100, 20, 50, 100, 20, 50, 100]
N_PROGRAM_SIGS = 6          # first 6 signatures are co-expressed programs
OVERLAP_FRACTION = 0.2      # share of genes borrowed from the previous signature
N_MISSING_GENES = 3         # genes absent from the dataset, added to one signature


def dataset_name(n_cells: int, n_genes: int, density: float) -> str:
    return f"sc_{n_cells}x{n_genes}_d{int(round(density * 100)):02d}"


def make_gene_model(n_genes: int, density: float, rng: np.random.Generator):
    """Per-gene detection probability and magnitude scale."""
    raw = np.exp(rng.normal(0.0, 1.5, n_genes))
    p = raw / raw.mean() * density
    for _ in range(20):  # rescale after clipping so that mean(p) == density
        p = np.clip(p, 0.0, 0.95)
        p *= density / p.mean()
    p = np.clip(p, 0.0, 0.95)
    # Magnitude correlates with detection rate, as in real scRNA-seq.
    mu = np.exp(0.5 * np.log(p / p.mean()) + rng.normal(0.0, 0.5, n_genes)) * 2.0
    return p.astype(np.float32), mu.astype(np.float32)


def make_signatures(gene_names: np.ndarray, p_gene: np.ndarray,
                    rng: np.random.Generator) -> dict[str, list[str]]:
    """Signatures of 20/50/100 genes with partial overlap between neighbours."""
    n_genes = len(gene_names)
    # Program signatures draw from reasonably detected genes; random ones from all.
    expressed_pool = np.flatnonzero(p_gene >= np.quantile(p_gene, 0.5))
    sigs: dict[str, list[str]] = {}
    prev: np.ndarray | None = None
    for i, size in enumerate(SIG_SIZES):
        pool = expressed_pool if i < N_PROGRAM_SIGS else np.arange(n_genes)
        size = min(size, len(pool))
        n_shared = int(round(size * OVERLAP_FRACTION)) if prev is not None else 0
        n_shared = min(n_shared, len(prev)) if prev is not None else 0
        shared = rng.choice(prev, n_shared, replace=False) if n_shared else np.array([], dtype=int)
        fresh_pool = np.setdiff1d(pool, shared)
        fresh = rng.choice(fresh_pool, size - n_shared, replace=False)
        idx = np.concatenate([shared, fresh]).astype(int)
        rng.shuffle(idx)
        kind = "prog" if i < N_PROGRAM_SIGS else "rand"
        sigs[f"sig{i + 1:02d}_{kind}_{size}"] = [str(g) for g in gene_names[idx]]
        prev = idx
    # Exercise the missing-gene path in one signature.
    first = next(iter(sigs))
    sigs[first] = sigs[first] + [f"MISSING_{k}" for k in range(N_MISSING_GENES)]
    return sigs


def generate_chunk(n_cells: int, p_gene: np.ndarray, mu_gene: np.ndarray,
                   program_cols: list[np.ndarray],
                   rng: np.random.Generator) -> sp.csr_matrix:
    """One CSR chunk (n_cells x n_genes), float32."""
    n_genes = p_gene.shape[0]
    size = rng.lognormal(0.0, 0.35, n_cells).astype(np.float32)
    size /= np.float32(np.exp(0.35 ** 2 / 2))
    prob = np.outer(size, p_gene)                       # chunk x genes, float32
    scale = np.ones((n_cells, n_genes), dtype=np.float32) * size[:, None]
    for cols in program_cols:
        act = rng.gamma(0.7, 1.0, n_cells).astype(np.float32)
        boost = (1.0 + 2.0 * act) / np.float32(1.0 + 2.0 * 0.7)
        prob[:, cols] *= boost[:, None]
        scale[:, cols] *= boost[:, None]
    np.clip(prob, 0.0, 1.0, out=prob)
    mask = rng.random((n_cells, n_genes), dtype=np.float32) < prob
    rows, cols = np.nonzero(mask)                       # row-major order => CSR order
    vals = rng.gamma(2.0, 0.5, rows.size).astype(np.float32)
    vals *= mu_gene[cols] * scale[rows, cols]
    vals = np.log1p(vals, dtype=np.float32)
    np.maximum(vals, np.float32(1e-3), out=vals)        # never store an explicit 0
    indptr = np.zeros(n_cells + 1, dtype=np.int64)
    np.cumsum(mask.sum(axis=1), out=indptr[1:])
    return sp.csr_matrix((vals, cols.astype(np.int32), indptr),
                         shape=(n_cells, n_genes))


def generate(n_cells: int, n_genes: int, density: float, out_root: Path,
             seed: int = 0, chunk_cells: int = 2000, force: bool = False) -> Path:
    import anndata as ad
    import h5py
    from anndata.io import sparse_dataset, write_elem

    out_dir = out_root / dataset_name(n_cells, n_genes, density)
    h5_path = out_dir / "h5ad_csr.h5ad"
    if h5_path.exists() and (out_dir / "meta.json").exists() and not force:
        print(f"[generate] {out_dir.name}: already present, skipping")
        return out_dir
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    gene_names = np.array([f"G{g:05d}" for g in range(n_genes)])
    p_gene, mu_gene = make_gene_model(n_genes, density, rng)
    sigs = make_signatures(gene_names, p_gene, rng)
    name_to_col = {g: i for i, g in enumerate(gene_names)}
    program_cols = [
        np.array([name_to_col[g] for g in genes if g in name_to_col])
        for genes in list(sigs.values())[:N_PROGRAM_SIGS]
    ]

    obs = pd.DataFrame(index=[f"C{c:07d}" for c in range(n_cells)])
    var = pd.DataFrame(index=gene_names)
    ad.AnnData(obs=obs, var=var).write_h5ad(h5_path)

    nnz = 0
    with h5py.File(h5_path, "r+") as f:
        if "X" in f:
            del f["X"]
        dset = None
        for start in range(0, n_cells, chunk_cells):
            n = min(chunk_cells, n_cells - start)
            chunk = generate_chunk(n, p_gene, mu_gene, program_cols, rng)
            nnz += chunk.nnz
            if dset is None:
                write_elem(f, "X", chunk)
                dset = sparse_dataset(f["X"])
            else:
                dset.append(chunk)
            if (start // chunk_cells) % 50 == 0:
                print(f"[generate] {out_dir.name}: {start + n:>9,d}/{n_cells:,d} cells "
                      f"({time.perf_counter() - t0:6.1f}s)", flush=True)

    (out_dir / "signatures.json").write_text(json.dumps(sigs, indent=1))
    meta = {
        "name": out_dir.name, "n_cells": n_cells, "n_genes": n_genes,
        "target_density": density, "nnz": int(nnz),
        "density": nnz / (n_cells * n_genes), "dtype": "float32", "seed": seed,
        "generation_seconds": time.perf_counter() - t0,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"[generate] {out_dir.name}: nnz={nnz:,d} density={meta['density']:.4f} "
          f"in {meta['generation_seconds']:.1f}s")
    return out_dir


# ---------------------------------------------------------------------------
# Derived layouts
# ---------------------------------------------------------------------------

def _dir_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def cells_to_elements(cells_per_chunk: int, meta: dict) -> int:
    """Zarr/HDF5 chunk length (elements of data/indices) holding ~N cells."""
    return max(1, int(round(cells_per_chunk * meta["nnz"] / meta["n_cells"])))


def genes_to_elements(genes_per_chunk: int, meta: dict) -> int:
    return max(1, int(round(genes_per_chunk * meta["nnz"] / meta["n_genes"])))


def derive(out_dir: Path, layouts: list[str], force: bool = False) -> None:
    """Write other storage layouts of the canonical dataset.

    Layout names:
        h5ad_csc                    H5AD, X = CSC (gene-major)
        h5ad_csr_gzip / h5ad_csc_gzip
        zarr_csr / zarr_csc         Zarr v3, anndata default chunking/compression
        zarr_csr_c{N}k              CSR with chunks of ~N*1000 cells
        zarr_csc_g{N}               CSC with chunks of ~N genes
        zarr_csr_raw / zarr_csc_raw no compression
    """
    import anndata as ad

    meta = json.loads((out_dir / "meta.json").read_text())
    todo = [l for l in layouts
            if force or not (out_dir / _layout_filename(l)).exists()]
    if not todo:
        return
    adata = ad.read_h5ad(out_dir / "h5ad_csr.h5ad")
    x_csr = adata.X
    x_csc = None
    sizes = meta.setdefault("storage_bytes", {})
    sizes["h5ad_csr"] = _dir_size(out_dir / "h5ad_csr.h5ad")
    for layout in todo:
        t0 = time.perf_counter()
        target = out_dir / _layout_filename(layout)
        if target.exists():
            shutil.rmtree(target) if target.is_dir() else target.unlink()
        fmt = "csc" if "_csc" in layout else "csr"
        if fmt == "csc" and x_csc is None:
            x_csc = x_csr.tocsc()
        adata.X = x_csc if fmt == "csc" else x_csr
        if layout.startswith("h5ad"):
            adata.write_h5ad(target, compression="gzip" if layout.endswith("gzip") else None)
        else:
            _write_zarr(adata, target, layout, meta)
        sizes[layout] = _dir_size(target)
        print(f"[derive] {out_dir.name}/{target.name}: "
              f"{sizes[layout] / 1e9:.2f} GB in {time.perf_counter() - t0:.1f}s", flush=True)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))


def _layout_filename(layout: str) -> str:
    return f"{layout}.h5ad" if layout.startswith("h5ad") else f"{layout}.zarr"


def _write_zarr(adata, target: Path, layout: str, meta: dict) -> None:
    import zarr
    from anndata.io import write_elem

    suffix = layout.split("_", 2)[2] if layout.count("_") >= 2 else ""
    kwargs: dict = {}
    if suffix.startswith("c") and suffix.endswith("k"):
        kwargs["chunks"] = (cells_to_elements(int(suffix[1:-1]) * 1000, meta),)
    elif suffix.startswith("g"):
        kwargs["chunks"] = (genes_to_elements(int(suffix[1:]), meta),)
    elif suffix == "raw":
        kwargs["compressors"] = None
    if not kwargs:
        adata.write_zarr(target)
        return
    # Explicit chunking / compression: write without consolidated metadata,
    # replace X, then consolidate (a consolidated store cannot be edited).
    adata.write_zarr(target, consolidate_metadata=False)
    g = zarr.open_group(target, mode="r+", use_consolidated=False)
    del g["X"]
    write_elem(g, "X", adata.X, dataset_kwargs=kwargs)
    zarr.consolidate_metadata(target)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", type=int, nargs="+", default=[1000])
    ap.add_argument("--genes", type=int, default=2000)
    ap.add_argument("--density", type=float, nargs="+", default=[0.05])
    ap.add_argument("--out", type=Path, default=DATA_DIR)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk-cells", type=int, default=2000)
    ap.add_argument("--derive", nargs="*", default=None,
                    help="also write these layouts (default when flag given "
                         "without values: h5ad_csc zarr_csr zarr_csc)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    for n_cells in args.cells:
        for density in args.density:
            out_dir = generate(n_cells, args.genes, density, args.out,
                               seed=args.seed, chunk_cells=args.chunk_cells,
                               force=args.force)
            if args.derive is not None:
                layouts = args.derive or ["h5ad_csc", "zarr_csr", "zarr_csc"]
                derive(out_dir, layouts, force=args.force)


if __name__ == "__main__":
    main()
