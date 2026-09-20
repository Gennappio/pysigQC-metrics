"""Prepare a public single-cell H5AD (CELLxGENE schema) for the benchmarks.

Produces, under ``benchmarks/data/<name>/``, the same layout the synthetic
generator writes (``h5ad_csr.h5ad``, ``signatures.json``, ``meta.json``), so
``benchmark_pipeline.py`` runs on it unchanged. In addition the file *as
downloaded* (gzip-compressed CSR with small HDF5 chunks) is exposed as the
``h5ad_orig`` layout when its X is already log-normalised.

Preprocessing is limited to what the methodology requires and is recorded in
``meta.json``:
    --x-is counts   library-size normalisation to 10,000 + log1p, streamed
                    chunk by chunk (the matrix is never fully in memory);
    --x-is lognorm  X is used as is.
No filtering of cells or genes, no batch correction.

Datasets used in the report (CELLxGENE Discover, CC-BY):
    covid_lung_116k  https://datasets.cellxgene.cziscience.com/1792df55-7cfe-4564-afcf-4bb3ffb73bc0.h5ad
                     "A molecular single-cell lung atlas of lethal COVID-19", X = log-normalised
    onek1k_1p25M     https://datasets.cellxgene.cziscience.com/1e44db10-b572-46cc-adae-dcc7acd44ca6.h5ad
                     "Single-cell eQTL mapping identifies cell type specific genetic control
                     of autoimmune disease" (OneK1K), X = raw counts
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from generate_singlecell import DATA_DIR, _dir_size, derive

# Curated gene-symbol signatures (well-known marker / programme genes).
SIGNATURES = {
    "ifn_response": [
        "ISG15", "IFI6", "IFI27", "IFI44", "IFI44L", "IFIT1", "IFIT2", "IFIT3", "IFITM1",
        "IFITM3", "MX1", "MX2", "OAS1", "OAS2", "OAS3", "OASL", "RSAD2", "STAT1", "IRF7",
        "XAF1", "EPSTI1", "LY6E", "HERC5", "CMPK2", "SAMD9L", "DDX58", "IFIH1", "USP18",
        "BST2", "PLSCR1"],
    "cytotoxic_t_nk": [
        "GZMA", "GZMB", "GZMH", "GZMK", "GZMM", "PRF1", "NKG7", "GNLY", "CST7", "CTSW",
        "KLRD1", "KLRB1", "KLRF1", "KLRK1", "CD8A", "CD8B", "CCL5", "CCL4", "FGFBP2", "FCGR3A"],
    "cell_cycle_g2m": [
        "HMGB2", "CDK1", "NUSAP1", "UBE2C", "BIRC5", "TPX2", "TOP2A", "NDC80", "CKS2", "NUF2",
        "CKS1B", "MKI67", "TMPO", "CENPF", "TACC3", "SMC4", "CCNB2", "CKAP2L", "CKAP2", "AURKB",
        "BUB1", "KIF11", "ANP32E", "TUBB4B", "GTSE1", "KIF20B", "HJURP", "CDCA3", "CDC20", "TTK",
        "CDC25C", "KIF2C", "RANGAP1", "NCAPD2", "DLGAP5", "CDCA2", "CDCA8", "ECT2", "KIF23", "HMMR",
        "AURKA", "PSRC1", "ANLN", "LBR", "CKAP5", "CENPE", "CTCF", "NEK2", "G2E3", "CBX5", "CENPA"],
    "mhc_class_ii": [
        "HLA-DRA", "HLA-DRB1", "HLA-DRB5", "HLA-DPA1", "HLA-DPB1", "HLA-DQA1", "HLA-DQB1",
        "HLA-DMA", "HLA-DMB", "HLA-DOA", "HLA-DOB", "CD74", "CIITA", "CTSS", "IFI30", "LGMN",
        "CD4", "RFX5", "RFXAP", "RFXANK"],
    "inflammatory_myeloid": [
        "S100A8", "S100A9", "S100A12", "LYZ", "CD14", "FCN1", "VCAN", "IL1B", "CXCL8", "TNF",
        "CCL2", "CCL3", "CCL7", "NFKBIA", "NFKB1", "PTGS2", "SOD2", "NLRP3", "TLR2", "TLR4",
        "FCGR1A", "CSF3R", "TREM1", "PLAUR", "IER3", "ICAM1", "IL6", "CXCL2", "CXCL3", "OSM"],
    "ribosomal": [f"RPL{i}" for i in (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 19, 21, 22,
                                     23, 24, 26, 27, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39)]
                 + [f"RPS{i}" for i in (2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19,
                                        20, 21, 23, 24, 25, 26, 27, 28, 29)],
    "hypoxia": [
        "VEGFA", "SLC2A1", "PGK1", "LDHA", "CA9", "ADM", "BNIP3", "BNIP3L", "NDRG1", "ENO1",
        "ALDOA", "P4HA1", "EGLN3", "ANKRD37", "KDM3A", "PDK1", "HK2", "PFKFB3", "DDIT4", "MXI1",
        "HILPDA", "ERO1A", "TPI1", "GAPDH", "PGAM1", "ANGPTL4", "LOX", "STC2", "IGFBP3", "HIF1A"],
    "b_plasma": [
        "CD79A", "CD79B", "MS4A1", "CD19", "BANK1", "IGHM", "IGHD", "IGKC", "JCHAIN", "MZB1",
        "XBP1", "SDC1", "TNFRSF17", "PRDM1", "IGHG1", "IGHA1", "CD22", "FCRLA", "TCL1A", "VPREB3"],
    # B-cell genes again plus T-cell genes: overlaps with two other signatures.
    "lymphoid_mixed": [
        "CD3D", "CD3E", "CD3G", "CD2", "CD5", "CD7", "IL7R", "CCR7", "LTB", "TCF7", "LEF1",
        "SELL", "CD79A", "MS4A1", "CD19", "GZMK", "CCL5", "NKG7", "CD8A", "KLRB1"],
}


def normalise_streaming(src: Path, dst: Path, chunk_cells: int = 50_000) -> int:
    """counts -> log1p(counts / libsize * 1e4), float32 CSR, streamed to ``dst``."""
    import anndata as ad
    import h5py
    from anndata.io import sparse_dataset, write_elem

    a = ad.read_h5ad(src, backed="r")
    n_cells = a.n_obs
    ad.AnnData(obs=pd.DataFrame(index=a.obs_names.astype(str)),
               var=pd.DataFrame({"feature_name": a.var["feature_name"].astype(str).to_numpy()},
                                index=a.var_names.astype(str))).write_h5ad(dst)
    nnz, dset, t0 = 0, None, time.perf_counter()
    with h5py.File(dst, "r+") as f:
        if "X" in f:
            del f["X"]
        for start in range(0, n_cells, chunk_cells):
            m = a.X[start:start + chunk_cells].tocsr().astype(np.float32)
            lib = np.asarray(m.sum(axis=1)).ravel()
            lib[lib == 0] = 1.0
            m.data *= np.repeat((1e4 / lib).astype(np.float32), np.diff(m.indptr))
            np.log1p(m.data, out=m.data)
            m.indices = m.indices.astype(np.int32)
            nnz += m.nnz
            if dset is None:
                write_elem(f, "X", m)
                dset = sparse_dataset(f["X"])
            else:
                dset.append(m)
            print(f"[prepare] normalised {min(start + chunk_cells, n_cells):,d}/{n_cells:,d} "
                  f"cells ({time.perf_counter() - t0:.0f}s)", flush=True)
    a.file.close()
    return nnz


def rewrite_streaming(src: Path, dst: Path, chunk_cells: int = 50_000) -> int:
    """Copy X unchanged into an uncompressed anndata-default H5AD."""
    import anndata as ad
    import h5py
    from anndata.io import sparse_dataset, write_elem

    a = ad.read_h5ad(src, backed="r")
    ad.AnnData(obs=pd.DataFrame(index=a.obs_names.astype(str)),
               var=pd.DataFrame({"feature_name": a.var["feature_name"].astype(str).to_numpy()},
                                index=a.var_names.astype(str))).write_h5ad(dst)
    nnz, dset = 0, None
    with h5py.File(dst, "r+") as f:
        if "X" in f:
            del f["X"]
        for start in range(0, a.n_obs, chunk_cells):
            m = a.X[start:start + chunk_cells].tocsr()
            m.indices = m.indices.astype(np.int32)
            nnz += m.nnz
            if dset is None:
                write_elem(f, "X", m)
                dset = sparse_dataset(f["X"])
            else:
                dset.append(m)
    a.file.close()
    return nnz


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", type=Path, help="downloaded .h5ad")
    ap.add_argument("--name", required=True)
    ap.add_argument("--x-is", choices=["counts", "lognorm"], required=True)
    ap.add_argument("--derive", nargs="*", default=["h5ad_csc", "zarr_csr", "zarr_csc"])
    args = ap.parse_args()

    import anndata as ad

    out_dir = DATA_DIR / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "h5ad_csr.h5ad"
    t0 = time.perf_counter()
    if not target.exists():
        if args.x_is == "counts":
            nnz = normalise_streaming(args.source, target)
        else:
            nnz = rewrite_streaming(args.source, target)
            link = out_dir / "h5ad_orig.h5ad"
            if not link.exists():
                os.symlink(args.source.resolve(), link)

        a = ad.read_h5ad(target, backed="r")
        symbol_to_id = dict(zip(a.var["feature_name"].astype(str), a.var_names))
        sigs = {name: [symbol_to_id.get(sym, f"MISSING_{sym}") for sym in genes]
                for name, genes in SIGNATURES.items()}
        found = {n: sum(not g.startswith("MISSING_") for g in v) for n, v in sigs.items()}
        meta = {
            "name": args.name, "source": str(args.source.name), "n_cells": a.n_obs,
            "n_genes": a.n_vars, "nnz": int(nnz), "density": nnz / (a.n_obs * a.n_vars),
            "dtype": "float32",
            "anndata_layer": "X",
            "preprocessing": ("library-size normalisation to 1e4 + log1p (computed here, streamed)"
                              if args.x_is == "counts" else "none: X already log-normalised by the authors"),
            "signature_genes_found": found,
            "prepare_seconds": time.perf_counter() - t0,
        }
        a.file.close()
        (out_dir / "signatures.json").write_text(json.dumps(sigs, indent=1))
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
        print(f"[prepare] {args.name}: {meta['n_cells']:,d} x {meta['n_genes']:,d}, "
              f"density {meta['density']:.4f}; signature genes found: {found}")

    if args.derive:
        derive(out_dir, args.derive)
    meta = json.loads((out_dir / "meta.json").read_text())
    if (out_dir / "h5ad_orig.h5ad").exists():
        meta.setdefault("storage_bytes", {})["h5ad_orig"] = _dir_size((out_dir / "h5ad_orig.h5ad").resolve())
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
