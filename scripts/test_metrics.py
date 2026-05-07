"""Run the pysigqc-metrics radar pipeline on a TCGA matrix + GMT signatures.

Standalone CLI: takes TCGA-style inputs (Ensembl-indexed expression matrix,
phenotype TSV, MSigDB GMT), runs the full pysigqc-metrics radar pipeline and
prints the resulting radar table + areas. No R dependency, no sibling-repo
dependency (uses ``_io_helpers`` co-located in this scripts/ folder).

Example:
    python pysigQC-metrics/scripts/test_metrics.py \\
        --expr      tcga_RSEM_gene_tpm.gz \\
        --phenotype TCGA_phenotype_denseDataOnlyDownload.tsv \\
        --gmt       BUFFA_HYPOXIA_METAGENE.v2026.1.Hs.gmt \\
        --sample-limit 100 \\
        --plot      ./radar_out
"""

from __future__ import annotations

import argparse
import gzip
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
# Make the package importable when running as a plain script (no install).
sys.path.insert(0, str(HERE.parent))
# Co-located helpers; HERE is on sys.path[0] automatically when run as script.
sys.path.insert(0, str(HERE))

from _io_helpers import (  # noqa: E402
    convert_ensembl_matrix_to_hgnc,
    parse_gmt_signatures,
)
from pysigqc_metrics.pipeline import run_pipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--expr", type=Path, required=True,
                   help="TCGA expression matrix (TSV or .gz, Ensembl rows).")
    p.add_argument("--phenotype", type=Path, required=True,
                   help="TCGA phenotype TSV (sample, sample_type, _primary_disease).")
    p.add_argument("--gmt", type=Path, required=True,
                   help="MSigDB-style GMT signature file.")
    p.add_argument("--sample-type", default="Primary Tumor",
                   help='Phenotype sample_type filter (default "Primary Tumor").')
    p.add_argument("--cancer-type", default="breast invasive carcinoma",
                   help='Phenotype _primary_disease filter '
                        '(default "breast invasive carcinoma").')
    p.add_argument("--dataset-name", default=None,
                   help="Dataset id (default derived from filters, e.g. BIC_PrimaryTumor).")
    p.add_argument("--sample-limit", type=int, default=0,
                   help="Cap number of samples (0 = all, default).")
    p.add_argument("--signature", action="append", default=[],
                   help="Restrict to specific signature names from the GMT (repeatable).")
    p.add_argument("--cache-dir", type=Path, default=None,
                   help="Working directory for preprocessed inputs "
                        "(default: pysigQC-metrics/scripts/test_metrics_cache).")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="If given, also write radarchart_table.txt under this directory.")
    p.add_argument("--plot", type=Path, default=None, metavar="PATH",
                   help="Render the radar chart as sig_radarplot.pdf under PATH "
                        "(requires matplotlib; install via 'pip install pysigqc-metrics[plot]').")
    p.add_argument("--rebuild-data", action="store_true",
                   help="Force re-preprocessing of TCGA -> HGNC matrix.")
    p.add_argument("--verbose", action="store_true",
                   help="Print per-module pipeline progress.")
    return p.parse_args()


def derive_dataset_name(sample_type: str, cancer_type: str) -> str:
    """Build a compact dataset id like 'BIC_PrimaryTumor' from the CLI filters."""
    cancer_short = "".join(w[0].upper() for w in cancer_type.split() if w)
    st_clean = "".join(w.capitalize() for w in sample_type.split())
    return f"{cancer_short}_{st_clean}" if cancer_short else st_clean or "dataset"


def load_filtered_expression(expr_file: Path, phenotype_file: Path,
                             sample_type: str, cancer_type: str) -> pd.DataFrame:
    """Load TCGA expression matrix, keep only the requested phenotype subset."""
    pheno = pd.read_csv(phenotype_file, sep="\t", index_col=False)
    keep = pheno[(pheno["sample_type"] == sample_type)
                 & (pheno["_primary_disease"] == cancer_type)]
    samples_use = set(keep["sample"].tolist())
    if not samples_use:
        sys.exit(f"[error] no samples match sample_type={sample_type!r} "
                 f"and _primary_disease={cancer_type!r}")
    opener = gzip.open if str(expr_file).endswith(".gz") else open
    with opener(expr_file, "rt") as f:
        expr = pd.read_csv(f, sep="\t", index_col=0)
    cols_keep = [c for c in expr.columns if c in samples_use]
    if not cols_keep:
        sys.exit("[error] no overlap between expression columns and phenotype filter.")
    return expr[cols_keep].astype(float)


def subsample_columns(expr: pd.DataFrame, max_n: int, seed: int = 7) -> pd.DataFrame:
    """Cap ``expr`` to at most ``max_n`` randomly-chosen samples (deterministic)."""
    if max_n <= 0 or expr.shape[1] <= max_n:
        return expr
    rng = np.random.default_rng(seed)
    cols_idx = sorted(rng.choice(expr.shape[1], size=max_n, replace=False))
    return expr.iloc[:, cols_idx]


def load_signatures(gmt_path: Path, requested: list[str]) -> dict[str, list[str]]:
    """Parse the GMT, optionally restrict to ``requested`` entry names."""
    sigs_all = parse_gmt_signatures(gmt_path)
    if not sigs_all:
        sys.exit(f"[error] no signatures parsed from {gmt_path}")
    if not requested:
        return dict(sigs_all)
    missing = [s for s in requested if s not in sigs_all]
    if missing:
        sys.exit(f"[error] signatures not in GMT: {missing}\n"
                 f"        available: {list(sigs_all)}")
    return {s: sigs_all[s] for s in requested}


def preprocess_inputs(args: argparse.Namespace, cache_dir: Path
                      ) -> tuple[pd.DataFrame, dict[str, list[str]], str]:
    """Build (HGNC matrix, gene_sigs_list, dataset_name) and persist to cache_dir."""
    dataset_name = args.dataset_name or derive_dataset_name(
        args.sample_type, args.cancer_type)
    dataset_csv = cache_dir / "dataset.csv"
    ens_cache = cache_dir / "ensembl_hgnc_map.tsv"

    gene_sigs_list = load_signatures(args.gmt, args.signature)

    if dataset_csv.exists() and not args.rebuild_data:
        print(f"[data] reuse cached HGNC matrix {dataset_csv} "
              f"(use --rebuild-data to redo)")
        expr_hgnc = pd.read_csv(dataset_csv, index_col=0)
    else:
        print(f"[data] loading expr={args.expr} phenotype={args.phenotype}")
        expr = load_filtered_expression(args.expr, args.phenotype,
                                        args.sample_type, args.cancer_type)
        print(f"[data] filter sample_type={args.sample_type!r} "
              f"cancer={args.cancer_type!r} -> "
              f"{expr.shape[0]} genes x {expr.shape[1]} samples")
        expr = subsample_columns(expr, args.sample_limit)
        if args.sample_limit > 0:
            print(f"[data] sample-limit={args.sample_limit} -> {expr.shape[1]} samples")
        print(f"[data] mapping {expr.shape[0]} Ensembl IDs -> HGNC (cache={ens_cache})")
        expr_hgnc = convert_ensembl_matrix_to_hgnc(expr, ens_cache)
        expr_hgnc.to_csv(dataset_csv)
        print(f"[data] wrote {dataset_csv}: "
              f"{expr_hgnc.shape[0]} genes x {expr_hgnc.shape[1]} samples")

    overlap = {s: sum(g in expr_hgnc.index for g in gs)
               for s, gs in gene_sigs_list.items()}
    print(f"[data] {len(gene_sigs_list)} signatures, overlap: {overlap}")
    return expr_hgnc, gene_sigs_list, dataset_name


def print_radar(result: dict, names_sigs: list[str], names_datasets: list[str]
                ) -> None:
    """Pretty-print the radar table, per-module timings and area scores."""
    radar = result["radar_result"]
    table = radar["output_table"]
    areas = radar["areas"]
    rownames = radar["radarplot_rownames"]

    print("\n================ MODULE TIMES ================")
    for key in ("var_result", "expr_result", "compact_result",
                "metrics_result", "stan_result"):
        mod = key.replace("_result", "")
        print(f"  {mod:12s} {result[key]['elapsed_seconds']:.4f}s")
    print(f"  {'radar':12s} {radar['elapsed_seconds']:.4f}s")
    print(f"  {'TOTAL':12s} {result['elapsed_seconds']:.4f}s")

    print("\n================ RADAR TABLE ================")
    with pd.option_context("display.max_columns", None,
                           "display.width", 200,
                           "display.float_format", "{:.4f}".format):
        print(table)

    print("\n================ AREAS ================")
    for name, area in zip(rownames, areas):
        print(f"  {name:50s}  {area:.4f}")


def main() -> int:
    args = parse_args()
    cache_dir = args.cache_dir or (HERE / "test_metrics_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"[setup] cache_dir = {cache_dir}")

    expr_hgnc, gene_sigs_list, dataset_name = preprocess_inputs(args, cache_dir)
    expr_dict = {dataset_name: expr_hgnc}
    names_datasets = [dataset_name]
    names_sigs = list(gene_sigs_list.keys())
    print(f"[data] datasets={names_datasets}  sigs={names_sigs}")

    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    result = run_pipeline(gene_sigs_list, names_sigs, expr_dict, names_datasets,
                          out_dir=args.out_dir, verbose=args.verbose)
    print(f"\n[time] pipeline wall: {time.perf_counter() - t0:.4f}s "
          f"(internal: {result['elapsed_seconds']:.4f}s)")

    print_radar(result, names_sigs, names_datasets)

    if args.plot is not None:
        try:
            from pysigqc_metrics.plots import plot_radar
        except ImportError as err:
            sys.exit(f"[error] --plot requires matplotlib: {err}\n"
                     f"        install via: pip install pysigqc-metrics[plot]")
        args.plot.mkdir(parents=True, exist_ok=True)
        pdf_path = plot_radar(result["radar_result"], names_sigs,
                              names_datasets, args.plot)
        print(f"\n[plot] wrote {pdf_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

