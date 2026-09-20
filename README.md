# pysigqc-metrics

Isolated radar-metrics pipeline extracted from
[pysigQC](https://github.com/Gennappio/pysigQC) — the Python port of sigQC
(Dhawan et al., *Nature Protocols*, 2019).

This package contains exactly the modules needed to compute the 14 radar
metrics that summarize gene signature quality on a (signature × dataset)
grid, plus the radar chart assembly and rendering.

It does **not** include negative/permutation controls, hierarchical
clustering / biclustering (`eval_struct`), or the parallel `joblib` variant
of the pipeline. For those, use the full [pysigQC](https://github.com/Gennappio/pysigQC).

## Modules included

| Module | Radar metrics produced |
|---|---|
| `eval_var` | `sd_median_ratio`, `abs_skewness_ratio`, `prop_top_10/25/50_percent`, `coeff_of_var_ratio` |
| `eval_expr` | `med_prop_na`, `med_prop_above_med` |
| `eval_compactness` | `autocor_median` |
| `compare_metrics` | `rho_mean_med`, `rho_pca1_med`, `rho_mean_pca1`, `prop_pca1_var` |
| `eval_stan` | `standardization_comp` |
| `radar_chart` | assembles the 14 metrics into the radar matrix + area ratios |
| `plots.plot_radar` | renders the radar chart as PDF (matplotlib) |

## Installation

From source:

```bash
pip install "git+ssh://git@github.com/Gennappio/pysigQC-metrics.git@main"
```

For the optional plotting layer (matplotlib):

```bash
pip install "pysigqc-metrics[plot] @ git+ssh://git@github.com/Gennappio/pysigQC-metrics.git@main"
```

Or pin to a tag for stability:

```bash
pip install "git+ssh://git@github.com/Gennappio/pysigQC-metrics.git@v0.1.0"
```

## Usage

### Full pipeline

```python
import pandas as pd
from pysigqc_metrics import run_pipeline

# expression: dict of dataset name -> DataFrame (genes x samples)
expr = {"dataset_A": pd.read_csv("ds_a.csv", index_col=0),
        "dataset_B": pd.read_csv("ds_b.csv", index_col=0)}
sigs = {"my_signature": ["GENE1", "GENE2", "GENE3"]}

result = run_pipeline(
    gene_sigs_list=sigs,
    names_sigs=list(sigs),
    mRNA_expr_matrix=expr,
    names_datasets=list(expr),
    out_dir="out/",        # optional: writes radarchart_table.txt
)

print(result["radar_result"]["output_table"])  # 14-column table
print(result["radar_result"]["areas"])         # area ratio per row
```

### Render the radar plot

```python
from pysigqc_metrics.plots import plot_radar

plot_radar(
    result["radar_result"],
    names_sigs=list(sigs),
    names_datasets=list(expr),
    out_dir="out/",
)  # writes out/sig_radarplot.pdf
```

### Calling individual modules

```python
from pysigqc_metrics import (
    compute_var, compute_expr, compute_compactness,
    compute_metrics, compute_stan, compute_radar,
)

var_r = compute_var(sigs, list(sigs), expr, list(expr))
# ... merge radar_values from each module, then:
radar = compute_radar(radar_values, list(sigs), list(expr))
```

## Input requirements

- Gene signatures: dict of `{name: list[str]}`. Gene IDs must match the row
  index of the expression matrices.
- Expression matrices: dict of `{name: pandas.DataFrame}` with shape
  *(genes × samples)*. Should be normalized, batch-corrected and
  log-transformed before use.
- Minimum 2 genes per signature, 2 samples per dataset.

## Single-cell / sparse inputs

Besides DataFrames, every dataset in `mRNA_expr_matrix` may be an `AnnData`
(in memory, `backed="r"`, or lazy) or an `ExpressionBackend`. Sparse inputs
are never densified as a whole — only the genes of the signatures are — and
the 14 metrics are numerically identical to the dense path (see
`tests/test_backends.py`).

```bash
pip install "pysigqc-metrics[anndata] @ git+ssh://git@github.com/Gennappio/pysigQC-metrics.git@main"
```

```python
import anndata as ad
from pysigqc_metrics import run_pipeline, SparseBackend, AnnDataBackend

# AnnData (cells x genes), out-of-core: only the needed parts are read
adata = ad.read_h5ad("atlas.h5ad", backed="r")
result = run_pipeline(sigs, list(sigs), {"atlas": adata}, ["atlas"])

# a specific layer
backend = AnnDataBackend(adata, layer="lognorm")

# a bare scipy matrix: say which axis holds the genes
backend = SparseBackend(csr, gene_names, cell_names, gene_axis=1)   # cells x genes
```

`run_pipeline` shares per-dataset work between the five modules by default
(`share_cache=True`): per-gene statistics, the expression threshold and the
signature-gene fetch are computed once — two scans of the matrix instead of
one or two per module. Results are unchanged.

Per-gene statistics depend on the matrix only, not on the signatures. With
`stats_cache="some/dir"` they are saved as `<dataset>.gene_stats.npz` (< 1 MB)
and reused by later runs on the same matrix, which then skip the scans of the
whole matrix (1M cells: ~30 s -> ~16 s):

```python
result = run_pipeline(sigs, list(sigs), {"atlas": adata}, ["atlas"],
                      stats_cache="~/.cache/pysigqc")
```

The file is validated against a fingerprint of the matrix (shape, dtype, gene
names, digest of its first and last vectors) and ignored with a warning when
it does not match. The fingerprint does not read the whole matrix: delete the
file if you modify a matrix in place while keeping its shape.

Memory: out-of-core inputs need about 4 GB per million cells for signatures
of up to 100 genes (peak RSS reads higher on macOS, whose allocator keeps
freed blocks resident); a DataFrame needs about 9 bytes per matrix element.

Storage advice, measured in [`SCALABILITY_REPORT.md`](SCALABILITY_REPORT.md):
CSC (gene-major) or CSR both work; avoid gzip-compressed H5AD for repeated
analyses (2–3× slower), prefer uncompressed H5AD or Zarr with zstd/Blosc.

## CLI — running on TCGA inputs

`scripts/test_metrics.py` is a standalone CLI that runs the full pipeline on a
TCGA-style dataset (Ensembl expression matrix + phenotype TSV + MSigDB GMT) and
prints the radar table. No R dependency.

```bash
python scripts/test_metrics.py \
    --expr      tcga_RSEM_gene_tpm.gz \
    --phenotype TCGA_phenotype_denseDataOnlyDownload.tsv \
    --gmt       BUFFA_HYPOXIA_METAGENE.v2026.1.Hs.gmt \
    --sample-limit 100 \
    --plot      ./radar_out          # optional: writes sig_radarplot.pdf
```

By default every run re-processes the input files from scratch. Pass
`--cache-dir PATH` to persist the preprocessed HGNC matrix between runs and
skip the slow Ensembl→HGNC mapping step on subsequent calls.

```
python scripts/test_metrics.py --help
```

## Tests

```bash
pip install -e ".[dev]"
pytest tests/
```

The test suite includes cross-validation against the original R reference
outputs (`tests/fixtures/reference_outputs/`) and backend-parity tests that
compare every sparse / on-disk backend with the dense pipeline.

Scalability benchmarks live in [`benchmarks/`](benchmarks/README.md).

## License

MIT — see `LICENSE`.
