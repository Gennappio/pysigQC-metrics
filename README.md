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
outputs (`tests/fixtures/reference_outputs/`).

## License

MIT — see `LICENSE`.
