"""Render the benchmark figures from ``benchmarks/results.csv`` into ``benchmarks/plots/``.

    python benchmarks/make_plots.py

Figures (every one is backed by a table in SCALABILITY_REPORT.md):
    scaling_time.png       total wall time vs number of cells, per backend
    scaling_memory.png     peak RSS vs number of cells, per backend
    modules_1M.png         time per pysigQC module at 1M cells
    operations_1M.png      time per kind of operation at 1M cells (where the time goes)
    layout_variants.png    Zarr chunking / compression / dask variants
    interventions.png      pipeline time before/after the shared cache and the exact fast paths
    density.png            effect of matrix density
    round2_time.png        second optimisation round: pipeline time
    round2_memory.png      second optimisation round: peak memory
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PLOTS = HERE / "plots"

# Validated categorical palette (light surface), fixed slot order.
SURFACE = "#fcfcfb"
TEXT, TEXT_2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

FAMILY_COLOR = {"dense": SLOTS[0], "scipy": SLOTS[1], "h5ad": SLOTS[2], "zarr": SLOTS[3]}
FAMILY_LABEL = {"dense": "Current dense (pandas)", "scipy": "SciPy sparse (in memory)",
                "h5ad": "H5AD backed", "zarr": "Zarr lazy"}
MAIN = ["dense", "scipy_csr", "scipy_csc", "h5ad_csr", "h5ad_csc", "zarr_csr", "zarr_csc"]
MODULES = ["eval_var", "eval_expr", "eval_compactness", "compare_metrics", "eval_stan"]

# backend_timings key -> operation category
OPERATIONS = {
    "Storage read (I/O + decompression)": ["backend_scan_read_seconds"],
    "Sparse traversal (per-gene stats)": ["backend_scan_compute_seconds", "backend_median_partition_seconds"],
    # module-level extract timers already include the backend's load_genes
    "Gene extraction (signature genes)": ["backend_compactness_extract_seconds",
                                          "backend_metrics_extract_seconds", "backend_stan_extract_seconds"],
    "Ranking (Spearman)": ["backend_compactness_rank_seconds", "backend_metrics_spearman_seconds",
                           "backend_stan_spearman_seconds"],
    "Per-cell median/mean scores": ["backend_metrics_scores_seconds", "backend_stan_scores_seconds"],
    "PCA": ["backend_metrics_pca_seconds"],
    "Gene-gene correlation": ["backend_compactness_corr_seconds"],
}


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.size": 10,
        "text.color": TEXT, "axes.labelcolor": TEXT_2, "axes.titlecolor": TEXT,
        "xtick.color": TEXT_2, "ytick.color": TEXT_2,
        "axes.edgecolor": GRID, "axes.linewidth": 1.0,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1.0, "grid.linestyle": "-",
        "axes.axisbelow": True, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "legend.frameon": False, "lines.linewidth": 2.0,
        "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    })


def save(fig, name):
    PLOTS.mkdir(exist_ok=True)
    fig.savefig(PLOTS / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] {name}")


def family(backend: str) -> str:
    return backend.split("_")[0]


def fmt_cells(n: float) -> str:
    return f"{n / 1e6:g}M" if n >= 1e6 else f"{n / 1e3:g}k"


def main_runs(df: pd.DataFrame) -> pd.DataFrame:
    d = df[(df["tag"].fillna("") == "") & df["dataset"].str.match(r"sc_\d+x20000_d05$")]
    return d[d["backend"].isin(MAIN)]


def scaling(df: pd.DataFrame, column: str, scale: float, ylabel: str, title: str, name: str):
    d = main_runs(df)
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for backend in MAIN:
        g = d[(d["backend"] == backend) & (d["status"] == "OK")].sort_values("cells")
        if g.empty:
            continue
        csc = backend.endswith("csc")
        ax.plot(g["cells"], g[column] * scale, color=FAMILY_COLOR[family(backend)],
                linestyle=(0, (4, 2)) if csc else "-", marker="o", markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
    oom = d[(d["backend"] == "dense") & (d["status"] == "OOM_EXPECTED")].sort_values("cells")
    dense_ok = d[(d["backend"] == "dense") & (d["status"] == "OK")].sort_values("cells")
    if not oom.empty and not dense_ok.empty:
        last = dense_ok.iloc[-1]
        ax.annotate(f"dense: OOM expected\nfrom {fmt_cells(oom['cells'].iloc[0])} cells",
                    (last["cells"], last[column] * scale), xytext=(8, 10),
                    textcoords="offset points", color=TEXT_2, fontsize=9, va="bottom")
    ax.set_xscale("log")
    ax.set_yscale("log")
    cells = sorted(d["cells"].unique())
    ax.set_xticks(cells, [fmt_cells(c) for c in cells])
    ax.minorticks_off()
    ax.set_xlabel("cells (20,000 genes, 5% density)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    handles = [plt.Line2D([], [], color=FAMILY_COLOR[f], lw=2, label=FAMILY_LABEL[f])
               for f in FAMILY_COLOR]
    handles += [plt.Line2D([], [], color=TEXT_2, lw=2, label="CSR (cell-major)"),
                plt.Line2D([], [], color=TEXT_2, lw=2, linestyle=(0, (4, 2)), label="CSC (gene-major)")]
    ax.legend(handles=handles, loc="upper left", fontsize=9)
    save(fig, name)


def stacked_bars(table: pd.DataFrame, title: str, xlabel: str, name: str):
    """Horizontal stacked bars, one row per backend, 2px surface gap between segments."""
    fig, ax = plt.subplots(figsize=(8.2, 0.55 * len(table) + 1.9))
    left = np.zeros(len(table))
    y = np.arange(len(table))
    for i, col in enumerate(table.columns):
        vals = table[col].to_numpy(dtype=float)
        ax.barh(y, vals, left=left, height=0.5, color=SLOTS[i], edgecolor=SURFACE,
                linewidth=2, label=col)
        left += vals
    for yi, total in zip(y, left):
        ax.text(total, yi, f"  {total:.0f} s", va="center", ha="left", color=TEXT, fontsize=9)
    ax.set_yticks(y, table.index)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, left.max() * 1.12)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=9)
    save(fig, name)


def modules_and_operations(df: pd.DataFrame, n_cells: int = 1_000_000):
    d = main_runs(df)
    d = d[(d["cells"] == n_cells) & (d["status"] == "OK")].set_index("backend")
    d = d.loc[[b for b in MAIN if b in d.index]]
    if d.empty:
        return
    mods = d[[m + "_seconds" for m in MODULES]].copy()
    mods.columns = MODULES
    stacked_bars(mods, f"Time per pysigQC module — {fmt_cells(n_cells)} cells × 20k genes",
                 "wall time (s)", f"modules_{fmt_cells(n_cells)}.png")
    ops = pd.DataFrame({label: d[[c for c in cols if c in d.columns]].sum(axis=1)
                        for label, cols in OPERATIONS.items()})
    stacked_bars(ops, f"Where the time goes — {fmt_cells(n_cells)} cells × 20k genes",
                 "wall time (s)", f"operations_{fmt_cells(n_cells)}.png")


def simple_bars(labels, values, title, xlabel, name, notes=None):
    fig, ax = plt.subplots(figsize=(7.6, 0.42 * len(labels) + 1.4))
    y = np.arange(len(labels))
    ax.barh(y, values, height=0.5, color=SLOTS[0])
    for yi, v in zip(y, values):
        ax.text(v, yi, f"  {v:.1f} s", va="center", ha="left", color=TEXT, fontsize=9)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(values) * 1.15)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    save(fig, name)


def layout_variants(df: pd.DataFrame, n_cells: int = 1_000_000):
    d = df[(df["dataset"] == f"sc_{n_cells}x20000_d05") & (df["status"] == "OK")
           & (df["tag"].fillna("") == "")]
    order = ["h5ad_csr", "h5ad_csr_gzip", "h5ad_csc", "h5ad_csc_gzip",
             "zarr_csr", "zarr_csr_raw", "zarr_csr_c1k", "zarr_csr_c10k", "zarr_csr_c50k", "zarr_csr+dask",
             "zarr_csc", "zarr_csc_raw", "zarr_csc_g10", "zarr_csc_g100", "zarr_csc_g1000", "zarr_csc+dask"]
    d = d.set_index("backend")
    order = [b for b in order if b in d.index]
    if len(order) < 3:
        return
    simple_bars(order, d.loc[order, "total_seconds"].to_numpy(),
                f"Storage layout, chunking and compression — {fmt_cells(n_cells)} cells",
                "total pipeline wall time (s)", "layout_variants.png")


def interventions(df: pd.DataFrame, n_cells: int = 1_000_000):
    d = df[(df["dataset"] == f"sc_{n_cells}x20000_d05") & (df["status"] == "OK")]
    tags = [("pipeline", "Sparse backends (per-module work)"),
            ("shared_cache", "+ shared DatasetStatsCache"),
            ("fastpaths", "+ exact fast median & sparse-aware ranking"),
            ("fastpaths_cache", "+ both")]
    tags = [(t, l) for t, l in tags if (d["tag"] == t).any()]
    backends = [b for b in MAIN[1:] if ((d["backend"] == b) & (d["tag"] == tags[0][0])).any()] if tags else []
    if not backends:
        return
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    width = 0.8 / len(tags)
    x = np.arange(len(backends))
    for i, (tag, label) in enumerate(tags):
        vals = [d[(d["backend"] == b) & (d["tag"] == tag)]["total_seconds"].mean() for b in backends]
        ax.bar(x + (i - (len(tags) - 1) / 2) * width, vals, width=width * 0.92, color=SLOTS[i],
               edgecolor=SURFACE, linewidth=2, label=label)
        for xi, v in zip(x, vals):
            if np.isfinite(v):
                ax.text(xi + (i - (len(tags) - 1) / 2) * width, v, f"{v:.0f}", ha="center",
                        va="bottom", fontsize=8, color=TEXT)
    ax.set_xticks(x, backends)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("total pipeline wall time (s)")
    ax.set_title(f"Effect of each intervention — {fmt_cells(n_cells)} cells × 20k genes")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=9)
    save(fig, "interventions.png")


def grouped_bars(df, n_cells, column, scale, tags, ylabel, title, name, fmt):
    d = df[(df["dataset"] == f"sc_{n_cells}x20000_d05") & (df["status"] == "OK")]
    tags = [(t, l) for t, l in tags if (d["tag"] == t).any()]
    backends = [b for b in MAIN[1:] if all(((d["backend"] == b) & (d["tag"] == t)).any() for t, _ in tags)]
    if not tags or not backends:
        return
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    width = 0.8 / len(tags)
    x = np.arange(len(backends))
    for i, (tag, label) in enumerate(tags):
        vals = [d[(d["backend"] == b) & (d["tag"] == tag)][column].mean() * scale for b in backends]
        offs = x + (i - (len(tags) - 1) / 2) * width
        ax.bar(offs, vals, width=width * 0.92, color=SLOTS[i], edgecolor=SURFACE, linewidth=2, label=label)
        for xi, v in zip(offs, vals):
            ax.text(xi, v, fmt.format(v), ha="center", va="bottom", fontsize=8, color=TEXT)
    ax.set_xticks(x, backends)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=1, fontsize=9)
    save(fig, name)


def round2(df: pd.DataFrame, n_cells: int = 1_000_000):
    size = f"{fmt_cells(n_cells)} cells × 20k genes"
    grouped_bars(df, n_cells, "total_seconds", 1.0,
                 [("fastpaths_cache", "Before (shared cache + exact fast paths)"),
                  ("v2", "After the memory work (run_pipeline defaults)"),
                  ("v2_stats_reuse", "After, with persisted per-gene statistics (stats_cache=)")],
                 "total pipeline wall time (s)", f"Second round, time — {size}", "round2_time.png", "{:.0f}")
    grouped_bars(df, n_cells, "peak_rss", 1e-9,
                 [("fastpaths_cache", "Before — peak RSS"),
                  ("v2", "After — peak RSS (default macOS allocator)"),
                  ("v2_ws", "After — real working set (allocator cache off)")],
                 "peak memory (GB)", f"Second round, memory — {size}", "round2_memory.png", "{:.1f}")


def density(df: pd.DataFrame, n_cells: int = 1_000_000):
    d = df[df["dataset"].str.match(rf"sc_{n_cells}x20000_d\d+$") & (df["status"] == "OK")
           & (df["tag"].fillna("") == "") & df["backend"].isin(MAIN)]
    if d["dataset"].nunique() < 2:
        return
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for backend in MAIN[1:]:
        g = d[d["backend"] == backend].sort_values("density")
        ax.plot(g["density"] * 100, g["total_seconds"], color=FAMILY_COLOR[family(backend)],
                linestyle=(0, (4, 2)) if backend.endswith("csc") else "-", marker="o",
                markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax.set_xlabel("matrix density (% non-zero)")
    ax.set_ylabel("total pipeline wall time (s)")
    ax.set_ylim(bottom=0)
    ax.set_title(f"Effect of density — {fmt_cells(n_cells)} cells × 20k genes")
    handles = [plt.Line2D([], [], color=FAMILY_COLOR[f], lw=2, label=FAMILY_LABEL[f])
               for f in ("scipy", "h5ad", "zarr")]
    handles += [plt.Line2D([], [], color=TEXT_2, lw=2, label="CSR"),
                plt.Line2D([], [], color=TEXT_2, lw=2, linestyle=(0, (4, 2)), label="CSC")]
    ax.legend(handles=handles, loc="upper left", fontsize=9)
    save(fig, "density.png")


def main():
    style()
    df = pd.read_csv(HERE / "results.csv")
    df["tag"] = df.get("tag", pd.Series("", index=df.index)).fillna("")
    scaling(df, "total_seconds", 1.0, "total pipeline wall time (s)",
            "Pipeline wall time vs dataset size", "scaling_time.png")
    scaling(df, "peak_rss", 1e-9, "peak RSS (GB)",
            "Peak memory vs dataset size", "scaling_memory.png")
    modules_and_operations(df)
    layout_variants(df)
    interventions(df)
    round2(df)
    density(df)


if __name__ == "__main__":
    main()
