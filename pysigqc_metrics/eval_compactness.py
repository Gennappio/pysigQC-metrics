"""Evaluate compactness (internal coherence) of gene signatures via autocorrelation.

Port of R_refactored/eval_compactness_loc.R — compute_compactness() only.
Produces 1 radar metric: autocor_median (median of Spearman gene-gene correlation matrix).

Also includes RankProd analysis (rank product) for identifying genes with
consistently high/low autocorrelation across datasets.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from .utils import gene_intersection


def _compute_rank_product(data: np.ndarray, n_permutations: int = 1000, seed: int = 42) -> dict:
    """Compute rank product statistics for multi-dataset gene consistency.

    Implements the RankProd algorithm:
    1. For each gene, compute its rank in each dataset
    2. Compute the geometric mean of ranks (rank product)
    3. Estimate p-values via permutation

    Args:
        data: (n_genes, n_datasets) matrix of values to rank
        n_permutations: number of permutations for p-value estimation
        seed: random seed

    Returns:
        dict with:
            - rp_up: rank products for up-regulation (high values)
            - rp_down: rank products for down-regulation (low values)
            - pval_up: p-values for up-regulation
            - pval_down: p-values for down-regulation
            - pfp_up: percentage of false positives for up
            - pfp_down: percentage of false positives for down
    """
    n_genes, n_datasets = data.shape
    rng = np.random.default_rng(seed)

    # Handle NaN by giving them worst rank
    data_filled = np.where(np.isnan(data), -np.inf, data)

    # Compute ranks for each dataset (1 = lowest, n = highest)
    # For up-regulation: rank high values first (descending)
    ranks_up = np.zeros_like(data)
    ranks_down = np.zeros_like(data)

    for j in range(n_datasets):
        col = data_filled[:, j]
        # Up: high value = low rank (good)
        ranks_up[:, j] = sp_stats.rankdata(-col, method='average')
        # Down: low value = low rank (good)
        ranks_down[:, j] = sp_stats.rankdata(col, method='average')

    # Compute rank products (geometric mean of ranks)
    rp_up = np.exp(np.mean(np.log(ranks_up), axis=1))
    rp_down = np.exp(np.mean(np.log(ranks_down), axis=1))

    # Permutation test for p-values
    null_rp_up = np.zeros((n_permutations, n_genes))
    null_rp_down = np.zeros((n_permutations, n_genes))

    for p in range(n_permutations):
        perm_ranks_up = np.zeros_like(ranks_up)
        perm_ranks_down = np.zeros_like(ranks_down)

        for j in range(n_datasets):
            perm_idx = rng.permutation(n_genes)
            perm_ranks_up[:, j] = ranks_up[perm_idx, j]
            perm_ranks_down[:, j] = ranks_down[perm_idx, j]

        null_rp_up[p] = np.exp(np.mean(np.log(perm_ranks_up), axis=1))
        null_rp_down[p] = np.exp(np.mean(np.log(perm_ranks_down), axis=1))

    # P-values: proportion of null RPs <= observed RP
    pval_up = np.zeros(n_genes)
    pval_down = np.zeros(n_genes)

    for i in range(n_genes):
        pval_up[i] = np.mean(null_rp_up[:, :].ravel() <= rp_up[i])
        pval_down[i] = np.mean(null_rp_down[:, :].ravel() <= rp_down[i])

    # PFP (percentage of false positives) = estimated FDR
    # Sort genes by RP, compute expected false positives at each threshold
    sorted_idx_up = np.argsort(rp_up)
    sorted_idx_down = np.argsort(rp_down)

    pfp_up = np.zeros(n_genes)
    pfp_down = np.zeros(n_genes)

    for rank_pos, gene_idx in enumerate(sorted_idx_up, 1):
        # Expected false positives at this rank
        expected_fp = pval_up[gene_idx] * n_genes
        pfp_up[gene_idx] = expected_fp / rank_pos

    for rank_pos, gene_idx in enumerate(sorted_idx_down, 1):
        expected_fp = pval_down[gene_idx] * n_genes
        pfp_down[gene_idx] = expected_fp / rank_pos

    return {
        "rp_up": rp_up,
        "rp_down": rp_down,
        "pval_up": pval_up,
        "pval_down": pval_down,
        "pfp_up": np.clip(pfp_up, 0, 1),
        "pfp_down": np.clip(pfp_down, 0, 1),
    }


def compute_compactness(
    gene_sigs_list: dict[str, list[str]],
    names_sigs: list[str],
    mRNA_expr_matrix: dict[str, pd.DataFrame],
    names_datasets: list[str],
    compute_rank_product: bool = True,
    n_permutations: int = 100,
) -> dict:
    """Compute compactness metrics for each signature-dataset pair.

    Args:
        gene_sigs_list: dict of signature name -> gene list
        names_sigs: list of signature names
        mRNA_expr_matrix: dict of dataset name -> DataFrame (genes x samples)
        names_datasets: list of dataset names
        compute_rank_product: if True and >1 dataset, compute RankProd analysis
        n_permutations: number of permutations for RankProd p-values

    Returns dict with keys:
        radar_values: nested dict [sig][dataset] -> {"autocor_median": val}
        autocor_matrices: nested dict [sig][dataset] -> gene-gene Spearman correlation matrix
        rank_product_tables: dict [sig] -> DataFrame with RankProd results
        elapsed_seconds: wall-clock time
    """
    _t0 = time.perf_counter()
    radar_values: dict = {}
    autocor_matrices: dict = {}
    rank_product_tables: dict = {}

    # First pass: compute autocorrelation matrices and median autocor per gene
    gene_median_autocor: dict = {}  # [sig][gene][ds] = median autocor

    for sig in names_sigs:
        gene_sig = gene_sigs_list[sig]
        radar_values[sig] = {}
        autocor_matrices[sig] = {}
        gene_median_autocor[sig] = {}

        for ds in names_datasets:
            data_matrix = mRNA_expr_matrix[ds]
            inter = gene_intersection(gene_sig, data_matrix)

            # Get sig gene data, drop genes with any NA (matches R's na.omit)
            sig_df = data_matrix.loc[inter].dropna(axis=0, how="any")
            genes_present = list(sig_df.index)
            sig_data = sig_df.values.astype(float)

            # Spearman correlation between genes (rows)
            n_genes = sig_data.shape[0]
            if n_genes > 1:
                autocors = np.eye(n_genes)
                for gi in range(n_genes):
                    for gj in range(gi + 1, n_genes):
                        rho, _ = sp_stats.spearmanr(sig_data[gi], sig_data[gj])
                        autocors[gi, gj] = rho
                        autocors[gj, gi] = rho
            else:
                autocors = np.array([[1.0]])

            autocor_df = pd.DataFrame(autocors, index=genes_present, columns=genes_present)
            autocor_matrices[sig][ds] = autocor_df

            # Compute median autocorrelation per gene (for RankProd)
            for i, gene in enumerate(genes_present):
                if gene not in gene_median_autocor[sig]:
                    gene_median_autocor[sig][gene] = {}
                gene_median_autocor[sig][gene][ds] = float(np.nanmedian(autocors[i, :]))

            if autocors.shape[0] > 1:
                autocor_median = float(np.nanmedian(autocors))
            else:
                autocor_median = 0.0

            radar_values[sig][ds] = {"autocor_median": autocor_median}

    # Second pass: RankProd analysis (only if >1 dataset)
    if compute_rank_product and len(names_datasets) > 1:
        for sig in names_sigs:
            genes = list(gene_median_autocor[sig].keys())
            if len(genes) < 2:
                continue

            # Build matrix: genes x datasets
            overall_rank_mat = np.full((len(genes), len(names_datasets)), np.nan)
            for i, gene in enumerate(genes):
                for j, ds in enumerate(names_datasets):
                    if ds in gene_median_autocor[sig][gene]:
                        overall_rank_mat[i, j] = gene_median_autocor[sig][gene][ds]

            # Run RankProd
            rp_result = _compute_rank_product(overall_rank_mat, n_permutations=n_permutations)

            # Build output table (matching R's format)
            table = pd.DataFrame({
                "pfp_up": rp_result["pfp_up"],
                "pfp_down": rp_result["pfp_down"],
                "pval_up": rp_result["pval_up"],
                "pval_down": rp_result["pval_down"],
                "rp_up": rp_result["rp_up"],
                "rp_down": rp_result["rp_down"],
            }, index=genes)

            # Sort by rp_up (genes with consistently high autocorrelation first)
            table = table.sort_values("rp_up")
            rank_product_tables[sig] = table

    return {
        "radar_values": radar_values,
        "autocor_matrices": autocor_matrices,
        "rank_product_tables": rank_product_tables,
        "elapsed_seconds": time.perf_counter() - _t0,
    }
