"""Expression-matrix backends: one logical view, several storage layouts.

Every pysigqc module sees an expression matrix as **genes x samples**. The
backends below provide that view without ever densifying (or densely
transposing) the whole matrix:

    DatasetStatsCache  optional memoising wrapper shared by the five modules.
    DenseBackend     pandas DataFrame (genes x samples). Runs exactly the
                     historical NumPy code, so results are bit-identical to
                     the pre-backend implementation.
    SparseBackend    scipy.sparse CSR/CSC held in memory, *or* an on-disk
                     sparse matrix (AnnData backed H5AD / Zarr, dask-lazy)
                     read chunk by chunk.
    AnnDataBackend   factory that picks the right one for an AnnData object.

Storage convention for sparse data is AnnData's: ``cells x genes``. A
``genes x samples`` scipy matrix is accepted too and is re-viewed as
``cells x genes`` through ``.T`` (CSR <-> CSC, zero copy).

What "gene-major" / "cell-major" means
--------------------------------------
    cells x genes CSC  -> gene-major: one gene = one contiguous slice.
    cells x genes CSR  -> cell-major: one gene is scattered over all rows;
                          any gene selection is a full scan of the matrix.
Both are supported by the same algorithms; only the cost differs.

NaN / Inf
---------
Sparse matrices may store explicit NaN/Inf entries. They are handled in the
sparse path itself with the semantics of the dense code (``nanmean``,
``nanstd(ddof=1)``, ``x < threshold`` being False for NaN), so there is no
separate "compatibility" path. Implicit zeros always count as real zeros.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import scipy.sparse as sp

# Target number of stored entries handled per scan chunk. Bounds the scan's
# temporaries to a few hundred MB independently of the dataset size.
DEFAULT_CHUNK_NNZ = 1 << 24


@dataclass
class ExprStats:
    """Per-gene quantities needed by eval_expr."""
    nan_counts: np.ndarray      # number of NaN samples per gene
    below_counts: np.ndarray    # number of samples with value < threshold per gene
    threshold: float


class GeneStore:
    """A small set of genes pulled out of a backend, ready to be densified.

    ``dense(idx)`` returns a writable float64 ``len(idx) x n_samples`` array —
    the only place where expression values are densified, and only for the
    requested genes. It is C-contiguous unless ``order="K"`` is requested,
    which keeps whatever layout the DataFrame code historically produced
    (row-wise sums round differently on C and F layouts; eval_stan relies on
    it to stay bit-identical for DataFrame inputs).
    """

    def dense(self, idx: np.ndarray, order: str = "C") -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


class ExpressionBackend(ABC):
    """Logical genes x samples expression matrix."""

    gene_names: pd.Index
    sample_names: list
    timings: dict

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.gene_names), len(self.sample_names))

    @property
    def index(self) -> pd.Index:
        """Alias of ``gene_names`` (DataFrame-like access used by utils)."""
        return self.gene_names

    def _tick(self, key: str, t0: float) -> None:
        """Accumulate wall time under ``timings[key]`` (profiling aid; kept on
        the backend so that module return values stay unchanged)."""
        self.timings[key] = self.timings.get(key, 0.0) + time.perf_counter() - t0

    def gene_indices(self, genes) -> np.ndarray:
        """Row positions of the genes present in the dataset, in signature order."""
        idx = self.gene_names.get_indexer(list(genes))
        return idx[idx >= 0]

    def get_genes(self, idx) -> np.ndarray:
        """Dense float64 block (len(idx) x n_samples) for the given gene positions."""
        idx = np.asarray(idx, dtype=np.intp)
        return self.load_genes(idx).dense(idx)

    def iter_gene_blocks(self, block_size: int = 256) -> Iterator[tuple[int, int, np.ndarray]]:
        """Yield ``(start, stop, dense_block)`` covering all genes in order."""
        n_genes = self.shape[0]
        for start in range(0, n_genes, block_size):
            stop = min(start + block_size, n_genes)
            yield start, stop, self.get_genes(np.arange(start, stop))

    def fingerprint(self) -> dict:
        """Cheap identity of the matrix, used to validate persisted statistics:
        shape, dtype, gene names and a digest of its first and last vectors.
        It does not read the whole matrix, so a change confined to the middle
        of a matrix of unchanged shape and size goes unnoticed."""
        raise NotImplementedError

    def _base_fingerprint(self) -> dict:
        names = "\x1f".join(map(str, self.gene_names)).encode()
        return {"shape": [int(v) for v in self.shape],
                "genes": hashlib.sha1(names).hexdigest()}

    @abstractmethod
    def load_genes(self, idx) -> GeneStore:
        """Fetch a set of genes (typically the union of all signatures) once."""

    @abstractmethod
    def gene_mean_sd(self) -> tuple[np.ndarray, np.ndarray]:
        """Per-gene nanmean and nanstd(ddof=1) over samples, for all genes."""

    @abstractmethod
    def expr_stats(self, threshold: float | None) -> ExprStats:
        """Per-gene NaN / below-threshold counts.

        With ``threshold=None`` the threshold is the exact median of all
        values of the genes that carry no NaN (R: ``median(na.omit(m))``).
        """


# ---------------------------------------------------------------------------
# Dense (pandas) backend — the historical code path
# ---------------------------------------------------------------------------

class _DenseStore(GeneStore):
    def __init__(self, df: pd.DataFrame):
        self._df = df

    def dense(self, idx: np.ndarray, order: str = "C") -> np.ndarray:
        # copy=True: callers own (and may overwrite) the block; pandas may
        # otherwise hand out a read-only view.
        block = self._df.iloc[np.asarray(idx, dtype=np.intp)].to_numpy(dtype=float, copy=True)
        return block if order == "K" else np.ascontiguousarray(block)


class DenseBackend(ExpressionBackend):
    """pandas DataFrame, genes x samples.

    Whole-matrix statistics run the historical NumPy calls (``nanmean``,
    ``nanstd(ddof=1)``, ``isnan``, ``<``, exact median) on **blocks of gene
    rows** instead of on one float64 copy of the whole frame. Every one of
    those reductions is per row, so the values are bit-identical to the
    unblocked computation — provided a block never has exactly one row (NumPy
    then switches to pairwise summation), which ``_row_blocks`` guarantees.
    Peak memory drops from ~29 to ~8 bytes per element for a float32 frame.
    Gene selection densifies the selected rows only.
    """

    def __init__(self, df: pd.DataFrame, block_elements: int = 1 << 25):
        self._df = df
        self.gene_names = df.index
        self.sample_names = list(df.columns)
        self.block_elements = int(block_elements)
        self.timings = {}

    def load_genes(self, idx) -> GeneStore:
        return _DenseStore(self._df)

    def fingerprint(self) -> dict:
        arr = self._native()
        k = min(32, arr.shape[0])
        fp = self._base_fingerprint()
        fp.update(kind="dense", dtype=str(arr.dtype),
                  head=hashlib.sha1(np.ascontiguousarray(arr[:k]).tobytes()).hexdigest(),
                  tail=hashlib.sha1(np.ascontiguousarray(arr[-k:]).tobytes()).hexdigest())
        return fp

    # -- blocked access -------------------------------------------------------

    def _native(self) -> np.ndarray:
        """The frame as a 2-D array in its own dtype (a view for homogeneous
        numeric frames — no float64 copy of the whole matrix)."""
        arr = self._df.to_numpy()
        if arr.dtype.kind not in "fiu":          # object / bool / ...: historical conversion
            arr = self._df.to_numpy(dtype=float)
        return arr

    def _row_blocks(self, arr: np.ndarray) -> Iterator[tuple[int, int]]:
        n_genes, n_samples = arr.shape
        step = max(2, self.block_elements // max(1, n_samples))
        start = 0
        while start < n_genes:
            stop = min(start + step, n_genes)
            if n_genes - stop == 1:              # never leave a 1-row block behind
                stop = n_genes
            yield start, stop
            start = stop

    # -- whole-dataset statistics --------------------------------------------

    def gene_mean_sd(self):
        mean, sd, _, _ = self._stats(want_mean_sd=True, want_expr=False, threshold=None)
        return mean, sd

    def expr_stats(self, threshold):
        _, _, stats, _ = self._stats(want_mean_sd=False, want_expr=True, threshold=threshold)
        return stats

    def fused_stats(self, threshold: float | None, gene_idx=None):
        """mean/SD and expression counts from the same pass over the blocks."""
        mean, sd, stats, _ = self._stats(want_mean_sd=True, want_expr=True, threshold=threshold)
        return mean, sd, stats, (None if gene_idx is None else self.load_genes(gene_idx))

    def _stats(self, want_mean_sd: bool, want_expr: bool, threshold):
        arr = self._native()
        n_genes, n_samples = arr.shape
        mean = np.empty(n_genes) if want_mean_sd else None
        sd = np.empty(n_genes) if want_mean_sd else None
        nan_counts = np.zeros(n_genes, dtype=np.int64)
        below = np.zeros(n_genes, dtype=np.int64)
        need_median = want_expr and threshold is None
        # R: median(unlist(na.omit(matrix))) — na.omit drops rows with any NA.
        # Values of NaN-free rows are gathered in the frame's own dtype (order
        # statistics do not depend on the dtype) and partitioned in place.
        pool = np.empty(arr.size, dtype=arr.dtype) if need_median else None
        filled = 0

        t0 = time.perf_counter()
        for a, b in self._row_blocks(arr):
            blk = np.asarray(arr[a:b], dtype=float)
            if want_mean_sd:
                sd[a:b] = np.nanstd(blk, axis=1, ddof=1)      # ddof=1 to match R's sd()
                mean[a:b] = np.nanmean(blk, axis=1)
            if want_expr:
                nan_mask = np.isnan(blk)
                nan_counts[a:b] = nan_mask.sum(axis=1)
                if need_median:
                    clean = arr[a:b][~nan_mask.any(axis=1)]
                    pool[filled:filled + clean.size] = clean.ravel()
                    filled += clean.size
                else:
                    # NaN comparison is False in numpy.
                    below[a:b] = (blk < float(threshold)).sum(axis=1)
        self._tick("dense_blocks", t0)

        stats = None
        if want_expr:
            if need_median:
                t0 = time.perf_counter()
                threshold = self._median_in_place(pool[:filled])
                del pool
                self._tick("median_partition", t0)
                for a, b in self._row_blocks(arr):
                    below[a:b] = (np.asarray(arr[a:b], dtype=float) < threshold).sum(axis=1)
            stats = ExprStats(nan_counts, below, float(threshold))
        return mean, sd, stats, None

    @staticmethod
    def _median_in_place(values: np.ndarray) -> float:
        """np.median of NaN-free values: the two middle order statistics,
        averaged in float64 exactly as np.median does after conversion."""
        n = values.size
        if n == 0:
            return float("nan")
        lo, hi = (n - 1) // 2, n // 2
        values.partition((lo, hi) if lo != hi else (hi,))
        return float(np.mean(np.array([values[lo], values[hi]], dtype=np.float64)))


PandasBackend = DenseBackend


# ---------------------------------------------------------------------------
# Sparse readers: uniform chunked access along the major axis
# ---------------------------------------------------------------------------

class _MajorReader(ABC):
    """Reads contiguous runs of major-axis vectors of a cells x genes matrix."""

    fmt: str                     # "csr" (cell-major) or "csc" (gene-major)
    shape: tuple[int, int]       # (n_cells, n_genes)
    nnz: int

    @property
    def n_major(self) -> int:
        return self.shape[0] if self.fmt == "csr" else self.shape[1]

    @abstractmethod
    def read(self, start: int, stop: int):
        """scipy matrix (same format) for major vectors [start, stop)."""

    def read_major_list(self, idx: np.ndarray):
        """scipy matrix of the given (sorted, unique) major vectors."""
        parts = [self.read(int(i), int(i) + 1) for i in idx]
        return sp.vstack(parts, format="csr") if self.fmt == "csr" else sp.hstack(parts, format="csc")

    def bounds(self, chunk_nnz: int) -> list[tuple[int, int]]:
        per_vec = max(1.0, self.nnz / max(1, self.n_major))
        step = int(max(1, min(self.n_major, chunk_nnz // per_vec)))
        return [(s, min(s + step, self.n_major)) for s in range(0, self.n_major, step)]


class _ScipyReader(_MajorReader):
    def __init__(self, matrix):
        if not (sp.isspmatrix_csr(matrix) or sp.isspmatrix_csc(matrix)
                or getattr(matrix, "format", None) in ("csr", "csc")):
            matrix = sp.csr_matrix(matrix)
        self._m = matrix
        self.fmt = matrix.format
        self.shape = matrix.shape
        self.nnz = int(matrix.nnz)

    def read(self, start, stop):
        m = self._m
        a, b = int(m.indptr[start]), int(m.indptr[stop])
        indptr = m.indptr[start:stop + 1] - m.indptr[start]
        n = stop - start
        shape = (n, m.shape[1]) if self.fmt == "csr" else (m.shape[0], n)
        cls = sp.csr_matrix if self.fmt == "csr" else sp.csc_matrix
        return cls((m.data[a:b], m.indices[a:b], indptr), shape=shape, copy=False)

    def read_major_list(self, idx):
        return self._m[idx] if self.fmt == "csr" else self._m[:, idx]


class _BackedReader(_MajorReader):
    """anndata on-disk sparse dataset (H5AD backed or Zarr via ``sparse_dataset``)."""

    def __init__(self, dataset):
        self._d = dataset
        self.fmt = dataset.format
        self.shape = tuple(dataset.shape)
        self.nnz = int(dataset.group["data"].shape[0])

    def read(self, start, stop):
        return self._d[start:stop] if self.fmt == "csr" else self._d[:, start:stop]

    def read_major_list(self, idx):
        return self._d[idx] if self.fmt == "csr" else self._d[:, idx]


class _DaskReader(_MajorReader):
    """dask array of scipy-sparse chunks (``anndata.experimental.read_lazy``)."""

    def __init__(self, darr):
        self._d = darr
        self.shape = tuple(int(s) for s in darr.shape)
        meta = getattr(darr, "_meta", None)
        self.fmt = getattr(meta, "format", None) or "csr"
        if self.fmt not in ("csr", "csc"):
            raise TypeError(f"Unsupported lazy sparse format: {self.fmt!r}")
        self._axis = 0 if self.fmt == "csr" else 1
        edges = np.concatenate([[0], np.cumsum(darr.chunks[self._axis])])
        self._bounds = [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:])]
        self.nnz = -1  # unknown without reading; bounds come from dask chunks

    def bounds(self, chunk_nnz):
        return self._bounds

    def read(self, start, stop):
        sl = self._d[start:stop] if self.fmt == "csr" else self._d[:, start:stop]
        out = sl.compute()
        return out.tocsr() if self.fmt == "csr" else out.tocsc()

    def read_major_list(self, idx):
        # One compute per dask block that holds a wanted vector, not one per vector.
        idx = np.asarray(idx)
        parts = []
        for start, stop in self._bounds:
            local = idx[(idx >= start) & (idx < stop)] - start
            if local.size:
                block = self.read(start, stop)
                parts.append(block[local] if self.fmt == "csr" else block[:, local])
        if self.fmt == "csr":
            return sp.vstack(parts, format="csr")
        return sp.hstack(parts, format="csc")


# ---------------------------------------------------------------------------
# Sparse backend
# ---------------------------------------------------------------------------

class _SparseStore(GeneStore):
    """Signature genes held as an in-memory CSC (cells x loaded genes)."""

    def __init__(self, csc, gene_idx: np.ndarray):
        self._csc = csc
        self._col = {int(g): j for j, g in enumerate(gene_idx)}

    def dense(self, idx, order: str = "C"):
        idx = np.asarray(idx, dtype=np.intp)
        m = self._csc
        out = np.zeros((idx.size, m.shape[0]), dtype=np.float64)
        for k, g in enumerate(idx):
            j = self._col[int(g)]
            a, b = m.indptr[j], m.indptr[j + 1]
            out[k, m.indices[a:b]] = m.data[a:b]
        return out


class SparseBackend(ExpressionBackend):
    """Sparse expression matrix (in memory or on disk), never densified.

    Args:
        matrix: scipy CSR/CSC matrix, an anndata backed sparse dataset, or a
            dask array of sparse chunks.
        gene_names, sample_names: labels.
        gene_axis: 1 if ``matrix`` is cells x genes (AnnData convention,
            default), 0 if it is genes x samples.
        chunk_nnz: stored entries processed per scan chunk.
    """

    def __init__(self, matrix, gene_names, sample_names, gene_axis: int = 1,
                 chunk_nnz: int = DEFAULT_CHUNK_NNZ):
        if gene_axis not in (0, 1):
            raise ValueError("gene_axis must be 0 (genes x samples) or 1 (cells x genes)")
        if sp.issparse(matrix):
            if gene_axis == 0:
                matrix = matrix.T  # CSR <-> CSC view, no data copy
            if matrix.format not in ("csr", "csc"):
                matrix = matrix.tocsr()
            reader: _MajorReader = _ScipyReader(matrix)
        elif hasattr(matrix, "group") and hasattr(matrix, "format"):
            if gene_axis == 0:
                raise ValueError("on-disk sparse datasets must be cells x genes")
            reader = _BackedReader(matrix)
        elif hasattr(matrix, "chunks") and hasattr(matrix, "compute"):
            if gene_axis == 0:
                raise ValueError("lazy sparse arrays must be cells x genes")
            reader = _DaskReader(matrix)
        else:
            raise TypeError(f"Unsupported sparse matrix type: {type(matrix)!r}")

        self._reader = reader
        self.gene_names = pd.Index(gene_names)
        self.sample_names = list(sample_names)
        if reader.shape != (len(self.sample_names), len(self.gene_names)):
            raise ValueError(
                f"matrix shape {reader.shape} (cells x genes) does not match "
                f"{len(self.sample_names)} samples x {len(self.gene_names)} genes")
        self.chunk_nnz = int(chunk_nnz)
        self.timings = {}

    @property
    def layout(self) -> str:
        return "gene-major" if self._reader.fmt == "csc" else "cell-major"

    def fingerprint(self) -> dict:
        r = self._reader
        bounds = r.bounds(1 << 16)               # small head / tail chunks
        fp = self._base_fingerprint()
        fp.update(kind="sparse", format=r.fmt, nnz=int(r.nnz))
        for label, (a, b) in (("head", bounds[0]), ("tail", bounds[-1])):
            m = r.read(a, b)
            h = hashlib.sha1()
            for part in (m.data, m.indices, m.indptr):
                h.update(np.ascontiguousarray(part).tobytes())
            fp[label] = h.hexdigest()
            fp["dtype"] = str(m.data.dtype)
        return fp

    # -- chunked scan -------------------------------------------------------

    def _scan_chunks(self) -> Iterator[tuple[np.ndarray, np.ndarray, object]]:
        """Yield ``(gene_ids, values, chunk_matrix)`` for every stored entry."""
        r = self._reader
        for start, stop in r.bounds(self.chunk_nnz):
            t0 = time.perf_counter()
            m = r.read(start, stop)
            if r.fmt == "csr":
                gene_ids = m.indices
            else:
                gene_ids = np.repeat(np.arange(start, stop, dtype=np.intp), np.diff(m.indptr))
            self._tick("scan_read", t0)
            yield gene_ids, m.data, m

    def _scan(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        for gene_ids, values, _ in self._scan_chunks():
            yield gene_ids, values

    # -- gene selection -----------------------------------------------------

    def load_genes(self, idx) -> GeneStore:
        t0 = time.perf_counter()
        uniq = np.unique(np.asarray(idx, dtype=np.intp))
        r = self._reader
        if r.fmt == "csc":
            sub = r.read_major_list(uniq)
        else:
            # Cell-major storage: a gene is spread over every row, so gene
            # selection is one full scan, filtered chunk by chunk.
            parts = [r.read(a, b)[:, uniq] for a, b in r.bounds(self.chunk_nnz)]
            sub = sp.vstack(parts, format="csr") if parts else sp.csr_matrix((0, uniq.size))
        sub = sp.csc_matrix(sub)
        sub.sort_indices()
        self._tick("load_genes", t0)
        return _SparseStore(sub, uniq)

    # -- whole-dataset statistics --------------------------------------------

    def gene_mean_sd(self):
        n_genes, n_samples = self.shape
        nan_counts = np.zeros(n_genes)
        stored = np.zeros(n_genes)
        sums = np.zeros(n_genes)
        for gid, vals in self._scan():
            t0 = time.perf_counter()
            v = vals.astype(np.float64)
            nan = np.isnan(v)
            if nan.any():
                nan_counts += np.bincount(gid[nan], minlength=n_genes)
                v[nan] = 0.0
            stored += np.bincount(gid, minlength=n_genes)
            sums += np.bincount(gid, weights=v, minlength=n_genes)
            self._tick("scan_compute", t0)

        valid = n_samples - nan_counts
        implicit_zeros = n_samples - stored
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = sums / valid

        # Second pass: squared deviations from the mean (same formulation as
        # np.nanstd — no sum-of-squares cancellation).
        ssd = np.zeros(n_genes)
        for gid, vals in self._scan():
            t0 = time.perf_counter()
            v = vals.astype(np.float64)
            with np.errstate(invalid="ignore"):
                dev = v - mean[gid]
                dev *= dev
            nan = np.isnan(v)
            if nan.any():
                dev[nan] = 0.0
            ssd += np.bincount(gid, weights=dev, minlength=n_genes)
            self._tick("scan_compute", t0)

        with np.errstate(invalid="ignore", divide="ignore"):
            zero_term = np.where(implicit_zeros > 0, implicit_zeros * mean * mean, 0.0)
            var = (ssd + zero_term) / (valid - 1)
            var[valid - 1 <= 0] = np.nan
            sd = np.sqrt(var)
        return mean, sd

    def expr_stats(self, threshold):
        n_genes, n_samples = self.shape
        nan_counts = np.zeros(n_genes, dtype=np.int64)
        stored = np.zeros(n_genes, dtype=np.int64)

        if threshold is not None:
            threshold = float(threshold)
            below = self._count_below(threshold, nan_counts, stored)
        else:
            neg = np.zeros(n_genes, dtype=np.int64)
            pos = np.zeros(n_genes, dtype=np.int64)
            for gid, vals in self._scan():
                t0 = time.perf_counter()
                stored += np.bincount(gid, minlength=n_genes)
                nan_counts += np.bincount(gid[np.isnan(vals)], minlength=n_genes)
                neg += np.bincount(gid[vals < 0], minlength=n_genes)
                pos += np.bincount(gid[vals > 0], minlength=n_genes)
                self._tick("scan_compute", t0)
            threshold = self._exact_median(nan_counts == 0, neg, pos)
            if threshold == 0.0:
                below = neg  # stored negatives; implicit zeros are not < 0
            else:
                nan_counts[:] = 0
                stored[:] = 0
                below = self._count_below(threshold, nan_counts, stored)

        return ExprStats(nan_counts, below, threshold)

    def fused_stats(self, threshold: float | None, gene_idx=None):
        """Everything eval_var / eval_expr need, plus the signature genes, in
        two scans of the matrix (used by :class:`DatasetStatsCache`).

        Scan 1: per-gene sums and counts; on cell-major storage the requested
                genes are filtered out of the very same chunks.
        Scan 2: squared deviations (and below-threshold counts when the
                threshold is not 0).
        Numerically identical to ``gene_mean_sd`` + ``expr_stats``.
        """
        n_genes, n_samples = self.shape
        uniq = None if gene_idx is None else np.unique(np.asarray(gene_idx, dtype=np.intp))
        cell_major = self._reader.fmt == "csr"
        parts = []
        nan_counts = np.zeros(n_genes, dtype=np.int64)
        stored = np.zeros(n_genes, dtype=np.int64)
        neg = np.zeros(n_genes, dtype=np.int64)
        pos = np.zeros(n_genes, dtype=np.int64)
        sums = np.zeros(n_genes)
        for gid, vals, m in self._scan_chunks():
            t0 = time.perf_counter()
            v = vals.astype(np.float64)
            nan = np.isnan(v)
            if nan.any():
                nan_counts += np.bincount(gid[nan], minlength=n_genes)
                v[nan] = 0.0
            stored += np.bincount(gid, minlength=n_genes)
            sums += np.bincount(gid, weights=v, minlength=n_genes)
            neg += np.bincount(gid[vals < 0], minlength=n_genes)
            pos += np.bincount(gid[vals > 0], minlength=n_genes)
            self._tick("scan_compute", t0)
            if uniq is not None and cell_major:
                t0 = time.perf_counter()
                parts.append(m[:, uniq])
                self._tick("load_genes", t0)

        store = None
        if uniq is not None:
            if cell_major:
                t0 = time.perf_counter()
                sub = sp.csc_matrix(sp.vstack(parts, format="csr"))
                sub.sort_indices()
                store = _SparseStore(sub, uniq)
                self._tick("load_genes", t0)
            else:
                store = self.load_genes(uniq)

        if threshold is None:
            threshold = self._exact_median(nan_counts == 0, neg, pos)
        threshold = float(threshold)
        count_below = threshold != 0.0

        valid = n_samples - nan_counts
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = sums / valid
        ssd = np.zeros(n_genes)
        below = np.zeros(n_genes, dtype=np.int64)
        for gid, vals in self._scan():
            t0 = time.perf_counter()
            v = vals.astype(np.float64)
            with np.errstate(invalid="ignore"):
                dev = v - mean[gid]
                dev *= dev
            nan = np.isnan(v)
            if nan.any():
                dev[nan] = 0.0
            ssd += np.bincount(gid, weights=dev, minlength=n_genes)
            if count_below:
                below += np.bincount(gid[vals < threshold], minlength=n_genes)
            self._tick("scan_compute", t0)

        implicit_zeros = n_samples - stored
        with np.errstate(invalid="ignore", divide="ignore"):
            zero_term = np.where(implicit_zeros > 0, implicit_zeros * mean * mean, 0.0)
            var = (ssd + zero_term) / (valid - 1)
            var[valid - 1 <= 0] = np.nan
            sd = np.sqrt(var)
        if not count_below:
            below = neg                      # stored negatives; implicit zeros are not < 0
        elif 0.0 < threshold:
            below = below + implicit_zeros   # implicit zeros are below too
        return mean, sd, ExprStats(nan_counts, below, threshold), store

    def _count_below(self, threshold, nan_counts, stored):
        n_genes, n_samples = self.shape
        below = np.zeros(n_genes, dtype=np.int64)
        for gid, vals in self._scan():
            t0 = time.perf_counter()
            stored += np.bincount(gid, minlength=n_genes)
            nan_counts += np.bincount(gid[np.isnan(vals)], minlength=n_genes)
            below += np.bincount(gid[vals < threshold], minlength=n_genes)
            self._tick("scan_compute", t0)
        if 0.0 < threshold:
            below = below + (n_samples - stored)  # implicit zeros are below too
        return below

    def _exact_median(self, clean, neg, pos) -> float:
        """Exact median of every value (implicit zeros included) of clean genes.

        Only counts are needed when the median falls among the zeros — the
        usual single-cell case. Otherwise the stored values of the relevant
        sign are gathered and partitioned; nothing is approximated.
        """
        n_samples = self.shape[1]
        total = int(clean.sum()) * n_samples
        if total == 0:
            return float("nan")
        n_neg = int(neg[clean].sum())
        n_pos = int(pos[clean].sum())
        n_zero = total - n_neg - n_pos
        ranks = sorted({(total - 1) // 2, total // 2})
        if all(n_neg <= k < n_neg + n_zero for k in ranks):
            return 0.0

        t0 = time.perf_counter()
        need_neg = any(k < n_neg for k in ranks)
        need_pos = any(k >= n_neg + n_zero for k in ranks)
        neg_vals, pos_vals = [], []
        for gid, vals in self._scan():
            keep = clean[gid]
            if need_neg:
                neg_vals.append(vals[keep & (vals < 0)])
            if need_pos:
                pos_vals.append(vals[keep & (vals > 0)])
        neg_arr = np.concatenate(neg_vals) if neg_vals else np.empty(0)
        pos_arr = np.concatenate(pos_vals) if pos_vals else np.empty(0)

        picked = []
        for k in ranks:
            if k < n_neg:
                picked.append(float(np.partition(neg_arr, k)[k]))
            elif k < n_neg + n_zero:
                picked.append(0.0)
            else:
                j = k - n_neg - n_zero
                picked.append(float(np.partition(pos_arr, j)[j]))
        self._tick("median_partition", t0)
        return float(np.mean(picked))


# ---------------------------------------------------------------------------
# Shared per-dataset cache
# ---------------------------------------------------------------------------

class DatasetStatsCache(ExpressionBackend):
    """Memoising wrapper that lets the five modules share per-dataset work.

    Reused across eval_var / eval_expr / eval_compactness / compare_metrics /
    eval_stan: per-gene mean and SD, NaN and below-threshold counts, the
    expression threshold, gene positions, and the fetched signature genes.

    ``prepare()`` is optional: for sparse backends it computes all of the
    above in two scans of the matrix instead of one or two scans per module
    (on cell-major storage every gene fetch is a full scan on its own).
    Results are those of the wrapped backend — the cache never changes values.
    """

    def __init__(self, backend):
        self._b = as_backend(backend)
        self.gene_names = self._b.gene_names
        self.sample_names = self._b.sample_names
        self.timings = self._b.timings       # one shared profile
        self._mean_sd = None
        self._expr: dict = {}
        self._store = None
        self._store_genes: set = set()
        self._positions: dict = {}

    @property
    def backend(self) -> ExpressionBackend:
        return self._b

    def fingerprint(self) -> dict:
        return self._b.fingerprint()

    def prepare(self, gene_idx=None, threshold: float | None = None) -> None:
        """Compute (or complete) everything the modules will ask for. Whatever
        was seeded from persisted statistics is not recomputed."""
        t0 = time.perf_counter()
        key = None if threshold is None else float(threshold)
        fused = getattr(self._b, "fused_stats", None)
        if fused is not None and self._mean_sd is None and key not in self._expr:
            mean, sd, stats, store = fused(threshold, gene_idx)
            self._mean_sd = (mean, sd)
            self._expr[key] = stats
            self._expr[stats.threshold] = stats
            if store is not None:
                self._set_store(store, gene_idx)
        else:
            self.gene_mean_sd()
            self.expr_stats(threshold)
            if gene_idx is not None:
                self.load_genes(gene_idx)
        self._tick("cache_prepare", t0)

    def has_stats(self, threshold: float | None = None) -> bool:
        """True when mean/SD and the expression counts for ``threshold`` are
        already available (computed or loaded)."""
        key = None if threshold is None else float(threshold)
        return self._mean_sd is not None and key in self._expr

    # -- persistence -----------------------------------------------------------

    STATS_FORMAT = 1

    def save_stats(self, path) -> None:
        """Persist the per-gene statistics computed so far (they depend on the
        matrix only, not on the signatures) together with a fingerprint of the
        matrix. Written atomically."""
        if self._mean_sd is None:
            raise ValueError("no statistics computed yet: call prepare() or gene_mean_sd() first")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        entries = {}                               # one entry per distinct threshold
        for key, st in self._expr.items():
            label = "median" if key is None else repr(float(key))
            entries.setdefault(label, st)
        if "median" in entries:                    # the median's own value is a duplicate key
            entries.pop(repr(float(entries["median"].threshold)), None)
        arrays = {"mean": self._mean_sd[0], "sd": self._mean_sd[1]}
        labels = []
        for i, (label, st) in enumerate(entries.items()):
            labels.append([label, float(st.threshold)])
            arrays[f"nan_counts_{i}"] = st.nan_counts
            arrays[f"below_counts_{i}"] = st.below_counts
        header = {"format": self.STATS_FORMAT, "fingerprint": self._b.fingerprint(),
                  "thresholds": labels}
        tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
        with open(tmp, "wb") as fh:
            np.savez(fh, header=np.array(json.dumps(header)), **arrays)
        os.replace(tmp, path)

    def load_stats(self, path) -> bool:
        """Seed the cache from :meth:`save_stats` output. Returns False — and
        loads nothing — when the file is missing, unreadable or was written
        for a different matrix (fingerprint mismatch)."""
        path = Path(path)
        if not path.exists():
            return False
        try:
            with np.load(path, allow_pickle=False) as z:
                header = json.loads(str(z["header"]))
                if header.get("format") != self.STATS_FORMAT:
                    return False
                if header["fingerprint"] != self._b.fingerprint():
                    warnings.warn(f"{path}: statistics were computed for a different "
                                  "matrix; ignoring them", RuntimeWarning, stacklevel=2)
                    return False
                mean, sd = z["mean"], z["sd"]
                loaded = [(label, ExprStats(z[f"nan_counts_{i}"], z[f"below_counts_{i}"], float(thr)))
                          for i, (label, thr) in enumerate(header["thresholds"])]
        except (OSError, ValueError, KeyError) as e:
            warnings.warn(f"{path}: unreadable statistics file ({e}); ignoring it",
                          RuntimeWarning, stacklevel=2)
            return False
        if mean.shape[0] != self.shape[0]:
            return False
        self._mean_sd = (mean, sd)
        for label, st in loaded:
            if label == "median":
                self._expr[None] = st
            self._expr[st.threshold] = st
        return True

    def export_stats(self) -> dict:
        """Per-gene statistics as plain arrays, e.g. to persist next to the
        dataset (they depend on the matrix only, not on the signatures)."""
        mean, sd = self.gene_mean_sd()
        stats = self.expr_stats(None)
        return {"mean": mean, "sd": sd, "nan_counts": stats.nan_counts,
                "below_counts": stats.below_counts, "threshold": np.float64(stats.threshold)}

    def seed_stats(self, stats: dict) -> None:
        """Load statistics produced by :meth:`export_stats` for the same matrix,
        so that no scan of the matrix is needed for eval_var / eval_expr."""
        self._mean_sd = (np.asarray(stats["mean"]), np.asarray(stats["sd"]))
        expr = ExprStats(np.asarray(stats["nan_counts"]), np.asarray(stats["below_counts"]),
                         float(stats["threshold"]))
        self._expr[None] = expr
        self._expr[expr.threshold] = expr

    def _set_store(self, store, gene_idx) -> None:
        self._store = store
        self._store_genes = set(np.asarray(gene_idx, dtype=np.intp).tolist())

    def gene_indices(self, genes) -> np.ndarray:
        key = tuple(genes)
        if key not in self._positions:
            self._positions[key] = self._b.gene_indices(key)
        return self._positions[key]

    def load_genes(self, idx) -> GeneStore:
        wanted = set(np.asarray(idx, dtype=np.intp).tolist())
        if self._store is None or not wanted <= self._store_genes:
            union = np.array(sorted(wanted | self._store_genes), dtype=np.intp)
            self._set_store(self._b.load_genes(union), union)
        return self._store

    def gene_mean_sd(self):
        if self._mean_sd is None:
            self._mean_sd = self._b.gene_mean_sd()
        return self._mean_sd

    def expr_stats(self, threshold):
        key = None if threshold is None else float(threshold)
        if key not in self._expr:
            self._expr[key] = self._b.expr_stats(threshold)
        return self._expr[key]


# ---------------------------------------------------------------------------
# AnnData
# ---------------------------------------------------------------------------

def AnnDataBackend(adata, layer: str | None = None,
                   chunk_nnz: int = DEFAULT_CHUNK_NNZ) -> ExpressionBackend:
    """Backend for an AnnData object (in memory, ``backed='r'`` or lazy).

    ``adata.X`` (or ``adata.layers[layer]``) is used as is — cells x genes —
    and is never loaded or transposed as a whole. Works with scipy sparse
    matrices, backed H5AD/Zarr sparse datasets and dask-lazy sparse arrays;
    a dense ``X`` falls back to :class:`DenseBackend` through a transposed view.
    """
    x = adata.X if layer is None else adata.layers[layer]
    genes = pd.Index(adata.var_names)
    cells = list(adata.obs_names)
    if isinstance(x, np.ndarray):
        return DenseBackend(pd.DataFrame(x.T, index=genes, columns=cells, copy=False))
    return SparseBackend(x, genes, cells, gene_axis=1, chunk_nnz=chunk_nnz)


def as_backend(obj) -> ExpressionBackend:
    """Coerce a pipeline input (DataFrame, backend, AnnData) to a backend."""
    if isinstance(obj, ExpressionBackend):
        return obj
    if isinstance(obj, pd.DataFrame):
        return DenseBackend(obj)
    if hasattr(obj, "var_names") and hasattr(obj, "obs_names"):
        return AnnDataBackend(obj)
    raise TypeError(
        f"Unsupported expression matrix type {type(obj)!r}: expected a pandas "
        "DataFrame (genes x samples), an ExpressionBackend or an AnnData. Wrap "
        "scipy sparse matrices with SparseBackend(matrix, gene_names, sample_names).")
