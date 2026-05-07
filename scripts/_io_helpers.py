"""Self-contained I/O helpers for the test_metrics.py CLI script.

Provides a minimal GMT parser and an Ensembl-to-HGNC mapping (mygene.info,
disk-cached) used to feed the pysigqc-metrics radar pipeline from raw TCGA
inputs. Kept private to scripts/ so the package itself stays free of
TCGA/network-specific dependencies.
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib import request as urlrequest
from urllib.error import URLError

import pandas as pd


def parse_gmt_signatures(path: Path) -> dict[str, list[str]]:
    """Parse an MSigDB-style GMT file (``name<TAB>url<TAB>gene1<TAB>gene2...``).

    Some sources use spaces instead of tabs; we fall back to space splitting
    when no tab is present. Empty gene tokens are dropped.
    """
    signatures: dict[str, list[str]] = {}
    for line in path.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) == 1:
            parts = line.split(" ")
        if len(parts) < 3:
            continue
        name, _description, *genes = parts
        signatures[name] = [g for g in genes if g]
    return signatures


def _read_mygene_cache(cache_path: Path) -> dict[str, str]:
    """Load a previously written ensembl-id -> hgnc-symbol map from disk.

    Lines may be ``ENSGxxxxxx<TAB>SYMBOL`` (mapped) or just ``ENSGxxxxxx``
    (queried, no HGNC symbol — recorded so we don't re-query it).
    """
    cache: dict[str, str] = {}
    if not cache_path.exists():
        return cache
    for ln in cache_path.read_text().splitlines():
        if "\t" in ln:
            ens, sym = ln.split("\t", 1)
            if ens:
                cache[ens] = sym
        else:
            ens = ln.strip()
            if ens:
                cache[ens] = ""
    return cache


def _write_mygene_cache(cache: dict[str, str], cache_path: Path) -> None:
    cache_path.write_text(
        "\n".join(f"{e}\t{s}" if s else e
                  for e, s in sorted(cache.items())) + "\n"
    )


def query_mygene_ensembl_to_symbol(ensembl_ids: list[str],
                                   cache_path: Path,
                                   batch_size: int = 500,
                                   timeout: float = 30.0) -> dict[str, str]:
    """Map Ensembl gene IDs to HGNC symbols via mygene.info, caching to disk."""
    cache = _read_mygene_cache(cache_path)
    missing = [eid for eid in ensembl_ids if eid not in cache]
    if not missing:
        return cache

    url = "https://mygene.info/v3/query"
    total_batches = (len(missing) + batch_size - 1) // batch_size
    print(f"[mygene] querying {len(missing)} Ensembl IDs in {total_batches} "
          f"batches of {batch_size}")
    for batch_start in range(0, len(missing), batch_size):
        chunk = missing[batch_start:batch_start + batch_size]
        body = ("q=" + ",".join(chunk)
                + "&scopes=ensembl.gene&fields=symbol&species=human&size=1").encode()
        req = urlrequest.Request(
            url, data=body,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        try:
            with urlrequest.urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode())
        except (URLError, TimeoutError, OSError) as err:
            batch_index = batch_start // batch_size + 1
            print(f"[mygene] batch {batch_index}/{total_batches} "
                  f"network error: {err}; stopping early")
            break
        for eid in chunk:
            cache.setdefault(eid, "")
        for hit in payload:
            eid = hit.get("query")
            sym = hit.get("symbol")
            if eid and isinstance(sym, str) and sym:
                cache[eid] = sym
        _write_mygene_cache(cache, cache_path)
        batch_index = batch_start // batch_size + 1
        if batch_index % 10 == 0:
            mapped = sum(1 for v in cache.values() if v)
            print(f"[mygene] progress {batch_index}/{total_batches} "
                  f"batches, mapped so far: {mapped}")
    return cache


def convert_ensembl_matrix_to_hgnc(expr: pd.DataFrame,
                                   cache_path: Path) -> pd.DataFrame:
    """Convert an Ensembl-indexed expression matrix to HGNC-indexed.

    1. strip Ensembl version suffix (``ENSG...123.7`` -> ``ENSG...123``)
    2. query mygene.info for ``{ensembl_id: hgnc_symbol}``
    3. drop rows whose HGNC symbol is missing/empty
    4. rename remaining rows to the HGNC symbol
    5. drop duplicate HGNC symbols, keeping the first occurrence
    """
    ensembl_no_version = (
        expr.index.to_series().str.replace(r"\.[0-9]+$", "", regex=True).tolist()
    )
    unique_ensembl = sorted(set(ensembl_no_version))
    ensembl_to_symbol = query_mygene_ensembl_to_symbol(unique_ensembl, cache_path)

    symbols = [ensembl_to_symbol.get(e, "") for e in ensembl_no_version]
    keep_mask = [bool(s) for s in symbols]
    n_total = len(symbols)
    n_keep = sum(keep_mask)
    print(f"[map] Ensembl->HGNC: {n_keep}/{n_total} rows have an HGNC symbol")

    expr_with_symbols = expr.loc[keep_mask].copy()
    expr_with_symbols.index = [s for s, k in zip(symbols, keep_mask) if k]

    duplicate_mask = expr_with_symbols.index.duplicated(keep="first")
    if duplicate_mask.any():
        print(f"[map] dropping {int(duplicate_mask.sum())} duplicated HGNC symbols")
    return expr_with_symbols.loc[~duplicate_mask]
