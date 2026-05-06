"""Shared test fixtures for pysigqc-metrics — loads the same data as R tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
REF_DIR = FIXTURES_DIR / "reference_outputs"


@pytest.fixture(scope="session")
def fixture_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture(scope="session")
def ref_dir() -> Path:
    return REF_DIR


@pytest.fixture(scope="session")
def datasets() -> dict[str, pd.DataFrame]:
    """Load expression matrices from CSV (genes x samples)."""
    ds = {}
    for name in ["dataset_A", "dataset_B"]:
        df = pd.read_csv(FIXTURES_DIR / f"fixture_{name}.csv", index_col=0)
        ds[name] = df
    return ds


@pytest.fixture(scope="session")
def signatures() -> dict[str, list[str]]:
    """Load gene signatures from CSV (long format: signature, gene)."""
    df = pd.read_csv(FIXTURES_DIR / "fixture_signatures.csv")
    sigs: dict[str, list[str]] = {}
    for sig_name, group in df.groupby("signature"):
        sigs[sig_name] = group["gene"].tolist()
    return sigs


@pytest.fixture(scope="session")
def names_sigs() -> list[str]:
    return ["compact_sig", "diffuse_sig"]


@pytest.fixture(scope="session")
def names_datasets() -> list[str]:
    return ["dataset_A", "dataset_B"]
