"""Shared pytest fixtures for the HIPPIE release test suite."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
TOY_DATASET = REPO_ROOT / "datasets_hippie" / "toy"


@pytest.fixture(scope="session")
def toy_dataset_dir() -> Path:
    """Path to the committed toy dataset (20 synthetic units).

    The toy dataset is regenerable via `scripts/generate_toy_dataset.py`
    and shipped in the repo so tests can run with no external downloads.
    """
    assert TOY_DATASET.is_dir(), (
        f"Toy dataset missing at {TOY_DATASET}. "
        "Regenerate with: python scripts/generate_toy_dataset.py"
    )
    return TOY_DATASET


@pytest.fixture
def tmp_dataset(tmp_path: Path, toy_dataset_dir: Path) -> Path:
    """A writable copy of the toy dataset for tests that mutate CSVs."""
    dst = tmp_path / "toy"
    shutil.copytree(toy_dataset_dir, dst)
    return dst
