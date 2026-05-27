"""Schema validation tests against the toy dataset.

These tests exercise the canonical CSV layout HIPPIE consumes:
    waveforms.csv (N x T) -- trough-centered spike templates
    isi_dist.csv  (N x ~100) -- 1 ms bins, 0..100 ms
    acg.csv       (N x ~201) -- 1 ms bins, -100..+100 ms  (optional)
    labels.csv    (N x >=1)  -- cell-type label in last column (optional)

The validator under test is hippie.cli.cmd_validate.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hippie.cli import cmd_validate


class _Args:
    """Minimal stand-in for argparse.Namespace."""

    def __init__(self, path: str) -> None:
        self.path = path


def test_validate_minimal_dataset_schema(toy_dataset_dir: Path) -> None:
    """The shipped toy dataset should validate cleanly (return code 0)."""
    rc = cmd_validate(_Args(str(toy_dataset_dir)))
    assert rc == 0


def test_reject_missing_required_waveforms(tmp_dataset: Path) -> None:
    os.remove(tmp_dataset / "waveforms.csv")
    rc = cmd_validate(_Args(str(tmp_dataset)))
    assert rc == 1, "validator must reject a dataset missing waveforms.csv"


def test_reject_missing_required_isi(tmp_dataset: Path) -> None:
    os.remove(tmp_dataset / "isi_dist.csv")
    rc = cmd_validate(_Args(str(tmp_dataset)))
    assert rc == 1, "validator must reject a dataset missing isi_dist.csv"


def test_reject_row_count_mismatch(tmp_dataset: Path) -> None:
    isi = pd.read_csv(tmp_dataset / "isi_dist.csv")
    isi.iloc[:10].to_csv(tmp_dataset / "isi_dist.csv", index=False)
    rc = cmd_validate(_Args(str(tmp_dataset)))
    assert rc == 1, "validator must reject mismatched row counts across modalities"


def test_accept_unlabeled_dataset_for_embedding(tmp_dataset: Path) -> None:
    """labels.csv is optional -- embedding extraction must still work."""
    os.remove(tmp_dataset / "labels.csv")
    rc = cmd_validate(_Args(str(tmp_dataset)))
    assert rc == 0, "validator must accept datasets without labels.csv (embedding-only)"


def test_accept_missing_acg(tmp_dataset: Path) -> None:
    """acg.csv is optional -- bimodal mode is supported."""
    os.remove(tmp_dataset / "acg.csv")
    rc = cmd_validate(_Args(str(tmp_dataset)))
    assert rc == 0


def test_toy_waveform_shape(toy_dataset_dir: Path) -> None:
    """Canonical layout: 50-sample waveform, trough at sample 20."""
    wf = pd.read_csv(toy_dataset_dir / "waveforms.csv").to_numpy()
    assert wf.shape == (20, 50), f"toy waveforms.csv has shape {wf.shape}"
    troughs = np.argmin(wf, axis=1)
    assert np.median(troughs) == pytest.approx(20, abs=2)


def test_toy_isi_shape(toy_dataset_dir: Path) -> None:
    isi = pd.read_csv(toy_dataset_dir / "isi_dist.csv").to_numpy()
    assert isi.shape == (20, 100), f"toy isi_dist.csv has shape {isi.shape}"
    assert np.all(isi >= 0), "ISI must be non-negative spike counts"


def test_toy_acg_shape(toy_dataset_dir: Path) -> None:
    acg = pd.read_csv(toy_dataset_dir / "acg.csv").to_numpy()
    assert acg.shape == (20, 201), f"toy acg.csv has shape {acg.shape}"
    assert acg[:, 100].max() == 0.0, "center bin (lag 0) must be zeroed"


def test_toy_labels(toy_dataset_dir: Path) -> None:
    labels = pd.read_csv(toy_dataset_dir / "labels.csv")
    assert len(labels) == 20
    assert set(labels.iloc[:, -1].unique()) == {"PV", "SOM"}
