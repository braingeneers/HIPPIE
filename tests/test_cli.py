"""Smoke tests for the hippie-cli entry point.

These cover the user-facing CLI surface declared in pyproject.toml:
    hippie-cli validate-data <dir>
    hippie-cli embed --datasets-root ... --datasets ... --output ...
    hippie-cli predict --reference ... --query ... --output ...

The embed/predict tests are gated behind a `network` marker because they
require the HuggingFace Hub download (~290 MB checkpoint). Run with:

    pytest tests/test_cli.py -m "not network"     # default, fast
    pytest tests/test_cli.py -m network           # full smoke test
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


def _run(*argv: str) -> subprocess.CompletedProcess:
    """Invoke `python -m hippie.cli` so the test is independent of $PATH."""
    return subprocess.run(
        [sys.executable, "-m", "hippie.cli", *argv],
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_help_lists_subcommands() -> None:
    result = _run("--help")
    assert result.returncode == 0
    out = result.stdout
    for sub in ("validate-data", "embed", "predict"):
        assert sub in out, f"`hippie-cli --help` is missing the `{sub}` subcommand"


def test_cli_validate_toy_dataset(toy_dataset_dir: Path) -> None:
    result = _run("validate-data", str(toy_dataset_dir))
    assert result.returncode == 0
    assert "OK" in result.stdout


def test_cli_validate_rejects_nonexistent_path(tmp_path: Path) -> None:
    result = _run("validate-data", str(tmp_path / "does-not-exist"))
    assert result.returncode != 0


def test_cli_validate_rejects_missing_csv(tmp_dataset: Path) -> None:
    (tmp_dataset / "waveforms.csv").unlink()
    result = _run("validate-data", str(tmp_dataset))
    assert result.returncode == 1
    assert "waveforms.csv" in (result.stdout + result.stderr)


@pytest.mark.network
def test_cli_embed_writes_npz_with_expected_keys(
    toy_dataset_dir: Path, tmp_path: Path
) -> None:
    """End-to-end: download the pretrained ckpt and embed the toy dataset."""
    out = tmp_path / "toy_emb.npz"
    result = _run(
        "embed",
        "--datasets-root", str(toy_dataset_dir.parent),
        "--datasets", "toy",
        "--output", str(out),
        "--batch-size", "8",
        "--device", "cpu",
    )
    assert result.returncode == 0, result.stderr
    assert out.exists()

    import numpy as np
    npz = np.load(out, allow_pickle=True)
    for key in ("embeddings", "labels", "dataset_ids", "technology_ids", "neuron_ids"):
        assert key in npz.files, f"output npz missing key: {key}"
    assert npz["embeddings"].shape[0] == 20
