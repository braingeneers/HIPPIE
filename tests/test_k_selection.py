"""Tests for the CV-based k selector in hippie.inference.select_k_via_cv.

The selector mirrors the procedure in
``hippie_benchmarking_release/scripts/cross_dataset_script.py``:
  * L2-normalize reference embeddings
  * cv = min(5, n_classes)
  * scoring = "balanced_accuracy"
  * try k = 1..20, return argmax
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hippie.inference import select_k_via_cv


def _two_blobs(rng: np.random.Generator, n_per_class: int = 40, dim: int = 8):
    """Two well-separated Gaussian blobs in `dim`-D."""
    a = rng.normal(loc=+2.0, scale=0.5, size=(n_per_class, dim))
    b = rng.normal(loc=-2.0, scale=0.5, size=(n_per_class, dim))
    emb = np.vstack([a, b]).astype(np.float32)
    labels = np.array(["A"] * n_per_class + ["B"] * n_per_class)
    return emb, labels


def test_select_k_returns_int_and_score_table() -> None:
    rng = np.random.default_rng(0)
    emb, labels = _two_blobs(rng)
    best_k, scores = select_k_via_cv(emb, labels)
    assert isinstance(best_k, int)
    assert 1 <= best_k <= 20
    assert best_k in scores
    # Every evaluated k must score in [0, 1] for balanced_accuracy.
    for k, s in scores.items():
        assert 0.0 <= s <= 1.0, f"k={k} got score={s}"


def test_select_k_picks_low_k_on_easy_separable_data() -> None:
    """Well-separated classes -> small k is enough -> selector picks k <= 5."""
    rng = np.random.default_rng(1)
    emb, labels = _two_blobs(rng, n_per_class=30, dim=8)
    best_k, scores = select_k_via_cv(emb, labels)
    assert best_k <= 5, f"expected small k on easy data; got {best_k} (scores={scores})"
    assert scores[best_k] > 0.9, f"expected near-perfect CV score; got {scores[best_k]:.3f}"


def test_select_k_rejects_single_class() -> None:
    rng = np.random.default_rng(2)
    emb = rng.normal(size=(20, 8)).astype(np.float32)
    labels = np.array(["only"] * 20)
    with pytest.raises(ValueError, match="2 classes"):
        select_k_via_cv(emb, labels)


def test_select_k_handles_small_class_count() -> None:
    """If a class has fewer samples than the default cv=5, the helper should
    shrink cv rather than crash."""
    rng = np.random.default_rng(3)
    # 4 samples of class A, 4 of class B -- can't do 5-fold but 4-fold is fine.
    emb_a = rng.normal(loc=+2.0, scale=0.3, size=(4, 6))
    emb_b = rng.normal(loc=-2.0, scale=0.3, size=(4, 6))
    emb = np.vstack([emb_a, emb_b]).astype(np.float32)
    labels = np.array(["A"] * 4 + ["B"] * 4)
    best_k, scores = select_k_via_cv(emb, labels)
    assert 1 <= best_k <= len(scores) * 2  # sanity
    assert len(scores) >= 1, "should evaluate at least one k"


def test_select_k_skips_k_larger_than_fold_size() -> None:
    """If the per-fold training set is smaller than some candidate k, that k
    must be skipped silently (not crash)."""
    rng = np.random.default_rng(4)
    emb_a = rng.normal(loc=+2.0, scale=0.3, size=(4, 6))
    emb_b = rng.normal(loc=-2.0, scale=0.3, size=(4, 6))
    emb = np.vstack([emb_a, emb_b]).astype(np.float32)
    labels = np.array(["A"] * 4 + ["B"] * 4)
    best_k, scores = select_k_via_cv(emb, labels, k_candidates=range(1, 21))
    # 8 samples, cv=4 -> each training fold has 6 samples -> k=7,8,...,20 skipped
    assert all(k <= 6 for k in scores), f"large k should be skipped; got {sorted(scores)}"


def _write_two_blob_npz(rng, ref_npz, qry_npz, n_ref_per_class=25, n_qry=10):
    """Write a labeled-reference + unlabeled-query pair of .npz files."""
    ref_emb = np.vstack([
        rng.normal(loc=+2.0, scale=0.4, size=(n_ref_per_class, 8)),
        rng.normal(loc=-2.0, scale=0.4, size=(n_ref_per_class, 8)),
    ]).astype(np.float32)
    ref_labels = np.array(["A"] * n_ref_per_class + ["B"] * n_ref_per_class)
    qry_emb = rng.normal(loc=+2.0, scale=0.4, size=(n_qry, 8)).astype(np.float32)

    np.savez(ref_npz, embeddings=ref_emb, labels=ref_labels,
             dataset_ids=np.array(["ref"] * len(ref_labels)),
             technology_ids=np.zeros(len(ref_labels), dtype=np.int32),
             neuron_ids=np.arange(len(ref_labels)))
    np.savez(qry_npz, embeddings=qry_emb,
             dataset_ids=np.array(["qry"] * n_qry),
             technology_ids=np.zeros(n_qry, dtype=np.int32),
             neuron_ids=np.arange(n_qry))


def test_cli_predict_default_k_is_10_and_skips_cv(tmp_path: Path) -> None:
    """`hippie-cli predict` without --k must use k=10 and not run CV."""
    rng = np.random.default_rng(5)
    ref_npz = tmp_path / "ref.npz"
    qry_npz = tmp_path / "qry.npz"
    out_csv = tmp_path / "predictions.csv"
    _write_two_blob_npz(rng, ref_npz, qry_npz)

    result = subprocess.run(
        [sys.executable, "-m", "hippie.cli", "predict",
         "--reference", str(ref_npz), "--query", str(qry_npz),
         "--output", str(out_csv)],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Using k = 10" in result.stdout
    assert "Selected best k" not in result.stdout, "default must not invoke CV"

    preds = pd.read_csv(out_csv)
    assert (preds["k_used"] == 10).all()


def test_cli_predict_writes_confidence_matrix(tmp_path: Path) -> None:
    """Output CSV must include a `confidence` column AND one `prob_<class>`
    column per training-set class. Each row's per-class probs must sum to ~1."""
    rng = np.random.default_rng(6)
    ref_npz = tmp_path / "ref.npz"
    qry_npz = tmp_path / "qry.npz"
    out_csv = tmp_path / "predictions.csv"
    _write_two_blob_npz(rng, ref_npz, qry_npz)

    result = subprocess.run(
        [sys.executable, "-m", "hippie.cli", "predict",
         "--reference", str(ref_npz), "--query", str(qry_npz),
         "--output", str(out_csv)],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr

    preds = pd.read_csv(out_csv)
    # One probability column per training class.
    prob_cols = [c for c in preds.columns if c.startswith("prob_")]
    assert sorted(prob_cols) == ["prob_A", "prob_B"]
    # Confidence is the max per-class probability.
    np.testing.assert_allclose(
        preds["confidence"].values,
        preds[prob_cols].max(axis=1).values,
        atol=1e-9,
    )
    # Probabilities sum to ~1.0.
    row_sums = preds[prob_cols].sum(axis=1).values
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-9)
    # On well-separated data, blob A is the unanimous prediction.
    assert (preds["prediction"] == "A").all()


def test_cli_predict_auto_k_runs_cv(tmp_path: Path) -> None:
    """`hippie-cli predict --k auto` should run CV and write a k_used column."""
    rng = np.random.default_rng(7)
    ref_npz = tmp_path / "ref.npz"
    qry_npz = tmp_path / "qry.npz"
    out_csv = tmp_path / "predictions.csv"
    _write_two_blob_npz(rng, ref_npz, qry_npz)

    result = subprocess.run(
        [sys.executable, "-m", "hippie.cli", "predict",
         "--reference", str(ref_npz), "--query", str(qry_npz),
         "--output", str(out_csv), "--k", "auto"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Selected best k" in result.stdout

    preds = pd.read_csv(out_csv)
    assert "k_used" in preds.columns
    assert preds["k_used"].nunique() == 1
    assert 1 <= int(preds["k_used"].iloc[0]) <= 20
    assert (preds["prediction"] == "A").all()


def test_cli_predict_explicit_k_skips_cv(tmp_path: Path) -> None:
    """`hippie-cli predict --k 3` should honor the user's k and not run CV."""
    rng = np.random.default_rng(8)
    ref_npz = tmp_path / "ref.npz"
    qry_npz = tmp_path / "qry.npz"
    out_csv = tmp_path / "predictions.csv"
    _write_two_blob_npz(rng, ref_npz, qry_npz, n_qry=5)

    result = subprocess.run(
        [sys.executable, "-m", "hippie.cli", "predict",
         "--reference", str(ref_npz), "--query", str(qry_npz),
         "--output", str(out_csv), "--k", "3"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Selected best k" not in result.stdout
    preds = pd.read_csv(out_csv)
    assert (preds["k_used"] == 3).all()
