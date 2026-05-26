"""Inference-API tests for hippie.inference.HIPPIEClassifier.

These tests exercise the public inference surface without requiring a
real pretrained checkpoint. Network-dependent tests (from_pretrained on
HuggingFace Hub) are marked with the `network` pytest marker.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hippie.inference import HIPPIEClassifier, TECHNOLOGY_IDS


def test_technology_ids_known_keys() -> None:
    """The released checkpoint maps these three keys; preserve them."""
    assert "neuropixels" in TECHNOLOGY_IDS
    assert "silicon_probe" in TECHNOLOGY_IDS
    assert "juxtacellular" in TECHNOLOGY_IDS
    assert TECHNOLOGY_IDS["neuropixels"] == 0
    assert len(set(TECHNOLOGY_IDS.values())) == len(TECHNOLOGY_IDS)


def test_classifier_rejects_unknown_tech_string() -> None:
    """get_embeddings(tech_id='nonsense') must fail with a clear error."""

    class _Stub:
        def to(self, _device):
            return self
        def eval(self):
            return self

    clf = HIPPIEClassifier(_Stub(), device="cpu")
    wave = np.zeros((1, 50), dtype=np.float32)
    isi = np.zeros((1, 100), dtype=np.float32)
    acg = np.zeros((1, 100), dtype=np.float32)
    with pytest.raises(ValueError, match="Unknown tech_id"):
        clf.get_embeddings(wave=wave, isi=isi, acg=acg, tech_id="not_a_real_tech")


@pytest.mark.network
def test_load_pretrained_checkpoint_from_hub_and_embed_toy(
    toy_dataset_dir: Path,
) -> None:
    """Full inference round-trip against the released checkpoint."""
    import pandas as pd
    import torch.nn.functional as F
    import torch

    classifier = HIPPIEClassifier.from_pretrained(device="cpu")

    def _resample_minmax(raw, target_len):
        t = torch.as_tensor(raw, dtype=torch.float32).unsqueeze(1)
        t = F.interpolate(t, size=(target_len,), mode="linear", align_corners=False).squeeze(1)
        mn = t.amin(dim=-1, keepdim=True)
        mx = t.amax(dim=-1, keepdim=True)
        return ((t - mn) / (mx - mn + 1e-8) * 2.0 - 1.0).numpy().astype(np.float32)

    wf = pd.read_csv(toy_dataset_dir / "waveforms.csv").to_numpy().astype(np.float32)
    isi = pd.read_csv(toy_dataset_dir / "isi_dist.csv").to_numpy().astype(np.float32)
    acg = pd.read_csv(toy_dataset_dir / "acg.csv").to_numpy().astype(np.float32)

    emb = classifier.get_embeddings(
        wave=_resample_minmax(wf, 50),
        isi=_resample_minmax(np.log(isi + 1.0), 100),
        acg=_resample_minmax(acg, 100),
        tech_id="neuropixels",
        batch_size=8,
    )
    assert emb.shape[0] == 20
    assert emb.shape[1] == 30  # locked z_dim for the released checkpoint
    assert np.isfinite(emb).all()
