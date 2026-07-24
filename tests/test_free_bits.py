#!/usr/bin/env python3
"""Verify the free-bits KL floor behaves as documented."""

import torch

from hippie.multimodal_model import (
    CVAEConfig,
    ExperimentConfigs,
    MultiModalCVAE,
    MultiModalCVAETrainModule,
)


def _module(free_bits: float) -> MultiModalCVAETrainModule:
    config = ExperimentConfigs.unconditioned()
    config.free_bits = free_bits
    model = MultiModalCVAE(
        modalities={"wave": 50, "isi": 100, "acg": 100},
        z_dim=8,
        num_sources=None,
        num_classes=None,
        num_super_regions=0,
        num_layers=0,
        config=config,
    )
    return MultiModalCVAETrainModule(base_model=model, config=config)


def test_defaults_off_for_every_preset():
    """Every published preset keeps the original objective."""
    assert CVAEConfig().free_bits == 0.0
    for name in (m for m in dir(ExperimentConfigs) if not m.startswith("_")):
        assert getattr(ExperimentConfigs, name)().free_bits == 0.0, name


def test_disabled_matches_plain_kl():
    """free_bits=0 reproduces sum-over-dims, mean-over-batch KL exactly."""
    zmean = torch.randn(16, 8)
    zlogvar = torch.randn(16, 8) * 0.1

    penalty, per_sample, _ = _module(0.0)._compute_kl(zmean, zlogvar)

    expected = -0.5 * torch.sum(
        1 + zlogvar - zmean.pow(2) - torch.exp(zlogvar.clamp(-30, 20)), axis=1
    )
    assert torch.allclose(per_sample, expected)
    assert torch.allclose(penalty, expected.mean())


def test_collapsed_latent_is_floored():
    """A fully collapsed posterior pays z_dim * free_bits instead of ~0."""
    zmean = torch.zeros(16, 8)
    zlogvar = torch.zeros(16, 8)  # q == prior, so true KL == 0

    penalty, per_sample, active = _module(0.5)._compute_kl(zmean, zlogvar)

    assert torch.allclose(per_sample, torch.zeros(16))  # logged KL stays truthful
    assert torch.isclose(penalty, torch.tensor(8 * 0.5))  # but the loss sees the floor
    assert active.item() == 0


def test_active_dims_counts_informative_dims():
    """Dims carrying real information are counted and not floored away."""
    zmean = torch.zeros(64, 8)
    zmean[:, :3] = torch.randn(64, 3) * 2.0  # 3 informative dims
    zlogvar = torch.zeros(64, 8)

    penalty, _, active = _module(0.5)._compute_kl(zmean, zlogvar)

    assert active.item() == 3
    # 3 informative dims (well above 0.5 nats) + 5 floored at 0.5
    assert penalty.item() > 5 * 0.5
