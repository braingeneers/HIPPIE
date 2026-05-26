"""Checkpoint loading for HIPPIE models."""

from __future__ import annotations

import torch


def infer_model_dims(state_dict: dict) -> tuple[int, int]:
    """Infer num_sources and num_classes from a checkpoint state dict."""
    src_key = next(k for k in state_dict if "source_embed" in k and "weight" in k)
    num_sources = state_dict[src_key].shape[0]
    cls_keys = [k for k in state_dict if "class_embed" in k and "weight" in k]
    num_classes = state_dict[cls_keys[0]].shape[0] if cls_keys else 2
    return num_sources, num_classes


def build_unconditioned_model(ckpt_path: str, backbone_base_width: int = 64):
    """Load a MultiModalCVAE trained with ExperimentConfigs.unconditioned().

    z_dim is inferred from the z_mean layer shape in the state dict so no
    extra config file is needed alongside the checkpoint.
    """
    from .multimodal_model import MultiModalCVAE, ExperimentConfigs

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    sd = {
        k.replace("model.", "", 1) if k.startswith("model.") else k: v
        for k, v in sd.items()
    }

    z_dim = sd["z_mean.weight"].shape[0]

    model = MultiModalCVAE(
        modalities={"wave": 50, "isi": 100, "acg": 100},
        z_dim=z_dim,
        num_sources=None,
        num_classes=None,
        num_super_regions=0,
        num_layers=0,
        config=ExperimentConfigs.unconditioned(),
        backbone_base_width=backbone_base_width,
    )
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def build_model(ckpt_path: str, strict: bool = True):
    """Load a MultiModalCVAE from a Lightning checkpoint file.

    Strips the Lightning "model." prefix from the state dict and infers
    ``z_dim``, ``num_sources``, and ``num_classes`` from the saved tensor
    shapes so the same loader works for both the z=30 discriminative
    checkpoint and the z=16 generative (Figure 6) checkpoint. Returns the
    model in eval mode on CPU.

    Args:
        ckpt_path: Path to a Lightning .ckpt file.
        strict: If True (default), require an exact shape/key match between
            the checkpoint and the constructed model — surfaces silent
            mismatches that would otherwise produce garbage embeddings.
            Pass strict=False only when intentionally loading a partial
            checkpoint (e.g. fine-tuning from a different architecture).
    """
    from .multimodal_model import MultiModalCVAE, ExperimentConfigs

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    sd = {
        k.replace("model.", "", 1) if k.startswith("model.") else k: v
        for k, v in sd.items()
    }

    # Infer latent dimensionality from the saved z_mean projection so a
    # mismatched-z_dim checkpoint cannot load silently. Mirrors the inference
    # done by build_unconditioned_model.
    z_dim = sd["z_mean.weight"].shape[0]
    num_sources, num_classes = infer_model_dims(sd)

    model = MultiModalCVAE(
        modalities={"wave": 50, "isi": 100, "acg": 100},
        z_dim=z_dim,
        num_sources=num_sources,
        num_classes=num_classes,
        num_super_regions=0,
        num_layers=0,
        config=ExperimentConfigs.class_decoder_source_bn_aug_reg(),
        backbone_base_width=64,
    )
    model.load_state_dict(sd, strict=strict)
    model.eval()
    return model
