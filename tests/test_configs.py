#!/usr/bin/env python3
"""Verify that each ExperimentConfigs preset matches its documented ablation schema."""

import pytest

from hippie.multimodal_model import ExperimentConfigs


# Expected feature values for each named configuration (the ablation ladder).
EXPECTED_CONFIGS = {
    "baseline": {
        "use_source_embedding": False,
        "use_class_embedding": False,
        "use_fusion_encoder": False,
        "use_batch_norm": False,
        "use_augmentations": False,
        "class_embedding_dropout": 0.0,
        "reconstruction_consistency_weight": 0.0,
    },
    "with_source": {
        "use_source_embedding": True,
        "use_class_embedding": False,
        "use_fusion_encoder": True,
        "use_batch_norm": False,
        "use_augmentations": False,
        "class_embedding_dropout": 0.0,
        "reconstruction_consistency_weight": 0.0,
    },
    "with_class": {
        "use_source_embedding": False,
        "use_class_embedding": True,
        "use_fusion_encoder": True,
        "use_batch_norm": False,
        "use_augmentations": False,
        "class_embedding_dropout": 0.0,
        "reconstruction_consistency_weight": 0.0,
    },
    "with_both_embeddings": {
        "use_source_embedding": True,
        "use_class_embedding": True,
        "use_fusion_encoder": True,
        "use_batch_norm": False,
        "use_augmentations": False,
        "class_embedding_dropout": 0.0,
        "reconstruction_consistency_weight": 0.0,
    },
    "with_light_augmentations": {
        "use_source_embedding": False,
        "use_class_embedding": False,
        "use_fusion_encoder": False,
        "use_batch_norm": False,
        "use_augmentations": True,
        "augment_prob": 0.3,
        "noise_std": 0.03,
        "class_embedding_dropout": 0.0,
        "reconstruction_consistency_weight": 0.0,
    },
    "with_heavy_augmentations": {
        "use_source_embedding": True,
        "use_class_embedding": True,
        "use_fusion_encoder": False,
        "use_batch_norm": False,
        "use_augmentations": True,
        "augment_prob": 0.7,
        "noise_std": 0.08,
        "class_embedding_dropout": 0.0,
        "reconstruction_consistency_weight": 0.0,
    },
    "with_batch_norm": {
        "use_source_embedding": True,
        "use_class_embedding": True,
        "use_fusion_encoder": True,
        "use_batch_norm": True,
        "use_augmentations": True,
        "augment_prob": 0.3,
        "noise_std": 0.03,
        "class_embedding_dropout": 0.0,
        "reconstruction_consistency_weight": 0.0,
    },
    "no_fusion": {
        "use_source_embedding": True,
        "use_class_embedding": True,
        "use_fusion_encoder": False,
        "use_batch_norm": False,
        "use_augmentations": False,
        "class_embedding_dropout": 0.0,
        "reconstruction_consistency_weight": 0.0,
    },
    "full_architecture": {
        "use_source_embedding": True,
        "use_class_embedding": True,
        "use_fusion_encoder": True,
        "use_batch_norm": True,
        "use_augmentations": True,
        "augment_prob": 0.3,
        "noise_std": 0.03,
        "class_embedding_dropout": 0.3,
        "reconstruction_consistency_weight": 0.15,
        "embedding_warmup_epochs": 5,
    },
    "class_decoder_source_bn_aug_reg": {
        "use_source_embedding": True,
        "use_class_embedding": True,
        "encoder_uses_class_embedding": False,
        "use_fusion_encoder": True,
        "use_batch_norm": True,
        "use_augmentations": True,
        "augment_prob": 0.3,
        "noise_std": 0.03,
        "class_embedding_dropout": 0.3,
        "reconstruction_consistency_weight": 0.15,
        "embedding_warmup_epochs": 5,
    },
}


@pytest.mark.parametrize("config_name", list(EXPECTED_CONFIGS))
def test_config_matches_schema(config_name):
    """Each preset's fields match the documented ablation schema."""
    config = getattr(ExperimentConfigs, config_name)()
    for feature, expected in EXPECTED_CONFIGS[config_name].items():
        actual = getattr(config, feature)
        assert actual == expected, (
            f"{config_name}.{feature}: got {actual!r}, expected {expected!r}"
        )


@pytest.mark.parametrize("config_name", list(EXPECTED_CONFIGS))
def test_preset_exists_and_is_callable(config_name):
    """Every config referenced in the schema is a real ExperimentConfigs factory."""
    assert hasattr(ExperimentConfigs, config_name), f"missing config: {config_name}"
    assert callable(getattr(ExperimentConfigs, config_name))
