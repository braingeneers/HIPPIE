#!/usr/bin/env python3
"""
Test script to verify all configurations match the expected schema.
"""

from hippie.multimodal_model import ExperimentConfigs


def _check_config(name, expected_features):
    """Test a single config against expected features."""
    config_func = getattr(ExperimentConfigs, name)
    config = config_func()

    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")

    errors = []

    for feature, expected_value in expected_features.items():
        actual_value = getattr(config, feature)
        match = "OK" if actual_value == expected_value else "FAIL"
        print(f"  {match} {feature}: {actual_value} (expected: {expected_value})")

        if actual_value != expected_value:
            errors.append(f"{feature}: got {actual_value}, expected {expected_value}")

    if errors:
        print(f"\nFAILED: {len(errors)} errors")
        for error in errors:
            print(f"    - {error}")
        return False
    else:
        print(f"\nPASSED")
        return True


def main():
    configs_to_test = {
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

    print("="*60)
    print("Testing All Configuration Schemas")
    print("="*60)

    results = {}
    for config_name, expected in configs_to_test.items():
        results[config_name] = _check_config(config_name, expected)

    print("\n" + "="*60)
    print("Summary")
    print("="*60)

    passed = sum(results.values())
    total = len(results)

    for name, passed_test in results.items():
        status = "PASS" if passed_test else "FAIL"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total} passed")

    if passed == total:
        print("\nAll configs match the expected schema!")
        return 0
    else:
        print(f"\n{total - passed} config(s) failed validation")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
