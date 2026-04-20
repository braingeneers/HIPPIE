# Quick Configuration Reference

## 🎯 At a Glance

### The Progression (Simplest → Most Complex)

```
1.  baseline                          → Pure VAE
2.  with_source                       → + Dataset ID
3.  with_class                        → + Cell type
4.  with_both_embeddings              → + Both embeddings
5.  with_light_augmentations          → Baseline + Light data augmentation
6.  with_heavy_augmentations          → Embeddings + Heavy augmentation
7.  with_batch_norm                   → + Batch normalization + Light aug
8.  no_fusion                         → Variant: embeddings without fusion
9.  no_augmentations                  → Full arch, no aug/reg
10. full_architecture                  → Full arch + Light aug + regularization
11. conditional_decoder_only          → Decoder-only class conditioning + light aug
12. class_decoder_source              → Decoder-only class, no BN/aug/reg
13. class_decoder_source_bn           → + Batch normalization
14. class_decoder_source_bn_strong_aug → + Strong augmentations
15. class_decoder_source_bn_aug_reg   → 🏆 PRODUCTION DEFAULT: decoder-only + BN + light aug + reg
```

## 📊 What Each Config Has

| # | Config | Components | Purpose |
|---|--------|------------|---------|
| 1 | `baseline` | Nothing | Pure VAE baseline |
| 2 | `with_source` | Source embedding + fusion | Add dataset context |
| 3 | `with_class` | Class embedding + fusion | Add cell type conditioning |
| 4 | `with_both_embeddings` | Both embeddings + fusion | Combine source + class |
| 5 | `with_light_augmentations` | Light augmentation only | Test augmentation alone |
| 6 | `with_heavy_augmentations` | Embeddings + heavy aug | Aggressive augmentation |
| 7 | `with_batch_norm` | Full arch + BatchNorm + light aug | Add normalization |
| 8 | `no_fusion` | Embeddings, no fusion | Test fusion importance |
| 9 | `no_augmentations` | Full arch, no aug/reg | Full architecture |
| 10 | `full_architecture` | Full arch + light aug + reg | Kitchen sink (encoder sees class) |
| 11 | `conditional_decoder_only` | Decoder-only class + light aug | Asymmetric CVAE variant |
| 12 | `class_decoder_source` | Decoder-only class, no BN/aug | Ladder rung 4 |
| 13 | `class_decoder_source_bn` | + Batch normalization | Ladder rung 5 |
| 14 | `class_decoder_source_bn_strong_aug` | + Strong augmentations | Ladder rung 7 |
| 15 | `class_decoder_source_bn_aug_reg` | + Light aug + regularization | 🏆 Production default (rung 8) |

## 🔧 Component Legend

| Symbol | Component | Description |
|--------|-----------|-------------|
| 🔵 | Source Embedding | Dataset ID (e.g., Allen Institute = 7) |
| 🟢 | Class Embedding | Cell type (e.g., PV, SST, VIP) |
| 🟡 | Fusion Encoder | MLP that combines features + embeddings |
| 🟣 | Batch Norm | Normalization layers |
| 🔶 | Augmentation | Data augmentation (light/heavy) |
| 🔴 | Regularization | Dropout + consistency loss |

## 🎨 Visual Representation

```
baseline:                [VAE]

with_source:             [VAE] + 🔵 + 🟡

with_class:              [VAE] + 🟢 + 🟡

with_both_embeddings:    [VAE] + 🔵 + 🟢 + 🟡

with_light_augmentations: [VAE] + 🔶(light)

with_heavy_augmentations: [VAE] + 🔵 + 🟢 + 🔶(heavy)

with_batch_norm:         [VAE] + 🔵 + 🟢 + 🟡 + 🟣 + 🔶(light)

no_fusion:               [VAE] + 🔵 + 🟢

no_augmentations:        [VAE] + 🔵 + 🟢 + 🟡 + 🟣

full_architecture:       [VAE] + 🔵 + 🟢 + 🟡 + 🟣 + 🔶(light) + 🔴

class_decoder_source_bn_aug_reg: [VAE] + 🔵 + 🟢(decoder) + 🟡 + 🟣 + 🔶(light) + 🔴
```

## 🎯 When to Use Which Config

### For Easy Datasets (naturally separable):
✅ **Use**: `no_augmentations` or `class_decoder_source_bn_aug_reg`
❌ **Avoid**: Over-complicating with augmentation

### For Hard Datasets (overlapping/imbalanced):
✅ **Use**: `with_light_augmentations` or `class_decoder_source_bn_aug_reg`
❌ **Avoid**: `with_class`, `with_both_embeddings`, `no_fusion` without regularization

### For Ablation Studies:
Run the full ladder to show progression

### For Production:
Use `class_decoder_source_bn_aug_reg` (post-rebuttal production default, +0.17 over prior default)

## 📈 Expected Performance Order

### Easy Dataset (lissberger):
```
with_both_embeddings ≈ no_augmentations ≈ class_decoder_source_bn_aug_reg > with_batch_norm > baseline
```

### Hard Dataset (cellexplorer):
```
with_light_augmentations > class_decoder_source_bn_aug_reg >> baseline > ... > with_both_embeddings > no_fusion
```

## 🚨 Known Issues

### Conditional Models Fail on Hard Datasets Without Regularization:
- ❌ `with_class`: ~42% (cellexplorer)
- ❌ `with_both_embeddings`: ~30%
- ❌ `no_fusion`: ~15%

**Solution**: Use `class_decoder_source_bn_aug_reg` — the encoder never sees the class label (no train/test mismatch), and decoder-side regularization further stabilizes training.

### Why This Happens:
1. Model trains WITH class labels
2. Model tested WITHOUT class labels (masked to prevent leakage)
3. Train/test mismatch → performance collapse

**The asymmetric CVAE fix**: `encoder_uses_class_embedding=False` — the encoder is class-agnostic by construction, so there is no drift at test time regardless of masking.

## 🎓 Key Takeaways

1. **Progression matters**: Start simple (baseline), add complexity gradually
2. **Dataset difficulty matters**: Easy vs hard datasets need different approaches
3. **Decoder-only conditioning is superior**: Moving class embedding to decoder-only eliminates train/test mismatch
4. **Augmentation is robust**: Works across dataset difficulties
5. **Production default is `class_decoder_source_bn_aug_reg`**: +0.17 mean accuracy over prior default
