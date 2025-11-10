# Quick Configuration Reference

## 🎯 At a Glance

### The Progression (Simplest → Most Complex)

```
1. baseline              → Pure VAE
2. with_source          → + Dataset ID
3. with_class           → + Cell type
4. with_both_embeddings → + Both embeddings
5. with_light_augmentations → Baseline + Light data augmentation
6. with_heavy_augmentations → Embeddings + Heavy augmentation
7. with_batch_norm      → + Batch normalization + Light aug
8. no_fusion            → Variant: embeddings without fusion
9. full_model           → Everything EXCEPT regularization
10. augmentation_ablation → 🏆 MOST COMPLEX: Everything + regularization
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
| 9 | `full_model` | Everything except reg | Full architecture |
| 10 | `augmentation_ablation` | Everything | Kitchen sink |

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

full_model:              [VAE] + 🔵 + 🟢 + 🟡 + 🟣

augmentation_ablation:   [VAE] + 🔵 + 🟢 + 🟡 + 🟣 + 🔶(light) + 🔴
```

## 🎯 When to Use Which Config

### For Easy Datasets (naturally separable):
✅ **Use**: `full_model` or `augmentation_ablation`
❌ **Avoid**: Over-complicating with augmentation

### For Hard Datasets (overlapping/imbalanced):
✅ **Use**: `with_light_augmentations` or `augmentation_ablation`
❌ **Avoid**: `with_class`, `with_both_embeddings`, `no_fusion` without regularization

### For Ablation Studies:
Run all 10 configs to show progression

### For Production:
Use `augmentation_ablation` (most robust)

## 📈 Expected Performance Order

### Easy Dataset (lissberger):
```
with_both_embeddings ≈ full_model ≈ augmentation_ablation > with_batch_norm > baseline
```

### Hard Dataset (cellexplorer):
```
with_light_augmentations > augmentation_ablation >> baseline > ... > with_both_embeddings > no_fusion
```

## 🚨 Known Issues

### Conditional Models Fail on Hard Datasets Without Regularization:
- ❌ `with_class`: ~42% (cellexplorer)
- ❌ `with_both_embeddings`: ~30%
- ❌ `no_fusion`: ~15%

**Solution**: Use `augmentation_ablation` which adds regularization

### Why This Happens:
1. Model trains WITH class labels
2. Model tested WITHOUT class labels (masked to prevent leakage)
3. Train/test mismatch → performance collapse

**Regularization fixes this** by:
- Dropout (30%) on class embeddings during training
- Consistency loss (0.15 weight) ensures similar outputs with/without labels
- Warmup schedule (5 epochs) for gradual enforcement

## 🎓 Key Takeaways

1. **Progression matters**: Start simple (baseline), add complexity gradually
2. **Dataset difficulty matters**: Easy vs hard datasets need different approaches
3. **Regularization is critical**: For conditional models on hard datasets
4. **Augmentation is robust**: Works across dataset difficulties
5. **Full model is not always best**: Sometimes simpler models win

## 📝 Quick Commands

```bash
# Run single config
python train_multimodal_transductive.py --config augmentation_ablation --dataset cellexplorer_cell_type

# Run all configs (ablation)
./run_transductive_jobs.sh

# Plot results
jupyter notebook analysis/plotting_single_dataset_ablation.ipynb
```

## 🔗 Related Files

- Config definitions: `hippie/multimodal_model.py`
- Training scripts: `train_multimodal_transductive.py`, `train_multimodal_holdout.py`
- Job submission: `run_transductive_jobs.sh`, `run_holdout_jobs.sh`
- Plotting: `analysis/plotting_single_dataset_ablation.ipynb`
- Detailed docs: `CONFIG_PROGRESSION.md`
