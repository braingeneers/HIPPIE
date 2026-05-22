# HIPPIE: High-dimensional Interpretation of Physiological Patterns In Intercellular Electrophysiology

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

**HIPPIE** is a trimodal deep learning framework for neuroscience autocuration, designed for classification and clustering of neurons based on extracellular electrophysiological recordings (e.g., HD-MEA and Neuropixel). The framework implements Conditional Variational Autoencoders (CVAEs) that operate jointly on three modalities of electrophysiological data: waveforms, ISI distributions, and autocorrelograms.

## Overview

HIPPIE addresses the challenge of automated neuron classification and clustering by leveraging multiple electrophysiological features simultaneously:
- **Waveforms**: Spike waveform morphology (50-100 time points)
- **ISI Distributions**: Interspike interval histograms (100 bins)
- **Autocorrelograms**: Temporal firing patterns (200 bins)

The framework uses a trimodal CVAE architecture with configurable ablation studies, data augmentation strategies, and transfer learning capabilities for cross-dataset prediction.

## Key Features

- **Multimodal Learning**: Simultaneously processes waveforms, ISI distributions, and autocorrelograms
- **Flexible Architecture**: 10 predefined configurations from baseline VAE to fully regularized models
- **Data Augmentation**: Light, heavy, and ablation modes with configurable noise, scaling, and smoothing
- **Transfer Learning**: Cross-dataset pretraining and fine-tuning capabilities
- **Regularization**: Class embedding dropout, reconstruction consistency loss, and warmup schedules
- **Evaluation**: K-NN and MLP classifier heads with balanced accuracy metrics
- **Experiment Tracking**: Integrated Weights & Biases logging
- **Docker Support**: Containerized deployment for reproducibility
- **Kubernetes Ready**: Job deployment scripts for cluster computing

## Installation

### Prerequisites

- Python 3.9 or higher
- CUDA-compatible GPU (optional, but recommended)
- Docker (optional, for containerized deployment)

### System Requirements                                                                                                
                                                                                                                        
**Tested Operating Systems:**                                                                                          
- macOS 14.x (Sonoma)                                                                                                  
- Ubuntu 22.04 LTS                                                                                                     
                                                                                                                        
**Tested Dependency Versions:**                                                                                        
- Python 3.9.x, 3.10.x, 3.11.x                                                                                         
- PyTorch 2.1.0                                                                                                        
- pytorch-lightning 2.1.0                                                                                              
- CUDA 11.8 / 12.1 (for GPU support)  

### Local Installation
Installation takes 2 to 3 minutes on a typical laptop. 
```bash
# Clone the repository
git clone https://github.com/braingeneers/HIPPIE.git
cd HIPPIE

# Create virtual environment
python -m venv hippie_venv
source hippie_venv/bin/activate  # On Windows: hippie_venv\Scripts\activate

# Install package
pip install -e .

```

### Docker Installation

```bash
# Build Docker image
make build

# Run container
make run

# Push to Docker Hub (requires login)
make push
```


## Dataset Structure

All datasets are stored in `datasets_hippie/` with the following standardized structure:

```
datasets_hippie/
├── <dataset_name>/
│   ├── waveforms.csv       # Spike waveform data (n_samples × n_timepoints)
│   ├── isi_dist.csv        # ISI distributions (n_samples × 100 bins)
│   ├── acg.csv             # Autocorrelograms (n_samples × 200 bins)
│   └── labels.csv          # Ground truth labels (n_samples × 1; may also be named celltypes.csv)
```

### Available Datasets

| Dataset | Description | Cell Types | Samples |
|---------|-------------|------------|---------|
| `allen_scope_neuropixel_area` | Allen Institute Neuropixel recordings | Brain regions | 82000+ |
| `cellexplorer_cell_type` | CellExplorer cortical interneurons | PV, SST, VIP, Pyramidal | 431 |
| `hausser_cell_type` | Häusser lab cerebellar recordings | GoC, GrC, MFB, MLI, PkC_ss, PkC_cs | 113 |
| `hull_cell_type` | Hull lab cerebellar recordings | PkC, GoC, MLI, MFB | 103 |
| `lissberger_labeled_cell_type` | Lisberger lab cerebellar data | PkC_ss, PkC_cs, GoC, MLI, MFB | 1152 |
| `mouse_organoids_cell_line` | Mouse organoid electrophysiology | Dorsal, Ventral | 4746 |
| `juxtacellular_mouse_s1_area` | Juxtacellular S1 recordings | Brain regions | 224 |

## Model Configurations

HIPPIE provides 10 predefined configurations for systematic ablation studies:

| Configuration | Source Emb | Class Emb | Fusion | Batch Norm | Augmentation | Regularization |
|---------------|------------|-----------|--------|------------|--------------|----------------|
| `baseline` | ❌ | ❌ | ❌ | ❌ | None | ❌ |
| `with_source` | ✅ | ❌ | ✅ | ❌ | None | ❌ |
| `with_class` | ❌ | ✅ | ✅ | ❌ | None | ❌ |
| `with_both_embeddings` | ✅ | ✅ | ✅ | ❌ | None | ❌ |
| `with_light_augmentations` | ❌ | ❌ | ❌ | ❌ | Light | ❌ |
| `with_heavy_augmentations` | ✅ | ✅ | ❌ | ❌ | Heavy | ❌ |
| `with_batch_norm` | ✅ | ✅ | ✅ | ✅ | Light | ❌ |
| `no_fusion` | ✅ | ✅ | ❌ | ❌ | None | ❌ |
| `no_augmentations` | ✅ | ✅ | ✅ | ✅ | None | ❌ |
| `full_architecture` | ✅ | ✅ | ✅ | ✅ | Light | ✅ |
| `class_decoder_source_bn_aug_reg` | ✅ | decoder-only | ✅ | ✅ | Light | ✅ |

**See [QUICK_CONFIG_REFERENCE.md](QUICK_CONFIG_REFERENCE.md) for detailed configuration parameters.**

## Usage

### Quick start: extract embeddings with the pretrained checkpoint

For inference-only use, the canonical example downloads the pretrained
checkpoint from the HuggingFace Hub and writes 30-D latent embeddings to a
single `.npz`:

```bash
python examples/extract_embeddings.py \
  --datasets-root ./datasets_hippie \
  --output ./embeddings.npz
```

Pass `--checkpoint ./hippie_techcond_v1.ckpt` to use a local checkpoint
instead, or `--datasets <name> <name> ...` to embed a subset. See
[`examples/extract_embeddings.py`](examples/extract_embeddings.py) for the
expected CSV layout (`waveforms.csv`, `isi_dist.csv`, `acg.csv`,
`labels.csv` per dataset folder).

### Cross-Dataset Training and Prediction

The main workflow uses the cross-dataset training pipeline and takes around 30 minutes on a single GPU:

```bash
# Using the Python script
python cross_dataset_script.py \
  --training-dataset hausser_cell_type \
  --predict-dataset lissberger_labeled_cell_type \
  --config class_decoder_source_bn_aug_reg \
  --z_dim 30 \
  --beta 1.0 \
  --pretrain-max-epochs 100 \
  --supervised-max-epochs 5

# Using the shell script (simplified)
bash cross_dataset_prediction_with_wandb.sh
```

### Jupyter Notebook Interface

For interactive experimentation, use the Jupyter notebook:

```bash
jupyter notebook cross_dataset_training.ipynb
```

The notebook provides a step-by-step walkthrough of:
1. **Pretraining**: Unsupervised learning on multiple datasets
2. **Fine-tuning**: Adaptation to target dataset without labels
3. **Supervised Training**: Training with labels using balanced sampling
4. **Evaluation**: K-NN and MLP classifier evaluation with confusion matrices

**Expected Output:**                                                                                                   
- Training logs printed to console                                                                                     
- Weights & Biases dashboard with loss curves and confusion matrices                                                   
- Final accuracy metrics and csv outputs with embeddings and cluster labels.

### Key Parameters

```bash
# Model Configuration
--config <name>                    # Model configuration (see table above)
--z_dim <int>                      # Latent space dimensionality (default: 30; 20 bimodal, 10 unimodal)
--beta <float>                     # β-VAE regularization (default: 1.0)

# Training
--pretrain-max-epochs <int>        # Pretraining epochs (default: 100)
--finetune-max-epochs <int>        # Fine-tuning epochs (default: 10)
--supervised-max-epochs <int>      # Supervised epochs (default: 5)
--batch-size <int>                 # Batch size (default: 128)
--learning-rate <float>            # Learning rate (default: 0.001)
--early-stopping-patience <int>    # Early stopping patience (default: 5)

# Data
--training-dataset <name>          # Dataset to train on (must have labels)
--predict-dataset <name>           # Dataset to predict on
--use_balanced_sampling            # Enable class-balanced sampling

# Experiment Tracking
--project <name>                   # W&B project name (default: HIPPIE)
--wandb-tag <tag>                  # W&B run tag (default: Hippie_cross_dataset)
```

## Data Augmentation

HIPPIE includes two augmentation strategies.

**Light Augmentations** (as reported in the paper):
```python
augment_prob: 0.3              # 30% chance of applying
noise_std: 0.03                # Additive Gaussian noise σ
amplitude_scale: (0.9, 1.1)    # ±10% amplitude variation
smoothing_sigma: (0.5, 1.5)    # Gaussian smoothing σ range
time_warp_strength: 0.05       # Non-linear time warping
baseline_shift: (-0.05, 0.05)  # Additive DC offset
```

**Heavy Augmentations**: the paper specifies the higher application
probability (`augment_prob = 0.7`); the remaining numeric values below
are code-side defaults for the `with_heavy_augmentations` config:
```python
augment_prob: 0.7              # 70% chance of applying (from paper)
noise_std: 0.08                # code-only
amplitude_scale: (0.7, 1.3)    # code-only
smoothing_sigma: (0.5, 3.0)    # code-only
```

## Regularization Techniques

To prevent data leakage and improve generalization:

1. **Class Embedding Dropout** (30%): Forces model to learn robust representations
2. **Reconstruction Consistency Loss**: Ensures consistent outputs with/without class labels
3. **Embedding Warmup Schedule**: Gradually increases regularization over first 5 epochs

**See [CLAUDE.md](CLAUDE.md) for detailed explanations of the data leakage fix and regularization strategies.**

## Module Reference

### Core Modules (`hippie/`)

- **`multimodal_model.py`**: MultiModal CVAE with configurable ablation studies
- **`unimodal_model.py`**: Single-modality CVAE implementation
- **`vae.py`**: Unconditioned VAE for unsupervised data compression
- **`dataloading.py`**: Dataset classes (`EphysDatasetLabeled`, `MultiModalEphysDataset`)
- **`backbones.py`**: ResNet18 encoder/decoder architectures
- **`augmentations.py`**: Data augmentation transformations
- **`optimizers.py`**: Custom optimizers (AdamWScheduleFree)
- **`utils.py`**: Utility functions (embeddings, confusion matrices, plotting)
- **`checkpoint.py`**: Checkpoint loading/saving helpers
- **`inference.py`**: Inference utilities for trained models
- **`compute_parity.py`**: Compute-parity timing/memory logger

### Scripts

- **`cross_dataset_script.py`**: Main training script with all features
- **`cross_dataset_prediction.sh`**: Shell wrapper for quick experiments
- **`cross_dataset_prediction_with_wandb.sh`**: Shell wrapper with Weights & Biases logging enabled
- **`cross_dataset_training.ipynb`**: Interactive Jupyter notebook
- **`hippie_nwb_classify.py`**: Classify neurons directly from NWB recordings
- **`Makefile`**: Docker build and deployment commands

### Data Wrangling (`data_wrangling_scripts/`)

- **`allen_nwb_to_csv_converter.ipynb`**: Convert Allen Institute NWB files to CSV
- **`acqm_to_csv_converter.ipynb`**: Convert proprietary formats to CSV
- **`neurocurator.py`**: Manual curation interface
- **`download_sessions_to_json.py`**: Batch data download utilities

## Architecture Details

### MultiModal CVAE

```
Input Modalities (Wave, ISI, ACG)
    ↓
Separate Encoders (ResNet18-based)
    ↓
[Optional] Fusion Encoder
    ↓
Latent Space (z_dim)
    ↓
[Optional] Class/Source Embeddings
    ↓
Separate Decoders (ResNet18-based)
    ↓
Reconstructions + KL Divergence Loss
```

**Loss Function:**
```
L = Σ(λ_m × MSE(x_m, x̂_m)) + β × KL(q(z|x) || p(z))
    + λ_c × ConsistencyLoss(x̂_with_class, x̂_without_class)
```

### Evaluation Pipeline

```
Trained CVAE
    ↓
Extract Embeddings (encoder only, no class labels)
    ↓
Train K-NN Classifier (k selected via cross-validation)
    ↓
Train MLP Classifier (3-layer with BatchNorm)
    ↓
Compute Balanced Accuracy & Confusion Matrices
```

## Experiment Tracking

HIPPIE uses [Weights & Biases](https://wandb.ai) for experiment tracking:

```bash
# Set API key
export WANDB_API_KEY=<your_key>

# Runs are automatically logged with:
# - Training/validation losses
# - Resource usage (GPU/CPU memory)
# - Confusion matrices
# - Embeddings (optional)
# - Model checkpoints (optional)
```

**Logged Metrics:**
- `train_loss`, `val_loss`: Reconstruction + KL loss
- `train_consistency_loss`: Consistency regularization
- `mlp_train_acc`, `mlp_val_acc`: MLP classifier accuracy
- `mlp_holdout_accuracy`: Final test accuracy
- `cross_dataset_balanced_accuracy`: Cross-dataset performance
- `resources/*`: GPU/CPU memory, step time

## Results

### Typical Performance (Balanced Accuracy)

| Dataset | Task Difficulty | Baseline | Full Model | Aug Ablation |
|---------|----------------|----------|------------|--------------|
| `lissberger_labeled_cell_type` | Easy | 60-65% | 73-78% | 72-77% |
| `cellexplorer_cell_type` | Hard | 40-45% | 45-50% | 60-65% |
| `hausser_cell_type` | Medium | 55-60% | 65-70% | 70-75% |

**Key Findings:**
- Augmentation strategies outperform conditional models on hard datasets with imbalanced/overlapping classes
- Conditional models (with embeddings) excel on easy datasets with well-separated classes
- Regularization is critical for preventing over-reliance on class labels during training

**See [QUICK_CONFIG_REFERENCE.md](QUICK_CONFIG_REFERENCE.md) for detailed configuration parameters and ablation study results.**

## Development

### Running Tests

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Code formatting
black hippie/
isort hippie/

# Type checking
mypy hippie/
```

### Docker Development

```bash
# Build and test locally
make build
make run

# Push to registry
make go  # Builds, tags, and pushes in one command
```

## Citation

If you use HIPPIE in your research, please cite:

```bibtex
@article{gonzalez2025hippie,
  title={HIPPIE: A Multimodal Deep Learning Model for Electrophysiological Classification of Neurons},
  author={Gonzalez-Ferrer, Jesus and Lehrer, Julian and Schweiger, Hunter E and Geng, Jinghui and Hernandez, Sebastian and Reyes, Francisco and Sevetson, Jess L and Salama, Sofie R and Teodorescu, Mircea and Haussler, David and others},
  journal={bioRxiv},
  year={2025}
}
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the BSD 3-Clause License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Braingeneers Lab** at UC Santa Cruz for project support
- **Allen Institute for Brain Science** for open-access Neuropixel datasets
- **CellExplorer** team for cortical interneuron data
- **Häusser, Hull, and Lisberger labs** for cerebellar recordings
- **PyTorch Lightning** and **Weights & Biases** teams for excellent frameworks

## Contact

- **Jesus Gonzalez Ferrer**: jgonz373@ucsc.edu
- **Project Homepage**: https://github.com/braingeneers/HIPPIE
- **Issues**: https://github.com/braingeneers/HIPPIE/issues

## Documentation

- [CLAUDE.md](CLAUDE.md): Detailed technical documentation for Claude Code
- [QUICK_CONFIG_REFERENCE.md](QUICK_CONFIG_REFERENCE.md): Configuration cheat sheet and ablation study results
- [data_wrangling_scripts/README.md](data_wrangling_scripts/README.md): Data conversion utilities
