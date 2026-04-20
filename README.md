# HIPPIE: HD-MEA Integration Pipeline for Phenotypic Inference and Electrophysiology

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

**HIPPIE** is a deep learning framework for neuroscience autocuration, designed for classification and clustering of neurons based on high-density microelectrode array (HD-MEA) recordings. The framework implements Conditional Variational Autoencoders (CVAEs) that can operate on multiple modalities of electrophysiological data.

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
Installation time takes 2 to 3 minutes in a laptop. 
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
│   └── labels.csv          # Ground truth labels (n_samples × 1)
```

### Available Datasets

| Dataset | Description | Cell Types | Samples |
|---------|-------------|------------|---------|
| `allen_scope_neuropixel_area` | Allen Institute Neuropixel recordings | Brain regions | 82000+ |
| `cellexplorer_cell_type` | CellExplorer cortical interneurons | PV, SST, VIP, Pyramidal | 431 |
| `hausser_cell_type` | Häusser lab cerebellar recordings | PkC, GoC, MLI, MFB | ~4000 |
| `hull_cell_type` | Hull lab cerebellar recordings | PkC, GoC, MLI, MFB | 206 |
| `lissberger_labeled_cell_type` | Lisberger lab cerebellar data | PkC_ss, PkC_cs, GoC, MLI, MFB | 1152 |
| `mouse_organoids_cell_line` | Mouse organoid electrophysiology | Dorsal, Ventral | 4746 |
| `juxtacellular_mouse_s1_area` | Juxtacellular S1 recordings | Brain regions | 225 |

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

### Cross-Dataset Training and Prediction

The main workflow uses the cross-dataset training pipeline and it takes around 30 Minutes:

```bash
# Using the Python script
python cross_dataset_script.py \
  --training-dataset hausser_cell_type \
  --predict-dataset lissberger_labeled_cell_type \
  --config class_decoder_source_bn_aug_reg \
  --z-dim 20 \
  --beta 0.9 \
  --pretrain-max-epochs 100 \
  --supervised-max-epochs 50

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
--z-dim <int>                      # Latent space dimensionality (default: 10)
--beta <float>                     # β-VAE regularization (default: 1.0)

# Training
--pretrain-max-epochs <int>        # Pretraining epochs (default: 100)
--finetune-max-epochs <int>        # Fine-tuning epochs (default: 10)
--supervised-max-epochs <int>      # Supervised epochs (default: 50)
--batch-size <int>                 # Batch size (default: 512)
--learning-rate <float>            # Learning rate (default: 0.001)

# Data
--training-dataset <name>          # Dataset to train on (must have labels)
--predict-dataset <name>           # Dataset to predict on
--use-balanced-sampling            # Enable class-balanced sampling

# Experiment Tracking
--wandb-project <name>             # W&B project name (default: HIPPIE)
--wandb-tag <tag>                  # W&B run tag
```

## Data Augmentation

HIPPIE includes three augmentation strategies:

**Light Augmentations** (Conservative):
```python
augment_prob: 0.3              # 30% chance of applying
noise_std: 0.03                # Low noise level
amplitude_scale: (0.9, 1.1)    # ±10% amplitude variation
smoothing_sigma: (0.5, 1.5)    # Mild smoothing
```

**Heavy Augmentations** (Aggressive):
```python
augment_prob: 0.7              # 70% chance of applying
noise_std: 0.08                # Higher noise level
amplitude_scale: (0.7, 1.3)    # ±30% amplitude variation
smoothing_sigma: (0.5, 3.0)    # Stronger smoothing
```

**Augmentation Ablation** (Most Extreme):
```python
augment_prob: 0.8              # 80% chance of applying
noise_std: 0.1                 # Highest noise level
amplitude_scale: (0.6, 1.4)    # ±40% amplitude variation
smoothing_sigma: (0.3, 3.5)    # Strongest smoothing
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
- **`dataloading.py`**: Dataset classes (`EphysDatasetLabeled`, `MultiModalEphysDataset`)
- **`backbones.py`**: ResNet18 encoder/decoder architectures
- **`augmentations.py`**: Data augmentation transformations
- **`optimizers.py`**: Custom optimizers (AdamWScheduleFree)
- **`utils.py`**: Utility functions (embeddings, confusion matrices, plotting)

### Scripts

- **`cross_dataset_script.py`**: Main training script with all features
- **`cross_dataset_prediction.sh`**: Shell wrapper for quick experiments
- **`cross_dataset_training.ipynb`**: Interactive Jupyter notebook
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

**See [CONFIG_PROGRESSION.md](CONFIG_PROGRESSION.md) for detailed ablation study results.**

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

- **Jesus Gonzalez Ferrer**: jesusgzlezferrer@gmail.com
- **Julian Lehrer**: jlehrer@ucsc.edu
- **Project Homepage**: https://github.com/braingeneers/HIPPIE
- **Issues**: https://github.com/braingeneers/HIPPIE/issues

## Documentation

- [CLAUDE.md](CLAUDE.md): Detailed technical documentation for Claude Code
- [QUICK_CONFIG_REFERENCE.md](QUICK_CONFIG_REFERENCE.md): Configuration cheat sheet
- [CONFIG_PROGRESSION.md](CONFIG_PROGRESSION.md): Ablation study design and results
- [data_wrangling_scripts/README.md](data_wrangling_scripts/README.md): Data conversion utilities
