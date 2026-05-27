# HIPPIE: High-dimensional Interpretation of Physiological Patterns In Intercellular Electrophysiology

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

**HIPPIE** is a generative model for electrophysiological analysis across species, technologies, and modalities. It implements parallel Conditional Variational Autoencoders (CVAEs) that jointly embed three modalities — waveforms, ISI distributions, and autocorrelograms — into a shared 30-D latent space. The same model supports neuron classification, unsupervised clustering, cross-species/cross-technology transfer, and generative use cases (counterfactual decoding, cross-modal imputation).

The framework has been validated on recordings from Neuropixels (1.0/2.0), silicon probes (incl. NeuroNexus 32-channel), and juxtacellular micropipettes, across mouse, rat, and macaque, spanning cerebellum and neocortex.

## Overview

HIPPIE addresses the challenge of automated neuron classification and clustering by leveraging multiple electrophysiological features simultaneously:
- **Waveforms**: Spike waveform morphology — trough-centered, resampled to 50 time points
- **ISI distributions**: Interspike interval histograms — 1 ms bins, 0–100 ms
- **Autocorrelograms**: Spike-pair counts at lags −100…+100 ms (201 bins, 1 ms wide, center bin zeroed)

The framework uses a trimodal CVAE architecture with configurable ablation studies, data augmentation strategies, and transfer learning capabilities for cross-dataset prediction.

## Key Features

- **Multimodal Learning**: Simultaneously processes waveforms, ISI distributions, and autocorrelograms
- **Flexible Architecture**: Predefined configurations from a pure VAE baseline to the fully regularized production default
- **Data Augmentation**: Light, heavy, and ablation modes with configurable noise, scaling, and smoothing
- **Transfer Learning**: Cross-dataset pretraining and fine-tuning capabilities
- **Regularization**: Class-embedding dropout, reconstruction consistency loss, and warmup schedules
- **Pretrained Checkpoint**: Downloaded automatically from the HuggingFace Hub (repo `Jesusgf23/hippie`, file `hippie_techcond_v1.ckpt`)
- **CLI**: `hippie-cli validate-data | embed | predict` for the common inference flows
- **NWB-native classification**: `hippie_nwb_classify.py` runs straight from NWB sessions (precomputed units or raw traces via SpikeInterface)
- **Docker Support**: Containerized deployment for reproducibility

> Paper-reproducing benchmarking (K-fold, KNN/MLP accuracy heads, balanced-accuracy reporting, compute-parity timing, figure generation) lives in the companion repository [`hippie_benchmarking_release`](https://github.com/JesusGF1/hippie_benchmarking_release), not here.

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

# Install package (core deps only -- inference, training, embedding)
pip install -e .

# Optional extras for specific data wrangling paths:
pip install -e ".[ibl]"     # ONE-api + iblatlas for IBL Brain Wide Map
pip install -e ".[dev]"     # pytest, black, isort, mypy for development
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


## Data format

HIPPIE consumes one folder per dataset under `datasets_hippie/<name>/`
with up to four CSVs. Only `waveforms.csv` and `isi_dist.csv` are
required; the rest are optional. Headers are not parsed — the first row
is treated as a header and discarded, so column names are free.

| File | Shape | Required? | Units / encoding |
|---|---|---|---|
| `waveforms.csv` | (N, T) — any T; trough-centered window, 20 pre / 30 post in the canonical layout (T = 50) | yes | raw amplitude |
| `isi_dist.csv`  | (N, ~100) — bin width 1 ms, range 0–100 ms | yes | spike counts (non-negative) |
| `acg.csv`       | (N, 201) — bin width 1 ms, range −100…+100 ms, center bin = 0 | no (falls back to bimodal mode with zero ACG) | spike-pair counts |
| `labels.csv`    | (N, ≥1) — cell-type label in the **last** column; also accepts `celltypes.csv` | no (required for supervised training / KNN heads, not for embedding extraction) | string |

Optional per-dataset extras (`metadata.csv`, `area.csv`, `super_regions.csv`) are recognized by some training paths but ignored by the embedding API.

Normalization is applied internally — do **not** pre-normalize your inputs:
- waveform: min-max to [−1, 1] per row
- ISI: `log(x + 1)`, then min-max to [−1, 1]
- ACG: min-max to [−1, 1]

Each modality is also resampled to a fixed internal length at load time
(waveform → 50, ISI → 100, ACG → 100 via linear interpolation), so the exact
input bin count is flexible — the canonical `(N, 201)` ACG and `(N, 100)` ISI
above are resampled to the model's expected sizes automatically.

**Sampling-rate assumption**: the feature-extraction code in `data_wrangling_scripts/neurocurator.py` assumes fs = 20 kHz when computing trough-to-peak and FWHM. If you record at a different rate, resample your waveforms before constructing `waveforms.csv`.

Validate any dataset folder you build with:

```bash
hippie-cli validate-data datasets_hippie/<your_dataset>
```

### Available datasets

These are the paper datasets that ship in this repo under `datasets_hippie/`. `Samples` is the number of rows in `labels.csv`. Paper benchmarks use the labeled subset of each.

| Dataset (paper name) | Directory | Recording technology | Cell types / labels | Samples |
|----------------------|-----------|----------------------|---------------------|---------|
| **Toy (synthetic)** | `toy` | — | PV / SOM (fake) | 20 |
| Häusser (mouse cerebellum) | `hausser_cell_type` | C4 database — Beau et al. 2024 | GoC, MFB, MLI, PkC_ss, PkC_cs (paper Methods also list GrCs; not present in this dump) | ~3,996 |
| Hull (mouse cerebellum) | `hull_cell_type` | C4 database — Beau et al. 2024 | GoC, MFB, MLI, PkC_ss, PkC_cs | 206 |
| Lisberger (macaque cerebellum) | `lisberger_labeled_cell_type` | C4 database — Beau et al. 2024 | GoC, MFB, MLI, PkC_ss, PkC_cs | 1,152 |
| Lakunina Mouse A1 | `a1data_remove_undef` | Silicon probe | EXC, PV, SOM | 285 |
| Juxtacellular Mouse S1 | `juxtacellular_mouse_s1_area` | Juxtacellular micropipette | E/FS × layer + SOM (5 cell types) | 223 |
| CellExplorer (mouse VC + HPC) | `cellexplorer_cell_type` | Neuropixels 1.0 | PV, SST, Pyramidal, Axo-axonic, Juxtacellular, VIP, VGAT | 430 |
| Allen Visual Coding (labeled subset) | `allen_scope_neuropixel_area_subset` | Neuropixels | 19 brain regions | (subset of the 82,094 full session set) |

Paper datasets **not** shipped in this repo (download separately to reproduce those figures): Watson rat frontal cortex (DANDI 000041, 64-site silicon probes), Ramachandran rat S1 (NeuroNexus 32-ch), Calvigioni mouse PFC (Neuropixel), IBL Brain Wide Map, and the full 82,094-unit Allen Visual Coding session set. See `data_wrangling_scripts/README.md` for download + conversion recipes.

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

### Quick start (30 seconds, no GPU required)

After `pip install -e .`, embed the bundled toy dataset (20 synthetic
neurons, ships in the repo) with the pretrained checkpoint:

```bash
hippie-cli embed \
  --datasets-root ./datasets_hippie \
  --datasets toy \
  --output ./toy_embeddings.npz \
  --device cpu
```

This downloads the public checkpoint from the HuggingFace Hub
(repo `Jesusgf23/hippie`, file `hippie_techcond_v1.ckpt`, ~290 MB; cached after the
first run), embeds the 20 toy units into the locked 30-D latent space,
and writes a single `.npz` with keys `embeddings`, `labels`,
`dataset_ids`, `technology_ids`, and `neuron_ids`. On a laptop CPU this
completes in well under a minute.

### Embed one or more paper datasets

```bash
hippie-cli embed \
  --datasets-root ./datasets_hippie \
  --datasets hausser_cell_type lisberger_labeled_cell_type \
  --output ./paper_embeddings.npz
```

The equivalent script form (slightly more flexible, exposes label
canonicalization and per-dataset technology IDs) is:

```bash
python examples/extract_embeddings.py \
  --datasets-root ./datasets_hippie \
  --datasets hausser_cell_type lisberger_labeled_cell_type \
  --output ./paper_embeddings.npz
```

Pass `--checkpoint ./hippie_techcond_v1.ckpt` to use a local checkpoint
instead of the Hub download. With no `--datasets` argument, the script
defaults to the set of datasets actually shipped under `datasets_hippie/`.

### Bring your own data

If you do not have NWB / ACQM recordings (and so cannot use
`data_wrangling_scripts/`), build the four CSVs directly from any sorted
spike train. Minimal recipe:

```python
import numpy as np, pandas as pd

# spike_times: dict {unit_id: 1-D np.array of times in seconds}
# templates  : dict {unit_id: 1-D np.array of mean waveform, trough-centered}

def isi_hist(spikes_s, bin_ms=1, max_ms=100):
    isi_ms = np.diff(np.sort(spikes_s)) * 1000
    edges = np.arange(0, max_ms + bin_ms, bin_ms)
    return np.histogram(isi_ms, bins=edges)[0]

def acg(spikes_s, bin_ms=1, max_ms=100):
    s_ms = np.sort(spikes_s) * 1000
    lags = (s_ms[:, None] - s_ms[None, :]).ravel()
    lags = lags[(lags != 0) & (np.abs(lags) <= max_ms)]
    edges = np.arange(-max_ms - bin_ms / 2, max_ms + bin_ms, bin_ms)
    h = np.histogram(lags, bins=edges)[0]
    h[len(h) // 2] = 0  # zero the center bin
    return h

units = sorted(spike_times.keys())
pd.DataFrame([templates[u]              for u in units]).to_csv("waveforms.csv", index=False)
pd.DataFrame([isi_hist(spike_times[u])  for u in units]).to_csv("isi_dist.csv",  index=False)
pd.DataFrame([acg(spike_times[u])       for u in units]).to_csv("acg.csv",       index=False)
pd.DataFrame({"label": ["unknown"] * len(units)}).to_csv("labels.csv", index=False)
```

Then validate and embed:

```bash
mkdir -p datasets_hippie/my_data && mv waveforms.csv isi_dist.csv acg.csv labels.csv datasets_hippie/my_data/
hippie-cli validate-data datasets_hippie/my_data
hippie-cli embed --datasets-root ./datasets_hippie --datasets my_data --output my_embeddings.npz
```

For NWB / ACQM / DANDI / IBL / Allen-SDK sources, use the
`Neurocurator` class and notebooks in `data_wrangling_scripts/`:

| Source | Notebook |
|---|---|
| Allen Institute Visual Coding (Allen SDK) | [`allen_nwb_to_csv_converter.ipynb`](data_wrangling_scripts/allen_nwb_to_csv_converter.ipynb) |
| IBL Brain Wide Map (ONE API) | [`ibl_one_to_csv_converter.ipynb`](data_wrangling_scripts/ibl_one_to_csv_converter.ipynb) |
| ACQM `.zip` (HD-MEA) | [`acqm_to_csv_converter.ipynb`](data_wrangling_scripts/acqm_to_csv_converter.ipynb) |
| DANDI NWB (Watson, Calvigioni, Ramachandran, …) | Allen notebook template — swap the download cell for `dandi download` |

### Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `FileNotFoundError: .../<name>/waveforms.csv` | The folder `datasets_hippie/<name>/` does not exist | Either build that folder via `hippie-cli validate-data` (see "Bring your own data"), or restrict `--datasets` to one of the shipped names |
| `ModuleNotFoundError: huggingface_hub` | Optional dep, only needed for `from_pretrained` | `pip install huggingface-hub` |
| `ValueError: Unknown tech_id` from `get_embeddings` | Passed a string not in `TECHNOLOGY_IDS` | Use `"neuropixels"`, `"silicon_probe"`, or `"juxtacellular"`. For an unseen rig, pass integer `0` (zero-init source embedding) |
| `scripts/train.py` can't find `--data-dir` | Default is `./datasets_hippie`; if running from a subdir, pass `--data-dir <path>` explicitly | — |
| Validator warns about non-finite values | Source data has NaNs/Infs (some shipped paper datasets do) | Loaders sanitize at runtime; this is a warning, not an error |
| `make build` fails on Linux with "no such file: Dockerfile" | Case-insensitive macOS used to mask a lowercase `dockerfile` | Already fixed — `Dockerfile` (capital D) ships in the current tree |

### Train HIPPIE on your own data

```bash
python scripts/train.py \
  --dataset my_data \
  --data-dir ./datasets_hippie \
  --output checkpoints/my_run.ckpt \
  --epochs 100 \
  --config class_decoder_source_bn_aug_reg
```

This runs the locked production-default architecture
(`class_decoder_source_bn_aug_reg`, β = 1.0, z_dim = 30,
batch_size = 128) for `--epochs` epochs and writes a single Lightning
checkpoint. No held-out validation, no KNN/MLP heads — the trainer is
intentionally minimal. For paper-reproducing benchmarking (K-fold,
holdout, balanced accuracy, W&B logging), use the companion
[`hippie_benchmarking_release`](https://github.com/JesusGF1/hippie_benchmarking_release)
repository.

You can then either:

```bash
# extract embeddings against your trained model
hippie-cli embed \
  --checkpoint checkpoints/my_run.ckpt \
  --datasets-root ./datasets_hippie --datasets my_data \
  --output my_embeddings.npz

# classify a new NWB recording against a labeled reference
python hippie_nwb_classify.py session.nwb \
  --checkpoint checkpoints/my_run.ckpt \
  --train-embeddings labeled_reference_embeddings.csv \
  --z-dim 30
```

### Generative defaults

The production trainer above uses the **discriminative defaults** that
prioritize classification/clustering accuracy on the embedding. For the
**generative experiments** (counterfactual decoding, cross-modal
imputation), use:

```bash
python scripts/train.py \
  --dataset my_data \
  --output checkpoints/my_generative.ckpt \
  --z-dim 16 --beta 0.1 --batch-size 256
```

Lowering β (1.0 → 0.1) gives the decoder more reconstruction capacity at
the cost of latent regularization; `z_dim=16` is the value used in the
paper for the generative figures.

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

**Regularization and evaluation details are described in the Methods section of the manuscript and in the [benchmarking repository](https://github.com/JesusGF1/hippie_benchmarking_release).**

## Module Reference

### Core package (`hippie/`)

- **`multimodal_model.py`**: MultiModal CVAE with configurable ablations (`CVAEConfig` + `ExperimentConfigs`)
- **`unimodal_model.py`**: Single-modality CVAE implementation
- **`vae.py`**: Unconditioned VAE for unsupervised data compression
- **`dataloading.py`**: Dataset classes (`EphysDatasetLabeled`, `MultiModalEphysDataset`, `none_safe_collate`)
- **`backbones.py`**: 1D ResNet-18 encoder/decoder architectures (referred to as "1dResNet" in the paper)
- **`augmentations.py`**: Data augmentation transformations
- **`optimizers.py`**: Custom optimizers (AdamWScheduleFree)
- **`checkpoint.py`**: Checkpoint loading helpers (`build_model`, `build_unconditioned_model`)
- **`inference.py`**: Pretrained-model inference API (`HIPPIEClassifier`, `TECHNOLOGY_IDS`)
- **`cli.py`**: `hippie-cli` entry point (`validate-data`, `embed`, `predict`)
- **`utils.py`**: Legacy bimodal embedding helper

### Scripts and entry points

- **`hippie-cli`**: User-facing CLI installed by `pip install -e .` (see `hippie/cli.py`)
- **`scripts/train.py`**: Minimal trainer — load one dataset, pretrain, save a `.ckpt`
- **`scripts/generate_toy_dataset.py`**: Deterministically regenerate `datasets_hippie/toy/`
- **`examples/extract_embeddings.py`**: Reference script for the full embedding flow
- **`hippie_nwb_classify.py`**: End-to-end pipeline from an NWB file to classified neurons (precomputed units or raw traces via SpikeInterface)
- **`Makefile`**: Docker build/run targets

### Data wrangling (`data_wrangling_scripts/`)

- **`neurocurator.py`**: Core `Neurocurator` class — loads ACQM zips or NWB files, computes mean waveforms, ISI distributions, autocorrelograms, and per-unit shape features
- **`allen_nwb_to_csv_converter.ipynb`**: Convert one Allen Institute Visual Coding session (Allen SDK) to HIPPIE CSVs; also a template for DANDI NWB files
- **`ibl_one_to_csv_converter.ipynb`**: Convert one IBL Brain Wide Map insertion (ONE API, public Open Alyx mirror) to HIPPIE CSVs
- **`acqm_to_csv_converter.ipynb`**: Convert ACQM `.zip` HD-MEA recordings to HIPPIE CSVs

## Architecture Details

### MultiModal CVAE

```
Input Modalities (Wave, ISI, ACG)
    ↓
Separate Encoders (1D ResNet-18)
    ↓
[Optional] Fusion Encoder
    ↓
Latent Space (z_dim)
    ↓
[Optional] Class/Source Embeddings
    ↓
Separate Decoders (1D ResNet-18)
    ↓
Reconstructions + KL Divergence Loss
```

**Loss Function:**
```
L = Σ(λ_m × MSE(x_m, x̂_m)) + β × KL(q(z|x) || p(z))
    + λ_c × ConsistencyLoss(x̂_with_class, x̂_without_class)
```

See [`QUICK_CONFIG_REFERENCE.md`](QUICK_CONFIG_REFERENCE.md) for the
list of supported configurations and the asymmetric-CVAE design that
backs the production default.

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

- [QUICK_CONFIG_REFERENCE.md](QUICK_CONFIG_REFERENCE.md): Configuration cheat sheet and ablation study results
- [data_wrangling_scripts/README.md](data_wrangling_scripts/README.md): Data conversion utilities
