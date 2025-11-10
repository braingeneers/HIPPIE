import sys
import os
import time

code_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'hippie'))
sys.path.append(code_dir)

# Now import directly from the modules (no 'code.' prefix)
from dataloading import MultiModalEphysDataset, EphysDatasetLabeled, none_safe_collate
from multimodal_model import MultiModalCVAE, MultiModalCVAETrainModule
from utils import make_confmat, get_embeddings
from augmentations import AugmentedMultiModalEphysDataset

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Timer
import pandas as pd
import argparse
import wandb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score
import numpy as np
from torch.utils.data import random_split, WeightedRandomSampler
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from multimodal_model import CVAEConfig, ExperimentConfigs

# -------------------------------
# Resource monitoring callback
# -------------------------------
try:
    import psutil
    _HAS_PSUTIL = True
except Exception:
    _HAS_PSUTIL = False


class ResourceMonitor(pl.Callback):
    """Logs GPU/CPU memory and average step time to W&B every N steps."""
    def __init__(self, log_every_n_steps: int = 50, namespace: str = "resources"):
        self.log_every_n_steps = max(1, log_every_n_steps)
        self.ns = namespace
        self._last_time = None
        self._accum_step_time = 0.0
        self._accum_steps = 0

    def on_train_start(self, trainer, pl_module):
        self._reset_cuda_peaks()
        self._last_time = time.perf_counter()

    def on_train_epoch_start(self, trainer, pl_module):
        self._reset_cuda_peaks()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Step timing
        now = time.perf_counter()
        if self._last_time is not None:
            self._accum_step_time += (now - self._last_time)
            self._accum_steps += 1
        self._last_time = now

        # Periodic log
        global_step = trainer.global_step
        if global_step % self.log_every_n_steps == 0 and global_step > 0:
            metrics = {}
            # Avg step time over the interval
            if self._accum_steps > 0:
                metrics[f"{self.ns}/avg_step_time_s"] = self._accum_step_time / self._accum_steps
                self._accum_step_time, self._accum_steps = 0.0, 0

            # CPU memory (RSS)
            if _HAS_PSUTIL:
                process = psutil.Process(os.getpid())
                rss_mb = process.memory_info().rss / (1024 ** 2)
                metrics[f"{self.ns}/cpu_rss_mb"] = rss_mb

            # GPU memory for each device
            if torch.cuda.is_available():
                for d in range(torch.cuda.device_count()):
                    device = torch.device(f"cuda:{d}")
                    curr = torch.cuda.memory_allocated(device) / (1024 ** 2)
                    reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
                    peak = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                    metrics[f"{self.ns}/gpu{d}_mem_alloc_mb"] = curr
                    metrics[f"{self.ns}/gpu{d}_mem_reserved_mb"] = reserved
                    metrics[f"{self.ns}/gpu{d}_mem_peak_mb"] = peak

            if metrics:
                wandb.log(metrics, step=global_step)

    def on_validation_epoch_end(self, trainer, pl_module):
        # Log peaks at epoch boundary, then reset peaks.
        if torch.cuda.is_available():
            metrics = {}
            for d in range(torch.cuda.device_count()):
                peak = torch.cuda.max_memory_allocated(d) / (1024 ** 2)
                metrics[f"{self.ns}/val_gpu{d}_mem_peak_mb"] = peak
            if metrics:
                wandb.log(metrics, step=trainer.global_step)
        self._reset_cuda_peaks()

    def _reset_cuda_peaks(self):
        if torch.cuda.is_available():
            for d in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(d)


def create_balanced_sampler(dataset, labels):
    """
    Create a WeightedRandomSampler for class-balanced sampling.

    Args:
        dataset: The dataset to sample from
        labels: numpy array of class labels

    Returns:
        WeightedRandomSampler configured for balanced sampling
    """
    # Count samples per class
    unique_labels, label_counts = np.unique(labels, return_counts=True)

    # Calculate class weights (inverse frequency)
    class_weights = 1.0 / label_counts

    # Create a weight for each sample based on its class
    sample_weights = np.zeros(len(labels))
    for label_idx, label in enumerate(unique_labels):
        mask = labels == label
        sample_weights[mask] = class_weights[label_idx]

    # Create sampler
    sampler = WeightedRandomSampler(
        weights=torch.FloatTensor(sample_weights),
        num_samples=len(dataset),
        replacement=True  # Sample with replacement to ensure balanced batches
    )

    print(f"\n{'='*60}")
    print("Class-Balanced Sampling Enabled")
    print(f"{'='*60}")
    print(f"Number of classes: {len(unique_labels)}")
    print(f"Class distribution before balancing:")
    for label, count in zip(unique_labels, label_counts):
        print(f"  Class {label}: {count} samples ({100*count/len(labels):.2f}%)")
    print(f"Class weights (1/frequency): {class_weights}")
    print(f"{'='*60}\n")

    return sampler


def _log_timer(timer_obj: Timer, prefix: str):
    """Helper to robustly pull timings from Lightning Timer across versions."""
    def _safe_elapsed(phase):
        try:
            val = timer_obj.time_elapsed(phase)
            return float(val) if val is not None else None
        except Exception:
            return None

    elapsed_fit = _safe_elapsed("fit")
    elapsed_train = _safe_elapsed("train")
    elapsed_validate = _safe_elapsed("validate")

    payload = {}
    if elapsed_fit is not None:
        payload[f"time/{prefix}_fit_s"] = elapsed_fit
    if elapsed_train is not None:
        payload[f"time/{prefix}_train_s"] = elapsed_train
    if elapsed_validate is not None:
        payload[f"time/{prefix}_val_s"] = elapsed_validate

    if payload:
        wandb.log(payload)


# ---------- FIX PART 1: Robust embedding extraction ----------
def get_embeddings_multimodal(loader, model):
    """Extract embeddings from a multimodal model (robust to call styles & zero-variance rows)."""
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for sample in loader:
            try:
                out = model(sample)
            except TypeError:
                out = model(*sample)
            embedding = out[0].detach().cpu().numpy()

            # Normalize embeddings row-wise (avoid div by zero)
            std = np.std(embedding, axis=1, keepdims=True)
            std[std == 0] = 1.0
            embedding = (embedding - np.mean(embedding, axis=1, keepdims=True)) / std
            all_embeddings.extend(embedding)

            label = sample[1]
            if getattr(label, "ndim", 1) == 2:
                cls_label, _ = label.unbind(1)
            else:
                cls_label = label
            all_labels.extend(cls_label.detach().cpu().numpy())

    return np.array(all_embeddings), np.array(all_labels)


# ---------- FIX PART 2: NaN sanitize ----------
def _nan_sanitize(x: np.ndarray, name: str, dataset: str):
    if np.isnan(x).any():
        print(f"NaN values detected in dataset '{dataset}', modality '{name}'. Replacing with 0.")
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x


def load_dataset_data(dataset_name, dataset_files):
    """Load waveform, ISI, and ACG data for a dataset."""
    wf = pd.read_csv(f"../datasets_hippie/{dataset_name}/waveforms.csv").to_numpy()
    isi = pd.read_csv(f"../datasets_hippie/{dataset_name}/isi_dist.csv").to_numpy()

    # Load ACG data if it exists, otherwise use zeros
    acg_path = f"../datasets_hippie/{dataset_name}/acg.csv"
    acg = pd.read_csv(acg_path).to_numpy() if os.path.exists(acg_path) else np.zeros_like(isi)

    # Sanitize NaNs
    wf = _nan_sanitize(wf, "wave", dataset_name)
    isi = _nan_sanitize(isi, "isi", dataset_name)
    acg = _nan_sanitize(acg, "acg", dataset_name)

    # Load labels if they exist
    labels = None
    labels_path = f"../datasets_hippie/{dataset_name}/labels.csv"
    if os.path.exists(labels_path):
        labels_df = pd.read_csv(labels_path)
        labels = labels_df[labels_df.columns[0]].values
    elif os.path.exists(f"../datasets_hippie/{dataset_name}/celltypes.csv"):
        labels_df = pd.read_csv(f"../datasets_hippie/{dataset_name}/celltypes.csv")
        labels = labels_df[labels_df.columns[0]].values

    source_id = dataset_files[dataset_name]

    print(f"Dataset {dataset_name} has shapes - waveform: {wf.shape}, isi: {isi.shape}, acg: {acg.shape}")

    return wf, isi, acg, labels, source_id


# ---------- FIX PART 3: Map predict labels to training encoder space ----------
def map_labels_to_training_encoder(le_train: LabelEncoder, labels: np.ndarray, fallback: int = 0):
    """Map string labels to indices in le_train; unseen labels map to `fallback` (default 0)."""
    train_set = set(le_train.classes_)
    out = np.empty(labels.shape[0], dtype=int)
    for i, lbl in enumerate(labels):
        if lbl in train_set:
            out[i] = le_train.transform([lbl])[0]
        else:
            out[i] = fallback
    return out


if __name__ == '__main__':
    # -------------------------------
    # Parse arguments
    # -------------------------------
    parser = argparse.ArgumentParser()
    # Common arguments
    parser.add_argument("--z_dim", type=int, default=10, help="Dimension of latent space")
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--beta', type=float, default=1, help="Weight for KL divergence loss")
    parser.add_argument('--training-dataset', type=str, required=True, help="Dataset to train on")
    parser.add_argument('--predict-dataset', type=str, required=True, help="Dataset to predict on")
    parser.add_argument('--upload-model', action='store_true')
    parser.add_argument('--wandb-tag', type=str, default="Hippie_cross_dataset")
    parser.add_argument('--project', type=str, default="HIPPIE")
    parser.add_argument('--finetune-without-labels', type=bool, default=True)
    parser.add_argument('--pretrain-max-epochs', type=int, default=100)
    parser.add_argument('--finetune-max-epochs', type=int, default=10)
    parser.add_argument('--supervised-max-epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--supervised-batch-size', type=int, default=64)
    parser.add_argument('--early-stopping-patience', type=int, default=30)
    parser.add_argument('--gradient-clip-val', type=float, default=1.0)
    parser.add_argument('--train-val-split', type=float, default=0.9)
    parser.add_argument('--finetune-split', type=float, default=0.1)
    parser.add_argument('--limit-train-batches', type=float, default=None)
    parser.add_argument('--limit-val-batches', type=float, default=None)

    # New arguments for multimodal approach
    parser.add_argument('--model-type', type=str, choices=['unimodal', 'multimodal'], default='multimodal',
                        help='Whether to use separate models for each modality or a joint model')
    parser.add_argument('--mod1-weight', type=float, default=1.0,
                        help='Weight for the waveform modality loss in multimodal model')
    parser.add_argument('--mod2-weight', type=float, default=1.0,
                        help='Weight for the ISI modality loss in multimodal model')

    parser.add_argument('--wave-weight', type=float, default=1.0,
                        help='Weight for the waveform modality loss')
    parser.add_argument('--isi-weight', type=float, default=1.0,
                        help='Weight for the ISI modality loss')
    parser.add_argument('--acg-weight', type=float, default=1.0,
                        help='Weight for the ACG modality loss')

    parser.add_argument('--config', type=str, default="baseline", choices=["baseline", "with_source", "with_class", "with_both_embeddings", "with_batch_norm", "full_model", "no_fusion", "with_light_augmentations", "with_heavy_augmentations", "augmentation_ablation"])

    # Class balancing argument
    parser.add_argument("--use_balanced_sampling", action="store_true",
                        help="Use class-balanced sampling via WeightedRandomSampler during supervised training")

    # DataLoader workers
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Number of worker processes for data loading (default: 4)")

    args = parser.parse_args()

    # -------------------------------
    # Common setup
    # -------------------------------
    accelerator = "gpu"
    limit_train_batches = args.limit_train_batches
    limit_val_batches = args.limit_val_batches
    project = args.project
    FINETUNE_WITHOUT_LABELS = args.finetune_without_labels
    trainer_kwargs = {}

    torch.manual_seed(42)

    # Initialize single wandb run at start
    wandb.init(
        project=project,
        name=f"{args.wandb_tag}-train_{args.training_dataset}-predict_{args.predict_dataset}-config_{args.config}_zdim_{args.z_dim}_B_{args.beta}",
        config=vars(args)
    )

    # -------------------------------
    # Dataset setup
    # -------------------------------
    dataset_files = {
        #"braingeneers_manual_curation": 1,
        #"cellexplorer_area": 2,
        #"cellexplorer_cell_type": 2,
        "hausser_cell_type": 1,#
        "hull_cell_type": 2,#
        "lissberger_labeled_cell_type": 3,#
        # "mouse_organoids_cell_line": 4,
        # "mouse_slice_area": 5,
        #"juxtacellular_mouse_s1_area": 6,
        #"juxtacellular_mouse_s1_cell_type": 6,
        #"cortical_labs_3": 7
    }

    all_dataset_files = dataset_files.copy()
    num_sources = max(all_dataset_files.values()) + 1

    # Validate dataset arguments
    if args.training_dataset not in dataset_files:
        raise ValueError(f"Training dataset '{args.training_dataset}' not found in available datasets: {list(dataset_files.keys())}")
    if args.predict_dataset not in dataset_files:
        raise ValueError(f"Predict dataset '{args.predict_dataset}' not found in available datasets: {list(dataset_files.keys())}")

    print(f"Training on: {args.training_dataset}")
    print(f"Predicting on: {args.predict_dataset}")
    
    # Remove training and predict datasets from pretraining
    pretrain_dataset_files = dataset_files.copy()
    if args.training_dataset in pretrain_dataset_files:
        pretrain_dataset_files.pop(args.training_dataset)
    if args.predict_dataset in pretrain_dataset_files:
        pretrain_dataset_files.pop(args.predict_dataset)
    
    # Remove juxtacellular datasets if they're related to training/predict datasets
    if "juxtacellular" in args.training_dataset or "juxtacellular" in args.predict_dataset:
        pretrain_dataset_files.pop("juxtacellular_mouse_s1_area", None)
        pretrain_dataset_files.pop("juxtacellular_mouse_s1_cell_type", None)
    
    if "cellexplorer" in args.training_dataset or "cellexplorer" in args.predict_dataset:
        pretrain_dataset_files.pop("cellexplorer_cell_type", None)
        pretrain_dataset_files.pop("cellexplorer_area", None)
    
    print(f"Pretraining on: {list(pretrain_dataset_files.keys())}")

    # -------------------------------
    # PHASE 1: PRETRAINING
    # -------------------------------
    print("=" * 50)
    print("PHASE 1: PRETRAINING")
    print("=" * 50)

    time_phase1_start = time.time()
    
    # Define standard modality sizes - the MultiModalEphysDataset will handle padding/truncation
    modalities = {
        "wave": 50,   # Standard waveform size
        "isi": 100,   # Standard ISI size
        "acg": 200    # Standard ACG size
    }
    
    # Get config
    EXPERIMENT_CONFIGS = {
        "baseline": ExperimentConfigs.baseline(),
        "with_source": ExperimentConfigs.with_source(),
        "with_class": ExperimentConfigs.with_class(),
        "with_both_embeddings": ExperimentConfigs.with_both_embeddings(),
        "with_batch_norm": ExperimentConfigs.with_batch_norm(),
        "full_model": ExperimentConfigs.full_model(),
        "no_fusion": ExperimentConfigs.no_fusion(),
        "with_light_augmentations": ExperimentConfigs.with_light_augmentations(),
        "with_heavy_augmentations": ExperimentConfigs.with_heavy_augmentations(),
        "augmentation_ablation": ExperimentConfigs.augmentation_ablation(),
    }
    config = EXPERIMENT_CONFIGS[args.config]
    
    # Load data for pretraining from all datasets except training and predict
    all_waveforms = []
    all_isi = []
    all_acg = []
    labels = []
    datasets_multi = []
    
    for folder in pretrain_dataset_files:
        wf = pd.read_csv(f"../datasets_hippie/{folder}/waveforms.csv").to_numpy()
        isi = pd.read_csv(f"../datasets_hippie/{folder}/isi_dist.csv").to_numpy()
        # Load ACG data if it exists, otherwise use zeros
        acg_path = f"../datasets_hippie/{folder}/acg.csv"
        acg = pd.read_csv(acg_path).to_numpy() if os.path.exists(acg_path) else np.zeros_like(isi)
        
        # Sanitize NaNs
        wf = _nan_sanitize(wf, "wave", folder)
        isi = _nan_sanitize(isi, "isi", folder)
        acg = _nan_sanitize(acg, "acg", folder)

        source = np.full((wf.shape[0]), pretrain_dataset_files[folder])
        print(f"Pretraining folder {folder} has shapes - waveform: {wf.shape}, isi: {isi.shape}, acg: {acg.shape}")

        all_waveforms.append(wf)
        all_isi.append(isi)
        all_acg.append(acg)
        labels.append(source)

        # Create multimodal dataset with all modalities
        data_dict = {
            "wave": wf,
            "isi": isi,
            "acg": acg
        }

        dataset_multi = MultiModalEphysDataset(data_dict, source, mode="multi", modality_sizes=modalities)
        
        # Wrap with augmentations if enabled
        if config.use_augmentations and config.augment_pretraining:
            dataset_multi = AugmentedMultiModalEphysDataset(dataset_multi, config, phase="pretraining")
        
        datasets_multi.append(dataset_multi)
    
    if not datasets_multi:
        print("Warning: No datasets available for pretraining. Skipping pretraining phase.")
        joint_model = None
        joint_path = None
    else:
        labels = np.concatenate(labels, axis=0)
        all_multi_dataset = torch.utils.data.ConcatDataset(datasets_multi)
        
        # Split datasets for pretraining
        prop = args.train_val_split
        indices = list(range(len(all_multi_dataset)))
        train_indices, test_indices = random_split(
            indices, [int(prop * len(indices)), len(indices) - int(prop * len(indices))]
        )
        
        # Create dataloaders for pretraining
        train_multi_dataset = torch.utils.data.Subset(all_multi_dataset, train_indices)
        test_multi_dataset = torch.utils.data.Subset(all_multi_dataset, test_indices)
        
        train_loader_multi = torch.utils.data.DataLoader(
            train_multi_dataset, batch_size=args.batch_size, shuffle=True,
            collate_fn=none_safe_collate, num_workers=args.num_workers
        )
        test_loader_multi = torch.utils.data.DataLoader(
            test_multi_dataset, batch_size=args.batch_size, shuffle=False,
            collate_fn=none_safe_collate, num_workers=args.num_workers
        )
        
        
        # Create multimodal model for pretraining
        joint_model = MultiModalCVAE(
            modalities=modalities,
            z_dim=args.z_dim,
            num_sources=num_sources,
            num_classes=5,  # Dummy value for pretraining
            config=config,
        )
        
        # Define modality weights
        modality_weights = {
            "wave": args.wave_weight,
            "isi": args.isi_weight,
            "acg": args.acg_weight
        }
        
        joint_model = MultiModalCVAETrainModule(
            joint_model,
            modality_weights=modality_weights,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            config=config,
        )
        
        # PRETRAIN: callbacks & trainer
        joint_checkpoint = pl.callbacks.ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min")
        joint_earlystop = pl.callbacks.EarlyStopping(monitor="val_loss", patience=args.early_stopping_patience, mode="min")
        timer_pretrain = Timer(duration=None)
        resource_cb_pretrain = ResourceMonitor(log_every_n_steps=50)
        
        # Phase marker for pretrain start
        wandb.log({"phase": "pretrain_start"})
        
        joint_trainer = pl.Trainer(
            max_epochs=args.pretrain_max_epochs,
            accelerator=accelerator,
            logger=pl.loggers.WandbLogger(experiment=wandb.run),
            callbacks=[joint_checkpoint, joint_earlystop, timer_pretrain, resource_cb_pretrain],
            limit_train_batches=limit_train_batches,
            limit_val_batches=limit_val_batches,
            gradient_clip_val=args.gradient_clip_val,
            **trainer_kwargs,
        )
        
        # Train joint model
        joint_trainer.fit(joint_model, train_loader_multi, test_loader_multi)
        # Log pretrain timing
        _log_timer(timer_pretrain, prefix="pretrain")
        
        joint_path = joint_checkpoint.best_model_path
        joint_model.load_state_dict(torch.load(joint_path)["state_dict"])

    time_phase1_end = time.time()
    time_phase1 = time_phase1_end - time_phase1_start
    print(f"\nPhase 1 (Pretraining) completed in {time_phase1:.2f} seconds ({time_phase1/60:.2f} minutes)")
    wandb.log({"time/phase1_pretrain_seconds": time_phase1, "time/phase1_pretrain_minutes": time_phase1/60})

    # -------------------------------
    # PHASE 2: FINETUNING
    # -------------------------------
    print("=" * 50)
    print("PHASE 2: FINETUNING")
    print("=" * 50)

    time_phase2_start = time.time()
    
    # Load training dataset for finetuning
    train_wf, train_isi, train_acg, train_labels, train_source_id = load_dataset_data(args.training_dataset, all_dataset_files)
    
    if train_labels is None:
        raise ValueError(f"Training dataset '{args.training_dataset}' must have labels for supervised training")
    
    # Create finetuning dataset (without labels for unsupervised adaptation)
    finetune_data_dict = {
        "wave": train_wf,
        "isi": train_isi,
        "acg": train_acg
    }
    
    # Keep modalities consistent with pretraining - all datasets will be resized to these dimensions
    
    if joint_model is not None and FINETUNE_WITHOUT_LABELS:
        label_ft = np.full((train_wf.shape[0]), train_source_id)
        finetune_dataset_multi = MultiModalEphysDataset(finetune_data_dict, label_ft, mode="multi", modality_sizes=modalities)
        
        # Debug: Check what sizes we're actually getting
        print(f"Finetuning modality sizes config: {modalities}")
        sample_data = finetune_dataset_multi[0][0]  # Get first sample
        for mod_name, tensor in sample_data.items():
            print(f"Finetune {mod_name} tensor shape: {tensor.shape}")
        print(f"Original training dataset shapes - waveform: {train_wf.shape}, isi: {train_isi.shape}, acg: {train_acg.shape}")
        
        # Split for finetuning
        prop = args.finetune_split
        indices = list(range(len(finetune_dataset_multi)))
        train_indices, test_indices = random_split(
            indices, [int(prop * len(indices)), len(indices) - int(prop * len(indices))]
        )
        
        # Create new model instance for fine-tuning with lower learning rate
        # Keep the original model architecture
        original_model = joint_model.model
        joint_model = MultiModalCVAETrainModule(
            original_model,
            modality_weights=modality_weights,
            learning_rate=(1/10)*args.learning_rate,
            weight_decay=args.weight_decay,
            config=config,
        )
        
        # Create dataloaders for fine-tuning
        # Apply augmentations only to training subset for fine-tuning
        if config.use_augmentations and config.augment_finetuning:
            augmented_finetune_dataset = AugmentedMultiModalEphysDataset(finetune_dataset_multi, config, phase="finetuning")
            train_finetune_dataset = torch.utils.data.Subset(augmented_finetune_dataset, train_indices)
        else:
            train_finetune_dataset = torch.utils.data.Subset(finetune_dataset_multi, train_indices)
        
        # Validation dataset is never augmented
        test_finetune_dataset = torch.utils.data.Subset(finetune_dataset_multi, test_indices)
        
        train_finetune_loader_multi = torch.utils.data.DataLoader(
            train_finetune_dataset, batch_size=args.batch_size, shuffle=False,
            collate_fn=none_safe_collate, num_workers=args.num_workers
        )
        test_finetune_loader_multi = torch.utils.data.DataLoader(
            test_finetune_dataset, batch_size=args.batch_size, shuffle=False,
            collate_fn=none_safe_collate, num_workers=args.num_workers
        )
        
        # FINE-TUNE: callbacks & trainer
        joint_checkpoint = pl.callbacks.ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min")
        joint_earlystop = pl.callbacks.EarlyStopping(monitor="val_loss", patience=args.early_stopping_patience, mode="min")
        timer_finetune = Timer(duration=None)
        resource_cb_finetune = ResourceMonitor(log_every_n_steps=50)
        
        # Phase marker for finetune start
        wandb.log({"phase": "finetune_start"})
        
        joint_trainer = pl.Trainer(
            max_epochs=args.finetune_max_epochs,
            accelerator=accelerator,
            logger=pl.loggers.WandbLogger(experiment=wandb.run),
            callbacks=[joint_checkpoint, joint_earlystop, timer_finetune, resource_cb_finetune],
            limit_train_batches=limit_train_batches,
            limit_val_batches=limit_val_batches,
            gradient_clip_val=args.gradient_clip_val,
            **trainer_kwargs,
        )
        
        joint_trainer.fit(joint_model, train_finetune_loader_multi, test_finetune_loader_multi)
        # Log finetune timing
        _log_timer(timer_finetune, prefix="finetune")
        
        joint_path = joint_checkpoint.best_model_path
        joint_model.load_state_dict(torch.load(joint_path)["state_dict"])

    time_phase2_end = time.time()
    time_phase2 = time_phase2_end - time_phase2_start
    print(f"\nPhase 2 (Finetuning) completed in {time_phase2:.2f} seconds ({time_phase2/60:.2f} minutes)")
    wandb.log({"time/phase2_finetune_seconds": time_phase2, "time/phase2_finetune_minutes": time_phase2/60})

    # -------------------------------
    # PHASE 3: SUPERVISED TRAINING
    # -------------------------------
    print("=" * 50)
    print("PHASE 3: SUPERVISED TRAINING")
    print("=" * 50)

    time_phase3_start = time.time()
    
    # Encode labels for supervised training
    le = LabelEncoder().fit(train_labels)
    train_labels_encoded = le.transform(train_labels)

    # Create train/val split for supervised training
    indices = list(range(len(train_wf)))
    train_size = int(args.train_val_split * len(indices))
    train_indices, val_indices = random_split(indices, [train_size, len(indices) - train_size])

    wf_train = train_wf[train_indices]
    wf_val = train_wf[val_indices]
    isi_train = train_isi[train_indices]
    isi_val = train_isi[val_indices]
    acg_train = train_acg[train_indices]
    acg_val = train_acg[val_indices]
    label_train = train_labels_encoded[train_indices]
    label_val = train_labels_encoded[val_indices]

    num_class_labels = len(np.unique(label_train))

    # Keep modalities consistent across all phases - already defined in pretraining

    # Create supervised model
    supervised_joint_model = MultiModalCVAE(
        modalities=modalities,
        z_dim=args.z_dim,
        num_sources=num_sources,
        num_classes=num_class_labels,
        config=config,
    )
    
    # Load pretrained weights if available, but skip the class embedding layer
    if joint_model is not None and joint_path is not None:
        joint_seq = torch.load(joint_path)
        if "model.class_embedding.weight" in joint_seq["state_dict"]:
            joint_seq["state_dict"].pop("model.class_embedding.weight")
        
        supervised_joint_model = MultiModalCVAETrainModule(
            supervised_joint_model,
            modality_weights=modality_weights,
            learning_rate=(1/10)*args.learning_rate,  # Lower learning rate for fine-tuning
            weight_decay=args.weight_decay,
            config=config,
        )
        supervised_joint_model.load_state_dict(joint_seq["state_dict"], strict=False)
        print("Loaded pretrained weights for supervised training")
    else:
        # No pretraining, start from scratch
        supervised_joint_model = MultiModalCVAETrainModule(
            supervised_joint_model,
            modality_weights=modality_weights,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            config=config,
        )
        print("No pretrained model available, starting supervised training from scratch")

    # Create source labels for embedding
    label_train_for_embedding = train_source_id * np.ones_like(label_train)
    label_val_for_embedding = train_source_id * np.ones_like(label_val)

    # Create supervised training datasets
    train_data_dict = {
        "wave": wf_train,
        "isi": isi_train,
        "acg": acg_train
    }

    val_data_dict = {
        "wave": wf_val,
        "isi": isi_val,
        "acg": acg_val
    }

    dataset_train_multi = MultiModalEphysDataset(
        train_data_dict,
        np.vstack((label_train, label_train_for_embedding)).T,
        mode="multi",
        modality_sizes=modalities
    )
    
    # Apply augmentations to supervised training dataset if enabled
    if config.use_augmentations and config.augment_supervised:
        dataset_train_multi = AugmentedMultiModalEphysDataset(dataset_train_multi, config, phase="supervised")

    dataset_val_multi = MultiModalEphysDataset(
        val_data_dict,
        np.vstack((label_val, label_val_for_embedding)).T,
        mode="multi",
        modality_sizes=modalities
    )

    # Create DataLoader with optional class-balanced sampling
    if args.use_balanced_sampling:
        # Create balanced sampler for training data
        train_sampler = create_balanced_sampler(dataset_train_multi, label_train)
        train_loader_multi = torch.utils.data.DataLoader(
            dataset_train_multi,
            batch_size=args.supervised_batch_size,
            sampler=train_sampler,  # Use sampler instead of shuffle
            collate_fn=none_safe_collate,
            num_workers=args.num_workers
        )
    else:
        train_loader_multi = torch.utils.data.DataLoader(
            dataset_train_multi,
            batch_size=args.supervised_batch_size,
            shuffle=True,
            collate_fn=none_safe_collate,
            num_workers=args.num_workers
        )

    test_loader_multi = torch.utils.data.DataLoader(
        dataset_val_multi, batch_size=args.supervised_batch_size,
        shuffle=False, collate_fn=none_safe_collate, num_workers=args.num_workers
    )

    # SUPERVISED: callbacks & trainer
    joint_checkpoint = pl.callbacks.ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min")
    joint_earlystop = pl.callbacks.EarlyStopping(monitor="val_loss", patience=args.early_stopping_patience, mode="min")
    lr_monitor_joint = pl.callbacks.LearningRateMonitor(logging_interval="step")
    timer_supervised = Timer(duration=None)
    resource_cb_supervised = ResourceMonitor(log_every_n_steps=50)

    # Phase marker for supervised start
    wandb.log({"phase": "supervised_start"})

    joint_trainer = pl.Trainer(
        max_epochs=args.supervised_max_epochs,
        accelerator=accelerator,
        logger=pl.loggers.WandbLogger(experiment=wandb.run),
        callbacks=[joint_checkpoint, joint_earlystop, lr_monitor_joint, timer_supervised, resource_cb_supervised],
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        gradient_clip_val=args.gradient_clip_val,
        **trainer_kwargs,
    )

    joint_trainer.fit(supervised_joint_model, train_loader_multi, test_loader_multi)
    # Log supervised timing
    _log_timer(timer_supervised, prefix="supervised")

    # Load best model checkpoint
    joint_path = joint_checkpoint.best_model_path
    wandb.log({"best_epoch_joint": joint_path})

    joint_seq = torch.load(joint_path)
    supervised_joint_model.load_state_dict(joint_seq["state_dict"])
    supervised_joint_model.eval()

    time_phase3_end = time.time()
    time_phase3 = time_phase3_end - time_phase3_start
    print(f"\nPhase 3 (Supervised Training) completed in {time_phase3:.2f} seconds ({time_phase3/60:.2f} minutes)")
    wandb.log({"time/phase3_supervised_seconds": time_phase3, "time/phase3_supervised_minutes": time_phase3/60})

    # -------------------------------
    # GET EMBEDDINGS FOR TRAINING DATASET
    # -------------------------------
    time_embedding_start = time.time()

    # Create full training dataset
    train_full_data_dict = {
        "wave": train_wf,
        "isi": train_isi,
        "acg": train_acg
    }

    train_full_dataset = MultiModalEphysDataset(
        train_full_data_dict,
        np.vstack((train_labels_encoded, np.ones_like(train_labels_encoded) * train_source_id)).T,
        mode="multi",
        modality_sizes=modalities
    )

    train_full_loader = torch.utils.data.DataLoader(
        train_full_dataset, batch_size=128, collate_fn=none_safe_collate, num_workers=args.num_workers
    )

    # Get embeddings for training dataset
    train_embeddings, train_labels_final = get_embeddings_multimodal(train_full_loader, supervised_joint_model)

    # Save training embeddings
    os.makedirs(f"../results_hippie_cross_dataset/train_{args.training_dataset}_predict_{args.predict_dataset}/config_{args.config}_zdim_{args.z_dim}_B_{args.beta}", exist_ok=True)

    train_embeddings_df = pd.DataFrame(train_embeddings)
    train_embeddings_df["label"] = le.inverse_transform(train_labels_final.astype(int))
    train_embeddings_df.to_csv(f"../results_hippie_cross_dataset/train_{args.training_dataset}_predict_{args.predict_dataset}/config_{args.config}_zdim_{args.z_dim}_B_{args.beta}/train_embeddings.csv", index=False)

    # -------------------------------
    # LOAD AND PROCESS PREDICT DATASET
    # -------------------------------
    predict_wf, predict_isi, predict_acg, predict_labels, predict_source_id = load_dataset_data(args.predict_dataset, all_dataset_files)

    if predict_labels is None:
        print(f"Warning: Predict dataset '{args.predict_dataset}' has no labels. Creating dummy labels for processing.")
        predict_labels = np.zeros(len(predict_wf), dtype=str)
        predict_labels[:] = "unknown"

    # Keep a combined encoder for reporting/saving (not used by the model)
    # Convert all labels to strings to avoid type mixing issues
    train_labels_str = [str(label) for label in train_labels]
    predict_labels_str = [str(label) for label in predict_labels]
    all_unique_labels = np.unique(np.concatenate([train_labels_str, predict_labels_str]))
    le_combined = LabelEncoder().fit(all_unique_labels)

    # ---------- FIX PART 3 applied: map predict labels for MODEL space ----------
    predict_labels_for_model = map_labels_to_training_encoder(le, predict_labels, fallback=0)

    # Create predict dataset
    predict_data_dict = {
        "wave": predict_wf,
        "isi": predict_isi,
        "acg": predict_acg
    }

    predict_dataset = MultiModalEphysDataset(
        predict_data_dict,
        np.vstack((predict_labels_for_model, np.ones_like(predict_labels_for_model) * predict_source_id)).T,
        mode="multi",
        modality_sizes=modalities
    )

    predict_loader = torch.utils.data.DataLoader(
        predict_dataset, batch_size=128, collate_fn=none_safe_collate, num_workers=args.num_workers
    )

    # Get embeddings for predict dataset
    predict_embeddings, predict_labels_final = get_embeddings_multimodal(predict_loader, supervised_joint_model)

    time_embedding_end = time.time()
    time_embedding = time_embedding_end - time_embedding_start
    print(f"\nEmbedding extraction completed in {time_embedding:.2f} seconds ({time_embedding/60:.2f} minutes)")
    wandb.log({"time/embedding_extraction_seconds": time_embedding, "time/embedding_extraction_minutes": time_embedding/60})

    # Save predict embeddings
    predict_embeddings_df = pd.DataFrame(predict_embeddings)
    # Labels in training-encoder space (may contain fallback)
    predict_embeddings_df["label_training_space"] = le.inverse_transform(np.clip(predict_labels_final.astype(int), 0, len(le.classes_) - 1))
    # Also store the original labels for reference
    predict_embeddings_df["label_original"] = predict_labels
    predict_embeddings_df.to_csv(f"../results_hippie_cross_dataset/train_{args.training_dataset}_predict_{args.predict_dataset}/config_{args.config}_zdim_{args.z_dim}_B_{args.beta}/predict_embeddings.csv", index=False)

    # -------------------------------
    # TRAIN KNN ON TRAINING EMBEDDINGS AND PREDICT ON TEST EMBEDDINGS
    # -------------------------------
    time_knn_start = time.time()

    # We need to use the original training label encoder for KNN training
    # since we want to predict classes that exist in the training set
    train_knn_labels = train_labels_final.astype(int)

    # Evaluate using KNN with different neighbor counts
    neighbor_options = list(range(5, min(20, len(np.unique(train_knn_labels)) * 3)))
    if not neighbor_options:  # If we have very few classes
        neighbor_options = [3, 5]

    best_accuracy = -1
    best_neighbors = neighbor_options[0]
    best_predictions = None

    print(f"Training KNN with {len(train_embeddings)} training samples and {len(np.unique(train_knn_labels))} classes")
    print(f"Testing different neighbor counts: {neighbor_options}")

    # For cross-validation on training set to select best k
    from sklearn.model_selection import cross_val_score

    cv_scores = {}
    for neighbor in neighbor_options:
        knn = KNeighborsClassifier(n_neighbors=neighbor)
        cv_score = cross_val_score(knn, train_embeddings, train_knn_labels, cv=min(5, len(np.unique(train_knn_labels))), scoring='balanced_accuracy')
        cv_scores[neighbor] = np.mean(cv_score)
        print(f"KNN with {neighbor} neighbors: CV balanced accuracy = {np.mean(cv_score):.4f} ± {np.std(cv_score):.4f}")

    # Select best k based on cross-validation
    best_neighbors = max(cv_scores, key=cv_scores.get)
    print(f"Selected best k = {best_neighbors} with CV score = {cv_scores[best_neighbors]:.4f}")

    # Train final KNN model with best k
    final_knn = KNeighborsClassifier(n_neighbors=best_neighbors)
    final_knn.fit(train_embeddings, train_knn_labels)

    # Make predictions on predict dataset
    predictions = final_knn.predict(predict_embeddings)
    prediction_probabilities = final_knn.predict_proba(predict_embeddings)

    # Convert predictions back to original labels (training encoder space)
    predicted_labels = le.inverse_transform(predictions.astype(int))

    # Calculate accuracy if we have true labels for the predict dataset
    accuracy_calculated = False
    if not (predict_labels == "unknown").all():  # If we have real labels
        # Create a label encoder for the predict dataset to get numeric labels for evaluation
        le_predict = LabelEncoder().fit(predict_labels)
        predict_labels_encoded_for_eval = le_predict.transform(predict_labels)

        # Find overlapping classes between training and predict datasets
        train_classes = set(le.classes_)
        predict_classes = set(le_predict.classes_)
        overlapping_classes = train_classes.intersection(predict_classes)

        if overlapping_classes:
            print(f"Overlapping classes: {overlapping_classes}")

            # Create mapping for evaluation
            true_labels_for_eval = []
            pred_labels_for_eval = []

            for i, (true_label, pred_idx) in enumerate(zip(predict_labels, predictions)):
                if true_label in overlapping_classes:
                    # Map both true and predicted labels to the training label encoder
                    true_labels_for_eval.append(le.transform([true_label])[0])
                    pred_labels_for_eval.append(pred_idx)

            if len(true_labels_for_eval) > 0:
                accuracy = balanced_accuracy_score(true_labels_for_eval, pred_labels_for_eval)
                print(f"Cross-dataset balanced accuracy: {accuracy:.4f}")
                accuracy_calculated = True

                # Confusion matrix
                conf_matrix = confusion_matrix(true_labels_for_eval, pred_labels_for_eval)
                available_classes = [c for c in le.classes_ if c in overlapping_classes]

                # Log metrics
                wandb.log({
                    "cross_dataset_balanced_accuracy": accuracy,
                    "best_k_neighbors": best_neighbors,
                    "num_overlapping_classes": len(overlapping_classes),
                    "num_evaluated_samples": len(true_labels_for_eval)
                })

                # Make confusion matrix figure if we have the function
                try:
                    figure_multi = make_confmat(conf_matrix, available_classes, best_neighbors)
                    wandb.log({
                        f"cross_dataset_confusion_matrix": wandb.Image(figure_multi),
                    })
                except Exception as e:
                    print(f"Could not create confusion matrix figure: {e}")
            else:
                print("No samples with overlapping classes found for evaluation")
        else:
            print("No overlapping classes between training and predict datasets")

    if not accuracy_calculated:
        wandb.log({"best_k_neighbors": best_neighbors})

    time_knn_end = time.time()
    time_knn = time_knn_end - time_knn_start
    print(f"\nKNN training and evaluation completed in {time_knn:.2f} seconds ({time_knn/60:.2f} minutes)")
    wandb.log({"time/knn_training_seconds": time_knn, "time/knn_training_minutes": time_knn/60})

    # -------------------------------
    # SAVE PREDICTIONS
    # -------------------------------
    # Create predictions dataframe
    predictions_df = pd.DataFrame({
        "predicted_label": predicted_labels,        # in training encoder space
        "true_label": predict_labels,               # original predict labels
        "prediction_confidence": np.max(prediction_probabilities, axis=1)
    })

    # Add probability columns for each class
    for i, class_name in enumerate(le.classes_):
        predictions_df[f"prob_{class_name}"] = prediction_probabilities[:, i]

    # Save predictions
    predictions_df.to_csv(f"../results_hippie_cross_dataset/train_{args.training_dataset}_predict_{args.predict_dataset}/config_{args.config}_zdim_{args.z_dim}_B_{args.beta}/predictions.csv", index=False)

    # -------------------------------
    # LOG ARTIFACTS TO WANDB
    # -------------------------------
    # Upload embeddings and predictions
    wandb.log_artifact(
        f"../results_hippie_cross_dataset/train_{args.training_dataset}_predict_{args.predict_dataset}/config_{args.config}_zdim_{args.z_dim}_B_{args.beta}/train_embeddings.csv",
        name=f"train_embeddings_{args.training_dataset}_config_{args.config}_zdim_{args.z_dim}_B_{args.beta}",
        type="embeddings"
    )

    wandb.log_artifact(
        f"../results_hippie_cross_dataset/train_{args.training_dataset}_predict_{args.predict_dataset}/config_{args.config}_zdim_{args.z_dim}_B_{args.beta}/predict_embeddings.csv",
        name=f"predict_embeddings_{args.predict_dataset}_config_{args.config}_zdim_{args.z_dim}_B_{args.beta}",
        type="embeddings"
    )

    wandb.log_artifact(
        f"../results_hippie_cross_dataset/train_{args.training_dataset}_predict_{args.predict_dataset}/config_{args.config}_zdim_{args.z_dim}_B_{args.beta}/predictions.csv",
        name=f"predictions_train_{args.training_dataset}_predict_{args.predict_dataset}_config_{args.config}_zdim_{args.z_dim}_B_{args.beta}",
        type="predictions"
    )

    # Upload model if requested
    if args.upload_model:
        wandb.log_artifact(joint_path, name=f'cross_dataset_model_train_{args.training_dataset}_predict_{args.predict_dataset}_z{args.z_dim}_lr{args.learning_rate}.pt', type='model')

    # Hyperparameters already logged during wandb.init()
    # Final log
    time_total = time_phase1 + time_phase2 + time_phase3 + time_embedding + time_knn

    print("\n" + "="*50)
    print("TIMING SUMMARY")
    print("="*50)
    print(f"Phase 1 (Pretraining):     {time_phase1:8.2f}s ({time_phase1/60:6.2f} min) - {100*time_phase1/time_total:5.1f}%")
    print(f"Phase 2 (Finetuning):      {time_phase2:8.2f}s ({time_phase2/60:6.2f} min) - {100*time_phase2/time_total:5.1f}%")
    print(f"Phase 3 (Supervised):      {time_phase3:8.2f}s ({time_phase3/60:6.2f} min) - {100*time_phase3/time_total:5.1f}%")
    print(f"Embedding Extraction:      {time_embedding:8.2f}s ({time_embedding/60:6.2f} min) - {100*time_embedding/time_total:5.1f}%")
    print(f"KNN Training/Evaluation:   {time_knn:8.2f}s ({time_knn/60:6.2f} min) - {100*time_knn/time_total:5.1f}%")
    print("-" * 50)
    print(f"TOTAL TIME:                {time_total:8.2f}s ({time_total/60:6.2f} min)")
    print("="*50 + "\n")

    wandb.log({
        "time/total_seconds": time_total,
        "time/total_minutes": time_total/60,
        "time/phase1_percentage": 100*time_phase1/time_total,
        "time/phase2_percentage": 100*time_phase2/time_total,
        "time/phase3_percentage": 100*time_phase3/time_total,
        "time/embedding_percentage": 100*time_embedding/time_total,
        "time/knn_percentage": 100*time_knn/time_total,
        "phase": "complete"
    })
    wandb.finish()

    print("="*50)
    print("CROSS-DATASET CLASSIFICATION COMPLETE")
    print("="*50)
    print(f"Training dataset: {args.training_dataset}")
    print(f"Predict dataset: {args.predict_dataset}")
    print(f"Training classes: {list(le.classes_)}")
    print(f"Best k for KNN: {best_neighbors}")
    print(f"Output directory: ../results_hippie_cross_dataset/train_{args.training_dataset}_predict_{args.predict_dataset}/config_{args.config}_zdim_{args.z_dim}_B_{args.beta}/")
    print("\nFiles generated:")
    print("- train_embeddings.csv: Embeddings for training dataset")
    print("- predict_embeddings.csv: Embeddings for predict dataset")
    print("- predictions.csv: KNN predictions on predict dataset")
    print("="*50)