"""Train a HIPPIE multimodal CVAE on one labeled or unlabeled dataset.

This is the minimal user-facing trainer: it loads one dataset folder
(canonical CSV layout), runs pretraining for ``--epochs`` epochs, and
writes a Lightning checkpoint. There is no held-out validation split,
no KNN/MLP head, no balanced accuracy, no W&B logging built in --
those live in the companion ``hippie_benchmarking_release`` repo.

To use the resulting checkpoint:

    # extract embeddings
    hippie-cli embed --checkpoint checkpoints/my_run.ckpt \\
        --datasets-root ./datasets_hippie --datasets my_data \\
        --output my_data_embeddings.npz

    # classify an NWB recording
    python hippie_nwb_classify.py session.nwb \\
        --checkpoint checkpoints/my_run.ckpt \\
        --train-embeddings labeled_reference_embeddings.csv \\
        --z-dim 30

Typical invocation:

    python scripts/train.py \\
        --dataset hausser_cell_type \\
        --data-dir ./datasets_hippie \\
        --output checkpoints/hausser.ckpt \\
        --epochs 100
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from hippie.augmentations import AugmentedMultiModalEphysDataset
from hippie.dataloading import MultiModalEphysDataset, none_safe_collate
from hippie.inference import TECHNOLOGY_IDS
from hippie.multimodal_model import (
    CVAEConfig,
    ExperimentConfigs,
    MultiModalCVAE,
    MultiModalCVAETrainModule,
)

logger = logging.getLogger(__name__)


def _load_dataset(data_dir: str, dataset_name: str):
    """Read the canonical 4-CSV dataset layout and return numpy arrays.

    Labels are encoded as integers via sklearn.LabelEncoder; if no labels
    file is present, every unit is assigned class 0 (unsupervised
    pretraining still works -- the encoder ignores class labels in the
    production config).
    """
    root = Path(data_dir) / dataset_name
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {root}")

    wf = pd.read_csv(root / "waveforms.csv").to_numpy().astype(np.float32)
    isi = pd.read_csv(root / "isi_dist.csv").to_numpy().astype(np.float32)
    acg_path = root / "acg.csv"
    if acg_path.exists():
        acg = pd.read_csv(acg_path).to_numpy().astype(np.float32)
    else:
        logger.warning("No acg.csv in %s -- using zeros (bimodal mode)", root)
        acg = np.zeros((len(wf), 100), dtype=np.float32)

    # Sanitize NaNs/Infs in source data -- the loaders also handle this,
    # but it is cleaner to do it once here.
    wf = np.nan_to_num(wf, nan=0.0, posinf=0.0, neginf=0.0)
    isi = np.nan_to_num(isi, nan=0.0, posinf=0.0, neginf=0.0)
    acg = np.nan_to_num(acg, nan=0.0, posinf=0.0, neginf=0.0)

    labels_path = root / "labels.csv"
    if not labels_path.exists():
        labels_path = root / "celltypes.csv"
    if labels_path.exists():
        labels_df = pd.read_csv(labels_path)
        raw_labels = labels_df.iloc[:, -1].astype(str).values
        le = LabelEncoder()
        labels = le.fit_transform(raw_labels)
    else:
        logger.warning("No labels.csv in %s -- training without supervision", root)
        labels = np.zeros(len(wf), dtype=np.int64)
        le = None

    return wf, isi, acg, labels, le


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", required=True, help="Dataset folder name under --data-dir")
    p.add_argument("--data-dir", default="./datasets_hippie",
                   help="Root directory containing dataset folders (default: ./datasets_hippie)")
    p.add_argument("--output", required=True, help="Output .ckpt path")
    p.add_argument("--config", default="class_decoder_source_bn_aug_reg",
                   choices=sorted(m for m in dir(ExperimentConfigs) if not m.startswith("_")),
                   help="ExperimentConfigs preset (default: production config)")
    p.add_argument("--z-dim", dest="z_dim", type=int, default=30)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--free-bits", dest="free_bits", type=float, default=0.5,
                   help="Floor in nats on each latent dim's KL, preventing dims from "
                        "collapsing onto the prior (default: 0.5). Pass 0 to disable "
                        "and reproduce the published objective exactly.")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=128)
    p.add_argument("--learning-rate", dest="learning_rate", type=float, default=1e-3)
    p.add_argument("--num-workers", dest="num_workers", type=int, default=0)
    p.add_argument("--accelerator", default=("gpu" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--devices", default=1, type=int)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--tech",
        default="neuropixels",
        choices=sorted(TECHNOLOGY_IDS.keys()),
        help="Recording-technology covariate for this dataset. Mapped to a "
             "source_id via hippie.inference.TECHNOLOGY_IDS so the trained "
             "model embeds the technology consistently with the public "
             "checkpoint's vocabulary.",
    )
    p.add_argument(
        "--num-sources",
        dest="num_sources",
        type=int,
        default=4,
        help="Size of the source-embedding table. Default 4 matches the "
             "vocabulary of the public HuggingFace checkpoint (slots 0-2 = "
             "neuropixels / silicon_probe / juxtacellular; slot 3 reserved). "
             "Pass --num-sources 1 for a single-technology fresh run.",
    )
    args = p.parse_args()

    if args.num_sources < 1:
        p.error("--num-sources must be >= 1")
    src_idx = TECHNOLOGY_IDS[args.tech]
    if src_idx >= args.num_sources:
        p.error(
            f"--tech {args.tech} maps to source_id {src_idx}, which is out of "
            f"range for --num-sources {args.num_sources}. Pass a larger "
            f"--num-sources or pick a tech with a smaller id."
        )

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    pl.seed_everything(args.seed, workers=True)

    wf, isi, acg, labels, le = _load_dataset(args.data_dir, args.dataset)
    n = len(wf)
    logger.info("Loaded %s: %d units, %d unique labels", args.dataset, n, len(set(labels)))

    # The dataset class expects labels as (class_idx, source_idx) pairs.
    # source_idx is set from --tech so the trained model carries the correct
    # recording-technology covariate (paper Methods § "HIPPIE training
    # details": "Recording technology covariates were included during
    # training").
    paired_labels = np.stack(
        [labels, np.full(n, src_idx, dtype=np.int64)], axis=1
    )
    logger.info(
        "Assigning every unit to source_id=%d (tech=%s); "
        "source-embedding table sized for %d technologies.",
        src_idx, args.tech, args.num_sources,
    )

    dataset = MultiModalEphysDataset(
        {"wave": wf, "isi": isi, "acg": acg},
        labels=paired_labels,
        mode="multi",
        normalize=True,
    )

    config: CVAEConfig = getattr(ExperimentConfigs, args.config)()
    config.beta = args.beta
    config.free_bits = args.free_bits

    # Wrap with the augmentation dataset when the config requests it.
    # Without this wrap, configs that set use_augmentations=True (notably the
    # production default class_decoder_source_bn_aug_reg) would silently train
    # without augmentations — drifting from the paper's reported architecture.
    if config.use_augmentations and config.augment_pretraining:
        dataset = AugmentedMultiModalEphysDataset(
            dataset, config, phase="pretraining"
        )
        logger.info(
            "Augmentations enabled (phase=pretraining, prob=%.2f, noise_std=%.3f)",
            config.augment_prob, config.noise_std,
        )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=none_safe_collate,
        drop_last=True,
    )

    num_classes = max(int(labels.max()) + 1, 1) if le is not None else 1

    model = MultiModalCVAE(
        modalities={"wave": 50, "isi": 100, "acg": 100},
        z_dim=args.z_dim,
        config=config,
        num_sources=args.num_sources,
        num_classes=num_classes,
    )
    train_module = MultiModalCVAETrainModule(
        base_model=model,
        config=config,
        learning_rate=args.learning_rate,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        default_root_dir=str(out_path.parent),
        enable_checkpointing=False,  # we write a single final checkpoint manually
        logger=False,                # benchmarking lives in hippie_benchmarking_release
        log_every_n_steps=10,
    )

    logger.info("Starting training (%d epochs, %s config)", args.epochs, args.config)
    trainer.fit(train_module, train_dataloaders=loader)

    trainer.save_checkpoint(str(out_path))
    logger.info("Saved checkpoint to %s", out_path)


if __name__ == "__main__":
    main()
