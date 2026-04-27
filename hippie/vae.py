"""Unsupervised multimodal VAE — training and compression without conditioning.

Uses the same ResNet18 + fusion encoder architecture as the pretrained HIPPIE
model but with all class and technology conditioning removed, so the latent
space is shaped purely by waveform + ISI + autocorrelogram reconstruction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split


class VAECompressor:
    """Trained unconditioned VAE encoder for data compression.

    Unlike HIPPIEClassifier, no tech_id is required — the model was
    trained without any conditioning signal.
    """

    def __init__(self, model, device: str = "cpu") -> None:
        self.model = model.to(device).eval()
        self.device = device

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        device: str = "cpu",
        backbone_base_width: int = 64,
    ) -> "VAECompressor":
        """Load from a checkpoint saved by train_vae().

        Args:
            checkpoint_path: Path to a .ckpt file produced by train_vae().
            device: "cuda" or "cpu".
            backbone_base_width: Must match the value used during training (default 64).
        """
        from .checkpoint import build_unconditioned_model

        model = build_unconditioned_model(str(checkpoint_path), backbone_base_width)
        return cls(model, device=device)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_embeddings(
        self,
        wave: np.ndarray,
        isi: np.ndarray,
        acg: np.ndarray,
        batch_size: int = 256,
    ) -> np.ndarray:
        """Encode neurons into the latent space.

        Inputs must already be preprocessed (same normalization as
        hippie_adapter.extract_features):
          - wave: (N, 50)  min-max → [-1, 1]
          - isi:  (N, 100) log(x+1) → min-max → [-1, 1]
          - acg:  (N, 100) min-max → [-1, 1]

        Returns:
            embeddings: (N, z_dim) float32 array of latent z_mean vectors.
        """
        N = len(wave)
        all_mu = []

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            wave_t = torch.from_numpy(wave[start:end]).float().unsqueeze(1).to(self.device)
            isi_t  = torch.from_numpy(isi[start:end]).float().unsqueeze(1).to(self.device)
            acg_t  = torch.from_numpy(acg[start:end]).float().unsqueeze(1).to(self.device)

            _, mu, _ = self.model.encode({"wave": wave_t, "isi": isi_t, "acg": acg_t})
            all_mu.append(mu.cpu().numpy())

        return np.concatenate(all_mu, axis=0)

    # ------------------------------------------------------------------
    # Downstream: UMAP + HDBSCAN (same helpers as HIPPIEClassifier)
    # ------------------------------------------------------------------

    @staticmethod
    def umap_reduce(
        embeddings: np.ndarray,
        n_components: int = 2,
        n_neighbors: int = 30,
        min_dist: float = 0.1,
        metric: str = "cosine",
        random_state: int = 42,
    ) -> np.ndarray:
        """Project embeddings to low-dimensional space with UMAP."""
        try:
            from umap import UMAP
        except ImportError:
            raise ImportError("umap-learn is required. Install with: pip install umap-learn")

        return UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
        ).fit_transform(embeddings).astype(np.float32)

    @staticmethod
    def hdbscan_cluster(
        embeddings: np.ndarray,
        min_cluster_size: int = 5,
        min_samples: Optional[int] = None,
        metric: str = "euclidean",
    ) -> np.ndarray:
        """Cluster embeddings with HDBSCAN. Returns (N,) int32; -1 = noise."""
        try:
            import hdbscan
        except ImportError:
            raise ImportError("hdbscan is required. Install with: pip install hdbscan")

        return hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric,
        ).fit_predict(embeddings).astype(np.int32)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_vae(
    wave: np.ndarray,
    isi: np.ndarray,
    acg: np.ndarray,
    output_dir: str,
    z_dim: int = 30,
    n_epochs: int = 100,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-2,
    val_fraction: float = 0.1,
    backbone_base_width: int = 64,
    device: str = "cpu",
    random_state: int = 42,
) -> VAECompressor:
    """Train an unconditioned multimodal VAE on preprocessed electrophysiology features.

    The model uses the same ResNet18 + fusion encoder as the pretrained HIPPIE
    checkpoint but with no class or technology conditioning.  Loss is the
    standard beta-VAE ELBO: MSE reconstruction + KL divergence (beta=1).

    Inputs must be preprocessed by hippie_adapter.extract_features() or the
    equivalent normalization (wave: min-max [-1,1]; isi: log1p then min-max;
    acg: min-max [-1,1]).

    Args:
        wave: (N, 50) preprocessed waveform array.
        isi:  (N, 100) preprocessed ISI histogram array.
        acg:  (N, 100) preprocessed autocorrelogram array.
        output_dir: Directory to save the best checkpoint.
        z_dim: Latent space dimensionality (default 30, matching pretrained model).
        n_epochs: Number of training epochs.
        batch_size: Minibatch size.
        learning_rate: AdamW learning rate.
        weight_decay: AdamW weight decay.
        val_fraction: Fraction of data held out for validation.
        backbone_base_width: ResNet18 base channel width (default 64).
        device: "cuda" or "cpu".
        random_state: Random seed for train/val split.

    Returns:
        VAECompressor loaded from the best checkpoint.
    """
    try:
        import pytorch_lightning as pl
    except ImportError:
        raise ImportError(
            "pytorch_lightning is required for training. "
            "Install with: pip install pytorch-lightning"
        )

    from .multimodal_model import MultiModalCVAE, MultiModalCVAETrainModule, ExperimentConfigs

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Dataset — tensors are already normalised, add channel dim (B,1,L)
    # ------------------------------------------------------------------
    torch.manual_seed(random_state)
    N = len(wave)

    wave_t = torch.from_numpy(wave).float().unsqueeze(1)  # (N, 1, 50)
    isi_t  = torch.from_numpy(isi).float().unsqueeze(1)   # (N, 1, 100)
    acg_t  = torch.from_numpy(acg).float().unsqueeze(1)   # (N, 1, 100)
    # Dummy labels — 1-D so _forward_model treats them as source_labels,
    # which _get_embeddings ignores when use_source_embedding=False.
    labels_t = torch.zeros(N, dtype=torch.long)

    full_dataset = TensorDataset(wave_t, isi_t, acg_t, labels_t)

    n_val   = max(1, int(N * val_fraction))
    n_train = N - n_val
    train_ds, val_ds = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(random_state),
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, drop_last=(n_train > batch_size),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    config = ExperimentConfigs.unconditioned()

    base_model = MultiModalCVAE(
        modalities={"wave": 50, "isi": 100, "acg": 100},
        z_dim=z_dim,
        num_sources=None,
        num_classes=None,
        num_super_regions=0,
        num_layers=0,
        config=config,
        backbone_base_width=backbone_base_width,
    )

    train_module = MultiModalCVAETrainModule(
        base_model=base_model,
        config=config,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    ckpt_callback = pl.callbacks.ModelCheckpoint(
        dirpath=str(output_dir),
        filename="vae_best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )

    trainer = pl.Trainer(
        max_epochs=n_epochs,
        accelerator="gpu" if device == "cuda" and torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[ckpt_callback],
        enable_progress_bar=True,
        enable_model_summary=False,
        logger=False,
    )

    trainer.fit(train_module, train_loader, val_loader)

    best_ckpt = ckpt_callback.best_model_path or str(output_dir / "vae_best.ckpt")
    return VAECompressor.from_checkpoint(best_ckpt, device=device, backbone_base_width=backbone_base_width)
