"""HIPPIE inference API — latent embeddings, UMAP projection, and HDBSCAN clustering."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Tuple, Union

import numpy as np
import torch


# ----------------------------------------------------------------------
# KNN k-selection (shared by hippie-cli predict and hippie_nwb_classify)
# ----------------------------------------------------------------------

def select_k_via_cv(
    reference_embeddings: np.ndarray,
    reference_labels: np.ndarray,
    k_candidates: Iterable[int] = range(1, 21),
    metric: str = "cosine",
    scoring: str = "balanced_accuracy",
) -> Tuple[int, dict]:
    """Pick the best k for cosine-metric KNN via stratified CV on the reference set.

    Mirrors the selection procedure used in
    ``hippie_benchmarking_release/scripts/cross_dataset_script.py``:
      * L2-normalize reference embeddings (no-op for cosine metric but kept
        for byte-for-byte compatibility with the benchmark recipe).
      * cv = min(5, n_classes).
      * scoring = "balanced_accuracy".
      * Try k = 1..20.
      * Return the k that maximizes mean CV score (ties broken by smaller k).

    Args:
        reference_embeddings: (N, z_dim) labeled reference embeddings.
        reference_labels:     (N,)       integer or string labels.
        k_candidates:         which neighbour counts to consider.
        metric:               sklearn KNN metric (default "cosine").
        scoring:              sklearn CV scoring (default "balanced_accuracy").

    Returns:
        best_k, cv_scores -- where cv_scores is a {k: mean_score} dict for
        the k values that were actually evaluated.

    Raises:
        ValueError: if there are fewer than 2 classes, or if every candidate
        k is larger than the per-fold training set size.
    """
    from sklearn.model_selection import cross_val_score
    from sklearn.neighbors import KNeighborsClassifier

    labels = np.asarray(reference_labels)
    n_samples = len(labels)
    classes, class_counts = np.unique(labels, return_counts=True)
    n_classes = len(classes)
    if n_classes < 2:
        raise ValueError(
            f"k-selection needs >= 2 classes in the reference set; got {n_classes}. "
            "Pass an explicit k (e.g. --k 5) to bypass CV selection."
        )

    cv = min(5, n_classes)
    # StratifiedKFold (default for classifiers) requires >= cv samples in
    # every class -- shrink cv to the smallest class size if needed.
    cv = max(2, min(cv, int(class_counts.min())))
    fold_train_size = n_samples - (n_samples // cv)

    emb = np.asarray(reference_embeddings, dtype=np.float64)
    emb_l2 = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)

    scores: dict = {}
    for k in k_candidates:
        if k > fold_train_size:
            # KNN needs k neighbours available in each CV training fold.
            continue
        knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
        cv_score = cross_val_score(knn, emb_l2, labels, cv=cv, scoring=scoring)
        scores[int(k)] = float(np.mean(cv_score))

    if not scores:
        raise ValueError(
            f"Reference set too small ({n_samples} samples, cv={cv}) to evaluate any "
            f"k in {list(k_candidates)}. Pass --k <int> to bypass CV selection."
        )

    # Ties broken by smaller k (max() with a key returning the score is stable;
    # iteration is in insertion order so the first/smallest k wins on ties).
    best_k = max(scores, key=lambda k: (scores[k], -k))
    return best_k, scores


# Recording-technology IDs that received gradients during pretraining.
# The released checkpoint's source-embedding table has 4 rows (one was
# reserved for tetrode data that was never added to the pretraining
# corpus). Only the three classes below are usable at inference time;
# row 3 of the embedding matrix is still at initialization.
TECHNOLOGY_IDS: dict[str, int] = {
    "neuropixels": 0,
    "silicon_probe": 1,
    "juxtacellular": 2,
}


class HIPPIEClassifier:
    """Pretrained HIPPIE encoder for embedding, UMAP visualization, and HDBSCAN clustering."""

    def __init__(self, model, device: str = "cpu") -> None:
        self.model = model.to(device).eval()
        self.device = device

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str = "Jesusgf23/hippie",
        filename: str = "hippie_techcond_v1.ckpt",
        device: str = "cpu",
        cache_dir: Optional[str] = None,
    ) -> "HIPPIEClassifier":
        """Download and load the pretrained HIPPIE model from HuggingFace Hub.

        Args:
            repo_id: HuggingFace repository ID.
            filename: Checkpoint filename inside the repo.
            device: "cuda" or "cpu".
            cache_dir: Local directory to cache the downloaded checkpoint.
        """
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError(
                "huggingface_hub is required. Install with: pip install huggingface-hub"
            )

        from .checkpoint import build_model

        ckpt_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=cache_dir,
        )
        model = build_model(ckpt_path)
        return cls(model, device=device)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        device: str = "cpu",
    ) -> "HIPPIEClassifier":
        """Load from a local Lightning checkpoint file.

        Args:
            checkpoint_path: Path to a .ckpt file.
            device: "cuda" or "cpu".
        """
        from .checkpoint import build_model

        model = build_model(str(checkpoint_path))
        return cls(model, device=device)

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_embeddings(
        self,
        wave: np.ndarray,
        isi: np.ndarray,
        acg: np.ndarray,
        tech_id: Union[int, str] = 0,
        batch_size: int = 256,
    ) -> np.ndarray:
        """Encode neurons and return latent z_mean embeddings.

        Inputs must be preprocessed (see the README for the expected normalization):
          - wave: min-max normalized to [-1, 1], resampled to 50 points
          - isi:  log(x+1) transformed, min-max normalized to [-1, 1], 100 bins
          - acg:  min-max normalized to [-1, 1], 100 bins

        Args:
            wave: (N, 50) waveform array.
            isi:  (N, 100) ISI histogram array.
            acg:  (N, 100) autocorrelogram array.
            tech_id: Recording technology index (int) or name from TECHNOLOGY_IDS.
            batch_size: Neurons per forward pass.

        Returns:
            embeddings: (N, z_dim) float32 array of latent means.
        """
        if isinstance(tech_id, str):
            if tech_id not in TECHNOLOGY_IDS:
                raise ValueError(
                    f"Unknown tech_id '{tech_id}'. Choose from: {list(TECHNOLOGY_IDS)}"
                )
            tech_id = TECHNOLOGY_IDS[tech_id]

        N = len(wave)
        all_mu = []

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            wave_t = torch.from_numpy(wave[start:end]).float().unsqueeze(1).to(self.device)
            isi_t = torch.from_numpy(isi[start:end]).float().unsqueeze(1).to(self.device)
            acg_t = torch.from_numpy(acg[start:end]).float().unsqueeze(1).to(self.device)
            src_t = torch.full((end - start,), tech_id, dtype=torch.long, device=self.device)

            _, mu, _ = self.model.encode(
                {"wave": wave_t, "isi": isi_t, "acg": acg_t},
                source_labels=src_t,
                apply_dropout=False,
            )
            all_mu.append(mu.cpu().numpy())

        return np.concatenate(all_mu, axis=0)

    # ------------------------------------------------------------------
    # UMAP
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
        """Project embeddings to low-dimensional space with UMAP.

        Args:
            embeddings: (N, z_dim) array.
            n_components: Output dimensionality (2 for visualization).
            n_neighbors: UMAP neighbourhood size.
            min_dist: Minimum distance between embedded points.
            metric: Distance metric.
            random_state: Reproducibility seed.

        Returns:
            coords: (N, n_components) float32 UMAP coordinates.
        """
        try:
            from umap import UMAP
        except ImportError:
            raise ImportError(
                "umap-learn is required. Install with: pip install umap-learn"
            )

        reducer = UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
        )
        return reducer.fit_transform(embeddings).astype(np.float32)

    # ------------------------------------------------------------------
    # HDBSCAN
    # ------------------------------------------------------------------

    @staticmethod
    def hdbscan_cluster(
        embeddings: np.ndarray,
        min_cluster_size: int = 5,
        min_samples: Optional[int] = None,
        metric: str = "euclidean",
    ) -> np.ndarray:
        """Cluster embeddings with HDBSCAN.

        Args:
            embeddings: (N, D) array — full latent embeddings or UMAP coords.
            min_cluster_size: Minimum neurons to form a cluster.
            min_samples: Core-distance sample requirement (defaults to min_cluster_size).
            metric: Distance metric.

        Returns:
            labels: (N,) int32 array. -1 = noise / unclustered.
        """
        try:
            import hdbscan
        except ImportError:
            raise ImportError(
                "hdbscan is required. Install with: pip install hdbscan"
            )

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric,
        )
        return clusterer.fit_predict(embeddings).astype(np.int32)
