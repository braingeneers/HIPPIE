#!/usr/bin/env python3
"""
hippie_nwb_classify.py
======================
End-to-end pipeline: NWB file  →  HIPPIE embeddings  →  classified neurons.

Two input paths are supported (auto-detected unless overridden with --path):

  Path A – precomputed / pre-sorted  (NWB has a ``units`` table with spike_times)
    The pipeline reads waveforms from the units table.  When waveforms are
    stored per-channel, the channel with the largest peak-to-peak amplitude is
    kept.  If the units table contains only spike times (no waveform column),
    a zero waveform is used as a placeholder and a warning is printed.
    ISI distributions and autocorrelograms are computed from the spike times.

  Path B – raw traces  (NWB has no sorted units, only raw ElectricalSeries)
    The pipeline spike-sorts the raw traces via SpikeInterface
    (default sorter: tridesclous2 — lightweight, no GPU, pure Python).
    Any other SpikeInterface-compatible sorter can be selected with --sorter.
    After sorting, the mean waveform, ISI and ACG are computed exactly as in
    Path A.

Usage examples
--------------
# Path A – precomputed units
python hippie_nwb_classify.py session.nwb \\
    --checkpoint path/to/best.ckpt \\
    --train-embeddings path/to/train_embeddings.csv \\
    --config class_decoder_source_bn_aug_reg \\
    --z-dim 20 --num-sources 4 --num-classes 5

# Path B – raw traces, automatic sorter choice
python hippie_nwb_classify.py session.nwb \\
    --path raw \\
    --checkpoint path/to/best.ckpt \\
    --train-embeddings path/to/train_embeddings.csv \\
    --config class_decoder_source_bn_aug_reg \\
    --z-dim 20 --num-sources 4 --num-classes 5

# Feature-extraction only (no model / no checkpoint required)
python hippie_nwb_classify.py session.nwb --features-only --output session_features.csv

Notes
-----
- The checkpoint can be the public pretrained HIPPIE checkpoint from
  HuggingFace Hub (downloaded automatically by hippie.inference) or one
  produced by `python scripts/train.py`. The training-set embeddings
  CSV is the labeled reference set you want to classify against (see
  `hippie-cli embed` to generate it).
- --num-sources must match the value used during training (it determines the
  source_embedding table size; a mismatch will raise an error on load).
- --source-id controls which source embedding is injected at inference time.
  Use 0 to inject zeros (safe default for an unseen recording site).
"""

import sys
import os
import argparse
import warnings
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import torch
from torch.utils.data import DataLoader
from hippie.dataloading import MultiModalEphysDataset, none_safe_collate
from hippie.multimodal_model import (
    CVAEConfig,
    ExperimentConfigs,
    MultiModalCVAE,
    MultiModalCVAETrainModule,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  AUTO-DETECT INPUT PATH
# ═══════════════════════════════════════════════════════════════════════════════

def detect_nwb_path(nwb_path: str) -> str:
    """Return 'precomputed' if the file has a units table with spike times,
    'raw' otherwise."""
    from pynwb import NWBHDF5IO

    with NWBHDF5IO(nwb_path, "r", load_namespaces=True) as io:
        nwb = io.read()
        has_units = (
            nwb.units is not None
            and len(nwb.units.id.data) > 0
            and "spike_times" in (nwb.units.colnames or [])
        )
    return "precomputed" if has_units else "raw"


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  PATH A – PRECOMPUTED / PRE-SORTED UNITS
# ═══════════════════════════════════════════════════════════════════════════════

def _max_amp_channel(wf: np.ndarray) -> np.ndarray:
    """Given (n_channels, n_timepoints), return the max peak-to-peak channel."""
    wf = np.asarray(wf, float)
    if wf.ndim == 1:
        return wf
    return wf[int(np.argmax(wf.ptp(axis=1)))]


def _cut_to_trough(wf1d: np.ndarray, n: int = 50) -> np.ndarray:
    """Centre-cut n points around the trough; zero-pad if waveform too short."""
    wf1d = np.asarray(wf1d, float).flatten()
    if wf1d.size == 0:
        return np.zeros(n)
    mid = int(np.argmin(wf1d))
    left = int(n * 2 / 5)
    lo = max(0, mid - left)
    hi = min(wf1d.size, lo + n)
    seg = wf1d[lo:hi]
    if seg.size < n:
        seg = np.pad(seg, (0, n - seg.size))
    return seg


def load_precomputed_path(
    nwb_path: str,
    n_wf_points: int = 50,
) -> tuple:
    """Load spike times, waveforms, and unit IDs from a pre-sorted NWB file.

    Handles waveform columns:
      - ``waveform_mean``   – per-unit mean; shape (n_tp,) or (n_ch, n_tp)
      - ``spike_waveforms`` – per-spike ragged; shape (n_sp, n_tp) or (n_sp, n_ch, n_tp)
      - No waveform column  – zeros + warning

    Returns
    -------
    spike_times_ms : list[np.ndarray]   spike times in milliseconds
    waveforms      : list[np.ndarray]   each (n_wf_points,)
    unit_ids       : list[str]
    """
    from pynwb import NWBHDF5IO

    spike_times_ms, waveforms, unit_ids = [], [], []

    with NWBHDF5IO(nwb_path, "r", load_namespaces=True) as io:
        nwb = io.read()
        units = nwb.units
        n_units = len(units.id.data)
        cols = set(units.colnames or [])

        wf_key = next((k for k in ("waveform_mean", "spike_waveforms") if k in cols), None)
        if wf_key is None:
            warnings.warn(
                "No waveform column found in units table "
                "(tried 'waveform_mean', 'spike_waveforms'). "
                "Waveform features will be zero — classification may be degraded.",
                stacklevel=2,
            )

        for i in range(n_units):
            unit_ids.append(str(units.id.data[i]))

            # spike times (seconds → milliseconds)
            spike_times_ms.append(
                np.asarray(units.get_unit_spike_times(i), float) * 1000.0
            )

            # waveform
            if wf_key is None:
                waveforms.append(np.zeros(n_wf_points))
                continue

            if wf_key == "spike_waveforms":
                raw = units.get_unit_spike_waveforms(i)
                if raw is None or len(raw) == 0:
                    waveforms.append(np.zeros(n_wf_points))
                    continue
                raw = np.asarray(raw, float)
                # (n_spikes, n_tp) or (n_spikes, n_ch, n_tp)
                mean_wf = raw.mean(axis=0)
                if mean_wf.ndim == 2:          # (n_ch, n_tp) after mean
                    mean_wf = _max_amp_channel(mean_wf)
            else:  # waveform_mean
                raw = np.asarray(units[wf_key][i], float)
                mean_wf = _max_amp_channel(raw) if raw.ndim == 2 else raw

            waveforms.append(_cut_to_trough(mean_wf, n_wf_points))

    return spike_times_ms, waveforms, unit_ids


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  PATH B – SPIKE SORTING FROM RAW TRACES
# ═══════════════════════════════════════════════════════════════════════════════

def load_raw_path(
    nwb_path: str,
    sorter: str = "tridesclous2",
    sort_output_dir: str = "sort_output",
    n_wf_points: int = 50,
    min_spikes: int = 20,
    n_jobs: int = -1,
) -> tuple:
    """Spike-sort raw ElectricalSeries from NWB via SpikeInterface.

    Parameters
    ----------
    nwb_path        : Path to NWB file.
    sorter          : SpikeInterface sorter name.
                      Lightweight, no-GPU options: 'tridesclous2', 'mountainsort5'.
    sort_output_dir : Directory for sorting intermediate files.
    n_wf_points     : Target waveform length.
    min_spikes      : Minimum spikes per unit to keep.
    n_jobs          : Workers for waveform extraction.

    Returns
    -------
    spike_times_ms, waveforms, unit_ids  (same contract as load_precomputed_path)
    """
    try:
        import spikeinterface as si
        import spikeinterface.extractors as se
        import spikeinterface.sorters as ss
        import spikeinterface.preprocessing as sp
    except ImportError:
        raise ImportError(
            "SpikeInterface is required for Path B.\n"
            "Install:  pip install spikeinterface[full]"
        )

    print(f"[Path B] Loading raw traces from {nwb_path}")
    recording = se.NwbRecordingExtractor(nwb_path)
    fs = recording.get_sampling_frequency()
    print(f"[Path B] Sampling rate: {fs:.0f} Hz, "
          f"{recording.get_num_channels()} channels, "
          f"{recording.get_num_frames() / fs:.1f} s")

    # Standard preprocessing: bandpass + common median reference
    rec = sp.bandpass_filter(recording, freq_min=300.0, freq_max=6000.0)
    rec = sp.common_reference(rec, reference="global", operator="median")

    print(f"[Path B] Running sorter: {sorter}")
    sorting = ss.run_sorter(
        sorter,
        rec,
        output_folder=sort_output_dir,
        remove_existing_folder=True,
        verbose=True,
    )

    # Quality filter: keep units with enough spikes
    keep = [
        u for u in sorting.unit_ids
        if len(sorting.get_unit_spike_train(u, segment_index=0)) >= min_spikes
    ]
    if not keep:
        raise RuntimeError(
            f"No units with ≥{min_spikes} spikes after sorting. "
            "Try a different sorter or check your recording quality."
        )
    sorting = sorting.select_units(keep)
    print(f"[Path B] {len(sorting.unit_ids)} units kept (≥{min_spikes} spikes).")

    # Extract spike waveforms to get templates
    wf_ext = si.extract_waveforms(
        rec,
        sorting,
        folder=os.path.join(sort_output_dir, "waveforms"),
        overwrite=True,
        ms_before=1.0,
        ms_after=2.0,
        max_spikes_per_unit=500,
        n_jobs=n_jobs,
    )
    templates = wf_ext.get_all_templates()  # (n_units, n_timepoints, n_channels)

    spike_times_ms, waveforms, unit_ids = [], [], []

    for idx, uid in enumerate(sorting.unit_ids):
        unit_ids.append(str(uid))

        train_samp = sorting.get_unit_spike_train(uid, segment_index=0)
        spike_times_ms.append(train_samp.astype(float) / fs * 1000.0)

        # template: (n_tp, n_ch) → pick max peak-to-peak channel
        tmpl = templates[idx]                       # (n_tp, n_ch)
        best_ch = int(np.argmax(tmpl.ptp(axis=0)))
        waveforms.append(_cut_to_trough(tmpl[:, best_ch], n_wf_points))

    return spike_times_ms, waveforms, unit_ids


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  FEATURE EXTRACTION  (ISI + ACG)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_isi_distributions(
    spike_times_ms: list,
    window_ms: float = 100.0,
    n_bins: int = 100,
) -> np.ndarray:
    """Return (n_units, n_bins) ISI histogram, 1 ms bins over [0, window_ms)."""

    def _single(st):
        st = np.sort(np.asarray(st, float).flatten())
        if st.size < 2:
            return np.zeros(n_bins)
        isi = np.diff(st)
        hist, _ = np.histogram(isi[isi < window_ms], bins=n_bins,
                               range=(0.0, window_ms))
        return hist.astype(float)

    return np.array(
        Parallel(n_jobs=-1)(delayed(_single)(st) for st in spike_times_ms)
    )


def compute_autocorrelograms(
    spike_times_ms: list,
    bin_ms: float = 1.0,
    window_ms: float = 100.0,
) -> np.ndarray:
    """Return (n_units, 2*window_ms/bin_ms) ACG array (±window_ms, central bin removed)."""
    bin_edges = np.arange(-window_ms, window_ms + bin_ms, bin_ms)
    n_bins = len(bin_edges) - 1

    def _single(st):
        st = np.sort(np.asarray(st, float).flatten())
        if st.size < 2:
            return np.zeros(n_bins)
        diffs = []
        for i in range(len(st)):
            hi = np.searchsorted(st, st[i] + window_ms, side="right")
            if hi > i + 1:
                d = st[i + 1:hi] - st[i]
                diffs.extend(d)
                diffs.extend(-d)
        if not diffs:
            return np.zeros(n_bins)
        diffs = np.asarray(diffs)
        diffs = diffs[np.abs(diffs) > 1e-10]
        counts, _ = np.histogram(diffs, bins=bin_edges)
        return counts.astype(float)

    return np.array(
        Parallel(n_jobs=-1)(delayed(_single)(st) for st in spike_times_ms)
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════════

#: Modality sizes must match what the checkpoint was trained with.
MODALITIES = {"wave": 50, "isi": 100, "acg": 100}


def load_model(
    checkpoint_path: str,
    config_name: str,
    z_dim: int,
    num_sources: int,
    num_classes: int,
    device: torch.device,
) -> MultiModalCVAETrainModule:
    """Reconstruct a MultiModalCVAETrainModule and load weights from a checkpoint.

    Parameters
    ----------
    checkpoint_path : Path to a PyTorch Lightning .ckpt file.
    config_name     : Name of an ExperimentConfigs static method, e.g.
                      'class_decoder_source_bn_aug_reg'.
    z_dim           : Latent dimensionality (must match training run).
    num_sources     : Number of source datasets (sets embedding table size).
    num_classes     : Number of cell-type classes (sets embedding table size).
    device          : torch.device to load the model onto.
    """
    config_fn = getattr(ExperimentConfigs, config_name, None)
    if config_fn is None:
        available = [m for m in dir(ExperimentConfigs) if not m.startswith("_")]
        raise ValueError(
            f"Unknown config '{config_name}'.\n"
            f"Available: {available}"
        )
    config: CVAEConfig = config_fn()

    base_model = MultiModalCVAE(
        modalities=MODALITIES,
        z_dim=z_dim,
        config=config,
        num_sources=num_sources,
        num_classes=num_classes,
    )
    module = MultiModalCVAETrainModule(base_model=base_model, config=config)

    ckpt = torch.load(checkpoint_path, map_location=device)
    try:
        module.load_state_dict(ckpt["state_dict"])
    except RuntimeError as e:
        raise RuntimeError(
            f"Checkpoint loading failed — this usually means --config, "
            f"--z-dim, --num-sources, or --num-classes does not match the "
            f"training run.\n\nOriginal error: {e}"
        ) from e

    module.eval()
    module.to(device)
    print(f"Model loaded from {checkpoint_path}  (config={config_name}, z_dim={z_dim})")
    return module


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  EMBEDDING EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def extract_embeddings(
    waveforms: np.ndarray,
    isi: np.ndarray,
    acg: np.ndarray,
    module: MultiModalCVAETrainModule,
    source_id: int,
    device: torch.device,
    batch_size: int = 128,
) -> np.ndarray:
    """Run the HIPPIE encoder and return (n_units, z_dim) embeddings.

    Class labels are always masked at inference time — this is correct for all
    configs (the new decoder-only configs were designed precisely so the encoder
    never relies on them; symmetric configs use class_labels=None as a safe
    test-time path).

    Embeddings are row-wise z-normalised to match the normalisation used
    during training (the same transform applied by hippie.inference).
    """
    n = len(waveforms)
    # Source ID is broadcast to every unit; class label = 0 (masked below anyway)
    labels = np.stack(
        [np.zeros(n, dtype=int), np.full(n, source_id, dtype=int)], axis=1
    )

    dataset = MultiModalEphysDataset(
        {"wave": waveforms, "isi": isi, "acg": acg},
        labels=labels,
        mode="multi",
        modality_sizes=MODALITIES,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=none_safe_collate, num_workers=0,
    )

    underlying = module.model if hasattr(module, "model") else module
    all_mu = []

    with torch.no_grad():
        for data_dict, lbl in loader:
            data_dict = {k: v.to(device) for k, v in data_dict.items()}
            lbl = lbl.to(device)
            source_labels = lbl[:, 1]

            _, mu, _ = underlying.encode(
                data_dict,
                source_labels=source_labels,
                class_labels=None,   # always masked at inference
            )
            emb = mu.cpu().numpy()
            # row-wise z-normalise
            std = emb.std(axis=1, keepdims=True)
            std[std == 0] = 1.0
            emb = (emb - emb.mean(axis=1, keepdims=True)) / std
            all_mu.append(emb)

    return np.vstack(all_mu)


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  KNN CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

def knn_classify(
    train_embeddings_csv: str,
    query_embeddings: np.ndarray,
    k: Union[int, str] = 10,
) -> tuple:
    """KNN classification against pre-saved training-set embeddings.

    Parameters
    ----------
    train_embeddings_csv : CSV of labeled reference embeddings. Must have
        a ``label`` column; all other columns are treated as embedding
        dimensions. Convert a `hippie-cli embed` .npz to this format by
        loading `embeddings` and `labels` and saving as CSV.
    query_embeddings     : (n_query, z_dim) float array.
    k                    : Number of neighbours. Default 10. Pass "auto" to
        select k via 5-fold cross-validation on the reference set with
        balanced-accuracy scoring over k=1..20 -- matching the procedure
        in hippie_benchmarking_release.

    Returns
    -------
    predicted   : (n_query,)          string array of predicted class names.
    probas      : (n_query, n_classes) float array of per-class probabilities.
    classes     : (n_classes,)        string array of class labels (column order of `probas`).
    k_used      : int                  the neighbour count actually used.
    """
    from sklearn.neighbors import KNeighborsClassifier

    from hippie.inference import select_k_via_cv

    df = pd.read_csv(train_embeddings_csv)
    label_col = "label"
    if label_col not in df.columns:
        raise ValueError(
            f"Column 'label' not found in {train_embeddings_csv}. "
            f"Available columns: {list(df.columns)}"
        )
    train_labels = df[label_col].values
    emb_cols = [c for c in df.columns if c != label_col]
    train_emb = df[emb_cols].values

    if train_emb.shape[1] != query_embeddings.shape[1]:
        raise ValueError(
            f"Embedding dimension mismatch: training has {train_emb.shape[1]} dims "
            f"but query has {query_embeddings.shape[1]}. "
            f"Check that --z-dim matches the training run."
        )

    if k == "auto":
        k, scores = select_k_via_cv(train_emb, train_labels)
        print(f"Selected best k = {k} via 5-fold CV on the reference set "
              f"(balanced_acc = {scores[k]:.4f}).")
    else:
        k = int(k)
        print(f"Using k = {k} (pass -k auto to enable CV-based selection).")

    knn = KNeighborsClassifier(n_neighbors=k, metric="cosine")
    knn.fit(train_emb, train_labels)

    predicted = knn.predict(query_embeddings)
    probas = knn.predict_proba(query_embeddings)
    classes = knn.classes_

    return predicted, probas, classes, k


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(args: argparse.Namespace) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── 1. detect / validate input path ────────────────────────────────────
    path = args.path or detect_nwb_path(args.nwb)
    print(f"Input path: {path}  ({args.nwb})")

    # ── 2. load features ────────────────────────────────────────────────────
    if path == "precomputed":
        spike_times_ms, waveforms_raw, unit_ids = load_precomputed_path(args.nwb)
    else:
        spike_times_ms, waveforms_raw, unit_ids = load_raw_path(
            args.nwb,
            sorter=args.sorter,
            sort_output_dir=args.sort_output_dir,
            min_spikes=args.min_spikes,
        )

    n_units = len(unit_ids)
    print(f"Units: {n_units}")

    waveforms = np.array(waveforms_raw)              # (n_units, 50)
    isi       = compute_isi_distributions(spike_times_ms)  # (n_units, 100)
    acg       = compute_autocorrelograms(spike_times_ms)   # (n_units, 200)
    print(f"Features: waveforms {waveforms.shape}, ISI {isi.shape}, ACG {acg.shape}")

    # ── 3. features-only mode ───────────────────────────────────────────────
    if args.features_only:
        wf_df  = pd.DataFrame(waveforms,  index=unit_ids)
        isi_df = pd.DataFrame(isi,        index=unit_ids)
        acg_df = pd.DataFrame(acg,        index=unit_ids)

        stem = Path(args.output).stem
        parent = Path(args.output).parent
        wf_df.to_csv(parent / f"{stem}_waveforms.csv")
        isi_df.to_csv(parent / f"{stem}_isi.csv")
        acg_df.to_csv(parent / f"{stem}_acg.csv")
        print(f"Features saved to {parent}/{stem}_{{waveforms,isi,acg}}.csv")
        return pd.concat([wf_df, isi_df, acg_df], axis=1)

    # ── 4. model inference ──────────────────────────────────────────────────
    num_classes = (
        len(args.class_names) if args.class_names else args.num_classes
    )

    module = load_model(
        checkpoint_path=args.checkpoint,
        config_name=args.config,
        z_dim=args.z_dim,
        num_sources=args.num_sources,
        num_classes=num_classes,
        device=device,
    )

    embeddings = extract_embeddings(
        waveforms, isi, acg, module,
        source_id=args.source_id,
        device=device,
    )
    print(f"Embeddings: {embeddings.shape}")

    # ── 5. KNN classification ───────────────────────────────────────────────
    predicted, probas, classes, k_used = knn_classify(
        args.train_embeddings,
        embeddings,
        k=args.k,
    )
    confidence = probas.max(axis=1)

    # ── 6. assemble output ──────────────────────────────────────────────────
    n_spikes = [len(st) for st in spike_times_ms]
    firing_rates = [
        len(st) / ((st[-1] - st[0]) / 1000.0) if len(st) > 1 else 0.0
        for st in spike_times_ms
    ]

    out = pd.DataFrame({
        "unit_id":          unit_ids,
        "predicted_class":  predicted,
        "knn_confidence":   np.round(confidence, 4),
        "k_used":           k_used,
        "n_spikes":         n_spikes,
        "firing_rate_hz":   np.round(firing_rates, 3),
    })
    # Per-class probability columns ("confidence matrix").
    for j, cls in enumerate(classes):
        out[f"prob_{cls}"] = np.round(probas[:, j], 4)
    for dim in range(embeddings.shape[1]):
        out[f"emb_{dim}"] = np.round(embeddings[:, dim], 6)

    out.to_csv(args.output, index=False)
    print(f"\nResults → {args.output}")
    print(out[["unit_id", "predicted_class", "knn_confidence",
               "n_spikes", "firing_rate_hz"]].to_string(index=False))

    return out


# ═══════════════════════════════════════════════════════════════════════════════
# 9.  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hippie_nwb_classify.py",
        description="HIPPIE: NWB file → classified neurons.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── positional ──────────────────────────────────────────────────────────
    p.add_argument("nwb", help="Path to the input NWB file.")

    # ── input path ──────────────────────────────────────────────────────────
    g_path = p.add_argument_group("Input path")
    g_path.add_argument(
        "--path", choices=["precomputed", "raw"], default=None,
        help="Force a specific input path (auto-detected if omitted).",
    )

    # ── spike sorting (Path B) ───────────────────────────────────────────────
    g_sort = p.add_argument_group("Spike sorting (Path B only)")
    g_sort.add_argument(
        "--sorter", default="tridesclous2",
        help="SpikeInterface sorter name (default: tridesclous2). "
             "Other lightweight options: mountainsort5.",
    )
    g_sort.add_argument(
        "--sort-output-dir", dest="sort_output_dir", default="sort_output",
        help="Directory for sorting intermediate files (default: sort_output).",
    )
    g_sort.add_argument(
        "--min-spikes", dest="min_spikes", type=int, default=20,
        help="Minimum spike count to keep a sorted unit (default: 20).",
    )

    # ── model ────────────────────────────────────────────────────────────────
    g_model = p.add_argument_group("Model (required unless --features-only)")
    g_model.add_argument(
        "--checkpoint",
        help="Path to a PyTorch Lightning .ckpt file (from scripts/train.py "
             "or downloaded from the HuggingFace Hub).",
    )
    g_model.add_argument(
        "--train-embeddings", dest="train_embeddings",
        help="CSV of labeled reference embeddings used by the KNN classifier. "
             "Generate with `hippie-cli embed` + a labeled reference dataset, "
             "then convert the .npz to CSV with columns: <z_dim> + 'label'.",
    )
    g_model.add_argument(
        "--config", default="class_decoder_source_bn_aug_reg",
        help="ExperimentConfigs method name used during training "
             "(default: class_decoder_source_bn_aug_reg).",
    )
    g_model.add_argument(
        "--z-dim", dest="z_dim", type=int, default=30,
        help="Latent dimension — must match training run (default: 30, matches the released checkpoint).",
    )
    g_model.add_argument(
        "--num-sources", dest="num_sources", type=int, default=4,
        help="Number of source datasets the model was trained on (default: 4). "
             "Sets the source_embedding table size — must match training.",
    )
    g_model.add_argument(
        "--num-classes", dest="num_classes", type=int, default=4,
        help="Number of cell-type classes (default: 4). "
             "Ignored when --class-names is supplied.",
    )
    g_model.add_argument(
        "--class-names", dest="class_names", nargs="+", default=None,
        help="Ordered list of class names as used in training "
             "(sets --num-classes automatically).",
    )
    g_model.add_argument(
        "--source-id", dest="source_id", type=int, default=0,
        help="Source embedding index to inject at inference (default: 0). "
             "Use 0 for an unseen recording site (maps to embedding row 0). "
             "Match to the closest dataset in your training set for best results.",
    )

    # ── KNN ──────────────────────────────────────────────────────────────────
    g_knn = p.add_argument_group("KNN classifier")
    g_knn.add_argument(
        "-k", default="10",
        help="Number of KNN neighbours. Default 10. Pass `-k auto` to select "
             "k via 5-fold cross-validation on the reference set with "
             "balanced-accuracy scoring over k=1..20 -- matching the "
             "procedure in hippie_benchmarking_release.",
    )

    # ── output ───────────────────────────────────────────────────────────────
    g_out = p.add_argument_group("Output")
    g_out.add_argument(
        "--output", default="hippie_results.csv",
        help="Output CSV path (default: hippie_results.csv).",
    )
    g_out.add_argument(
        "--features-only", dest="features_only", action="store_true",
        help="Extract and save features (waveforms / ISI / ACG) without "
             "running model inference. No checkpoint required.",
    )

    return p


def _validate_args(args: argparse.Namespace):
    if not args.features_only:
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required (or use --features-only).")
        if not args.train_embeddings:
            raise SystemExit("--train-embeddings is required (or use --features-only).")
        if not Path(args.checkpoint).exists():
            raise SystemExit(f"Checkpoint not found: {args.checkpoint}")
        if not Path(args.train_embeddings).exists():
            raise SystemExit(f"Training embeddings not found: {args.train_embeddings}")
    if not Path(args.nwb).exists():
        raise SystemExit(f"NWB file not found: {args.nwb}")


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()
    _validate_args(args)
    run_pipeline(args)
