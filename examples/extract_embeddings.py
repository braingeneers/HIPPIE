"""
Extract HIPPIE latent embeddings for one or more datasets.

This example uses the public ``hippie.inference`` API. By default it pulls the
pretrained checkpoint from the HuggingFace Hub (``Jesusgf23/hippie``); pass
``--checkpoint`` to use a local ``.ckpt`` file instead.

Datasets must follow the canonical HIPPIE CSV format:

    <datasets-root>/<dataset_name>/
        waveforms.csv     (N x T)        spike waveforms (any T; resampled to 50)
        isi_dist.csv      (N x T)        ISI histograms     (any T; resampled to 100; 1 ms bins, 0-100 ms)
        acg.csv           (N x T)        autocorrelograms   (any T; resampled to 100; 1 ms bins, -100..+100 ms) -- optional, zero-filled if missing
        labels.csv        (N x ...)      ground-truth labels (last column is read)

Usage:
    # Default: load weights from HuggingFace Hub
    python examples/extract_embeddings.py \
        --datasets-root ./datasets_hippie \
        --output ./embeddings.npz

    # Use a local checkpoint
    python examples/extract_embeddings.py \
        --datasets-root ./datasets_hippie \
        --checkpoint ./hippie_techcond_v1.ckpt \
        --output ./embeddings.npz

    # Only embed a subset of datasets
    python examples/extract_embeddings.py \
        --datasets-root ./datasets_hippie \
        --datasets hausser_cell_type lisberger_labeled_cell_type
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from hippie.inference import HIPPIEClassifier, TECHNOLOGY_IDS


# Recording technology per dataset (maps to TECHNOLOGY_IDS slots 0-3).
# Edit this dict to add your own datasets. Datasets shipped under
# datasets_hippie/ are listed first; the rest are paper datasets that
# users must wrangle themselves (see data_wrangling_scripts/README.md).
DATASET_TECHNOLOGY = {
    # Shipped under datasets_hippie/
    "toy":                                 "neuropixels",
    "hausser_cell_type":                   "neuropixels",
    "hull_cell_type":                      "neuropixels",
    "lisberger_labeled_cell_type":        "silicon_probe",
    "cellexplorer_cell_type":              "neuropixels",
    "a1data_remove_undef":                 "silicon_probe",
    "juxtacellular_mouse_s1_area":         "juxtacellular",
    "allen_scope_neuropixel_area_subset":  "neuropixels",
    # Paper datasets not shipped -- wrangle from DANDI / IBL first
    "dandi_000041_cell_type":              "silicon_probe",
    "dandi_000473_cell_type":              "neuropixels",
    "dandi_000955_cell_type":              "silicon_probe",
    "ibl_brainwide_good":                  "neuropixels",
}

# Default --datasets list when none is passed: only the folders that
# actually ship in this repo. Run with --datasets <name>... to embed
# something you wrangled yourself.
_SHIPPED_DATASETS = [
    "hausser_cell_type",
    "hull_cell_type",
    "lisberger_labeled_cell_type",
    "cellexplorer_cell_type",
    "a1data_remove_undef",
    "juxtacellular_mouse_s1_area",
    "allen_scope_neuropixel_area_subset",
]

WAVE_LEN = 50
ISI_LEN  = 100
ACG_LEN  = 100


def _resample_minmax(raw: np.ndarray, target_len: int) -> np.ndarray:
    """Resample to target_len and min-max normalize each row to [-1, 1]."""
    t = torch.as_tensor(raw, dtype=torch.float32)
    if t.dim() == 1:
        t = t.unsqueeze(0)
    if t.shape[-1] != target_len:
        t = F.interpolate(
            t.unsqueeze(1), size=(target_len,), mode="linear", align_corners=False
        ).squeeze(1)
    mn = t.amin(dim=-1, keepdim=True)
    mx = t.amax(dim=-1, keepdim=True)
    return ((t - mn) / (mx - mn + 1e-8) * 2.0 - 1.0).numpy().astype(np.float32)


def preprocess_waveforms(arr: np.ndarray) -> np.ndarray:
    return _resample_minmax(arr, WAVE_LEN)


def preprocess_isi(arr: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(arr, nan=0.0).astype(np.float32)
    return _resample_minmax(np.log(arr + 1.0), ISI_LEN)


def preprocess_acg(arr: np.ndarray) -> np.ndarray:
    return _resample_minmax(
        np.nan_to_num(arr, nan=0.0).astype(np.float32), ACG_LEN
    )


_SYNONYMS = {
    "pvalb": "PV", "parvalbumin": "PV",
    "sst": "SOM", "somatostatin": "SOM", "somnan": "SOM",
}


def canonicalize_label(lbl: str) -> str:
    norm = " ".join(str(lbl).strip().lower().split())
    return _SYNONYMS.get(norm, lbl)


def embed_dataset(
    classifier: HIPPIEClassifier,
    dataset: str,
    datasets_root: str,
    batch_size: int,
):
    """Return (embeddings, labels, tech_ids, neuron_ids) for one dataset."""
    dst = os.path.join(datasets_root, dataset)
    if not os.path.isdir(dst):
        raise FileNotFoundError(
            f"Dataset folder not found: {dst}\n"
            f"  Expected: <datasets-root>/<dataset_name>/{{waveforms,isi_dist,acg,labels}}.csv\n"
            f"  Either copy your CSVs into {dst}, or pass --datasets explicitly.\n"
            f"  See data_wrangling_scripts/README.md for how to wrangle a paper dataset."
        )

    wf  = pd.read_csv(os.path.join(dst, "waveforms.csv")).to_numpy().astype(np.float32)
    isi = pd.read_csv(os.path.join(dst, "isi_dist.csv")).to_numpy().astype(np.float32)

    acg_path = os.path.join(dst, "acg.csv")
    if os.path.exists(acg_path):
        acg = pd.read_csv(acg_path).to_numpy().astype(np.float32)
    else:
        print(f"  [{dataset}] No acg.csv found, using zeros (bimodal mode)")
        acg = np.zeros((len(wf), ACG_LEN), dtype=np.float32)

    labels_df = pd.read_csv(os.path.join(dst, "labels.csv"))
    labels = [canonicalize_label(l) for l in labels_df.iloc[:, -1].astype(str).tolist()]

    tech_name = DATASET_TECHNOLOGY.get(dataset, "neuropixels")
    tech_id   = TECHNOLOGY_IDS[tech_name]

    embeddings = classifier.get_embeddings(
        wave=preprocess_waveforms(wf),
        isi=preprocess_isi(isi),
        acg=preprocess_acg(acg),
        tech_id=tech_id,
        batch_size=batch_size,
    )
    tech_ids   = np.full(len(wf), tech_id, dtype=np.int32)
    neuron_ids = np.arange(len(wf))
    return embeddings, labels, tech_ids, neuron_ids


def main():
    p = argparse.ArgumentParser(description="Extract HIPPIE latent embeddings.")
    p.add_argument("--datasets-root", required=True,
                   help="Directory containing <dataset>/{waveforms,isi_dist,acg,labels}.csv subfolders")
    p.add_argument("--datasets", nargs="+", default=_SHIPPED_DATASETS,
                   help="Subset of datasets to embed. Default: datasets shipped under datasets_hippie/.")
    p.add_argument("--checkpoint", default=None,
                   help="Local .ckpt path. If omitted, the checkpoint is downloaded from HuggingFace Hub.")
    p.add_argument("--hf-repo", default="Jesusgf23/hippie",
                   help="HuggingFace repo ID (used only when --checkpoint is not given)")
    p.add_argument("--hf-filename", default="hippie_techcond_v1.ckpt")
    p.add_argument("--output", default="embeddings.npz")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint} ...")
        classifier = HIPPIEClassifier.from_checkpoint(args.checkpoint, device=args.device)
    else:
        print(f"Loading checkpoint from HuggingFace Hub: {args.hf_repo}/{args.hf_filename} ...")
        classifier = HIPPIEClassifier.from_pretrained(
            repo_id=args.hf_repo, filename=args.hf_filename, device=args.device
        )
    print(f"Model loaded on {args.device}.")

    all_emb, all_labels, all_tech, all_ds, all_nid = [], [], [], [], []
    for i, dataset in enumerate(args.datasets):
        print(f"\n[{i+1}/{len(args.datasets)}] {dataset}")
        emb, labels, tech_ids, nids = embed_dataset(
            classifier, dataset, args.datasets_root, args.batch_size
        )
        all_emb.append(emb)
        all_labels.extend(labels)
        all_tech.append(tech_ids)
        all_ds.extend([dataset] * len(labels))
        all_nid.append(nids)
        print(f"  {len(labels)} neurons embedded, shape {emb.shape}")

    embeddings = np.concatenate(all_emb, axis=0)
    np.savez_compressed(
        args.output,
        embeddings=embeddings,
        labels=np.array(all_labels),
        dataset_ids=np.array(all_ds),
        technology_ids=np.concatenate(all_tech, axis=0),
        neuron_ids=np.concatenate(all_nid, axis=0),
    )
    print(f"\nSaved {len(embeddings)} embeddings to {args.output}")
    print(f"  Shape: {embeddings.shape}  (N x z_dim)")
    print(f"  Datasets: {dict(zip(*np.unique(np.array(all_ds), return_counts=True)))}")


if __name__ == "__main__":
    main()
