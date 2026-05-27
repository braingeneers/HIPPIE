"""HIPPIE command-line interface.

Subcommands:

    hippie-cli validate-data <dataset_dir>
        Check that a dataset folder follows the canonical CSV layout.

    hippie-cli embed --datasets-root <dir> [--datasets <name>...]
        Run the pretrained HIPPIE encoder over one or more dataset folders
        and write a single .npz of latent embeddings. Thin wrapper around
        examples/extract_embeddings.py for users who prefer a CLI verb.

    hippie-cli predict --reference <emb.npz> --query <emb.npz>
        KNN-classify query embeddings against a labeled reference set.

The entry point is registered in pyproject.toml; after ``pip install -e .``
the script is available on PATH as ``hippie-cli``.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# validate-data
# ---------------------------------------------------------------------------

REQUIRED_FILES = ["waveforms.csv", "isi_dist.csv"]
OPTIONAL_FILES = ["acg.csv", "labels.csv"]


def _read_csv(path: str) -> np.ndarray:
    return pd.read_csv(path).to_numpy()


def cmd_validate(args: argparse.Namespace) -> int:
    dataset_dir = args.path
    if not os.path.isdir(dataset_dir):
        print(f"ERROR: not a directory: {dataset_dir}", file=sys.stderr)
        return 2

    errors: list[str] = []
    warnings: list[str] = []

    # 1. Required files present.
    for fname in REQUIRED_FILES:
        if not os.path.exists(os.path.join(dataset_dir, fname)):
            errors.append(f"missing required file: {fname}")
    if errors:
        for e in errors:
            print(f"  [ERROR] {e}")
        return 1

    # 2. Load required + optional.
    wf = _read_csv(os.path.join(dataset_dir, "waveforms.csv"))
    isi = _read_csv(os.path.join(dataset_dir, "isi_dist.csv"))
    N = wf.shape[0]
    print(f"  Found {N} units in {dataset_dir}")

    # 3. Shape checks.
    if isi.shape[0] != N:
        errors.append(f"isi_dist.csv has {isi.shape[0]} rows, expected {N} (matching waveforms.csv)")

    acg_path = os.path.join(dataset_dir, "acg.csv")
    if os.path.exists(acg_path):
        acg = _read_csv(acg_path)
        if acg.shape[0] != N:
            errors.append(f"acg.csv has {acg.shape[0]} rows, expected {N}")
        if acg.shape[1] not in (200, 201):
            warnings.append(
                f"acg.csv has {acg.shape[1]} columns; expected 201 (lag -100..+100 ms, 1 ms bins). "
                "HIPPIE will resample at inference time, but the canonical shape is 201."
            )
        if np.any(acg < 0):
            warnings.append("acg.csv contains negative values; expected non-negative pair counts.")
    else:
        warnings.append("acg.csv not found; embedding will run in bimodal mode (zero-filled ACG).")

    labels_path = os.path.join(dataset_dir, "labels.csv")
    if os.path.exists(labels_path):
        labels = pd.read_csv(labels_path)
        if len(labels) != N:
            errors.append(f"labels.csv has {len(labels)} rows, expected {N}")
        if labels.shape[1] == 0:
            errors.append("labels.csv has no columns")
    else:
        warnings.append("labels.csv not found; supervised training and KNN/MLP evaluation will be skipped.")

    # 4. Sanity checks on values. NaN/Inf is a warning -- the dataloaders
    # sanitize NaNs at load time (scripts/train.py and
    # examples/extract_embeddings.py both call np.nan_to_num), so
    # non-finite values are tolerated in practice -- but a fresh
    # wrangling pipeline should not produce them.
    if not np.isfinite(wf).all():
        warnings.append(
            f"waveforms.csv contains {int(np.sum(~np.isfinite(wf)))} non-finite values; "
            "HIPPIE will replace them with 0 at load time."
        )
    if not np.isfinite(isi).all():
        warnings.append(
            f"isi_dist.csv contains {int(np.sum(~np.isfinite(isi)))} non-finite values; "
            "HIPPIE will replace them with 0 at load time."
        )
    if np.any(isi < 0):
        warnings.append(
            "isi_dist.csv contains negative values; expected non-negative spike counts "
            "(the loader applies log(x+1) which is undefined for x<-1)."
        )

    # 5. Waveform trough-centering sanity check (tolerant of any length).
    finite_wf = np.where(np.isfinite(wf), wf, 0.0)
    trough_positions = np.argmin(finite_wf, axis=1)
    T = wf.shape[1]
    # Canonical layout has the trough roughly 40% of the way through (20 of 50);
    # warn only if the median trough is in the outer 10% on either side.
    if np.median(trough_positions) < 0.1 * T or np.median(trough_positions) > 0.9 * T:
        warnings.append(
            f"waveform median trough is at sample {int(np.median(trough_positions))} of {T}; "
            "HIPPIE expects roughly trough-centered waveforms "
            "(see data_wrangling_scripts/neurocurator.py for the canonical layout)."
        )

    # 6. Report.
    if warnings:
        print(f"\n  {len(warnings)} warning(s):")
        for w in warnings:
            print(f"  [WARN]  {w}")
    if errors:
        print(f"\n  {len(errors)} error(s):")
        for e in errors:
            print(f"  [ERROR] {e}")
        print("\n  FAIL")
        return 1

    print("\n  OK")
    return 0


# ---------------------------------------------------------------------------
# embed
# ---------------------------------------------------------------------------

def cmd_embed(args: argparse.Namespace) -> int:
    # Import lazily so `hippie-cli validate-data` works without torch loaded.
    from .inference import HIPPIEClassifier, TECHNOLOGY_IDS
    import torch
    import torch.nn.functional as F

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        classifier = HIPPIEClassifier.from_checkpoint(args.checkpoint, device=device)
    else:
        print(f"Loading checkpoint from HuggingFace Hub: {args.hf_repo}/{args.hf_filename}")
        classifier = HIPPIEClassifier.from_pretrained(
            repo_id=args.hf_repo, filename=args.hf_filename, device=device
        )

    def _resample_minmax(raw, target_len):
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

    all_emb, all_labels, all_tech, all_ds, all_nid = [], [], [], [], []
    tech_name = args.tech or "neuropixels"
    if tech_name not in TECHNOLOGY_IDS:
        print(f"ERROR: --tech {tech_name} not in {list(TECHNOLOGY_IDS)}", file=sys.stderr)
        return 2
    tech_id = TECHNOLOGY_IDS[tech_name]

    for i, dataset in enumerate(args.datasets):
        dst = os.path.join(args.datasets_root, dataset)
        if not os.path.isdir(dst):
            print(f"ERROR: dataset folder not found: {dst}", file=sys.stderr)
            return 2

        wf = pd.read_csv(os.path.join(dst, "waveforms.csv")).to_numpy().astype(np.float32)
        isi = pd.read_csv(os.path.join(dst, "isi_dist.csv")).to_numpy().astype(np.float32)
        acg_path = os.path.join(dst, "acg.csv")
        if os.path.exists(acg_path):
            acg = pd.read_csv(acg_path).to_numpy().astype(np.float32)
        else:
            acg = np.zeros((len(wf), 100), dtype=np.float32)

        wave_p = _resample_minmax(wf, 50)
        isi_p = _resample_minmax(np.log(np.nan_to_num(isi, nan=0.0) + 1.0), 100)
        acg_p = _resample_minmax(np.nan_to_num(acg, nan=0.0), 100)

        print(f"[{i + 1}/{len(args.datasets)}] {dataset}: {len(wf)} units")
        emb = classifier.get_embeddings(wave=wave_p, isi=isi_p, acg=acg_p,
                                        tech_id=tech_id, batch_size=args.batch_size)

        labels_path = os.path.join(dst, "labels.csv")
        if os.path.exists(labels_path):
            labels = pd.read_csv(labels_path).iloc[:, -1].astype(str).tolist()
        else:
            labels = ["unknown"] * len(wf)

        all_emb.append(emb)
        all_labels.extend(labels)
        all_tech.append(np.full(len(wf), tech_id, dtype=np.int32))
        all_ds.extend([dataset] * len(wf))
        all_nid.append(np.arange(len(wf)))

    embeddings = np.concatenate(all_emb, axis=0)
    np.savez_compressed(
        args.output,
        embeddings=embeddings,
        labels=np.array(all_labels),
        dataset_ids=np.array(all_ds),
        technology_ids=np.concatenate(all_tech, axis=0),
        neuron_ids=np.concatenate(all_nid, axis=0),
    )
    print(f"\nSaved {len(embeddings)} embeddings to {args.output}  shape={embeddings.shape}")
    return 0


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------

def cmd_predict(args: argparse.Namespace) -> int:
    from sklearn.neighbors import KNeighborsClassifier

    from .inference import select_k_via_cv

    ref = np.load(args.reference, allow_pickle=True)
    qry = np.load(args.query, allow_pickle=True)
    if "embeddings" not in ref.files or "labels" not in ref.files:
        print("ERROR: --reference must contain 'embeddings' and 'labels' arrays", file=sys.stderr)
        return 2
    if "embeddings" not in qry.files:
        print("ERROR: --query must contain an 'embeddings' array", file=sys.stderr)
        return 2

    ref_emb = ref["embeddings"]
    ref_labels = ref["labels"]

    # Pick k. "--k auto" (default) runs CV on the *reference set only*,
    # matching the procedure in hippie_benchmarking_release. An explicit
    # integer bypasses CV.
    if args.k == "auto":
        try:
            best_k, scores = select_k_via_cv(ref_emb, ref_labels)
            print(f"Selected best k = {best_k} via 5-fold CV on the reference set.")
            for k_val, score in sorted(scores.items()):
                marker = " <-- selected" if k_val == best_k else ""
                print(f"  k={k_val:>2}  balanced_acc={score:.4f}{marker}")
        except ValueError as exc:
            print(f"ERROR: k-selection failed: {exc}", file=sys.stderr)
            return 2
        k = best_k
    else:
        k = int(args.k)
        print(f"Using k = {k} (pass --k auto to enable CV-based selection).")

    knn = KNeighborsClassifier(n_neighbors=k, metric="cosine")
    knn.fit(ref_emb, ref_labels)
    pred = knn.predict(qry["embeddings"])
    probas = knn.predict_proba(qry["embeddings"])  # (n_query, n_classes)
    classes = knn.classes_                          # class label per probas column

    out = pd.DataFrame({
        "neuron_id": qry["neuron_ids"] if "neuron_ids" in qry.files else np.arange(len(pred)),
        "dataset_id": qry["dataset_ids"] if "dataset_ids" in qry.files else "",
        "prediction": pred,
        "confidence": probas.max(axis=1),
        "k_used": k,
    })
    # Per-class probability columns ("confidence matrix"): one prob_<class>
    # column per training-set class. Each row sums to 1.0.
    for j, cls in enumerate(classes):
        out[f"prob_{cls}"] = probas[:, j]

    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out)} predictions ({len(classes)} classes) to {args.output}")
    return 0


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="hippie-cli", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)

    v = sub.add_parser("validate-data", help="Check a dataset folder for schema compliance")
    v.add_argument("path", help="Path to a <dataset> folder containing the canonical CSVs")
    v.set_defaults(func=cmd_validate)

    e = sub.add_parser("embed", help="Run the pretrained HIPPIE encoder")
    e.add_argument("--datasets-root", required=True)
    e.add_argument("--datasets", nargs="+", required=True)
    e.add_argument("--checkpoint", default=None,
                   help="Local .ckpt path. If omitted, downloads from HuggingFace Hub.")
    e.add_argument("--hf-repo", default="Jesusgf23/hippie")
    e.add_argument("--hf-filename", default="hippie_techcond_v1.ckpt")
    e.add_argument("--output", default="embeddings.npz")
    e.add_argument("--batch-size", type=int, default=512)
    e.add_argument("--tech", default=None,
                   help="Recording technology key (neuropixels | silicon_probe | juxtacellular)")
    e.add_argument("--device", default=None)
    e.set_defaults(func=cmd_embed)

    pr = sub.add_parser("predict", help="KNN-classify query embeddings against a labeled reference")
    pr.add_argument("--reference", required=True, help="Reference .npz from `hippie-cli embed`")
    pr.add_argument("--query", required=True, help="Query .npz from `hippie-cli embed`")
    pr.add_argument("--output", default="predictions.csv")
    pr.add_argument("--k", default="10",
                    help="KNN neighbour count. Default 10. Pass --k auto to run 5-fold CV on the "
                         "reference set with balanced-accuracy scoring over k=1..20 and pick the "
                         "best -- matching the principled procedure in hippie_benchmarking_release.")
    pr.set_defaults(func=cmd_predict)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
