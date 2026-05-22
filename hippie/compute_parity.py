"""
Unified compute-parity logging for HIPPIE / PhysMAP / NEMO benchmark runs.

The goal is a single CSV schema that every benchmark method appends to, so we
can build a per-method x per-dataset wall-clock and peak-memory table for the
Nature Comms supplement (the "compute parity" reviewers will look for).

Schema (one row per phase per run, appended to a single CSV):

    method        : "hippie" | "physmap" | "nemo"
    dataset       : dataset folder name (e.g. "cellexplorer_cell_type")
    fold          : integer CV / holdout fold index, or -1 if N/A
    phase         : free-form phase label
                    (e.g. "pretrain", "finetune", "supervised", "embedding",
                     "knn", "total"; PhysMAP uses "wnn", "knn_cv", "total"; NEMO
                     uses "train", "embedding", "evaluate", "total")
    wall_seconds  : float, wall clock seconds for this phase
    peak_gpu_mb   : float, peak CUDA memory across all visible devices in MB
                    (NaN if no GPU was used)
    peak_cpu_mb   : float, peak resident set size in MB during the phase
                    (NaN if psutil unavailable)
    n_cells       : int, number of cells the phase operated on (or 0)
    n_classes     : int, number of distinct classes (or 0)
    control       : "" | "shuffled" | "shifted"
    config        : free-form config label
                    (HIPPIE: --config value; NEMO: model variant; PhysMAP: "")
    run_id        : unique string per script invocation, used to group rows
                    that came from the same launch
    started_at    : ISO 8601 timestamp at start of phase (UTC)
    extra         : JSON-encoded dict of method-specific metadata (z_dim, beta,
                    epochs, etc). Free-form so each method can stash whatever
                    it wants without breaking the schema.

The default output CSV path is
    <project_root>/results/compute_parity/timings.csv
where <project_root> is the parent of the `hippie/` package directory unless
the COMPUTE_PARITY_CSV environment variable is set, in which case that path is
used verbatim. Each row is appended atomically with a file lock so that
multiple folds running in parallel on NRP / SLURM don't trample each other.

PhysMAP (R) emits rows by calling out to a tiny R helper that writes the same
schema; see comparison_methods/physmap/compute_parity.r.

This module has no hard dependency on torch or psutil - both are best-effort.
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
import os
import time
import uuid
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

try:
    import psutil  # type: ignore
    _HAS_PSUTIL = True
except Exception:
    _HAS_PSUTIL = False

try:
    import torch  # type: ignore
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


# Schema columns, in stable order. Anything new goes at the end.
COLUMNS = (
    "method",
    "dataset",
    "fold",
    "phase",
    "wall_seconds",
    "peak_gpu_mb",
    "peak_cpu_mb",
    "n_cells",
    "n_classes",
    "control",
    "config",
    "run_id",
    "started_at",
    "extra",
)


def default_csv_path() -> str:
    """Resolve the default unified-CSV path.

    Order of precedence:
      1. $COMPUTE_PARITY_CSV (full path; used verbatim)
      2. <package parent>/results/compute_parity/timings.csv
    """
    env = os.environ.get("COMPUTE_PARITY_CSV")
    if env:
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(here)  # parent of hippie/ -> repo root
    return os.path.join(project_root, "results", "compute_parity", "timings.csv")


def make_run_id(prefix: str = "") -> str:
    """Stable per-launch ID. Reused across all phases of the same script run."""
    stamp = _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    short = uuid.uuid4().hex[:8]
    return f"{prefix}{stamp}-{short}" if prefix else f"{stamp}-{short}"


def _peak_cpu_mb() -> float:
    if not _HAS_PSUTIL:
        return float("nan")
    try:
        return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    except Exception:
        return float("nan")


def _peak_gpu_mb_and_reset() -> float:
    """Return peak CUDA memory in MB across visible devices and reset counters.

    Returns NaN if torch is unavailable, no CUDA devices are visible, or the
    underlying call fails.
    """
    if not _HAS_TORCH:
        return float("nan")
    try:
        if not torch.cuda.is_available():
            return float("nan")
        peak = 0
        for d in range(torch.cuda.device_count()):
            peak = max(peak, torch.cuda.max_memory_allocated(d))
            torch.cuda.reset_peak_memory_stats(d)
        return peak / (1024 ** 2)
    except Exception:
        return float("nan")


def _reset_gpu_peak() -> None:
    if not _HAS_TORCH:
        return
    try:
        if not torch.cuda.is_available():
            return
        for d in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(d)
    except Exception:
        pass


def append_row(
    *,
    method: str,
    dataset: str,
    fold: int,
    phase: str,
    wall_seconds: float,
    peak_gpu_mb: float = float("nan"),
    peak_cpu_mb: float = float("nan"),
    n_cells: int = 0,
    n_classes: int = 0,
    control: str = "",
    config: str = "",
    run_id: str = "",
    started_at: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
    csv_path: Optional[str] = None,
) -> str:
    """Append one row to the unified compute-parity CSV. Returns the path used.

    Creates parent directories and writes a header if the file doesn't exist.
    Uses a best-effort cross-process advisory lock (fcntl on POSIX) so parallel
    fold runs on the same node don't corrupt the file. The lock is best-effort:
    on platforms without fcntl we fall back to plain append, which is still
    safe for single-line writes on most filesystems.
    """
    csv_path = csv_path or default_csv_path()
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    if started_at is None:
        started_at = _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"

    row = {
        "method": method,
        "dataset": dataset,
        "fold": fold,
        "phase": phase,
        "wall_seconds": float(wall_seconds),
        "peak_gpu_mb": float(peak_gpu_mb),
        "peak_cpu_mb": float(peak_cpu_mb),
        "n_cells": int(n_cells),
        "n_classes": int(n_classes),
        "control": control,
        "config": config,
        "run_id": run_id,
        "started_at": started_at,
        "extra": json.dumps(extra or {}, sort_keys=True, default=str),
    }

    write_header = not os.path.exists(csv_path)

    # Best-effort advisory lock
    try:
        import fcntl  # type: ignore
        _has_fcntl = True
    except Exception:
        _has_fcntl = False

    with open(csv_path, "a", newline="") as f:
        if _has_fcntl:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            except Exception:
                pass
        try:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass
        finally:
            if _has_fcntl:
                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                except Exception:
                    pass

    return csv_path


@contextmanager
def phase(
    method: str,
    dataset: str,
    phase: str,
    *,
    fold: int = -1,
    n_cells: int = 0,
    n_classes: int = 0,
    control: str = "",
    config: str = "",
    run_id: str = "",
    extra: Optional[Dict[str, Any]] = None,
    csv_path: Optional[str] = None,
    reset_gpu_peak: bool = True,
) -> Iterator[Dict[str, Any]]:
    """Context manager that times a phase and appends one CSV row on exit.

    Yields a mutable dict the caller can stash extra metadata into; the dict
    will be merged into the row's `extra` JSON column when the phase ends.

    Example:
        with phase("hippie", dataset, "pretrain", fold=cv_fold,
                   n_cells=n_train, n_classes=n_classes, run_id=run_id,
                   config=args.config, extra={"epochs": args.pretrain_max_epochs}) as info:
            trainer.fit(...)
            info["final_val_loss"] = trainer.callback_metrics.get("val_loss")
    """
    if reset_gpu_peak:
        _reset_gpu_peak()
    started_at = _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    t0 = time.perf_counter()
    info: Dict[str, Any] = dict(extra or {})
    try:
        yield info
    finally:
        wall = time.perf_counter() - t0
        peak_gpu = _peak_gpu_mb_and_reset() if reset_gpu_peak else float("nan")
        peak_cpu = _peak_cpu_mb()
        try:
            append_row(
                method=method,
                dataset=dataset,
                fold=fold,
                phase=phase,
                wall_seconds=wall,
                peak_gpu_mb=peak_gpu,
                peak_cpu_mb=peak_cpu,
                n_cells=n_cells,
                n_classes=n_classes,
                control=control,
                config=config,
                run_id=run_id,
                started_at=started_at,
                extra=info,
                csv_path=csv_path,
            )
        except Exception as e:
            # Never let timing logging break a real benchmark run.
            print(f"[compute_parity] WARNING: failed to append row for "
                  f"{method}/{dataset}/fold={fold}/phase={phase}: {e}")