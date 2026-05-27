"""Regenerate datasets_hippie/toy/ from a fixed seed.

The toy dataset is 20 synthetic neurons (10 fake-PV / 10 fake-SOM) used
as a smoke-test fixture for the loader, the inference API, and the
release test suite. It is committed to the repo so that a fresh clone
can run examples/extract_embeddings.py with no external downloads.

Run from the repo root:

    python scripts/generate_toy_dataset.py

The script is deterministic; re-running overwrites the four CSVs with
byte-identical content (modulo pandas column order).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..", "datasets_hippie", "toy")
N_UNITS = 20
WAVE_LEN = 50          # trough at sample 20 (20 pre, 30 post -- canonical layout)
ISI_BINS = 100         # 1 ms bins, 0..100 ms
ACG_BINS = 201         # 1 ms bins, -100..+100 ms (center bin = 0)
SEED = 42


def _waveform(rng: np.random.Generator, narrow: bool) -> np.ndarray:
    """Synthetic biphasic spike. Narrow=PV-like, wide=SOM-like."""
    t = np.arange(WAVE_LEN, dtype=np.float32)
    trough_t = 20
    width = 1.5 if narrow else 3.5
    trough = -np.exp(-((t - trough_t) ** 2) / (2 * width ** 2))
    peak_t = trough_t + (3 if narrow else 8)
    peak_w = width * 1.5
    peak = 0.4 * np.exp(-((t - peak_t) ** 2) / (2 * peak_w ** 2))
    wf = trough + peak
    wf = wf + rng.normal(0.0, 0.03, size=wf.shape).astype(np.float32)
    return wf.astype(np.float32)


def _isi_hist(rng: np.random.Generator, mean_rate_hz: float) -> np.ndarray:
    """Synthetic ISI histogram: exponential ISI distribution + refractory."""
    # Sample ~600 ISIs from an exponential at the requested rate, drop any
    # below a 2 ms refractory period, then histogram in 1 ms bins 0..100 ms.
    isis_ms = rng.exponential(scale=1000.0 / mean_rate_hz, size=600)
    isis_ms = isis_ms[isis_ms >= 2.0]
    hist, _ = np.histogram(isis_ms, bins=np.arange(ISI_BINS + 1))
    return hist.astype(np.float32)


def _acg(rng: np.random.Generator, mean_rate_hz: float) -> np.ndarray:
    """Synthetic autocorrelogram on lags -100..+100 ms, center bin zeroed."""
    # Generate a short Poisson spike train, then compute the empirical ACG.
    duration_s = 60.0
    n_spikes = int(rng.poisson(mean_rate_hz * duration_s))
    spike_times_ms = np.sort(rng.uniform(0.0, duration_s * 1000.0, size=n_spikes))
    edges = np.arange(-100.5, 100.5 + 1.0, 1.0)
    acg = np.zeros(ACG_BINS, dtype=np.float32)
    # Pair up nearby spikes with a sliding window for efficiency.
    j_lo = 0
    for i, t in enumerate(spike_times_ms):
        while j_lo < n_spikes and spike_times_ms[j_lo] < t - 100.0:
            j_lo += 1
        j = j_lo
        while j < n_spikes and spike_times_ms[j] <= t + 100.0:
            if j != i:
                lag = spike_times_ms[j] - t
                idx = int(np.searchsorted(edges, lag) - 1)
                if 0 <= idx < ACG_BINS:
                    acg[idx] += 1.0
            j += 1
    # Zero the center bin (lag ~ 0 ms) per the canonical layout.
    acg[ACG_BINS // 2] = 0.0
    return acg


def main() -> None:
    os.makedirs(ROOT, exist_ok=True)
    rng = np.random.default_rng(SEED)

    waveforms = np.zeros((N_UNITS, WAVE_LEN), dtype=np.float32)
    isi_dists = np.zeros((N_UNITS, ISI_BINS), dtype=np.float32)
    acgs = np.zeros((N_UNITS, ACG_BINS), dtype=np.float32)
    labels = []

    for i in range(N_UNITS):
        is_pv = (i < N_UNITS // 2)
        waveforms[i] = _waveform(rng, narrow=is_pv)
        isi_dists[i] = _isi_hist(rng, mean_rate_hz=18.0 if is_pv else 6.0)
        acgs[i] = _acg(rng, mean_rate_hz=18.0 if is_pv else 6.0)
        labels.append("PV" if is_pv else "SOM")

    pd.DataFrame(waveforms).to_csv(os.path.join(ROOT, "waveforms.csv"), index=False)
    pd.DataFrame(isi_dists).to_csv(os.path.join(ROOT, "isi_dist.csv"), index=False)
    # ACG column names = lag in ms (matches the shipped paper datasets).
    acg_cols = [f"{lag:.2f}" for lag in np.arange(-100, 101, 1, dtype=float)]
    pd.DataFrame(acgs, columns=acg_cols).to_csv(os.path.join(ROOT, "acg.csv"), index=False)
    pd.DataFrame({"label": labels}).to_csv(os.path.join(ROOT, "labels.csv"), index=False)

    print(f"Wrote {N_UNITS} synthetic neurons to {os.path.abspath(ROOT)}")


if __name__ == "__main__":
    main()
