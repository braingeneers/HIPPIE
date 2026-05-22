import os
import pandas as pd
import numpy as np
import zipfile

from scipy import interpolate, signal
from joblib import Parallel, delayed
from pynwb import NWBHDF5IO


class Neurocurator:
    def __init__(self):
        self.waveforms = None
        self.spike_times_train = None
        self.sampling_rate = None
        self.metadata_obs = None
        self.isi_distribution = None
        self.acgs = None

    def load_curation_file(self, qm_path):
        """Load spike times, neuron data, config, and sampling rate from an ACQM zip."""
        if not os.path.exists(qm_path):
            raise FileNotFoundError(f"Curation file not found: {qm_path}")

        try:
            with zipfile.ZipFile(qm_path, "r") as f_zip:
                if "qm.npz" not in f_zip.namelist():
                    raise ValueError(f"Invalid curation file: missing qm.npz in {qm_path}")
                data = np.load(f_zip.open("qm.npz"), allow_pickle=True)

                for key in ["train", "fs", "neuron_data"]:
                    if key not in data:
                        raise KeyError(f"Missing required key '{key}' in curation file")

                spike_times = data["train"].item()
                fs = data["fs"]
                train = [times / fs for _, times in spike_times.items()]
                config = data["config"].item() if "config" in data else None
                neuron_data = data["neuron_data"].item()

        except zipfile.BadZipFile:
            raise ValueError(f"Not a valid zip archive: {qm_path}")

        return train, neuron_data, config, fs

    def load_acqm(self, acqm_path):
        """Load an ACQM zip file and populate all fields."""
        train, neuron_data, config, fs = self.load_curation_file(acqm_path)
        self.sampling_rate = fs
        self.spike_times_train = [np.array(st) * 1000 for st in train]  # convert to ms
        self.waveforms = self.extract_waveforms(neuron_data)
        self.isi_distribution = self.compute_isi_distribution()
        self.acgs = self.compute_autocorrelogram(train)
        self.metadata_obs = self.extract_neuron_positions(neuron_data)
        self.validate_data_integrity()

    def extract_waveforms(self, neuron_data, n_datapoints=50):
        """Extract mean waveforms, centered on trough (20 pts before, 30 after)."""
        datapoints_before = int(n_datapoints / 5 * 2)
        datapoints_after = n_datapoints - datapoints_before

        neuron_array = []
        for neuron in neuron_data.values():
            if "waveforms" in neuron:
                neuron_array.append(np.mean(neuron["waveforms"], axis=0))
            else:
                neuron_array.append(neuron["template"])

        neuron_cut = []
        for wf in neuron_array:
            min_idx = np.argmin(wf)
            segment = wf[min_idx - datapoints_before : min_idx + datapoints_after]
            neuron_cut.append(segment if len(segment) == n_datapoints else np.zeros(n_datapoints))

        return pd.DataFrame(neuron_cut)

    def extract_neuron_positions(self, neuron_data):
        """Extract x, y electrode positions from neuron data."""
        positions = [neuron["position"][:2] for neuron in neuron_data.values()]
        return pd.DataFrame(positions, columns=["x", "y"])

    def compute_single_isi(self, spike_times):
        """Return interspike intervals for a sorted spike-times array."""
        spike_times.sort()
        if len(spike_times) < 2:
            return np.array([])
        return np.diff(spike_times)

    def compute_all_isis(self, spike_times_train):
        """Compute ISIs for all units."""
        isis = []
        for neuron in spike_times_train:
            if isinstance(neuron, list):
                neuron = np.array(neuron)
            elif not isinstance(neuron, np.ndarray):
                raise ValueError("Spike times must be a numpy array or list")
            isis.append(self.compute_single_isi(neuron.flatten()))
        return isis

    def compute_isi_distribution(self, time_window=100):
        """Compute ISI histograms with 1 ms bins up to time_window ms."""
        isis = self.compute_all_isis(self.spike_times_train)

        def compute_hist(isi):
            if len(isi) == 0:
                return np.zeros(time_window)
            return np.histogram(isi[isi < time_window], bins=time_window)[0]

        return pd.DataFrame(Parallel(n_jobs=-1)(delayed(compute_hist)(isi) for isi in isis))

    def compute_autocorrelogram(
        self, spike_trains, bin_size_ms=1, window_size_ms=100, normalize=False, remove_central_bin=True
    ):
        """
        Compute autocorrelograms for all spike trains.
        Returns a DataFrame (units × bins) with bin centers (ms) as column names.
        """
        bin_edges = np.arange(-window_size_ms, window_size_ms + bin_size_ms, bin_size_ms)
        bin_centers = bin_edges[:-1] + bin_size_ms / 2
        n_bins = len(bin_centers)

        def compute_single_acg(train):
            train = np.sort(np.asarray(train, dtype=np.float64).flatten())
            if len(train) < 2:
                return np.zeros(n_bins)

            differences = []
            for i in range(len(train)):
                j_max = np.searchsorted(train, train[i] + window_size_ms, side="right")
                if j_max > i + 1:
                    diffs = train[i + 1 : j_max] - train[i]
                    differences.extend(diffs)
                    differences.extend(-diffs)

            differences = np.array(differences) if differences else np.array([])
            if remove_central_bin and len(differences) > 0:
                differences = differences[np.abs(differences) > 1e-10]

            if len(differences) > 0:
                counts, _ = np.histogram(differences, bins=bin_edges)
            else:
                counts = np.zeros(n_bins)

            if normalize and len(train) > 0:
                counts = counts.astype(np.float64) / len(train)
            return counts

        all_acgs = Parallel(n_jobs=-1)(delayed(compute_single_acg)(t) for t in spike_trains)
        return pd.DataFrame(np.array(all_acgs), columns=[f"{x:.2f}" for x in bin_centers])

    def compute_cross_correlation(self, bt1, bt2, ccg_win=[-10, 10], t_lags_shift=0, bin_size=1):
        """Cross-correlation between two binary spike trains."""
        if np.all((np.array(ccg_win) / bin_size) % 1) != 0:
            raise ValueError("Window and shift must be multiples of bin size")

        left_edge, right_edge = np.subtract(np.array(ccg_win) / bin_size, t_lags_shift)
        lags = np.arange(ccg_win[0], ccg_win[1] + bin_size, bin_size)
        pad_width = min(max(-int(left_edge), 0), max(int(right_edge), 0))
        bt2_pad = np.pad(bt2, pad_width=pad_width, mode="constant")
        return np.round(signal.fftconvolve(bt2_pad, bt1[::-1], mode="valid")), lags

    def compute_waveform_features(self, waveform, mid=20, right_only=True, fs=20000.0):
        """
        Compute waveform shape features.
        Returns dict with: lpeak, rpeak, trough_to_peak (ms), fwhm (ms), fwhm_x.
        """
        if waveform[mid] > 0:
            waveform = -waveform

        left = waveform[: mid + 1][::-1]
        right = waveform[mid:]

        lpeak_ind = 0
        for ind in np.where(left[1:] - left[:-1] < 0)[0]:
            if waveform[mid - ind] > 0:
                lpeak_ind = mid - ind
                break
        lpeak = (lpeak_ind, waveform[lpeak_ind])

        rpeak_ind = len(waveform) - 1
        for ind in np.where(right[1:] - right[:-1] < 0)[0]:
            if waveform[mid + ind] > 0:
                rpeak_ind = mid + ind
                break
        rpeak = (rpeak_ind, waveform[rpeak_ind])

        post_hyper = rpeak if right_only else (lpeak if lpeak[1] > rpeak[1] else rpeak)
        trough_to_peak = abs(post_hyper[0] - mid) / (fs / 1000)

        half_amp = waveform[mid] / 2.0
        xx_left = np.arange(lpeak_ind, mid + 1)
        xx_right = np.arange(mid, rpeak_ind + 1)
        fl = interpolate.interp1d(waveform[lpeak_ind : mid + 1], xx_left)
        fr = interpolate.interp1d(waveform[mid : rpeak_ind + 1], xx_right)

        seg_left = waveform[lpeak_ind : mid + 1]
        seg_right = waveform[mid : rpeak_ind + 1]
        inter_1 = fl(half_amp) if seg_left.min() <= half_amp <= seg_left.max() else None
        inter_2 = fr(half_amp) if seg_right.min() <= half_amp <= seg_right.max() else None

        try:
            fwhm = (abs(inter_2) - abs(inter_1)) / (fs / 1000)
        except TypeError:
            fwhm = 0

        return {
            "lpeak": lpeak,
            "rpeak": rpeak,
            "trough_to_peak": trough_to_peak,
            "fwhm": fwhm,
            "fwhm_x": np.array([inter_1, inter_2]),
        }

    def compute_all_waveform_features(self):
        """Compute and store waveform shape features in metadata_obs."""
        features = [self.compute_waveform_features(row.to_numpy()) for _, row in self.waveforms.iterrows()]
        for key in ["lpeak", "rpeak", "trough_to_peak", "fwhm", "fwhm_x"]:
            self.metadata_obs[key] = [f[key] for f in features]
        return self.metadata_obs

    def compute_firing_rate(self):
        """Compute spikes / recording duration for each unit."""
        self.metadata_obs["firing_rate"] = [
            len(st) / (max(st) - min(st)) for st in self.spike_times_train
        ]
        return self.metadata_obs

    def compute_minimum_isi(self):
        """Compute minimum interspike interval for each unit."""
        self.metadata_obs["minimum_isi"] = [np.diff(st).min() for st in self.spike_times_train]
        return self.metadata_obs

    def set_experiment_condition(self, column_name, condition):
        """Add a constant metadata column to all units."""
        self.metadata_obs[column_name] = condition

    def validate_data_integrity(self):
        """Warn about missing or empty data structures. Returns True if all present."""
        checks = {
            "waveforms": self.waveforms is not None and not self.waveforms.empty,
            "spike_times": self.spike_times_train is not None and len(self.spike_times_train) > 0,
            "sampling_rate": self.sampling_rate is not None and self.sampling_rate > 0,
            "metadata": self.metadata_obs is not None and not self.metadata_obs.empty,
            "isi_distribution": self.isi_distribution is not None and not self.isi_distribution.empty,
            "autocorrelograms": self.acgs is not None and not self.acgs.empty,
        }
        missing = [k for k, v in checks.items() if not v]
        if missing:
            print(f"Warning: missing or empty: {', '.join(missing)}")
        return not missing

    def load_nwb_spike_times(self, nwb_path):
        """Load spike times (ms) from the units table of an NWB file."""
        with NWBHDF5IO(nwb_path, "r", load_namespaces=True) as io:
            nwb = io.read()
            self.spike_times_train = [
                np.asarray(nwb.units.get_unit_spike_times(i), dtype=float) * 1000.0
                for i in range(len(nwb.units.id.data))
            ]

    def load_nwb_waveforms(self, nwb_path, n_datapoints=50, candidates=("waveform_mean", "spike_waveforms")):
        """Load and standardize waveforms from an NWB file to n_datapoints length."""
        def cut_or_pad(wf, n=50):
            wf = np.asarray(wf, float).flatten()
            if wf.size == 0:
                return np.zeros(n, float)
            mid = int(np.argmin(wf))
            left = int(n * 2 / 5)
            lo = max(0, mid - left)
            seg = wf[lo : min(wf.size, lo + n)]
            return np.pad(seg, (0, n - seg.size)) if seg.size < n else seg

        with NWBHDF5IO(nwb_path, "r", load_namespaces=True) as io:
            nwb = io.read()
            if nwb.units is None:
                self.waveforms = pd.DataFrame()
                return self.waveforms

            key = next((k for k in candidates if k in set(nwb.units.colnames)), None)
            wf_rows = []
            for i in range(len(nwb.units.id.data)):
                if key is None:
                    wf_rows.append(np.zeros(n_datapoints))
                elif key == "spike_waveforms":
                    wfs = nwb.units.get_unit_spike_waveforms(i)
                    if wfs is not None and len(wfs) > 0:
                        wf_rows.append(cut_or_pad(np.mean(np.asarray(wfs, float), axis=0), n_datapoints))
                    else:
                        wf_rows.append(np.zeros(n_datapoints))
                else:
                    wf_rows.append(cut_or_pad(np.asarray(nwb.units[key][i], float), n_datapoints))

        self.waveforms = pd.DataFrame(wf_rows)
        return self.waveforms
