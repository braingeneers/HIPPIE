import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import torch.utils.data


def extend_sequence_interpolation(tensor, target_size):
    """
    Resample a sequence tensor to target_size via linear interpolation.
    Accepts shape (batch, channels, length) or (batch, length).
    """
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(1)
    if tensor.size(-1) == target_size:
        return tensor
    return F.interpolate(tensor, size=(target_size,), mode="linear", align_corners=False)


def none_safe_collate(data):
    samples = [x[0] for x in data if x is not None]
    labels = [x[1] for x in data if x is not None]

    data_results = {key: [] for key in samples[0].keys()}
    for sample in samples:
        for key, value in sample.items():
            data_results[key].append(value)
    samples = {key: torch.stack(value) for key, value in data_results.items()}

    if labels[0].ndim == 0:
        labels = torch.as_tensor(labels).long()
    else:
        labels = torch.stack(labels).long()

    return samples, labels


class EphysDatasetLabeled(Dataset):
    def __init__(self, waveforms, isi_dists, mode, normalize=True):
        self.waveforms = np.array(waveforms)
        self.isi_dists = np.array(isi_dists)
        assert mode in ("wave", "time", "both")
        self.mode = mode
        assert len(self.waveforms) == len(self.isi_dists)
        self.normalize = normalize

    def __getitem__(self, idx):
        waveform = torch.as_tensor(self.waveforms[idx, ...]).float()
        isi_dist = torch.as_tensor(self.isi_dists[idx, ...]).float()
        isi_dist = torch.log(isi_dist + 1)

        if self.normalize:
            min_val = waveform.min()
            max_val = waveform.max()
            waveform = (waveform - min_val) / (max_val - min_val) * 2 - 1
            isi_dist = (isi_dist - isi_dist.mean()) / isi_dist.std()

        waveform = extend_sequence_interpolation(waveform.view(1, 1, -1), 50).view(1, -1)
        isi_dist = extend_sequence_interpolation(isi_dist.view(1, 1, -1), 100).view(1, -1)

        if self.mode == "wave":
            return waveform, -1
        elif self.mode == "time":
            return isi_dist, -1
        elif self.mode == "both":
            return waveform, isi_dist

    def __len__(self):
        return len(self.waveforms)


class MultiModalEphysDataset(Dataset):
    """Dataset for multiple modalities (wave, isi, acg) with consistent preprocessing."""

    def __init__(self, data_dict, labels, mode="multi", normalize=True, modality_sizes=None):
        """
        Args:
            data_dict: dict mapping modality names to data arrays
            labels: array of integer labels
            mode: "multi" or one of the modality keys
            normalize: apply per-sample min-max normalization to [-1, 1]
            modality_sizes: override default target sizes {wave: 50, isi: 100, acg: 100}
        """
        self.modalities = {name: np.array(data) for name, data in data_dict.items()}
        self.labels = np.array(labels)

        if mode != "multi" and mode not in self.modalities:
            print(f"Mode '{mode}' must be 'multi' or one of: {list(self.modalities.keys())}")
        self.mode = mode

        n_samples = len(self.labels)
        for name, data in self.modalities.items():
            assert len(data) == n_samples, f"Modality '{name}' has {len(data)} samples, expected {n_samples}"

        self.modality_sizes = {"wave": 50, "isi": 100, "acg": 100}
        if modality_sizes:
            self.modality_sizes.update(modality_sizes)

        self.normalize = normalize

    def process_modality(self, mod_name, data, idx):
        """Normalize and interpolate a single modality sample."""
        tensor = torch.as_tensor(data[idx, ...]).float()

        if mod_name == "isi":
            tensor = torch.log(tensor + 1)

        if self.normalize:
            min_val = tensor.min()
            max_val = tensor.max()
            tensor = (tensor - min_val) / (max_val - min_val + 1e-8) * 2 - 1

        tensor = tensor.view(1, 1, -1)
        target_size = self.modality_sizes.get(mod_name)
        if target_size:
            tensor = extend_sequence_interpolation(tensor, target_size)
        return tensor.view(1, -1)

    def __getitem__(self, idx):
        label = torch.as_tensor(self.labels[idx]).long()

        if self.mode == "multi":
            result = {}
            for mod_name, data in self.modalities.items():
                result[mod_name] = self.process_modality(mod_name, data, idx)
                if torch.isnan(result[mod_name]).any() or torch.isinf(result[mod_name]).any():
                    print(f"NaN/Inf in sample {idx}, modality '{mod_name}'")
                    return None
            return result, label
        else:
            tensor = self.process_modality(self.mode, self.modalities[self.mode], idx)
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                print(f"NaN/Inf in sample {idx}, modality '{self.mode}'")
                return None
            return tensor, label

    def __len__(self):
        return len(self.labels)


class BalancedBatchSampler(torch.utils.data.sampler.Sampler):
    """Sampler that oversamples minority classes so each class appears equally per epoch."""

    def __init__(self, dataset, labels):
        self.labels = labels
        self.dataset = {}
        self.balanced_max = 0

        for idx in range(len(dataset)):
            label = self.labels[idx].item()
            self.dataset.setdefault(label, []).append(idx)
            self.balanced_max = max(self.balanced_max, len(self.dataset[label]))

        # Oversample minority classes to match the largest class
        for label in self.dataset:
            while len(self.dataset[label]) < self.balanced_max:
                self.dataset[label].append(random.choice(self.dataset[label]))

        self.keys = list(self.dataset.keys())
        self.currentkey = 0
        self.indices = [-1] * len(self.keys)

    def __iter__(self):
        while self.indices[self.currentkey] < self.balanced_max - 1:
            self.indices[self.currentkey] += 1
            yield self.dataset[self.keys[self.currentkey]][self.indices[self.currentkey]]
            self.currentkey = (self.currentkey + 1) % len(self.keys)
        self.indices = [-1] * len(self.keys)

    def __len__(self):
        return self.balanced_max * len(self.keys)
