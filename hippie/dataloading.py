import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import torch.utils.data

is_torchvision_installed = True
try:
    import torchvision
except:
    is_torchvision_installed = False
import torch.utils.data
import random


def extend_sequence_interpolation(tensor, target_size):
    """
    Extend a sequence to target_size by interpolating additional points within the existing boundaries.
    If the sequence is already longer than target_size, it will be downsampled.
    
    Args:
        tensor: Input tensor of shape (batch, channels, length) or (batch, length)  
        target_size: Desired length
        
    Returns:
        Extended/resampled tensor with target_size length
    """
    if tensor.dim() == 2:
        # Add channel dimension if missing
        tensor = tensor.unsqueeze(1)
    
    current_size = tensor.size(-1)
    
    if current_size == target_size:
        return tensor
    else:
        # For both extension and downsampling, F.interpolate works well
        # The key is that it maintains the signal boundaries and interpolates within them
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
            # waveform = (waveform - waveform.mean()) / waveform.std()
            # 0 1 normalization
            min_val = np.min(waveform)
            max_val = np.max(waveform)
            waveform = (waveform - min_val) / (max_val - min_val)
            # Scale to range [-1, 1]
            waveform = waveform * 2 - 1
            # waveform = (waveform - waveform.min()) / (waveform.max() - waveform.min())
            isi_dist = (isi_dist - isi_dist.mean()) / isi_dist.std()

        waveform = waveform.view(1, 1, -1)
        # waveform = waveform.view(1,-1)
        waveform = extend_sequence_interpolation(waveform, 50).view(1, -1)

        isi_dist = isi_dist.view(1, 1, -1)
        # isi_dist = isi_dist.view(1,-1)
        isi_dist = extend_sequence_interpolation(isi_dist, 100).view(1, -1)

        if self.mode == "wave":
            return waveform, -1
        elif self.mode == "time":
            return isi_dist, -1
        elif self.mode == "both":
            return waveform, isi_dist

    def __len__(self):
        return len(self.waveforms)


class MultiModalEphysDataset(Dataset):
    """Dataset for multiple modalities with consistent preprocessing."""
    def __init__(self, data_dict, labels, mode="multi", normalize=True, modality_sizes=None):
        """
        Initialize a multi-modal ephys dataset.
        
        Args:
            data_dict (dict): Dictionary mapping modality names to data arrays
            labels (np.ndarray): Array of labels
            mode (str): Either one of the modality keys or "multi"
            normalize (bool): Whether to perform normalization
            modality_sizes (dict, optional): Target sizes for each modality
        """
        self.modalities = {}
        for mod_name, data in data_dict.items():
            self.modalities[mod_name] = np.array(data)
        
        self.labels = np.array(labels)
        
        if mode != "multi" and mode not in self.modalities:
            print(f"Mode '{mode}' must be 'multi' or one of the modality keys: {list(self.modalities.keys())}")
        self.mode = mode
        
        n_samples = len(self.labels)
        for mod_name, data in self.modalities.items():
            assert len(data) == n_samples, f"Modality {mod_name} has {len(data)} samples but expected {n_samples}"
        
        default_sizes = {
            "wave": 50,
            "isi": 100,
            "acg": 100,
        }
        
        self.modality_sizes = default_sizes.copy()
        if modality_sizes:
            self.modality_sizes.update(modality_sizes)
        
        self.normalize = normalize
    
    def process_modality(self, mod_name, data, idx):
        """Process a single modality sample."""
        tensor = torch.as_tensor(data[idx, ...]).float()
        
        # Check for NaN or infinite values in raw data
        # if torch.isnan(tensor).any():
        #     print(f"NaN values detected in raw data for sample {idx}, modality '{mod_name}'")
            
        # if torch.isinf(tensor).any():
        #     print(f"Infinite values detected in raw data for sample {idx}, modality '{mod_name}'")
            
        # Apply log transform for ISI
        if mod_name == "isi":
            tensor = torch.log(tensor + 1)
            
        # Check after log transform
        # if torch.isnan(tensor).any():
        #     print(f"NaN values detected after log transform for sample {idx}, modality '{mod_name}'")
            
        # Apply normalization
        if self.normalize:
            min_val = tensor.min()
            max_val = tensor.max()
            
            tensor = (tensor - min_val) / (max_val - min_val + 1e-8)
            tensor = tensor * 2 - 1
            
            # if torch.isnan(tensor).any():
            #     print(f"NaN values detected after normalization for sample {idx}, modality '{mod_name}'")
            
        tensor = tensor.view(1, 1, -1)
        
        target_size = self.modality_sizes.get(mod_name)
        if target_size:
            tensor = extend_sequence_interpolation(tensor, target_size)
            
            # if torch.isnan(tensor).any():
            #     print(f"NaN values detected after interpolation for sample {idx}, modality '{mod_name}'")
            
        tensor = tensor.view(1, -1)
        return tensor

    def __getitem__(self, idx):
        label = torch.as_tensor(self.labels[idx]).long()
        
        if self.mode == "multi":
            result = {}
            for mod_name, data in self.modalities.items():
                result[mod_name] = self.process_modality(mod_name, data, idx)
                
                if torch.isnan(result[mod_name]).any() or torch.isinf(result[mod_name]).any():
                    print(f"NaN values detected in sample {idx}, modality '{mod_name}'")
                    return None
            return result, label
        else:
            tensor = self.process_modality(self.mode, self.modalities[self.mode], idx)
            
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                print(f"NaN values detected in sample {idx}, modality '{self.mode}'")
                return None 
            return tensor, label
        
    def __len__(self):
        return len(self.labels)


class BalancedBatchSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, labels=None):
        self.labels = labels
        self.dataset = dict()
        self.balanced_max = 0
        # Save all the indices for all the classes
        for idx in range(0, len(dataset)):
            label = self._get_label(dataset, idx)
            if label not in self.dataset:
                self.dataset[label] = list()
            self.dataset[label].append(idx)
            self.balanced_max = (
                len(self.dataset[label]) if len(self.dataset[label]) > self.balanced_max else self.balanced_max
            )

        # Oversample the classes with fewer elements than the max
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

    def _get_label(self, dataset, idx, labels=None):
        if self.labels is not None:
            return self.labels[idx].item()
        else:
            # Trying guessing
            dataset_type = type(dataset)
            if is_torchvision_installed and dataset_type is torchvision.datasets.MNIST:
                return dataset.train_labels[idx].item()
            elif is_torchvision_installed and dataset_type is torchvision.datasets.ImageFolder:
                return dataset.imgs[idx][1]
            else:
                raise Exception("You should pass the tensor of labels to the constructor as second argument")

    def __len__(self):
        return self.balanced_max * len(self.keys)
