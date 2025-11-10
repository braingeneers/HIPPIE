import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import random
from typing import Dict, Any, Optional
from scipy import ndimage


class NeuralAugmentations:
    """Data augmentation methods for 1D neural electrophysiology signals."""
    
    @staticmethod
    def add_gaussian_noise(signal, noise_std=0.05):
        """Add Gaussian noise to signal."""
        if noise_std <= 0:
            return signal
        noise = torch.randn_like(signal) * noise_std
        return signal + noise
    
    @staticmethod
    def amplitude_scaling(signal, scale_range=(0.8, 1.2)):
        """Scale signal amplitude by random factor."""
        scale_min, scale_max = scale_range
        if scale_min >= scale_max:
            return signal
        scale_factor = torch.rand(1) * (scale_max - scale_min) + scale_min
        return signal * scale_factor
    
    @staticmethod
    def time_warping(signal, warp_strength=0.1):
        """Apply non-linear time warping to signal."""
        if warp_strength <= 0:
            return signal
            
        length = signal.shape[-1]
        # Create warping grid
        warp_amount = int(length * warp_strength)
        if warp_amount < 2:
            return signal
            
        # Generate random warp points
        num_warp_points = max(2, length // 10)
        warp_points = torch.linspace(0, length-1, num_warp_points)
        warp_offsets = (torch.rand(num_warp_points) - 0.5) * 2 * warp_amount
        
        # Interpolate to create smooth warping
        indices = torch.arange(length, dtype=torch.float32)
        
        # Manual 1D interpolation since torch.interp doesn't exist
        warped_indices = torch.zeros_like(indices)
        for i, idx in enumerate(indices):
            # Find surrounding warp points
            left_idx = torch.searchsorted(warp_points, idx, right=False)
            left_idx = torch.clamp(left_idx, 0, len(warp_points) - 1)
            right_idx = torch.clamp(left_idx + 1, 0, len(warp_points) - 1)
            
            if left_idx == right_idx:
                warped_indices[i] = warp_points[left_idx] + warp_offsets[left_idx]
            else:
                # Linear interpolation
                t = (idx - warp_points[left_idx]) / (warp_points[right_idx] - warp_points[left_idx] + 1e-8)
                warped_point = (1 - t) * warp_points[left_idx] + t * warp_points[right_idx]
                warped_offset = (1 - t) * warp_offsets[left_idx] + t * warp_offsets[right_idx]
                warped_indices[i] = warped_point + warped_offset
        
        warped_indices = torch.clamp(warped_indices, 0, length-1)
        
        # Apply warping using interpolation
        signal_1d = signal.view(-1)
        warped_signal = torch.zeros_like(signal_1d)
        
        for i in range(length):
            idx = warped_indices[i]
            idx_floor = int(torch.floor(idx))
            idx_ceil = min(idx_floor + 1, length - 1)
            alpha = idx - idx_floor
            
            warped_signal[i] = (1 - alpha) * signal_1d[idx_floor] + alpha * signal_1d[idx_ceil]
        
        return warped_signal.view_as(signal)
    
    @staticmethod
    def baseline_shift(signal, shift_range=(-0.1, 0.1)):
        """Add random baseline shift to signal."""
        shift_min, shift_max = shift_range
        if shift_min >= shift_max:
            return signal
        shift = torch.rand(1) * (shift_max - shift_min) + shift_min
        return signal + shift
    
    @staticmethod
    def gaussian_smoothing(signal, sigma_range=(0.5, 2.0)):
        """Apply Gaussian smoothing to signal."""
        sigma_min, sigma_max = sigma_range
        if sigma_min >= sigma_max or sigma_min <= 0:
            return signal
            
        sigma = torch.rand(1) * (sigma_max - sigma_min) + sigma_min
        
        # Convert to numpy for scipy filtering
        signal_np = signal.detach().cpu().numpy()
        smoothed_np = ndimage.gaussian_filter1d(signal_np, sigma=sigma.item(), axis=-1)
        
        return torch.from_numpy(smoothed_np).to(signal.device)
    
    @staticmethod
    def mixup(signal1, signal2, alpha=0.2):
        """Apply mixup between two signals."""
        if alpha <= 0:
            return signal1
        lam = np.random.beta(alpha, alpha)
        return lam * signal1 + (1 - lam) * signal2


class AugmentedMultiModalEphysDataset(Dataset):
    """Wrapper dataset that applies augmentations to MultiModalEphysDataset."""
    
    def __init__(self, base_dataset, config, phase="pretraining"):
        """
        Initialize augmented dataset wrapper.
        
        Args:
            base_dataset: Original MultiModalEphysDataset instance
            config: CVAEConfig with augmentation parameters
            phase: Training phase ("pretraining", "finetuning", "supervised")
        """
        self.base_dataset = base_dataset
        self.config = config
        self.phase = phase
        self.training = True  # Default to training mode
        
        # Determine if augmentations should be applied for this phase
        self.augment_enabled = (
            config.use_augmentations and 
            getattr(config, f"augment_{phase}", False)
        )
        
        self.augmentations = NeuralAugmentations()
    
    def train(self):
        """Set dataset to training mode (augmentations enabled)."""
        self.training = True
        return self
    
    def eval(self):
        """Set dataset to evaluation mode (augmentations disabled)."""
        self.training = False
        return self
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        # Get original sample from base dataset
        sample = self.base_dataset[idx]
        
        # If not in training mode or augmentations disabled, return original
        if not (self.training and self.augment_enabled):
            return sample
        
        # Apply augmentations with probability
        if random.random() > self.config.augment_prob:
            return sample
        
        # Extract data and label
        if isinstance(sample, tuple) and len(sample) == 2:
            data, label = sample
        else:
            return sample
        
        # Apply augmentations to each modality
        if isinstance(data, dict):
            # Multi-modal case
            augmented_data = {}
            for modality, signal in data.items():
                augmented_data[modality] = self._augment_signal(signal, modality)
            return augmented_data, label
        else:
            # Single modality case
            augmented_signal = self._augment_signal(data, "single")
            return augmented_signal, label
    
    def _augment_signal(self, signal, modality):
        """Apply augmentations to a single signal."""
        augmented = signal.clone()
        
        # Apply different augmentations based on config
        if random.random() < 0.5:  # 50% chance for each augmentation
            augmented = self.augmentations.add_gaussian_noise(
                augmented, self.config.noise_std
            )
        
        if random.random() < 0.5:
            augmented = self.augmentations.amplitude_scaling(
                augmented, self.config.amplitude_scale_range
            )
        
        if random.random() < 0.3:  # Less frequent time warping
            augmented = self.augmentations.time_warping(
                augmented, warp_strength=0.05
            )
        
        if random.random() < 0.4:
            augmented = self.augmentations.baseline_shift(
                augmented, shift_range=(-0.05, 0.05)
            )
        
        if random.random() < 0.3:
            augmented = self.augmentations.gaussian_smoothing(
                augmented, self.config.smoothing_sigma_range
            )
        
        return augmented