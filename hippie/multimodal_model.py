import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl

from .backbones import ResNet18Enc, ResNet18Dec

logger = logging.getLogger(__name__)

@dataclass
class CVAEConfig:
    """Configuration class for MultiModal CVAE ablation studies."""
    use_source_embedding: bool = True
    use_class_embedding: bool = True
    use_super_region_embedding: bool = False  # for hierarchical brain-region conditioning (available at inference)
    use_layer_embedding: bool = False          # E1-cond4: cortical/cerebellar layer (PCL/GCL/ML/L4/L5…)
    # If False, the encoder ignores class_labels (forces them to None internally) while the
    # decoder still uses them — asymmetric "decoder-only" conditional variant. Defaults True
    # to preserve the original paper's behavior for every existing config.
    encoder_uses_class_embedding: bool = True
    # If False, the encoder ignores region labels (decoder-only region conditioning).
    # Use for cross-species transfer to prevent region→cell-type shortcuts in the embedding.
    encoder_uses_region_embedding: bool = True
    use_fusion_encoder: bool = True
    use_batch_norm: bool = True
    # β=1.0 locked after hyperparameter sweep on Hausser dataset (42 configs swept).
    # β=0.9 was the pre-sweep default; 1.0 was frozen before evaluating any other dataset.
    beta: float = 1.0
    # Free bits (Kingma et al. 2016, IAF §2.3): floor, in nats, on each latent
    # dimension's batch-averaged KL. Dimensions already above the floor are
    # regularized normally; dimensions below it stop paying a KL penalty, which
    # removes the gradient pressure that collapses them onto the prior.
    # 0.0 disables it — every preset below keeps the paper's exact objective.
    free_bits: float = 0.0
    fusion_layers: Optional[List[int]] = None
    class_hidden_dim: int = 5

    # Augmentation parameters
    use_augmentations: bool = False
    augment_pretraining: bool = True
    augment_finetuning: bool = True
    augment_supervised: bool = False
    augment_prob: float = 0.5
    noise_std: float = 0.05
    amplitude_scale_range: tuple = (0.8, 1.2)
    smoothing_sigma_range: tuple = (0.5, 2.0)

    # Regularization to prevent over-reliance on class embeddings
    class_embedding_dropout: float = 0.3  # Dropout probability for class embeddings during training
    embedding_warmup_epochs: int = 5  # Epochs to warmup embedding usage
    reconstruction_consistency_weight: float = 0.1  # Weight for consistency loss

    # Optional supervised contrastive regularizer on the encoder latent mean.
    # This is off for all legacy HIPPIE configs and enabled by HIPPIE_contrastive.
    use_contrastive_loss: bool = False
    contrastive_weight: float = 0.0
    contrastive_temperature: float = 0.5

class ExperimentConfigs:
    """Predefined configurations for common ablation studies."""
    
    @staticmethod
    def baseline():
        """Baseline: Basic VAE without any conditional components.
        No embeddings, no fusion, no batch norm, no augmentations, no regularization."""
        return CVAEConfig(
            use_source_embedding=False,
            use_class_embedding=False,
            use_fusion_encoder=False,
            use_batch_norm=False,
            use_augmentations=False,
            class_embedding_dropout=0.0,
            reconstruction_consistency_weight=0.0,
            embedding_warmup_epochs=0
        )
    
    @staticmethod
    def with_source():
        """Baseline + source embeddings.
        Has fusion encoder to process source embeddings, but no other features."""
        return CVAEConfig(
            use_source_embedding=True,
            use_class_embedding=False,
            use_fusion_encoder=True,
            use_batch_norm=False,
            use_augmentations=False,
            class_embedding_dropout=0.0,
            reconstruction_consistency_weight=0.0,
            embedding_warmup_epochs=0
        )

    @staticmethod
    def with_class():
        """Baseline + class embeddings.
        Has fusion encoder to process class embeddings, but no other features."""
        return CVAEConfig(
            use_source_embedding=False,
            use_class_embedding=True,
            use_fusion_encoder=True,
            use_batch_norm=False,
            use_augmentations=False,
            class_embedding_dropout=0.0,
            reconstruction_consistency_weight=0.0,
            embedding_warmup_epochs=0
        )
    
    @staticmethod
    def with_both_embeddings():
        """Model with both source and class embeddings + fusion encoder.
        No batch norm, no augmentations, no regularization."""
        return CVAEConfig(
            use_source_embedding=True,
            use_class_embedding=True,
            use_fusion_encoder=True,
            use_batch_norm=False,
            use_augmentations=False,
            class_embedding_dropout=0.0,
            reconstruction_consistency_weight=0.0,
            embedding_warmup_epochs=0
        )
    
    @staticmethod
    def with_batch_norm():
        """Model with batch normalization + light augmentations.
        Has source & class embeddings, fusion encoder, no regularization."""
        return CVAEConfig(
            use_source_embedding=True,
            use_class_embedding=True,
            use_fusion_encoder=True,
            use_batch_norm=True,
            use_augmentations=True,
            augment_pretraining=True,
            augment_finetuning=True,
            augment_supervised=False,
            augment_prob=0.3,
            noise_std=0.03,
            amplitude_scale_range=(0.9, 1.1),
            smoothing_sigma_range=(0.5, 1.5),
            class_embedding_dropout=0.0,
            reconstruction_consistency_weight=0.0,
            embedding_warmup_epochs=0
        )
    
    @staticmethod
    def no_augmentations():
        """Full conditional architecture without augmentations or the augmentation-coupled regularization.
        Has all components: source & class embeddings, fusion encoder, batch norm.
        Ablation rung below full_architecture: removes augmentations, class-embedding dropout, and consistency loss."""
        return CVAEConfig(
            use_source_embedding=True,
            use_class_embedding=True,
            use_fusion_encoder=True,
            use_batch_norm=True,
            use_augmentations=False,
            class_embedding_dropout=0.0,
            reconstruction_consistency_weight=0.0,
            embedding_warmup_epochs=0
        )
    
    @staticmethod
    def no_fusion():
        """Model with source & class embeddings but no fusion encoder.
        No augmentations, no regularization."""
        return CVAEConfig(
            use_source_embedding=True,
            use_class_embedding=True,
            use_fusion_encoder=False,
            use_batch_norm=False,
            use_augmentations=False,
            class_embedding_dropout=0.0,
            reconstruction_consistency_weight=0.0,
            embedding_warmup_epochs=0
        )
    
    @staticmethod
    def with_light_augmentations():
        """Baseline model with light/conservative augmentations.
        No embeddings, no fusion, just data augmentation."""
        return CVAEConfig(
            use_source_embedding=False,
            use_class_embedding=False,
            use_fusion_encoder=False,
            use_batch_norm=False,
            use_augmentations=True,
            augment_pretraining=True,
            augment_finetuning=True,
            augment_supervised=False,
            augment_prob=0.3,
            noise_std=0.03,
            amplitude_scale_range=(0.9, 1.1),
            smoothing_sigma_range=(0.5, 1.5),
            class_embedding_dropout=0.0,
            reconstruction_consistency_weight=0.0,
            embedding_warmup_epochs=0
        )
    
    @staticmethod
    def with_heavy_augmentations():
        """Model with source & class embeddings + heavy/aggressive augmentations.
        No fusion encoder, no regularization."""
        return CVAEConfig(
            use_source_embedding=True,
            use_class_embedding=True,
            use_fusion_encoder=False,
            use_batch_norm=False,
            use_augmentations=True,
            augment_pretraining=True,
            augment_finetuning=True,
            augment_supervised=True,
            augment_prob=0.7,
            noise_std=0.08,
            amplitude_scale_range=(0.7, 1.3),
            smoothing_sigma_range=(0.5, 3.0),
            class_embedding_dropout=0.0,
            reconstruction_consistency_weight=0.0,
            embedding_warmup_epochs=0
        )
    
    @staticmethod
    def full_architecture():
        """The full HIPPIE model: conditional architecture + light augmentations + regularization.
        Everything enabled — this is the production configuration used for benchmarking."""
        return CVAEConfig(
            use_source_embedding=True,
            use_class_embedding=True,
            use_fusion_encoder=True,
            use_batch_norm=True,
            use_augmentations=True,
            augment_pretraining=True,
            augment_finetuning=True,
            augment_supervised=False,  # Light augmentations (not supervised)
            augment_prob=0.3,  # Light augmentation probability
            noise_std=0.03,  # Light noise
            amplitude_scale_range=(0.9, 1.1),  # Light amplitude scaling
            smoothing_sigma_range=(0.5, 1.5),  # Light smoothing
            class_embedding_dropout=0.3,  # Regularization
            reconstruction_consistency_weight=0.15,  # Regularization
            embedding_warmup_epochs=5  # Regularization
        )

    @staticmethod
    def conditional_decoder_only():
        """Asymmetric CVAE: encoder never sees the class label, decoder does.
        Targets the post-leakage-fix spatial-drift failure mode: configs that put the class
        embedding into the encoder drift to a single "trap" class at test time when the label
        is masked. Here the encoder path is class-agnostic by construction, so train/test
        behavior matches and no drift is possible. The class embedding is still active in the
        decoder during training, so reconstruction is class-conditioned and z is free from the
        burden of encoding class identity."""
        return CVAEConfig(
            use_source_embedding=True,
            use_class_embedding=True,
            encoder_uses_class_embedding=False,  # <- the single bit that matters
            use_fusion_encoder=True,
            use_batch_norm=True,
            use_augmentations=True,
            augment_pretraining=True,
            augment_finetuning=True,
            augment_supervised=False,
            augment_prob=0.3,
            noise_std=0.03,
            amplitude_scale_range=(0.9, 1.1),
            smoothing_sigma_range=(0.5, 1.5),
            # No need for dropout/consistency regularization since the encoder never sees the
            # label; train/test mismatch is zero.
            class_embedding_dropout=0.0,
            reconstruction_consistency_weight=0.0,
            embedding_warmup_epochs=0,
        )

    @staticmethod
    def full_architecture_heavy_reg():
        """full_architecture with heavier regularization against class-label dependency.
        Tests whether the existing train/test alignment regularizers just need to be cranked
        up to prevent the spatial drift observed in post-leakage-fix ablations."""
        return CVAEConfig(
            use_source_embedding=True,
            use_class_embedding=True,
            use_fusion_encoder=True,
            use_batch_norm=True,
            use_augmentations=True,
            augment_pretraining=True,
            augment_finetuning=True,
            augment_supervised=False,
            augment_prob=0.3,
            noise_std=0.03,
            amplitude_scale_range=(0.9, 1.1),
            smoothing_sigma_range=(0.5, 1.5),
            class_embedding_dropout=0.7,            # 0.3 -> 0.7
            reconstruction_consistency_weight=0.5,  # 0.15 -> 0.5
            embedding_warmup_epochs=10,             # 5 -> 10
        )

    # ----- Paper Figure 2 ladder rungs (decoder-only conditional variants) -----
    #
    # The asymmetric CVAE fix (encoder never sees the class label) forms the base; the rungs
    # below build it up layer by layer. All four use encoder_uses_class_embedding=False.

    @staticmethod
    def class_decoder_source():
        """Ladder rung 4: source + class-in-decoder (no BN, no aug, no reg).
        Direct comparison to with_both_embeddings (rung 3): same inputs, but class embedding
        is only applied in the decoder path."""
        return CVAEConfig(
            use_source_embedding=True,
            use_class_embedding=True,
            encoder_uses_class_embedding=False,
            use_fusion_encoder=True,
            use_batch_norm=False,
            use_augmentations=False,
            class_embedding_dropout=0.0,
            reconstruction_consistency_weight=0.0,
            embedding_warmup_epochs=0,
        )

    @staticmethod
    def class_decoder_source_bn():
        """Ladder rung 5: rung 4 + batch normalization."""
        return CVAEConfig(
            use_source_embedding=True,
            use_class_embedding=True,
            encoder_uses_class_embedding=False,
            use_fusion_encoder=True,
            use_batch_norm=True,
            use_augmentations=False,
            class_embedding_dropout=0.0,
            reconstruction_consistency_weight=0.0,
            embedding_warmup_epochs=0,
        )

    @staticmethod
    def class_decoder_source_bn_strong_aug():
        """Ladder rung 7: rung 5 + strong augmentations.
        Strong aug parameters match with_heavy_augmentations so the strong/light distinction is
        defined consistently across the two augmentation variants."""
        return CVAEConfig(
            use_source_embedding=True,
            use_class_embedding=True,
            encoder_uses_class_embedding=False,
            use_fusion_encoder=True,
            use_batch_norm=True,
            use_augmentations=True,
            augment_pretraining=True,
            augment_finetuning=True,
            augment_supervised=True,            # strong: augment supervised phase too
            augment_prob=0.7,                   # strong: higher prob
            noise_std=0.08,                     # strong: more noise
            amplitude_scale_range=(0.7, 1.3),   # strong: wider amplitude
            smoothing_sigma_range=(0.5, 3.0),   # strong: wider smoothing
            class_embedding_dropout=0.0,
            reconstruction_consistency_weight=0.0,
            embedding_warmup_epochs=0,
        )

    @staticmethod
    def class_decoder_source_bn_aug_reg():
        """Ladder rung 8 and the production default for the 11-dataset benchmark.

        Why this is the production configuration:

        1. Class-agnostic encoder (encoder_uses_class_embedding=False): the encoder path
           contains no class embedding, so there is no train/test mismatch when class labels
           are unavailable at inference. An earlier default put the class embedding into the
           encoder and relied on regularizers to absorb the mismatch; under 5-fold
           cross-validation on the two tuning datasets that approach reached mean balanced
           accuracy 0.468 (cellexplorer 0.460, hausser 0.475), whereas this configuration
           reaches 0.635 (cellexplorer 0.557, hausser 0.713), a +0.17 mean improvement from
           moving class conditioning to the decoder while keeping the same regularizers.

        2. Hausser is the representative tuning dataset: a dataset-level cosine-distance
           analysis places hausser 2nd-closest to the 11-dataset center (mean distance 0.475)
           while cellexplorer is a mild outlier (9th of 11, 0.707). This configuration's
           +0.069 hausser advantage over the simpler class_decoder_source_bn (rung 5) is
           therefore more likely to generalize than rung 5's +0.044 cellexplorer advantage.

        3. The decoder-side regularization (class_embedding_dropout=0.3,
           reconstruction_consistency_weight=0.15, embedding_warmup_epochs=5) reuses the
           original values without refitting. Because encoder_uses_class_embedding=False,
           these regularizers act entirely on the decoder path: the dropout regularizes the
           decoder's class-embedding input, and the consistency loss pushes the decoder to
           produce similar reconstructions with and without the class embedding. The
           downstream KNN probe only touches the encoder's z, so they regularize without
           affecting the evaluation path.

        Alternative: class_decoder_source_bn (rung 5) is the minimal variant without
        augmentation or regularization. It is strictly better on cellexplorer and slightly
        faster to train (10.0 vs 13.4 min/fold); rung 8 is selected for its better expected
        generalization (see point 2)."""
        return CVAEConfig(
            use_source_embedding=True,
            use_class_embedding=True,
            encoder_uses_class_embedding=False,
            use_fusion_encoder=True,
            use_batch_norm=True,
            use_augmentations=True,
            augment_pretraining=True,
            augment_finetuning=True,
            augment_supervised=False,
            augment_prob=0.3,
            noise_std=0.03,
            amplitude_scale_range=(0.9, 1.1),
            smoothing_sigma_range=(0.5, 1.5),
            class_embedding_dropout=0.3,
            reconstruction_consistency_weight=0.15,
            embedding_warmup_epochs=5,
        )

    @staticmethod
    def HIPPIE_contrastive():
        """Contrastive upgrade over the locked HIPPIE production config.

        Starts from class_decoder_source_bn_aug_reg (the current best HIPPIE
        architecture) and adds a supervised contrastive loss on the encoder
        latent mean. The class label remains decoder-only for reconstruction,
        so the anti-drift behavior is preserved; the contrastive term is the
        explicit discriminative pressure missing from the original cVAE setup.
        """
        cfg = ExperimentConfigs.class_decoder_source_bn_aug_reg()
        cfg.use_contrastive_loss = True
        cfg.contrastive_weight = 0.1
        cfg.contrastive_temperature = 0.5
        return cfg

    @staticmethod
    def unconditioned():
        """Winner architecture with all conditioning disabled.

        Identical to class_decoder_source_bn_aug_reg except both source and
        class embeddings are turned off, so neither the encoder nor the decoder
        receives any conditioning signal.  The model is a pure multimodal VAE
        that learns to compress waveform + ISI + ACG into a shared latent space
        without any label or technology supervision.

        Use this when you want unsupervised data compression / dimensionality
        reduction on a new dataset and do not have cell-type or technology labels.
        """
        return CVAEConfig(
            use_source_embedding=False,
            use_class_embedding=False,
            encoder_uses_class_embedding=False,
            use_fusion_encoder=True,
            use_batch_norm=True,
            use_augmentations=True,
            augment_pretraining=True,
            augment_finetuning=True,
            augment_supervised=False,
            augment_prob=0.3,
            noise_std=0.03,
            amplitude_scale_range=(0.9, 1.1),
            smoothing_sigma_range=(0.5, 1.5),
            class_embedding_dropout=0.0,
            embedding_warmup_epochs=0,
            reconstruction_consistency_weight=0.0,
            beta=1.0,
        )

class MultiModalCVAE(nn.Module):
    """Multimodal Conditional Variational Autoencoder model for joint processing of multiple modalities."""
    
    def __init__(self, modalities, z_dim, config: CVAEConfig, num_sources=None, num_classes=None, num_super_regions=None, num_layers=None, backbone_base_width=64):
        """
        Initialize the MultiModal CVAE with configurable components.

        modalities (dict): Dictionary mapping modality names to their output sizes
        z_dim (int): Dimension of the latent space
        config (CVAEConfig): Configuration object specifying which components to use
        num_sources (int, optional): Number of sources for source embedding
        num_classes (int, optional): Number of classes for class embedding
        num_super_regions (int, optional): Number of super regions for super_region embedding
        backbone_base_width (int): Base channel width for ResNet-18 encoder/decoder.
            Default 64 matches vanilla ResNet-18. Use 32 for ~0.25× params, 128 for ~4×.
        """
        super().__init__()
        self.z_dim = z_dim
        self.config = config
        self.class_hidden_dim = config.class_hidden_dim
        self.num_sources = num_sources
        self.num_classes = num_classes
        self.num_super_regions = num_super_regions
        self.num_layers = num_layers
        self.modalities = modalities
        self.num_modalities = len(modalities)
        self.embedding_dropout = nn.Dropout(config.class_embedding_dropout)

        self.encoders = nn.ModuleDict({
            mod_name: ResNet18Enc(z_dim=z_dim, base_width=backbone_base_width)
            for mod_name in modalities.keys()
        })

        embedding_dim = 0
        if config.use_source_embedding and num_sources:
            self.source_embedding = nn.Embedding(num_sources, self.class_hidden_dim)
            embedding_dim += self.class_hidden_dim
        else:
            self.source_embedding = None

        if config.use_class_embedding and num_classes:
            self.class_embedding = nn.Embedding(num_classes, self.class_hidden_dim)
            embedding_dim += self.class_hidden_dim
        else:
            self.class_embedding = None

        # Super region embedding (available at inference time, unlike class labels)
        if config.use_super_region_embedding and num_super_regions:
            self.super_region_embedding = nn.Embedding(num_super_regions, self.class_hidden_dim)
            embedding_dim += self.class_hidden_dim
        else:
            self.super_region_embedding = None

        # Layer embedding (E1-cond4: cortical/cerebellar layer, available at inference)
        if config.use_layer_embedding and num_layers:
            self.layer_embedding = nn.Embedding(num_layers, self.class_hidden_dim)
            embedding_dim += self.class_hidden_dim
        else:
            self.layer_embedding = None
        
        if config.use_fusion_encoder:
            fusion_input_size = (z_dim * 2) * self.num_modalities + embedding_dim
            
            if config.fusion_layers:
                layers = []
                input_size = fusion_input_size
                for hidden_size in config.fusion_layers:
                    layers.append(nn.Linear(input_size, hidden_size))
                    if config.use_batch_norm:
                        layers.append(nn.BatchNorm1d(hidden_size))
                    layers.append(nn.LeakyReLU(0.2))
                    input_size = hidden_size
                layers.append(nn.Linear(input_size, z_dim))
                self.fusion_encoder = nn.Sequential(*layers)
            else:
                layers = [
                    nn.Linear(fusion_input_size, z_dim * 2),
                ]
                if config.use_batch_norm:
                    layers.append(nn.BatchNorm1d(z_dim * 2))
                layers.extend([
                    nn.LeakyReLU(0.2),
                    nn.Linear(z_dim * 2, z_dim),
                ])
                self.fusion_encoder = nn.Sequential(*layers)
        else:
            fusion_input_size = (z_dim * 2) * self.num_modalities + embedding_dim
            self.fusion_encoder = nn.Linear(fusion_input_size, z_dim)

        self.z_mean = nn.Linear(z_dim, z_dim)
        self.z_log_var = nn.Linear(z_dim, z_dim)

        decoder_input_size = z_dim + embedding_dim
        self.decoder_fcs = nn.ModuleDict({
            mod_name: self._build_decoder_fc(decoder_input_size, z_dim * 2, config.use_batch_norm)
            for mod_name in modalities.keys()
        })
        
        self.decoders = nn.ModuleDict({
            mod_name: ResNet18Dec(z_dim=z_dim, output_size=output_size, base_width=backbone_base_width)
            for mod_name, output_size in modalities.items()
        })

    def _build_decoder_fc(self, input_size, hidden_size, use_batch_norm):
        """Build decoder fully connected layers based on configuration."""
        layers = [
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, hidden_size),
        ]
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_size))
        layers.append(nn.LeakyReLU(0.2))
        return nn.Sequential(*layers)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _get_embeddings(self, source_labels=None, class_labels=None, super_region_labels=None, layer_labels=None, apply_dropout=False):
        """Get source, class, super_region, and layer embeddings if they exist.

        Args:
            source_labels: Source/tech labels tensor
            class_labels: Class labels tensor
            super_region_labels: Brain-region labels tensor (available at inference)
            layer_labels: Cortical/cerebellar layer labels tensor (E1-cond4)
            apply_dropout: Whether to apply dropout to class embeddings (training only)
        """
        embeddings = []

        # Determine batch size from available labels
        if source_labels is not None:
            batch_size = source_labels.shape[0]
        elif class_labels is not None:
            batch_size = class_labels.shape[0]
        elif super_region_labels is not None:
            batch_size = super_region_labels.shape[0]
        elif layer_labels is not None:
            batch_size = layer_labels.shape[0]
        else:
            batch_size = 1

        if self.source_embedding is not None:
            if source_labels is not None:
                source_emb = self.source_embedding(source_labels)
            else:
                source_emb = torch.zeros(batch_size, self.class_hidden_dim, device=self.source_embedding.weight.device)
            embeddings.append(source_emb)

        if self.class_embedding is not None:
            if class_labels is not None:
                class_labels = class_labels.long()
                valid = class_labels >= 0
                class_emb = torch.zeros(
                    batch_size,
                    self.class_hidden_dim,
                    device=self.class_embedding.weight.device,
                )
                if valid.any():
                    class_emb[valid] = self.class_embedding(class_labels[valid])
                if apply_dropout and self.training:
                    class_emb = self.embedding_dropout(class_emb)
            else:
                class_emb = torch.zeros(batch_size, self.class_hidden_dim, device=self.class_embedding.weight.device)
            embeddings.append(class_emb)

        # Super region embedding (always available at inference, unlike class labels)
        if self.super_region_embedding is not None:
            if super_region_labels is not None:
                super_region_emb = self.super_region_embedding(super_region_labels)
            else:
                super_region_emb = torch.zeros(batch_size, self.class_hidden_dim, device=self.super_region_embedding.weight.device)
            embeddings.append(super_region_emb)

        # Layer embedding (E1-cond4: cortical/cerebellar layer)
        if self.layer_embedding is not None:
            if layer_labels is not None:
                layer_emb = self.layer_embedding(layer_labels)
            else:
                layer_emb = torch.zeros(batch_size, self.class_hidden_dim, device=self.layer_embedding.weight.device)
            embeddings.append(layer_emb)

        if embeddings:
            return torch.cat(embeddings, dim=1)
        else:
            return None

    def encode(self, data_dict, source_labels=None, class_labels=None, super_region_labels=None, layer_labels=None, apply_dropout=False):
        encoded_features = []
        for mod_name, encoder in self.encoders.items():
            h = encoder(data_dict[mod_name])
            encoded_features.append(h)

        combined_features = torch.cat(encoded_features, dim=1)

        # Asymmetric decoder-only conditional variant: encoder path never sees the class label,
        # even during training. The class embedding still lives in the model (the decoder uses
        # it), but _get_embeddings substitutes zeros when class_labels is None.
        encoder_class_labels = class_labels if self.config.encoder_uses_class_embedding else None
        encoder_region_labels = super_region_labels if self.config.encoder_uses_region_embedding else None
        embeddings = self._get_embeddings(source_labels, encoder_class_labels, encoder_region_labels, layer_labels, apply_dropout=apply_dropout)
        if embeddings is not None:
            combined_features = torch.cat([combined_features, embeddings], dim=1)

        if self.fusion_encoder is not None:
            h = self.fusion_encoder(combined_features)
        else:
            h = combined_features

        return h, self.z_mean(h), self.z_log_var(h)

    def decode(self, z, source_labels=None, class_labels=None, super_region_labels=None, layer_labels=None, apply_dropout=False):
        results = {}

        embeddings = self._get_embeddings(source_labels, class_labels, super_region_labels, layer_labels, apply_dropout=apply_dropout)

        for mod_name in self.modalities.keys():
            if embeddings is not None:
                z_mod = torch.cat([z, embeddings], dim=1)
            else:
                z_mod = z
            z_mod = self.decoder_fcs[mod_name](z_mod)
            results[mod_name] = self.decoders[mod_name](z_mod)

        return results

    def forward(self, data_dict, source_labels=None, class_labels=None, super_region_labels=None, layer_labels=None, apply_dropout=False):
        encoded, mu, logvar = self.encode(data_dict, source_labels, class_labels, super_region_labels, layer_labels, apply_dropout=apply_dropout)
        z = self.reparameterize(mu, logvar)
        decoded = self.decode(z, source_labels, class_labels, super_region_labels, layer_labels, apply_dropout=apply_dropout)

        return encoded, mu, logvar, decoded


class MultiModalCVAETrainModule(pl.LightningModule):
    """PyTorch Lightning module for training the MultiModalCVAE model."""
    
    def __init__(self,
                 base_model,
                 config: CVAEConfig,
                 modality_weights=None,
                 learning_rate=0.01,
                 weight_decay=0.01
        ):
        """
        Initialize the training module.

        Args:
            base_model: The MultiModalCVAE model to train
            config: Configuration object containing model architecture parameters
            modality_weights: Dictionary of weights for each modality in loss computation
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
        """
        super().__init__()
        self.model = base_model
        self.config = config
        self.modalities = base_model.modalities
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.mse_loss = nn.MSELoss()
        self.val_loss = []
        self.train_loss = []

        if modality_weights is None:
            self.modality_weights = {mod_name: 1.0 for mod_name in self.modalities.keys()}
        else:
            self.modality_weights = modality_weights

        self.beta = config.beta
        self.free_bits = config.free_bits
        self.reconstruction_consistency_weight = config.reconstruction_consistency_weight
        self.embedding_warmup_epochs = config.embedding_warmup_epochs
        self.use_contrastive_loss = (
            config.use_contrastive_loss and config.contrastive_weight > 0
        )
        self.contrastive_weight = config.contrastive_weight
        self.contrastive_temperature = config.contrastive_temperature
        self.train_epoch_history = []
        self.val_epoch_history = []
        self._train_component_buffer = []
        self._val_component_buffer = []

    def process_batch(self, batch):
        if isinstance(batch[0], dict):
            data_dict, labels = batch
        else:
            data_list, labels = batch[:-1], batch[-1]
            data_dict = {mod_name: data for mod_name, data in zip(self.modalities.keys(), data_list)}
        
        return data_dict, labels

    def _compute_losses(self, data_dict, decoded_dict):
        mse_losses = {}
        for mod_name in self.modalities.keys():
            try:
                mse_losses[mod_name] = F.mse_loss(data_dict[mod_name], decoded_dict[mod_name])
            except Exception as e:
                logger.error(
                    "MSE loss failed for '%s': %s; shapes %s vs %s",
                    mod_name, e, data_dict[mod_name].shape, decoded_dict[mod_name].shape,
                )
                raise

        mse_loss = sum(self.modality_weights[mod_name] * mse_losses[mod_name] 
                      for mod_name in self.modalities.keys())
        
        return mse_loss, mse_losses

    def _forward_model(self, data_dict, labels, apply_dropout=False):
        """Forward pass handling different label configurations.

        Supports 1D (source only), 2D (class + source), 3D (class + source + super_region),
        or 4D (class + source + super_region + layer).
        """
        if labels.ndim == 2:
            if labels.shape[1] == 2:
                # Traditional: (class, source)
                class_labels, source_labels = labels.unbind(1)
                return self.model(data_dict, source_labels=source_labels, class_labels=class_labels, apply_dropout=apply_dropout)
            elif labels.shape[1] == 3:
                # (class, source, super_region)
                class_labels, source_labels, super_region_labels = labels.unbind(1)
                return self.model(data_dict, source_labels=source_labels, class_labels=class_labels,
                                  super_region_labels=super_region_labels, apply_dropout=apply_dropout)
            elif labels.shape[1] == 4:
                # E1-cond4: (class, source/tech, super_region, layer)
                class_labels, source_labels, super_region_labels, layer_labels = labels.unbind(1)
                return self.model(data_dict, source_labels=source_labels, class_labels=class_labels,
                                  super_region_labels=super_region_labels, layer_labels=layer_labels,
                                  apply_dropout=apply_dropout)
        else:
            # Just source labels
            return self.model(data_dict, source_labels=labels, apply_dropout=apply_dropout)

    def _extract_class_labels(self, labels):
        """Return class labels when the batch contains them, else None.

        Negative labels are treated as intentionally masked and are ignored by
        the contrastive objective.
        """
        if labels.ndim == 2 and labels.shape[1] >= 2:
            return labels[:, 0]
        return None

    def _supervised_contrastive_loss(self, zmean, class_labels):
        """Semi-supervised contrastive loss on zmean.

        Anchors and positives come only from labeled cells (label >= 0).
        ALL cells — labeled and unlabeled — appear in the denominator as
        negatives.  This gives unlabeled test cells a contrastive gradient
        (they are pushed away from every labeled class cluster) without
        requiring pseudo-labels or augmentation.

        Design details
        --------------
        * Logit matrix shape: (n_labeled, n_all) — rows are labeled anchors,
          columns are every cell in the batch.
        * Self-exclusion: each anchor's own entry in z_all is masked out via
          its global index (valid_idx), so the anchor never counts as its own
          negative.
        * Positives: a column j is a positive for anchor i only when j is also
          a labeled cell (j in valid_idx) and shares the same class label.
          Unlabeled cells are always negatives and never positives.
        * Returns zero when the batch cannot support the objective (pretrain /
          finetune phases where class_labels is None, fewer than two labeled
          cells, or only one class represented among labeled cells).
        """
        if class_labels is None:
            return zmean.new_zeros(())

        labels = class_labels.long()
        valid = labels >= 0
        n_labeled = int(valid.sum())
        if n_labeled < 2:
            return zmean.new_zeros(())

        labels_labeled = labels[valid]
        if labels_labeled.unique().numel() < 2:
            return zmean.new_zeros(())

        # L2-normalise all cells; extract labeled anchor rows.
        z_all = F.normalize(zmean, dim=1)             # (n_all, d)
        valid_idx = torch.where(valid)[0]             # global indices of labeled cells
        z_labeled = z_all[valid_idx]                  # (n_labeled, d)
        n_all = zmean.shape[0]

        # (n_labeled, n_all) similarity logits — anchors are labeled cells,
        # keys include every cell in the batch.
        logits = z_labeled @ z_all.T                  # (n_labeled, n_all)
        logits = logits / max(self.contrastive_temperature, 1e-8)
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        # Mask out each anchor's own entry using its global position in z_all.
        self_mask = torch.zeros(n_labeled, n_all, dtype=torch.bool, device=zmean.device)
        self_mask.scatter_(1, valid_idx.unsqueeze(1), True)
        logits = logits.masked_fill(self_mask, -torch.finfo(logits.dtype).max)

        # Positive-pair mask: (n_labeled, n_all).
        # Only labeled cells (columns in valid_idx) can be positives.
        pos_among_labeled = labels_labeled.unsqueeze(0).eq(labels_labeled.unsqueeze(1))
        pos_among_labeled.fill_diagonal_(False)         # no self-positive
        pos_mask = torch.zeros(n_labeled, n_all, dtype=torch.bool, device=zmean.device)
        pos_mask[:, valid_idx] = pos_among_labeled      # scatter to global columns

        anchors_with_pos = pos_mask.any(dim=1)
        if not anchors_with_pos.any():
            return zmean.new_zeros(())

        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        positives_per_anchor = pos_mask.sum(dim=1).clamp_min(1)
        mean_log_prob_pos = (pos_mask.float() * log_prob).sum(dim=1) / positives_per_anchor
        return -mean_log_prob_pos[anchors_with_pos].mean()

    def _compute_embedding_warmup_factor(self):
        """Compute warmup factor based on current epoch."""
        if self.current_epoch < self.embedding_warmup_epochs:
            return self.current_epoch / self.embedding_warmup_epochs
        return 1.0

    def _compute_kl(self, zmean, zlogvar):
        """KL to the standard normal prior, with optional free bits.

        Returns (kl_penalty, kl_per_sample, active_dims):
          kl_penalty    scalar actually added to the loss (free-bits floored)
          kl_per_sample (B,) true unfloored KL — logged so runs stay comparable
                        across free_bits settings
          active_dims   how many latent dims carry >0.01 nats; the number to
                        watch when tuning free_bits

        With free_bits=0 the penalty is exactly kl_per_sample.mean(), i.e. the
        original objective, bit for bit.
        """
        # Clamp zlogvar before exp to prevent float32 overflow (exp > 88 → inf → nan KL).
        kl_per_dim = -0.5 * (
            1 + zlogvar - zmean.pow(2) - torch.exp(zlogvar.clamp(-30, 20))
        )  # (B, z_dim)
        kl_per_sample = kl_per_dim.sum(dim=1)  # (B,)

        # Average over the batch *within* each dimension before applying the
        # floor — this is the original formulation. Flooring per sample instead
        # makes the gradient far noisier.
        kl_per_dim_mean = kl_per_dim.mean(dim=0)  # (z_dim,)
        active_dims = (kl_per_dim_mean > 0.01).sum()

        if self.free_bits > 0:
            kl_penalty = torch.clamp(kl_per_dim_mean, min=self.free_bits).sum()
        else:
            kl_penalty = kl_per_sample.mean()

        return kl_penalty, kl_per_sample, active_dims

    def training_step(self, batch, batch_idx):
        data_dict, labels = self.process_batch(batch)

        # Apply dropout to embeddings during training
        enc, zmean, zlogvar, decoded_dict = self._forward_model(data_dict, labels, apply_dropout=True)

        mse_loss, mse_losses = self._compute_losses(data_dict, decoded_dict)
        kl_penalty, kl_loss, active_dims = self._compute_kl(zmean, zlogvar)

        total_loss = mse_loss + self.beta * kl_penalty
        contrastive_loss = self._supervised_contrastive_loss(
            zmean, self._extract_class_labels(labels)
        )
        if self.use_contrastive_loss:
            total_loss = total_loss + self.contrastive_weight * contrastive_loss

        # Add reconstruction consistency loss: forward pass WITHOUT class labels
        if self.reconstruction_consistency_weight > 0 and self.config.use_class_embedding:
            # Get reconstruction without class labels (simulating test-time).
            # Keep source/tech, super_region, and layer labels — all available at inference.
            if labels.ndim == 2:
                if labels.shape[1] == 2:
                    class_labels, source_labels = labels.unbind(1)
                    _, _, _, decoded_dict_no_class = self.model(
                        data_dict, source_labels=source_labels, class_labels=None, apply_dropout=False
                    )
                elif labels.shape[1] == 3:
                    class_labels, source_labels, super_region_labels = labels.unbind(1)
                    _, _, _, decoded_dict_no_class = self.model(
                        data_dict, source_labels=source_labels, class_labels=None,
                        super_region_labels=super_region_labels, apply_dropout=False
                    )
                elif labels.shape[1] == 4:
                    class_labels, source_labels, super_region_labels, layer_labels = labels.unbind(1)
                    _, _, _, decoded_dict_no_class = self.model(
                        data_dict, source_labels=source_labels, class_labels=None,
                        super_region_labels=super_region_labels, layer_labels=layer_labels,
                        apply_dropout=False
                    )
                else:
                    _, _, _, decoded_dict_no_class = self.model(
                        data_dict, source_labels=labels[:, 1], class_labels=None, apply_dropout=False
                    )
            else:
                _, _, _, decoded_dict_no_class = self.model(
                    data_dict, source_labels=labels, class_labels=None, apply_dropout=False
                )

            # Compute consistency loss between reconstructions
            consistency_loss = 0.0
            for mod_name in self.modalities.keys():
                consistency_loss += F.mse_loss(decoded_dict[mod_name], decoded_dict_no_class[mod_name])

            # Apply warmup schedule to consistency loss
            warmup_factor = self._compute_embedding_warmup_factor()
            total_loss = total_loss + warmup_factor * self.reconstruction_consistency_weight * consistency_loss

            self.log("train_consistency_loss", consistency_loss)
            self.log("embedding_warmup_factor", warmup_factor)

        self.log("train_loss", total_loss)
        for mod_name, mod_loss in mse_losses.items():
            self.log(f"train_mse_loss_{mod_name}", mod_loss)
        self.log("train_kl_loss", kl_loss.mean())
        self.log("train_kl_penalty", kl_penalty)
        self.log("train_active_dims", active_dims.float())
        self.log("train_contrastive_loss", contrastive_loss)
        self.log("train_contrastive_weighted", self.contrastive_weight * contrastive_loss)
        self.train_loss.append(total_loss.item())
        self._train_component_buffer.append({
            "total": float(total_loss.detach().cpu()),
            "mse": float(mse_loss.detach().cpu()),
            "kl": float(kl_loss.mean().detach().cpu()),
            "contrastive": float(contrastive_loss.detach().cpu()),
        })

        return total_loss

    def validation_step(self, batch, batch_idx):
        data_dict, labels = self.process_batch(batch)
        enc, zmean, zlogvar, decoded_dict = self._forward_model(data_dict, labels)
        
        mse_loss, mse_losses = self._compute_losses(data_dict, decoded_dict)
        kl_penalty, kl_loss, active_dims = self._compute_kl(zmean, zlogvar)

        loss = mse_loss + self.beta * kl_penalty
        contrastive_loss = self._supervised_contrastive_loss(
            zmean, self._extract_class_labels(labels)
        )
        if self.use_contrastive_loss:
            loss = loss + self.contrastive_weight * contrastive_loss

        self.val_loss.append(loss.item())
        self.log("val_loss", loss)
        for mod_name, mod_loss in mse_losses.items():
            self.log(f"val_mse_loss_{mod_name}", mod_loss)
        self.log("val_kl_loss", kl_loss.mean())
        self.log("val_kl_penalty", kl_penalty)
        self.log("val_active_dims", active_dims.float())
        self.log("val_contrastive_loss", contrastive_loss)
        self.log("val_contrastive_weighted", self.contrastive_weight * contrastive_loss)
        self._val_component_buffer.append({
            "total": float(loss.detach().cpu()),
            "mse": float(mse_loss.detach().cpu()),
            "kl": float(kl_loss.mean().detach().cpu()),
            "contrastive": float(contrastive_loss.detach().cpu()),
        })

        return loss

    def forward(self, batch):
        data_dict, labels = self.process_batch(batch)
        return self._forward_model(data_dict, labels)

    def on_validation_epoch_end(self):
        if self.val_loss:
            avg_loss = sum(self.val_loss) / len(self.val_loss)
            logger.info("Average validation loss is %.2f", avg_loss)
            if self._val_component_buffer:
                self.val_epoch_history.append({
                    "epoch": int(self.current_epoch),
                    **{
                        k: float(sum(row[k] for row in self._val_component_buffer) / len(self._val_component_buffer))
                        for k in self._val_component_buffer[0]
                    },
                })
                self._val_component_buffer = []
            self.val_loss = []
        
    def on_train_epoch_end(self):
        if self.train_loss:
            avg_loss = sum(self.train_loss) / len(self.train_loss)
            logger.info("Average training loss is %.2f", avg_loss)
            if self._train_component_buffer:
                self.train_epoch_history.append({
                    "epoch": int(self.current_epoch),
                    **{
                        k: float(sum(row[k] for row in self._train_component_buffer) / len(self._train_component_buffer))
                        for k in self._train_component_buffer[0]
                    },
                })
                self._train_component_buffer = []
            self.train_loss = []

    def configure_optimizers(self):
        return optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
