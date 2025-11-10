from dataclasses import dataclass
from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from backbones import ResNet18Enc, ResNet18Dec

@dataclass
class CVAEConfig:
    """Configuration class for MultiModal CVAE ablation studies."""
    use_source_embedding: bool = True
    use_class_embedding: bool = True
    use_super_region_embedding: bool = False  # NEW: for hierarchical region conditioning (available at inference)
    use_fusion_encoder: bool = True
    use_batch_norm: bool = True
    beta: float = 1.0
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
    def full_model():
        """Full model WITHOUT regularization.
        Has all components: source & class embeddings, fusion encoder, batch norm.
        No augmentations, no dropout/consistency regularization."""
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
    def augmentation_ablation():
        """Most complex model: Full model + light augmentations + regularization.
        This is the kitchen sink - everything enabled for maximum performance."""
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

class MultiModalCVAE(nn.Module):
    """Multimodal Conditional Variational Autoencoder model for joint processing of multiple modalities."""
    
    def __init__(self, modalities, z_dim, config: CVAEConfig, num_sources=None, num_classes=None, num_super_regions=None):
        """
        Initialize the MultiModal CVAE with configurable components.

        modalities (dict): Dictionary mapping modality names to their output sizes
        z_dim (int): Dimension of the latent space
        config (CVAEConfig): Configuration object specifying which components to use
        num_sources (int, optional): Number of sources for source embedding
        num_classes (int, optional): Number of classes for class embedding
        num_super_regions (int, optional): Number of super regions for super_region embedding
        """
        super().__init__()
        self.z_dim = z_dim
        self.config = config
        self.class_hidden_dim = config.class_hidden_dim
        self.num_sources = num_sources
        self.num_classes = num_classes
        self.num_super_regions = num_super_regions
        self.modalities = modalities
        self.num_modalities = len(modalities)
        self.embedding_dropout = nn.Dropout(config.class_embedding_dropout)

        self.encoders = nn.ModuleDict({
            mod_name: ResNet18Enc(z_dim=z_dim)
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

        # NEW: Super region embedding (available at inference time, unlike class labels)
        if config.use_super_region_embedding and num_super_regions:
            self.super_region_embedding = nn.Embedding(num_super_regions, self.class_hidden_dim)
            embedding_dim += self.class_hidden_dim
        else:
            self.super_region_embedding = None
        
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
            mod_name: ResNet18Dec(z_dim=z_dim, output_size=output_size)
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

    def _get_embeddings(self, source_labels=None, class_labels=None, super_region_labels=None, apply_dropout=False):
        """Get source, class, and super_region embeddings if they exist.

        Args:
            source_labels: Source labels tensor
            class_labels: Class labels tensor
            super_region_labels: Super region labels tensor (NEW - available at inference)
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
        else:
            batch_size = 1

        if self.source_embedding is not None:
            if source_labels is not None:
                source_emb = self.source_embedding(source_labels)
            else:
                # If source embedding exists but no labels provided, use zeros
                source_emb = torch.zeros(batch_size, self.class_hidden_dim, device=self.source_embedding.weight.device)
            embeddings.append(source_emb)

        if self.class_embedding is not None:
            if class_labels is not None:
                class_emb = self.class_embedding(class_labels)
                # Apply dropout during training to prevent over-reliance
                if apply_dropout and self.training:
                    class_emb = self.embedding_dropout(class_emb)
            else:
                # If class embedding exists but no labels provided, use zeros
                class_emb = torch.zeros(batch_size, self.class_hidden_dim, device=self.class_embedding.weight.device)
            embeddings.append(class_emb)

        # NEW: Super region embedding (always available at inference, unlike class labels)
        if self.super_region_embedding is not None:
            if super_region_labels is not None:
                super_region_emb = self.super_region_embedding(super_region_labels)
            else:
                # If super_region embedding exists but no labels provided, use zeros
                super_region_emb = torch.zeros(batch_size, self.class_hidden_dim, device=self.super_region_embedding.weight.device)
            embeddings.append(super_region_emb)

        if embeddings:
            return torch.cat(embeddings, dim=1)
        else:
            return None

    def encode(self, data_dict, source_labels=None, class_labels=None, super_region_labels=None, apply_dropout=False):
        encoded_features = []
        for mod_name, encoder in self.encoders.items():
            h = encoder(data_dict[mod_name])
            encoded_features.append(h)

        combined_features = torch.cat(encoded_features, dim=1)

        embeddings = self._get_embeddings(source_labels, class_labels, super_region_labels, apply_dropout=apply_dropout)
        if embeddings is not None:
            combined_features = torch.cat([combined_features, embeddings], dim=1)

        if self.fusion_encoder is not None:
            h = self.fusion_encoder(combined_features)
        else:
            h = combined_features

        return h, self.z_mean(h), self.z_log_var(h)

    def decode(self, z, source_labels=None, class_labels=None, super_region_labels=None, apply_dropout=False):
        results = {}

        embeddings = self._get_embeddings(source_labels, class_labels, super_region_labels, apply_dropout=apply_dropout)

        for mod_name in self.modalities.keys():
            if embeddings is not None:
                z_mod = torch.cat([z, embeddings], dim=1)
            else:
                z_mod = z
            z_mod = self.decoder_fcs[mod_name](z_mod)
            results[mod_name] = self.decoders[mod_name](z_mod)

        return results

    def forward(self, data_dict, source_labels=None, class_labels=None, super_region_labels=None, apply_dropout=False):
        encoded, mu, logvar = self.encode(data_dict, source_labels, class_labels, super_region_labels, apply_dropout=apply_dropout)
        z = self.reparameterize(mu, logvar)
        decoded = self.decode(z, source_labels, class_labels, super_region_labels, apply_dropout=apply_dropout)

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
        self.reconstruction_consistency_weight = config.reconstruction_consistency_weight
        self.embedding_warmup_epochs = config.embedding_warmup_epochs

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
                print(f"Error computing MSE loss for {mod_name}: {e}")
                print(f"Shapes are {data_dict[mod_name].shape} and {decoded_dict[mod_name].shape}")

        mse_loss = sum(self.modality_weights[mod_name] * mse_losses[mod_name] 
                      for mod_name in self.modalities.keys())
        
        return mse_loss, mse_losses

    def _forward_model(self, data_dict, labels, apply_dropout=False):
        """Forward pass handling different label configurations.

        Supports 1D (source only), 2D (class + source), or 3D (class + source + super_region).
        """
        if labels.ndim == 2:
            if labels.shape[1] == 2:
                # Traditional: (class, source)
                class_labels, source_labels = labels.unbind(1)
                return self.model(data_dict, source_labels=source_labels, class_labels=class_labels, apply_dropout=apply_dropout)
            elif labels.shape[1] == 3:
                # NEW: (class, source, super_region)
                class_labels, source_labels, super_region_labels = labels.unbind(1)
                return self.model(data_dict, source_labels=source_labels, class_labels=class_labels,
                                super_region_labels=super_region_labels, apply_dropout=apply_dropout)
        else:
            # Just source labels
            return self.model(data_dict, source_labels=labels, apply_dropout=apply_dropout)

    def _compute_embedding_warmup_factor(self):
        """Compute warmup factor based on current epoch."""
        if self.current_epoch < self.embedding_warmup_epochs:
            return self.current_epoch / self.embedding_warmup_epochs
        return 1.0

    def training_step(self, batch, batch_idx):
        data_dict, labels = self.process_batch(batch)

        # Apply dropout to embeddings during training
        enc, zmean, zlogvar, decoded_dict = self._forward_model(data_dict, labels, apply_dropout=True)

        mse_loss, mse_losses = self._compute_losses(data_dict, decoded_dict)
        kl_loss = -0.5 * torch.sum(1 + zlogvar - zmean.pow(2) - torch.exp(zlogvar), axis=1)

        total_loss = mse_loss + self.beta * kl_loss.mean()

        # Add reconstruction consistency loss: forward pass WITHOUT class labels
        if self.reconstruction_consistency_weight > 0 and self.config.use_class_embedding:
            # Get reconstruction without class labels (simulating test-time)
            # But keep source and super_region labels (they're available at inference)
            if labels.ndim == 2:
                if labels.shape[1] == 2:
                    # Traditional: (class, source)
                    class_labels, source_labels = labels.unbind(1)
                    _, _, _, decoded_dict_no_class = self.model(
                        data_dict, source_labels=source_labels, class_labels=None, apply_dropout=False
                    )
                elif labels.shape[1] == 3:
                    # NEW: (class, source, super_region) - keep source and super_region at test time
                    class_labels, source_labels, super_region_labels = labels.unbind(1)
                    _, _, _, decoded_dict_no_class = self.model(
                        data_dict, source_labels=source_labels, class_labels=None,
                        super_region_labels=super_region_labels, apply_dropout=False
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
        for mod_name, loss in mse_losses.items():
            self.log(f"train_mse_loss_{mod_name}", loss)
        self.log("train_kl_loss", kl_loss.mean())
        self.train_loss.append(total_loss.item())

        return total_loss

    def validation_step(self, batch, batch_idx):
        data_dict, labels = self.process_batch(batch)
        enc, zmean, zlogvar, decoded_dict = self._forward_model(data_dict, labels)
        
        mse_loss, mse_losses = self._compute_losses(data_dict, decoded_dict)
        kl_loss = -0.5 * torch.sum(1 + zlogvar - zmean.pow(2) - torch.exp(zlogvar), axis=1)
        
        loss = mse_loss + self.beta * kl_loss.mean()

        self.val_loss.append(loss.item())
        self.log("val_loss", loss)
        for mod_name, loss in mse_losses.items():
            self.log(f"val_mse_loss_{mod_name}", loss)
        self.log("val_kl_loss", kl_loss.mean())

        return loss

    def forward(self, batch):
        data_dict, labels = self.process_batch(batch)
        return self._forward_model(data_dict, labels)

    def on_validation_epoch_end(self):
        if self.val_loss:
            avg_loss = sum(self.val_loss) / len(self.val_loss)
            print(f"Average validation loss is {avg_loss:.2f}")
            self.val_loss = []
        
    def on_train_epoch_end(self):
        if self.train_loss:
            avg_loss = sum(self.train_loss) / len(self.train_loss)
            print(f"Average training loss is {avg_loss:.2f}")
            self.train_loss = []

    def configure_optimizers(self):
        return optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)