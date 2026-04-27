import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from .backbones import ResNet18Enc, ResNet18Dec
from pytorch_lightning.utilities import grad_norm
from torch.nn.functional import normalize
from .optimizers import AdamWScheduleFree


class hippieUnimodalCVAE(nn.Module):
    def __init__(self, z_dim, output_size, class_hidden_dim, num_sources, num_classes):
        super().__init__()
        self.z_dim = z_dim
        self.class_hidden_dim = class_hidden_dim
        self.num_sources = num_sources
        self.num_classes = num_classes

        self.encoder = ResNet18Enc(z_dim=z_dim)  # Assuming this already includes appropriate normalization
        self.encoder_fc = nn.Sequential(
            nn.Linear(z_dim * 2 + class_hidden_dim * 2, z_dim * 2),
            nn.BatchNorm1d(z_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(z_dim * 2, z_dim),
            nn.BatchNorm1d(z_dim),
            nn.LeakyReLU(0.2),
        )

        self.source_embedding = nn.Embedding(num_sources, class_hidden_dim)
        self.class_embedding = nn.Embedding(num_classes, class_hidden_dim)
        self.z_mean = nn.Linear(z_dim, z_dim)
        self.z_log_var = nn.Linear(z_dim, z_dim)

        self.decoder_fc = nn.Sequential(
            nn.Linear(z_dim + class_hidden_dim * 2, z_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(z_dim * 2, z_dim * 2),
            nn.BatchNorm1d(z_dim * 2),
            nn.LeakyReLU(0.2),
        )
        self.decoder = ResNet18Dec(
            z_dim=z_dim, output_size=output_size
        )  # Assuming this includes appropriate normalization

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x, source_emb, class_emb):
        x = self.encoder(x)
        x = torch.cat([x, source_emb, class_emb], dim=1)
        x = self.encoder_fc(x)
        mu = self.z_mean(x)
        logvar = self.z_log_var(x)
        return x, mu, logvar

    def decode(self, z, source_emb, class_emb):
        z = torch.cat([z, source_emb, class_emb], dim=1)
        z = self.decoder_fc(z)
        return self.decoder(z)

    def forward(self, data, source_labels, class_labels=None):
        source_emb = self.source_embedding(source_labels)
        class_emb = self.class_embedding(class_labels) if class_labels is not None else torch.zeros_like(source_emb)

        encoded, mu, logvar = self.encode(data, source_emb, class_emb)
        z = self.reparameterize(mu, logvar)
        decoded = self.decode(z, source_emb, class_emb)

        return encoded, mu, logvar, decoded


class hippieUnimodalEmbeddingModelCVAE(pl.LightningModule):
    def __init__(
        self,
        base_model,
        alpha_max=0.5,
        learning_rate=0.01,
        weight_decay=0.01,
        beta=1,
    ):
        super().__init__()
        self.alpha_max = alpha_max
        self.beta = beta
        self.model = base_model
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.mse_loss = nn.MSELoss()
        self.val_loss = []
        self.train_loss = []
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def training_step(self, batch, batch_idx):
        data, labels = batch
        if labels.ndim == 2:
            class_labels, source_labels = labels.unbind(1)
            enc, zmean, zlogvar, dec = self.model(data, source_labels=source_labels, class_labels=class_labels)
        else:
            enc, zmean, zlogvar, dec = self.model(data, source_labels=labels)

        mse_loss = F.mse_loss(data, dec)
        kl_loss = -0.5 * torch.sum(1 + zlogvar - zmean.pow(2) - torch.exp(zlogvar), axis=1)

        total_epochs = self.trainer.max_epochs
        current_epoch = self.current_epoch

        loss = mse_loss + self.beta * kl_loss.mean()

        self.log("train_loss", loss)
        self.log("train_mse_loss", mse_loss)
        self.log("train_kl_loss", kl_loss.mean())
        self.train_loss.append(loss.item())

        return loss

    def validation_step(self, batch, batch_idx):
        data, labels = batch
        if labels.ndim == 2:
            class_labels, source_labels = labels.unbind(1)
            enc, zmean, zlogvar, dec = self.model(data, source_labels=source_labels, class_labels=class_labels)
        else:
            enc, zmean, zlogvar, dec = self.model(data, source_labels=labels)

        mse_loss = F.mse_loss(data, dec)
        kl_loss = -0.5 * torch.sum(1 + zlogvar - zmean.pow(2) - torch.exp(zlogvar), axis=1)

        total_epochs = self.trainer.max_epochs
        current_epoch = self.current_epoch

        loss = mse_loss + self.beta * kl_loss.mean()

        self.val_loss.append(loss.item())
        self.log("val_loss", loss)
        self.log("val_mse_loss", mse_loss)
        self.log("val_kl_loss", kl_loss.mean())

        return loss

    def on_validation_epoch_end(self):
        avg_loss = sum(self.val_loss) / len(self.val_loss)
        print(f"Average validation loss is {avg_loss:.2f}")
        self.val_loss = []

    def on_train_epoch_end(self):
        avg_loss = sum(self.train_loss) / len(self.train_loss)
        print(f"Average training loss is {avg_loss:.2f}")
        self.train_loss = []

    def configure_optimizers(self):
        return self.optimizer

    def forward(self, batch):
        data, labels = batch
        if labels.ndim == 2:
            class_labels, source_labels = labels.unbind(1)
            enc, zmean, zlogvar, dec = self.model(data, source_labels=source_labels, class_labels=class_labels)
        else:
            enc, zmean, zlogvar, dec = self.model(data, source_labels=labels)

        return enc, zmean, zlogvar, dec

class MultiModalCVAE(nn.Module):
    """Multimodal Conditional Variational Autoencoder model for joint processing of multiple modalities."""
    def __init__(self, modalities, z_dim, class_hidden_dim, num_sources, num_classes):
        """
        Initialize the MultiModal CVAE.
        
        Args:
            modalities (dict): Dictionary mapping modality names to their output sizes
            z_dim (int): Dimension of the latent space
            class_hidden_dim (int): Hidden dimension for class embeddings
            num_sources (int): Number of sources for source embedding
            num_classes (int): Number of classes for class embedding
        """
        super().__init__()
        self.z_dim = z_dim
        self.class_hidden_dim = class_hidden_dim
        self.num_sources = num_sources
        self.num_classes = num_classes
        self.modalities = modalities
        self.num_modalities = len(modalities)
        
        self.encoders = nn.ModuleDict({
            mod_name: ResNet18Enc(z_dim=z_dim) 
            for mod_name in modalities.keys()
        })
        
        fusion_input_size = (z_dim * 2) * self.num_modalities + class_hidden_dim * 2
        self.fusion_encoder = nn.Sequential(
            nn.Linear(fusion_input_size, z_dim * 2),
            nn.BatchNorm1d(z_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(z_dim * 2, z_dim),
        )

        self.source_embedding = nn.Embedding(num_sources, class_hidden_dim)
        self.class_embedding = nn.Embedding(num_classes, class_hidden_dim)
        
        self.z_mean = nn.Linear(z_dim, z_dim)
        self.z_log_var = nn.Linear(z_dim, z_dim)

        self.decoder_fcs = nn.ModuleDict({
            mod_name: nn.Sequential(
                nn.Linear(z_dim + class_hidden_dim * 2, z_dim * 2),
                nn.LeakyReLU(0.2),
                nn.Linear(z_dim * 2, z_dim * 2),
                nn.BatchNorm1d(z_dim * 2),
                nn.LeakyReLU(0.2),
            ) for mod_name in modalities.keys()
        })
        
        self.decoders = nn.ModuleDict({
            mod_name: ResNet18Dec(z_dim=z_dim, output_size=output_size)
            for mod_name, output_size in modalities.items()
        })

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, data_dict, source_emb, class_emb):
        encoded_features = []
        for mod_name, encoder in self.encoders.items():
            h = encoder(data_dict[mod_name])
            encoded_features.append(h)
        
        h = torch.cat(encoded_features + [source_emb, class_emb], dim=1)
        h = self.fusion_encoder(h)
        return h, self.z_mean(h), self.z_log_var(h)

    def decode(self, z, source_emb, class_emb):
        results = {}
        
        # Decode each modality
        for mod_name in self.modalities.keys():
            z_mod = torch.cat([z, source_emb, class_emb], dim=1)
            z_mod = self.decoder_fcs[mod_name](z_mod)
            results[mod_name] = self.decoders[mod_name](z_mod)
        
        return results

    def forward(self, data_dict, source_labels, class_labels=None):
        source_emb = self.source_embedding(source_labels)
        class_emb = self.class_embedding(class_labels) if class_labels is not None else torch.zeros_like(source_emb)

        encoded, mu, logvar = self.encode(data_dict, source_emb, class_emb)
        z = self.reparameterize(mu, logvar)
        decoded = self.decode(z, source_emb, class_emb)

        return encoded, mu, logvar, decoded


class MultiModalCVAETrainModule(pl.LightningModule):
    """PyTorch Lightning module for training the MultiModalCVAE model."""
    def __init__(
        self, base_model, modality_weights=None, alpha_max=0.5, learning_rate=0.01, weight_decay=0.01, beta=1
    ):
        super().__init__()
        self.model = base_model
        self.modalities = base_model.modalities
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.mse_loss = nn.MSELoss()
        self.val_loss = []
        self.train_loss = []
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        # Set default modality weights if not provided
        if modality_weights is None:
            self.modality_weights = {mod_name: 1.0 for mod_name in self.modalities.keys()}
        else:
            self.modality_weights = modality_weights
        
        self.alpha_max = alpha_max
        self.beta = beta

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
                breakpoint()

        mse_loss = sum(self.modality_weights[mod_name] * mse_losses[mod_name] 
                      for mod_name in self.modalities.keys())
        
        return mse_loss, mse_losses

    def training_step(self, batch, batch_idx):
        data_dict, labels = self.process_batch(batch)
        
        if labels.ndim == 2:
            class_labels, source_labels = labels.unbind(1)
            enc, zmean, zlogvar, decoded_dict = self.model(
                data_dict, source_labels=source_labels, class_labels=class_labels
            )
        else:
            enc, zmean, zlogvar, decoded_dict = self.model(data_dict, source_labels=labels)
        
        mse_loss, mse_losses = self._compute_losses(data_dict, decoded_dict)
        kl_loss = -0.5 * torch.sum(1 + zlogvar - zmean.pow(2) - torch.exp(zlogvar), axis=1)
        
        total_loss = mse_loss + self.beta * kl_loss.mean()
        
        self.log("train_loss", total_loss)
        for mod_name, loss in mse_losses.items():
            self.log(f"train_mse_loss_{mod_name}", loss)
        self.log("train_kl_loss", kl_loss.mean())
        self.train_loss.append(total_loss.item())
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        data_dict, labels = self.process_batch(batch)
        
        if labels.ndim == 2:
            class_labels, source_labels = labels.unbind(1)
            enc, zmean, zlogvar, decoded_dict = self.model(
                data_dict, source_labels=source_labels, class_labels=class_labels
            )
        else:
            enc, zmean, zlogvar, decoded_dict = self.model(data_dict, source_labels=labels)
        
        # Compute losses
        mse_loss, mse_losses = self._compute_losses(data_dict, decoded_dict)
        
        # Single KL loss for joint latent space
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
        
        if labels.ndim == 2:
            class_labels, source_labels = labels.unbind(1)
            enc, zmean, zlogvar, decoded_dict = self.model(
                data_dict, source_labels=source_labels, class_labels=class_labels
            )
        else:
            enc, zmean, zlogvar, decoded_dict = self.model(data_dict, source_labels=labels)

        return enc, zmean, zlogvar, decoded_dict

    def on_validation_epoch_end(self):
        avg_loss = sum(self.val_loss) / len(self.val_loss)
        print(f"Average validation loss is {avg_loss:.2f}")
        self.val_loss = []
        
    def on_train_epoch_end(self):
        avg_loss = sum(self.train_loss) / len(self.train_loss)
        print(f"Average training loss is {avg_loss:.2f}")
        self.train_loss = []

    def configure_optimizers(self):
        return self.optimizer
