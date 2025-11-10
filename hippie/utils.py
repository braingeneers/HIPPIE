import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import torch


def make_confmat(cm, label_names, best_neighbors_waveform):
    # Handle division by zero when some classes have no samples
    row_sums = cm.sum(axis=1)[:, np.newaxis]
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    normalized_cm = cm / row_sums

    # Create annotations with both normalized values and raw counts
    annotations = np.empty_like(normalized_cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annotations[i, j] = f"{normalized_cm[i, j]:.2f}\n({cm[i, j]})"

    # Create heatmap with blue color scheme
    ax = sns.heatmap(
        normalized_cm,
        annot=annotations,
        fmt="",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
    )

    # Explicitly set the tick labels
    ax.set_xticklabels(label_names, rotation=45, ha="right")
    ax.set_yticklabels(label_names, rotation=0)

    # Set the title
    plt.title(f"{best_neighbors_waveform} neighbors")

    # Get the figure to return
    figure = ax.get_figure()
    plt.close(figure)  # Close the plot to avoid displaying it in some environments
    return figure


def generate_kfolds(dataset_path):
    # Load the cell explorer data
    cell_explorer_wf = pd.read_csv(f"datasets/{dataset_path}/waveforms.csv")
    cell_explorer_isi = pd.read_csv(f"datasets/{dataset_path}/isi_dist.csv")
    cell_explorer_labels = pd.read_csv(f"../datasets/{dataset_path}/celltypes.csv", index_col=0)

    # Turn into numpy arrays
    cell_explorer_wf = cell_explorer_wf.to_numpy()
    cell_explorer_isi = cell_explorer_isi.to_numpy()
    cell_explorer_labels = cell_explorer_labels.to_numpy()

    le = LabelEncoder()
    cell_explorer_labels = le.fit_transform(cell_explorer_labels)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # Generate the folds
    folds = []
    for train_index, val_index in skf.split(cell_explorer_wf, cell_explorer_labels):
        wf_train = cell_explorer_wf[train_index]
        wf_val = cell_explorer_wf[val_index]
        isi_train = cell_explorer_isi[train_index]
        isi_val = cell_explorer_isi[val_index]
        label_train = cell_explorer_labels[train_index]
        label_val = cell_explorer_labels[val_index]

        folds.append((wf_train, wf_val, isi_train, isi_val, label_train, label_val, le))

    return folds


# Try K nearest neighbor
def get_embeddings(dataloader_wave, dataloader_time, wave_model, time_model):
    embedding_waveform = []
    embedding_isi = []
    for i, ((wave, label_wave), (time, label_time)) in enumerate(zip(dataloader_wave, dataloader_time)):
        assert (label_wave == label_time).all()
        w_out = wave_model((wave, label_wave))
        t_out = time_model((time, label_time))
        e_wave, d_wave = w_out[0], w_out[-1]  # w_out = enc, zmean, zlogvar, dec
        e_time, d_time = t_out[0], t_out[-1]  # t_out = enc, zmean, zlogvar, dec

        e_wave = (e_wave - e_wave.mean(dim=1)[:, None]) / e_wave.std(dim=1)[:, None]
        e_time = (e_time - e_time.mean(dim=1)[:, None]) / e_time.std(dim=1)[:, None]

        embedding_waveform.append(e_wave)
        embedding_isi.append(e_time)
    embedding_waveform = torch.cat(embedding_waveform, dim=0)
    embedding_isi = torch.cat(embedding_isi, dim=0)
    # Run Umap in the embeddings
    embedding_waveform = embedding_waveform.detach().numpy()
    embedding_isi = embedding_isi.detach().numpy()
    # labels = torch.cat(labels, dim=0).detach().numpy()
    joint_embeddings = np.concatenate([embedding_waveform, embedding_isi], axis=1)
    # normalize the embeddings

    return embedding_waveform, embedding_isi, joint_embeddings

def get_embeddings_multimodal(loader, model):
    """Extract embeddings from a multimodal model."""
    model.eval()
    all_embeddings = []
    all_labels = []
    all_data = []
    
    with torch.no_grad():
        for sample in loader:
            embedding = model(sample)[0].detach().cpu().numpy()
            # Normalize embeddings
            embedding = (embedding - np.mean(embedding, axis=1, keepdims=True)) / np.std(embedding, axis=1, keepdims=True)
            all_embeddings.extend(embedding)
            label = sample[1]
            if label.ndim == 2:
                cls_label, source_label = label.unbind(1)
            else:
                cls_label = label
            all_labels.extend(cls_label.detach().cpu().numpy())
    
    return np.array(all_embeddings), np.array(all_labels)


def generate_embeddings(model_path, data_dir, model_type="multimodal", z_dim=5, batch_size=128):
    """
    Generate embeddings using a pretrained model.
    
    Args:
        model_path (str): Path to the model checkpoint or directory containing model checkpoints
        data_dir (str): Directory containing data files (waveforms.csv, isi_dist.csv, etc.)
        model_type (str): Either 'unimodal' or 'multimodal'
        z_dim (int): Dimension of latent space used in the model
        batch_size (int): Batch size for data processing
        
    Returns:
        If unimodal: tuple of (waveform_embeddings, isi_embeddings, joint_embeddings)
        If multimodal: multimodal_embeddings
    """
    import os
    import torch
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import LabelEncoder
    
    # Assuming these imports from the original code
    from dataloading import MultiModalEphysDataset, EphysDatasetLabeled, none_safe_collate
    from model import hippieUnimodalCVAE, hippieUnimodalEmbeddingModelCVAE, MultiModalCVAE, MultiModalCVAETrainModule
    from utils import get_embeddings, get_embeddings_multimodal
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    print(f"Loading data from {data_dir}")
    wf = pd.read_csv(os.path.join(data_dir, "waveforms.csv")).to_numpy()
    isi = pd.read_csv(os.path.join(data_dir, "isi_dist.csv")).to_numpy()
    
    # Load labels if available for metadata
    label_path = os.path.join(data_dir, "labels.csv")
    if os.path.exists(label_path):
        labels_df = pd.read_csv(label_path)
        labels = labels_df.iloc[:, 0].values  # Use first column
        le = LabelEncoder().fit(labels)
        encoded_labels = le.transform(labels)
    else:
        print("No labels.csv found. Using dummy labels.")
        encoded_labels = np.zeros(len(wf))
        le = LabelEncoder().fit(encoded_labels)
    
    # Load ACG data for multimodal if available
    acg_path = os.path.join(data_dir, "acg.csv")
    if os.path.exists(acg_path):
        acg = pd.read_csv(acg_path).to_numpy()
    else:
        print("No ACG data found. Using zeros.")
        acg = np.zeros_like(isi)
    
    # Number of classes and sources
    num_classes = len(np.unique(encoded_labels))
    num_sources = 7  # Default from the code
    print(f"Found {num_classes} classes")
    
    # Prepare source labels for the datasets
    source_label = np.ones_like(encoded_labels)  # Assume default source = 1
    combined_labels = np.vstack((encoded_labels, source_label)).T
    
    if model_type == "unimodal":
        print("Using unimodal approach...")
        
        # Check for model paths
        if os.path.isdir(model_path):
            wave_model_path = os.path.join(model_path, "wave_model.pt")
            time_model_path = os.path.join(model_path, "time_model.pt")
            if not (os.path.exists(wave_model_path) and os.path.exists(time_model_path)):
                raise ValueError(f"Could not find wave_model.pt and time_model.pt in {model_path}")
        else:
            raise ValueError("For unimodal, model_path should be a directory containing wave_model.pt and time_model.pt")
        
        # Create datasets and dataloaders
        wave_dataset = EphysDatasetLabeled(wf, isi, encoded_labels, mode="wave", normalize=False)
        time_dataset = EphysDatasetLabeled(wf, isi, encoded_labels, mode="time", normalize=False)
        
        wave_loader = torch.utils.data.DataLoader(
            wave_dataset, batch_size=batch_size, shuffle=False, collate_fn=none_safe_collate
        )
        time_loader = torch.utils.data.DataLoader(
            time_dataset, batch_size=batch_size, shuffle=False, collate_fn=none_safe_collate
        )
        
        # Load wave model
        print(f"Loading wave model from {wave_model_path}")
        wave_model = hippieUnimodalCVAE(
            z_dim=z_dim, output_size=50, class_hidden_dim=5, 
            num_sources=num_sources, num_classes=num_classes
        ).to(device)
        wave_model = hippieUnimodalEmbeddingModelCVAE(wave_model)
        wave_checkpoint = torch.load(wave_model_path, map_location=device)
        wave_model.load_state_dict(wave_checkpoint["state_dict"])
        wave_model.eval()
        
        # Load time model
        print(f"Loading time model from {time_model_path}")
        time_model = hippieUnimodalCVAE(
            z_dim=z_dim, output_size=100, class_hidden_dim=5, 
            num_sources=num_sources, num_classes=num_classes
        ).to(device)
        time_model = hippieUnimodalEmbeddingModelCVAE(time_model)
        time_checkpoint = torch.load(time_model_path, map_location=device)
        time_model.load_state_dict(time_checkpoint["state_dict"])
        time_model.eval()
        
        # Get embeddings
        print("Generating embeddings...")
        waveform_embeddings, isi_embeddings, joint_embeddings = get_embeddings(
            wave_loader, time_loader, wave_model, time_model
        )
        
        # Save embeddings
        embeddings_dir = os.path.join(data_dir, "embeddings")
        os.makedirs(embeddings_dir, exist_ok=True)
        
        # Save as dataframes with encoded labels
        df_wave = pd.DataFrame(waveform_embeddings)
        df_isi = pd.DataFrame(isi_embeddings)
        df_joint = pd.DataFrame(joint_embeddings)
        
        # Add labels
        df_wave['label'] = le.inverse_transform(encoded_labels)
        df_isi['label'] = le.inverse_transform(encoded_labels)
        df_joint['label'] = le.inverse_transform(encoded_labels)
        
        # Save to CSV
        wave_path = os.path.join(embeddings_dir, "waveform_embeddings.csv")
        isi_path = os.path.join(embeddings_dir, "isi_embeddings.csv")
        joint_path = os.path.join(embeddings_dir, "joint_embeddings.csv")
        
        df_wave.to_csv(wave_path, index=False)
        df_isi.to_csv(isi_path, index=False)
        df_joint.to_csv(joint_path, index=False)
        
        print(f"Saved embeddings to {embeddings_dir}")
        
        return waveform_embeddings, isi_embeddings, joint_embeddings
        
    else:  # multimodal
        print("Using multimodal approach...")
        
        # Check model path
        if not os.path.exists(model_path):
            raise ValueError(f"Model path {model_path} does not exist")
        
        # Create data dictionary
        data_dict = {
            "wave": wf,
            "isi": isi,
            "acg": acg
        }
        
        # Create dataset and dataloader
        multi_dataset = MultiModalEphysDataset(
            data_dict, combined_labels, mode="multi"
        )
        
        multi_loader = torch.utils.data.DataLoader(
            multi_dataset, batch_size=batch_size, shuffle=False, collate_fn=none_safe_collate
        )
        
        # Define modality sizes
        modalities = {
            "wave": 50,
            "isi": 100,
            "acg": 100
        }
        
        # Create model
        print(f"Loading multimodal model from {model_path}")
        model = MultiModalCVAE(
            modalities=modalities,
            z_dim=z_dim,
            class_hidden_dim=5,
            num_sources=num_sources,
            num_classes=num_classes
        ).to(device)
        
        # Create training module
        modality_weights = {
            "wave": 1.0,
            "isi": 1.0,
            "acg": 1.0
        }
        
        model = MultiModalCVAETrainModule(
            model, 
            modality_weights=modality_weights,
            learning_rate=0.001,
            weight_decay=0.01,
            beta=1.0
        )
        
        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        
        # Get embeddings
        print("Generating embeddings...")
        embeddings, _ = get_embeddings_multimodal(multi_loader, model)
        
        # Save embeddings
        embeddings_dir = os.path.join(data_dir, "embeddings")
        os.makedirs(embeddings_dir, exist_ok=True)
        
        # Save as dataframe with encoded labels
        df_embeddings = pd.DataFrame(embeddings)
        df_embeddings['label'] = le.inverse_transform(encoded_labels)
        
        # Save to CSV
        output_path = os.path.join(embeddings_dir, "multimodal_embeddings.csv")
        df_embeddings.to_csv(output_path, index=False)
        
        print(f"Saved embeddings to {output_path}")
        
        return embeddings