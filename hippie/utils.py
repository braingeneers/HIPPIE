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
