"""HIPPIE utility helpers.

The canonical inference path lives in ``hippie.inference.HIPPIEClassifier``;
the function below is a low-level legacy helper retained for callers that
want to run two separate per-modality encoders side by side. New code
should prefer ``HIPPIEClassifier.get_embeddings``.
"""

from __future__ import annotations

import numpy as np
import torch


def get_embeddings(dataloader_wave, dataloader_time, wave_model, time_model):
    """Run two unimodal encoders in parallel and return concatenated embeddings.

    Legacy helper from the bimodal era. Returns ``(wave_emb, time_emb, joint_emb)``
    where ``joint_emb`` is the channel-wise concatenation. Wave/time embeddings
    are normalized to zero mean / unit std per sample.
    """
    embedding_waveform = []
    embedding_isi = []
    for (wave, label_wave), (time, label_time) in zip(dataloader_wave, dataloader_time):
        assert (label_wave == label_time).all()
        w_out = wave_model((wave, label_wave))
        t_out = time_model((time, label_time))
        e_wave = w_out[0]  # (enc, zmean, zlogvar, dec)
        e_time = t_out[0]

        e_wave = (e_wave - e_wave.mean(dim=1)[:, None]) / e_wave.std(dim=1)[:, None]
        e_time = (e_time - e_time.mean(dim=1)[:, None]) / e_time.std(dim=1)[:, None]

        embedding_waveform.append(e_wave)
        embedding_isi.append(e_time)

    embedding_waveform = torch.cat(embedding_waveform, dim=0).detach().numpy()
    embedding_isi = torch.cat(embedding_isi, dim=0).detach().numpy()
    joint_embeddings = np.concatenate([embedding_waveform, embedding_isi], axis=1)
    return embedding_waveform, embedding_isi, joint_embeddings
