"""Preprocessing helpers for raw EEG and PSD inputs."""

from __future__ import annotations

import numpy as np
from scipy.signal import welch
from sklearn.preprocessing import StandardScaler


def compute_psd_dataset(x: np.ndarray, fs: int = 100, nperseg: int = 256) -> np.ndarray:
    """Compute Welch PSD for each epoch and channel.

    Args:
        x: EEG epochs with shape (n_epochs, n_channels, n_times).
        fs: Sampling frequency.
        nperseg: Segment length for Welch PSD.

    Returns:
        PSD array with shape (n_epochs, n_channels, n_freqs).
    """
    psd_list = []
    for epoch in x:
        psd_epoch = []
        for channel in epoch:
            _, pxx = welch(channel, fs=fs, nperseg=nperseg)
            psd_epoch.append(pxx)
        psd_list.append(psd_epoch)
    return np.asarray(psd_list)


def zscore_with_scaler_per_channel(x: np.ndarray) -> np.ndarray:
    """Apply sklearn StandardScaler to each EEG channel independently."""
    n_epochs, n_channels, n_times = x.shape
    x_scaled = np.zeros_like(x)

    for channel_idx in range(n_channels):
        scaler = StandardScaler()
        channel_data = x[:, channel_idx, :]
        x_scaled[:, channel_idx, :] = scaler.fit_transform(channel_data)

    return x_scaled.reshape(n_epochs, n_channels, n_times)
