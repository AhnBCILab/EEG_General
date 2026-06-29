"""PyTorch datasets."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class EEGRawAndPSDDataset(Dataset):
    """Sequence dataset centered on the label from the middle epoch."""

    def __init__(
        self,
        raw_data: np.ndarray,
        psd_data: np.ndarray,
        labels: np.ndarray,
        seq_len: int = 15,
    ) -> None:
        self.raw = torch.tensor(raw_data, dtype=torch.float32)
        self.psd = torch.tensor(psd_data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.seq_len = seq_len
        self.valid_indices = list(range(len(labels) - seq_len + 1))

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        start = self.valid_indices[idx]
        raw_seq = self.raw[start : start + self.seq_len]
        psd_seq = self.psd[start : start + self.seq_len]
        label = self.labels[start + self.seq_len // 2]
        return raw_seq, psd_seq, label
