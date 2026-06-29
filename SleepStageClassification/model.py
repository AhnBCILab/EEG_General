"""Raw EEG plus PSD sleep-stage model."""

from __future__ import annotations

import torch
import torch.nn as nn


class MultiResBranchEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 8, sampling_rate: int = 100) -> None:
        super().__init__()
        durations = [0.1, 1.0, 5.0]
        kernel_sizes = [max(3, int(sampling_rate * duration)) | 1 for duration in durations]

        self.branches = nn.ModuleList()
        for kernel_size in kernel_sizes:
            self.branches.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                )
            )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return [branch(x) for branch in self.branches]


class PSDEncoder(nn.Module):
    def __init__(self, psd_dim: int = 129, fs: int = 100, out_channels: int = 8) -> None:
        super().__init__()
        df = fs / (2 * (psd_dim - 1))
        k1 = max(3, int(round(1.0 / df))) | 1
        k5 = max(3, int(round(5.0 / df))) | 1

        def make_branch(kernel_size: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.Conv1d(16, 32, kernel_size=3, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(32, 32, kernel_size=3, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )

        self.branch_1hz = make_branch(k1)
        self.branch_5hz = make_branch(k5)
        self.head_1 = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, out_channels))
        self.head_2 = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, out_channels))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b1 = self.branch_1hz(x).squeeze(-1)
        b2 = self.branch_5hz(x).squeeze(-1)
        return self.head_1(b1), self.head_2(b2)


class TransformerFusion(nn.Module):
    def __init__(self, embed_dim: int = 8, num_heads: int = 2) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(inputs, inputs, inputs)
        return self.norm(inputs + attn_out)


class BiLSTMEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, num_layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        self.output_dim = hidden_dim * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return out.mean(dim=1)


class DeepMLPClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        num_classes: int = 5,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [64, 32]
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(hidden_dims)):
            layers.extend(
                [
                    nn.Linear(dims[i], dims[i + 1]),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(hidden_dims[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.mlp(x))


class RawAndPSDModel(nn.Module):
    def __init__(
        self,
        psd_dim: int = 129,
        num_classes: int = 5,
        sampling_rate: int = 100,
        out_channels: int = 8,
        lstm_hidden: int = 32,
    ) -> None:
        super().__init__()
        self.raw_encoder = MultiResBranchEncoder(
            in_channels=1,
            out_channels=out_channels,
            sampling_rate=sampling_rate,
        )
        self.psd_encoder = PSDEncoder(psd_dim=psd_dim, fs=sampling_rate, out_channels=out_channels)
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        self.fusion = TransformerFusion(embed_dim=out_channels, num_heads=2)
        self.sequence_model = BiLSTMEncoder(input_dim=out_channels, hidden_dim=lstm_hidden)
        self.classifier = DeepMLPClassifier(input_dim=lstm_hidden * 2, hidden_dims=[64, 32], num_classes=num_classes)

    def forward(self, raw_x: torch.Tensor, psd_x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _, n_times = raw_x.shape
        raw_flat = raw_x.reshape(batch_size * seq_len, 1, n_times)
        psd_flat = psd_x.reshape(batch_size * seq_len, 1, -1)

        raw_features = self.raw_encoder(raw_flat)
        pooled_raw = [self.temporal_pool(features).squeeze(-1) for features in raw_features]
        combined_raw = pooled_raw[0] + pooled_raw[1] + pooled_raw[2]

        psd_f1, psd_f2 = self.psd_encoder(psd_flat)
        heads = torch.stack(pooled_raw + [combined_raw, psd_f1, psd_f2], dim=1)
        fused = self.fusion(heads)
        epoch_repr = fused.mean(dim=1)

        x_seq = epoch_repr.reshape(batch_size, seq_len, -1)
        x_encoded = self.sequence_model(x_seq)
        return self.classifier(x_encoded)
