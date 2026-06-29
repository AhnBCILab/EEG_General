"""Run LOSO training for the Sleep-EDF raw EEG + PSD model."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from data import load_sleep_edf_raw_psd
from datasets import EEGRawAndPSDDataset
from metrics import CLASS_NAMES, detailed_metrics, smooth_class_weights
from model import RawAndPSDModel
from trainer import predict, train


DEFAULT_DATA_ROOT = str(
    Path(__file__).resolve().parent
    / "MouseData/sleep_data/sleep-edf-database-expanded-1.0.0/"
    "sleep-edf-database-expanded-1.0.0"
)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_loso(args: argparse.Namespace) -> None:
    device = get_device()
    print(f"Using {device} device")

    raw_by_subject, psd_by_subject, y_by_subject, subject_ids = load_sleep_edf_raw_psd(
        base_path=args.data_root,
        channel=args.channel,
        sampling_rate=args.sampling_rate,
        wake_epochs=args.wake_epochs,
        cassette_only=not args.include_telemetry,
    )

    if len(raw_by_subject) < 2:
        raise RuntimeError("Need at least two subjects for LOSO training.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_fold_metrics = []
    all_y_true = []
    all_y_pred = []
    fold_histories = {}
    total_start_time = time.time()
    n_subjects = len(raw_by_subject)

    for fold in range(n_subjects):
        print(f"\n=== LOSO Fold {fold + 1}/{n_subjects} subject={subject_ids[fold]} ===")
        fold_start_time = time.time()

        train_idx = [idx for idx in range(n_subjects) if idx != fold]
        raw_train = np.vstack([raw_by_subject[idx] for idx in train_idx])
        psd_train = np.vstack([psd_by_subject[idx] for idx in train_idx])
        y_train = np.hstack([y_by_subject[idx] for idx in train_idx])

        raw_test = raw_by_subject[fold]
        psd_test = psd_by_subject[fold]
        y_test = y_by_subject[fold]

        train_ds = EEGRawAndPSDDataset(raw_train, psd_train, y_train, seq_len=args.seq_len)
        valid_ds = EEGRawAndPSDDataset(raw_test, psd_test, y_test, seq_len=args.seq_len)
        test_ds = EEGRawAndPSDDataset(raw_test, psd_test, y_test, seq_len=args.seq_len)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

        print("Train class counts:", np.bincount(y_train, minlength=5))
        print("Test class counts:", np.bincount(y_test, minlength=5))

        class_weights = smooth_class_weights(y_train, device=device, scale=args.class_weight_scale)
        print("Class weights:", class_weights)

        model = RawAndPSDModel(sampling_rate=args.sampling_rate).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)

        _, model, history = train(
            model=model,
            loader_train=train_loader,
            loader_valid=valid_loader,
            optimizer=optimizer,
            criterion=criterion,
            n_epochs=args.epochs,
            patience=args.patience,
            device=device,
            metric=detailed_metrics,
            scheduler=scheduler,
        )
        fold_histories[fold + 1] = history
        pd.DataFrame(history).to_csv(output_dir / f"history_fold_{fold + 1}.csv", index=False)

        y_true, y_pred = predict(model, test_loader, device)
        metrics = detailed_metrics(y_true, y_pred)
        metrics["subject_id"] = subject_ids[fold]
        all_fold_metrics.append(metrics)
        all_y_true.append(y_true)
        all_y_pred.append(y_pred)

        fold_time = time.time() - fold_start_time
        print(f"\nFold {fold + 1} completed in {fold_time:.2f} seconds ({fold_time / 60:.2f} minutes)")
        print(f"Fold {fold + 1} test metrics:")
        for key, value in metrics.items():
            if key.startswith("recall_class_") or key == "subject_id":
                continue
            print(f"{key}: {value * 100:.2f}%" if key != "kappa" else f"{key}: {value:.4f}")

        print("Per-class recall:")
        for i, name in enumerate(CLASS_NAMES):
            print(f"  {name}: {metrics[f'recall_class_{i}'] * 100:.2f}%")

    total_time = time.time() - total_start_time
    print(f"\nAll {n_subjects} LOSO folds completed in {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")

    y_true_total = np.concatenate(all_y_true)
    y_pred_total = np.concatenate(all_y_pred)
    aggregate_metrics = detailed_metrics(y_true_total, y_pred_total)

    print("\n=== Aggregated LOSO Metrics ===")
    for key, value in aggregate_metrics.items():
        if key.startswith("recall_class_"):
            continue
        print(f"{key}: {value * 100:.2f}%" if key != "kappa" else f"{key}: {value:.4f}")

    pd.DataFrame(all_fold_metrics).to_csv(output_dir / "20_RAWPSD15stack_LOSO_fold_metrics.csv", index=True)
    pd.DataFrame({"y_true": y_true_total, "y_pred": y_pred_total}).to_csv(
        output_dir / "20_RAWPSD15stack_LOSO_fold_predictions.csv",
        index=False,
    )

    cm = confusion_matrix(y_true_total, y_pred_total)
    cm_df = pd.DataFrame(
        cm,
        index=[f"True_{i}" for i in range(cm.shape[0])],
        columns=[f"Pred_{i}" for i in range(cm.shape[1])],
    )
    cm_df.to_csv(output_dir / "20_RAWPSD15stack_LOSO_fold_confusion_matrix.csv")
    pd.DataFrame([aggregate_metrics]).to_csv(output_dir / "20_RAWPSD15stack_LOSO_aggregate_metrics.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Sleep-EDF raw EEG + PSD model with LOSO evaluation.")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT, help="Sleep-EDF expanded database root.")
    parser.add_argument("--output-dir", default="outputs/raw_psd_loso", help="Directory for CSV outputs.")
    parser.add_argument("--channel", default="EEG Fpz-Cz", help="EEG channel to load from EDF files.")
    parser.add_argument("--include-telemetry", action="store_true", help="Include ST telemetry records.")
    parser.add_argument("--sampling-rate", type=int, default=100, help="Sampling rate used by Sleep-EDF.")
    parser.add_argument("--wake-epochs", type=int, default=60, help="Wake epochs kept before and after sleep.")
    parser.add_argument("--seq-len", type=int, default=15, help="Sequence length in epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=50, help="Maximum epochs per fold.")
    parser.add_argument("--patience", type=int, default=999, help="Early stopping patience.")
    parser.add_argument("--lr", type=float, default=0.001, help="Adam learning rate.")
    parser.add_argument("--lr-step-size", type=int, default=5, help="StepLR step size.")
    parser.add_argument("--lr-gamma", type=float, default=0.5, help="StepLR gamma.")
    parser.add_argument("--label-smoothing", type=float, default=0.1, help="Cross-entropy label smoothing.")
    parser.add_argument(
        "--class-weight-scale",
        choices=["none", "log", "sqrt"],
        default="sqrt",
        help="Class weight smoothing method.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_loso(parse_args())
