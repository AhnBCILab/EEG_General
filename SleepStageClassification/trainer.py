"""Training and evaluation loops."""

from __future__ import annotations

import copy

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score


def _do_train(model, loader, optimizer, criterion, device, metric):
    model.train()
    train_loss = np.zeros(len(loader))
    y_pred_all = []
    y_true_all = []

    for idx_batch, (raw_x, psd_x, batch_y) in enumerate(loader):
        raw_x = raw_x.to(device, dtype=torch.float32)
        psd_x = psd_x.to(device, dtype=torch.float32)
        batch_y = batch_y.to(device, dtype=torch.int64)

        optimizer.zero_grad()
        output = model(raw_x, psd_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()

        y_pred_all.append(torch.argmax(output, dim=1).cpu().numpy())
        y_true_all.append(batch_y.cpu().numpy())
        train_loss[idx_batch] = loss.item()

    y_pred = np.concatenate(y_pred_all)
    y_true = np.concatenate(y_true_all)
    return float(np.mean(train_loss)), metric(y_true, y_pred)


def validate(model, loader, criterion, device, metric):
    model.eval()
    val_loss = np.zeros(len(loader))
    y_pred_all = []
    y_true_all = []

    with torch.no_grad():
        for idx_batch, (raw_x, psd_x, batch_y) in enumerate(loader):
            raw_x = raw_x.to(device, dtype=torch.float32)
            psd_x = psd_x.to(device, dtype=torch.float32)
            batch_y = batch_y.to(device, dtype=torch.int64)

            output = model(raw_x, psd_x)
            loss = criterion(output, batch_y)

            y_pred_all.append(torch.argmax(output, dim=1).cpu().numpy())
            y_true_all.append(batch_y.cpu().numpy())
            val_loss[idx_batch] = loss.item()

    y_pred = np.concatenate(y_pred_all)
    y_true = np.concatenate(y_true_all)
    return float(np.mean(val_loss)), metric(y_true, y_pred)


def train(
    model,
    loader_train,
    loader_valid,
    optimizer,
    criterion,
    n_epochs,
    patience,
    device,
    metric=None,
    scheduler=None,
):
    best_valid_loss = np.inf
    best_model = copy.deepcopy(model)
    waiting = 0
    history = []

    if metric is None:
        metric = balanced_accuracy_score

    print("epoch\ttrain_loss\tvalid_loss\tacc\tbacc\tf1\tval_acc\tval_bacc\tval_f1")
    print("-" * 96)

    for epoch in range(1, n_epochs + 1):
        train_loss, train_perf = _do_train(model, loader_train, optimizer, criterion, device, metric=metric)
        valid_loss, valid_perf = validate(model, loader_valid, criterion, device, metric=metric)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                **{f"train_{key}": value for key, value in train_perf.items()},
                **{f"valid_{key}": value for key, value in valid_perf.items()},
            }
        )

        print(
            f"{epoch}\t{train_loss:.4f}\t{valid_loss:.4f}"
            f"\t{train_perf['accuracy']:.4f}\t{train_perf['balanced_accuracy']:.4f}\t{train_perf['f1']:.4f}"
            f"\t{valid_perf['accuracy']:.4f}\t{valid_perf['balanced_accuracy']:.4f}\t{valid_perf['f1']:.4f}"
        )

        if valid_loss < best_valid_loss:
            print(f"best val loss {best_valid_loss:.4f} -> {valid_loss:.4f}")
            best_valid_loss = valid_loss
            best_model = copy.deepcopy(model)
            waiting = 0
        else:
            waiting += 1

        if scheduler is not None:
            scheduler.step()

        if waiting >= patience:
            print(f"Stop training at epoch {epoch}")
            print(f"Best val loss: {best_valid_loss:.4f}")
            break

    return best_model, model, history


def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_pred_all = []
    y_true_all = []

    with torch.no_grad():
        for raw_x, psd_x, batch_y in loader:
            raw_x = raw_x.to(device, dtype=torch.float32)
            psd_x = psd_x.to(device, dtype=torch.float32)
            output = model(raw_x, psd_x)
            y_pred_all.append(torch.argmax(output, dim=1).cpu().numpy())
            y_true_all.append(batch_y.numpy())

    return np.concatenate(y_true_all), np.concatenate(y_pred_all)
