from __future__ import annotations

import copy
from collections import OrderedDict

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from .data import ClientPool


def train_one_client(
    global_model: nn.Module,
    dataset,
    indices: list[int],
    device: torch.device,
    batch_size: int,
    lr: float,
    epochs: int,
) -> tuple[OrderedDict[str, torch.Tensor], int]:
    if not indices:
        return copy.deepcopy(global_model.state_dict()), 0

    model = copy.deepcopy(global_model).to(device)
    model.train()
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
    return copy.deepcopy(model.cpu().state_dict()), len(indices)


def federated_train_round(
    model: nn.Module,
    dataset,
    clients: list[ClientPool],
    available_clients: list[int],
    device: torch.device,
    batch_size: int,
    lr: float,
    epochs: int,
) -> nn.Module:
    local_states: list[tuple[OrderedDict[str, torch.Tensor], int]] = []
    for client_id in available_clients:
        indices = clients[client_id].labeled_indices()
        state, sample_count = train_one_client(model, dataset, indices, device, batch_size, lr, epochs)
        if sample_count > 0:
            local_states.append((state, sample_count))

    if not local_states:
        return model

    total = sum(count for _, count in local_states)
    averaged = OrderedDict()
    for key in local_states[0][0].keys():
        averaged[key] = sum(state[key] * (count / total) for state, count in local_states)
    model.load_state_dict(averaged)
    return model


@torch.no_grad()
def evaluate(model: nn.Module, dataset, device: torch.device, batch_size: int) -> dict[str, float]:
    model = model.to(device)
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    correct = 0
    total = 0
    all_preds: list[int] = []
    all_targets: list[int] = []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1).cpu()
        correct += int((preds == y).sum())
        total += int(y.numel())
        all_preds.extend(preds.tolist())
        all_targets.extend(y.tolist())

    accuracy = correct / total if total else 0.0
    return {
        "accuracy": float(accuracy),
        "macro_f1": macro_f1(np.asarray(all_targets), np.asarray(all_preds)),
    }


def macro_f1(targets: np.ndarray, preds: np.ndarray) -> float:
    labels = np.unique(targets)
    scores = []
    for label in labels:
        tp = np.sum((targets == label) & (preds == label))
        fp = np.sum((targets != label) & (preds == label))
        fn = np.sum((targets == label) & (preds != label))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        scores.append(f1)
    return float(np.mean(scores)) if scores else 0.0
