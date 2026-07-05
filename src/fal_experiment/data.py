from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms


@dataclass
class ClientPool:
    client_id: int
    indices: list[int]
    labeled: set[int] = field(default_factory=set)
    total_queries: int = 0

    def unlabeled_indices(self) -> list[int]:
        return [idx for idx in self.indices if idx not in self.labeled]

    def labeled_indices(self) -> list[int]:
        return [idx for idx in self.indices if idx in self.labeled]

    def add_labels(self, indices: Iterable[int]) -> None:
        new_indices = list(indices)
        self.labeled.update(new_indices)
        self.total_queries += len(new_indices)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_dataset(
    name: str,
    root: str | Path,
    train: bool,
    download: bool,
    max_samples: int | None = None,
) -> Dataset:
    transform = transforms.Compose([transforms.ToTensor()])
    dataset_cls = {
        "MNIST": datasets.MNIST,
        "FashionMNIST": datasets.FashionMNIST,
        "CIFAR10": datasets.CIFAR10,
    }.get(name)
    if dataset_cls is None:
        raise ValueError(f"Unsupported dataset: {name}")
    dataset = dataset_cls(root=str(root), train=train, download=download, transform=transform)
    if max_samples is None:
        return dataset
    return limit_dataset(dataset, max_samples=max_samples, seed=17)


def get_targets(dataset: Dataset) -> np.ndarray:
    if hasattr(dataset, "targets"):
        return np.asarray(dataset.targets)
    if isinstance(dataset, Subset):
        parent_targets = get_targets(dataset.dataset)
        return parent_targets[np.asarray(dataset.indices)]
    return np.asarray([dataset[i][1] for i in range(len(dataset))])


def limit_dataset(dataset: Dataset, max_samples: int, seed: int) -> Subset:
    targets = get_targets(dataset)
    rng = np.random.default_rng(seed)
    classes = np.unique(targets)
    per_class = max(1, max_samples // len(classes))
    selected: list[int] = []
    for label in classes:
        class_indices = np.flatnonzero(targets == label)
        rng.shuffle(class_indices)
        selected.extend(class_indices[:per_class].tolist())
    if len(selected) < max_samples:
        remaining = np.setdiff1d(np.arange(len(targets)), np.asarray(selected), assume_unique=False)
        rng.shuffle(remaining)
        selected.extend(remaining[: max_samples - len(selected)].tolist())
    rng.shuffle(selected)
    return Subset(dataset, selected[:max_samples])


def make_dirichlet_clients(
    targets: np.ndarray,
    num_clients: int,
    alpha: float,
    initial_labels_per_client: int,
    seed: int,
) -> list[ClientPool]:
    rng = np.random.default_rng(seed)
    client_indices = [[] for _ in range(num_clients)]
    for label in np.unique(targets):
        class_indices = np.flatnonzero(targets == label)
        rng.shuffle(class_indices)
        proportions = rng.dirichlet(np.full(num_clients, alpha))
        split_points = (np.cumsum(proportions)[:-1] * len(class_indices)).astype(int)
        splits = np.split(class_indices, split_points)
        for client_id, split in enumerate(splits):
            client_indices[client_id].extend(split.tolist())

    clients: list[ClientPool] = []
    for client_id, indices in enumerate(client_indices):
        rng.shuffle(indices)
        labeled = set(indices[: min(initial_labels_per_client, len(indices))])
        clients.append(ClientPool(client_id=client_id, indices=indices, labeled=labeled))
    return clients


def availability_weights(num_clients: int, skew: str) -> np.ndarray:
    if skew == "uniform":
        weights = np.ones(num_clients)
    elif skew == "two_group":
        weights = np.ones(num_clients)
        split = max(1, num_clients // 2)
        weights[:split] = 3.0
        weights[split:] = 0.6
    elif skew == "long_tail":
        ranks = np.arange(1, num_clients + 1)
        weights = 1.0 / np.sqrt(ranks)
    else:
        raise ValueError(f"Unsupported availability skew: {skew}")
    return weights / weights.sum()


def sample_available_clients(
    num_clients: int,
    participation_rate: float,
    skew: str,
    rng: np.random.Generator,
) -> list[int]:
    count = max(1, int(round(num_clients * participation_rate)))
    weights = availability_weights(num_clients, skew)
    return rng.choice(num_clients, size=count, replace=False, p=weights).tolist()
