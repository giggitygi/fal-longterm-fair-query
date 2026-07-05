from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 42,
    "device": "auto",
    "output_root": "runs/fal_longterm_fair_query",
    "dataset": {
        "name": "FashionMNIST",
        "root": "data/torchvision",
        "download": True,
        "max_train_samples": None,
        "max_test_samples": None,
    },
    "federated": {
        "num_clients": 20,
        "dirichlet_alpha": 0.3,
        "initial_labels_per_client": 20,
        "num_rounds": 30,
        "participation_rate": 0.2,
        "availability_skew": "long_tail",
        "local_epochs": 1,
        "batch_size": 64,
        "lr": 0.01,
        "optimizer": "sgd",
        "round_order": "train_eval_query",
        "warmup_rounds": 0,
        "warmup_client_scope": "all",
    },
    "active_learning": {
        "strategy": "entropy",
        "query_budget": 100,
        "candidate_pool_per_client": 80,
        "lambda_q": 0.6,
        "lambda_r": 0.4,
        "memory_size": 2048,
        "debt_target_mode": "available_equal",
    },
    "model": {
        "name": "simple_cnn",
    },
}


def deep_update(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path) -> dict[str, Any]:
    config = deepcopy(DEFAULT_CONFIG)
    with Path(path).open("r", encoding="utf-8") as handle:
        user_config = yaml.safe_load(handle) or {}
    return deep_update(config, user_config)


def save_config(config: dict[str, Any], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=False)
