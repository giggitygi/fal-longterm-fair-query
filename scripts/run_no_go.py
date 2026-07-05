from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from fal_experiment.active import QueryMemory, select_queries, update_debt  # noqa: E402
from fal_experiment.config import load_config, save_config  # noqa: E402
from fal_experiment.data import (  # noqa: E402
    availability_weights,
    get_dataset,
    get_targets,
    make_dirichlet_clients,
    sample_available_clients,
    set_seed,
)
from fal_experiment.federated import evaluate, federated_train_round  # noqa: E402
from fal_experiment.metrics import gini, jain_index, low_availability_query_share  # noqa: E402
from fal_experiment.models import make_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--strategy", default=None)
    parser.add_argument("--output-root", default=None)
    return parser.parse_args()


def choose_device(config_device: str) -> torch.device:
    if config_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Config requests CUDA, but torch.cuda.is_available() is False.")
    return torch.device(config_device)


def make_run_dir(output_root: Path, dataset_name: str, strategy: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = output_root / f"{timestamp}_{dataset_name}_{strategy}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.strategy:
        config["active_learning"]["strategy"] = args.strategy
    if args.output_root:
        config["output_root"] = args.output_root

    seed = int(config["seed"])
    set_seed(seed)
    rng = np.random.default_rng(seed)
    device = choose_device(config["device"])

    dataset_cfg = config["dataset"]
    fed_cfg = config["federated"]
    al_cfg = config["active_learning"]

    output_root = ROOT / config["output_root"]
    run_dir = make_run_dir(output_root, dataset_cfg["name"], al_cfg["strategy"])
    save_config(config, run_dir / "resolved_config.yaml")
    write_run_info(run_dir, device)

    train_dataset = get_dataset(
        dataset_cfg["name"],
        ROOT / dataset_cfg["root"],
        train=True,
        download=bool(dataset_cfg["download"]),
        max_samples=dataset_cfg["max_train_samples"],
    )
    test_dataset = get_dataset(
        dataset_cfg["name"],
        ROOT / dataset_cfg["root"],
        train=False,
        download=bool(dataset_cfg["download"]),
        max_samples=dataset_cfg["max_test_samples"],
    )
    targets = get_targets(train_dataset)
    num_classes = int(len(np.unique(targets)))

    clients = make_dirichlet_clients(
        targets=targets,
        num_clients=int(fed_cfg["num_clients"]),
        alpha=float(fed_cfg["dirichlet_alpha"]),
        initial_labels_per_client=int(fed_cfg["initial_labels_per_client"]),
        seed=seed,
    )

    model = make_model(dataset_cfg["name"], num_classes=num_classes)
    debt = np.zeros(len(clients), dtype=np.float64)
    memory = QueryMemory(max_size=int(al_cfg["memory_size"]))
    availability_probs = availability_weights(len(clients), fed_cfg["availability_skew"])

    metrics_path = run_dir / "round_metrics.csv"
    query_counts_path = run_dir / "query_counts.csv"

    with metrics_path.open("w", newline="", encoding="utf-8") as metrics_file:
        writer = csv.DictWriter(
            metrics_file,
            fieldnames=[
                "round",
                "strategy",
                "accuracy",
                "macro_f1",
                "query_gini",
                "jain_index",
                "low_availability_query_share",
                "mean_redundancy",
                "selected_count",
                "available_clients",
                "labeled_total",
            ],
        )
        writer.writeheader()

        for round_idx in range(int(fed_cfg["num_rounds"])):
            available = sample_available_clients(
                num_clients=len(clients),
                participation_rate=float(fed_cfg["participation_rate"]),
                skew=fed_cfg["availability_skew"],
                rng=rng,
            )
            model = federated_train_round(
                model=model,
                dataset=train_dataset,
                clients=clients,
                available_clients=available,
                device=device,
                batch_size=int(fed_cfg["batch_size"]),
                lr=float(fed_cfg["lr"]),
                epochs=int(fed_cfg["local_epochs"]),
            )
            eval_result = evaluate(model, test_dataset, device=device, batch_size=int(fed_cfg["batch_size"]))
            selection = select_queries(
                model=model,
                dataset=train_dataset,
                clients=clients,
                available_clients=available,
                strategy=al_cfg["strategy"],
                query_budget=int(al_cfg["query_budget"]),
                candidate_pool_per_client=int(al_cfg["candidate_pool_per_client"]),
                debt=debt,
                query_memory=memory,
                lambda_q=float(al_cfg["lambda_q"]),
                lambda_r=float(al_cfg["lambda_r"]),
                device=device,
                batch_size=int(fed_cfg["batch_size"]),
                rng=rng,
            )

            selected_by_client: dict[int, list[int]] = {}
            for client_id, sample_idx in selection.selected:
                selected_by_client.setdefault(client_id, []).append(sample_idx)
            for client_id, sample_indices in selected_by_client.items():
                clients[client_id].add_labels(sample_indices)
            memory.add(selection.embeddings)
            debt = update_debt(
                debt=debt,
                available_clients=available,
                selected=selection.selected,
                query_budget=int(al_cfg["query_budget"]),
                num_clients=len(clients),
            )

            query_counts = [client.total_queries for client in clients]
            row = {
                "round": round_idx,
                "strategy": al_cfg["strategy"],
                "accuracy": eval_result["accuracy"],
                "macro_f1": eval_result["macro_f1"],
                "query_gini": gini(query_counts),
                "jain_index": jain_index(query_counts),
                "low_availability_query_share": low_availability_query_share(query_counts, availability_probs),
                "mean_redundancy": selection.mean_redundancy,
                "selected_count": len(selection.selected),
                "available_clients": " ".join(map(str, available)),
                "labeled_total": sum(len(client.labeled) for client in clients),
            }
            writer.writerow(row)
            metrics_file.flush()
            print(
                f"round={round_idx} strategy={al_cfg['strategy']} "
                f"acc={row['accuracy']:.4f} macro_f1={row['macro_f1']:.4f} "
                f"gini={row['query_gini']:.4f} red={row['mean_redundancy']:.4f}"
            )

    with query_counts_path.open("w", newline="", encoding="utf-8") as counts_file:
        writer = csv.DictWriter(counts_file, fieldnames=["client_id", "samples", "labeled", "queries", "debt"])
        writer.writeheader()
        for client_id, client in enumerate(clients):
            writer.writerow(
                {
                    "client_id": client_id,
                    "samples": len(client.indices),
                    "labeled": len(client.labeled),
                    "queries": client.total_queries,
                    "debt": debt[client_id],
                }
            )

    print(f"Run complete: {run_dir}")


def write_run_info(run_dir: Path, device: torch.device) -> None:
    info_path = run_dir / "run_info.txt"
    lines = [
        f"torch={torch.__version__}",
        f"cuda_available={torch.cuda.is_available()}",
        f"cuda_runtime={torch.version.cuda}",
        f"device={device}",
    ]
    if torch.cuda.is_available():
        lines.append(f"gpu_name={torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        lines.append(f"gpu_total_memory_mb={props.total_memory // (1024 * 1024)}")
    info_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
