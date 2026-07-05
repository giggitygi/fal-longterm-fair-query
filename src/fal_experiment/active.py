from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset

from .data import ClientPool


@dataclass
class SelectionResult:
    selected: list[tuple[int, int]]
    embeddings: torch.Tensor
    mean_redundancy: float


class QueryMemory:
    def __init__(self, max_size: int) -> None:
        self.max_size = max_size
        self.embeddings: torch.Tensor | None = None

    def redundancy(self, embeddings: torch.Tensor) -> torch.Tensor:
        if self.embeddings is None or self.embeddings.numel() == 0:
            return torch.zeros(embeddings.shape[0])
        memory = F.normalize(self.embeddings, dim=1)
        current = F.normalize(embeddings.cpu(), dim=1)
        return (current @ memory.T).max(dim=1).values.clamp(min=0.0)

    def add(self, embeddings: torch.Tensor) -> None:
        embeddings = embeddings.detach().cpu()
        if embeddings.numel() == 0:
            return
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = torch.cat([self.embeddings, embeddings], dim=0)
        if self.embeddings.shape[0] > self.max_size:
            self.embeddings = self.embeddings[-self.max_size :]


def acquisition_scores(logits: torch.Tensor, strategy: str, rng: np.random.Generator) -> torch.Tensor:
    if strategy == "random":
        return torch.from_numpy(rng.random(logits.shape[0]).astype(np.float32))
    probs = logits.softmax(dim=1)
    entropy = -(probs * (probs + 1e-12).log()).sum(dim=1)
    entropy = entropy / np.log(logits.shape[1])
    if strategy in {"entropy", "quota_entropy", "debt_entropy", "red_entropy", "qfair"}:
        return entropy.cpu()
    if strategy == "margin":
        top2 = torch.topk(probs, k=2, dim=1).values
        margin_score = 1.0 - (top2[:, 0] - top2[:, 1])
        return margin_score.cpu()
    raise ValueError(f"Unsupported strategy: {strategy}")


@torch.no_grad()
def select_queries(
    model: nn.Module,
    dataset,
    clients: list[ClientPool],
    available_clients: list[int],
    strategy: str,
    query_budget: int,
    candidate_pool_per_client: int,
    debt: np.ndarray,
    query_memory: QueryMemory,
    lambda_q: float,
    lambda_r: float,
    device: torch.device,
    batch_size: int,
    rng: np.random.Generator,
) -> SelectionResult:
    candidate_pairs = sample_candidates(clients, available_clients, candidate_pool_per_client, rng)
    if not candidate_pairs:
        return SelectionResult(selected=[], embeddings=torch.empty(0, 64), mean_redundancy=0.0)

    all_indices = [idx for _, idx in candidate_pairs]
    logits, embeddings = score_dataset(model, dataset, all_indices, device, batch_size)
    base = acquisition_scores(logits, strategy, rng)
    redundancy = query_memory.redundancy(embeddings)

    if strategy == "quota_entropy":
        return select_with_quota(candidate_pairs, embeddings, base, redundancy, query_budget, available_clients)

    debt_norm = normalize_debt(debt)
    scores = base.clone()
    if strategy in {"debt_entropy", "qfair"}:
        debt_terms = torch.tensor([debt_norm[client_id] for client_id, _ in candidate_pairs], dtype=torch.float32)
        scores = scores + float(lambda_q) * debt_terms
    if strategy in {"red_entropy", "qfair"}:
        scores = scores - float(lambda_r) * redundancy

    top_k = min(query_budget, len(candidate_pairs))
    selected_positions = torch.topk(scores, k=top_k).indices.cpu().tolist()
    selected = [candidate_pairs[pos] for pos in selected_positions]
    selected_embeddings = embeddings[selected_positions].cpu()
    mean_redundancy = float(redundancy[selected_positions].mean()) if selected_positions else 0.0
    return SelectionResult(selected=selected, embeddings=selected_embeddings, mean_redundancy=mean_redundancy)


def sample_candidates(
    clients: list[ClientPool],
    available_clients: list[int],
    candidate_pool_per_client: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for client_id in available_clients:
        pool = clients[client_id].unlabeled_indices()
        if not pool:
            continue
        sample_size = min(candidate_pool_per_client, len(pool))
        sampled = rng.choice(pool, size=sample_size, replace=False)
        pairs.extend((client_id, int(idx)) for idx in sampled)
    return pairs


@torch.no_grad()
def score_dataset(
    model: nn.Module,
    dataset,
    indices: list[int],
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    model = model.to(device)
    model.eval()
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False)
    logits_batches = []
    embedding_batches = []
    for x, _ in loader:
        x = x.to(device)
        logits, embeddings = model(x, return_features=True)
        logits_batches.append(logits.cpu())
        embedding_batches.append(embeddings.cpu())
    return torch.cat(logits_batches, dim=0), torch.cat(embedding_batches, dim=0)


def normalize_debt(debt: np.ndarray) -> np.ndarray:
    max_debt = float(np.max(debt)) if debt.size else 0.0
    if max_debt <= 1e-12:
        return np.zeros_like(debt, dtype=np.float64)
    return debt / max_debt


def select_with_quota(
    candidate_pairs: list[tuple[int, int]],
    embeddings: torch.Tensor,
    base: torch.Tensor,
    redundancy: torch.Tensor,
    query_budget: int,
    available_clients: list[int],
) -> SelectionResult:
    selected_positions: list[int] = []
    per_client = max(1, query_budget // max(1, len(available_clients)))
    for client_id in available_clients:
        positions = [pos for pos, pair in enumerate(candidate_pairs) if pair[0] == client_id]
        if not positions:
            continue
        local_scores = base[positions]
        local_k = min(per_client, len(positions), query_budget - len(selected_positions))
        if local_k <= 0:
            break
        local_top = torch.topk(local_scores, k=local_k).indices.cpu().tolist()
        selected_positions.extend([positions[pos] for pos in local_top])

    if len(selected_positions) < min(query_budget, len(candidate_pairs)):
        remaining = [pos for pos in range(len(candidate_pairs)) if pos not in set(selected_positions)]
        fill_k = min(query_budget - len(selected_positions), len(remaining))
        if fill_k > 0:
            fill_top = torch.topk(base[remaining], k=fill_k).indices.cpu().tolist()
            selected_positions.extend([remaining[pos] for pos in fill_top])

    selected = [candidate_pairs[pos] for pos in selected_positions]
    selected_embeddings = embeddings[selected_positions].cpu() if selected_positions else torch.empty(0, 64)
    mean_redundancy = float(redundancy[selected_positions].mean()) if selected_positions else 0.0
    return SelectionResult(selected=selected, embeddings=selected_embeddings, mean_redundancy=mean_redundancy)


def update_debt(
    debt: np.ndarray,
    available_clients: list[int],
    selected: list[tuple[int, int]],
    query_budget: int,
    num_clients: int,
) -> np.ndarray:
    queried = np.zeros(num_clients, dtype=np.float64)
    for client_id, _ in selected:
        queried[client_id] += 1.0

    target = np.zeros(num_clients, dtype=np.float64)
    fair_share = query_budget / max(1, num_clients)
    for client_id in available_clients:
        target[client_id] = fair_share
    return np.maximum(0.0, debt + target - queried)
