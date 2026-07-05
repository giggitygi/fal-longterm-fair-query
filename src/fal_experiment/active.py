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
    labels: torch.Tensor
    mean_redundancy: float


class QueryMemory:
    def __init__(self, max_size: int) -> None:
        self.max_size = max_size
        self.embeddings: torch.Tensor | None = None
        self.labels: torch.Tensor | None = None

    def redundancy(self, embeddings: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        if self.embeddings is None or self.embeddings.numel() == 0:
            return torch.zeros(embeddings.shape[0])
        memory = F.normalize(self.embeddings, dim=1)
        current = F.normalize(embeddings.cpu(), dim=1)
        similarities = current @ memory.T
        if labels is not None and self.labels is not None:
            label_mask = labels.cpu().view(-1, 1) == self.labels.view(1, -1)
            similarities = similarities.masked_fill(~label_mask, -1.0)
        return similarities.max(dim=1).values.clamp(min=0.0)

    def add(self, embeddings: torch.Tensor, labels: torch.Tensor) -> None:
        embeddings = embeddings.detach().cpu()
        labels = labels.detach().cpu().long()
        if embeddings.numel() == 0:
            return
        if self.embeddings is None:
            self.embeddings = embeddings
            self.labels = labels
        else:
            self.embeddings = torch.cat([self.embeddings, embeddings], dim=0)
            self.labels = torch.cat([self.labels, labels], dim=0) if self.labels is not None else labels
        if self.embeddings.shape[0] > self.max_size:
            self.embeddings = self.embeddings[-self.max_size :]
            if self.labels is not None:
                self.labels = self.labels[-self.max_size :]


def acquisition_scores(logits: torch.Tensor, strategy: str, rng: np.random.Generator) -> torch.Tensor:
    if strategy == "random":
        return torch.from_numpy(rng.random(logits.shape[0]).astype(np.float32))
    probs = logits.softmax(dim=1)
    entropy = -(probs * (probs + 1e-12).log()).sum(dim=1)
    entropy = entropy / np.log(logits.shape[1])
    if strategy in {
        "entropy",
        "quota_entropy",
        "quota_red_entropy",
        "class_aware_quota_red",
        "debt_entropy",
        "red_entropy",
        "qfair",
    }:
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
        return SelectionResult(
            selected=[],
            embeddings=torch.empty(0, 64),
            labels=torch.empty(0, dtype=torch.long),
            mean_redundancy=0.0,
        )

    all_indices = [idx for _, idx in candidate_pairs]
    logits, embeddings = score_dataset(model, dataset, all_indices, device, batch_size)
    predicted_labels = logits.argmax(dim=1).cpu()
    base = acquisition_scores(logits, strategy, rng)
    redundancy = query_memory.redundancy(embeddings)
    class_redundancy = query_memory.redundancy(embeddings, labels=predicted_labels)

    if strategy == "quota_entropy":
        return select_with_quota(
            candidate_pairs,
            embeddings,
            predicted_labels,
            base,
            redundancy,
            query_budget,
            available_clients,
        )
    if strategy == "quota_red_entropy":
        quota_red_score = base - float(lambda_r) * redundancy
        return select_with_quota(
            candidate_pairs,
            embeddings,
            predicted_labels,
            quota_red_score,
            redundancy,
            query_budget,
            available_clients,
        )
    if strategy == "class_aware_quota_red":
        class_red_score = base - float(lambda_r) * class_redundancy
        return select_with_quota(
            candidate_pairs,
            embeddings,
            predicted_labels,
            class_red_score,
            class_redundancy,
            query_budget,
            available_clients,
        )
    if strategy == "qfair":
        qfair_score = base - float(lambda_r) * redundancy
        return select_with_debt_quota(
            candidate_pairs=candidate_pairs,
            embeddings=embeddings,
            labels=predicted_labels,
            scores=qfair_score,
            redundancy=redundancy,
            query_budget=query_budget,
            available_clients=available_clients,
            debt=debt,
            lambda_q=lambda_q,
        )

    debt_norm = normalize_debt(debt)
    scores = base.clone()
    if strategy == "debt_entropy":
        debt_terms = torch.tensor([debt_norm[client_id] for client_id, _ in candidate_pairs], dtype=torch.float32)
        scores = scores + float(lambda_q) * debt_terms
    if strategy == "red_entropy":
        scores = scores - float(lambda_r) * redundancy

    top_k = min(query_budget, len(candidate_pairs))
    selected_positions = torch.topk(scores, k=top_k).indices.cpu().tolist()
    selected = [candidate_pairs[pos] for pos in selected_positions]
    selected_embeddings = embeddings[selected_positions].cpu()
    selected_labels = predicted_labels[selected_positions].cpu()
    mean_redundancy = float(redundancy[selected_positions].mean()) if selected_positions else 0.0
    return SelectionResult(
        selected=selected,
        embeddings=selected_embeddings,
        labels=selected_labels,
        mean_redundancy=mean_redundancy,
    )


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
    positive = debt[debt > 1e-12]
    if positive.size == 0:
        return np.zeros_like(debt, dtype=np.float64)
    scale = float(np.percentile(positive, 90))
    if scale <= 1e-12:
        scale = float(np.max(positive))
    return np.clip(debt / max(scale, 1e-12), 0.0, 1.0)


def select_with_quota(
    candidate_pairs: list[tuple[int, int]],
    embeddings: torch.Tensor,
    labels: torch.Tensor,
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
    selected_labels = labels[selected_positions].cpu() if selected_positions else torch.empty(0, dtype=torch.long)
    mean_redundancy = float(redundancy[selected_positions].mean()) if selected_positions else 0.0
    return SelectionResult(
        selected=selected,
        embeddings=selected_embeddings,
        labels=selected_labels,
        mean_redundancy=mean_redundancy,
    )


def select_with_debt_quota(
    candidate_pairs: list[tuple[int, int]],
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    scores: torch.Tensor,
    redundancy: torch.Tensor,
    query_budget: int,
    available_clients: list[int],
    debt: np.ndarray,
    lambda_q: float,
) -> SelectionResult:
    positions_by_client: dict[int, list[int]] = {}
    for pos, (client_id, _) in enumerate(candidate_pairs):
        positions_by_client.setdefault(client_id, []).append(pos)

    eligible_clients = [client_id for client_id in available_clients if positions_by_client.get(client_id)]
    if not eligible_clients:
        return SelectionResult(
            selected=[],
            embeddings=torch.empty(0, 64),
            labels=torch.empty(0, dtype=torch.long),
            mean_redundancy=0.0,
        )

    caps = {client_id: len(positions_by_client[client_id]) for client_id in eligible_clients}
    debt_norm = normalize_debt(debt)
    weights = {client_id: 1.0 + float(lambda_q) * float(debt_norm[client_id]) for client_id in eligible_clients}
    quotas = allocate_quotas(caps=caps, weights=weights, query_budget=query_budget)

    selected_positions: list[int] = []
    for client_id in eligible_clients:
        local_k = quotas.get(client_id, 0)
        if local_k <= 0:
            continue
        positions = positions_by_client[client_id]
        local_scores = scores[positions]
        local_top = torch.topk(local_scores, k=min(local_k, len(positions))).indices.cpu().tolist()
        selected_positions.extend([positions[pos] for pos in local_top])

    selected = [candidate_pairs[pos] for pos in selected_positions]
    selected_embeddings = embeddings[selected_positions].cpu() if selected_positions else torch.empty(0, 64)
    selected_labels = labels[selected_positions].cpu() if selected_positions else torch.empty(0, dtype=torch.long)
    mean_redundancy = float(redundancy[selected_positions].mean()) if selected_positions else 0.0
    return SelectionResult(
        selected=selected,
        embeddings=selected_embeddings,
        labels=selected_labels,
        mean_redundancy=mean_redundancy,
    )


def allocate_quotas(caps: dict[int, int], weights: dict[int, float], query_budget: int) -> dict[int, int]:
    total_budget = min(query_budget, sum(caps.values()))
    if total_budget <= 0:
        return {client_id: 0 for client_id in caps}

    weight_sum = sum(max(weights[client_id], 1e-12) for client_id in caps)
    raw = {client_id: total_budget * max(weights[client_id], 1e-12) / weight_sum for client_id in caps}
    quotas = {client_id: min(int(np.floor(raw[client_id])), caps[client_id]) for client_id in caps}

    while sum(quotas.values()) < total_budget:
        candidates = [client_id for client_id in caps if quotas[client_id] < caps[client_id]]
        if not candidates:
            break
        best_client = max(candidates, key=lambda client_id: (raw[client_id] - quotas[client_id], weights[client_id]))
        quotas[best_client] += 1
    return quotas


def update_debt(
    debt: np.ndarray,
    available_clients: list[int],
    selected: list[tuple[int, int]],
    query_budget: int,
    num_clients: int,
    target_mode: str = "available_equal",
) -> np.ndarray:
    queried = np.zeros(num_clients, dtype=np.float64)
    for client_id, _ in selected:
        queried[client_id] += 1.0

    target = np.zeros(num_clients, dtype=np.float64)
    if target_mode == "global_equal":
        fair_share = query_budget / max(1, num_clients)
    elif target_mode == "available_equal":
        effective_budget = len(selected) if selected else query_budget
        fair_share = effective_budget / max(1, len(available_clients))
    else:
        raise ValueError(f"Unsupported debt target mode: {target_mode}")
    for client_id in available_clients:
        target[client_id] = fair_share
    return np.maximum(0.0, debt + target - queried)
