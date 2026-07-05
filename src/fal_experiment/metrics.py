from __future__ import annotations

import numpy as np


def gini(values: list[int] | np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or np.allclose(array.sum(), 0):
        return 0.0
    sorted_values = np.sort(array)
    n = array.size
    cumulative = np.cumsum(sorted_values)
    return float((n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n)


def jain_index(values: list[int] | np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    denom = array.size * np.sum(array**2)
    if denom <= 0:
        return 0.0
    return float((np.sum(array) ** 2) / denom)


def low_availability_query_share(query_counts: list[int], availability_probs: np.ndarray) -> float:
    counts = np.asarray(query_counts, dtype=np.float64)
    if counts.sum() <= 0:
        return 0.0
    threshold = np.median(availability_probs)
    low_group = availability_probs <= threshold
    return float(counts[low_group].sum() / counts.sum())
