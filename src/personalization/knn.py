from __future__ import annotations

import numpy as np
import torch


def predict_knn_from_embeddings(
    support_embeddings: np.ndarray,
    support_labels: np.ndarray,
    query_embeddings: np.ndarray,
    neighbors: int = 1,
) -> np.ndarray:
    """Predict query labels with deterministic Euclidean k-NN.

    Ties in the neighbor vote are resolved by the smaller summed squared
    distance among tied classes and then by the smaller numeric class label.
    The paper comparison uses ``neighbors=1``; the general implementation keeps
    the baseline explicit and testable.
    """

    support = torch.as_tensor(support_embeddings, dtype=torch.float32)
    query = torch.as_tensor(query_embeddings, dtype=torch.float32)
    labels = np.asarray(support_labels)
    if support.ndim != 2 or query.ndim != 2:
        raise ValueError("support_embeddings and query_embeddings must be 2-D")
    if support.shape[0] != labels.shape[0]:
        raise ValueError("support embeddings and labels must have the same length")
    if support.shape[1] != query.shape[1]:
        raise ValueError("support and query embedding dimensions must match")
    if not 1 <= neighbors <= support.shape[0]:
        raise ValueError("neighbors must be between 1 and the support-set size")

    distances = torch.cdist(query, support, p=2).square().cpu().numpy()
    nearest = np.argsort(distances, axis=1, kind="stable")[:, :neighbors]
    predictions: list[int] = []
    for row, neighbor_idx in enumerate(nearest):
        neighbor_labels = labels[neighbor_idx]
        classes, votes = np.unique(neighbor_labels, return_counts=True)
        max_votes = int(votes.max())
        tied = classes[votes == max_votes]
        if len(tied) == 1:
            predictions.append(int(tied[0]))
            continue
        distance_sums = {
            int(cls): float(distances[row, neighbor_idx[neighbor_labels == cls]].sum())
            for cls in tied
        }
        predictions.append(min(distance_sums, key=lambda cls: (distance_sums[cls], cls)))
    return np.asarray(predictions, dtype=labels.dtype)


def knn_resource_cost(
    shots: int,
    num_classes: int,
    embedding_dim: int,
    bytes_per_value: int = 4,
) -> dict[str, int]:
    """Return deployment-state and per-query distance dimensions for 1-NN."""

    support_count = int(shots) * int(num_classes)
    values = support_count * int(embedding_dim)
    return {
        "support_embeddings": support_count,
        "state_bytes": values * int(bytes_per_value),
        "distance_dimensions_per_query": values,
    }
