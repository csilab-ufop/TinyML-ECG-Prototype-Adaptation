from __future__ import annotations

import copy

import numpy as np
import torch

from src.models.cnn1d_backbone import BackboneWithHead
from src.personalization.knn import knn_resource_cost, predict_knn_from_embeddings
from src.personalization.last_layer_linear import adapt_linear_head
from src.personalization.prototypes import prototypes_from_embeddings


def test_one_nn_equals_nearest_prototype_at_one_shot() -> None:
    support = np.asarray([[0.0, 0.0], [2.0, 2.0]], dtype=np.float32)
    labels = np.asarray([0, 1], dtype=np.int64)
    query = np.asarray([[0.1, 0.2], [1.8, 2.1]], dtype=np.float32)
    prototypes = prototypes_from_embeddings(support, labels, num_classes=2)

    knn_pred = predict_knn_from_embeddings(support, labels, query, neighbors=1)
    proto_pred = np.square(query[:, None, :] - prototypes[None, :, :]).sum(axis=2).argmin(axis=1)

    np.testing.assert_array_equal(knn_pred, proto_pred)


def test_knn_resource_scaling() -> None:
    assert knn_resource_cost(shots=10, num_classes=2, embedding_dim=32) == {
        "support_embeddings": 20,
        "state_bytes": 2560,
        "distance_dimensions_per_query": 640,
    }


def test_linear_adaptation_matches_precomputed_full_batch_reference() -> None:
    torch.manual_seed(7)
    rng = np.random.default_rng(7)
    model = BackboneWithHead()
    reference = copy.deepcopy(model)
    x = rng.normal(size=(6, 1, 200)).astype(np.float32)
    y = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64)

    stats = adapt_linear_head(model, x, y, steps=4, lr=0.05)

    for parameter in reference.backbone.parameters():
        parameter.requires_grad = False
    xs = torch.from_numpy(x)
    ys = torch.from_numpy(y)
    reference.eval()
    with torch.no_grad():
        embeddings = reference.backbone(xs).detach()
    optimizer = torch.optim.SGD(reference.head.parameters(), lr=0.05)
    loss_fn = torch.nn.CrossEntropyLoss()
    for _ in range(4):
        optimizer.zero_grad()
        loss = loss_fn(reference.head(embeddings), ys)
        loss.backward()
        optimizer.step()

    assert stats["update_semantics"] == "full_batch"
    assert stats["batch_size"] == 6
    for actual, expected in zip(model.head.parameters(), reference.head.parameters()):
        torch.testing.assert_close(actual, expected)
