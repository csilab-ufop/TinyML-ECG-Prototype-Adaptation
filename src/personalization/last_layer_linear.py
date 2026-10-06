from __future__ import annotations

import time

import numpy as np
import torch


def adapt_linear_head(
    model: torch.nn.Module,
    x_support: np.ndarray,
    y_support: np.ndarray,
    steps: int = 50,
    lr: float = 0.05,
    momentum: float = 0.0,
    weight_decay: float = 0.0,
) -> dict[str, float]:
    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.head.parameters():
        p.requires_grad = True

    opt = torch.optim.SGD(
        model.head.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    criterion = torch.nn.CrossEntropyLoss()
    xs = torch.from_numpy(x_support).float()
    ys = torch.from_numpy(y_support).long()

    # The backbone is frozen, so extract support embeddings exactly once.  In
    # addition to avoiding redundant work, this matches the deployment
    # semantics: adaptation consists of one backbone pass per support beat
    # followed by head-only optimization.
    model.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        embeddings = model.backbone(xs).detach()
    embedding_elapsed_ms = (time.perf_counter() - t0) * 1e3

    model.head.train()
    head_t0 = time.perf_counter()
    for _ in range(steps):
        opt.zero_grad()
        logits = model.head(embeddings)
        loss = criterion(logits, ys)
        loss.backward()
        opt.step()
    head_elapsed_ms = (time.perf_counter() - head_t0) * 1e3
    elapsed_ms = (time.perf_counter() - t0) * 1e3
    return {
        "adaptation_time_ms": elapsed_ms,
        "embedding_time_ms": embedding_elapsed_ms,
        "head_update_time_ms": head_elapsed_ms,
        "updates": steps,
        "batch_size": int(len(y_support)),
        "update_semantics": "full_batch",
        "lr": float(lr),
        "momentum": float(momentum),
        "weight_decay": float(weight_decay),
    }
