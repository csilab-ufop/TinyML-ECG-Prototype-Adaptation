#!/usr/bin/env python3
"""Evaluate normal-only adaptation on the full protocol's identical query sets."""

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

from src.dataset.mitbih import load_processed_npz
from src.dataset.splits import build_de_chazal_interpatient_split
from src.models.cnn1d_backbone import build_model_from_kwargs, normalize_model_kwargs
from src.personalization.protocols import sample_few_shot
from src.personalization.prototypes import (
    build_prototypes,
    extract_embeddings,
    predict_with_prototypes,
    prototypes_from_embeddings,
)
from src.utils.io import save_json


def main() -> None:
    torch.set_num_threads(4)
    data = load_processed_npz("data/processed/mitbih_binary.npz")
    x, y, records = data["x"], data["y"], data["records"]
    checkpoint = torch.load("artifacts/checkpoints/baseline.pt", map_location="cpu")
    kwargs = normalize_model_kwargs(checkpoint.get("model_kwargs", {"embedding_dim": int(checkpoint.get("embedding_dim", 32))}))
    model = build_model_from_kwargs(num_classes=int(checkpoint["num_classes"]), model_kwargs=kwargs)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    split = build_de_chazal_interpatient_split(records, seed=42)
    train = np.isin(records, split.train_records)
    ds1_prototypes = build_prototypes(model, x[train], y[train], num_classes=2)
    output = {"protocol": "de_chazal_interpatient_ds1_ds2", "seed": 42,
              "comparison": "same eligible records and identical full-protocol queries within each K; cohorts may differ across K",
              "shots": {}}
    for k in (1, 5, 10):
        full = json.loads(Path(f"results/offline/protocol_{k}shot.json").read_text())
        rows = []
        for original in full["per_record"]:
            record = str(original["record"])
            mask = records == record
            support_x, support_y, query_x, query_y = sample_few_shot(x[mask], y[mask], shots=k, seed=42)
            normal = support_y == 0
            embeddings = extract_embeddings(model, support_x[normal])
            prototypes = prototypes_from_embeddings(embeddings, support_y[normal], 2, base_prototypes=ds1_prototypes)
            predicted = predict_with_prototypes(model, query_x, prototypes)
            rows.append({"record": record, "queries": int(len(query_y)),
                         "pre_f1_macro": float(original["pre"]["f1_macro"]),
                         "restricted_f1_macro": float(f1_score(query_y, predicted, average="macro")),
                         "full_f1_macro": float(original["post_prototypes"]["f1_macro"])})
        output["shots"][str(k)] = {"eligible_records": len(rows),
                                    "mean_pre_f1_macro": float(np.mean([r["pre_f1_macro"] for r in rows])),
                                    "mean_restricted_f1_macro": float(np.mean([r["restricted_f1_macro"] for r in rows])),
                                    "mean_full_f1_macro": float(np.mean([r["full_f1_macro"] for r in rows])),
                                    "per_record": rows}
    save_json(output, "results/revision_round1/paired_normal_only.json")


if __name__ == "__main__":
    main()
