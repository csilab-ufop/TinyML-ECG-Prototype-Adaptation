#!/usr/bin/env python3
"""Tune the head-only SGD rule on held-out DS1 records, never on DS2."""

from __future__ import annotations

import argparse
import copy
from collections import defaultdict

import numpy as np
import torch

from src.dataset.mitbih import load_processed_npz
from src.dataset.splits import build_de_chazal_interpatient_split
from src.models.cnn1d_backbone import build_model_from_kwargs, normalize_model_kwargs
from src.personalization.protocols import few_shot_feasible, sample_few_shot
from src.training.metrics import compute_metrics
from src.utils.io import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/processed/mitbih_binary.npz")
    parser.add_argument("--checkpoint", default="artifacts/checkpoints/baseline.pt")
    parser.add_argument("--out", default="results/revision_round1/sgd_validation_sensitivity.json")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_processed_npz(args.dataset)
    records = data["records"].astype(str)
    split = build_de_chazal_interpatient_split(records, seed=args.seed, val_fraction=0.1)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    kwargs = normalize_model_kwargs(
        checkpoint.get("model_kwargs", {"embedding_dim": int(checkpoint.get("embedding_dim", 32))})
    )
    model = build_model_from_kwargs(int(checkpoint["num_classes"]), kwargs)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    shots_grid = (1, 5, 10)
    lr_grid = (0.01, 0.05, 0.1)
    steps_grid = (25, 50, 100)
    required_classes = np.array([0, 1])
    rows: list[dict] = []
    skipped: list[dict] = []

    for record in split.val_records:
        mask = records == record
        x_record = data["x"][mask]
        y_record = data["y"][mask]
        with torch.no_grad():
            embeddings = model.backbone(torch.from_numpy(x_record).float()).cpu().numpy()

        for shots in shots_grid:
            feasible, reason = few_shot_feasible(y_record, shots, required_classes=required_classes)
            if not feasible:
                skipped.append({"record": record, "shots": shots, "reason": reason})
                continue
            emb_sup, y_sup, emb_query, y_query = sample_few_shot(
                embeddings,
                y_record,
                shots=shots,
                seed=args.seed,
                required_classes=required_classes,
            )
            xs = torch.from_numpy(emb_sup).float()
            ys = torch.from_numpy(y_sup).long()
            xq = torch.from_numpy(emb_query).float()
            for lr in lr_grid:
                for steps in steps_grid:
                    head = copy.deepcopy(model.head)
                    optimizer = torch.optim.SGD(head.parameters(), lr=lr)
                    for _ in range(steps):
                        optimizer.zero_grad()
                        loss = torch.nn.functional.cross_entropy(head(xs), ys)
                        loss.backward()
                        optimizer.step()
                    with torch.no_grad():
                        pred = head(xq).argmax(dim=1).cpu().numpy()
                    rows.append(
                        {
                            "record": record,
                            "shots": shots,
                            "lr": lr,
                            "steps": steps,
                            **compute_metrics(y_query, pred),
                        }
                    )

    grouped: dict[tuple[float, int], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["lr"], row["steps"])].append(row["f1_macro"])
    summary = [
        {
            "lr": lr,
            "steps": steps,
            "mean_f1_macro_across_eligible_validation_tasks": float(np.mean(values)),
            "num_tasks": len(values),
        }
        for (lr, steps), values in sorted(grouped.items())
    ]
    best_observed = max(
        summary,
        key=lambda row: (
            row["mean_f1_macro_across_eligible_validation_tasks"],
            -row["steps"],
            -row["lr"],
        ),
    )
    save_json(
        {
            "selection_partition": "DS1 held-out validation records only",
            "validation_records": list(split.val_records),
            "shots_grid": list(shots_grid),
            "lr_grid": list(lr_grid),
            "steps_grid": list(steps_grid),
            "skipped_tasks": skipped,
            "per_task": rows,
            "summary": summary,
            "pre_registered_firmware_setting": {"lr": 0.05, "steps": 50},
            "best_observed_not_adopted": best_observed,
            "selection_note": (
                "Only record 209 is eligible among the two held-out DS1 records; "
                "the original fixed firmware setting is retained to avoid post-hoc "
                "selection from one patient."
            ),
        },
        args.out,
    )
    print(f"Sensitivity saved to {args.out}")
    print("Best observed (not adopted):", best_observed)


if __name__ == "__main__":
    main()
