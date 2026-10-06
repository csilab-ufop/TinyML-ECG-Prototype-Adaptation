#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.dataset.mitbih import load_processed_npz
from src.models.cnn1d_backbone import build_model_from_kwargs, normalize_model_kwargs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="artifacts/checkpoints/baseline.pt")
    ap.add_argument("--dataset", default="data/processed/mitbih_binary.npz")
    ap.add_argument("--out", default="artifacts/embeddings/embeddings.npz")
    args = ap.parse_args()

    data = load_processed_npz(args.dataset)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model_kwargs = normalize_model_kwargs(ckpt.get("model_kwargs", {"embedding_dim": int(ckpt.get("embedding_dim", 32))}))
    model = build_model_from_kwargs(num_classes=int(ckpt["num_classes"]), model_kwargs=model_kwargs)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    with torch.no_grad():
        emb, _ = model(torch.from_numpy(data["x"]).float())
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, embeddings=emb.numpy(), y=data["y"], records=data["records"])
    print("Embeddings exportados para", args.out)


if __name__ == "__main__":
    main()
