"""Computational cost analysis for the ECG personalization system.

Measures:
  - Operation count (MACs) per model component
  - Per-beat backbone inference time on host CPU (proxy for MCU)
  - Per-sample adaptation time for SGD and prototype methods
  - Estimated CM4 @ 150 MHz inference time from MAC count
  - Clinical feasibility: beat interval at typical heart rates vs system latency

Outputs JSON to results/offline/compute_cost.json, which is consumed by
generate_paper_assets.py to produce the figure and table.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Architecture constants (must match model_params.h)
# ---------------------------------------------------------------------------
INPUT_LENGTH = 200          # samples per beat window
CONV1_IN_CH  = 1
CONV1_OUT_CH = 8
CONV1_KERNEL = 5
POOL1_FACTOR = 2

CONV2_IN_CH  = 8
CONV2_OUT_CH = 16
CONV2_KERNEL = 5
POOL2_FACTOR = 2

PROJ_IN      = 16
PROJ_OUT     = 32           # embedding dim
HEAD_IN      = 32
HEAD_OUT     = 2            # num classes

# PSoC6 CM4 @ 150 MHz, Cortex-M4F with FPU
# Realistic throughput for a hand-written loop with FPU MAC:
# ~75 M single-precision MACs/s is a conservative estimate for this kind
# of unvectorised C code (no CMSIS-DSP), accounting for memory latency.
CM4_MACS_PER_SEC = 75e6

# Heart-rate range for clinical feasibility (BPM)
HR_LOW  = 50    # slow sinus
HR_HIGH = 150   # moderate tachycardia

# MIT-BIH sample rate
FS_HZ = 360


# ---------------------------------------------------------------------------
# MAC count
# ---------------------------------------------------------------------------

def count_macs() -> dict[str, int]:
    """Return multiply-accumulate counts per layer (weight MACs only)."""
    # Conv1: Cout * Cin * Kw * L_out  (same padding → L_out = L_in)
    conv1  = CONV1_OUT_CH * CONV1_IN_CH  * CONV1_KERNEL * INPUT_LENGTH
    # Conv2: input length after Pool1
    l_pool1 = INPUT_LENGTH // POOL1_FACTOR
    conv2  = CONV2_OUT_CH * CONV2_IN_CH  * CONV2_KERNEL * l_pool1
    # Projection (linear)
    proj   = PROJ_IN * PROJ_OUT
    # Head (linear)
    head   = HEAD_IN * HEAD_OUT
    # Prototype update: running mean over PROJ_OUT elements (not a MAC, trivial)
    proto_update = PROJ_OUT          # additions (per sample)
    # SGD step: logit forward + softmax approx + gradient + weight update
    # logit: HEAD_IN * HEAD_OUT MACs; gradient: HEAD_OUT * HEAD_IN MACs
    sgd_step = 2 * HEAD_IN * HEAD_OUT
    return {
        "conv1": conv1,
        "conv2": conv2,
        "proj":  proj,
        "head":  head,
        "backbone_total": conv1 + conv2 + proj,
        "full_inference": conv1 + conv2 + proj + head,
        "proto_update_ops": proto_update,
        "sgd_step_macs": sgd_step,
    }


# ---------------------------------------------------------------------------
# Host-CPU timing (per-beat, single-sample, averaged over many runs)
# ---------------------------------------------------------------------------

def _load_model() -> torch.nn.Module:
    import sys
    sys.path.insert(0, str(ROOT))
    from src.models.cnn1d_backbone import BackboneWithHead
    ckpt = ROOT / "artifacts" / "checkpoints" / "baseline.pt"
    model = BackboneWithHead()
    ckpt_data = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = ckpt_data["state_dict"] if "state_dict" in ckpt_data else ckpt_data
    model.load_state_dict(state)
    model.eval()
    return model


def measure_inference_time(model: torch.nn.Module, n_warmup: int = 50, n_runs: int = 500) -> float:
    """Return mean per-beat inference time in ms (CPU, no batch)."""
    x = torch.randn(1, 1, INPUT_LENGTH)
    with torch.no_grad():
        for _ in range(n_warmup):
            model(x)
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - t0) * 1e3)
    return float(np.mean(times))


def measure_prototype_update_time(model: torch.nn.Module, n_runs: int = 2000) -> float:
    """Return mean per-sample prototype update time in ms (CPU)."""
    from src.personalization.prototypes import extract_embeddings
    x = np.random.randn(1, 1, INPUT_LENGTH).astype(np.float32)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        emb = extract_embeddings(model, x)      # backbone
        # running mean for one class (32-d vector)
        _ = emb.mean(axis=0)
        times.append((time.perf_counter() - t0) * 1e3)
    return float(np.mean(times))


def measure_sgd_step_time(model: torch.nn.Module, n_runs: int = 500) -> float:
    """Return mean time for one SGD step (forward + backward + update) in ms."""
    import sys
    sys.path.insert(0, str(ROOT))
    sgd_model = copy.deepcopy(model)
    for p in sgd_model.backbone.parameters():
        p.requires_grad = False
    for p in sgd_model.head.parameters():
        p.requires_grad = True
    opt = torch.optim.SGD(sgd_model.head.parameters(), lr=0.05)
    criterion = torch.nn.CrossEntropyLoss()
    x = torch.randn(2, 1, INPUT_LENGTH)        # 1-shot → 2 support samples
    y = torch.tensor([0, 1])
    # warmup
    for _ in range(20):
        opt.zero_grad()
        _, logits = sgd_model(x)
        criterion(logits, y).backward()
        opt.step()
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        opt.zero_grad()
        _, logits = sgd_model(x)
        criterion(logits, y).backward()
        opt.step()
        times.append((time.perf_counter() - t0) * 1e3)
    return float(np.mean(times))


# ---------------------------------------------------------------------------
# MCU estimate
# ---------------------------------------------------------------------------

def estimate_mcu_time_ms(mac_count: int, macs_per_sec: float = CM4_MACS_PER_SEC) -> float:
    return (mac_count / macs_per_sec) * 1e3


# ---------------------------------------------------------------------------
# Clinical feasibility
# ---------------------------------------------------------------------------

def beat_interval_ms(bpm: float) -> float:
    return 60_000.0 / bpm


def acquisition_time_ms() -> float:
    """Time to acquire one beat window at FS_HZ sample rate."""
    return (INPUT_LENGTH / FS_HZ) * 1e3


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import sys
    sys.path.insert(0, str(ROOT))

    print("Loading model...")
    model = _load_model()

    print("Counting MACs...")
    macs = count_macs()

    print("Measuring host CPU inference time (500 runs)...")
    infer_ms = measure_inference_time(model)

    print("Measuring prototype update time (2000 runs)...")
    proto_update_ms = measure_prototype_update_time(model)

    print("Measuring SGD step time (500 runs)...")
    sgd_step_ms = measure_sgd_step_time(model)

    # MCU estimates
    mcu_infer_ms   = estimate_mcu_time_ms(macs["full_inference"])
    mcu_infer_bb_ms = estimate_mcu_time_ms(macs["backbone_total"])

    # Full adaptation budget estimates on MCU (K-shot)
    mcu_adapt = {}
    for K in [1, 5, 10]:
        n_support = K * 2   # 2 classes
        # Prototype: n_support backbone inferences + proto update (trivial)
        proto_total = n_support * mcu_infer_bb_ms
        # SGD: n_support backbone inferences (frozen) + 50 steps each 1 forward+backward
        # The SGD step on MCU is just the head (2×32 = 64 MACs) × 50 steps, very fast
        # But embedding extraction dominates: n_support backbone inferences
        sgd_total = n_support * mcu_infer_bb_ms + 50 * estimate_mcu_time_ms(macs["sgd_step_macs"])
        mcu_adapt[K] = {
            "proto_ms": proto_total,
            "sgd_ms":   sgd_total,
        }

    # Clinical window
    acq_ms = acquisition_time_ms()
    window_low  = beat_interval_ms(HR_LOW)
    window_high = beat_interval_ms(HR_HIGH)

    result = {
        "macs": macs,
        "host_cpu": {
            "inference_per_beat_ms": infer_ms,
            "proto_update_per_sample_ms": proto_update_ms,
            "sgd_step_ms": sgd_step_ms,
        },
        "mcu_cm4_150mhz_estimate": {
            "macs_per_sec": CM4_MACS_PER_SEC,
            "inference_per_beat_ms": mcu_infer_ms,
            "backbone_only_ms": mcu_infer_bb_ms,
            "adapt_budget": mcu_adapt,
        },
        "clinical": {
            "fs_hz": FS_HZ,
            "window_samples": INPUT_LENGTH,
            "acquisition_ms": acq_ms,
            "beat_interval_bpm50_ms": window_low,
            "beat_interval_bpm150_ms": window_high,
            "max_budget_ms": window_high - acq_ms,
        },
    }

    out_path = ROOT / "results" / "offline" / "compute_cost.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nSaved → {out_path}")

    print("\n=== Summary ===")
    print(f"Total MACs (inference):   {macs['full_inference']:,}")
    print(f"  Conv1:                  {macs['conv1']:,}")
    print(f"  Conv2:                  {macs['conv2']:,}")
    print(f"  Proj:                   {macs['proj']:,}")
    print(f"  Head:                   {macs['head']:,}")
    print(f"Host CPU inference:       {infer_ms:.3f} ms/beat")
    print(f"Prototype update:         {proto_update_ms:.3f} ms/sample (incl. embed)")
    print(f"SGD step (50 steps):      {sgd_step_ms:.3f} ms")
    print(f"MCU CM4 inference est.:   {mcu_infer_ms:.2f} ms/beat")
    print(f"Acquisition window:       {acq_ms:.1f} ms ({INPUT_LENGTH} samples @ {FS_HZ} Hz)")
    print(f"Beat interval @ 50 BPM:   {window_low:.0f} ms")
    print(f"Beat interval @ 150 BPM:  {window_high:.0f} ms")
    for K, v in mcu_adapt.items():
        print(f"MCU adapt {K}-shot: proto={v['proto_ms']:.1f} ms, SGD={v['sgd_ms']:.1f} ms")


if __name__ == "__main__":
    main()
