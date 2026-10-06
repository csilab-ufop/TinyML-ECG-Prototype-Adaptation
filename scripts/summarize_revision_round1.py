#!/usr/bin/env python3
"""Consolidate the verified revision experiments into submission-facing JSON."""

from __future__ import annotations

import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def device_mean(device: dict) -> dict[str, float]:
    episodes = device["per_episode_device"].values()
    return {
        "pre_f1_macro": statistics.fmean(x["pre_f1_macro"] for x in episodes),
        "post_linear_sgd_f1_macro": statistics.fmean(x["post_linear_sgd_f1_macro"] for x in episodes),
        "post_prototype_f1_macro": statistics.fmean(x["post_prototype_f1_macro"] for x in episodes),
    }


def host_mean(reference: dict) -> dict[str, float]:
    aggregate = reference["aggregate"]
    return {
        "pre_f1_macro": aggregate["pre"]["f1_macro"],
        "post_linear_sgd_f1_macro": aggregate["post_linear"]["f1_macro"],
        "post_prototype_f1_macro": aggregate["post_prototypes"]["f1_macro"],
    }


def main() -> None:
    offline: dict[str, dict] = {}
    hardware: dict[str, dict] = {}
    device_paths = {
        1: RESULTS / "firmware" / "ondevice_batch_1shot_device_corrected.json",
        5: RESULTS / "firmware" / "ondevice_batch_5shot_device.json",
        10: RESULTS / "firmware" / "ondevice_batch_10shot_device.json",
    }

    for k in (1, 5, 10):
        protocol = load(RESULTS / "offline" / f"protocol_{k}shot.json")
        aggregate = protocol["aggregate"]
        offline[str(k)] = {
            "eligible_records": protocol["num_evaluated_records"],
            "excluded_records": protocol["num_skipped_records"],
            "mean_per_record_f1_macro": {
                "pre": aggregate["pre"]["f1_macro"],
                "linear_sgd": aggregate["post_linear"]["f1_macro"],
                "prototype": aggregate["post_prototypes"]["f1_macro"],
                "knn_1": aggregate["post_knn1"]["f1_macro"],
            },
        }

        device = load(device_paths[k])
        reference = load(RESULTS / "firmware" / f"ondevice_batch_{k}shot_reference.json")
        dmean = device_mean(device)
        hmean = host_mean(reference)
        hardware[str(k)] = {
            "device_result": str(device_paths[k].relative_to(ROOT)),
            "host_reference": f"results/firmware/ondevice_batch_{k}shot_reference.json",
            "episodes": device["config"]["episodes"],
            "queries": device["config"]["query"],
            "max_queries_per_episode": device["config"]["max_query"],
            "host_mean_per_episode_f1_macro": hmean,
            "device_mean_per_episode_f1_macro": dmean,
            "absolute_difference": {key: abs(hmean[key] - dmean[key]) for key in hmean},
        }

    cycles = load(device_paths[1])["profiling_cycles"]
    timing = {
        "clock_hz": 100_000_000,
        "inference": {"cycles": cycles["infer_single_beat"], "milliseconds": cycles["infer_single_beat"] / 100_000},
        "prototype_adaptation": {
            str(k): {
                "cycles": cycles[f"proto_adapt_K{k}_total"],
                "milliseconds": cycles[f"proto_adapt_K{k}_total"] / 100_000,
            }
            for k in (1, 5, 10)
        },
        "linear_sgd_adaptation": {
            str(k): {
                "cycles": cycles[f"sgd_adapt_K{k}_total"],
                "milliseconds": cycles[f"sgd_adapt_K{k}_total"] / 100_000,
            }
            for k in (1, 5, 10)
        },
        "scope": "support embedding plus head update; excludes query inference, replay control, and UART",
    }

    paired = load(RESULTS / "revision_round1" / "paired_normal_only.json")["shots"]
    summary = {
        "revision": "round1",
        "platform": "CY8CPROTO-063-BLE / Cortex-M4F",
        "offline_full_eligible_ds2": offline,
        "paired_normal_only": {
            "result": "results/revision_round1/paired_normal_only.json",
            "scope": "identical eligible records and full-protocol query beats within each shot count",
            "mean_per_record_f1_macro": {
                str(k): {
                    "pre": paired[str(k)]["mean_pre_f1_macro"],
                    "restricted": paired[str(k)]["mean_restricted_f1_macro"],
                    "full": paired[str(k)]["mean_full_f1_macro"],
                }
                for k in (1, 5, 10)
            },
        },
        "hardware_capped_replay": hardware,
        "hardware_timing": timing,
        "prototype_streaming_state": {
            "classes": 2,
            "embedding_dimension": 32,
            "bytes": 264,
            "contents": "two float32 class means plus two uint32 counters",
        },
        "richer_estimator_reference_configs": {
            "ProtoDiff": {
                "repository": "https://github.com/YDU-uva/ProtoDiff",
                "commit": "b120f1bff56b27a563fd87b03df86d65a604e57d",
                "configuration": "public miniImageNet Gpt: 12 layers, width 512, 16 heads",
                "parameters": 38_487_040,
                "float32_mib": 146.81640625,
                "int8_mib": 36.7041015625,
                "scope": "published vision configuration; not a PSoC ECG port or timing measurement",
            },
            "PNN": {
                "repository": "https://github.com/Dracula-funny/PNN",
                "commit": "dd5a90ec5bbc95eec10d31b139d5c9be6305c9d8",
                "configuration": "public ResNet-12 backbone",
                "parameters": 12_424_320,
                "float32_mib": 47.39501953125,
                "convolution_macs_at_84x84": 3_507_980_288,
                "scope": "published vision configuration; not a PSoC ECG port or timing measurement",
            },
        },
        "sgd_semantics": {
            "loss": "cross_entropy",
            "learning_rate": 0.05,
            "full_batch_updates": 50,
            "momentum": 0.0,
            "weight_decay": 0.0,
            "support_embeddings_computed_once": True,
            "sensitivity_result": "results/revision_round1/sgd_validation_sensitivity.json",
        },
    }
    dump(RESULTS / "revision_round1" / "summary.json", summary)

    # Replace the obsolete pre-correction one-shot summary so that no tracked
    # result continues to advertise the former 0.007 SGD mismatch.
    d1 = load(device_paths[1])
    h1 = hardware["1"]["host_mean_per_episode_f1_macro"]
    m1 = hardware["1"]["device_mean_per_episode_f1_macro"]
    legacy = {
        "scenario": "ondevice_batch_benchmark_ds2_1shot_capped_queries",
        "platform": "CY8CPROTO-063-BLE",
        "build": "Release -Os",
        "shots": 1,
        "episodes": d1["config"]["episodes"],
        "support_samples_total": d1["config"]["support"],
        "query_samples_total": d1["config"]["query"],
        "query_samples_per_record": d1["config"]["max_query"],
        "source": {
            "replay_summary": "results/firmware/ondevice_batch_1shot.json",
            "host_reference": "results/firmware/ondevice_batch_1shot_reference.json",
            "device_capture": "results/firmware/ondevice_batch_1shot_device_corrected.json",
            "collection": "UART per-episode logs from CM4",
        },
        "aggregate_device_mean_by_episode": m1,
        "aggregate_host_mean_by_episode": h1,
        "host_device_abs_diff": hardware["1"]["absolute_difference"],
        "per_episode_device_f1_macro": {
            record: {
                "pre": row["pre_f1_macro"],
                "sgd": row["post_linear_sgd_f1_macro"],
                "proto": row["post_prototype_f1_macro"],
            }
            for record, row in d1["per_episode_device"].items()
        },
    }
    dump(RESULTS / "firmware" / "ondevice_batch_1shot_device_summary.json", legacy)
    print("Wrote results/revision_round1/summary.json and corrected one-shot summary")


if __name__ == "__main__":
    main()
