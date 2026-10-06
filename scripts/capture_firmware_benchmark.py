#!/usr/bin/env python3
"""Capture one complete PSoC 6 benchmark run from the UART."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import serial


EPISODE_RE = re.compile(
    r"^BATCH rec=(?P<record>\d+) support=(?P<support>\d+) query=(?P<query>\d+) "
    r"pre_f1_macro=(?P<pre>[0-9.]+) sgd_f1_macro=(?P<sgd>[0-9.]+) "
    r"proto_f1_macro=(?P<proto>[0-9.]+)$"
)
CFG_RE = re.compile(
    r"^BATCH_CFG shots=(?P<shots>\d+) episodes=(?P<episodes>\d+) "
    r"total_support=(?P<support>\d+) total_query=(?P<query>\d+) max_query=(?P<max_query>\d+)$"
)
AGG_RE = re.compile(
    r"^BATCH_AGG episodes=(?P<episodes>\d+) total_support=(?P<support>\d+) "
    r"total_query=(?P<query>\d+) total_cyc=(?P<cycles>\d+) avg_episode_cyc=(?P<avg_cycles>\d+)$"
)
METRIC_RE = re.compile(
    r"^batch_(?P<name>pre_linear|post_linear_sgd|post_prototype) "
    r"acc=(?P<accuracy>[0-9.]+) f1_macro=(?P<f1>[0-9.]+)$"
)
PROF_RE = re.compile(r"^PROF (?P<name>\S.*?)\s+cyc=(?P<cycles>\d+)(?P<rest>.*)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--raw-log", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    deadline = time.monotonic() + args.timeout
    raw_lines: list[str] = []
    episodes: dict[str, dict[str, float | int]] = {}
    metrics: dict[str, dict[str, float]] = {}
    profiling: dict[str, int] = {}
    config: dict[str, int] | None = None
    aggregate: dict[str, int] | None = None

    with serial.Serial(args.port, args.baud, timeout=0.5) as uart:
        while time.monotonic() < deadline:
            raw = uart.readline()
            if not raw:
                continue
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            print(line, flush=True)
            raw_lines.append(line)

            if match := CFG_RE.match(line):
                config = {key: int(value) for key, value in match.groupdict().items()}
            elif match := EPISODE_RE.match(line):
                values = match.groupdict()
                episodes[values["record"]] = {
                    "support": int(values["support"]),
                    "query": int(values["query"]),
                    "pre_f1_macro": float(values["pre"]),
                    "post_linear_sgd_f1_macro": float(values["sgd"]),
                    "post_prototype_f1_macro": float(values["proto"]),
                }
            elif match := AGG_RE.match(line):
                aggregate = {key: int(value) for key, value in match.groupdict().items()}
            elif match := METRIC_RE.match(line):
                values = match.groupdict()
                metrics[values["name"]] = {
                    "accuracy": float(values["accuracy"]),
                    "f1_macro": float(values["f1"]),
                }
                if values["name"] == "post_prototype":
                    break
            elif match := PROF_RE.match(line):
                profiling[match.group("name").strip()] = int(match.group("cycles"))

    if config is None or aggregate is None or "post_prototype" not in metrics:
        raise RuntimeError("UART capture ended before one complete benchmark was received")
    if len(episodes) != config["episodes"]:
        raise RuntimeError(
            f"expected {config['episodes']} episode lines, captured {len(episodes)}"
        )

    result = {
        "platform": "CY8CPROTO-063-BLE",
        "uart": {"port": args.port, "baud": args.baud},
        "config": config,
        "aggregate_cycles": aggregate,
        "aggregate_device": metrics,
        "per_episode_device": episodes,
        "profiling_cycles": profiling,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.raw_log:
        args.raw_log.parent.mkdir(parents=True, exist_ok=True)
        args.raw_log.write_text("\n".join(raw_lines) + "\n", encoding="utf-8")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
