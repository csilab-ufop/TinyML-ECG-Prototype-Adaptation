#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    results_dir = Path("results/offline")
    out_table = Path("results/tables/offline_summary.md")
    rows = ["| Arquivo | Acc pre | Acc linear | Acc proto |", "|---|---:|---:|---:|"]
    for p in sorted(results_dir.glob("protocol_*shot.json")):
        obj = json.loads(p.read_text(encoding="utf-8"))
        metrics = obj.get("aggregate", obj)
        if not metrics.get("pre"):
            rows.append(f"| {p.name} | n/a | n/a | n/a |")
            continue
        rows.append(
            f"| {p.name} | {metrics['pre']['accuracy']:.3f} | {metrics['post_linear']['accuracy']:.3f} | {metrics['post_prototypes']['accuracy']:.3f} |"
        )
    out_table.parent.mkdir(parents=True, exist_ok=True)
    out_table.write_text("\n".join(rows), encoding="utf-8")
    print("Resumo salvo em", out_table)


if __name__ == "__main__":
    main()
