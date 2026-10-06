#!/usr/bin/env python3
"""Generate the compact figures added during ESL revision round 1."""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np


OUT = Path("paper/revision_round1/figures")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "legend.fontsize": 6.8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "pdf.fonttype": 42,
    }
)

shots = np.array([1, 5, 10])
pre = np.array([0.634681, 0.638581, 0.645907])
sgd = np.array([0.671261, 0.703600, 0.709362])
proto = np.array([0.730913, 0.770859, 0.796921])
knn = np.array([0.730913, 0.764957, 0.797074])

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.35))
ax = axes[0]
ax.plot(shots, pre, "o-", lw=1.35, ms=4, color="#777777", label="Pre")
ax.plot(shots, sgd, "s-", lw=1.35, ms=4, color="#d55e00", label="Linear SGD")
ax.plot(shots, proto, "D-", lw=1.45, ms=4, color="#0072b2", label="Prototype")
ax.plot(shots, knn, "^-", lw=1.2, ms=4, color="#009e73", label="1-NN")
ax.set(xlabel="Support shots per class ($K$)", ylabel="Mean macro-F1", xticks=shots, ylim=(0.62, 0.82))
ax.grid(axis="y", alpha=0.22)
ax.legend(ncol=2, loc="lower right", frameon=False)
ax.set_title("(a) Full eligible DS2", fontsize=8)

ax = axes[1]
host = np.array([0.798114, 0.784914, 0.803623])
device = np.array([0.798222, 0.784933, 0.803643])
x = np.arange(3)
w = 0.34
ax.bar(x - w / 2, host, w, color="#9ecae1", edgecolor="#0072b2", linewidth=0.5, label="Host")
ax.bar(x + w / 2, device, w, color="#0072b2", label="PSoC 6")
ax.set(xlabel="Support shots per class ($K$)", ylabel="Mean macro-F1", xticks=x, xticklabels=shots, ylim=(0.75, 0.82))
ax.grid(axis="y", alpha=0.22)
ax.legend(frameon=False, loc="lower right")
ax.set_title("(b) Matched capped replay", fontsize=8)

fig.tight_layout(pad=0.45, w_pad=1.2)
fig.savefig(OUT / "offline_results.pdf", bbox_inches="tight")
fig.savefig(OUT / "offline_results.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# The submitted restricted-ablation figure mixed accuracy with Python host
# timing.  The latter is not a like-for-like proxy for the measured CM4 cost,
# so the revision retains only the accuracy ablation requested by the paper.
paired = json.loads(Path("results/revision_round1/paired_normal_only.json").read_text())["shots"]
restricted_pre = np.array([paired[str(k)]["mean_pre_f1_macro"] for k in shots])
restricted = np.array([paired[str(k)]["mean_restricted_f1_macro"] for k in shots])
restricted_full = np.array([paired[str(k)]["mean_full_f1_macro"] for k in shots])

fig, ax = plt.subplots(figsize=(3.45, 2.3))
ax.plot(shots, restricted_pre, "o-", lw=1.35, ms=4, color="#777777", label="Pre")
ax.plot(shots, restricted, "^-", lw=1.35, ms=4, color="#a6611a", label="Normal-only")
ax.plot(shots, restricted_full, "D-", lw=1.45, ms=4, color="#0072b2", label="Full prototype")
ax.set(
    xlabel="Support shots ($K$; normal only for restricted)",
    ylabel="Mean macro-F1",
    xticks=shots,
    ylim=(0.62, 0.82),
)
ax.grid(axis="y", alpha=0.22)
ax.legend(frameon=False, loc="lower right")
fig.tight_layout(pad=0.45)
fig.savefig(OUT / "restricted_ablation.pdf", bbox_inches="tight")
fig.savefig(OUT / "restricted_ablation.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# Hardware-only calibration cost. Host Python timings are deliberately
# omitted because they are platform-specific and did not cover the same work.
proto_ms = np.array([22.82658, 114.04145, 228.06116])
sgd_ms = np.array([25.97186, 124.51673, 247.69255])

fig, ax = plt.subplots(figsize=(3.45, 2.3))
ax.plot(shots, proto_ms, "D-", lw=1.45, ms=4, color="#0072b2", label="Prototype")
ax.plot(shots, sgd_ms, "s-", lw=1.35, ms=4, color="#d55e00", label="Linear SGD")
ax.set(
    xlabel="Support shots per class ($K$)",
    ylabel="CM4 adaptation time (ms)",
    xticks=shots,
    ylim=(0, 270),
)
ax.grid(axis="y", alpha=0.22)
ax.legend(frameon=False, loc="upper left")
fig.tight_layout(pad=0.45)
fig.savefig(OUT / "compute_cost.pdf", bbox_inches="tight")
fig.savefig(OUT / "compute_cost.png", dpi=300, bbox_inches="tight")
plt.close(fig)
