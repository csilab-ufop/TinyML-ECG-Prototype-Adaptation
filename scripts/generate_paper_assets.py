"""Generate all paper figures and tables from offline experiment results.

Run from the repository root:
    python scripts/generate_paper_assets.py

Outputs go to paper/figures/ and paper/tables/.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "offline"
PAPER_DIR = ROOT / "paper"
TABLES_DIR = PAPER_DIR / "tables"
FIGURES_DIR = PAPER_DIR / "figures"

# ---------------------------------------------------------------------------
# Unified IEEE-compatible colour palette
# ---------------------------------------------------------------------------
PAL = {
    # Adaptation methods (used consistently in ALL charts)
    "pre":        "#555555",   # dark gray       → pre-adaptation baseline
    "sgd":        "#D6604D",   # brick red        → Linear SGD
    "proto":      "#2166AC",   # deep blue        → Prototype (full)
    "rest":       "#8C510A",   # ochre/brown      → Prototype (restricted)

    # Lighter fills for bar charts and fills
    "pre_l":      "#AAAAAA",
    "sgd_l":      "#F4A582",
    "proto_l":    "#92C5DE",
    "rest_l":     "#DFC27D",

    # Block-diagram background fills (very light)
    "bk_train":   "#DAEAF6",   # light blue  → offline training
    "bk_export":  "#FDF2E0",   # light cream → export / transfer
    "bk_device":  "#E2F2DF",   # light green → on-device
    "bk_head":    "#FCE8E8",   # light pink  → head adaptation
    "bk_kitprog": "#E8EDF5",   # steel blue  → KitProg3
    "bk_target":  "#E8F4E8",   # sage green  → PSoC target
    "bk_ble":     "#EDE8F5",   # lavender    → BLE
    "bk_uart":    "#FEF9EC",   # warm cream  → UART / result

    # Edges and borders
    "edge":  "#2C2C2C",
    "edge2": "#555555",
    "snap":  "#888888",   # snap-away line
}

# ---------------------------------------------------------------------------
# Shared matplotlib style
# ---------------------------------------------------------------------------

def _apply_ieee_style() -> None:
    plt.rcParams.update({
        "font.family":          "serif",
        "font.serif":           ["Times New Roman", "DejaVu Serif"],
        "font.size":            8.5,
        "axes.titlesize":       9.5,
        "axes.labelsize":       8.5,
        "xtick.labelsize":      8.0,
        "ytick.labelsize":      8.0,
        "legend.fontsize":      7.5,
        "legend.frameon":       False,
        "legend.handlelength":  1.6,
        "lines.linewidth":      1.6,
        "lines.markersize":     5.0,
        "axes.linewidth":       0.8,
        "grid.linewidth":       0.5,
        "grid.color":           "#CCCCCC",
        "axes.grid":            True,
        "axes.grid.axis":       "y",
        "axes.spines.top":      False,
        "axes.spines.right":    False,
        "figure.dpi":           150,
    })


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _fmt(value: float, decimals: int = 3) -> str:
    return f"{value:.{decimals}f}"


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{stem}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _draw_box(ax, x, y, w, h, color, text, fontsize=8.5, edge=None, lw=0.9):
    edge = edge or PAL["edge"]
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.015,rounding_size=0.015",
        linewidth=lw, edgecolor=edge, facecolor=color,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center", fontsize=fontsize,
            color=PAL["edge"], multialignment="center")


def _arrow(ax, x0, y0, x1, y1, color="#2C2C2C", lw=1.2):
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle="->", lw=lw, color=color,
                        connectionstyle="arc3,rad=0.0"),
    )


# ===========================================================================
# FIGURE 1 — System overview (redesigned, IEEE block-diagram style)
# ===========================================================================

def _generate_system_overview() -> None:
    _apply_ieee_style()
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    # ---- Phase labels -------------------------------------------------------
    ax.text(0.175, 0.96, "Offline Phase",
            ha="center", va="top", fontsize=8, color="#444444",
            style="italic")
    ax.text(0.73, 0.96, "On-Device Phase  (PSoC 6 CM4)",
            ha="center", va="top", fontsize=8, color="#444444",
            style="italic")
    ax.axvline(0.415, ymin=0.05, ymax=0.92, color=PAL["snap"],
               linewidth=0.9, linestyle="--")

    # ---- Offline blocks -----------------------------------------------------
    _draw_box(ax, 0.01, 0.22, 0.22, 0.56, PAL["bk_train"],
              "DS1 offline training\nTiny 1-D CNN\n(backbone + head)")
    _draw_box(ax, 0.275, 0.30, 0.12, 0.42, PAL["bk_export"],
              "Export\nweights +\nepisodes")

    # ---- On-device blocks ---------------------------------------------------
    _draw_box(ax, 0.44, 0.22, 0.21, 0.56, PAL["bk_device"],
              "Frozen backbone\nPSoC 6 CM4\nemb. dim = 32")
    _draw_box(ax, 0.71, 0.54, 0.17, 0.24, PAL["bk_head"],
              "Linear head\nSGD on $W,b$")
    _draw_box(ax, 0.71, 0.22, 0.17, 0.24, PAL["bk_head"],
              "Prototype head\nclass means $p_c$")

    # ---- Arrows -------------------------------------------------------------
    _arrow(ax, 0.23, 0.50, 0.275, 0.50)
    _arrow(ax, 0.395, 0.50, 0.44, 0.50)
    _arrow(ax, 0.65, 0.62, 0.71, 0.65)
    _arrow(ax, 0.65, 0.38, 0.71, 0.35)

    # ---- DS2 annotation -----------------------------------------------------
    ax.text(0.67, 0.10,
            "DS2 support beats consumed only on-device",
            ha="center", va="bottom", fontsize=7.5, color="#555555",
            style="italic")

    fig.tight_layout(pad=0.3)
    _save_figure(fig, "system_overview")


# ===========================================================================
# FIGURE 2 — Prototype adaptation (redesigned, consistent palette)
# ===========================================================================

def _generate_prototype_figure() -> None:
    _apply_ieee_style()
    fig, ax = plt.subplots(figsize=(6.8, 2.5))
    ax.set_xlim(-0.2, 10.2)
    ax.set_ylim(-0.5, 4.2)
    ax.set_axis_off()

    left_x, right_x = 2.2, 7.5
    ax.text(left_x, 3.9, "Support embeddings", ha="center", fontsize=9.5,
            fontweight="bold", color=PAL["edge"])
    ax.text(right_x, 3.9, "Prototype classification", ha="center", fontsize=9.5,
            fontweight="bold", color=PAL["edge"])

    # Support points — left panel
    c0_pts = [(1.0, 1.0), (1.6, 1.4), (2.0, 1.1), (1.4, 0.7)]
    c1_pts = [(2.8, 2.4), (3.2, 2.0), (2.5, 1.8), (3.4, 2.6)]
    for (x, y) in c0_pts:
        ax.scatter(x, y, s=55, color=PAL["proto"], zorder=3, marker="o")
    for (x, y) in c1_pts:
        ax.scatter(x, y, s=55, color=PAL["sgd"], zorder=3, marker="s")

    ax.text(1.2, 0.22, "class 0 (N)", color=PAL["proto"], fontsize=8)
    ax.text(2.9, 2.95, "class 1 (Arr.)", color=PAL["sgd"], fontsize=8)

    # Arrow between panels
    ax.annotate("", xy=(5.1, 1.9), xytext=(4.0, 1.9),
                arrowprops=dict(arrowstyle="-|>", lw=1.5, color=PAL["edge"],
                                mutation_scale=14))

    # Prototype markers — right panel
    proto0 = (6.8, 1.05)
    proto1 = (8.2, 2.3)
    ax.scatter(*proto0, s=200, marker="X", color=PAL["proto"], zorder=4, linewidths=0.5)
    ax.scatter(*proto1, s=200, marker="X", color=PAL["sgd"],   zorder=4, linewidths=0.5)
    ax.text(proto0[0] - 0.05, proto0[1] - 0.48,
            r"$p_0 = \mathrm{mean}(z_i)$", ha="center", fontsize=8, color=PAL["proto"])
    ax.text(proto1[0] + 0.05, proto1[1] + 0.38,
            r"$p_1 = \mathrm{mean}(z_i)$", ha="center", fontsize=8, color=PAL["sgd"])

    # Query point
    query = (9.0, 1.45)
    ax.scatter(*query, s=80, marker="D", color=PAL["rest"], zorder=4)
    ax.text(query[0] + 0.15, query[1] + 0.10, "query $z_q$", fontsize=8, color=PAL["rest"])

    # Distance lines
    ax.plot([query[0], proto0[0]], [query[1], proto0[1]],
            color=PAL["proto"], linestyle="--", linewidth=1.4, zorder=2)
    ax.plot([query[0], proto1[0]], [query[1], proto1[1]],
            color=PAL["sgd"],   linestyle=":",  linewidth=1.4, zorder=2)

    # Mid-label on shorter distance
    mx, my = (query[0] + proto0[0]) / 2, (query[1] + proto0[1]) / 2
    ax.text(mx + 0.1, my - 0.28, r"$\|z_q - p_0\|$", fontsize=7.5, color=PAL["proto"])

    ax.text(7.5, 0.18, r"$\hat{y} = \arg\min_c \|z_q - p_c\|_2^2$",
            ha="center", fontsize=8.5, color=PAL["edge"])

    fig.tight_layout(pad=0.3)
    _save_figure(fig, "prototype_adaptation")


# ===========================================================================
# FIGURE 4 — Embedded platform (completely redesigned)
# ===========================================================================

def _generate_embedded_platform_figure() -> None:
    _apply_ieee_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_axis_off()

    # =========================================================================
    # LEFT PANEL: CY8CPROTO-063-BLE hardware block diagram
    # =========================================================================
    left = axes[0]
    left.set_title("CY8CPROTO-063-BLE Hardware", fontsize=9, pad=4,
                   fontweight="bold", color=PAL["edge"])

    # Background sections: KitProg3 subsystem | Target subsystem
    left.add_patch(FancyBboxPatch(
        (0.01, 0.03), 0.42, 0.90,
        boxstyle="round,pad=0.01,rounding_size=0.01",
        lw=0.8, edgecolor=PAL["edge2"], facecolor=PAL["bk_kitprog"], alpha=0.45))
    left.add_patch(FancyBboxPatch(
        (0.55, 0.03), 0.43, 0.90,
        boxstyle="round,pad=0.01,rounding_size=0.01",
        lw=0.8, edgecolor=PAL["edge2"], facecolor=PAL["bk_target"], alpha=0.45))

    # Section labels
    left.text(0.23, 0.965, "KitProg3 Subsystem",
              ha="center", fontsize=7.5, style="italic", color="#34557A")
    left.text(0.775, 0.965, "PSoC 63 BLE Target",
              ha="center", fontsize=7.5, style="italic", color="#2D6B3C")

    # Snap-away dashed line
    left.axvline(0.49, ymin=0.04, ymax=0.96, color=PAL["snap"],
                 linewidth=1.0, linestyle=(0, (4, 2)))
    left.text(0.49, 0.01, "snap-away", ha="center", fontsize=6.5,
              color=PAL["snap"])

    # ----- KitProg3 side -------------------------------------------------
    _draw_box(left, 0.04, 0.70, 0.18, 0.18, PAL["bk_export"],
              "USB\nMicro-B\n(J10)", fontsize=7.5)
    _draw_box(left, 0.04, 0.39, 0.35, 0.25, PAL["bk_kitprog"],
              "KitProg3 MCU\nSWD bridge\nUART bridge\nFirmware loader", fontsize=7.5,
              lw=1.0)
    _draw_box(left, 0.04, 0.08, 0.35, 0.22, PAL["bk_export"],
              "3.3 V LDO\nVTARG rail", fontsize=7.5)

    # USB → KitProg
    _arrow(left, 0.13, 0.70, 0.13, 0.64, lw=1.1)
    # LDO → KitProg (power)
    _arrow(left, 0.22, 0.30, 0.22, 0.39, lw=1.1)

    # ----- PSoC 63 target side -------------------------------------------
    # Main PSoC 63 block
    _draw_box(left, 0.57, 0.55, 0.39, 0.38, PAL["bk_device"],
              "PSoC 63 BLE\n(CYBLE-416045-02)\nCM4 @ 150 MHz\nCM0+ @ 100 MHz",
              fontsize=7.5, lw=1.0)
    _draw_box(left, 0.57, 0.25, 0.18, 0.22, PAL["bk_ble"],
              "BLE RF\nAntenna", fontsize=7.5)
    _draw_box(left, 0.78, 0.25, 0.18, 0.22, PAL["bk_uart"],
              "Debug UART\nP5_0 / P5_1", fontsize=7.5)

    # BLE and UART connected to PSoC
    _arrow(left, 0.66, 0.55, 0.66, 0.47, lw=1.0)
    _arrow(left, 0.87, 0.55, 0.87, 0.47, lw=1.0)

    # ----- Cross-section connections (KitProg ↔ Target) ------------------
    kw = dict(lw=1.0, color="#555555")
    # SWD
    left.annotate("", xy=(0.57, 0.72), xytext=(0.39, 0.57),
                  arrowprops=dict(arrowstyle="->", **kw))
    left.text(0.48, 0.68, "SWD", fontsize=6.5, color="#555555", ha="center")
    # UART bridge
    left.annotate("", xy=(0.57, 0.62), xytext=(0.39, 0.48),
                  arrowprops=dict(arrowstyle="<->", **kw))
    left.text(0.48, 0.52, "UART", fontsize=6.5, color="#555555", ha="center")
    # 3.3 V power
    left.annotate("", xy=(0.57, 0.57), xytext=(0.39, 0.18),
                  arrowprops=dict(arrowstyle="->", lw=1.0, color="#888888",
                                  connectionstyle="arc3,rad=-0.3"))
    left.text(0.48, 0.35, "3.3 V", fontsize=6.5, color="#888888", ha="center")

    # =========================================================================
    # RIGHT PANEL: Firmware replay execution path
    # =========================================================================
    right = axes[1]
    right.set_title("On-Device Replay Execution (CM4)", fontsize=9, pad=4,
                    fontweight="bold", color=PAL["edge"])

    # Boxes (top-to-bottom flow)
    _draw_box(right, 0.05, 0.76, 0.40, 0.18, PAL["bk_export"],
              "Exported replay pack\n(support + query beats)", fontsize=7.5)
    _draw_box(right, 0.55, 0.76, 0.40, 0.18, PAL["bk_device"],
              "CM4 app\nload episode", fontsize=7.5)
    _draw_box(right, 0.55, 0.50, 0.40, 0.18, PAL["bk_device"],
              "Frozen backbone\n32-D embedding", fontsize=7.5)
    _draw_box(right, 0.05, 0.26, 0.38, 0.17, PAL["bk_head"],
              "Linear head\nSGD on $W,b$", fontsize=7.5)
    _draw_box(right, 0.57, 0.26, 0.38, 0.17, PAL["bk_head"],
              "Prototype head\nclass means $p_c$", fontsize=7.5)
    _draw_box(right, 0.25, 0.04, 0.50, 0.14, PAL["bk_uart"],
              "UART log — accuracy, macro-F1", fontsize=7.5)

    # Arrows
    _arrow(right, 0.45, 0.85, 0.55, 0.85)          # replay → CM4 app
    _arrow(right, 0.75, 0.76, 0.75, 0.68)          # CM4 app → backbone
    _arrow(right, 0.55, 0.59, 0.43, 0.43)          # backbone → linear head
    _arrow(right, 0.75, 0.50, 0.76, 0.43)          # backbone → proto head
    _arrow(right, 0.24, 0.26, 0.42, 0.18)          # linear → UART
    _arrow(right, 0.76, 0.26, 0.58, 0.18)          # proto  → UART

    right.text(0.50, 0.61,
               "same embedding,\ntwo head rules",
               ha="center", fontsize=7, color="#555555", style="italic")

    fig.tight_layout(pad=0.4)
    _save_figure(fig, "embedded_platform")


# ===========================================================================
# ECG EXAMPLES
# ===========================================================================

def _generate_ecg_examples() -> None:
    _apply_ieee_style()
    dataset = np.load(ROOT / "data" / "processed" / "mitbih_binary.npz",
                      allow_pickle=True)
    signals = dataset["x"][:, 0, :]
    labels  = dataset["y"]
    symbols = dataset["symbols"]
    records = dataset["records"]
    split_g = dataset["split_group"]

    normal_idx = np.where((split_g == "DS2") & (labels == 0))[0]
    abnormal_idx = {
        s: np.where((split_g == "DS2") & (labels == 1) & (symbols == s))[0]
        for s in ["L", "R", "V", "A"]
    }

    chosen_normal: list[int] = []
    used_recs: set[str] = set()
    for idx in normal_idx:
        rec = str(records[idx])
        if rec in used_recs:
            continue
        chosen_normal.append(idx)
        used_recs.add(rec)
        if len(chosen_normal) == 4:
            break

    chosen_abnormal = [arr[0] for arr in abnormal_idx.values()]
    x_ax = np.arange(signals.shape[1]) - 90

    fig, axes = plt.subplots(2, 4, figsize=(7.0, 2.8), sharex=True, sharey=True)

    for ax, idx in zip(axes[0], chosen_normal):
        ax.plot(x_ax, signals[idx], color=PAL["proto"], linewidth=1.4)
        ax.set_title(f"N / rec {records[idx]}", fontsize=7.5)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_xticks([-80, 0, 80])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax, idx in zip(axes[1], chosen_abnormal):
        ax.plot(x_ax, signals[idx], color=PAL["sgd"], linewidth=1.4)
        ax.set_title(f"{symbols[idx]} / rec {records[idx]}", fontsize=7.5)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_xticks([-80, 0, 80])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes[1]:
        ax.set_xlabel("Samples from R-peak", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("z-score", fontsize=8)

    fig.suptitle("MIT-BIH beat windows: normal (top) and abnormal (bottom)",
                 fontsize=9, y=1.02)
    fig.tight_layout(pad=0.4)
    _save_figure(fig, "ecg_examples")


# ===========================================================================
# FIGURE 5 — Offline results (harmonised palette)
# ===========================================================================

def _generate_results_plot(protocols: list[dict]) -> None:
    _apply_ieee_style()
    protocols_s = sorted(protocols, key=lambda p: p["shots"])
    shots   = [p["shots"] for p in protocols_s]
    pre_f1  = [p["aggregate"]["pre"]["f1_macro"]          for p in protocols_s]
    lin_f1  = [p["aggregate"]["post_linear"]["f1_macro"]  for p in protocols_s]
    pro_f1  = [p["aggregate"]["post_prototypes"]["f1_macro"] for p in protocols_s]
    lin_ms  = [p["aggregate"]["mean_linear_adaptation_time_ms"] for p in protocols_s]
    pro_ms  = [p["aggregate"]["mean_prototype_time_ms"]   for p in protocols_s]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))

    ax = axes[0]
    ax.plot(shots, pre_f1, "o-", color=PAL["pre"],   label="Pre",          lw=1.6, ms=5)
    ax.plot(shots, lin_f1, "s-", color=PAL["sgd"],   label="Linear SGD",   lw=1.6, ms=5)
    ax.plot(shots, pro_f1, "D-", color=PAL["proto"], label="Prototypes",   lw=1.6, ms=5)
    ax.set_xlabel("Support shots per class ($K$)")
    ax.set_ylabel("Macro-F1")
    ax.set_xticks(shots)
    ax.set_ylim(0.60, 0.83)
    ax.legend(loc="lower right")
    ax.set_title("(a) DS2 macro-F1", fontsize=9)

    ax = axes[1]
    w, x = 0.35, np.arange(len(shots))
    ax.bar(x - w / 2, lin_ms, width=w, color=PAL["sgd"],   label="Linear SGD")
    ax.bar(x + w / 2, pro_ms, width=w, color=PAL["proto"], label="Prototypes")
    ax.set_xticks(x, [str(s) for s in shots])
    ax.set_xlabel("Support shots per class ($K$)")
    ax.set_ylabel("Adaptation time (ms)")
    ax.legend(loc="upper left")
    ax.set_title("(b) Reference adaptation cost", fontsize=9)

    fig.tight_layout(pad=0.4)
    _save_figure(fig, "offline_results")


# ===========================================================================
# FIGURE 6 — Restricted ablation (harmonised palette)
# ===========================================================================

def _common_subset_rows(full: dict, restr: dict) -> list[tuple[dict, dict]]:
    full_map = {r["record"]: r for r in full["per_record"]}
    return [(full_map[r["record"]], r)
            for r in restr["per_record"] if r["record"] in full_map]


def _generate_restricted_ablation_plot(full_p: list[dict], restr_p: list[dict]) -> None:
    _apply_ieee_style()
    full_s  = sorted(full_p,  key=lambda p: p["shots"])
    restr_s = sorted(restr_p, key=lambda p: p["shots"])

    shots, com_pre, restr_f1, full_f1, restr_ms, full_ms = [], [], [], [], [], []
    for fp, rp in zip(full_s, restr_s):
        common = _common_subset_rows(fp, rp)
        shots.append(rp["shots"])
        com_pre.append(np.mean([r["pre"]["f1_macro"] for _, r in common]))
        restr_f1.append(np.mean([r["post_restricted_prototypes"]["f1_macro"] for _, r in common]))
        full_f1.append(np.mean([f["post_prototypes"]["f1_macro"] for f, _ in common]))
        restr_ms.append(np.mean([r["restricted_prototype_time_ms"] for _, r in common]))
        full_ms.append(np.mean([f["prototype_time_ms"] for f, _ in common]))

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))

    ax = axes[0]
    ax.plot(shots, com_pre,   "o-", color=PAL["pre"],   label="Pre",        lw=1.6, ms=5)
    ax.plot(shots, restr_f1,  "^-", color=PAL["rest"],  label="Restr. proto", lw=1.6, ms=5)
    ax.plot(shots, full_f1,   "D-", color=PAL["proto"], label="Full proto",  lw=1.6, ms=5)
    ax.set_xlabel("Support shots ($K$, normal class only for restricted)")
    ax.set_ylabel("Macro-F1")
    ax.set_xticks(shots)
    ax.set_ylim(0.60, 0.83)
    ax.legend(loc="lower right")
    ax.set_title("(a) Macro-F1 comparison", fontsize=9)

    ax = axes[1]
    w, x = 0.35, np.arange(len(shots))
    ax.bar(x - w / 2, restr_ms, width=w, color=PAL["rest"],  label="Restricted")
    ax.bar(x + w / 2, full_ms,  width=w, color=PAL["proto"], label="Full proto")
    ax.set_xticks(x, [str(s) for s in shots])
    ax.set_xlabel("Support shots ($K$)")
    ax.set_ylabel("Adaptation time (ms)")
    ax.legend(loc="upper left")
    ax.set_title("(b) Reference adaptation cost", fontsize=9)

    fig.tight_layout(pad=0.4)
    _save_figure(fig, "restricted_ablation")


# ===========================================================================
# FIGURE 7 — Compute cost (harmonised palette)
# ===========================================================================

def _generate_compute_cost_figure(cost: dict, protocols: list[dict]) -> None:
    _apply_ieee_style()
    dwt   = cost["measured_dwt_release"]
    cli   = cost["clinical"]
    macs  = cost["macs"]
    shots = [1, 5, 10]

    mcu_proto = [dwt["adapt_end_to_end"][str(k)]["proto_ms"] for k in shots]
    mcu_sgd   = [dwt["adapt_end_to_end"][str(k)]["sgd_ms"]   for k in shots]
    prot_s    = sorted(protocols, key=lambda p: p["shots"])
    sw_proto  = [p["aggregate"]["mean_prototype_time_ms"]          for p in prot_s]
    sw_sgd    = [p["aggregate"]["mean_linear_adaptation_time_ms"]  for p in prot_s]

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8))

    # (a) MAC breakdown -------------------------------------------------------
    ax = axes[0]
    lbls    = ["Conv1\n(8k)", "Conv2\n(64k)", "Proj\n(0.5k)", "Head\n(0.06k)"]
    mac_v   = [macs["conv1"], macs["conv2"], macs["proj"], macs["head"]]
    cols    = [PAL["pre_l"], PAL["proto_l"], PAL["rest_l"], PAL["sgd_l"]]
    bars = ax.barh(lbls, mac_v, color=cols, edgecolor=PAL["edge"], linewidth=0.6)
    ax.set_xlabel("MACs")
    ax.set_title("(a) Operation count", fontsize=9)
    ax.set_xscale("log")
    ax.grid(True, axis="x", alpha=0.3)
    ax.grid(False, axis="y")
    for bar, val in zip(bars, mac_v):
        ax.text(val * 1.2, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # (b) Adaptation cost (log scale) -----------------------------------------
    ax = axes[1]
    x = np.arange(len(shots))
    bw = 0.20
    ax.bar(x - 1.5*bw, mcu_proto, width=bw, color=PAL["proto"],   label="MCU proto (meas.)")
    ax.bar(x - 0.5*bw, mcu_sgd,   width=bw, color=PAL["sgd"],     label="MCU SGD (meas.)")
    ax.bar(x + 0.5*bw, sw_proto,  width=bw, color=PAL["proto_l"], hatch="//",
           edgecolor=PAL["proto"], label="SW proto (ref.)")
    ax.bar(x + 1.5*bw, sw_sgd,    width=bw, color=PAL["sgd_l"],   hatch="//",
           edgecolor=PAL["sgd"],   label="SW SGD (ref.)")
    ax.set_xticks(x, [str(k) for k in shots])
    ax.set_xlabel("Support shots ($K$)")
    ax.set_ylabel("Adaptation time (ms)")
    ax.set_title("(b) Adaptation cost", fontsize=9)
    ax.set_yscale("log")
    ax.legend(fontsize=6.0, ncol=2, loc="upper left",
              handlelength=1.2, columnspacing=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # (c) Clinical timeline at 150 BPM ----------------------------------------
    ax = axes[2]
    rr      = cli["beat_interval_bpm150_ms"]
    post_r  = (110 / 360) * 1e3   # 110 post-R samples
    inf_ms  = dwt["inference"]["time_ms"]
    a10_ms  = mcu_proto[2]        # 10-shot proto adapt (measured)

    ax.set_xlim(0, rr + 40)
    ax.set_ylim(-0.5, 4.5)
    ax.set_xlabel("Time after R-peak (ms)")
    ax.set_title("(c) 150 BPM timeline", fontsize=9)
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_yticklabels(["Head\nclas.", "10-shot\nadapt.", "Infer.", "Post-R\nwait", "RR\ninterval"],
                       fontsize=7.5)
    ax.grid(True, axis="x", alpha=0.25)
    ax.grid(False, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    def _hbar(row, width, color, label=""):
        ax.barh(row, width, left=0, height=0.45, color=color,
                edgecolor=PAL["edge2"], linewidth=0.7)
        if label:
            ax.text(width / 2, row + 0.32, label,
                    ha="center", fontsize=6.8, color=PAL["edge"])

    _hbar(4, rr,        PAL["pre_l"],   f"RR = {rr:.0f} ms")
    _hbar(3, post_r,    PAL["rest_l"],  f"Post-R {post_r:.0f} ms")
    _hbar(2, inf_ms,    PAL["proto_l"], f"{inf_ms:.1f} ms")
    _hbar(1, a10_ms,    PAL["proto_l"], f"{a10_ms:.0f} ms")
    _hbar(0, max(0.01*inf_ms, 2.0), PAL["sgd_l"], "<0.01 ms")

    ax.axvline(rr, color="#888888", linestyle=":", linewidth=1.2, alpha=0.8)

    fig.tight_layout(pad=0.4)
    _save_figure(fig, "compute_cost")


# ===========================================================================
# FIGURE — Neural network architecture (new)
# ===========================================================================

def _generate_architecture_figure() -> None:  # noqa: C901
    """Elegant architecture diagram showing frozen backbone vs adaptive head."""
    _apply_ieee_style()

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.set_xlim(0, 10.5)
    ax.set_ylim(-0.55, 5.0)
    ax.set_axis_off()

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------
    CY = 2.65          # vertical centre of the main flow
    BH = 0.88          # box height
    BW_NORM = 1.00     # standard box width
    BW_WIDE = 1.20     # wide box width
    BW_IN   = 1.05     # input box

    def _box(xc, yc, w, h, color, lines, fontsize=7.8, lw=0.8, edge=None):
        edge = edge or PAL["edge"]
        rect = FancyBboxPatch(
            (xc - w / 2, yc - h / 2), w, h,
            boxstyle="round,pad=0.04,rounding_size=0.04",
            linewidth=lw, edgecolor=edge, facecolor=color, zorder=3,
        )
        ax.add_patch(rect)
        ax.text(xc, yc, "\n".join(lines),
                ha="center", va="center", fontsize=fontsize,
                color=PAL["edge"], multialignment="center", zorder=4)

    def _arr(x0, x1, y=CY, dy=0.0, lw=1.2):
        ax.annotate(
            "", xy=(x1, y + dy), xytext=(x0, y + dy),
            arrowprops=dict(arrowstyle="-|>", lw=lw, color="#444444",
                            mutation_scale=10),
            zorder=5,
        )

    def _dim(xc, y, text, color="#666666"):
        ax.text(xc, y, text, ha="center", va="top",
                fontsize=6.8, color=color, style="italic")

    def _fmap(xc, yc, n_ch, seq_len, color):
        """Mini feature-map: stacked horizontal bars, width ∝ seq_len."""
        max_ch_shown = min(n_ch, 7)
        bar_h = 0.090
        bar_w = (seq_len / 200.0) * 0.52
        x0    = xc - bar_w / 2
        base  = yc - max_ch_shown * bar_h / 2
        # colour alternation for visual depth
        light = tuple(min(1.0, c + 0.18) for c in matplotlib.colors.to_rgb(color))
        for i in range(max_ch_shown):
            fc = color if i % 2 == 0 else light
            ax.add_patch(plt.Rectangle(
                (x0, base + i * bar_h), bar_w, bar_h * 0.82,
                facecolor=fc, edgecolor="#999999", linewidth=0.30, zorder=3,
            ))
        if n_ch > max_ch_shown:
            ax.text(xc, base - 0.08, f"({n_ch} ch.)",
                    ha="center", va="top", fontsize=6.0, color="#888888")

    # -----------------------------------------------------------------------
    # Background regions
    # -----------------------------------------------------------------------
    # Frozen backbone region
    ax.add_patch(FancyBboxPatch(
        (0.12, 0.55), 6.80, 4.20,
        boxstyle="round,pad=0.08,rounding_size=0.08",
        linewidth=1.1, linestyle="--",
        edgecolor=PAL["proto"], facecolor=PAL["bk_device"], alpha=0.18, zorder=1,
    ))
    ax.text(3.55, 4.90, "Frozen Backbone  — no gradient on-device  (∇ = 0)",
            ha="center", va="top", fontsize=8.5, color=PAL["proto"],
            style="italic", fontweight="bold")

    # Adaptive head region
    ax.add_patch(FancyBboxPatch(
        (7.10, 0.55), 3.25, 4.20,
        boxstyle="round,pad=0.08,rounding_size=0.08",
        linewidth=1.1, linestyle="--",
        edgecolor=PAL["sgd"], facecolor=PAL["bk_head"], alpha=0.18, zorder=1,
    ))
    ax.text(8.72, 4.90, "Adaptive Head",
            ha="center", va="top", fontsize=8.5, color=PAL["sgd"],
            style="italic", fontweight="bold")

    # Lock symbol on backbone (top-left of backbone region)
    lx, ly = 0.50, 4.55
    arc = matplotlib.patches.Arc(
        (lx, ly + 0.17), 0.18, 0.22,
        theta1=0, theta2=180, color=PAL["proto"], lw=1.5, zorder=6,
    )
    ax.add_patch(arc)
    ax.add_patch(plt.Rectangle(
        (lx - 0.12, ly), 0.24, 0.17,
        facecolor=PAL["bk_device"], edgecolor=PAL["proto"], lw=1.2, zorder=6,
    ))
    ax.add_patch(plt.Circle((lx, ly + 0.085), 0.038, color=PAL["proto"], zorder=7))
    ax.plot([lx, lx], [ly + 0.085 - 0.038, ly + 0.03],
            color="#FFFFFF", lw=1.5, zorder=8)

    # Update arrow on head (top-right of head region)
    ax.annotate(
        "", xy=(9.10, 4.62), xytext=(9.10, 4.28),
        arrowprops=dict(arrowstyle="-|>", lw=1.4, color=PAL["sgd"],
                        mutation_scale=11),
        zorder=6,
    )
    ax.text(9.26, 4.45, "update", fontsize=7.0, color=PAL["sgd"],
            va="center", style="italic")

    # -----------------------------------------------------------------------
    # Layout: x-centres of feature maps and operation boxes
    # -----------------------------------------------------------------------
    # Feature map visual positions (between ops)
    #   x: input   conv1box  fmap1   conv2box  fmap2   pool+avg  fmap3  proj   emb
    #      0.80     —         2.20    —         3.85    —         5.00   —      6.15
    # Operation box positions (midway between feature maps)

    xs_fmap = [0.80, 2.18, 3.82, 5.00, 6.18]   # input, post-c1, post-c2, post-avg, embedding
    xs_op   = [1.50, 3.00, 4.42, 5.60]          # conv1, conv2, avg+proj, (head is separate)

    # -----------------------------------------------------------------------
    # Operation boxes
    # -----------------------------------------------------------------------
    # Conv Block 1
    _box(xs_op[0], CY, BW_NORM, BH, PAL["bk_device"], [
        "Conv1D", "1→8 ch", "kernel 5", "+ ReLU"
    ], lw=1.0, edge=PAL["proto"])

    # Conv Block 2
    _box(xs_op[1], CY, BW_NORM, BH, PAL["bk_device"], [
        "Conv1D", "8→16 ch", "kernel 5", "+ ReLU"
    ], lw=1.0, edge=PAL["proto"])

    # MaxPool labels (small, on the arrows)
    ax.text(2.68, CY + 0.54, "MaxPool /2", ha="center", fontsize=6.8, color="#555555")
    ax.text(4.20, CY + 0.54, "MaxPool /2", ha="center", fontsize=6.8, color="#555555")

    # Global AvgPool + Projection (combined narrow box)
    _box(xs_op[2], CY, BW_NORM * 0.88, BH, PAL["bk_device"], [
        "AvgPool", "(global)", "→ Linear", "16→32"
    ], lw=1.0, edge=PAL["proto"])

    # Head box (adaptive)
    _box(8.48, CY, BW_WIDE * 0.90, BH * 1.55, PAL["bk_head"], [
        "Head", "", "Linear SGD", "or Prototype", "class means"
    ], lw=1.2, edge=PAL["sgd"], fontsize=8.0)

    # -----------------------------------------------------------------------
    # Feature map mini-visualisations
    # -----------------------------------------------------------------------
    fmap_y_vis = CY - BH / 2 - 1.00   # below the main line
    fmap_y_dim = fmap_y_vis - 0.78    # dimension label

    specs = [
        # (n_ch, seq_len, color, dim_text)
        (1,  200, PAL["bk_export"],  "1 × 200"),
        (8,  100, PAL["proto_l"],    "8 × 100"),
        (16,  50, PAL["proto_l"],    "16 × 50"),
        (16,   1, PAL["proto_l"],    "16 × 1"),
        (32,   1, PAL["bk_train"],   "32-D emb. $z$"),
    ]
    for xc, (n_ch, seq_len, color, dim) in zip(xs_fmap, specs):
        _fmap(xc, fmap_y_vis, n_ch, seq_len, color)
        _dim(xc, fmap_y_dim, dim)
        # vertical connector from mini-fmap to main box
        ax.plot([xc, xc], [fmap_y_vis + min(n_ch, 7) * 0.090 / 2, CY - BH / 2],
                color="#AAAAAA", lw=0.6, linestyle=":", zorder=2)

    # Input ECG label below first feature map
    ax.text(xs_fmap[0], fmap_y_dim - 0.50, "Input ECG\n(z-score, 200 samp.)",
            ha="center", va="top", fontsize=6.5, color="#555555",
            style="italic", multialignment="center")

    # -----------------------------------------------------------------------
    # Arrows (main horizontal flow)
    # -----------------------------------------------------------------------
    # Input fmap → conv1
    _arr(xs_fmap[0] + 0.27, xs_op[0] - BW_NORM / 2)
    # conv1 → fmap1
    _arr(xs_op[0] + BW_NORM / 2, xs_fmap[1] - 0.27)
    # fmap1 → conv2 (with MaxPool note baked in above)
    _arr(xs_fmap[1] + 0.20, xs_op[1] - BW_NORM / 2)
    # conv2 → fmap2
    _arr(xs_op[1] + BW_NORM / 2, xs_fmap[2] - 0.18)
    # fmap2 → avg+proj
    _arr(xs_fmap[2] + 0.16, xs_op[2] - BW_NORM * 0.88 / 2)
    # avg+proj → embedding
    _arr(xs_op[2] + BW_NORM * 0.88 / 2, xs_fmap[3] - 0.16)
    # fmap3 (16×1) → embedding
    _arr(xs_fmap[3] + 0.14, xs_fmap[4] - 0.14)
    # embedding → head
    _arr(xs_fmap[4] + 0.20, 8.48 - BW_WIDE * 0.90 / 2)

    # -----------------------------------------------------------------------
    # Output classes
    # -----------------------------------------------------------------------
    cls_x = 9.62
    for letter, desc, yoff, col in [
        ("N",   "class 0 · Normal",     +0.55, PAL["proto"]),
        ("Arr", "class 1 · Arrhythmia", -0.55, PAL["sgd"]),
    ]:
        ax.add_patch(plt.Circle((cls_x, CY + yoff), 0.22,
                                facecolor=col, edgecolor=PAL["edge"],
                                linewidth=0.8, alpha=0.85, zorder=4))
        ax.text(cls_x, CY + yoff, letter,
                ha="center", va="center", fontsize=7.5,
                color="white", fontweight="bold", zorder=5)
        ax.text(cls_x, CY + yoff + (0.32 if yoff > 0 else -0.32),
                desc, ha="center", va="center", fontsize=6.2,
                color=col, zorder=5)
        _arr(8.48 + BW_WIDE * 0.90 / 2, cls_x - 0.22, y=CY + yoff, lw=1.1)

    # -----------------------------------------------------------------------
    # Parameter count strip at very bottom
    # -----------------------------------------------------------------------
    ax.text(4.8, -0.35, "Total: 1,314 param. · 72,576 MACs",
            ha="center", va="bottom", fontsize=7.2, color=PAL["edge"],
            fontweight="bold")

    fig.tight_layout(pad=0.2)
    _save_figure(fig, "architecture")


# ===========================================================================
# TABLES
# ===========================================================================

def _generate_offline_table(train_metrics: dict, protocols: list[dict]) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\caption{DS2 aggregate results for the de~Chazal inter-patient protocol.",
        r"Adaptation times are host-side reference measurements (Python/PyTorch);",
        r"MCU-estimated times appear in Table~\ref{tab:compute-cost}.}",
        r"\label{tab:offline-results}",
        r"\centering",
        r"\begin{tabular}{l ccc rr}",
        r"\toprule",
        r"$K$ & Pre F1 & Linear F1 & Proto F1 & Linear (ms) & Proto (ms) \\",
        r"\midrule",
    ]
    for p in sorted(protocols, key=lambda p: p["shots"]):
        a = p["aggregate"]
        lines.append(
            f"{p['shots']} & "
            f"{_fmt(a['pre']['f1_macro'])} & "
            f"{_fmt(a['post_linear']['f1_macro'])} & "
            f"\\textbf{{{_fmt(a['post_prototypes']['f1_macro'])}}} & "
            f"{a['mean_linear_adaptation_time_ms']:.0f} & "
            f"{a['mean_prototype_time_ms']:.0f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    _write_text(TABLES_DIR / "offline_results.tex", "\n".join(lines) + "\n")


def _generate_gain_table(protocols: list[dict]) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\caption{Mean absolute macro-F1 gain over the pre-adaptation baseline",
        r"and number of DS2 records with positive gain per method.}",
        r"\label{tab:gain-summary}",
        r"\centering",
        r"\begin{tabular}{l cc cc}",
        r"\toprule",
        r"& \multicolumn{2}{c}{Mean $\Delta$F1} & \multicolumn{2}{c}{Records improved} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"$K$ & Linear SGD & Prototypes & Linear SGD & Prototypes \\",
        r"\midrule",
    ]
    for p in sorted(protocols, key=lambda p: p["shots"]):
        per = p["per_record"]
        n = len(per)
        lin_pos  = sum(1 for r in per if r["linear_gain"]["f1_abs_gain"] > 0)
        pro_pos  = sum(1 for r in per if r["prototype_gain"]["f1_abs_gain"] > 0)
        agg = p["aggregate"]
        lines.append(
            f"{p['shots']} & "
            f"{_fmt(agg['linear_gain']['f1_abs_gain'])} & "
            f"\\textbf{{{_fmt(agg['prototype_gain']['f1_abs_gain'])}}} & "
            f"{lin_pos}/{n} & "
            f"\\textbf{{{pro_pos}/{n}}} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    _write_text(TABLES_DIR / "gain_summary.tex", "\n".join(lines) + "\n")


def _generate_restricted_ablation_table(full_p: list[dict], restr_p: list[dict]) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\caption{Restricted-prototype ablation: only normal-beat support from the",
        r"target patient; arrhythmia prototype inherited from DS1.",
        r"Evaluated on the common eligible DS2 subset.}",
        r"\label{tab:restricted-ablation}",
        r"\centering",
        r"\begin{tabular}{l c ccc r}",
        r"\toprule",
        r"$K$ & Records & Pre F1 & Restr.\ F1 & Full proto F1 & Restr.\ (ms) \\",
        r"\midrule",
    ]
    for fp, rp in zip(sorted(full_p, key=lambda p: p["shots"]),
                      sorted(restr_p, key=lambda p: p["shots"])):
        common = _common_subset_rows(fp, rp)
        n      = len(common)
        pre_f  = np.mean([r["pre"]["f1_macro"] for _, r in common])
        re_f   = np.mean([r["post_restricted_prototypes"]["f1_macro"] for _, r in common])
        fu_f   = np.mean([f["post_prototypes"]["f1_macro"] for f, _ in common])
        re_ms  = np.mean([r["restricted_prototype_time_ms"] for _, r in common])
        lines.append(
            f"{rp['shots']} & {n} & "
            f"{_fmt(pre_f)} & "
            f"{_fmt(re_f)} & "
            f"\\textbf{{{_fmt(fu_f)}}} & "
            f"{re_ms:.1f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    _write_text(TABLES_DIR / "restricted_ablation.tex", "\n".join(lines) + "\n")


def _generate_hardware_table() -> None:
    device = _load_json(ROOT / "results" / "firmware" / "ondevice_batch_1shot_device_summary.json")
    dev = device["aggregate_device_mean_by_episode"]
    host = device["aggregate_host_mean_by_episode"]
    diff = device["host_device_abs_diff"]
    lines = [
        r"\begin{table}[t]",
        r"\caption{Aggregate on-device batch benchmark on the CY8CPROTO-063-BLE.",
        r"The firmware replays 18 DS2 1-shot episodes with 32 query beats per record",
        r"(576 query beats total), and reports mean per-episode macro-F1 over UART.",
        r"Host values correspond to the same capped replay subset.}",
        r"\label{tab:hardware-sanity}",
        r"\centering",
        r"\begin{tabular}{l ccc}",
        r"\toprule",
        r"Method & Device F1 & Host F1 & $|$Diff$|$ \\",
        r"\midrule",
        f"Pre-adaptation & {_fmt(dev['pre_f1_macro'])} & {_fmt(host['pre_f1_macro'])} & {diff['pre_f1_macro']:.3f} \\\\",
        f"Linear SGD & {_fmt(dev['post_linear_sgd_f1_macro'])} & {_fmt(host['post_linear_sgd_f1_macro'])} & {diff['post_linear_sgd_f1_macro']:.3f} \\\\",
        f"Prototype & \\textbf{{{_fmt(dev['post_prototype_f1_macro'])}}} & \\textbf{{{_fmt(host['post_prototype_f1_macro'])}}} & {diff['post_prototype_f1_macro']:.3f} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    _write_text(TABLES_DIR / "hardware_sanity.tex", "\n".join(lines) + "\n")


def _generate_compute_cost_table(cost: dict) -> None:
    macs = cost["macs"]
    cli  = cost["clinical"]
    measured = cost.get("measured_dwt_release")

    total_params = 8*1*5+8 + 16*8*5+16 + 32*16+32 + 2*32+2

    if measured:
        lines = [
            r"\begin{table}[t]",
            r"\caption{Model complexity and measured on-device timing. DWT cycle counts were collected from the CM4 \emph{Release} build on the CY8CPROTO-063-BLE with a validated HF0 clock of 100\,MHz. Adaptation times are end-to-end and include support embedding extraction plus head update.}",
            r"\label{tab:compute-cost}",
            r"\centering",
            r"\begin{tabular}{l rr}",
            r"\toprule",
            r"Item & Cycles / MACs & Time / Params \\",
            r"\midrule",
            r"\multicolumn{3}{l}{\textit{Model complexity}} \\",
            f"Conv1 (1$\\to$8, $k$=5, $L$=200) & {macs['conv1']:,} MACs & {8*1*5+8} params \\\\",
            f"Conv2 (8$\\to$16, $k$=5, $L$=100) & {macs['conv2']:,} MACs & {16*8*5+16} params \\\\",
            f"Projection (16$\\to$32) & {macs['proj']:,} MACs & {32*16+32} params \\\\",
            f"Linear head (32$\\to$2) & {macs['head']:,} MACs & {2*32+2} params \\\\",
            f"\\textbf{{Total}} & \\textbf{{{macs['full_inference']:,} MACs}} & \\textbf{{{total_params} params}} \\\\",
            r"\midrule",
            r"\multicolumn{3}{l}{\textit{Measured CM4 timing at 100\,MHz}} \\",
            f"Inference / beat & {measured['inference']['cycles_mean']:,} cyc & {measured['inference']['time_ms']:.2f} ms \\\\",
            f"Prototype adapt ($K=1$) & {measured['adapt_end_to_end']['1']['proto_cycles']:,} cyc & {measured['adapt_end_to_end']['1']['proto_ms']:.2f} ms \\\\",
            f"Prototype adapt ($K=5$) & {measured['adapt_end_to_end']['5']['proto_cycles']:,} cyc & {measured['adapt_end_to_end']['5']['proto_ms']:.2f} ms \\\\",
            f"Prototype adapt ($K=10$) & {measured['adapt_end_to_end']['10']['proto_cycles']:,} cyc & {measured['adapt_end_to_end']['10']['proto_ms']:.2f} ms \\\\",
            f"Linear SGD adapt ($K=1$) & {measured['adapt_end_to_end']['1']['sgd_cycles']:,} cyc & {measured['adapt_end_to_end']['1']['sgd_ms']:.2f} ms \\\\",
            f"Linear SGD adapt ($K=5$) & {measured['adapt_end_to_end']['5']['sgd_cycles']:,} cyc & {measured['adapt_end_to_end']['5']['sgd_ms']:.2f} ms \\\\",
            f"Linear SGD adapt ($K=10$) & {measured['adapt_end_to_end']['10']['sgd_cycles']:,} cyc & {measured['adapt_end_to_end']['10']['sgd_ms']:.2f} ms \\\\",
            r"\midrule",
            r"\multicolumn{3}{l}{\textit{Clinical reference}} \\",
            f"Beat interval at 50\\,BPM & -- & {cli['beat_interval_bpm50_ms']:.0f} ms \\\\",
            f"Beat interval at 150\\,BPM & -- & {cli['beat_interval_bpm150_ms']:.0f} ms \\\\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    else:
        mcu  = cost["mcu_cm4_150mhz_estimate"]
        cpu  = cost["host_cpu"]
        lines = [
            r"\begin{table}[t]",
            r"\caption{Model complexity and per-beat inference cost.",
            r"MCU latency estimated from MAC count at 75\,MMAC/s on CM4 @ 150\,MHz.",
            r"Host CPU proxy includes Python/PyTorch overhead.}",
            r"\label{tab:compute-cost}",
            r"\centering",
            r"\begin{tabular}{l rr}",
            r"\toprule",
            r"Layer & MACs & Parameters \\",
            r"\midrule",
            r"\multicolumn{3}{l}{\textit{Backbone}} \\",
            f"Conv1 (1$\\to$8,  $k$=5, $L$=200) & {macs['conv1']:,} & {8*1*5+8} \\\\",
            f"Conv2 (8$\\to$16, $k$=5, $L$=100) & {macs['conv2']:,} & {16*8*5+16} \\\\",
            f"Projection (16$\\to$32) & {macs['proj']:,} & {32*16+32} \\\\",
            r"\midrule",
            r"\multicolumn{3}{l}{\textit{Head}} \\",
            f"Linear (32$\\to$2) & {macs['head']:,} & {2*32+2} \\\\",
            r"\midrule",
            f"\\textbf{{Total}} & \\textbf{{{macs['full_inference']:,}}} & \\textbf{{{total_params}}} \\\\",
            r"\midrule",
            r"\multicolumn{3}{l}{\textit{Per-beat inference}} \\",
            f"MCU CM4 estimate & \\multicolumn{{2}}{{r}}{{{mcu['inference_per_beat_ms']:.2f}\\,ms}} \\\\",
            f"Host CPU proxy & \\multicolumn{{2}}{{r}}{{{cpu['inference_per_beat_ms']:.2f}\\,ms}} \\\\",
            r"\midrule",
            r"\multicolumn{3}{l}{\textit{Clinical beat interval}} \\",
            f"At 50\\,BPM  & \\multicolumn{{2}}{{r}}{{{cli['beat_interval_bpm50_ms']:.0f}\\,ms}} \\\\",
            f"At 150\\,BPM & \\multicolumn{{2}}{{r}}{{{cli['beat_interval_bpm150_ms']:.0f}\\,ms}} \\\\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    _write_text(TABLES_DIR / "compute_cost.tex", "\n".join(lines) + "\n")


def _generate_scaling_study_table(sweep: dict, retune: dict) -> None:
    retune_map = {variant["name"]: variant for variant in retune["variants"]}
    lines = [
        r"\begin{table*}[t]",
        r"\caption{Scaling study across backbone sizes. DS2 test macro-F1 is reported before any personalization.",
        r"Few-shot columns use fixed settings across backbones (SGD 0.05/50; raw Euclidean prototypes) and identify the higher-scoring method.}",
        r"\label{tab:scaling-study}",
        r"\centering",
        r"\begin{tabular}{l r r c c c c}",
        r"\toprule",
        r"Variant & Params & MACs / beat & DS2 test F1 & Best 1-shot & Best 5-shot & Best 10-shot \\",
        r"\midrule",
    ]
    for variant in sweep["variants"]:
        best_cells = []
        for shot in [1, 5, 10]:
            shot_key = f"shot_{shot}"
            if variant["name"] in retune_map:
                shot_data = retune_map[variant["name"]]["shots"][shot_key]
                best_sgd = next(row for row in shot_data["sgd_grid"] if row["name"] == "sgd_lr0p05_s50")
                best_proto = next(row for row in shot_data["prototype_grid"] if row["name"] == "proto_euclidean_raw")
                if best_proto["f1_macro"] > best_sgd["f1_macro"]:
                    best_cells.append(f"Proto {_fmt(best_proto['f1_macro'])}")
                else:
                    best_cells.append(f"SGD {_fmt(best_sgd['f1_macro'])}")
            else:
                agg = variant["few_shot"][shot_key]["aggregate"]
                if agg["post_prototypes"]["f1_macro"] > agg["post_linear"]["f1_macro"]:
                    best_cells.append(f"Proto {_fmt(agg['post_prototypes']['f1_macro'])}")
                else:
                    best_cells.append(f"SGD {_fmt(agg['post_linear']['f1_macro'])}")
        lines.append(
            f"{variant['label']} & "
            f"{variant['parameter_count']:,} & "
            f"{variant['macs_per_beat']:,} & "
            f"{_fmt(variant['test_ds2']['f1_macro'])} & "
            f"{best_cells[0]} & "
            f"{best_cells[1]} & "
            f"{best_cells[2]} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    _write_text(TABLES_DIR / "scaling_study.tex", "\n".join(lines) + "\n")


def _generate_scaling_cost_table(cost: dict, medium_profile: dict) -> None:
    tiny = cost["measured_dwt_release"]
    medium = medium_profile["profiling"]["decoded"]
    lines = [
        r"\begin{table}[t]",
        r"\caption{Measured on-device scaling from the baseline backbone to the medium backbone.",
        r"Both were profiled with DWT cycle counts from the CM4 \emph{Release} build at a validated 100\,MHz HF0 clock.}",
        r"\label{tab:scaling-cost}",
        r"\centering",
        r"\begin{tabular}{l r r c c}",
        r"\toprule",
        r"Model & Params & MACs & Infer. (ms) & K{=}10 P / S (ms) \\",
        r"\midrule",
        f"Tiny & 1,314 & 72,576 & {tiny['inference']['time_ms']:.1f} & {tiny['adapt_end_to_end']['10']['proto_ms']:.0f} / {tiny['adapt_end_to_end']['10']['sgd_ms']:.0f} \\\\",
        f"Medium & 2,618 & 157,040 & {medium['inference']['time_ms']:.1f} & {medium['adapt_end_to_end']['10']['proto_ms']:.0f} / {medium['adapt_end_to_end']['10']['sgd_ms']:.0f} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    _write_text(TABLES_DIR / "scaling_cost.tex", "\n".join(lines) + "\n")


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> None:
    train_metrics = _load_json(RESULTS_DIR / "train_metrics.json")
    protocols = [
        _load_json(RESULTS_DIR / "protocol_1shot.json"),
        _load_json(RESULTS_DIR / "protocol_5shot.json"),
        _load_json(RESULTS_DIR / "protocol_10shot.json"),
    ]
    restr_protocols = [
        _load_json(RESULTS_DIR / "protocol_restricted_normal_1shot.json"),
        _load_json(RESULTS_DIR / "protocol_restricted_normal_5shot.json"),
        _load_json(RESULTS_DIR / "protocol_restricted_normal_10shot.json"),
    ]
    cost = _load_json(RESULTS_DIR / "compute_cost.json")
    sweep = _load_json(ROOT / "results" / "architecture_sweep" / "summary.json")
    retune = _load_json(ROOT / "results" / "head_retune" / "summary.json")
    medium_profile = _load_json(ROOT / "results" / "firmware" / "medium_profile.json")

    # Tables
    _generate_offline_table(train_metrics, protocols)
    _generate_gain_table(protocols)
    _generate_restricted_ablation_table(protocols, restr_protocols)
    _generate_hardware_table()
    _generate_compute_cost_table(cost)
    _generate_scaling_study_table(sweep, retune)
    _generate_scaling_cost_table(cost, medium_profile)

    # Figures
    _generate_architecture_figure()
    _generate_system_overview()
    _generate_prototype_figure()
    _generate_embedded_platform_figure()
    _generate_ecg_examples()
    _generate_results_plot(protocols)
    _generate_restricted_ablation_plot(protocols, restr_protocols)
    _generate_compute_cost_figure(cost, protocols)

    print("All assets generated.")


if __name__ == "__main__":
    main()
