"""Redraw the manuscript figures for the explicit-node workspace version.

Outputs (vector PDF, SVG and PNG):
  fig03_diagnosis_ablation.png  - layer read-out, read-out intervention, design ablation
  fig04_formal_timing.png       - formal accuracy vs controls, CPU timing
  fig05_belief_revision.png     - Belief-R: neural / rule / LLM baselines
  fig06_procurement.png         - procurement paired-instruction choices

Truth sources:
  runs/arithmetic_workspace/{diagnosis,probe,development,verified,controls,edit_timing}.json
  runs/neural_baselines/belief_summary.json
  runs/arithmetic_workspace/procurement_formal_grounded.jsonl
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
EXPORTS = ROOT / "figures" / "exports"
EXPORTS.mkdir(parents=True, exist_ok=True)

BLUE = "#2A78D6"
ORANGE = "#EB6834"
GREEN = "#1BAF7A"
GRAY = "#8A93A6"
LGRAY = "#C9D1D9"
INK = "#0B0B0B"
MUTED = "#52514E"
GRID = "#E0DED9"
WHITE = "#FFFFFF"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 9,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.edgecolor": "#B8C0C8",
    "figure.facecolor": WHITE,
    "axes.facecolor": WHITE,
    "savefig.facecolor": WHITE,
    "pdf.fonttype": 42,
})


def style_ax(ax, ygrid=True):
    ax.spines[["top", "right"]].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#B8C0C8")
        ax.spines[s].set_linewidth(0.8)
    if ygrid:
        ax.yaxis.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=3, width=0.8)


def save(fig, name):
    fig.savefig(EXPORTS / f"{name}.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    fig.savefig(EXPORTS / f"{name}.png", dpi=600, bbox_inches="tight")
    fig.savefig(EXPORTS / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def fig03():
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.9), dpi=300,
                             gridspec_kw={"width_ratios": [1.0, 1.0, 1.45]})

    # (a) layer-by-layer read-out
    ax = axes[0]
    layers = ["input", "layer 1", "layer 2", "root"]
    vals = [100.0, 1033 / 1200 * 100, 226 / 600 * 100, 41 / 300 * 100]
    bars = ax.bar(range(4), vals, 0.55, color=[GREEN, BLUE, BLUE, ORANGE], zorder=3)
    for r, v, n in zip(bars, vals, [1500, 1200, 600, 300]):
        ax.text(r.get_x() + r.get_width() / 2, v + 1.5,
                f"{v:.1f}\n({n})", ha="center", va="bottom", fontsize=7.4)
    ax.set_xticks(range(4)); ax.set_xticklabels(layers, fontsize=8)
    ax.set_ylim(0, 118); ax.set_ylabel("read-out accuracy (%)", fontsize=8.8)
    ax.set_title("(a) Continuous-state read-out", fontsize=9.8, fontweight="bold", pad=6)
    style_ax(ax)

    # (b) read-out intervention
    ax = axes[1]
    widths = ["2", "4", "8"]
    orig = [59.7, 24.3, 16.3]
    ridge = [80.3, 80.3, 42.7]
    x = list(range(3)); w = 0.34
    ax.bar([i - w / 2 for i in x], orig, w, color=GRAY, label="original read-out", zorder=3)
    ax.bar([i + w / 2 for i in x], ridge, w, color=BLUE,
           label="ridge read-out", zorder=3)
    for i, (a, b) in enumerate(zip(orig, ridge)):
        ax.text(i - w / 2, a + 1.5, f"{a:.1f}", ha="center", va="bottom", fontsize=7.2)
        ax.text(i + w / 2, b + 1.5, f"{b:.1f}", ha="center", va="bottom",
                fontsize=7.2, color=BLUE, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(widths, fontsize=8)
    ax.set_xlabel("branches", fontsize=8.5)
    ax.set_ylim(0, 100); ax.set_ylabel("dev accuracy (%)", fontsize=8.8)
    ax.set_title("(b) Read-out-only fix", fontsize=9.8, fontweight="bold", pad=6)
    ax.legend(loc="upper right", frameon=False, fontsize=7.2)
    style_ax(ax)

    # (c) design ablation
    ax = axes[2]
    settings = ["names,\nlinear", "canonical,\nlinear", "names,\nproduct",
                "canonical,\nproduct"]
    add = [720 / 960, 960 / 960, 720 / 960, 960 / 960]
    mixed = [225 / 960, 244 / 960, 720 / 960, 960 / 960]
    two = [259 / 800, 465 / 800, 400 / 800, 800 / 800]
    x = list(range(4)); w = 0.25
    ax.bar([i - w for i in x], [v * 100 for v in add], w, color=BLUE,
           label="single-digit add", zorder=3)
    ax.bar([i for i in x], [v * 100 for v in mixed], w, color=ORANGE,
           label="mixed operations", zorder=3)
    ax.bar([i + w for i in x], [v * 100 for v in two], w, color=GREEN,
           label="two-digit", zorder=3)
    for i, v in enumerate(add):
        ax.text(i - w, v * 100 + 1.5, f"{v*100:.0f}", ha="center", va="bottom", fontsize=6.6)
    for i, v in enumerate(mixed):
        ax.text(i, v * 100 + 1.5, f"{v*100:.0f}", ha="center", va="bottom", fontsize=6.6)
    for i, v in enumerate(two):
        ax.text(i + w, v * 100 + 1.5, f"{v*100:.0f}", ha="center", va="bottom", fontsize=6.6)
    ax.set_xticks(x); ax.set_xticklabels(settings, fontsize=7.6)
    ax.set_ylim(0, 128); ax.set_ylabel("dev accuracy (%)", fontsize=8.8)
    ax.set_title("(c) Design ablation", fontsize=9.8, fontweight="bold", pad=6)
    ax.legend(loc="upper left", frameon=False, fontsize=7.0, ncol=1)
    style_ax(ax)

    fig.tight_layout()
    save(fig, "fig03_diagnosis_ablation")


def fig04():
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.6), dpi=300)
    axes = axes.flat

    # (a) formal accuracy
    ax = axes[0]
    ranges = ["0–9", "60–99", "100–999"]
    ours = [100.0, 100.0, 100.0]
    mlp = [1448 / 3600 * 100, 3 / 3600 * 100, 1 / 3600 * 100]
    sym = [100.0, 100.0, 100.0]
    x = list(range(3)); w = 0.26
    ax.bar([i - w for i in x], ours, w, color=BLUE, label="learned workspace", zorder=3)
    ax.bar([i for i in x], mlp, w, color=ORANGE, label="two-layer ReLU", zorder=3)
    ax.bar([i + w for i in x], sym, w, color=GREEN, label="symbolic oracle", zorder=3)
    for i, v in enumerate(ours):
        ax.text(i - w, v + 1.5, f"{v:.0f}", ha="center", va="bottom", fontsize=7.0)
    for i, v in enumerate(mlp):
        ax.text(i, v + 1.5, f"{v:.1f}", ha="center", va="bottom", fontsize=7.0)
    for i, v in enumerate(sym):
        ax.text(i + w, v + 1.5, f"{v:.0f}", ha="center", va="bottom", fontsize=7.0)
    ax.set_xticks(x); ax.set_xticklabels(ranges, fontsize=8.5)
    ax.set_xlabel("input value range", fontsize=8.5)
    ax.set_ylim(0, 132); ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("exact-match accuracy (%)", fontsize=8.8)
    ax.set_title("(a) 10,800 edits, 3 seeds", fontsize=9.8, fontweight="bold", pad=6)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=3,
              frameon=False, fontsize=6.8, handlelength=1.0, columnspacing=0.7)
    style_ax(ax)

    # (b) timing
    ax = axes[1]
    widths = ["5", "7", "12"]
    local = [6.15, 6.40, 7.50]
    rebuild = [9.50, 12.65, 20.45]
    speed = [1.61, 2.01, 2.74]
    x = list(range(3)); w = 0.32
    ax.bar([i - w / 2 for i in x], local, w, color=BLUE, label="local edit", zorder=3)
    ax.bar([i + w / 2 for i in x], rebuild, w, color=GRAY, label="full rebuild", zorder=3)
    for i, (a, b, s) in enumerate(zip(local, rebuild, speed)):
        ax.text(i - w / 2, a + 0.4, f"{a:.2f}", ha="center", va="bottom", fontsize=7.0)
        ax.text(i + w / 2, b + 0.4, f"{b:.2f}", ha="center", va="bottom", fontsize=7.0)
        ax.text(i, max(a, b) + 1.6, f"{s:.2f}×", ha="center", va="bottom",
                fontsize=8.0, color=BLUE, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(widths, fontsize=8.5)
    ax.set_xlabel("branches (private edit)", fontsize=8.5)
    ax.set_ylim(0, 25); ax.set_ylabel("CPU time per edit (µs)", fontsize=8.8)
    ax.set_title("(b) Local vs full rebuild", fontsize=9.8, fontweight="bold", pad=6)
    ax.legend(loc="upper left", frameon=False, fontsize=7.2)
    style_ax(ax)

    # (c) amortization
    ax = axes[2]
    n = [1, 10, 100, 1000, 5000]
    cached = [0.7483, 2.3528, 3.4878, 3.8644, 3.9131]
    cuda = [0.5584, 1.1285, 1.2646, 1.2839, 1.2861]
    ax.axhline(1.0, color=MUTED, lw=0.8, ls="--", zorder=1)
    ax.plot(n, cached, "o-", color=BLUE, lw=1.6, ms=4.5, label="cached CPU", zorder=3)
    ax.plot(n, cuda, "s-", color=ORANGE, lw=1.6, ms=4.5,
            label="uncached RTX 4060", zorder=3)
    ax.set_xscale("log"); ax.set_xticks(n)
    ax.set_xticklabels(["1", "10", "100", "1k", "5k"])
    ax.minorticks_off(); ax.set_xlim(0.8, 6500); ax.set_ylim(0, 4.4)
    ax.set_xlabel("edits", fontsize=8.5)
    ax.set_ylabel("cumulative speedup (×)", fontsize=8.5)
    ax.set_title("(c) Amortization", fontsize=9.8, fontweight="bold", pad=6)
    ax.legend(loc="lower right", frameon=False, fontsize=7.2)
    style_ax(ax)

    # (d) width scaling
    ax = axes[3]
    branches = [5, 7, 12, 20, 50, 100]
    speed = [1.61, 2.01, 2.74, 3.7371, 6.1341, 8.1889]
    err = [[0, 0, 0, 3.7371 - 3.7032, 6.1341 - 6.1035, 8.1889 - 8.1390],
           [0, 0, 0, 3.7716 - 3.7371, 6.1660 - 6.1341, 8.2337 - 8.1889]]
    ax.errorbar(branches, speed, yerr=err, fmt="o-", color=GREEN, lw=1.6,
                ms=4.5, capsize=2.5, ecolor=INK, elinewidth=0.8, zorder=3)
    for b, s in zip(branches, speed):
        ax.text(b, s + 0.35, f"{s:.2f}×", ha="center", va="bottom",
                fontsize=7.0, color=GREEN)
    ax.set_xscale("log"); ax.set_xticks(branches)
    ax.set_xticklabels([str(b) for b in branches])
    ax.minorticks_off(); ax.set_xlim(4, 120); ax.set_ylim(0, 9.8)
    ax.set_xlabel("branches (private edit)", fontsize=8.5)
    ax.set_ylabel("speedup (×)", fontsize=8.5)
    ax.set_title("(d) Width scaling", fontsize=9.8, fontweight="bold", pad=6)
    ax.text(5.2, 7.6, "recomputed share 14.3% → 4.0%", fontsize=7.0, color=MUTED)
    style_ax(ax)

    fig.tight_layout()
    save(fig, "fig04_formal_timing")


def _binomial_ci(k, n, seed):
    rng = np.random.default_rng(seed)
    draws = rng.binomial(n, k / n, 4000) / n
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _breu_ci(bu, bu_n, bm, bm_n, seed):
    rng = np.random.default_rng(seed)
    bu_d = rng.binomial(bu_n, bu / bu_n, 4000) / bu_n
    bm_d = rng.binomial(bm_n, bm / bm_n, 4000) / bm_n
    draws = (bu_d + bm_d) / 2
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def fig05():
    # (label, colour, init k, init n, BU k, BU n, BM k, BM n, seed)
    methods = [
        ("Neural", BLUE, 343, 641, 110, 509, 67, 132, 101),
        ("Copy", LGRAY, 343, 641, 0, 509, 83, 132, 102),
        ("Rule", GREEN, 586, 641, 45, 509, 122, 132, 103),
        ("Fusion", ORANGE, 586, 641, 141, 509, 99, 132, 104),
        ("Qwen", GRAY, 374, 641, 2, 509, 53, 132, 105),
        ("Llama", "#9AA5B1", 84, 641, 5, 509, 3, 132, 106),
    ]
    names = [m[0] for m in methods]
    panels = [("init", "(a) initial inference", "%", 105),
              ("bu", "(b) belief update (BU)", "%", 35),
              ("bm", "(c) belief maintenance (BM)", "%", 105),
              ("breu", "(d) BREU", "BREU", 0.62)]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), dpi=300)
    axes = axes.flat
    for ax, (key, title, ylabel, top) in zip(axes, panels):
        vals, errs = [], []
        for _, _, ik, inn, buk, bun, bmk, bmn, seed in methods:
            if key == "init":
                val = ik / inn * 100
                lo, hi = _binomial_ci(ik, inn, seed)
                val_lo, val_hi = lo * 100, hi * 100
            elif key == "bu":
                val = buk / bun * 100
                lo, hi = _binomial_ci(buk, bun, seed)
                val_lo, val_hi = lo * 100, hi * 100
            elif key == "bm":
                val = bmk / bmn * 100
                lo, hi = _binomial_ci(bmk, bmn, seed)
                val_lo, val_hi = lo * 100, hi * 100
            else:
                val = (buk / bun + bmk / bmn) / 2
                val_lo, val_hi = _breu_ci(buk, bun, bmk, bmn, seed)
            vals.append(val)
            errs.append([max(0.0, val - val_lo), max(0.0, val_hi - val)])
        colors = [m[1] for m in methods]
        bars = ax.bar(range(len(methods)), vals, 0.62, color=colors, zorder=3)
        ax.errorbar(range(len(methods)), vals, yerr=np.array(errs).T,
                    fmt="none", ecolor=INK, elinewidth=0.9,
                    capsize=2.5, zorder=4)
        for r, v, e in zip(bars, vals, errs):
            txt = f"{v:.3f}" if key == "breu" else f"{v:.1f}"
            ax.text(r.get_x() + r.get_width() / 2, v + e[1] + top * 0.02,
                    txt, ha="center", va="bottom", fontsize=7.0)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(names, fontsize=8.2)
        ax.set_ylim(0, top)
        ax.set_ylabel(ylabel, fontsize=8.6)
        ax.set_title(title, fontsize=9.2, fontweight="bold", pad=5)
        style_ax(ax)
    fig.tight_layout()
    save(fig, "fig05_belief_revision")


def fig06():
    types = ["supplier\nname", "quote\norder", "delivery\ncomparison"]
    q_single = [164 / 200 * 100, 131 / 200 * 100, 101 / 200 * 100]
    q_pair = [65.0, 33.0, 1.0]
    p_single = [100.0, 100.0, 100.0]
    p_pair = [100.0, 100.0, 100.0]
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.0), dpi=300, sharey=True)
    for ax, (q, p, title) in zip(axes, [
        (q_single, p_single, "(a) Single instruction"),
        (q_pair, p_pair, "(b) Paired decisions"),
    ]):
        x = list(range(3)); w = 0.34
        ax.bar([i - w / 2 for i in x], q, w, color=GRAY, label="Qwen3-1.7B", zorder=3)
        ax.bar([i + w / 2 for i in x], p, w, color=BLUE, label="structured parser", zorder=3)
        for i, (a, b) in enumerate(zip(q, p)):
            ax.text(i - w / 2, a + 1.5, f"{a:.0f}", ha="center", va="bottom", fontsize=7.4)
            ax.text(i + w / 2, b + 1.5, f"{b:.0f}", ha="center", va="bottom",
                    fontsize=7.4, color=BLUE, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(types, fontsize=8.2)
        ax.set_ylim(0, 115); ax.set_ylabel("accuracy (%)", fontsize=8.8)
        ax.set_title(title, fontsize=9.8, fontweight="bold", pad=6)
        style_ax(ax)
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.05, -0.18), ncol=2,
                  frameon=False, fontsize=7.8)
    fig.tight_layout()
    save(fig, "fig06_procurement")


if __name__ == "__main__":
    fig03(); fig04(); fig05(); fig06()
    print("saved", [p.name for p in sorted(EXPORTS.glob("fig0*_*.png"))])
