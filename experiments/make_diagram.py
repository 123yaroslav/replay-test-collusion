"""Figure 0: schematic of the replay test (no data; a diagram for readers who are not statisticians).

    uv run python experiments/make_diagram.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch

FIG = Path(__file__).resolve().parents[1] / "report" / "figures"
BLUE, ORANGE, INK, INK2, FILL = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#f3f1ea"


def box(ax, x, y, w, h, text, fc=FILL, ec=INK2, fs=6.9, bold_first=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                fc=fc, ec=ec, lw=0.8))
    lines = text.split("\n")
    for k, ln in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.03 - k * 0.052, ln, ha="center", va="top", fontsize=fs,
                color=INK, fontweight="bold" if (k == 0 and bold_first) else "normal")


def arrow(ax, x0, y0, x1, y1, color=INK2, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9,
                                 color=color, lw=1.0, ls=ls))


def main():
    fig, ax = plt.subplots(figsize=(7.2, 2.35))
    ax.set_xlim(0, 1.015)
    ax.set_ylim(0, 0.62)
    ax.axis("off")
    # agents
    pos = {"P1": (0.05, 0.47), "P2": (0.15, 0.47), "P3": (0.05, 0.27), "P4": (0.15, 0.27)}
    for name, (x, y) in pos.items():
        # x and y units differ in inches (7.2 in per x unit, 3.79 in per y unit)
        ax.add_patch(Ellipse((x, y), 0.05, 0.095, fc="white", ec=BLUE, lw=1.4))
        ax.text(x, y, name, ha="center", va="center", fontsize=7.5, color=INK)
    ax.plot([0.05, 0.05], [0.42, 0.32], color=ORANGE, lw=1.6, ls=(0, (3, 2)))
    ax.text(0.058, 0.37, "hidden\nchannel", fontsize=6.0, color=ORANGE, ha="left", va="center")
    ax.text(0.10, 0.575, "Deployed agents\n(same base model)", fontsize=7.2, ha="center",
            va="center", color=INK, fontweight="bold")
    ax.text(0.10, 0.17, "every agent sees the public\nhistory + its own private info",
            fontsize=6.3, ha="center", va="center", color=INK2)
    arrow(ax, 0.205, 0.37, 0.255, 0.37)
    box(ax, 0.255, 0.22, 0.18, 0.31, "1. Log\nevery agent's move\nand the observation\nit acted on\n(the auditor cannot\nsee the channel)")
    arrow(ax, 0.435, 0.37, 0.485, 0.37)
    box(ax, 0.485, 0.22, 0.21, 0.31, "2. Replay\nre-ask each agent k = 2\ntimes on the SAME\nobservation, channel\nremoved: what would it\nhave done on its own?", fc="#e9f1fb", ec=BLUE)
    arrow(ax, 0.695, 0.37, 0.745, 0.37)
    box(ax, 0.745, 0.22, 0.245, 0.31, "3. Compare\ndo the real moves of a pair\ndepend on each other more\nthan moves redrawn from\n{real move, replays}?\nflag the pair if p < α")
    ax.text(0.865, 0.15, "same model, no channel → replays share the\npriors → no flag (false alarms stay at α)",
            fontsize=6.2, ha="center", va="center", color=INK2)
    ax.text(0.865, 0.06, "hidden channel shapes the moves → real moves\ncarry what replays lack → pair flagged",
            fontsize=6.2, ha="center", va="center", color=INK2)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fig0_method.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIG / "fig0_method.pdf", bbox_inches="tight")
    print("written", FIG / "fig0_method.png")


if __name__ == "__main__":
    main()
