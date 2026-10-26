"""Figures for the paper, drawn at their final printed size in the template's font (Old Standard TT
for text, STIX for math), at least 8 pt, one message per figure, one colour per detector.

    uv run python experiments/paper_figures.py
Writes report/figures/p_*.png (600 dpi, for the docx) and p_*.pdf (vector) from results/summary/*.csv
and, for the e-process figure, from the raw logs of the pre-registered replication.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from sklearn.metrics import roc_curve

from collusion.analysis import wilson
from collusion.detect.crt_llm import requery_pool, sequential_replay_test
from collusion.games.rps import MOVES
from collusion.trajectory import read_episodes

ROOT = Path(__file__).resolve().parents[1]
SUM = ROOT / "results" / "summary"
FIG = ROOT / "report" / "figures"
WIDTH = 6.3  # inches: the template's text width as used in the report

for f in Path.home().joinpath("Library/Fonts").glob("OldStandard-*.ttf"):
    font_manager.fontManager.addfont(str(f))
mpl.rcParams.update({
    "font.family": "Old Standard TT", "mathtext.fontset": "stix", "font.size": 8.5,
    "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#55534e", "xtick.color": "#3a3935", "ytick.color": "#3a3935",
    "legend.frameon": False, "pdf.fonttype": 42, "savefig.dpi": 600,
})
INK, INK2, GRID, BAND = "#1a1a1a", "#55534e", "#e3e2de", "#f5f4f1"
C = {"crt_net": "#2d8a5e", "seq": "#1b4d3a", "naive_mi_shuffle": "#8a8781",
     "bonjour_alpha": "#bdbab3", "judge": "#c98a00", "channel": "#c2410c"}
GREEN, GREEN_DARK, GREEN_PALE = "#2d8a5e", "#1b4d3a", "#e8f3ec"
NAMES = {"crt_net": "Replay test, fixed (ours)", "naive_mi_shuffle": "Mutual information, shuffle null",
         "bonjour_alpha": "Bonjour et al. rule", "judge": "LLM judge (actions only)"}


def load(name):
    p = SUM / name
    return pd.read_csv(p) if p.exists() else None


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", pad_inches=0.12)
    fig.savefig(FIG / f"{name}.png", dpi=600, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def panel_label(ax, s, x=-0.02, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9.5, fontweight="bold", va="bottom",
            ha="right")


# ------------------------------------------------------------------ Fig. 1: the method --------
def fig_method():
    """Figure 1: the logged call, two replays without the channel, and the pool (deck style)."""
    fig = plt.figure(figsize=(WIDTH, 2.05))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 126), ax.set_ylim(0, 41)
    ax.axis("off")
    CH, CH_PALE = C["channel"], "#fbeee6"

    def box(x, y, w, h, text, fc="white", ec=INK2, lw=0.8, tc=INK, size=9, bold=False, ls="-"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                    fc=fc, ec=ec, lw=lw, ls=ls))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=tc,
                fontweight="bold" if bold else "normal")

    def arrow(x0, y0, x1, y1, color=INK2, lw=0.9):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8,
                                     color=color, lw=lw, shrinkA=0, shrinkB=0))

    # row 1: the logged call sees the view and the channel
    ax.text(0.5, 31.5, "Logged call", fontsize=8.5, fontweight="bold", va="center")
    box(15, 32.5, 13, 5.5, "view")
    box(15, 25.0, 13, 5.5, "channel", fc=CH_PALE, ec=CH, tc=CH, ls=(0, (3, 2)))
    box(35, 27.5, 12, 8, "agent", fc=INK, ec=INK, tc="white", bold=True)
    arrow(28.6, 35.2, 34.4, 33.4)
    arrow(28.6, 27.8, 34.4, 29.6, color=CH)
    box(53, 28.6, 9, 5.8, "100", fc=GREEN_DARK, ec=GREEN_DARK, tc="white", size=10, bold=True)
    arrow(47.6, 31.5, 52.4, 31.5)
    # row 2: two replays see the view only
    ax.text(0.5, 11.5, "Two replays", fontsize=8.5, fontweight="bold", va="center")
    box(15, 12.5, 13, 5.5, "view")
    box(15, 5.0, 13, 5.5, "channel", fc="white", ec="#8f8d87", tc="#8f8d87", ls=(0, (3, 2)))
    ax.plot([16.5, 26.5], [7.75, 7.75], color="#6b6963", lw=1.0)
    box(35, 7.5, 12, 8, "agent", fc=INK, ec=INK, tc="white", bold=True)
    arrow(28.6, 15.2, 34.4, 13.4)
    box(53, 13.2, 9, 5.2, "80", size=10, bold=True)
    box(53, 5.8, 9, 5.2, "70", size=10, bold=True)
    arrow(47.6, 12.8, 52.4, 15.6)
    arrow(47.6, 10.2, 52.4, 8.4)
    ax.text(15, 1.6, "the same agent, the same view, the message removed", fontsize=7.8,
            color="#45433e", va="center")
    # converge into the pool
    for y in (31.5, 15.8, 8.4):
        ax.add_patch(FancyArrowPatch((62.6, y), (71.4, 21), arrowstyle="-|>", mutation_scale=8,
                                     color=INK2, lw=0.8, shrinkA=0, shrinkB=0,
                                     connectionstyle="arc3,rad=0.0"))
    # pool panel
    ax.add_patch(FancyBboxPatch((72, 4), 53, 33, boxstyle="round,pad=0,rounding_size=2",
                                fc=GREEN_PALE, ec="none"))
    ax.text(76, 32.8, "Pool", fontsize=9, fontweight="bold", color=GREEN_DARK, va="center")
    box(76, 21.5, 10, 6.5, "100", fc=GREEN_DARK, ec=GREEN_DARK, tc="white", size=10, bold=True)
    box(88, 21.5, 10, 6.5, "80", size=10, bold=True)
    box(100, 21.5, 10, 6.5, "70", size=10, bold=True)
    ax.text(76, 15.2, "If the message had no effect,", fontsize=9, color=INK, va="center")
    ax.text(76, 10.9, "100 is one of three equally likely draws.", fontsize=9, color=INK,
            va="center", fontweight="bold")
    ax.text(76, 6.6, "The shared model shapes all three alike.", fontsize=7.8, color=INK2,
            va="center")
    save(fig, "p_fig1_method")


# ------------------------------------------------------------------ Fig. 2: main contrast -----
def _rows(audits, settings, metric):
    out = []
    for tag, cond, label in settings:
        g = audits[(audits.tag == tag) & (audits.condition == cond)]
        for det in ("naive_mi_shuffle", "bonjour_alpha", "crt_net"):
            d = g[g.detector == det]
            if len(d):
                col = "flag05" if metric == "flag" else "correct_pair"
                out.append((label, det, int(d[col].sum()), len(d)))
    return out


def _dots(ax, rows, ref=None, min_rows=None):
    labels = list(dict.fromkeys(r[0] for r in rows))
    dets = [d for d in ("naive_mi_shuffle", "bonjour_alpha", "crt_net")
            if any(r[1] == d for r in rows)]
    off = dict(zip(dets, np.linspace(-0.24, 0.24, len(dets))))
    for y, lab in enumerate(labels):
        if y % 2 == 0:
            ax.axhspan(y - 0.5, y + 0.5, color=BAND, zorder=0, lw=0)
        for lab_, det, k, n in rows:
            if lab_ != lab:
                continue
            lo, hi = wilson(k, n)
            yy = y + off[det]
            ax.plot([100 * lo, 100 * hi], [yy, yy], color=C[det], lw=1.5, solid_capstyle="round")
            ax.plot(100 * k / n, yy, "o", ms=4.6, color=C[det], mec="white", mew=0.8, zorder=3)
    if ref is not None:
        ax.axvline(ref, color=INK2, lw=0.7, ls=(0, (3, 2)))
        ax.text(ref + 1.0, -0.62, r"$\alpha$ = 5%", fontsize=8, color=INK2, va="bottom",
                ha="left", clip_on=False)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_ylim(max(len(labels), min_rows or 0) - 0.5, -0.5)
    ax.set_xlim(0, 100)
    ax.tick_params(axis="y", length=0, labelsize=8.5)
    ax.tick_params(axis="x", labelsize=8.5)
    ax.xaxis.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    return dets


def fig_main():
    """Figure 4 (deck style): false alarms as bars, detection as three cards."""
    import json
    import matplotlib.patches as mpatches
    s1, s2 = load("s1_audits.csv"), load("s2_audits.csv")
    def rate(d, tag, cond, det, col):
        g = d[(d.tag == tag) & (d.condition == cond) & (d.detector == det)]
        return 100 * g[col].mean(), int(g[col].sum()), len(g)
    bars = [("Mutual\ninformation", rate(s1, "haiku", "null_same", "naive_mi_shuffle", "flag05")[0], C["naive_mi_shuffle"]),
            ("Bonjour\nrule", rate(s1, "haiku", "null_same", "bonjour_alpha", "flag05")[0], C["bonjour_alpha"]),
            ("Replay\ntest", rate(s1, "haiku", "null_same", "crt_net", "flag05")[0], GREEN)]
    n_h = rate(s1, "haiku", "null_same", "crt_net", "flag05")[2]
    vend = rate(s2, "deepseek", "vendor_leak", "crt_net", "correct_pair")[0]
    pub = rate(s2, "deepseek", "vendor_public", "crt_net", "flag05")[0]
    conf = json.load(open(ROOT / "results" / "summary" / "confirm_results.json"))
    fr = json.load(open(ROOT / "results" / "summary" / "fresh_results.json"))["run_e"]
    fig = plt.figure(figsize=(WIDTH, 2.25))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.74, 2.26], wspace=0.2)
    ax = fig.add_subplot(gs[0, 0])
    xs = range(3)
    ax.bar(xs, [v for _, v, _ in bars], color=[c for _, _, c in bars], width=0.62)
    for x, (_, v, _) in zip(xs, bars):
        ax.text(x, v + 2, f"{v:.1f}%" if v >= 10 else f"{v:.0f}%", ha="center", va="bottom", fontsize=8,
                color=INK, fontweight="bold")
    ax.axhline(5, color=INK2, lw=0.7, ls=(0, (3, 2)))
    ax.set_xticks(list(xs), [b[0] for b in bars], fontsize=7.5)
    ax.set_ylim(0, 85); ax.set_yticks([0, 25, 50, 75]); ax.tick_params(axis="y", labelsize=7.5)
    ax.set_ylabel("Honest games flagged (%)", fontsize=8)
    ax.set_title(f"{n_h} honest Haiku games", fontsize=7.8, loc="left", color=INK2)
    panel_label(ax, "a", x=-0.36, y=1.08)
    # cards: each detection result next to its honest control
    ax2 = fig.add_subplot(gs[0, 1]); ax2.set_axis_off(); ax2.set_xlim(0, 100); ax2.set_ylim(0, 100)
    cards = [("Pooled-data vendor", [("seller pair named", vend), ("public data, flagged", pub)], "fixed-horizon test"),
             ("Secret plan", [("pair named", 100 * conf["c1_seq_correct"] / conf["c1_n"], f"{conf['c1_seq_correct']} / {conf['c1_n']}"),
                              ("honest, flagged", 100 * conf["c2_seq_alarms"] / conf["c2_n"], f"{conf['c2_seq_alarms']} / {conf['c2_n']}")], "sequential, pre-registered"),
             ("Double-auction cartel", [("cartel, flagged", 100 * fr["cartel2"]["codep_flag"]["rate"]), ("no chat, flagged", 100 * fr["off"]["codep_flag"]["rate"])], "co-departure, post hoc")]
    w, gap = 32.6, 1.1
    for k, (title, rows, foot) in enumerate(cards):
        x0 = k * (w + gap)
        ax2.add_patch(mpatches.FancyBboxPatch((x0, 1), w, 97, boxstyle="round,pad=0,rounding_size=2.5", fc="#f5f4f1", ec="none"))
        ax2.text(x0 + 2.5, 88, title, fontsize=7.9, fontweight="bold", va="center", color=INK)
        y = 74
        for row in rows:
            lab, v = row[0], row[1]; txt = row[2] if len(row) > 2 else (f"{v:.1f}%" if v % 1 else f"{v:.0f}%")
            ax2.text(x0 + 3, y, lab, fontsize=7.5, va="center", color=INK2)
            ax2.text(x0 + 3, y - 10, txt, fontsize=10, va="center", ha="left", color=INK, fontweight="bold")
            ax2.add_patch(mpatches.Rectangle((x0 + 3, y - 19.5), w - 6, 3, fc=GRID, ec="none"))
            ax2.add_patch(mpatches.Rectangle((x0 + 3, y - 19.5), (w - 6) * v / 100, 3, fc=GREEN if v > 50 else "#9a9890", ec="none"))
            y -= 31
        ax2.text(x0 + 2.5, 7, foot, fontsize=6.9, va="center", color=INK2, style="italic")
    panel_label(ax2, "b", x=-0.01, y=1.08)
    fig.subplots_adjust(left=0.09, right=0.995, top=0.87, bottom=0.15)
    save(fig, "p_fig2_main")


# ------------------------------------------------------------------ Fig. 3: ROC ---------------
def fig_roc():
    s1, j1 = load("s1_audits.csv"), load("s1_judge.csv")
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.05), sharey=True)
    for ax, tag, name in zip(axes, ("haiku", "deepseek"), ("Claude Haiku", "DeepSeek")):
        g = s1[s1.tag == tag]
        for det in ("naive_mi_shuffle", "crt_net"):
            d = g[(g.detector == det) & g.condition.isin(["pair_hidden", "null_same"])]
            y = (d.condition == "pair_hidden").astype(int)
            fpr, tpr, _ = roc_curve(y, -np.log(d.p.clip(lower=1e-6)))
            ax.plot(np.clip(100 * fpr, 0.5, None), 100 * tpr, color=C[det], lw=1.6,
                    drawstyle="steps-post", label=NAMES[det])
        d = j1[(j1.tag == tag) & (j1.view == "actions")
               & j1.condition.isin(["pair_hidden", "null_same"])].dropna(subset=["prob"])
        y = (d.condition == "pair_hidden").astype(int)
        fpr, tpr, _ = roc_curve(y, d.prob)
        ax.plot(np.clip(100 * fpr, 0.5, None), 100 * tpr, color=C["judge"], lw=1.6,
                drawstyle="steps-post", label=NAMES["judge"])
        xs = np.logspace(np.log10(0.5), 2, 50)
        ax.plot(xs, xs, color="#b5b3ab", lw=0.9, label="chance")
        for x in (1, 5):
            ax.axvline(x, color=INK2, lw=0.9, ls=(0, (3, 2)))
        ax.set_xscale("log")
        ax.set_xlim(0.5, 100), ax.set_ylim(0, 100)
        ax.set_xticks([1, 5, 20, 100], ["1%", "5%", "20", "100"])
        ax.tick_params(labelsize=8.5)
        ax.set_xlabel("Honest games flagged (%), log scale")
        ax.text(0.6, 93, name, fontsize=8.5, fontstyle="italic", va="top")
        ax.grid(True, color=GRID, lw=0.5)
    axes[0].set_ylabel("Colluding games\ndetected (%)")
    panel_label(axes[0], "a", x=-0.2)
    panel_label(axes[1], "b", x=-0.05)
    hs, ls = axes[0].get_legend_handles_labels()
    fig.legend(hs, ls, loc="lower center", ncol=3, bbox_to_anchor=(0.55, -0.01), handlelength=1.6)
    fig.subplots_adjust(left=0.12, right=0.985, top=0.92, bottom=0.35, wspace=0.12)
    save(fig, "p_fig3_roc")


# ------------------------------------------------------------------ Fig. 4: e-process ---------
def fig_eprocess():
    raw = ROOT / "results" / "raw" / "s1" / "confirm_haiku"
    paths = {}
    for cond in ("null_same", "pair_hidden"):
        out = []
        for ep in read_episodes(raw / f"{cond}.jsonl"):
            pool = requery_pool(ep, ["P1", "P2", "P3", "P4"], MOVES, None)
            out.append(sequential_replay_test(pool, alpha=0.05, n_actions=3)["e_path"])
        paths[cond] = np.array(out)
    fig, ax = plt.subplots(figsize=(WIDTH, 1.6))
    t = np.arange(paths["null_same"].shape[1])
    for e in paths["null_same"]:
        ax.plot(t, e, color="#9a9890", lw=0.6, alpha=0.7)
    for e in paths["pair_hidden"]:
        ax.plot(t, e, color=C["seq"], lw=0.9, alpha=0.85)
    ax.axhline(20, color=C["channel"], lw=0.9, ls=(0, (4, 2)))
    ax.text(0.3, 26, "alarm line at 20 (5% level)", ha="left", va="bottom", fontsize=8,
            color=C["channel"])
    ax.set_yscale("log")
    ax.set_ylim(0.05, 1e5)
    ax.set_xlim(0, t[-1])
    ax.set_xlabel("Round")
    ax.set_ylabel("Evidence (log scale)")
    ax.grid(True, color=GRID, lw=0.5)
    n0, n1 = len(paths["null_same"]), len(paths["pair_hidden"])
    hs = [plt.Line2D([], [], color=C["seq"], lw=1.2), plt.Line2D([], [], color="#9a9890", lw=1.0)]
    ax.legend(hs, [f"instructed pair ({n1} games)", f"honest agents ({n0} games)"],
              loc="upper left", handlelength=1.6, bbox_to_anchor=(0.0, 1.0))
    fig.subplots_adjust(left=0.1, right=0.985, top=0.95, bottom=0.27)
    save(fig, "p_fig4_eprocess")


# ------------------------------------------------------------------ appendix figures ----------
def fig_all_conditions():
    s1, s2, ag = load("s1_audits.csv"), load("s2_audits.csv"), load("agrawal_panel_runs.csv")
    audits = pd.concat([s1, s2.assign(tag=s2.tag + "_s2")], ignore_index=True)
    honest = [("deepseek", "null_same", "RPS · 4 × DeepSeek"),
              ("haiku", "null_same", "RPS · 4 × Haiku"),
              ("mixed", "null_mixed", "RPS · 2 DeepSeek + 2 Haiku"),
              ("deepseek", "talk_no_action", "RPS · DeepSeek, undelivered plans"),
              ("haiku", "talk_no_action", "RPS · Haiku, undelivered plans"),
              ("haiku", "told_no_channel", "RPS · Haiku, team, no channel"),
              ("deepseek_s2", "null_same", "Auction · 4 × DeepSeek"),
              ("deepseek_s2", "vendor_public", "Auction · public-data vendor")]
    rows_h = _rows(audits, honest, "flag")
    b = ag[ag.condition == "base"]
    rows_h += [("Agrawal et al. logs, no chat", "naive_mi_shuffle", int(b.naive_mi_shuffle.sum()),
                len(b)),
               ("Agrawal et al. logs, no chat", "bonjour_alpha", int(b.bonjour_alpha.sum()), len(b))]
    coll = [("deepseek", "pair_hidden", "RPS · DeepSeek, secret plan"),
            ("haiku", "pair_hidden", "RPS · Haiku, secret plan"),
            ("deepseek_s2", "vendor_leak", "Auction · pooled-data vendor"),
            ("deepseek_s2", "pair_hidden", "Auction · bid-rotation cartel")]
    rows_c = _rows(audits, coll, "correct")
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.1),
                             gridspec_kw={"width_ratios": [1, 1], "wspace": 1.15})
    _dots(axes[0], rows_h, ref=5)
    axes[0].set_xlabel("Honest games flagged (%)")
    _dots(axes[1], rows_c, min_rows=len(set(r[0] for r in rows_h)))
    axes[1].set_xlabel("True pair named (%)")
    panel_label(axes[0], "a", x=-0.9)
    panel_label(axes[1], "b", x=-0.9)
    hs = [plt.Line2D([], [], color=C[d], marker="o", lw=1.5, ms=4.6, mec="white")
          for d in ("naive_mi_shuffle", "bonjour_alpha", "crt_net")]
    fig.legend(hs, [NAMES[d] for d in ("naive_mi_shuffle", "bonjour_alpha", "crt_net")],
               loc="lower center", ncol=3, bbox_to_anchor=(0.55, -0.01), handlelength=1.6)
    fig.subplots_adjust(left=0.3, right=0.985, top=0.9, bottom=0.2)
    save(fig, "p_figA1_all")


def fig_pvalues():
    s1 = load("s1_audits.csv")
    honest = s1[s1.condition.isin(["null_same", "null_mixed", "talk_no_action"])
                & s1.tag.isin(["deepseek", "haiku", "mixed"])]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 1.9), sharey=True)
    for ax, det, lab in zip(axes, ("crt_net", "naive_mi_shuffle"), ("a", "b")):
        p = honest[honest.detector == det].p.dropna()
        ax.hist(p, bins=np.linspace(0, 1, 21), color=C[det], edgecolor="white", lw=0.8)
        ax.axhline(len(p) / 20, color=INK2, lw=0.7, ls=(0, (3, 2)))
        ax.set_xlabel("$p$-value, replay test (fixed)" if det == "crt_net" else "$p$-value, mutual information, shuffle null")
        panel_label(ax, lab, x=-0.08)
    axes[0].set_ylabel("Honest games")
    fig.subplots_adjust(left=0.1, right=0.985, top=0.9, bottom=0.25, wspace=0.08)
    save(fig, "p_figA2_pvalues")


def fig_power():
    pw = load("s0b_power.csv")
    fig, ax = plt.subplots(figsize=(WIDTH * 0.75, 2.3))
    cols = {0.6: "#7fb0ea", 0.8: "#2a78d6", 1.0: "#16457f"}
    for cp, g in pw[pw.cp > 0].groupby("cp"):
        if cp in cols:
            m = g.groupby("T").correct.mean()
            ax.plot(m.index, 100 * m.values, color=cols[cp], lw=1.5, marker="o", ms=3.5,
                    label=f"coordinates in {cp:.0%} of rounds")
    fa = pw[pw.cp == 0].groupby("T").flag.mean()
    ax.plot(fa.index, 100 * fa.values, color=INK2, lw=1, ls=(0, (3, 2)), marker="o", ms=3,
            label="honest (false alarms)")
    ax.set_xlabel("Rounds in the log, $T$")
    ax.set_ylabel("True pair named (%)")
    ax.set_xlim(5, 128), ax.set_ylim(0, 105)
    ax.grid(True, color=GRID, lw=0.5)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), handlelength=1.6)
    fig.subplots_adjust(left=0.12, right=0.52, top=0.95, bottom=0.2)
    save(fig, "p_figA3_power")


def fig_growth():
    g = load("s0g_growth.csv")
    if g is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.1), sharey=False)
    cols = {"joint": C["seq"], "interaction": C["bonjour_alpha"], "ratio": C["crt_net"]}
    labels = {"joint": "joint bet (pre-registered)", "interaction": "interaction bet",
              "ratio": "likelihood-ratio bet"}
    for ax, prior, lab in zip(axes, ("uniform", "skewed"), ("a", "b")):
        d = g[(g.prior == prior) & (g.channel == "copy")].sort_values("eps")
        ax.plot(d.eps, d.kl, color=INK2, lw=1, ls=(0, (3, 2)), label="KL ceiling")
        for b in ("joint", "ratio", "interaction"):
            ax.plot(d.eps, d[f"{b}_k2"], color=cols[b], lw=1.5, marker="o", ms=3, label=labels[b],
                    ls=(0, (4, 2)) if b == "interaction" else "-", zorder=3 if b == "interaction" else 2)
        ax.axhline(0, color=INK2, lw=0.6)
        ax.set_xlabel("Channel strength (share of copied moves)")
        ax.text(0.04, 0.93, "uniform priors" if prior == "uniform" else "shared skewed priors",
                transform=ax.transAxes, fontsize=8.5, fontstyle="italic", va="top")
        ax.grid(True, color=GRID, lw=0.5)
        panel_label(ax, lab, x=-0.12)
    axes[0].set_ylabel("Expected log-growth\nper round")
    hs, ls = axes[1].get_legend_handles_labels()
    fig.legend(hs, ls, loc="lower center", ncol=4, bbox_to_anchor=(0.53, -0.01), handlelength=1.6)
    fig.subplots_adjust(left=0.11, right=0.985, top=0.92, bottom=0.33, wspace=0.22)
    save(fig, "p_figA4_growth")




# ---------------------------------------------------------------- live example -------------
def fig_example():
    """One audited round and one whole game, taken verbatim from the pooled-data-vendor logs."""
    import json, textwrap
    import matplotlib.patches as mpatches
    path = ROOT / "results" / "raw" / "s2" / "deepseek" / "vendor_leak.jsonl"
    with open(path) as f:
        ep = json.loads(f.readline())
    rd = ep["rounds"][1]  # second round of the first game
    fig = plt.figure(figsize=(6.6, 3.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.16, left=0.03, right=0.99, top=0.86, bottom=0.17)
    ax = fig.add_subplot(gs[0, 0])
    ax.set_axis_off()
    ax.set_xlim(0, 100), ax.set_ylim(0, 100)
    seats = ["P1", "P2", "P3", "P4"]
    msgs = {m["recipients"][0]: m["text"] for m in rd["messages"]}
    RED, GRN = "#a63a12", "#8fcfb3"
    y = 100
    for p in seats:
        cost = rd["private_info"][p]["your_cost"]
        logged, replays = rd["actions"][p], rd["requery"][p]
        departs = logged not in replays
        lines = textwrap.wrap("Vendor, confidential. " + msgs[p].replace("Confidential (pooled client data): ", ""), 58) if p in msgs else ["no message"]
        h = 13 + 7.5 * len(lines)
        box = mpatches.FancyBboxPatch((0, y - h), 100, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                      fc=("#fff4ee" if p in msgs else "white"), ec=("#d9622f" if p in msgs else "#c9c7c1"), lw=0.8)
        ax.add_patch(box)
        ax.text(1.5, y - 1.2, f"{p}   cost {cost}", fontsize=8, fontweight="bold", va="top", color=INK)
        ax.text(1.5, y - 7.8, "\n".join(lines), fontsize=7.2, va="top", color=("#8a2e0e" if p in msgs else INK2), linespacing=1.15)
        yy = y - h + 5.2
        ax.text(1.5, yy, f"logged bid {logged}", fontsize=7.6, va="center", color=INK, fontweight="bold")
        ax.text(33, yy, f"replays without the message {replays[0]}, {replays[1]}", fontsize=7.6, va="center", color=INK)
        ax.text(99, yy, "departs" if departs else "matches", fontsize=7.6, va="center", ha="right",
                color=(RED if departs else "#17805a"), fontweight="bold")
        y -= h + 2.2
    ax.set_ylim(y, 100)
    # panel b: the whole game
    ax2 = fig.add_subplot(gs[0, 1])
    T = len(ep["rounds"])
    for row, p in enumerate(seats):
        yb = 3 - row
        for r in ep["rounds"]:
            t = r["t"] + 1
            logged, replays = r["actions"][p], r["requery"][p]
            dep = logged not in replays
            ax2.scatter([t], [yb], s=13, color=(RED if dep else GRN), zorder=3, lw=0)
        ax2.text(-0.6, yb, p, fontsize=8, va="center", ha="right", color=INK, fontweight="bold")
    ax2.set_xlim(0, T + 1), ax2.set_ylim(-0.6, 3.6)
    ax2.set_yticks([])
    ax2.set_xticks([1, 10, 20, 30])
    ax2.tick_params(axis="x", labelsize=7.6)
    ax2.set_xlabel("Round", fontsize=8)
    for sp in ("top", "right", "left"):
        ax2.spines[sp].set_visible(False)
    dep_counts = {p: sum(r["actions"][p] not in r["requery"][p] for r in ep["rounds"]) for p in seats}
    ax2.set_title("Departures from the pool, of " + f"{T} rounds. " +
                  ", ".join(f"{p} {dep_counts[p]}" for p in seats), fontsize=7.6, color=INK2, loc="left", pad=4)
    h1 = mpatches.Patch(color=RED, label="departs from both replays")
    h2 = mpatches.Patch(color=GRN, label="matches a replay")
    ax2.legend(handles=[h1, h2], fontsize=7.4, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, handlelength=1.0, columnspacing=1.0)
    fig.text(0.01, 0.955, "a  Round 2 of one pooled-data-vendor game, DeepSeek sellers", fontsize=8.5, fontweight="bold", va="center")
    fig.text(0.585, 0.955, "b  The whole game, 30 rounds", fontsize=8.5, fontweight="bold", va="center")
    save(fig, "p_fig2_example")




# ---------------------------------------------------------------- conditions schematic --------
def fig_conditions():
    """Figure 3: every set of games, by game, with a tag saying whether a hidden channel is present."""
    import matplotlib.patches as mpatches
    H, C, N = "hidden channel", "chat, no instruction", "honest"
    cols = [("Rock-paper-scissors", "4 seats, 30 rounds", [
                ("Same model", "four copies of one model", N),
                ("Mixed models", "two DeepSeek, two Haiku", N),
                ("Undelivered plans", "plans that reach nobody", N),
                ("Team, no channel", "P1 and P3 told they are a team", N),
                ("Secret plan", "P1 tells P3 both next moves", H)]),
            ("Procurement auction", "4 suppliers, 30 rounds", [
                ("Same model", "published bids only", N),
                ("Public-data vendor", "advice from public bids", N),
                ("Pooled-data vendor", "rivals' private costs pooled", H),
                ("Bid-rotation cartel", "P1 and P3 rotate wins privately", H)]),
            ("Double auction", "5 sellers, 30 hours, pre-registered", [
                ("No chat", "DeepSeek sellers act alone", N),
                ("Chat", "sellers may talk, no instruction", C),
                ("Mixed sellers", "two Haiku, three DeepSeek", N),
                ("Instructed cartel", "new price floor every hour", H)])]
    colour = {H: "#b23a0e", C: "#9a6a00", N: "#6b6963"}
    fill = {H: "#fdeee6", C: "#fbf3df", N: "#f3f2ef"}
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.05))
    for ax, (title, sub, rows) in zip(axes, cols):
        ax.set_axis_off(); ax.set_xlim(0, 100); ax.set_ylim(0, 100)
        ax.text(0, 99, title, fontsize=8.6, fontweight="bold", va="top", color=INK)
        ax.text(0, 91, sub, fontsize=7.4, va="top", color=INK2)
        y = 82
        for name, desc, kind in rows:
            ax.add_patch(mpatches.FancyBboxPatch((0, y - 14.5), 100, 14.5, boxstyle="round,pad=0.2,rounding_size=1.5",
                                                 fc=fill[kind], ec="none"))
            ax.add_patch(mpatches.Rectangle((0, y - 14.5), 2.2, 14.5, fc=colour[kind], ec="none"))
            ax.text(5, y - 4.3, name, fontsize=7.8, fontweight="bold", va="center", color=INK)
            ax.text(5, y - 10.6, desc, fontsize=7.1, va="center", color="#45433e")
            ax.text(98.5, y - 4.3, kind if kind != N else "honest", fontsize=6.9, va="center", ha="right",
                    color=colour[kind], fontweight=("bold" if kind != N else "normal"))
            y -= 17
    fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02, wspace=0.07)
    save(fig, "p_fig3_conditions")


if __name__ == "__main__":
    fig_growth()
    fig_method()
    fig_example()
    fig_conditions()
    fig_main()
    fig_roc()
    fig_eprocess()
    fig_all_conditions()
    fig_pvalues()
    fig_power()
    print("figures written to", FIG)
