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
C = {"crt_net": "#2a78d6", "seq": "#16457f", "naive_mi_shuffle": "#eb6834",
     "bonjour_alpha": "#1baf7a", "judge": "#c98a00", "channel": "#c2410c"}
NAMES = {"crt_net": "Replay test (ours)", "naive_mi_shuffle": "Mutual information, shuffle null",
         "bonjour_alpha": "Bonjour et al. rule", "judge": "LLM judge (actions only)"}


def load(name):
    p = SUM / name
    return pd.read_csv(p) if p.exists() else None


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=600)
    plt.close(fig)


def panel_label(ax, s, x=-0.02, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9.5, fontweight="bold", va="bottom",
            ha="right")


# ------------------------------------------------------------------ Fig. 1: the method --------
def fig_method():
    fig = plt.figure(figsize=(WIDTH, 1.75))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 127), ax.set_ylim(3, 39.5)
    ax.axis("off")

    def box(x, y, w, h, text, fc="white", ec=INK2, lw=0.8, size=8.5, style="round,pad=0.4"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fc, ec=ec, lw=lw))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size)

    def arrow(x0, y0, x1, y1, color=INK2, ls="-", lw=0.9):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7,
                                     color=color, lw=lw, ls=ls, shrinkA=0, shrinkB=0))

    heads = [(1, "1  Deployment"), (35, "2  Replay"), (66, "3  Pool"), (92, "4  Test")]
    for x, h in heads:
        ax.text(x, 35.5, h, fontsize=9, fontweight="bold", va="center")
    # 1. deployment: agent i acts on its logged view V and on the hidden channel C
    ax.text(1.5, 26, r"$V_{i,t}$", fontsize=9.5, va="center")
    ax.text(1.5, 17, r"$C_{i,t}$", fontsize=9.5, va="center", color=C["channel"])
    box(16, 18.5, 7.5, 6, r"$\pi_i$", size=9.5)
    arrow(7.2, 26, 15.4, 23.6)
    arrow(7.2, 17, 15.4, 19.4, color=C["channel"], ls=(0, (3, 2)))
    arrow(24.2, 21.5, 31.5, 21.5)
    ax.text(27.8, 23.2, r"$a_{i,t}$", fontsize=9.5, ha="center")
    ax.text(1, 8.2, "acts on the logged view $V$\nand the hidden channel $C$", fontsize=7.6,
            color=INK2, va="center", linespacing=1.3)
    # 2. replay: the same policy on V alone, k = 2 draws
    ax.text(35.5, 26, r"$V_{i,t}$", fontsize=9.5, va="center")
    ax.text(35.5, 17, r"$C_{i,t}$", fontsize=9.5, va="center", color=C["channel"])
    ax.plot([35.0, 41.2], [15.3, 18.8], color=C["channel"], lw=1.0)
    box(46, 18.5, 7.5, 6, r"$\pi_i$", size=9.5)
    arrow(41.2, 26, 45.4, 23.6)
    arrow(54.2, 23.3, 57.6, 26.0)
    arrow(54.2, 19.7, 57.6, 17.0)
    ax.text(58.2, 26.6, r"$\tilde a^{(1)}_{i,t}$", fontsize=9.5, va="center")
    ax.text(58.2, 16.2, r"$\tilde a^{(2)}_{i,t}$", fontsize=9.5, va="center")
    ax.text(35, 8.2, "asked again on $V$ alone,\ntwice", fontsize=7.6, color=INK2,
            va="center", linespacing=1.3)
    # 3. pool: the logged move and its replays
    for j, (lab, logged) in enumerate(((r"$a_{i,t}$", True), (r"$\tilde a^{(1)}_{i,t}$", False),
                                       (r"$\tilde a^{(2)}_{i,t}$", False))):
        x = 66.5 + 7.0 * j
        ax.add_patch(Rectangle((x, 18.5), 6.2, 6, fc="#eaf2fc" if logged else "white",
                               ec=C["crt_net"] if logged else INK2, lw=1.3 if logged else 0.8))
        ax.text(x + 3.1, 21.5, lab, ha="center", va="center", fontsize=8.5)
    ax.text(76.6, 27.6, r"pool $P_{i,t}$", ha="center", fontsize=9)
    ax.text(66, 8.2, "under $H_0$ the logged move\nis uniform in its pool", fontsize=7.6,
            color=INK2, va="center", linespacing=1.3)
    arrow(87.6, 21.5, 90.8, 21.5)
    # 4. test: null distribution with the observed statistic, and an e-process
    rng = np.random.default_rng(3)
    hist = np.histogram(rng.gamma(3, 1, 4000), bins=16, range=(0, 9))[0]
    for x, h in zip(np.linspace(92.5, 103, 16), hist):
        ax.add_patch(Rectangle((x, 15), 0.6, 9 * h / hist.max(), fc="#c9c7c1", ec="none"))
    ax.plot([101.6, 101.6], [15, 25.5], color=C["crt_net"], lw=1.2)
    ax.text(101.6, 26.4, r"$S$", ha="center", fontsize=9, color=C["crt_net"])
    ax.text(98, 8.2, "fixed test:\n$p$-value", ha="center", va="center", fontsize=7.6,
            color=INK2, linespacing=1.3)
    tt = np.linspace(0, 1, 50)
    path = 15 + 11 * (np.exp(2.4 * tt) - 1) / (np.exp(2.4) - 1) + 0.7 * np.sin(13 * tt) * tt
    ax.plot(109 + 16 * tt, path, color=C["seq"], lw=1.1)
    ax.plot([109, 125], [22, 22], color=INK2, lw=0.7, ls=(0, (3, 2)))
    ax.text(109.2, 22.7, r"$1/\alpha$", fontsize=8.5, ha="left", va="bottom", color=INK2)
    ax.text(116.5, 8.2, "sequential test:\nalarm at $E_T \\geq 1/\\alpha$", ha="center",
            va="center", fontsize=7.6, color=INK2, linespacing=1.3)
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


def _dots(ax, rows, ref=None):
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
        ax.text(ref + 1.5, len(labels) - 0.55, r"$\alpha$ = 5%", fontsize=8, color=INK2,
                va="bottom")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_ylim(len(labels) - 0.5, -0.5)
    ax.set_xlim(0, 100)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    return dets


def fig_main():
    s1, s2, ag = load("s1_audits.csv"), load("s2_audits.csv"), load("agrawal_panel_runs.csv")
    audits = pd.concat([s1, s2.assign(tag=s2.tag + "_s2")], ignore_index=True)
    honest = [("haiku", "null_same", "4 × Claude Haiku"), ("deepseek", "null_same", "4 × DeepSeek")]
    rows_h = _rows(audits, honest, "flag")
    b = ag[ag.condition == "base"]
    rows_h += [("Agrawal et al. logs", "naive_mi_shuffle", int(b.naive_mi_shuffle.sum()), len(b)),
               ("Agrawal et al. logs", "bonjour_alpha", int(b.bonjour_alpha.sum()), len(b))]
    coll = [("haiku", "pair_hidden", "Haiku pair, secret plan"),
            ("deepseek", "pair_hidden", "DeepSeek pair, secret plan"),
            ("deepseek_s2", "vendor_leak", "Pooled-data vendor")]
    rows_c = _rows(audits, coll, "correct")
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.25))
    _dots(axes[0], rows_h, ref=5)
    axes[0].set_xlabel("Honest games flagged (%)")
    _dots(axes[1], rows_c)
    axes[1].set_xlabel("True pair named (%)")
    panel_label(axes[0], "a", x=-0.62)
    panel_label(axes[1], "b", x=-0.72)
    hs = [plt.Line2D([], [], color=C[d], marker="o", lw=1.5, ms=4.6, mec="white")
          for d in ("naive_mi_shuffle", "bonjour_alpha", "crt_net")]
    fig.legend(hs, [NAMES[d] for d in ("naive_mi_shuffle", "bonjour_alpha", "crt_net")],
               loc="lower center", ncol=3, bbox_to_anchor=(0.55, -0.01), handlelength=1.6)
    fig.subplots_adjust(left=0.19, right=0.985, top=0.9, bottom=0.33, wspace=0.95)
    save(fig, "p_fig2_main")


# ------------------------------------------------------------------ Fig. 3: ROC ---------------
def fig_roc():
    s1, j1 = load("s1_audits.csv"), load("s1_judge.csv")
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.25), sharey=True)
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
        ax.plot(xs, xs, color=GRID, lw=0.8)
        for x in (1, 5):
            ax.axvline(x, color=INK2, lw=0.6, ls=(0, (2, 2)))
        ax.set_xscale("log")
        ax.set_xlim(0.5, 100), ax.set_ylim(0, 100)
        ax.set_xticks([1, 5, 20, 100], ["1", "5", "20", "100"])
        ax.set_xlabel("Honest games flagged (%), log scale")
        ax.text(0.6, 93, name, fontsize=8.5, fontstyle="italic", va="top")
        ax.grid(True, color=GRID, lw=0.5)
    axes[0].set_ylabel("Colluding games\ndetected (%)")
    panel_label(axes[0], "a", x=-0.2)
    panel_label(axes[1], "b", x=-0.05)
    hs, ls = axes[0].get_legend_handles_labels()
    fig.legend(hs, ls, loc="lower center", ncol=3, bbox_to_anchor=(0.55, -0.01), handlelength=1.6)
    fig.subplots_adjust(left=0.12, right=0.985, top=0.92, bottom=0.32, wspace=0.12)
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
    fig, ax = plt.subplots(figsize=(WIDTH, 1.9))
    t = np.arange(paths["null_same"].shape[1])
    for e in paths["null_same"]:
        ax.plot(t, e, color="#9a9890", lw=0.6, alpha=0.7)
    for e in paths["pair_hidden"]:
        ax.plot(t, e, color=C["seq"], lw=0.9, alpha=0.85)
    ax.axhline(20, color=C["channel"], lw=0.9, ls=(0, (4, 2)))
    ax.text(0.3, 26, r"alarm threshold $1/\alpha = 20$", ha="left", va="bottom", fontsize=8,
            color=C["channel"])
    ax.set_yscale("log")
    ax.set_ylim(0.05, 1e4)
    ax.set_xlim(0, t[-1])
    ax.set_xlabel("Round $T$")
    ax.set_ylabel(r"Evidence $E_T$ (log scale)")
    ax.grid(True, color=GRID, lw=0.5)
    n0, n1 = len(paths["null_same"]), len(paths["pair_hidden"])
    hs = [plt.Line2D([], [], color=C["seq"], lw=1.2), plt.Line2D([], [], color="#9a9890", lw=1.0)]
    ax.legend(hs, [f"instructed pair ({n1} games)", f"honest agents ({n0} games)"],
              loc="upper left", handlelength=1.6, bbox_to_anchor=(0.0, 1.0))
    fig.subplots_adjust(left=0.1, right=0.985, top=0.95, bottom=0.23)
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
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.6),
                             gridspec_kw={"width_ratios": [1, 1], "wspace": 1.15})
    _dots(axes[0], rows_h, ref=5)
    axes[0].set_xlabel("Honest games flagged (%)")
    _dots(axes[1], rows_c)
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
        ax.set_xlabel(f"$p$-value, {NAMES[det].lower().replace(' (ours)', '')}")
        panel_label(ax, lab, x=-0.08)
    axes[0].set_ylabel("Honest games")
    fig.subplots_adjust(left=0.1, right=0.985, top=0.9, bottom=0.25, wspace=0.08)
    save(fig, "p_figA2_pvalues")


def fig_power():
    pw = load("s0b_power.csv")
    fig, ax = plt.subplots(figsize=(WIDTH * 0.62, 2.3))
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
    fig.subplots_adjust(left=0.15, right=0.6, top=0.95, bottom=0.2)
    save(fig, "p_figA3_power")


if __name__ == "__main__":
    fig_method()
    fig_main()
    fig_roc()
    fig_eprocess()
    fig_all_conditions()
    fig_pvalues()
    fig_power()
    print("figures written to", FIG)
