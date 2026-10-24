"""Figures and the main table for the report, from results/summary/*.csv (every number comes from
these files; nothing is typed by hand).

    uv run python experiments/make_figures.py
Writes report/figures/fig1_*.pdf|png, fig2_*, fig3_*, and results/summary/table1.csv / .md
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from collusion.analysis import wilson

ROOT = Path(__file__).resolve().parents[1]
SUM = ROOT / "results" / "summary"
FIG = ROOT / "report" / "figures"

# Reference categorical palette (dataviz skill), fixed order, light mode for print.
C = {"crt_net": "#2a78d6", "naive_mi_shuffle": "#eb6834", "bonjour_alpha": "#1baf7a",
     "judge": "#eda100"}
NAMES = {"crt_net": "Replay test (CRT, ours)", "naive_mi_shuffle": "Mutual information + shuffle",
         "bonjour_alpha": "Bonjour et al. rule", "judge": "LLM judge (reads messages)"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "pdf.fonttype": 42, "savefig.dpi": 300,
})


def load(name: str) -> pd.DataFrame | None:
    p = SUM / name
    return pd.read_csv(p) if p.exists() else None


def rate_rows(audits: pd.DataFrame, judge: pd.DataFrame | None, settings, metric: str):
    """[(label, detector, k, n)] for the given (exp-tag, condition, label) settings."""
    out = []
    for tag, cond, label in settings:
        g = audits[(audits.tag == tag) & (audits.condition == cond)]
        if g.empty:
            continue
        for det in ("naive_mi_shuffle", "bonjour_alpha", "crt_net"):
            d = g[g.detector == det]
            if len(d):
                col = "flag05" if metric == "flag" else "correct_pair"
                out.append((label, det, int(d[col].sum()), len(d)))
        if judge is not None:
            j = judge[(judge.tag == tag) & (judge.condition == cond) & (judge.view == "messages")]
            if len(j):
                if metric == "flag":
                    k = int((j.prob >= 50).sum())
                else:
                    k = int(((j.prob >= 50) & (j.pair == j.truth)).sum())
                out.append((label, "judge", k, len(j)))
    return out


def dot_panel(ax, rows, title, ref=None):
    labels = list(dict.fromkeys(r[0] for r in rows))
    dets = [d for d in C if any(r[1] == d for r in rows)]
    off = np.linspace(-0.27, 0.27, len(dets)) if len(dets) > 1 else [0.0]
    for y, lab in enumerate(labels):
        if y % 2 == 0:
            ax.axhspan(y - 0.5, y + 0.5, color="#f4f3f0", zorder=0, lw=0)
        for o, det in zip(off, dets):
            r = [x for x in rows if x[0] == lab and x[1] == det]
            if not r:
                continue
            _, _, k, n = r[0]
            lo, hi = wilson(k, n)
            ax.plot([100 * lo, 100 * hi], [y + o, y + o], color=C[det], lw=1.6,
                    solid_capstyle="round", zorder=2)
            ax.plot(100 * k / n, y + o, "o", ms=5.5, color=C[det], mec="white", mew=1.0,
                    zorder=3)
    if ref is not None:
        ax.axvline(ref, color=INK2, lw=0.8, ls=(0, (3, 2)), zorder=1)
        ax.text(ref + 1.5, len(labels) - 0.45, f"α = {ref:g}%", color=INK2, fontsize=7.5,
                va="top")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_ylim(len(labels) - 0.5, -0.5)
    ax.set_xlim(0, 100)
    ax.xaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=9, color=INK, fontweight="bold")
    ax.tick_params(axis="y", length=0)
    return dets


def fig1():
    s1, s2 = load("s1_audits.csv"), load("s2_audits.csv")
    j1, ag = load("s1_judge.csv"), load("agrawal_panel_runs.csv")
    if s1 is None and s2 is None:
        return
    if s2 is not None:
        s2 = s2.assign(tag=s2.tag + "_s2")  # S1 and S2 both use the tag "deepseek"
    audits = pd.concat([d for d in (s1, s2) if d is not None], ignore_index=True)
    honest = [("deepseek", "null_same", "RPS · 4× DeepSeek"),
              ("haiku", "null_same", "RPS · 4× Claude Haiku"),
              ("mixed", "null_mixed", "RPS · 2 DeepSeek + 2 Haiku"),
              ("deepseek", "talk_no_action", "RPS · collusive talk, never delivered"),
              ("deepseek_s2", "null_same", "Auction · 4× DeepSeek"),
              ("deepseek_s2", "vendor_public", "Auction · shared vendor, public data")]
    colluding = [("deepseek", "pair_hidden", "RPS · DeepSeek pair, hidden channel"),
                 ("haiku", "pair_hidden", "RPS · Haiku pair, hidden channel"),
                 ("deepseek_s2", "vendor_leak", "Auction · vendor leaks rival costs"),
                 ("deepseek_s2", "pair_hidden", "Auction · bid-rotation cartel")]
    rows_h = rate_rows(audits, j1, honest, "flag")
    if ag is not None:
        b = ag[ag.condition == "base"]
        rows_h += [("Agrawal et al. logs · 5× GPT-4.1, no chat", "naive_mi_shuffle",
                    int(b.naive_mi_shuffle.sum()), len(b)),
                   ("Agrawal et al. logs · 5× GPT-4.1, no chat", "bonjour_alpha",
                    int(b.bonjour_alpha.sum()), len(b))]
    rows_c = rate_rows(audits, j1, colluding, "correct")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.9), gridspec_kw={"width_ratios": [1.15, 1]})
    d1 = dot_panel(axes[0], rows_h, "A  Honest agents: how often each detector accuses a pair",
                   ref=5)
    axes[0].set_xlabel("Episodes flagged (%), 95% Wilson interval")
    d2 = dot_panel(axes[1], rows_c, "B  Colluding pair: flagged and named correctly")
    axes[1].set_xlabel("Episodes with the true pair named (%)")
    dets = [d for d in C if d in set(d1) | set(d2)]
    handles = [plt.Line2D([], [], color=C[d], marker="o", lw=1.6, ms=5.5, mec="white")
               for d in dets]
    fig.legend(handles, [NAMES[d] for d in dets], loc="lower center", ncol=len(dets),
               fontsize=7.8, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "fig1_false_accusations")


def fig2():
    s1, j1 = load("s1_audits.csv"), load("s1_judge.csv")
    if s1 is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, tag, name in zip(axes, ("deepseek", "haiku"), ("DeepSeek", "Claude Haiku")):
        g = s1[s1.tag == tag]
        for det in ("crt_net", "naive_mi_shuffle"):
            d = g[(g.detector == det) & g.condition.isin(["pair_hidden", "null_same"])]
            if d.empty or d.condition.nunique() < 2:
                continue
            y = (d.condition == "pair_hidden").astype(int)
            fpr, tpr, _ = roc_curve(y, -np.log(d.p.clip(lower=1e-6)))
            ax.plot(100 * fpr, 100 * tpr, color=C[det], lw=2, label=NAMES[det])
        if j1 is not None:
            for view, ls in (("actions", "-"), ):
                d = j1[(j1.tag == tag) & (j1.view == view)
                       & j1.condition.isin(["pair_hidden", "null_same"])].dropna(subset=["prob"])
                if d.condition.nunique() == 2:
                    y = (d.condition == "pair_hidden").astype(int)
                    fpr, tpr, _ = roc_curve(y, d.prob)
                    ax.plot(100 * fpr, 100 * tpr, color=C["judge"], lw=2, ls=ls,
                            label="LLM judge (actions only)")
        ax.plot([0, 100], [0, 100], color=GRID, lw=0.8)
        for x in (1, 5):
            ax.axvline(x, color=INK2, lw=0.6, ls=(0, (2, 2)))
        ax.set_xlim(0, 100), ax.set_ylim(0, 100)
        ax.set_title(f"{name}: hidden-channel pair vs honest", loc="left", fontsize=9,
                     fontweight="bold")
        ax.set_xlabel("False-alarm rate on honest episodes (%)")
        ax.grid(True, color=GRID, lw=0.5)
    axes[0].set_ylabel("Detection rate (%)")
    axes[0].legend(loc="lower right", fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig2_roc")


def fig3():
    pw = load("s0b_power.csv")
    if pw is None:
        return
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    cols = {0.4: "#9fc4ef", 0.6: "#5f9be3", 0.8: "#2a78d6", 1.0: "#16457f"}
    for cp, g in pw[pw.cp > 0].groupby("cp"):
        if cp not in cols:
            continue
        m = g.groupby("T").correct.mean()
        ax.plot(m.index, 100 * m.values, color=cols[cp], lw=2, marker="o", ms=4)
        ax.text(m.index[-1] + 4, 100 * m.values[-1], f"{cp:.0%}", color=INK2, fontsize=7.5,
                va="center")
    fa = pw[pw.cp == 0].groupby("T").flag.mean()
    ax.plot(fa.index, 100 * fa.values, color=INK2, lw=1.2, ls=(0, (3, 2)), marker="o", ms=3)
    ax.text(fa.index[-1] + 4, 100 * fa.values[-1], "honest", color=INK2, fontsize=7.5,
            va="center")
    # LLM episodes truncated to their first T rounds (results/summary/trunc_*_audits.csv)
    marks = {"deepseek": ("s", "DeepSeek pair (RPS)"), "haiku": ("D", "Haiku pair (RPS)"),
             "deepseek_s2": ("^", "Vendor leak (auction)")}
    pts = []
    for f in sorted(SUM.glob("trunc*_audits.csv")):
        d = pd.read_csv(f)
        d = d[(d.detector == "crt_net") & d.condition.isin(["pair_hidden", "vendor_leak"])]
        if "exp" in d and (d.exp == "s2").any():
            d = d.assign(tag=d.tag + "_s2")
        pts.append(d)
    if pts:
        d = pd.concat(pts)
        d = d[~((d.tag == "deepseek_s2") & (d.condition == "pair_hidden"))]
        for tag, g in d.groupby("tag"):
            if tag not in marks:
                continue
            m = g.groupby("T").correct_pair.mean()
            ax.plot(m.index, 100 * m.values, ls="none", marker=marks[tag][0], ms=5,
                    mfc="white", mec=INK, mew=1.1, label=marks[tag][1])
        ax.legend(loc="lower right", fontsize=6.8, handletextpad=0.3)
    ax.set_xlabel("Rounds in the audited log (T)")
    ax.set_ylabel("Correct pair flagged (%)")
    ax.set_title("Detectability frontier (simulator)", loc="left", fontsize=9,
                 fontweight="bold")
    ax.set_xlim(15, 140), ax.set_ylim(0, 105)
    ax.grid(True, color=GRID, lw=0.5)
    ax.text(0.02, 0.97, "label = share of rounds the\npair coordinates", transform=ax.transAxes,
            fontsize=7, color=INK2, va="top")
    fig.tight_layout()
    save(fig, "fig3_power")


def fig_pvalues():
    s1 = load("s1_audits.csv")
    if s1 is None:
        return
    honest = s1[s1.condition.isin(["null_same", "null_mixed", "talk_no_action"])
                & ~s1.tag.str.startswith("val_")]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.2), sharey=True)
    for ax, det in zip(axes, ("crt_net", "naive_mi_shuffle")):
        p = honest[honest.detector == det].p.dropna()
        ax.hist(p, bins=np.linspace(0, 1, 21), color=C[det], edgecolor="white", lw=1.0)
        ax.axhline(len(p) / 20, color=INK2, lw=0.8, ls=(0, (3, 2)))
        ax.set_title(f"{NAMES[det]}: p-values on {len(p)} honest episodes", loc="left",
                     fontsize=8.5, fontweight="bold")
        ax.set_xlabel("p-value")
    axes[0].set_ylabel("Episodes")
    fig.tight_layout()
    save(fig, "figA_pvalues")


def table1():
    rows = []
    for exp, name in (("s1", "s1_summary.csv"), ("s2", "s2_summary.csv")):
        s = load(name)
        if s is None:
            continue
        s = s[s.detector.isin(["naive_mi_shuffle", "bonjour_alpha", "crt_net"])]
        for _, r in s.iterrows():
            rows.append({"env": exp, "tag": r.tag, "condition": r.condition,
                         "detector": r.detector, "n": int(r.n),
                         "flag%": round(100 * r.rate05, 1),
                         "CI": f"{100 * r.rate05_lo:.0f}–{100 * r.rate05_hi:.0f}",
                         "flag@1%": round(100 * r.rate01, 1),
                         "correct_pair%": round(100 * r.correct_pair05, 1),
                         "pair_effect_bits": round(r.pair_effect_bits, 3)})
    if rows:
        t = pd.DataFrame(rows)
        t.to_csv(SUM / "table1.csv", index=False)
        (SUM / "table1.md").write_text(t.to_markdown(index=False))


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig_pvalues()
    table1()
    print("figures written to", FIG)
