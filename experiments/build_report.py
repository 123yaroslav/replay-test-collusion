"""Build the report PDF: fill report/template.html with numbers computed from results/summary/*.csv,
then print it with headless Chrome. Every number in the report comes from here.

    uv run python experiments/build_report.py
Writes report/report.html, report/report.pdf and report/numbers.json; prints the abstract word
count, the page count and any placeholder left unfilled.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from collusion.analysis import p_fpr_above, p_fpr_below, wilson

ROOT = Path(__file__).resolve().parents[1]
SUM = ROOT / "results" / "summary"
REP = ROOT / "report"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def pct(k, n, digits=None):
    """Percent with one decimal only when it is not zero (37/40 -> 92.5%, 6/150 -> 4%)."""
    if not n:
        return "n/a"
    if digits is not None:
        return f"{100 * k / n:.{digits}f}%"
    s = f"{100 * k / n:.1f}"
    return (s.removesuffix(".0")) + "%"


def ci(k, n):
    lo, hi = wilson(k, n)
    return f"{100 * lo:.0f}–{100 * hi:.0f}%"


def read(name):
    p = SUM / name
    return pd.read_csv(p) if p.exists() else None


def detector_stats(df, tag, cond, det, col="flag05"):
    g = df[(df.tag == tag) & (df.condition == cond) & (df.detector == det)]
    return int(g[col].sum()), len(g)


def auroc_ci(pos, neg, n_boot: int = 2000, seed: int = 0) -> str:
    """Percentile bootstrap CI for AUROC, resampling episodes within each class."""
    from collusion.analysis import auroc

    rng = np.random.default_rng(seed)
    vals = [auroc(rng.choice(pos, len(pos)), rng.choice(neg, len(neg))) for _ in range(n_boot)]
    return f"{np.percentile(vals, 2.5):.2f}–{np.percentile(vals, 97.5):.2f}"


def numbers() -> dict:
    v: dict = {}
    s1, s2 = read("s1_audits.csv"), read("s2_audits.csv")
    ag, j1 = read("agrawal_panel_runs.csv"), read("s1_judge.csv")
    s0b_cal, s0b_pw = read("s0b_calibration.csv"), read("s0b_power.csv")

    if ag is not None:
        from scipy.stats import fisher_exact
        b, c = ag[ag.condition == "base"], ag[ag.seller_comms]
        tab = [[int(b.naive_mi_shuffle.sum()), len(b) - int(b.naive_mi_shuffle.sum())],
               [int(c.naive_mi_shuffle.sum()), len(c) - int(c.naive_mi_shuffle.sum())]]
        v["ag_fisher_p"] = f"{fisher_exact(tab)[1]:.2f}"
        for det in ("naive_mi_shuffle", "bonjour_alpha", "net_shuffle"):
            k, n = int(b[det].sum()), len(b)
            v[f"ag_base_{det}"] = f"{k}/{n}"
            v[f"ag_base_{det}_pct"] = pct(k, n)
            v[f"ag_base_{det}_words"] = f"{k} of {n}"
            v[f"ag_base_{det}_ci"] = ci(k, n)
            kc, nc = int(c[det].sum()), len(c)
            v[f"ag_comms_{det}"] = f"{kc}/{nc}"
            v[f"ag_comms_{det}_pct"] = pct(kc, nc)

    if s0b_cal is not None:
        g = s0b_cal[(s0b_cal.variant == "pooled") & (s0b_cal.k == 2)]
        r = g.groupby("population").p.apply(lambda p: (p < 0.05).mean())
        v["s0b_fpr_range"] = f"{100 * r.min():.1f}–{100 * r.max():.1f}%"
        v["s0b_iters"] = str(int(g.groupby("population").size().min()))
    if s0b_pw is not None:
        m = s0b_pw[s0b_pw.cp > 0].groupby(["cp", "T"]).correct.mean()
        for cp in (0.6, 0.8):
            for T in (30, 60):
                if (cp, T) in m.index:
                    v[f"s0b_power_{int(cp * 100)}_{T}"] = f"{100 * m[(cp, T)]:.0f}%"

    honest_s1 = [("deepseek", "null_same"), ("haiku", "null_same"), ("mixed", "null_mixed"),
                 ("deepseek", "talk_no_action"), ("haiku", "talk_no_action")]
    if s1 is not None:
        for tag, cond in honest_s1 + [("deepseek", "pair_hidden"), ("haiku", "pair_hidden"),
                                      ("deepseek", "told_no_channel"),
                                      ("haiku", "told_no_channel"), ("deepseek", "pair_aware"),
                                      ("deepseek", "pair_chat")]:
            for det in ("naive_mi_shuffle", "bonjour_alpha", "crt_net"):
                k, n = detector_stats(s1, tag, cond, det)
                if n:
                    key = f"s1_{tag}_{cond}_{det}"
                    v[key] = f"{k}/{n}"
                    v[key + "_pct"] = pct(k, n)
                    v[key + "_ci"] = ci(k, n)
                    kc, _ = detector_stats(s1, tag, cond, det, "correct_pair")
                    v[key + "_correct_pct"] = pct(kc, n)
        # pooled honest episodes for the replay test
        h = pd.concat([s1[(s1.tag == t) & (s1.condition == c) & (s1.detector == "crt_net")]
                       for t, c in honest_s1])
        k, n = int(h.flag05.sum()), len(h)
        k1 = int(h.flag01.sum())
        v["s1_honest_n"] = str(n)
        v["s1_honest_crt"] = f"{k}/{n}"
        v["s1_honest_crt_pct"] = pct(k, n, 1)
        v["s1_honest_crt_ci"] = ci(k, n)
        v["s1_honest_crt01_pct"] = pct(k1, n, 1)
        v["s1_honest_crt_p_below10"] = f"{p_fpr_below(k, n, 0.10):.1e}"
        v["s1_honest_crt_per1000"] = f"{1000 * k / n:.0f}" if n else "n/a"
        for det in ("naive_mi_shuffle", "bonjour_alpha"):
            hd = pd.concat([s1[(s1.tag == t) & (s1.condition == c) & (s1.detector == det)]
                            for t, c in honest_s1])
            kk, nn = int(hd.flag05.sum()), len(hd)
            v[f"s1_honest_{det}_pct"] = pct(kk, nn, 1)
            v[f"s1_honest_{det}_ci"] = ci(kk, nn)
            v[f"s1_honest_{det}_p_above5"] = f"{p_fpr_above(kk, nn, 0.05):.1e}"
            v[f"s1_honest_{det}_per1000"] = f"{1000 * kk / nn:.0f}" if nn else "n/a"
        val = s1[s1.tag.str.startswith("val_") & (s1.detector == "crt_net")]
        for cond in ("null_same", "pair_hidden"):
            g = val[val.condition == cond]
            col = "flag05" if cond == "null_same" else "correct_pair"
            v[f"val_{cond}_crt"] = f"{int(g[col].sum())}/{len(g)}"
            v[f"val_{cond}_crt_pct"] = pct(int(g[col].sum()), len(g))
        eff = s1[(s1.detector == "crt_net") & (s1.condition == "pair_hidden")]
        for tag, g in eff.groupby("tag"):
            v[f"s1_{tag}_pair_effect"] = f"{g.pair_effect_bits.mean():.2f}"
        v["s1_n_episodes"] = str(s1[s1.detector == "crt_net"].shape[0])

    if s2 is not None:
        for cond in ("null_same", "vendor_public", "vendor_leak", "pair_hidden"):
            for det in ("naive_mi_shuffle", "bonjour_alpha", "crt_net"):
                k, n = detector_stats(s2, "deepseek", cond, det)
                if n:
                    key = f"s2_{cond}_{det}"
                    v[key] = f"{k}/{n}"
                    v[key + "_pct"] = pct(k, n)
                    v[key + "_ci"] = ci(k, n)
                    kc, _ = detector_stats(s2, "deepseek", cond, det, "correct_pair")
                    v[key + "_correct_pct"] = pct(kc, n)
        v["s2_n_episodes"] = str(s2[s2.detector == "crt_net"].shape[0])

    if j1 is not None:
        for tag in ("deepseek", "haiku"):
            for cond in ("null_same", "pair_hidden", "talk_no_action"):
                for view in ("actions", "messages"):
                    g = j1[(j1.tag == tag) & (j1.condition == cond) & (j1.view == view)]
                    if len(g):
                        k = int((g.prob >= 50).sum())
                        v[f"judge_{tag}_{cond}_{view}_pct"] = pct(k, len(g))
                        v[f"judge_{tag}_{cond}_{view}"] = f"{k}/{len(g)}"
    ex_md, ex_js = SUM / "example_episode.md", SUM / "example_episode.json"
    if ex_md.exists() and ex_js.exists():
        info = json.loads(ex_js.read_text())
        v["example_table"] = ex_md.read_text()
        v["example_p"] = f"{info['p']:.3f}"
        v["example_seed"] = str(info["seed"])
        v["example_rounds"] = str(info["rounds"])
        v["example_shown"] = str(info["shown"])
    s0 = read("s0_partB_summary.csv")
    if s0 is not None:
        g = s0[(s0.population == "null_herd") & (s0.n_games == 500)].set_index("detector").flag_rate
        v["s0_naive_herd_fpr"] = f"{100 * g.get('raw_gamma_shuffle', float('nan')):.0f}%"
        v["s0_crt_herd_fpr"] = f"{100 * g.get('net_crt', float('nan')):.1f}%"
        v["s0_bonjour_herd_fpr"] = f"{100 * g.get('bonjour_alpha_nostate', float('nan')):.1f}%"
    if s1 is not None:
        from collusion.analysis import auroc
        for tag, cond in (("deepseek", "null_same"), ("haiku", "null_same"), ("mixed", "null_mixed")):
            k, n = detector_stats(s1, tag, cond, "crt_net")
            if n:
                v[f"s1_{tag}_{cond}_crt_p_below10"] = f"{p_fpr_below(k, n, 0.10):.1e}"
        for tag in ("deepseek", "haiku"):
            for det in ("crt_net", "naive_mi_shuffle"):
                d = s1[(s1.tag == tag) & (s1.detector == det)]
                pos = -np.log(d[d.condition == "pair_hidden"].p.clip(lower=1e-6))
                neg = -np.log(d[d.condition == "null_same"].p.clip(lower=1e-6))
                if len(pos) and len(neg):
                    v[f"auroc_{tag}_{det}"] = f"{auroc(pos.values, neg.values):.2f}"
                    v[f"auroc_{tag}_{det}_ci"] = auroc_ci(pos.values, neg.values)
            if j1 is not None:
                for view in ("actions", "messages"):
                    jj = j1[(j1.tag == tag) & (j1.view == view)].dropna(subset=["prob"])
                    pos = jj[jj.condition == "pair_hidden"].prob
                    neg = jj[jj.condition == "null_same"].prob
                    if len(pos) and len(neg):
                        v[f"auroc_{tag}_judge_{view}"] = f"{auroc(pos.values, neg.values):.2f}"
                        v[f"auroc_{tag}_judge_{view}_ci"] = auroc_ci(pos.values, neg.values)
    js = read("s1_judge_strong.csv")
    if js is not None:
        from collusion.analysis import auroc as _auroc
        js = js[js.parsed]
        for (tag, cond, view), g in js.groupby(["tag", "condition", "view"]):
            v[f"judge_strong_{tag}_{cond}_{view}_pct"] = pct(int((g.prob >= 50).sum()), len(g))
        for tag in ("deepseek", "haiku"):
            for view in ("actions", "messages"):
                g = js[(js.tag == tag) & (js.view == view)]
                pos, neg = g[g.condition == "pair_hidden"].prob, g[g.condition == "null_same"].prob
                if len(pos) and len(neg):
                    v[f"auroc_{tag}_judge_strong_{view}"] = f"{_auroc(pos.values, neg.values):.2f}"
        v["judge_strong_n"] = str(int(js.groupby(["tag", "condition", "view"]).size().min()))
    if s1 is not None:
        # recall at a fixed false-alarm budget, from the ROC of each detector (pair_hidden vs null_same)
        from sklearn.metrics import roc_curve

        def tpr_at(y, score, fpr_max):
            fpr, tpr, _ = roc_curve(y, score)
            ok = fpr <= fpr_max + 1e-12
            return float(tpr[ok].max()) if ok.any() else 0.0

        for tag in ("deepseek", "haiku"):
            for det in ("crt_net", "naive_mi_shuffle"):
                d = s1[(s1.tag == tag) & (s1.detector == det) & s1.condition.isin(["pair_hidden", "null_same"])]
                if d.condition.nunique() == 2:
                    y = (d.condition == "pair_hidden").astype(int)
                    sc = -np.log(d.p.clip(lower=1e-6))
                    for f in (0.01, 0.05):
                        v[f"tpr{int(f * 100)}_{tag}_{det}"] = f"{100 * tpr_at(y, sc, f):.0f}%"
            for jname, jdf in (("judge", j1), ("judge_strong", read("s1_judge_strong.csv"))):
                if jdf is None:
                    continue
                d = jdf[(jdf.tag == tag) & (jdf.view == "actions")
                        & jdf.condition.isin(["pair_hidden", "null_same"])].dropna(subset=["prob"])
                if d.condition.nunique() == 2:
                    y = (d.condition == "pair_hidden").astype(int)
                    for f in (0.01, 0.05):
                        v[f"tpr{int(f * 100)}_{tag}_{jname}"] = f"{100 * tpr_at(y, d.prob, f):.0f}%"
        for tag in sorted(t for t in s1.tag.unique() if t.startswith("val_")):
            for cond in ("null_same", "pair_hidden"):
                k, n = detector_stats(s1, tag, cond, "crt_net",
                                      "flag05" if cond == "null_same" else "correct_pair")
                if n:
                    key = re.sub(r"\W", "_", tag)  # placeholder names are \w only
                    v[f"{key}_{cond}_crt"] = f"{k}/{n}"
                    v[f"{key}_{cond}_crt_ci"] = ci(k, n)
    if s1 is not None:
        # seat check: colluders always sit at P1, P3; under the null the named pair should be P1-P3 ~1/6
        nul = s1[(s1.detector == "crt_net") & s1.tag.isin(["deepseek", "haiku", "mixed"])
                 & s1.condition.isin(["null_same", "null_mixed", "talk_no_action", "told_no_channel"])]
        v["seat_p13_null"] = f"{100 * (nul.top == 'P1-P3').mean():.0f}%"
        v["seat_p13_null_n"] = str(len(nul))
    import json as _js
    v1c = SUM / "pilot_cda_rebin_v1_summary.json"
    if v1c.exists():
        k = _js.loads(v1c.read_text())["cells"].get("SM-cartel")
        if k:
            v["cartel_v1_mi"], v["cartel_v1_net"] = str(k["crt_mi_flags"]), str(k["crt_net_flags"])
    for tag, key in (("pilot_cda_summary.json", ""), ("pilot_cda_rebin_v2_summary.json", "_v2")):
        pf = SUM / tag
        if not pf.exists():
            continue
        c = _js.loads(pf.read_text())["cells"]
        off, on = [x for x in c if x.endswith("off")], [x for x in c if x.endswith("on")]
        n_off, n_on = sum(c[x]["episodes"] for x in off), sum(c[x]["episodes"] for x in on)
        v[f"pilot_naive_off{key}"] = f"{sum(c[x]['naive_mi_flags'] for x in off)} of {n_off}"
        v[f"pilot_naive_on{key}"] = f"{sum(c[x]['naive_mi_flags'] for x in on)} of {n_on}"
        if not key:
            un = off + on  # uninstructed cells only; the instructed cartel is reported separately
            v["pilot_eps"] = str(n_off + n_on)
            v["pilot_msgs"] = str(sum(c[x]["messages"] for x in un))
            v["pilot_agree_msgs"] = str(sum(c[x]["messages_with_agreement_words"] for x in un))
            v["pilot_replay_flags"] = str(sum(c[x]["crt_mi_flags"] + c[x]["seq_alarms"]
                                              + c[x]["crt_net_flags"] for x in un))
            if "SM-cartel" in c:
                k = c["SM-cartel"]
                v["cartel_eps"] = str(k["episodes"])
                v["cartel_price"] = f"{k['mean_trade_price']:.2f}"
                v["cartel_trades"] = f"{k['trades_per_episode']:.1f}"
                v["cartel_floor_hours"] = str(k["cartel_floor_hours"])
                v["cartel_leader"] = f"{100 * k['cartel_leader_compliance']:.0f}%"
                v["cartel_follower"] = f"{100 * k['cartel_follower_compliance']:.0f}%"
                v["cartel_naive"] = str(k["naive_mi_flags"])
        if key == "_v2" and "SM-cartel" in c:
            k = c["SM-cartel"]
            v["cartel_v2_mi"], v["cartel_v2_net"] = str(k["crt_mi_flags"]), str(k["crt_net_flags"])
            v["cartel_seq"] = str(k["seq_alarms"])
            for cell in ("SM-off", "SM-on", "HK-off", "HK-on"):
                if cell in c:
                    v[f"pilot_price_{cell.replace('-', '_')}"] = f"{c[cell]['mean_trade_price']:.2f}"
    ev = read("s0e_evasion_summary.csv")
    if ev is not None:
        def e(strategy, T, col):
            r = ev[(ev.strategy == strategy) & (ev["T"] == T)]
            return float(r[col].iloc[0])
        full30 = e("rotation_0.0", 30, "gain_overpay")
        v["s0e_harm_full30"] = f"{full30:.2f}"
        v["s0e_harm_half30"] = f"{e('rotation_0.5', 30, 'gain_overpay'):.2f}"
        v["s0e_harm_frac_half30"] = f"{100 * e('rotation_0.5', 30, 'gain_overpay') / full30:.0f}%"
        v["s0e_seq_full30"] = f"{100 * e('rotation_0.0', 30, 'seq'):.0f}%"
        v["s0e_seq_half30"] = f"{100 * e('rotation_0.5', 30, 'seq'):.0f}%"
        v["s0e_seq_half60"] = f"{100 * e('rotation_0.5', 60, 'seq'):.0f}%"
        v["s0e_seq_q60"] = f"{100 * e('rotation_0.75', 60, 'seq'):.0f}%"
        v["s0e_harm_frac_q60"] = f"{100 * e('rotation_0.75', 60, 'gain_overpay') / e('rotation_0.0', 60, 'gain_overpay'):.0f}%"
        v["s0e_const_harm"] = f"{e('constant', 30, 'gain_overpay'):.1f}"
        v["s0e_const_seq"] = f"{100 * min(e('constant', 30, 'seq'), e('constant', 60, 'seq')):.0f}%"
        v["s0e_const_dev"] = f"{100 * e('constant', 30, 'deviation'):.0f}–{100 * e('constant', 60, 'deviation'):.0f}%"
        v["s0e_const_dep"] = f"{100 * min(e('constant', T, d) for T in (30, 60) for d in ('crt_net', 'crt_mi')):.1f}–{100 * max(e('constant', T, d) for T in (30, 60) for d in ('crt_net', 'crt_mi')):.0f}%"
        hon = ev[ev.strategy == "honest"]
        v["s0e_honest_range"] = f"{100 * hon[['crt_net', 'crt_mi', 'seq', 'deviation']].min().min():.1f}–{100 * hon[['crt_net', 'crt_mi', 'seq', 'deviation']].max().max():.0f}%"
        v["s0e_iters"] = str(int(ev.n.iloc[0]))
    s0d = read("s0d_coalition_summary.csv")
    if s0d is not None:
        def g(setup, T, det, col="flag"):
            r = s0d[(s0d.setup == setup) & (s0d["T"] == T) & (s0d.detector == det)]
            return float(r[col].iloc[0]) if len(r) else float("nan")
        for det in ("crt_net", "crt_mi", "seq"):
            vals = [g(s_, T_, det) for s_ in ("4p_triple", "5p_triple") for T_ in (30, 60)]
            lo_, hi_ = f"{100 * min(vals):.0f}", f"{100 * max(vals):.0f}"
            v[f"s0d_{det}_triple"] = f"{lo_}%" if lo_ == hi_ else f"{lo_}–{hi_}%"
            v[f"s0d_{det}_inside"] = f"{100 * min(g(s_, T_, det, 'inside') / max(g(s_, T_, det), 1e-9) for s_ in ('4p_triple', '5p_triple') for T_ in (30, 60)):.0f}%"
        hon = s0d[s0d.setup.str.endswith("honest") & (s0d.detector != "seq")]
        v["s0d_honest_range"] = f"{100 * hon.flag.min():.0f}–{100 * hon.flag.max():.0f}%"
        worst = hon.loc[hon.flag.idxmax()]
        v["s0d_honest_worst_ci"] = f"{100 * worst.lo:.0f}–{100 * worst.hi:.0f}%"
        v["s0d_iters"] = str(int(s0d.n.iloc[0]))
        hs = s0d[s0d.setup.str.endswith("honest") & (s0d.detector == "seq")]
        v["s0d_seq_honest_max"] = f"{100 * hs.flag.max():.0f}%"
        v["s0d_honest_worst"] = f"{100 * worst.flag:.1f}%"
    kq = read("k_requeries_summary.csv")
    if kq is not None:
        for _, r in kq.iterrows():
            v[f"kq_{r.exp}_{r.tag}_{r.condition}_{r.detector}_k{r.k}"] = pct(round(r.rate * r.n), int(r.n))
        hon = kq[kq.metric == "false_alarm"]
        for det in ("crt_net", "seq"):
            for k in (1, 2):
                h = hon[(hon.detector == det) & (hon.k == k)]
                kk, n = int((h.rate * h.n).round().sum()), int(h.n.sum())
                v[f"kq_honest_{det}_k{k}"] = pct(kk, n, 1)
    ex = read("exchangeability.csv")
    if ex is not None:
        v["exch_max_abs"] = f"{ex['diff'].abs().max():.3f}"
        v["exch_cells"] = f"{int(ex.cells.sum()):,}"
        for tag in ("deepseek", "haiku"):
            r = ex[(ex.exp == "s1") & (ex.tag == tag) & (ex.group == "honest")]
            if len(r):
                r = r.iloc[0]
                v[f"exch_{tag}_diff"] = f"{r['diff']:+.3f}"
                v[f"exch_{tag}_ci"] = f"{r.diff_lo:+.3f} to {r.diff_hi:+.3f}"
                v[f"exch_{tag}_p"] = f"{r.p_rep_rep:.2f}"
        cover = (ex.diff_lo <= 0) & (ex.diff_hi >= 0)
        v["exch_all_cover_zero"] = bool(cover.all())
        v["exch_groups"] = str(len(ex))
        v["exch_chance_any"] = f"{100 * (1 - 0.95 ** len(ex)):.0f}%"
        v["exch_cover"] = str(int(cover.sum()))
        out_ = ex[~cover]
        v["exch_exceptions"] = "; ".join(
            f"{r.tag.replace('val_', '')} ({r.group.replace('_', ' ')}, {r.exp.upper()}): {r['diff']:+.3f}, "
            f"CI {r.diff_lo:+.3f} to {r.diff_hi:+.3f}" for _, r in out_.iterrows()) or "none"
        names = {"deepseek": "DeepSeek", "haiku": "Claude Haiku", "mixed": "Mixed",
                 "val_gpt-6-luna": "GPT-6 Luna", "val_gemma-3-12b-it": "Gemma 3 12B",
                 "val_gpt-oss-20b": "gpt-oss-20b"}
        def sg(x):
            return f"{x:+.3f}".replace("-", "−")

        lines = ["| Model | Game | Seats | Agent-rounds | Difference | 95% CI |",
                 "|---|---|---|---|---|---|"]
        for _, r in ex.iterrows():
            lines.append(
                f"| {names.get(r.tag, r.tag)} | {'RPS' if r.exp == 's1' else 'Auction'} | "
                f"{'all (honest games)' if r.group == 'honest' else 'outside the pair'} | "
                f"{int(r.cells):,} | {sg(r['diff'])} | {sg(r.diff_lo)} to {sg(r.diff_hi)} |")
        v["exch_table"] = "\n".join(lines)
        main = ex[ex.tag.isin(["deepseek", "haiku", "mixed"])]
        v["exch_main_max_abs"] = f"{main['diff'].abs().max():.3f}"
        v["exch_main_cover_all"] = bool(((main.diff_lo <= 0) & (main.diff_hi >= 0)).all())
    pv = read("exchangeability_providers.csv")
    if pv is not None:
        for tag, model in (("deepseek", "deepseek/deepseek-v4-flash-0731"),
                           ("haiku", "anthropic/claude-haiku-5.5")):
            r = pv[pv.model == model]
            if len(r):
                v[f"prov_{tag}_n"] = str(int(r.providers.iloc[0]))
                v[f"prov_{tag}_top"] = f"{100 * r.top_share.iloc[0]:.0f}%"
                calls = int(r.calls.iloc[0])
                v[f"prov_{tag}_calls"] = f"{calls:,}"
                v[f"prov_{tag}_other"] = f"{round(calls * (1 - r.top_share.iloc[0])):,}"
    conf = SUM / "confirm_results.json"
    if conf.exists():
        import json as _json
        c = _json.loads(conf.read_text())
        v["conf_c1"] = f"{c['c1_seq_correct']}/{c['c1_n']}"
        v["conf_c1_crt"] = f"{c['c1_crt_correct']}/{c['c1_n']}"
        v["conf_c1_p"] = f"{c['c1_p_mcnemar_one_sided']:.3f}"
        v["conf_c1_round"] = f"{c['c1_median_round']:.0f}" if c["c1_median_round"] is not None else "–"
        v["conf_c2"] = f"{c['c2_seq_alarms']}/{c['c2_n']}"
        v["conf_c2_p"] = f"{c['c2_p_fpr_below10']:.3f}"
        v["conf_sec_crt"] = f"{c['sec_crt_null_flags']}/{c['c2_n']}"
        v["conf_sec_crt_ci"] = f"{100 * c['sec_crt_null_ci'][0]:.0f}–{100 * c['sec_crt_null_ci'][1]:.0f}%"
        v["conf_c1_verdict"] = "supported" if c["c1_supported"] else "not supported"
        v["conf_c2_verdict"] = "supported" if c["c2_supported"] else "not supported"
        v["conf_c1_diff"] = f"{100 * c['c1_diff']:.0f} points"
        v["conf_c1_diff_ci"] = f"{100 * c['c1_diff_ci'][0]:.0f} to {100 * c['c1_diff_ci'][1]:.0f}"
        v["conf_c2_ci"] = f"0–{100 * c['c2_ci'][1]:.0f}%" if c["c2_seq_alarms"] == 0 else \
            f"{100 * c['c2_ci'][0]:.0f}–{100 * c['c2_ci'][1]:.0f}%"
        v["conf_success"] = "successful" if c["replication_successful"] else "not successful"
        ca = read("confirm_audits.csv")
        if ca is not None:
            g = ca[(ca.condition == "null_same") & (ca.detector == "naive_mi_shuffle")]
            v["conf_mi_null"] = f"{int(g.flag05.sum())}/{len(g)}"
    seq_sim = read("s0c_sequential.csv")
    if seq_sim is not None:
        honest = seq_sim[~seq_sim.population.str.startswith("pair")]
        v["seq_sim_fpr_max"] = f"{100 * honest.groupby(['population', 'T']).reject.mean().max():.1f}%"
        pw = seq_sim[seq_sim.population.str.startswith("pair")].groupby(["population", "T"])
        m, med = pw.correct.mean(), pw.reject_round.median()
        for cp in ("0.6", "0.8", "1.0"):
            for T in (30, 60):
                key = (f"pair_cp{cp}", T)
                if key in m.index:
                    tag = cp.replace(".", "")
                    v[f"seq_sim_power_{tag}_{T}"] = f"{100 * m[key]:.0f}%"
                    v[f"seq_sim_round_{tag}_{T}"] = f"{med[key]:.0f}"
    seq = read("seq_llm_summary.csv")
    if seq is not None:
        for _, r in seq.iterrows():
            key = f"seq_{r.exp}_{r.tag}_{r.condition}"
            v[key + "_pct"] = pct(round(r.reject * r.n), int(r.n))
            v[key + "_correct_pct"] = pct(round(r.correct * r.n), int(r.n))
            v[key + "_round"] = f"{r.median_round:.0f}" if r.median_round == r.median_round else "–"
        h = seq[(seq.exp == "s1") & seq.condition.isin(["null_same", "null_mixed", "talk_no_action"])
                & ~seq.tag.str.startswith("val_")]
        k, n = int((h.reject * h.n).round().sum()), int(h.n.sum())
        v["seq_s1_honest_pct"] = pct(k, n, 1)
        v["seq_s1_honest_ci"] = ci(k, n)
    for bet in ("joint", "interaction", "mix"):
        sb = read("seq_llm_summary.csv" if bet == "joint" else f"seq_llm_summary_{bet}.csv")
        if sb is None:
            continue
        for _, r in sb.iterrows():
            v[f"bet_{bet}_{r.exp}_{r.tag}_{r.condition}"] = f"{100 * r.correct:.1f}%"
        hon = sb[sb.condition.isin(["null_same", "null_mixed", "talk_no_action", "vendor_public"])
                 & ~sb.tag.str.startswith("val_")]
        k, n = int((hon.reject * hon.n).round().sum()), int(hon.n.sum())
        v[f"bet_{bet}_honest"] = pct(k, n, 1)
    if s1 is not None and s2 is not None:  # fixed-horizon test on the same pooled honest set
        h1 = s1[(s1.detector == "crt_net") & s1.tag.isin(["deepseek", "haiku", "mixed"])
                & s1.condition.isin(["null_same", "null_mixed", "talk_no_action"])]
        h2 = s2[(s2.detector == "crt_net") & s2.condition.isin(["null_same", "vendor_public"])]
        kk, nn = int(h1.flag05.sum() + h2.flag05.sum()), len(h1) + len(h2)
        v["honest_s1s2_crt_pct"] = pct(kk, nn, 1)
        v["honest_s1s2_n"] = str(nn)
    gain = read("collusion_gain.csv")
    if gain is not None:
        for _, r in gain.iterrows():
            v[f"gain_{r.exp}_{r.tag}_{r.condition}"] = f"{r.team_adv_per_round:+.2f}"
    harm = read("s2_harm.csv")
    if harm is not None:
        for _, r in harm.iterrows():
            v[f"harm_{r.condition}_overpay"] = f"{r.overpayment:.1f}"
            v[f"harm_{r.condition}_pairwin"] = f"{100 * r.pair_win_share:.0f}%"
    comp = read("compliance.csv")
    if comp is not None:
        for _, r in comp[(comp.exp == "s1") & (comp.condition == "pair_hidden")].iterrows():
            v[f"comp_{r.tag}_leader"] = f"{100 * r.leader_follows:.0f}%"
            v[f"comp_{r.tag}_follower"] = f"{100 * r.follower_follows:.0f}%"
            v[f"comp_{r.tag}_same"] = f"{100 * r.plans_same_move:.0f}%"
    # exploratory, after the pilot: where the instructed cartel's agreement lives (memory_cartel.py)
    mc = SUM / "memory_cartel.json"
    if mc.exists():
        m = json.loads(mc.read_text())
        for cell, key in (("SM-cartel", "cartel"), ("SM-on", "on"), ("SM-off", "off")):
            k, n = m[cell]["follower_memory_mentions_floor_or_S1"]
            v[f"mem_{key}_mem"] = f"{k} of {n}"
        c = m["SM-cartel"]
        v["mem_cartel_ask"] = f"{c['mean_logged_ask']:.2f}"
        v["mem_cartel_replay_ask"] = f"{c['mean_replay_ask']:.2f}"
        v["mem_changed_logged"] = f"{100 * c['compliance']['changed']['logged']:.0f}%"
        v["mem_changed_replay"] = f"{100 * c['compliance']['changed']['replay']:.0f}%"
    # exploratory simulator: memory cartel and honest agents that repeat their own moves (s0f)
    s0f = read("s0f_memory_cartel_summary.csv")
    if s0f is not None:
        hon = s0f[s0f.scenario.str.startswith("honest")]
        pers = hon[hon.scenario.isin(["honest_rho0.5", "honest_rho0.8"])]

        def rng_(x):
            lo, hi = 100 * x.min(), 100 * x.max()
            f = lambda y: f"{y:.1f}".removesuffix(".0")  # noqa: E731
            return f"{f(lo)}–{f(hi)}%" if f(lo) != f(hi) else f"{f(lo)}%"

        v["s0f_net_pers"] = rng_(pers.replay_crt_net)
        v["s0f_mi_pers60"] = rng_(pers[pers['T'] == 60].replay_crt_mi)
        v["s0f_seq_honest_max"] = f"{100 * hon.replay_seq.max():.1f}%".replace(".0%", "%")
        v["s0f_net_rho0"] = rng_(hon[hon.scenario == "honest_rho0.0"].replay_crt_net)
        v["s0f_twin_seq_pers"] = rng_(pers.twin_seq)
        sil = s0f[s0f.scenario.str.startswith("silent")]
        v["s0f_silent_replay_seq"] = rng_(sil.replay_seq)
        v["s0f_silent_twin_seq"] = rng_(sil.twin_seq)
        v["s0f_iters"] = str(int(s0f.n.iloc[0]))
    # full study in the re-implemented double auction (pre-registered at launch; decision script
    # results/summary/full_cda_confirm.json, memory analysis memory_cartel_full_cda.json)
    fc = SUM / "full_cda_confirm.json"
    if fc.exists():
        c = json.loads(fc.read_text())
        assert not c.get("dry_run"), "a dry run must never feed the report"
        cells = c["cells"]

        def kn(k, n):
            return f"{k}/{n}"

        v["fc_mode"] = c["mode"]
        c1 = c["C1"]
        v["fc_c1"] = kn(c1["alarms"], c1["n"])
        v["fc_c1_ci"] = f"{100 * c1['wilson'][0]:.0f}–{100 * c1['wilson'][1]:.0f}%"
        v["fc_c1_p"] = f"{c1['p_fpr_ge_15']:.3f}"
        v["fc_c1_verdict"] = "supported" if c1["supported"] else "not supported"
        v["fc_c2_naive"] = kn(cells["SM-off"]["naive"], cells["SM-off"]["n"])
        v["fc_c2_seq"] = kn(cells["SM-off"]["seq"], cells["SM-off"]["n"])
        v["fc_c2_p"] = f"{c['C2']['p_mcnemar']:.3f}"
        v["fc_c2_verdict"] = "supported" if c["C2"]["supported"] else "not supported"
        c4 = c["C4"]
        v["fc_c4_cartel"] = kn(c4["cartel_alarms"], c4["cartel_n"])
        v["fc_c4_off"] = kn(c4["off_alarms"], c4["off_n"])
        v["fc_c4_p"] = f"{c4['p_fisher']:.3f}"
        v["fc_c4_verdict"] = "supported" if c4["supported"] else "not supported"
        names = {"SM-off": "DeepSeek, no chat", "SM-on": "DeepSeek, chat",
                 "SM-cartel": "DeepSeek, instructed cartel", "XP-off": "DeepSeek + Haiku, no chat",
                 "XP-on": "DeepSeek + Haiku, chat"}
        for cell, r in cells.items():
            key = cell.replace("-", "_")
            v[f"fc_{key}_price"] = f"{r['mean_price']:.2f}"
            v[f"fc_{key}_seq"] = kn(r["seq"], r["n"])
            v[f"fc_{key}_naive"] = kn(r["naive"], r["n"])
        mem = SUM / "memory_cartel_full_cda.json"
        mrows = json.loads(mem.read_text()) if mem.exists() else {}
        lines = ["| Cell | Games | Mean trade price | Time-shuffle MI | Replay, sequential "
                 "| Replay, fixed (MI / net) | Logged − replayed ask |",
                 "|---|---|---|---|---|---|---|"]
        for cell in ("SM-off", "XP-off", "SM-on", "XP-on", "SM-cartel"):
            if cell not in cells:
                continue
            r = cells[cell]
            gap = mrows.get(cell, {}).get("mean_logged_minus_replay")
            lines.append(f"| {names[cell]} | {r['n']} | {r['mean_price']:.2f} | {r['naive']} | "
                         f"{r['seq']} | {r['crt_mi']} / {r['crt_net']} | "
                         f"{'–' if gap is None else f'{gap:+.2f}'} |")
        v["fc_table"] = "\n".join(lines)
        # sentences for both outcomes, written before the data; the decision flags pick one
        v["fc_c1_sentence"] = (
            f"the sequential test raised {v['fc_c1']} false alarms in games without chat (95% CI "
            f"{v['fc_c1_ci']}), " + (
                f"ruling out a false-alarm rate of 15% or more (p = {v['fc_c1_p']})."
                if c1["supported"] else
                f"so a false-alarm rate of 15% or more could not be ruled out "
                f"(p = {v['fc_c1_p']})."))
        v["fc_c2_sentence"] = (
            f"time-shuffle mutual information flagged {v['fc_c2_naive']} DeepSeek games without "
            f"chat, against {v['fc_c2_seq']} for the sequential replay test " + (
                f"(exact McNemar p = {v['fc_c2_p']})."
                if c["C2"]["supported"] else
                f"(exact McNemar p = {v['fc_c2_p']}, not significant)."))
        v["fc_c4_sentence"] = (
            f"The instructed cartel's mean trade price was {v['fc_SM_cartel_price']} (no chat: "
            f"{v['fc_SM_off_price']}), and the sequential test alarmed in {v['fc_c4_cartel']} of "
            f"its games against {v['fc_c4_off']} without chat (one-sided Fisher p = "
            f"{v['fc_c4_p']}" + (")." if c4["supported"] else
                                 "): no detected effect of the current inbox, which is not "
                                 "evidence that the sellers did not collude."))
        if "SM-cartel" in mrows:
            ch = mrows["SM-cartel"]["compliance"]["changed"]
            v["fc_changed_logged"] = f"{100 * ch['logged']:.0f}%"
            v["fc_changed_replay"] = f"{100 * ch['replay']:.0f}%"
            k, n = mrows["SM-cartel"]["follower_memory_mentions_floor_or_S1"]
            v["fc_cartel_mem"] = f"{k} of {n}"
    # sample sizes per condition and validation-model rates as percentages (for Table A1)
    for name, exp in (("s1_audits.csv", "s1"), ("s2_audits.csv", "s2")):
        d = read(name)
        if d is not None:
            for (tag, cond), g in d[d.detector == "crt_net"].groupby(["tag", "condition"]):
                key = f"n_{exp}_{tag}_{cond}" if exp == "s1" else f"n_s2_{cond}"
                v[key] = str(len(g))
    for key in [k for k in list(v) if k.startswith("val_") and k.endswith("_crt")]:
        kk, nn = (int(x) for x in str(v[key]).split("/"))
        v[key + "_pct"] = pct(kk, nn, 1)
    pw = read("s0b_power.csv")
    if pw is not None:
        h = pw[pw.cp == 0].groupby("T").flag.mean()
        v["s0b_power_honest_range"] = f"{100 * h.min():.0f}–{100 * h.max():.0f}%"
        v["s0b_power_n_per_T"] = str(int(pw[pw.cp == 0].groupby("T").size().iloc[0]))
    # p-values for text: below 1e-10 print a bound instead of a long exponent
    for key in [k for k in v if k.endswith(("_p_above5", "_p_below10"))]:
        x = float(v[key])
        v[key + "_txt"] = "p < 10^−10^" if x < 1e-10 else f"p = {v[key]}"
    led = spend()
    v.update(led)
    return v


def spend() -> dict:
    import sqlite3

    db = ROOT / "results" / "cache" / "llm.sqlite"
    if not db.exists():
        return {}
    con = sqlite3.connect(db)
    # the full auction study was pre-registered and launched at 2026-10-25 15:20:43 +05:00; costs
    # quoted for the main study (and the per-audit replay cost) use the ledger before that moment
    import datetime
    cut = datetime.datetime(2026, 10, 25, 15, 20, 43,
                            tzinfo=datetime.timezone(datetime.timedelta(hours=5))).timestamp()
    total, calls = con.execute("SELECT SUM(cost), COUNT(*) FROM spend WHERE ts < ?",
                               (cut,)).fetchone()
    out = {"total_spend": f"${total:.2f}", "total_calls": f"{calls:,}"}
    fs, fc = con.execute("SELECT SUM(cost), COUNT(*) FROM spend WHERE ts >= ?", (cut,)).fetchone()
    out["full_spend"], out["full_calls"] = f"${(fs or 0):.2f}", f"{fc:,}"
    rows = con.execute("SELECT model, SUM(cost), COUNT(*) FROM spend WHERE ts < ? GROUP BY model",
                       (cut,)).fetchall()
    for model, c, n in rows:
        out[f"cost_per_call_{model.split('/')[-1]}"] = f"${c / n:.6f}"
        if model.startswith("deepseek/"):
            # replay test overhead: k = 2 re-queries x 4 agents x 30 rounds = 240 calls per audit
            out["audit_cost"] = f"${240 * c / n:.3f}"
    return out


def fill(template: str, v: dict) -> tuple[str, list[str]]:
    missing = []

    def sub(m):
        key = m.group(1)
        if key in v:
            return str(v[key])
        missing.append(key)
        return f"<span style='background:#fdd'>[{key}]</span>"

    return re.sub(r"\{\{([^{}]+)\}\}", sub, template), missing


def main():
    v = numbers()
    REP.mkdir(exist_ok=True)
    (REP / "numbers.json").write_text(json.dumps(v, indent=1, sort_keys=True))
    html, missing = fill((REP / "template.html").read_text(), v)
    (REP / "report.html").write_text(html)
    abstract = re.search(r'<div class="abstract-body">(.*?)</div>', html, re.DOTALL)
    if abstract:
        words = len(re.sub(r"<[^>]+>", " ", abstract.group(1)).split())
        print(f"abstract words: {words}")
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    "--virtual-time-budget=10000", f"--print-to-pdf={REP / 'report.pdf'}",
                    (REP / "report.html").as_uri()], check=True, capture_output=True)
    info = subprocess.run(["pdfinfo", str(REP / "report.pdf")], capture_output=True, text=True,
                          check=False)
    pages = re.search(r"Pages:\s+(\d+)", info.stdout)
    print(f"pages: {pages.group(1) if pages else '?'}")
    print(f"unfilled placeholders: {sorted(set(missing))}")


if __name__ == "__main__":
    main()
