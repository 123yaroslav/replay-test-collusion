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
        v["exch_all_cover_zero"] = bool(((ex.diff_lo <= 0) & (ex.diff_hi >= 0)).all())
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
            v[key + "_pct"] = f"{100 * r.reject:.0f}%"
            v[key + "_correct_pct"] = f"{100 * r.correct:.0f}%"
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
    led = spend()
    v.update(led)
    return v


def spend() -> dict:
    import sqlite3

    db = ROOT / "results" / "cache" / "llm.sqlite"
    if not db.exists():
        return {}
    con = sqlite3.connect(db)
    total, calls = con.execute("SELECT SUM(cost), COUNT(*) FROM spend").fetchone()
    out = {"total_spend": f"${total:.2f}", "total_calls": f"{calls:,}"}
    rows = con.execute("SELECT model, SUM(cost), COUNT(*) FROM spend GROUP BY model").fetchall()
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
