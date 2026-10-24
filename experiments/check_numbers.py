"""Independent cross-check before each report commit: recompute key numbers straight from the CSVs
(not through build_report.numbers()), compare them with report/numbers.json, and check that the
rendered value appears in the PDF text.

    uv run python experiments/check_numbers.py
Exits non-zero on any mismatch.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SUM = ROOT / "results" / "summary"
REP = ROOT / "report"


def pct(x: float) -> str:
    s = f"{100 * x:.1f}"
    return (s[:-2] if s.endswith(".0") else s) + "%"


def main():
    v = json.loads((REP / "numbers.json").read_text())
    pdf = subprocess.run(["pdftotext", str(REP / "Apart_report.pdf"), "-"], capture_output=True,
                         text=True, check=True).stdout
    s1 = pd.read_csv(SUM / "s1_audits.csv")
    s2 = pd.read_csv(SUM / "s2_audits.csv")
    checks = []

    def rate(df, tag, cond, det, col="flag05"):
        d = df[(df.tag == tag) & (df.condition == cond) & (df.detector == det)]
        return d[col].mean(), len(d)

    for tag, cond, det, col, key in [
        ("haiku", "null_same", "naive_mi_shuffle", "flag05", "s1_haiku_null_same_naive_mi_shuffle_pct"),
        ("haiku", "null_same", "bonjour_alpha", "flag05", "s1_haiku_null_same_bonjour_alpha_pct"),
        ("haiku", "null_same", "crt_net", "flag05", "s1_haiku_null_same_crt_net_pct"),
        ("deepseek", "null_same", "crt_net", "flag05", "s1_deepseek_null_same_crt_net_pct"),
        ("haiku", "pair_hidden", "crt_net", "correct_pair", "s1_haiku_pair_hidden_crt_net_correct_pct"),
        ("deepseek", "pair_hidden", "crt_net", "correct_pair", "s1_deepseek_pair_hidden_crt_net_correct_pct"),
    ]:
        r, n = rate(s1, tag, cond, det, col)
        checks.append((key, pct(r), n))
    for cond, det, col, key in [
        ("vendor_leak", "crt_net", "correct_pair", "s2_vendor_leak_crt_net_correct_pct"),
        ("vendor_public", "crt_net", "flag05", "s2_vendor_public_crt_net_pct"),
    ]:
        r, n = rate(s2, "deepseek", cond, det, col)
        checks.append((key, pct(r), n))
    # pooled honest S1 replay-test rate (null_same, null_mixed, talk_no_action of the main models)
    hon = s1[s1.tag.isin(["deepseek", "haiku", "mixed"])
             & s1.condition.isin(["null_same", "null_mixed", "talk_no_action"])
             & (s1.detector == "crt_net")]
    checks.append(("s1_honest_n", str(len(hon)), len(hon)))
    checks.append(("s1_honest_crt_pct", f"{100 * hon.flag05.mean():.1f}%", len(hon)))

    bad = 0
    for key, mine, n in checks:
        theirs = str(v.get(key))
        in_pdf = theirs in pdf
        ok = theirs.rstrip("%").rstrip("0").rstrip(".") == mine.rstrip("%").rstrip("0").rstrip(".") \
            or theirs == mine
        flag = "OK " if ok and in_pdf else "BAD"
        bad += flag == "BAD"
        print(f"{flag} {key:48s} csv={mine:>7s} numbers.json={theirs:>7s} n={n:<4d} in_pdf={in_pdf}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
