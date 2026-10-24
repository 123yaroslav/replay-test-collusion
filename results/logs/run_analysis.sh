#!/bin/zsh
# Post-run pipeline: every number and figure in the report is produced here.
set -e
cd "/Users/yaroslav/Desktop/apart research/AI Collusion Research Sprint"
export COLLUSION_MAX_CONCURRENCY=40
uv run python experiments/analyse_s1.py --exp s1 > results/logs/analysis_s1.txt
uv run python experiments/analyse_s1.py --exp s2 > results/logs/analysis_s2.txt
uv run python experiments/analyse_s1.py --exp s2 --full --out-prefix s2_full > results/logs/analysis_s2_full.txt
for T in 10 20; do
  uv run python experiments/analyse_s1.py --exp s1 --truncate $T --conditions pair_hidden --tags deepseek haiku --out-prefix trunc_s1_T$T > /dev/null
  uv run python experiments/analyse_s1.py --exp s2 --truncate $T --conditions vendor_leak --tags deepseek --out-prefix trunc_s2_T$T > /dev/null
done
# T = 30 points come from the full audits
uv run python -c "
import pandas as pd
a=pd.read_csv('results/summary/s1_audits.csv'); a[a.condition=='pair_hidden'].to_csv('results/summary/trunc_s1_T30_audits.csv',index=False)
b=pd.read_csv('results/summary/s2_audits.csv'); b[b.condition=='vendor_leak'].to_csv('results/summary/trunc_s2_T30_audits.csv',index=False)"
uv run python experiments/bonjour_pooled.py > results/logs/analysis_bonjour_pooled.txt
uv run python experiments/harm_and_pairs.py > results/logs/analysis_harm_pairs.txt
# frozen judge on every main S1 episode; stronger judge on a subset
uv run python experiments/judge.py --tags deepseek haiku mixed "val_*" --views actions messages --out s1_judge > results/logs/judge_haiku.txt
uv run python experiments/judge.py --tags deepseek haiku --conditions null_same pair_hidden talk_no_action \
   --max-per-condition 40 --views actions messages --judge openai/gpt-6.1-sol --reasoning low --out s1_judge_strong > results/logs/judge_strong.txt
uv run python experiments/make_figures.py
uv run python experiments/build_report.py
echo ANALYSIS_DONE
