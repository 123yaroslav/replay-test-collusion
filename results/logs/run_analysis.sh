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
# stronger judge only if at least $5 of the $45 cap is left (it costs ~$0.01 per call)
LEFT=$(uv run python -c "import sqlite3;print(45-(sqlite3.connect('results/cache/llm.sqlite').execute('select sum(cost) from spend').fetchone()[0] or 0))")
if (( LEFT > 5 )); then
  uv run python experiments/judge.py --tags deepseek haiku --conditions null_same pair_hidden talk_no_action \
     --max-per-condition 20 --views actions messages --judge openai/gpt-6.1-sol --reasoning low --out s1_judge_strong > results/logs/judge_strong.txt
else
  echo "skipping strong judge: only \$$LEFT left" > results/logs/judge_strong.txt
fi
uv run python experiments/make_diagram.py
uv run python experiments/make_figures.py
uv run python experiments/compliance.py > results/logs/analysis_compliance.txt
uv run python experiments/example_episode.py > results/logs/analysis_example.txt
uv run python experiments/build_report.py
uv run python experiments/render_official.py
echo ANALYSIS_DONE
