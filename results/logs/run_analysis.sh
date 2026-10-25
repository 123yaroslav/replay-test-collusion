#!/bin/zsh
# Post-run pipeline: every number and figure in the report is produced here.
set -e
cd "$(dirname "$0")/../.."
export COLLUSION_MAX_CONCURRENCY=40
# external panel on Agrawal et al.'s shipped logs (needs third_party/llm-agent-collusion, see REPRODUCE.md)
if [ -d third_party/llm-agent-collusion ]; then uv run python experiments/agrawal_panel.py > results/logs/analysis_agrawal.txt; fi
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
uv run python experiments/collusion_gain.py > results/logs/analysis_gain.txt
uv run python experiments/s0c_sequential.py > results/logs/analysis_s0c.txt
uv run python experiments/sequential_llm.py > results/logs/analysis_seq_llm.txt
BETTOR=interaction uv run python experiments/sequential_llm.py > results/logs/analysis_seq_llm_interaction.txt
BETTOR=mix uv run python experiments/sequential_llm.py > results/logs/analysis_seq_llm_mix.txt
# frozen judge on every main S1 episode; stronger judge on a subset
uv run python experiments/judge.py --tags deepseek haiku mixed "val_*" --views actions messages --out s1_judge > results/logs/judge_haiku.txt
# stronger judge: already run on 20 episodes per cell (results/summary/s1_judge_strong.csv); rerunning
# would only hit the cache. Command used:
#   judge.py --tags deepseek haiku --conditions null_same pair_hidden talk_no_action --max-per-condition 20
#            --views actions messages --judge openai/gpt-6.1-sol --reasoning low --max-tokens 2000 --out s1_judge_strong
uv run python experiments/exchangeability.py > results/logs/analysis_exchangeability.txt
uv run python experiments/s0d_coalition.py > results/logs/analysis_s0d.txt
uv run python experiments/s0e_evasion.py > results/logs/analysis_s0e.txt
uv run python experiments/k_requeries.py > results/logs/analysis_k_requeries.txt
# pre-registered confirmatory replication (docs/confirmatory_plan.md)
uv run python experiments/analyse_s1.py --tags confirm_haiku --out-prefix confirm > results/logs/analysis_confirm.txt
uv run python experiments/confirm_sequential.py >> results/logs/analysis_confirm.txt
# pilot in the re-implemented Agrawal auction (descriptive; episodes come from experiments/s3_cda.py)
uv run python experiments/analyse_cda.py --tag pilot_cda > results/logs/analysis_pilot_cda.txt
uv run python experiments/analyse_cda.py --tag pilot_cda --rebin v2 > /dev/null
# full pre-registered auction study (data from results/logs/run_full_cda.sh), then exploratory follow-ups
if [ -d results/raw/s3/full_cda ]; then
  uv run python experiments/confirm_cda.py --tag full_cda > results/logs/analysis_full_cda_confirm.txt
  uv run python experiments/analyse_cda.py --tag full_cda > results/logs/analysis_full_cda.txt
  uv run python experiments/memory_cartel.py --tag full_cda > /dev/null
  uv run python experiments/gro_cda.py --tag full_cda > /dev/null
fi
# exploratory theory and post-hoc analyses of the likelihood-ratio bet ($0)
uv run python experiments/s0g_growth.py > results/logs/analysis_s0g.txt
uv run python experiments/gro_sim.py > results/logs/analysis_gro_sim.txt
uv run python experiments/gro_llm.py > results/logs/analysis_gro_llm.txt
uv run python experiments/fresh_predictions.py > /dev/null
uv run python experiments/departure_llm.py > results/logs/analysis_departure_llm.txt
# fresh pre-registered runs (data from results/logs/run_fresh.sh)
uv run python experiments/fresh_analysis.py > results/logs/analysis_fresh.txt
uv run python experiments/make_diagram.py
uv run python experiments/make_figures.py
uv run python experiments/paper_figures.py
uv run python experiments/compliance.py > results/logs/analysis_compliance.txt
uv run python experiments/example_episode.py > results/logs/analysis_example.txt
uv run python experiments/build_report.py
uv run python experiments/render_official.py
uv run python experiments/check_numbers.py
echo ANALYSIS_DONE
