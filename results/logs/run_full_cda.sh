#!/bin/zsh
# Full study in the re-implemented Agrawal et al. double auction (docs/agrawal_live_plan.md,
# Revision and Revision 2). Run only after the user approves the budget.
#   ./results/logs/run_full_cda.sh ds     # SM-off, SM-on, SM-cartel: about $17, about 45 min
#   ./results/logs/run_full_cda.sh full   # + XP-off, XP-on (2 Haiku sellers): about $34, about 80 min
# CAP=<dollars> sets the global spend cap passed to the runner (default: current spend + estimate).
# If a cell ends with fewer than 30 episodes (provider errors), rerun just that cell with the same
# s3_cda.py command; completed calls come from the cache, so the rerun costs only the missing ones.
set -e
set -o pipefail
cd "$(dirname "$0")/../.."
MODE=${1:-ds}
LOG=results/logs/full_cda.log
# 1) plan and code must be committed before any data
if ! grep -q "Status: PRE-REGISTERED at launch" docs/agrawal_live_plan.md; then
  echo "record the approval first: set the plan's status line to PRE-REGISTERED at launch and commit"; exit 1
fi
if [ -n "$(git status --porcelain docs/agrawal_live_plan.md src/collusion/games/cda.py experiments/s3_cda.py experiments/confirm_cda.py experiments/analyse_cda.py results/logs/run_full_cda.sh docs/agrawal_full_results_templates.md src/collusion/audit.py src/collusion/detect)" ]; then
  echo "plan or code has uncommitted changes: commit them before collecting data"; exit 1
fi
if [ -d results/raw/s3/full_cda ]; then echo "results/raw/s3/full_cda exists; refusing to overwrite"; exit 1; fi
PLAN=$(git log -1 --format='%h %ad' --date=iso -- docs/agrawal_live_plan.md)
SPENT=$(uv run python -c "import sqlite3;print(round(sqlite3.connect('results/cache/llm.sqlite').execute('select sum(cost) from spend').fetchone()[0],2))")
EST=$([ "$MODE" = full ] && echo 34 || echo 17)
CAP=${CAP:-$(python3 -c "print(round($SPENT + $EST, 2))")}
echo "plan commit: $PLAN | spent so far: \$$SPENT | cap: \$$CAP | mode: $MODE" | tee $LOG
CELLS=(SM-off SM-on SM-cartel)
[ "$MODE" = full ] && CELLS+=(XP-off XP-on)
export COLLUSION_MAX_CONCURRENCY=40
for cell in $CELLS; do
  BEFORE=$(uv run python -c "import sqlite3;print(sqlite3.connect('results/cache/llm.sqlite').execute('select sum(cost) from spend').fetchone()[0])")
  caffeinate -i uv run python experiments/s3_cda.py --cell $cell --episodes 30 --hours 30 \
      --seed-offset 0 --tag full_cda --budget $CAP --bins v3 >> $LOG 2>&1
  # 2) stop rule: actual cost per episode from the spend ledger, above 1.5 x the estimate
  #    (DeepSeek cells $0.156, mixed cells $0.236 per 30-hour episode), or fewer than 30 episodes
  AFTER=$(uv run python -c "import sqlite3;print(sqlite3.connect('results/cache/llm.sqlite').execute('select sum(cost) from spend').fetchone()[0])")
  N=$(wc -l < results/raw/s3/full_cda/$cell.jsonl | tr -d ' ')
  EST_EP=$([[ $cell == XP* ]] && echo 0.236 || echo 0.156)
  python3 -c "import sys; per=($AFTER-$BEFORE)/max($N,1); print(f'$cell: {$N} episodes, cost/episode \${per:.3f} (estimate \$$EST_EP)'); sys.exit(1 if per>1.5*$EST_EP or $N<30 else 0)" | tee -a $LOG || { echo "stop rule hit after $cell (see $LOG)"; exit 1; }
done
# 3) decision script (fixed before the data) and descriptives
uv run python experiments/confirm_cda.py --tag full_cda | tee -a $LOG
uv run python experiments/analyse_cda.py --tag full_cda > results/logs/analysis_full_cda.txt
echo "FULL_CDA_DONE" | tee -a $LOG
