#!/bin/zsh
# Full study in the re-implemented Agrawal et al. double auction (docs/agrawal_live_plan.md,
# Revision and Revision 2). Run only after the user approves the budget.
#   ./results/logs/run_full_cda.sh ds     # SM-off, SM-on, SM-cartel: about $17, about 45 min
#   ./results/logs/run_full_cda.sh full   # + XP-off, XP-on (2 Haiku sellers): about $34, about 80 min
# CAP=<dollars> sets the global spend cap passed to the runner (default: current spend + estimate).
# If a cell ends with fewer than 30 episodes (provider errors), rerun just that cell with the same
# s3_cda.py command; completed calls come from the cache, so the rerun costs only the missing ones.
set -e
cd "$(dirname "$0")/../.."
MODE=${1:-ds}
LOG=results/logs/full_cda.log
# 1) plan and code must be committed before any data
if [ -n "$(git status --porcelain docs/agrawal_live_plan.md src/collusion/games/cda.py experiments/s3_cda.py experiments/confirm_cda.py experiments/analyse_cda.py)" ]; then
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
  caffeinate -i uv run python experiments/s3_cda.py --cell $cell --episodes 30 --hours 30 \
      --seed-offset 0 --tag full_cda --budget $CAP --bins v3 >> $LOG 2>&1
  # 2) stop rule: cost per episode above 1.5 x the estimate (DeepSeek $0.156, mixed $0.236)
  uv run python - "$cell" <<'PY' >> $LOG 2>&1 || { echo "stop rule hit after $cell (see $LOG)"; exit 1; }
import json, sys
from pathlib import Path
cell = sys.argv[1]
price = {"deepseek/deepseek-v4-flash-0731": (0.065e-6, 0.272e-6), "anthropic/claude-haiku-5.5": (0.100e-6, 0.500e-6)}
eps = [json.loads(l) for l in Path(f"results/raw/s3/full_cda/{cell}.jsonl").read_text().splitlines()]
cost = []
for ep in eps:
    # buyers and most sellers are DeepSeek; Haiku sellers are priced conservatively at Haiku rates
    pin, pout = price["anthropic/claude-haiku-5.5" if cell.startswith("XP") else "deepseek/deepseek-v4-flash-0731"]
    cost.append(sum(ti * pin + to * pout for _, ti, to, _ in ep["meta"]["usage"].values()))
est = 0.236 if cell.startswith("XP") else 0.156
per = sum(cost) / max(len(cost), 1)
print(f"{cell}: {len(eps)} episodes, cost/episode ${per:.3f} (estimate ${est:.3f})")
sys.exit(1 if per > 1.5 * est or len(eps) < 30 else 0)
PY
done
# 3) decision script (fixed before the data) and descriptives
uv run python experiments/confirm_cda.py --tag full_cda | tee -a $LOG
uv run python experiments/analyse_cda.py --tag full_cda > results/logs/analysis_full_cda.txt
echo "FULL_CDA_DONE" | tee -a $LOG
