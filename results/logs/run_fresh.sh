#!/bin/zsh
# Fresh pre-registered runs (docs/fresh_plan.md): A (DeepSeek, pinned, T=60), C (Haiku public chat),
# B (Haiku evasive). Refuses to start while the auction study is running (its stop rule reads the
# shared ledger) or if the plan is not committed. Each run has its own spend cap.
set -e
set -o pipefail
cd "$(dirname "$0")/../.."
LOG=results/logs/fresh.log
grep -q "FULL_CDA_DONE" results/logs/full_cda.log || { echo "auction study not finished"; exit 1; }
[ -z "$(git status --porcelain docs/fresh_plan.md experiments/fresh_analysis.py experiments/s1_llm_rps.py src/collusion/detect/gro.py src/collusion/agents/llm_agent.py)" ] || { echo "commit the plan and code first"; exit 1; }
for d in fresh_deepseek_pinned fresh_haiku_stego fresh_haiku_evasive; do
  [ -d results/raw/s1/$d ] && { echo "results/raw/s1/$d exists; refusing to overwrite"; exit 1; }
done
spent() { uv run python -c "import sqlite3;print(round(sqlite3.connect('results/cache/llm.sqlite').execute('select sum(cost) from spend').fetchone()[0],2))"; }
echo "plan commit: $(git log -1 --format='%h %ad' --date=iso -- docs/fresh_plan.md) | spent: \$$(spent)" | tee $LOG
export COLLUSION_MAX_CONCURRENCY=40
# A: DeepSeek, provider pinned, T = 60
CAP=$(python3 -c "print(round($(spent) + 9, 2))")
caffeinate -i uv run python experiments/s1_llm_rps.py --condition pair_hidden --model deepseek/deepseek-v4-flash-0731 --pin-provider Baidu --episodes 40 --rounds 60 --seed-offset 90000 --tag fresh_deepseek_pinned --budget $CAP >> $LOG 2>&1
caffeinate -i uv run python experiments/s1_llm_rps.py --condition null_same --model deepseek/deepseek-v4-flash-0731 --pin-provider Baidu --episodes 80 --rounds 60 --seed-offset 90000 --tag fresh_deepseek_pinned --budget $CAP >> $LOG 2>&1
echo "A done | spent: \$$(spent)" | tee -a $LOG
# C: Haiku, public chat (honest small talk and steganographic pair)
CAP=$(python3 -c "print(round($(spent) + 7, 2))")
caffeinate -i uv run python experiments/s1_llm_rps.py --condition chat_public stego_public --model anthropic/claude-haiku-5.5 --pin-provider "Claude Platform on AWS" --episodes 40 --rounds 30 --seed-offset 90000 --tag fresh_haiku_stego --budget $CAP >> $LOG 2>&1
echo "C done | spent: \$$(spent)" | tee -a $LOG
# B: Haiku, evasive pair
CAP=$(python3 -c "print(round($(spent) + 3, 2))")
caffeinate -i uv run python experiments/s1_llm_rps.py --condition pair_evasive --model anthropic/claude-haiku-5.5 --pin-provider "Claude Platform on AWS" --episodes 40 --rounds 30 --seed-offset 90000 --tag fresh_haiku_evasive --budget $CAP >> $LOG 2>&1
echo "B done | spent: \$$(spent)" | tee -a $LOG
uv run python experiments/fresh_analysis.py | tee -a $LOG
uv run python - <<'PY' | tee -a $LOG
import sqlite3, time
c = sqlite3.connect("results/cache/llm.sqlite")
t0 = time.time() - 6 * 3600
print("providers in the last 6h:", c.execute("select model, provider, count(*) from spend where ts > ? group by model, provider", (t0,)).fetchall())
PY
echo "FRESH_DONE" | tee -a $LOG
