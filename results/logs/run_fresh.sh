#!/bin/zsh
# Fresh pre-registered runs (docs/fresh_plan.md): A (DeepSeek, pinned, T=60), C (Haiku public chat),
# B (Haiku evasive). Refuses to start while the auction study is running (its stop rule reads the
# shared ledger) or if the plan is not committed. Each run has its own spend cap.
set -e
set -o pipefail
cd "$(dirname "$0")/../.."
LOG=results/logs/fresh.log
grep -q "FULL_CDA_DONE" results/logs/full_cda.log || { echo "auction study not finished"; exit 1; }
[ -z "$(git status --porcelain docs/fresh_plan.md experiments/fresh_analysis.py experiments/s1_llm_rps.py experiments/s2_procurement.py src/collusion/detect/gro.py src/collusion/agents/llm_agent.py src/collusion/detect/departure.py results/logs/run_fresh.sh)" ] || { echo "commit the plan and code first"; exit 1; }
if [ -z "$RESUME" ]; then  # RESUME=1 reruns from the cache (finished games cost nothing)
  for d in s1/fresh_deepseek_pinned s1/fresh_haiku_stego s1/fresh_haiku_evasive s2/fresh_haiku_auction s3/fresh_cartel2; do
    [ -d results/raw/$d ] && { echo "results/raw/$d exists; refusing to overwrite (RESUME=1 to continue)"; exit 1; }
  done
fi
spent() { uv run python -c "import sqlite3;print(round(sqlite3.connect('results/cache/llm.sqlite').execute('select sum(cost) from spend').fetchone()[0],2))"; }
echo "plan commit: $(git log -1 --format='%h %ad' --date=iso -- docs/fresh_plan.md) | spent: \$$(spent)" | tee -a $LOG
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
# D: Haiku procurement auction (vendor conditions), only with at least $15 left on the key
REMAIN=$(uv run python -c "import json,urllib.request;from pathlib import Path;k=next(l.split('=',1)[1].strip().strip(chr(34)) for l in Path('.env').read_text().splitlines() if l.startswith('OPENROUTER_API_KEY'));d=json.load(urllib.request.urlopen(urllib.request.Request('https://openrouter.ai/api/v1/key',headers={'Authorization':'Bearer '+k}),timeout=30))['data'];print(round(d['limit_remaining'],2))")
if [ "$(python3 -c "print($REMAIN >= 15)")" = "True" ]; then
  CAP=$(python3 -c "print(round($(spent) + 9, 2))")
  caffeinate -i uv run python experiments/s2_procurement.py --condition vendor_leak vendor_public null_same --model anthropic/claude-haiku-5.5 --pin-provider "Claude Platform on AWS" --episodes 40 --rounds 30 --seed-offset 90000 --tag fresh_haiku_auction --budget $CAP >> $LOG 2>&1
  echo "D done | spent: \$$(spent)" | tee -a $LOG
else
  echo "D skipped: \$$REMAIN left on the key" | tee -a $LOG
fi
# E: second cartel series in the auction, only with at least $8 left on the key
REMAIN=$(uv run python -c "import json,urllib.request;from pathlib import Path;k=next(l.split('=',1)[1].strip().strip(chr(34)) for l in Path('.env').read_text().splitlines() if l.startswith('OPENROUTER_API_KEY'));d=json.load(urllib.request.urlopen(urllib.request.Request('https://openrouter.ai/api/v1/key',headers={'Authorization':'Bearer '+k}),timeout=30))['data'];print(round(d['limit_remaining'],2))")
if [ "$(python3 -c "print($REMAIN >= 8)")" = "True" ]; then
  CAP=$(python3 -c "print(round($(spent) + 6, 2))")
  caffeinate -i uv run python experiments/s3_cda.py --cell SM-cartel --episodes 30 --hours 30 --seed-offset 100 --tag fresh_cartel2 --budget $CAP --bins v3 >> $LOG 2>&1
  echo "E done | spent: \$$(spent)" | tee -a $LOG
else
  echo "E skipped: \$$REMAIN left on the key" | tee -a $LOG
fi
uv run python experiments/fresh_analysis.py | tee -a $LOG
uv run python - <<'PY' | tee -a $LOG
import sqlite3, time
c = sqlite3.connect("results/cache/llm.sqlite")
t0 = time.time() - 6 * 3600
print("providers in the last 6h:", c.execute("select model, provider, count(*) from spend where ts > ? group by model, provider", (t0,)).fetchall())
PY
echo "FRESH_DONE" | tee -a $LOG
