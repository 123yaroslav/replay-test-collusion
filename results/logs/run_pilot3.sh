#!/bin/zsh
# Pilot v3 (seeds >= 30000, excluded): clarified leader prompt (RPS) + first auction pilot
cd "/Users/yaroslav/Desktop/apart research/AI Collusion Research Sprint"
export COLLUSION_MAX_CONCURRENCY=24
for m in deepseek/deepseek-v4-flash-0731 mistralai/mistral-small-24b-instruct-2501 anthropic/claude-haiku-5.5; do
  t="${m##*/}"
  uv run python experiments/s1_llm_rps.py --model "$m" --condition pair_hidden --episodes 3 --rounds 20 \
     --seed-offset 30000 --tag "pilot3_$t" --budget 5 > "results/logs/pilot3_$t.log" 2>&1 &
done
for m in deepseek/deepseek-v4-flash-0731 mistralai/mistral-small-24b-instruct-2501; do
  t="${m##*/}"
  uv run python experiments/s2_procurement.py --model "$m" --condition null_same vendor_public vendor_leak pair_hidden \
     --episodes 3 --rounds 20 --seed-offset 30000 --tag "pilot3_$t" --budget 5 > "results/logs/pilot3_s2_$t.log" 2>&1 &
done
wait
echo PILOT3_DONE
