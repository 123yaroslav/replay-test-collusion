#!/bin/zsh
cd "/Users/yaroslav/Desktop/apart research/AI Collusion Research Sprint"
export COLLUSION_MAX_CONCURRENCY=24
for m in deepseek/deepseek-v4-flash-0731 mistralai/mistral-small-24b-instruct-2501 openai/gpt-6-luna; do
  t="${m##*/}"
  uv run python experiments/s2_procurement.py --model "$m" --condition null_same vendor_public vendor_leak pair_hidden \
     --episodes 3 --rounds 20 --seed-offset 30000 --tag "pilot3_$t" --budget 6 > "results/logs/pilot3_s2_$t.log" 2>&1 &
done
wait
echo PILOT3B_DONE
