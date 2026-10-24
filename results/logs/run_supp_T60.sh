#!/bin/zsh
# EXPLORATORY supplement (not pre-registered): longer logs, T = 60, to place real-LLM points on the
# detectability frontier beyond T = 30. Separate tags so the frozen main analysis is unaffected.
cd "/Users/yaroslav/Desktop/apart research/AI Collusion Research Sprint"
export COLLUSION_MAX_CONCURRENCY=40
HK=anthropic/claude-haiku-5.5; DS=deepseek/deepseek-v4-flash-0731
run() { uv run python experiments/s1_llm_rps.py --rounds 60 --k 2 --budget 45 --seed-offset 50000 "$@"; }
( run --model $HK --tag T60_haiku --condition pair_hidden --episodes 20
  run --model $HK --tag T60_haiku --condition null_same --episodes 20 ) > results/logs/supp_T60_haiku.log 2>&1 &
( run --model $DS --tag T60_deepseek --condition pair_hidden --episodes 20
  run --model $DS --tag T60_deepseek --condition null_same --episodes 20 ) > results/logs/supp_T60_deepseek.log 2>&1 &
wait
echo SUPP_DONE
