#!/bin/zsh
# Pilot v2: leader-follower positive control + exact-replay talk null; seeds >= 20000 (excluded)
cd "$(dirname "$0")/../.."
export COLLUSION_MAX_CONCURRENCY=24
for m in deepseek/deepseek-v4-flash-0731 mistralai/mistral-small-24b-instruct-2501 openai/gpt-6-luna anthropic/claude-haiku-5.5 google/gemma-3-12b-it; do
  tag="pilot2_${m##*/}"
  uv run python experiments/s1_llm_rps.py --model "$m" --condition pair_hidden talk_no_action \
     --episodes 3 --rounds 20 --k 2 --seed-offset 20000 --tag "$tag" --budget 4 > "results/logs/${tag}.log" 2>&1 &
done
wait
echo PILOT2_DONE
