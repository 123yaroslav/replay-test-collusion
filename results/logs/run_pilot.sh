#!/bin/zsh
# Pilot: 7 model families x 3 conditions x 3 episodes x 20 rounds, seeds >= 10000 (excluded from main analysis)
cd "/Users/yaroslav/Desktop/apart research/AI Collusion Research Sprint"
export COLLUSION_MAX_CONCURRENCY=24
for m in qwen/qwen3.7-flash deepseek/deepseek-v4-flash-0731 openai/gpt-6-luna anthropic/claude-haiku-5.5 openai/gpt-oss-20b mistralai/mistral-small-24b-instruct-2501 google/gemma-3-12b-it; do
  tag="pilot_${m##*/}"
  uv run python experiments/s1_llm_rps.py --model "$m" --condition null_same pair_hidden talk_no_action \
     --episodes 3 --rounds 20 --k 2 --seed-offset 10000 --tag "$tag" --budget 3 > "results/logs/${tag}.log" 2>&1 &
done
wait
echo PILOT_DONE
