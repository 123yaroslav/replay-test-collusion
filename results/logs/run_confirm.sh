#!/bin/zsh
# CONFIRMATORY sample, plan in docs/confirmatory_plan.md (committed before this run).
cd "/Users/yaroslav/Desktop/apart research/AI Collusion Research Sprint"
export COLLUSION_MAX_CONCURRENCY=40
HK=anthropic/claude-haiku-5.5
run() { uv run python experiments/s1_llm_rps.py --rounds 30 --k 2 --budget 45 --seed-offset 70000 --tag confirm_haiku "$@"; }
run --model $HK --condition pair_hidden --episodes 20
run --model $HK --condition null_same --episodes 40
