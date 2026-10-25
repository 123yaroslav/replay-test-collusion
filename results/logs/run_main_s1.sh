#!/bin/zsh
# MAIN S1 sample (frozen in docs/analysis_plan.md). Seeds from 0, disjoint blocks per condition.
# 3 processes x 40 concurrent calls (450 concurrent connections caused connection errors).
# All calls are cached, so restarts are free and resume where they stopped.
cd "$(dirname "$0")/../.."
export COLLUSION_MAX_CONCURRENCY=40
DS=deepseek/deepseek-v4-flash-0731; HK=anthropic/claude-haiku-5.5
run() { uv run python experiments/s1_llm_rps.py --rounds 30 --k 2 --budget 45 "$@"; }
( run --model $DS --tag deepseek --condition pair_hidden --episodes 40
  run --model $DS --tag deepseek --condition talk_no_action --episodes 60
  run --model $DS --tag deepseek --condition told_no_channel --episodes 40
  run --model $DS --tag deepseek --condition null_same --episodes 150
  run --model $DS --tag deepseek --condition pair_aware --episodes 30
  run --model $DS --tag deepseek --condition pair_chat --episodes 30 ) > results/logs/main_s1_deepseek.log 2>&1 &
( run --model $HK --tag haiku --condition pair_hidden --episodes 40
  run --model $HK --tag haiku --condition talk_no_action --episodes 40
  run --model $HK --tag haiku --condition told_no_channel --episodes 30
  run --model $HK --tag haiku --condition null_same --episodes 150 ) > results/logs/main_s1_haiku.log 2>&1 &
( run --model $DS --model-b $HK --tag mixed --condition null_mixed --episodes 150
  for m in openai/gpt-6-luna google/gemma-3-12b-it openai/gpt-oss-20b; do
    run --model $m --tag "val_${m##*/}" --condition pair_hidden --episodes 20
    run --model $m --tag "val_${m##*/}" --condition null_same --episodes 40
  done ) > results/logs/main_s1_other.log 2>&1 &
wait
echo MAIN_S1_DONE
