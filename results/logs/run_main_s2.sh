#!/bin/zsh
cd "$(dirname "$0")/../.."
export COLLUSION_MAX_CONCURRENCY=32
DS=deepseek/deepseek-v4-flash-0731
run() { uv run python experiments/s2_procurement.py --model $DS --tag deepseek --rounds 30 --k 2 --budget 45 "$@"; }
run --condition vendor_leak --episodes 40
run --condition pair_hidden --episodes 40
run --condition vendor_public --episodes 60
run --condition null_same --episodes 100
echo MAIN_S2_DONE
