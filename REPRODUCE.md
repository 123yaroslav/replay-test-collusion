# Reproduce every number in the report

## From the released data, at no cost

The analysis reads two folders that are too large for git. `results/raw/` holds the episode logs and `results/cache/llm.sqlite` holds every model reply. Both come from the Hugging Face dataset [123yaroslav/replay-test-collusion](https://huggingface.co/datasets/123yaroslav/replay-test-collusion) (CC-BY-4.0). The only other download is the public logs of Agrawal et al.

```bash
uv sync
uvx --from huggingface_hub hf download 123yaroslav/replay-test-collusion --repo-type dataset --local-dir results
git clone --depth 1 https://github.com/andyjliu/llm-agent-collusion third_party/llm-agent-collusion  # Agrawal et al. logs
./results/logs/run_analysis.sh
```

`run_analysis.sh` runs these steps in order.

1. The per-game audits of the rock-paper-scissors games and the procurement auction.
2. The truncation, team-advantage and harm analyses.
3. The simulator checks of the sequential test, coalitions, evasion and growth rates.
4. The sequential test with three betting rules.
5. The LLM judges, served from the cache.
6. The exchangeability diagnostic, the re-analysis with one replay and the pre-registered replication.
7. The pre-registered double-auction study (`confirm_cda.py`) and its exploratory follow-ups on memory, the likelihood-ratio bet and co-departure.
8. The fresh pre-registered runs A to E (`fresh_analysis.py`).
9. The figures, `report/numbers.json` and the report in the official template.
10. `experiments/check_numbers.py`, which recomputes key numbers from the result tables, compares them with the PDF and checks the abstract length.

| | |
|---|---|
| Time | About 13 minutes (781 s measured) on an Apple-silicon laptop. Most of it is the resampling tests, 999 null games per episode |
| Cost | Nothing while every reply is in the cache. A cache miss calls OpenRouter and needs `OPENROUTER_API_KEY` in `.env` |
| PDF | Pages on macOS exports the filled `.docx`. Elsewhere the script still writes `report/official/Apart_report.docx`, which can be exported by hand |
| LLM judges | Their replies come from the cache and reproduce exactly. Asked again live, a temperature-0 judge can answer differently (report, Appendix D, Table A9) |

The agent-round tables of the dataset viewer are rebuilt from `results/raw/` with `uv run --with pyarrow python experiments/export_agent_rounds.py`.

## From scratch, with new model calls

Each launcher sets its own budget cap.

```bash
cp .env.example .env                 # add OPENROUTER_API_KEY
./results/logs/run_main_s1.sh        # rock-paper-scissors, main models, mixed groups, validation models
./results/logs/run_main_s2.sh        # procurement auction
./results/logs/run_confirm.sh        # pre-registered replication (docs/confirmatory_plan.md)
./results/logs/run_full_cda.sh full  # pre-registered double-auction study, all five conditions (docs/agrawal_live_plan.md)
./results/logs/run_fresh.sh          # fresh pre-registered runs A to E (docs/fresh_plan.md)
./results/logs/run_analysis.sh
```

Seeds are fixed, with a separate block for each condition, so a rerun sends the same requests. Provider routing on OpenRouter was not pinned in the main runs, so live replies can differ from the cached ones. The whole sprint cost about $87 in API calls, pilots and judges included.

## Tests

```bash
uv run pytest -q
```
