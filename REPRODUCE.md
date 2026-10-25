# Reproduce every number in the report

## From the cache ($0)
Needs the data release: `results/raw/` (episode logs, JSONL) and `results/cache/llm.sqlite` (every LLM reply). They are not in git (too large) and come from the Hugging Face dataset [123yaroslav/replay-test-collusion](https://huggingface.co/datasets/123yaroslav/replay-test-collusion) (CC-BY-4.0). The only other network step is the clone of Agrawal et al.'s public logs below.

```bash
uv sync
huggingface-cli download 123yaroslav/replay-test-collusion --repo-type dataset --local-dir results
git clone --depth 1 https://github.com/andyjliu/llm-agent-collusion third_party/llm-agent-collusion  # Agrawal et al. logs
./results/logs/run_analysis.sh
```

This one script runs, in order:
- the per-episode audits for S1 and S2;
- the truncation, team-advantage and harm analyses;
- the simulator checks: s0c sequential, s0d coalitions, s0e evasion, s0g growth rates;
- the sequential test with three betting functions;
- the LLM judges (served from the cache);
- the exchangeability diagnostic, the k = 1 re-analysis and the confirmatory replication;
- the pre-registered double-auction study (`confirm_cda.py`) and its exploratory follow-ups (memory, likelihood-ratio bet, co-departure);
- the fresh pre-registered runs A-E (`fresh_analysis.py`);
- figures, `report/numbers.json` and the official-template report;
- `experiments/check_numbers.py`, which recomputes key numbers straight from the CSVs, compares them with the PDF and checks the abstract length.

- **Time:** about 13 minutes (781 s measured) on an Apple-silicon laptop. The CPU-heavy parts are the resampling tests, B = 999 per episode.
- **Cost:** $0 with the cache. A cache miss would call OpenRouter and needs `OPENROUTER_API_KEY` in `.env`.
- **PDF:** exported from the filled `.docx` by Pages on macOS. Elsewhere, `report/official/Apart_report.docx` is produced and can be exported by hand.
- The judges' replies are reproduced exactly from the cache. Temperature-0 replies are not deterministic if re-requested live (report, Appendix C).

## From scratch (OpenRouter calls; each launcher states its cap)
```bash
cp .env.example .env                 # add OPENROUTER_API_KEY
./results/logs/run_main_s1.sh        # rock-paper-scissors: main models, mixed, validation models
./results/logs/run_main_s2.sh        # procurement auction
./results/logs/run_confirm.sh        # pre-registered confirmatory replication (docs/confirmatory_plan.md)
./results/logs/run_full_cda.sh       # pre-registered double-auction study (docs/agrawal_live_plan.md)
./results/logs/run_fresh.sh          # fresh pre-registered runs A-E (docs/fresh_plan.md)
./results/logs/run_analysis.sh
```
Seeds are fixed (disjoint blocks per condition), so a rerun requests the same calls. Provider routing on OpenRouter was not pinned, so live replies can differ from the cached ones.

## Tests
```bash
uv run pytest -q
```
