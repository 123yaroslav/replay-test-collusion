# Reproduce every number in the report

## From the cache ($0)
Needs the data archive: `results/raw/` (episode logs, JSONL) and `results/cache/llm.sqlite` (every LLM reply). They are not in git (too large); the archive link goes in the report's Code and Data section when it is uploaded. The only network step is the clone of Agrawal et al.'s public logs below.

```bash
uv sync
git clone --depth 1 https://github.com/andyjliu/llm-agent-collusion third_party/llm-agent-collusion  # Agrawal et al. logs
./results/logs/run_analysis.sh
```

This one script runs, in order:
- the per-episode audits for S1 and S2;
- the truncation, team-advantage and harm analyses;
- the simulator checks: s0c sequential and s0d coalitions;
- the sequential test with three betting functions;
- the LLM judges (served from the cache);
- the exchangeability diagnostic, the k = 1 re-analysis and the confirmatory replication;
- figures, `report/numbers.json`, the official-template report and the work-summary page;
- `experiments/check_numbers.py`, which recomputes key numbers straight from the CSVs, compares them with the PDF and checks the abstract length.

- **Time:** about 13 minutes (781 s measured) on an Apple-silicon laptop. The CPU-heavy parts are the resampling tests, B = 999 per episode.
- **Cost:** $0 with the cache. A cache miss would call OpenRouter and needs `OPENROUTER_API_KEY` in `.env`.
- **PDF:** exported from the filled `.docx` by Pages on macOS. Elsewhere, `report/official/Apart_report.docx` is produced and can be exported by hand.
- The judges' replies are reproduced exactly from the cache. Temperature-0 replies are not deterministic if re-requested live (report, Appendix C).

## From scratch (about $42 of OpenRouter calls)
```bash
cp .env.example .env                 # add OPENROUTER_API_KEY; the code caps spend at $45
./results/logs/run_main_s1.sh        # rock-paper-scissors: main models, mixed, validation models
./results/logs/run_main_s2.sh        # procurement auction
./results/logs/run_confirm.sh        # pre-registered confirmatory replication (docs/confirmatory_plan.md)
./results/logs/run_analysis.sh
```
Seeds are fixed (disjoint blocks per condition), so a rerun requests the same calls. Provider routing on OpenRouter was not pinned, so live replies can differ from the cached ones.

## Tests
```bash
uv run pytest -q
```
