# Confirmatory replication of the sequential replay test (pre-registered)

Written and committed before any episode of this sample existed. The sequential test (§4.6 of the report) was built after the main freeze, and two of its three betting functions were tried after seeing data, so its results are exploratory. This plan fixes everything in advance and tests the first variant on fresh seeds.

## Data
- Model `anthropic/claude-haiku-5.5`, the same code and prompts as the main S1 run (`experiments/s1_llm_rps.py`), 30 rounds, k = 2 re-queries, temperature 1.0, reasoning disabled.
- Conditions and sizes: `pair_hidden` 20 episodes, `null_same` 40 episodes.
- Seeds: `--seed-offset 70000`, giving seeds 70000+ and disjoint from every earlier run (main runs start at 0 and stay below 1,000 within each condition block).
- Output tag `confirm_haiku`. The tag is excluded from all main analyses, so no reported number changes.
- Stopping rule: exactly these 60 episodes. An episode that fails for infrastructure reasons is rerun with the same seed. No additional data.
- Budget: about 21,600 calls at the ledger's Haiku price ($0.000132 per call), roughly $3. The run uses the global $45 cap in code.

## Tests (fixed now)
- **Sequential test:** `sequential_replay_test` with `bettor="joint"` (the first variant, as run before any variant was tried), α = 0.05, KT counts mixed 50/50 with uniform. The pair is named at the round the alarm fires (code as of the commit that adds this plan).
- **Fixed-horizon test:** the pre-registered `crt_net` (B = 999, α = 0.05), on the same episodes.

## Hypotheses
- **C1 (power).** On `pair_hidden`, the sequential test names the true pair in more episodes than the fixed-horizon test. Exact one-sided McNemar test on the paired outcomes (binomial on discordant episodes), α = 0.05.
- **C2 (calibration).** On `null_same`, the sequential test's false-alarm rate is below 10%. Exact one-sided binomial test of H0: FPR ≥ 10%, α = 0.05. With n = 40 this requires zero alarms; one alarm gives p = 0.08 and C2 is then reported as not supported.
- **Secondary, descriptive.** The median round of the alarm on `pair_hidden`, and the fixed-horizon test's false alarms on the fresh `null_same` episodes, with Wilson intervals.

## Reporting
Both outcomes are reported in the report (§4.6 and Appendix C) whichever way they go, with this file's commit hash. Analysis: `experiments/analyse_s1.py --tags confirm_haiku --out-prefix confirm`, then `experiments/confirm_sequential.py`.
