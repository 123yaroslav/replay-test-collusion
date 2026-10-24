# Monoculture looks like collusion: a replay test for hidden information flow between LLM agents

Code and data for our submission to the [Apart Research AI Collusion Research Sprint](https://apartresearch.com/sprints/ai-collusion-research-sprint-2026-10-23-to-2026-10-25)
(Oct 23–25, 2026).

**Question.** Given logs of several LLM agents and query access to their deployed policies, can
an auditor tell *which pair* exchanged non-public information, without accusing honest agents
that merely run the same model?

**Method.** For every round, the auditor re-queries each agent on exactly what it saw, with the
private channel removed. A pair is flagged only if their real moves depend on each other more
than their own replays do. This is a conditional randomization test (CRT) with an exact
per-cell null (logged action pooled with its re-queries), one audit per episode, family-wise
over pairs.

## What was done before the sprint (pre-existing) vs during the sprint

The sprint rules require pre-existing work to be marked. The full dated list is in
[`PRE_SPRINT_LOG.md`](PRE_SPRINT_LOG.md); the git history shows the boundary.

| Pre-existing (before Oct 23; commit `98f00dc` on `main`) | Sprint work (Oct 24–26; branch `sprint`) |
|---|---|
| Repo skeleton, OpenRouter client with cache and per-process cap, trajectory schema | Global spend ledger, reasoning control, seeds, 429 handling (`llm.py`) |
| Information-theory estimators and tests (`detect/infotheory.py`) | In-round CRT re-queries, undelivered talk, vendor messages (`runner.py`) |
| Bonjour et al. γ/Γ, Bonjour rule, family-wise resampling test (`detect/net_influence.py`) | Pooled exchangeable CRT and per-episode audit (`detect/crt_llm.py`) |
| Scripted RPS simulator, S0 replication of Bonjour Table 2 (`games/rps.py`, `experiments/s0_scripted_validation.py`) | S0b validity and power with estimated references (`experiments/s0b_crt_validity.py`) |
| Agrawal et al. adapter and pooled TE pilot (`adapters/agrawal.py`, `experiments/pilot_agrawal_te.py`) | Per-run Agrawal panel (`experiments/agrawal_panel.py`) |
| Replay-test idea and first implementation: CRT via post-hoc LLM policy re-querying (k = 8, reference distribution, not pooled with the logged action) and the first LLM RPS script; both run only with fake replies | All LLM experiments: S1 RPS, S2 procurement auction, LLM judge, analysis, figures |
| Literature notes, question selection, plan (`docs/answers.md`, `docs/plan.md`, `docs/literature/`) | Pre-registered analysis plan (`docs/analysis_plan.md`, frozen in commits `4333131` and `e65cabb`), report |

## Reproduce

```bash
uv sync && cp .env.example .env          # add OPENROUTER_API_KEY
uv run pytest -q

# $0: simulator validity/power, external panel on Agrawal et al. logs
uv run python experiments/s0b_crt_validity.py
git clone --depth 1 https://github.com/andyjliu/llm-agent-collusion third_party/llm-agent-collusion
uv run python experiments/agrawal_panel.py

# LLM experiments (exact commands and sample sizes: results/logs/run_main_s1.sh, run_main_s2.sh)
./results/logs/run_main_s1.sh
./results/logs/run_main_s2.sh
uv run python experiments/analyse_s1.py --exp s1
uv run python experiments/analyse_s1.py --exp s2
uv run python experiments/judge.py --tags deepseek haiku mixed --views actions messages --out s1_judge
uv run python experiments/make_figures.py
```

Every LLM call is cached in `results/cache/llm.sqlite`, so with the cache a rerun costs $0 and is
bit-identical. Episode logs (JSONL) are released separately (see the report's Code and Data
section).

## Layout

```
src/collusion/
  llm.py               OpenRouter client: cache, global spend ledger, reasoning control
  runner.py            repeated N-player game loop: private channels, in-round CRT re-queries
  trajectory.py        Episode/Round schema; for_detector() hides labels and condition
  agents/llm_agent.py  LLM agent (same procedure for logged move and re-query)
  games/rps.py         4-player rock-paper-scissors (Bonjour et al. 2022) + scripted simulator
  games/procurement.py sealed-bid procurement auction with pricing-vendor arms
  detect/              net influence, information theory, per-episode CRT audit
  analysis.py          Wilson intervals, binomial tests, AUROC
experiments/           one rerunnable script per experiment
results/summary/       small outputs used by the report (raw logs and cache are gitignored)
docs/analysis_plan.md  pre-registered analysis plan
```
