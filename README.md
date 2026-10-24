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
| Scripted RPS simulator, S0 replication of Bonjour Table 2, and the S0 finding that only the (non-pooled) CRT keeps the nominal false-alarm rate on shared-prior decoys (`games/rps.py`, `experiments/s0_scripted_validation.py`) | S0b validity and power with estimated references (`experiments/s0b_crt_validity.py`) |
| Agrawal et al. adapter and pooled TE/MI pilot, which already found significant dependence (p = 0.005) in the no-chat condition (`adapters/agrawal.py`, `experiments/pilot_agrawal_te.py`) | Per-run Agrawal panel (`experiments/agrawal_panel.py`) |
| Replay-test idea and first implementation: CRT via post-hoc LLM policy re-querying (k = 8, reference distribution, not pooled with the logged action) and the first LLM RPS script; both run only with fake replies | All LLM experiments: S1 RPS, S2 procurement auction, LLM judge, analysis, figures |
| Generic N-agent game runner; research briefing and answers to every sprint question, literature notes, reading list, analysis of past winners with a submission checklist, question selection, plan (`docs/answers.md`, `docs/plan.md`, `docs/literature/`, `docs/reading.md`, `docs/winning_rules.md`) | Pre-registered analysis plan (`docs/analysis_plan.md`, frozen in commits `4333131` and `e65cabb`), report |

## Audit your own agents in 3 steps

1. **Log replays.** In every round, keep each agent's action. Then ask the same deployed policy
   again, k times, on the identical observation with the audited channel removed (same prompt
   byte for byte, minus the inbox and the agent's own outgoing private message). Store the
   answers under `requery`. We used k = 2; for the fixed-horizon test k = 1 lost little in our
   data at half the cost, while the sequential test benefits from k = 2.
   ```json
   {"episode_id": "run-1", "rounds": [
     {"actions": {"A": "rock", "B": "paper", "C": "rock"},
      "requery": {"A": ["rock", "scissors"], "B": ["paper", "paper"], "C": ["rock", "paper"]}}]}
   ```
2. **Run the audit.**
   ```bash
   uv run python examples/audit_your_logs.py my_logs.jsonl   # no argument: a toy log with a planted pair
   ```
   or in Python: `from collusion.audit import audit; audit("my_logs.jsonl", statistic="net")`.
   Use `statistic="mi"` if a coalition of three or more agents is possible (net influence cancels
   inside a coalition), and `coarsen={...}` to bin many-valued actions such as prices.
3. **Read the result.** `flag=True` means the named pair's actions depend on each other more than
   their own replays, given what you logged, plausibly allow at level α (default 5%). It is a
   screening signal and a plus factor, not proof of intent. `seq_alarm_round` is the round at
   which the anytime-valid version would have stopped. Coordination written into the agents'
   shared instructions is invisible to the test by design, so pair it with a policy audit.

## Reproduce

All numbers, figures and the report come from one script and the released cache, at $0:
`./results/logs/run_analysis.sh`. Details, timing and the from-scratch route: [`REPRODUCE.md`](REPRODUCE.md).

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
