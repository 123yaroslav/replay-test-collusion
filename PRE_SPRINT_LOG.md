# Pre-sprint work log (for disclosure)

Sprint rules: "Building on existing work is allowed and encouraged, but you must clearly identify
what is new work done during the sprint. Undisclosed prior work can lead to disqualification."

Everything below was done **before** the sprint opened (Fri 2026-10-23). The report's
Methodology section must state this list, and everything not listed here was done during the
sprint. Add an entry for every pre-sprint change; keep it factual.

| Date | What | Files |
|---|---|---|
| 2026-10-13 | Repository skeleton, uv environment, OpenRouter LLM client with cache and budget cap, trajectory schema | `pyproject.toml`, `src/collusion/llm.py`, `src/collusion/trajectory.py` |
| 2026-10-13 | Generic discrete information-theory estimators (entropy, MI, CMI, transfer entropy, surrogate tests) with unit tests | `src/collusion/detect/infotheory.py`, `tests/test_infotheory.py` |
| 2026-10-13 | Literature notes (AI research agents, web search), research-question briefing, choice of RQ 1.4, project plan | `docs/` |
| 2026-10-13 | Bonjour et al. (2022) γ / net influence Γ / decision rule implementation; family-wise calibrated test | `src/collusion/detect/net_influence.py`, `tests/test_net_influence.py` |
| 2026-10-13 | Scripted RPS simulator with shared-prior decoys; S0 scripted validation run (Bonjour Table 2 replication + calibration comparison) | `src/collusion/games/rps.py`, `experiments/s0_scripted_validation.py`, `results/summary/s0_*` |
| 2026-10-13 | Adapter for Agrawal et al. shipped runs; $0 transfer-entropy pilot on them | `src/collusion/adapters/agrawal.py`, `experiments/pilot_agrawal_te.py`, `results/summary/pilot_agrawal_te.txt` |
| 2026-10-13 | LLM-agent CRT via policy re-querying and the S1 LLM-RPS experiment script (code only, run with fake replies, no LLM data collected) | `src/collusion/detect/crt_llm.py`, `experiments/s1_llm_rps.py` |
| 2026-10-13 | Reading list for the team | `docs/reading.md` |
| 2026-10-13 | Generic N-agent game runner (row added 2026-10-25 to match commit `98f00dc`) | `src/collusion/runner.py`, `tests/test_runner.py` |
| 2026-10-14 | Analysis of four past 1st-place Apart sprint projects (reports, public reviews, repos) distilled into rules and a submission checklist | `docs/winning_rules.md` |
