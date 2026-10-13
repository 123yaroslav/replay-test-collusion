# Catching colluding LLM agents with information-theoretic plus factors

Working repo for the [Apart Research AI Collusion Research Sprint](https://apartresearch.com)
(Fri 2026-10-23 → Sun 2026-10-25, **submission deadline Sun 23:59 AoE**).

**Question (RQ 1.4, framed by RQ 2.3):** can directional information measures (pairwise net
influence, transfer entropy, conditional mutual information) tell *which* agents collude, and
separate real collusion from the innocent dependence that shared model priors and competitive
reactions also produce, better than action correlation or an LLM judge?

- Why this question, answers to every sprint question: [`docs/answers.md`](docs/answers.md)
- Experimental plan and sprint timeline: [`docs/plan.md`](docs/plan.md)
- Literature notes and starter-repo survey: [`docs/literature/`](docs/literature/)
- Sprint brief (copied from the sprint page): [`docs/brief/`](docs/brief/)
- Pre-sprint work to disclose in the report: [`PRE_SPRINT_LOG.md`](PRE_SPRINT_LOG.md)

## Setup

```bash
uv sync                      # Python 3.12 env with all deps
cp .env.example .env         # add OPENROUTER_API_KEY
uv run pytest -q             # estimators + runner tests
# optional: starter repos (see third_party/README.md)
git clone --depth 1 https://github.com/andyjliu/llm-agent-collusion third_party/llm-agent-collusion
uv run python experiments/pilot_agrawal_te.py   # $0 pilot on Agrawal et al.'s shipped runs
```

## Layout

```
src/collusion/
  llm.py              async OpenRouter client, SQLite cache, hard budget cap
  trajectory.py       Episode/Round schema (JSONL); for_detector() hides ground truth
  runner.py           generic repeated N-player game loop with private channels
  agents/llm_agent.py LLM and scripted agents
  games/              environments
  detect/             detectors (infotheory.py: entropy, MI, CMI, transfer entropy, surrogates)
  adapters/agrawal.py loads the 75 shipped double-auction runs as Episodes
experiments/          one rerunnable script per experiment
results/summary/      small committed outputs; raw/ and cache/ are gitignored
report/               figures and notes for the PDF (written by the team, see CLAUDE.md)
```

## Models

All LLM traffic goes through OpenRouter, one key for every provider. Cheap defaults (prices
per 1M input/output tokens, checked 2026-10-13): `openai/gpt-6-luna` $0.10/$0.50,
`anthropic/claude-haiku-5.5` $0.10/$0.50, `deepseek/deepseek-v4.1-flash` $0.04/$0.62,
`qwen/qwen3.8-flash` $0.15/$0.47. A 25-round, 4-agent episode is roughly 100 calls, about $0.02.
