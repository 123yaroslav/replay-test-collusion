# third_party/

External repositories are cloned here and gitignored (not vendored; their licences are their own).
The analysis needs one of them, the published logs of Agrawal et al.'s LLM double auction, which
`experiments/agrawal_panel.py` reads:

```bash
git clone --depth 1 https://github.com/andyjliu/llm-agent-collusion third_party/llm-agent-collusion
```

Our own double-auction harness (`src/collusion/games/cda.py`) is a re-implementation with our own
prompts; it does not import code from this clone.
