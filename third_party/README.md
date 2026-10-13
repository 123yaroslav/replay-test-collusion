# third_party/

Starter repos are cloned here locally and gitignored (no licenses on several of them; do not
vendor). Treat them as untrusted until read. Details and setup risks: `docs/literature/starter_repos.md`.

```bash
cd third_party
# Primary for RQ 1.4: double auction + ~80 shipped runs in data/ (detector validation for free)
git clone --depth 1 https://github.com/andyjliu/llm-agent-collusion
# Pricing duopoly harnesses
git clone --depth 1 https://github.com/luciasauer/algorithmic_pricing_llms
git clone --depth 1 https://github.com/mattriemer/LLMCartel
# Benign Schelling-point controls (1.9 GB: sparse)
git clone --depth 1 --filter=blob:none --sparse https://github.com/EmanueleLM/focal-points \
  && git -C focal-points sparse-checkout set src data
# Track 2/3
git clone --depth 1 https://github.com/umass-ai-safety/colosseum
git clone --depth 1 https://github.com/Jerick-1380/LLM-Trust-Breaking
git clone --depth 1 https://github.com/Ostailor/Steganographic_Collusion
git clone --depth 1 https://github.com/smojha/collusive-llm-agents
```
