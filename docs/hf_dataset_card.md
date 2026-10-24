---
license: cc-by-4.0
pretty_name: Replay audit of LLM agent collusion (Apart AI Collusion Research Sprint 2026)
tags:
- multi-agent
- collusion
- ai-safety
- llm-agents
- auditing
size_categories:
- 1K<n<10K
---

# Replay audit of LLM agent collusion

Episode logs and the LLM response cache behind the report *Monoculture Looks Like Collusion: A Replay Test for Hidden Information Flow Between LLM Agents* (Apart AI Collusion Research Sprint, October 2026). Code: [repository link].

## Contents

| Path | Size | What |
|---|---|---|
| `results/raw/s1/<tag>/<condition>.jsonl` | 39 MB | Four-player rock-paper-scissors, 30 rounds per episode. Tags are the models (`deepseek`, `haiku`, `mixed`), the validation models (`val_*`), the confirmatory replication (`confirm_haiku`) and the pilots (`pilot*`, excluded from all analyses). |
| `results/raw/s2/deepseek/<condition>.jsonl` | 11 MB | Procurement auction, 30 rounds per episode. |
| `results/cache/llm.sqlite` | 389 MB | Every LLM reply (about 492k), keyed by a SHA-256 hash of the request, plus a spend ledger (time, model, provider, tokens, cost). It lets the code rebuild every number at $0. |

One line is one episode in the `collusion.trajectory.Episode` schema:
- `agents`: id, model, role;
- `colluders`: ground truth, which detectors never see;
- `rounds[t].actions`: logged action per agent;
- `rounds[t].requery[agent]`: k = 2 replays, the same policy asked again on the same observation with the private channel removed;
- `rounds[t].messages`: private messages, including undelivered ones in `talk_no_action`;
- `rounds[t].private`: each agent's short rationale;
- `meta`: models, temperature, k, parse failures.

Conditions: `null_same`, `null_mixed`, `talk_no_action`, `told_no_channel` (honest or no channel used); `pair_hidden`, `pair_aware`, `pair_chat` (a pair with a hidden channel); `vendor_public`, `vendor_leak` (auction pricing vendor). Section 3 of the report gives the full design.

## Reproduce
Put both folders in a clone of the code repository and run `./results/logs/run_analysis.sh` (about 13 minutes, $0). `REPRODUCE.md` in the repository gives the details.

## Not included
- Agrawal et al.'s double-auction logs, used for the external panel, are not redistributed: their repository has no licence. `REPRODUCE.md` shows how to clone them.
- Talk transcripts from the sprint.

## Notes
- All agents are LLMs (DeepSeek V4 Flash, Claude Haiku 5.5, GPT-6 Luna, Gemma 3 12B, gpt-oss-20b via OpenRouter). The colluding behaviour is instructed by us as a positive control.
- The texts are model outputs about games and contain no personal data.
- OpenRouter provider routing was not pinned, so the `provider` column in the ledger varies for DeepSeek.

## Citation
[Author], Monoculture Looks Like Collusion: A Replay Test for Hidden Information Flow Between LLM Agents. Apart AI Collusion Research Sprint, 2026.
