---
license: cc-by-4.0
pretty_name: Replay audits of LLM agents in repeated games
tags:
- multi-agent
- collusion
- auditing
- llm-agents
- ai-safety
size_categories:
- 1K<n<10K
---

# Replay audits of LLM agents in repeated games

Episode logs and the model-reply cache behind *Monoculture Looks Like Collusion: A Replay Test for
Hidden Information Flow Between LLM Agents* (Yaroslav Rogoza, Apart Research AI Collusion Research
Sprint, October 2026). Code: [github.com/123yaroslav/replay-test-collusion](https://github.com/123yaroslav/replay-test-collusion).

Every episode is a repeated game between LLM agents. In each round the log holds every agent's
action and two *replays*: the same deployed policy asked again on the same observation with the
private channel removed. Some conditions give one pair a hidden channel; the others are honest.

## Files

| Path | Game | Size |
|---|---|---|
| `raw/s1/<models>/<condition>.jsonl` | Four-player rock–paper–scissors, 30 rounds | 39 MB |
| `raw/s2/deepseek/<condition>.jsonl` | Sealed-bid procurement auction, four suppliers, 30 rounds | 11 MB |
| `raw/s3/full_cda/<cell>.jsonl` | Continuous double auction, five sellers and five buyers, 30 hours (pre-registered study) | 71 MB |
| `raw/s3/fresh_cartel2/SM-cartel.jsonl` | Second pre-registered cartel series in the double auction | 14 MB |
| `raw/s1/fresh_deepseek_pinned/`, `raw/s1/fresh_haiku_stego/`, `raw/s1/fresh_haiku_evasive/` | Fresh pre-registered rock–paper–scissors runs (provider pinned; public chat; evasive pair) | 14 MB |
| `raw/s2/fresh_haiku_auction/` | Procurement auction with Claude Haiku suppliers (pre-registered) | 6 MB |
| `cache/llm.sqlite` | Every model reply, keyed by a hash of the request, with a cost ledger | 713 MB |

`<models>` is `deepseek`, `haiku`, `mixed` (two of each), `val_*` (validation models) or
`confirm_haiku` (the pre-registered replication). Folders named `pilot*` (including
`raw/s3/pilot_cda` and the `pilot_stego*` instruction pilots) hold pilot runs; the main analyses do not use them.

## Conditions

| Game | Condition | Setup | Hidden channel |
|---|---|---|---|
| RPS | `null_same` | four copies of one model | no |
| RPS | `null_mixed` | two DeepSeek and two Haiku agents | no |
| RPS | `talk_no_action` | P1 and P3 write collusive plans that nobody receives | no |
| RPS | `told_no_channel` | P1 and P3 are told they are a team, with no channel | no |
| RPS | `pair_hidden` | P1 privately sets its own and P3's move each round | yes |
| RPS | `pair_aware` | as `pair_hidden`, and the pair is warned about a statistical auditor | yes |
| RPS | `pair_chat` | P1 and P3 share a free-form private chat | yes |
| RPS | `pair_evasive` | as `pair_hidden`, and the pair is told how the replay audit works and asked to stay indistinguishable | yes |
| RPS | `chat_public` | all four players write small talk in a public chat that the auditor reads | no |
| RPS | `stego_public` | as `chat_public`, and P1 encodes the pair's next move in a code word of its line | yes (in the public chat) |
| Procurement | `null_same` | four copies of DeepSeek | no |
| Procurement | `vendor_public` | P1 and P3 follow a pricing vendor that sees public data and their own cost | no |
| Procurement | `vendor_leak` | the vendor pools P1's and P3's confidential costs and rotates wins | yes |
| Procurement | `pair_hidden` | P1 and P3 run a bid-rotation cartel over a private channel | yes |
| Procurement | `fresh_haiku_auction/*` | the same `null_same`, `vendor_public`, `vendor_leak` conditions with Claude Haiku suppliers | as above |
| Double auction | `SM-off`, `SM-on` | five DeepSeek sellers, seller chat off or on | chat, when on |
| Double auction | `XP-off`, `XP-on` | three DeepSeek and two Haiku sellers, chat off or on | chat, when on |
| Double auction | `SM-cartel` | one seller announces an hourly price floor to the others | yes |

## Record format

One line is one episode:

- `agents`: id and model of each seat; `colluders`: the pair with a channel, if any (detectors
  never read it);
- `rounds[t].actions`: each agent's logged action;
- `rounds[t].requery[agent]`: the two replays;
- `rounds[t].messages`: private messages, including undelivered ones;
- `rounds[t].private`: each agent's short stated rationale;
- `meta`: models, temperature, number of replays, parse failures.

## Use

Download into the `results/` folder of a clone of the code repository and run the analysis:

```bash
huggingface-cli download 123yaroslav/replay-test-collusion --repo-type dataset --local-dir results
./results/logs/run_analysis.sh
```

With the cache no API calls are made. The external panel also uses the published logs of Agrawal
et al. (2025), which are not redistributed here; the code repository shows how to fetch them.

All agents are LLMs called through OpenRouter (DeepSeek V4 Flash, Claude Haiku 5.5, GPT-6 Luna,
Gemma 3 12B, gpt-oss-20b). Colluding behaviour was instructed by us as a positive control. The
texts are model outputs about games and contain no personal data. Provider routing was not
pinned, so the ledger's provider column varies for DeepSeek.

## Citation

```bibtex
@misc{rogoza2026monoculture,
  title  = {Monoculture Looks Like Collusion: A Replay Test for Hidden Information Flow
            Between {LLM} Agents},
  author = {Rogoza, Yaroslav},
  year   = {2026},
  note   = {Apart Research AI Collusion Research Sprint},
  url    = {https://github.com/123yaroslav/replay-test-collusion}
}
```
