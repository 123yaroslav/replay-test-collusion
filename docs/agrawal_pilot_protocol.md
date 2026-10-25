# Pilot protocol: replay test in a re-implemented Agrawal et al. double auction

**Purpose: cost and sanity only. No hypothesis is tested; pilot episodes never enter a main result.** The full 2×2 study (`docs/agrawal_live_plan.md`, draft) runs only after the user explicitly approves the extra budget.

## Implementation
- `src/collusion/games/cda.py` re-implements Agrawal et al.'s protocol from their design:
  - one order per agent per hour; a new order replaces the old one; "null" withdraws it;
  - end-of-hour matching of the highest bid with the lowest ask at the midpoint;
  - seller messages read the next hour;
  - each agent keeps its own memory and notebook.
- **The prompts are written by us**, with the same information content and section structure as theirs, not their text. Their code is not executed. Their repository has no licence.
- `experiments/s3_cda.py`:
  - all agents see one frozen view per hour, so there is no same-hour leakage (Codex found such a race in their `run_round`);
  - memory and notebook are snapshotted before the call;
  - k = 2 replays per seller and hour: empty inbox when chat is on, the byte-identical prompt when chat is off; replays are never written back;
  - replays are not truncated (`max_tokens` 1,500 for every call);
  - unparseable replies are coded "invalid", separately from "null".
- Tests (`tests/test_cda.py`, `tests/test_s3_harness.py`) check:
  - matching, order replacement and price bins;
  - chat off: replay prompt identical to the factual one;
  - chat on: replay differs only in the message line;
  - replies to replays are never written to memory;
  - episodes pass through `audit()`.

## Pilot design
- Cells `SM-off` and `SM-on`: 5 DeepSeek V4 Flash sellers (cost 80) and 5 DeepSeek buyers (value 100), temperature 0.7.
- 5 episodes per cell, **15 hours**, k = 2, seeds from 10 000 (disjoint from any main run).
- Tag `pilot_cda`.
- Cost estimate: about $0.8 (`docs/agrawal_live_plan.md`). It runs under the global $45 cap in code, with about $2.7 left at the time of writing.

## What is reported (descriptive)
- Measured tokens and cost per episode against the estimate.
- Parse and invalid rates for factual calls and replays.
- Ask distribution by hour and cell; trades and prices.
- Message content in `SM-on`: do sellers propose price coordination?
- Memory and notebook mentions of an agreement (the P1 keyword list).
- Replay-test p-values: global alarms of sequential and `crt_mi`.
- **Fidelity check against Agrawal et al.'s shipped logs:** prompt length and the distribution of seller asks, base against chat runs. A different model (GPT-4.1) and prompts are expected to shift these; the check is for gross mismatch, such as nobody trading.

## Stop rules
- If the measured cost per episode is above 1.5 × the estimate, stop and report.
- If more than 10% of factual seller calls are invalid, fix the prompt before any main run. A changed prompt is a deviation and is logged.
