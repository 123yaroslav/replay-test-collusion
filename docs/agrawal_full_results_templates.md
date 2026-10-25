# Text templates for the full auction study (written before its data)

Insert exactly one variant of each paragraph into Appendix C after the run, and at most one sentence in §4.1. Numbers come from `results/summary/full_cda_confirm.json` through placeholders added to `build_report.py`. Do not rewrite the text around the result; if neither variant fits, report the numbers plainly.

## C1, calibration (pooled no-chat cells)
- **Supported:** "In the full pre-registered study in our re-implementation of the organisers' auction (plan `<commit>`), the sequential replay test raised {{c1_alarms}} alarms in {{c1_n}} no-chat episodes (Wilson CI {{c1_ci}}), rejecting a false-alarm rate of 15% or more (p = {{c1_p}}). This does not by itself establish a rate of 5% or less."
- **Not supported:** "In the full pre-registered study …, the sequential replay test raised {{c1_alarms}} alarms in {{c1_n}} no-chat episodes (Wilson CI {{c1_ci}}), so a false-alarm rate of 15% or more could not be ruled out (p = {{c1_p}})."

## C2, monoculture in the target environment
- **Supported:** "On the same no-chat DeepSeek episodes the time-shuffle MI screen flagged {{c2_naive}} and the sequential replay test {{c2_seq}} (exact McNemar p = {{c2_p}})."
- **Not supported:** "On the same no-chat DeepSeek episodes the time-shuffle MI screen flagged {{c2_naive}} and the sequential replay test {{c2_seq}}; the difference was not significant (exact McNemar p = {{c2_p}})."

## C3, prices (descriptive)
"Mean trade prices: {{c3_off}} without chat, {{c3_on}} with uninstructed chat and {{c3_cartel}} for the instructed cartel."

## C4, detection of the instructed cartel
- **Supported:** "The sequential test alarmed in {{c4_cartel}} of the instructed-cartel episodes against {{c4_off}} without chat (one-sided Fisher p = {{c4_p}}). Scope: an instructed positive control in a re-implementation with our own prompts."
- **Not supported:** "The sequential test alarmed in {{c4_cartel}} of the instructed-cartel episodes against {{c4_off}} without chat (one-sided Fisher p = {{c4_p}}), so detection of the instructed cartel in this environment was not shown. The likely reasons are named in the pilot paragraph (bins, the leader's own message, compliance); we report the result as it is."

## One sentence for §4.1 (only if C1 and C4 point the same way)
- Both supported: "In a pre-registered re-implementation of the organisers' auction, the replay test kept its false-alarm rate and detected an instructed cartel (Appendix C)."
- Otherwise: no sentence in the body; Appendix C only.
