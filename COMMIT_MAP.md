# Commit map

The public history was cleaned before release. Internal notes, draft text, review logs, helper scripts and author emails were removed from every commit, commit messages were shortened to describe the change, and commits that then held nothing were dropped. Commit dates are unchanged, and the pre-registered plans in `docs/` are byte-identical to the originals, so the commit hashes quoted inside those plans are the original ones. This table maps each original hash to its hash here.

| Original | Published | Date (UTC+5) | Subject |
|---|---|---|---|
| `98f00dc` | `a5f531b` | 2026-10-13 20:59:22 | Pre-sprint setup: net-influence collusion detector with calibrated nulls |
| `4333131` | `fae61c5` | 2026-10-25 01:04:47 | Sprint: per-episode CRT audit, LLM experiments, pilots, frozen analysis plan |
| `e65cabb` | `3c2a302` | 2026-10-25 01:06:53 | Freeze S2 (procurement auction) analysis plan; coarsened bid statistic |
| `e05b522` | `af99b82` | 2026-10-25 01:19:56 | Robust runs (patient retries on network errors, per-episode failure isolation), report bui |
| `8ed0e66` | `7356b42` | 2026-10-25 01:25:51 | Deviations log; post-hoc replay-deviation statistic (reported, not used for claims) |
| `b1e5089` | `6feb362` | 2026-10-25 02:07:10 | Exploratory anytime-valid sequential replay test (s0c), example episode, official-template |
| `6deaa5c` | `ad6526c` | 2026-10-25 02:38:11 | Full analysis pipeline outputs, official Apart template report (docx -> PDF), fixes, stron |
| `d3f81ab` | `eab66d5` | 2026-10-25 02:52:25 | Report fixes and reference audit: fix sequential pair naming at stop time, verified Cendyn |
| `81ffe26` | `c841bb1` | 2026-10-25 02:59:14 | Pre-register confirmatory replication of the sequential replay test (Haiku, fresh seeds 70 |
| `346cd67` | `6c17eb1` | 2026-10-25 03:05:03 | Confirmatory plan addendum (joint decision rule, seed/sample enforcement, effect-size CI), |
| `3755519` | `a66ab8e` | 2026-10-25 03:06:30 | Disclose pre-sprint replay-test idea and first (non-exact) implementation in Appendix E, M |
| `a391573` | `21979e1` | 2026-10-25 03:13:07 | Full pre-sprint disclosure (S0 shared-prior finding, Agrawal pooled pilot result, runner,  |
| `dc7ce51` | `932b876` | 2026-10-25 03:26:13 | Confirmatory replication results (pre-registered 81ffe26/346cd67): C1 18/20 vs 11/20 (McNe |
| `841b2ac` | `f50ca01` | 2026-10-25 03:42:23 | Final rebuild from one snapshot (all validation models incl. gpt-oss, confirmatory replica |
| `db284e5` | `4610a4b` | 2026-10-25 03:45:49 | Exchangeability: report the one exception (gpt-oss honest) exactly; unified percent format |
| `7a50255` | `6006fc3` | 2026-10-25 03:56:09 | Coalitions of three (s0d): net influence cancels inside a coalition, raw-MI replay CRT and |
| `c1398cd` | dropped (held only removed files) | 2026-10-25 03:57:06 | Summary page: coalition result |
| `0d721ee` | `67122ee` | 2026-10-25 03:59:59 | Abstract cut to 227 words (234 under the strictest count); strict abstract check in render |
| `775c31b` | `760bf78` | 2026-10-25 04:20:49 | Practitioner kit: public collusion.audit.audit(), examples/audit_your_logs.py, README 3-st |
| `fe71a26` | `346d749` | 2026-10-25 04:21:32 | Code and Data: point to the audit() kit and REPRODUCE.md |
| `f44debd` | dropped (held only removed files) | 2026-10-25 04:22:08 | Notes: k=1 result, practitioner kit |
| `6b97780` | `97bb50f` | 2026-10-25 04:33:30 | Plainer abstract and pages 1-2, exactness intuition, blind spot in limitations; release ch |
| `6079f23` | `bd9c14c` | 2026-10-25 04:36:49 | Restore caveats lost in the plain-language rewrite (exchangeability is checked, not establ |
| `62866e2` | dropped (held only removed files) | 2026-10-25 04:37:07 | Progress note |
| `d59653a` | `c52685b` | 2026-10-25 04:41:04 | Pages 1-2 restructure (round 1): At-a-glance box, plain example before formulas, jargon tr |
| `c5c2d47` | `bd90ec6` | 2026-10-25 04:46:00 | Pages 1-2 round 2: plainer abstract (exactness in plain words), box with power and cost sc |
| `009347e` | `1605048` | 2026-10-25 04:51:02 | Pages 1-2 round 3: fixed vs sequential explained, 18/20 scoped (instructed Haiku, fixed-le |
| `ed4f3fc` | dropped (held only removed files) | 2026-10-25 04:52:45 | Reader notes |
| `3fd231b` | `b8069cb` | 2026-10-25 04:54:08 | Final precision fixes; text frozen |
| `ee62e20` | `14b1b23` | 2026-10-25 05:05:05 | Draft (not pre-registered): live replay test in Agrawal et al.'s auction - design, hypothe |
| `2a47ae8` | dropped (held only removed files) | 2026-10-25 05:05:34 | Progress note |
| `39410a9` | `cf2202b` | 2026-10-25 05:14:55 | Pilot protocol (no hypotheses) and re-implementation of Agrawal et al.'s hourly double auc |
| `dba316d` | `00d2527` | 2026-10-25 05:22:39 | DeepSeek pilot in the re-implemented auction (descriptive: $0.58, chat talk but no price r |
| `bd9bb80` | `01c81d0` | 2026-10-25 05:27:44 | Pilot results in the re-implemented auction (16 episodes, $1.45 total): chat talk but no p |
| `f6e572b` | `e1f5fdc` | 2026-10-25 05:31:09 | Auction harness: versioned price bins (v2: $1 bins 84-90), prompt hashes and hourly state  |
| `33565a9` | `35be755` | 2026-10-25 05:40:46 | Pilot in the target environment reported (Appendix C, one sentence in 4.1; descriptive, po |
| `7562049` | dropped (held only removed files) | 2026-10-25 05:41:16 | Progress note |
| `de1d190` | `2164cdf` | 2026-10-25 05:42:22 | Post-hoc pilot bins only in Appendix C, symmetric for both detectors |
| `64de3a0` | `cc392f0` | 2026-10-25 05:51:16 | Report: consistent honest pools, reference order, pre-registration header restored verbati |
| `af4ae79` | dropped (held only removed files) | 2026-10-25 05:51:36 | Progress note |
| `78c5cad` | `ba5fa71` | 2026-10-25 06:00:02 | s0e evasion simulator (exploratory, $0): rotation cartel with evasion eps, marginal-preser |
| `e7df5f3` | `4d55e96` | 2026-10-25 06:03:58 | Evasion-harm frontier in Appendix C, audit-sheet line and limitation |
| `7dccdcc` | `34d41db` | 2026-10-25 06:10:00 | Clarify what a sequential alarm means (single-agent replay departure vs pair dependence);  |
| `348fece` | `397606f` | 2026-10-25 06:13:07 | Sequential alarm = replay departure, named pair = candidate |
| `05ae10f` | `2aec398` | 2026-10-25 06:19:02 | Theory of change, explicit novelty vs Audit the Whisper and policy-query work, contributio |
| `1731942` | `6aacbeb` | 2026-10-25 06:23:50 | Theory of change and novelty wording; compact figures (larger labels at the same height) |
| `5e53a07` | `c91fdd0` | 2026-10-25 06:25:43 | Pilot addendum 2 (before data): instructed-cartel positive control in the re-implemented a |
| `b573cd5` | `7a3a81a` | 2026-10-25 06:26:08 | Pilot addendum 2 text (before any SM-cartel episode); lint fix |
| `de8297f` | `5807d30` | 2026-10-25 06:30:20 | Instructed-cartel pilot results (5 episodes, $0.33): price up, high compliance; replay tes |
| `c64b4cf` | `65a5a64` | 2026-10-25 06:36:19 | Instructed-cartel pilot reported in Appendix C (descriptive); plan revision 2 fixes v3 bin |
| `ea7cafc` | `fcee5e8` | 2026-10-25 06:39:21 | Full auction study ready to run after approval: v3 bins, decision script with seed/size gu |
| `2eff24c` | `32bbd15` | 2026-10-25 06:42:50 | Launch kit hardened: mode-specific C1, pre-registration check at launch, ledger-based stop |
| `f830fb5` | `074b735` | 2026-10-25 07:20:43 | Seed scheme in Appendix C, Methods pointer; runner row in PRE_SPRINT_LOG |
| `b5d59e8` | `cb01eb0` | 2026-10-25 14:50:18 | Exploratory ($0, not in the report yet): where the instructed cartel's agreement lives (pi |
| `d4a8cb4` | `507eab4` | 2026-10-25 15:20:43 | Pre-register the full auction study at launch (mode full; cap $89); Revision 4: persistenc |
| `0c9af36` | `916ed23` | 2026-10-25 15:22:08 | Clarity rewrite: setup with H0 as the channel's direct effect, Propositions 1-2, results b |
| `6aad15c` | `7e77e82` | 2026-10-25 15:32:17 | README rewritten as a paper repository: summary, results, audit quick start, reproduce, la |
| `4e0a0c2` | `ad52216` | 2026-10-25 15:41:33 | Rewrite: setup with typeset math (Word equations via a LaTeX-to-OMML converter), Propositi |
| `86d104f` | `4a7395e` | 2026-10-25 15:44:57 | README: point to the new method figure |
| `a2c96bb` | `982d382` | 2026-10-25 15:51:41 | Rewrite: H0 at realised inputs, pools as tuples, A1 with joint independence, Propositions  |
| `2c83aec` | `5c04141` | 2026-10-25 15:53:27 | Vector figures in the final PDF (swap the 600-dpi images Pages needs for the matching vect |
| `a319598` | `059266d` | 2026-10-25 15:54:04 | README: pass a fixed action space to audit() (needed for the sequential test's guarantee) |
| `72dda12` | `df970a0` | 2026-10-25 15:56:05 | Math precision (filtration in Proposition 1, fixed action coding, N>=3, meaning of a seque |
| `dba9c36` | `63b1f20` | 2026-10-25 15:57:09 | Dataset card links the code repository; report builder reads the full auction study's deci |
| `eb130a0` | `68f6273` | 2026-10-25 15:57:43 | Dataset card rewritten: files, conditions table, record format, use, citation |
| `fbd6717` | `64f78ca` | 2026-10-25 16:10:24 | Report fixes: claims matched to the tables (sequential test's power cost, bid-rotation car |
| `2ab0b58` | `919c976` | 2026-10-25 16:11:15 | Section 5.2: the sequential test keeps its level under A1 (not unconditionally) |
| `419f07c` | `666cf79` | 2026-10-25 16:13:22 | Full-study sentences for both outcomes, selected by the pre-registered decision flags (C1, |
| `612c5ee` | `330c9f9` | 2026-10-25 16:14:08 | Abstract sentence on the full auction study written before its data, selected by the C1 fl |
| `8c342bb` | `b978cde` | 2026-10-25 16:15:52 | Attribution: our re-implementation of Agrawal et al.'s double auction (the brief's target  |
| `3266086` | `a3c6f30` | 2026-10-25 16:42:23 | Exploratory theory and post-hoc analyses (likelihood-ratio bet: exact growth rates, simula |
| `e106cc8` | `86385bf` | 2026-10-25 16:45:15 | Fresh-run pre-registration: Haiku provider pinned too, 60-round prediction qualified, anal |
| `99e0351` | `dbd13cd` | 2026-10-25 16:48:05 | Pre-registered Run C: collusion through a public chat (steganographic code in small talk)  |
| `0b7e87f` | `b10fb36` | 2026-10-25 16:48:37 | Run C pre-registration: honest-chat endpoint with both interpretations, code-compliance ru |
| `d6a386f` | `bd8ecdf` | 2026-10-25 16:49:51 | One-command launcher for the fresh pre-registered runs (waits for the auction study, per-r |
| `5272834` | `fd1d700` | 2026-10-25 16:50:11 | Prediction for the auction cartel cell committed before its data: joint bet weak on floor  |
| `2ede19f` | `e549d44` | 2026-10-25 16:50:51 | Exploratory likelihood-ratio bet on the auction cells (to run after the pre-registered dec |
| `1590784` | `6c7b926` | 2026-10-25 16:51:19 | Reproduction pipeline covers the full auction study, the likelihood-ratio analyses and the |
| `055dad8` | `6e145c3` | 2026-10-25 16:52:12 | Pre-registered Run D: the vendor result on Haiku sellers (H4, H5, predictions from the Dee |
| `dd4d115` | `858d4f7` | 2026-10-25 17:00:01 | Run C after four instruction pilots (excluded): literal code words with two mandatory lead |
| `96dcd78` | `57eeb1d` | 2026-10-25 17:01:07 | Report builder: table of the floor-change mechanism from the exploratory per-hour analysis |
| `6671240` | `299fe88` | 2026-10-25 17:05:47 | Replay co-departure test (pooled exact null) as a pre-registered secondary statistic for t |
| `761c954` | `db7a4a7` | 2026-10-25 17:07:19 | Exploratory post-hoc: replay co-departure test on all existing LLM games (calibrated every |
| `accea5f` | `8b23650` | 2026-10-25 17:07:52 | Report placeholders for the co-departure analysis |
| `026d70e` | `2178c61` | 2026-10-25 17:58:32 | Full pre-registered auction study: C1 and C2 supported (0/60 false alarms; time-shuffle MI |
| `93bf818` | `dcfedfc` | 2026-10-25 17:59:21 | Exploratory: replay-vs-logged gap and per-hour e-process increments by floor-change hour i |
| `f54b844` | `40201ac` | 2026-10-25 17:59:25 | Full auction study inserted into the report from the pre-written templates (C1 sentence in |
| `7532296` | `366eb62` | 2026-10-25 18:00:28 | Exploratory post-hoc co-departure test on the full auction study, with report placeholders |
| `edda5cf` | `876e1b8` | 2026-10-25 18:01:30 | Appendix C.7: failed likelihood-ratio prediction disclosed, exact comparison cell named, p |
| `6ef4202` | `1b10974` | 2026-10-25 18:02:03 | Auction text: p-values as powers of ten, procurement auction named in 5.1, post-hoc label  |
| `bb41c91` | `3af345a` | 2026-10-25 18:02:35 | p-value bound uses the next power of ten above the value |
| `33903cd` | dropped (held only removed files) | 2026-10-25 18:02:57 | Progress note |
| `5437d3e` | `6e1a11d` | 2026-10-25 18:58:47 | Fresh-run launcher: RESUME=1 continues from the cache after an interruption |
| `e8ccb7f` | `da915e2` | 2026-10-25 19:01:12 | Result sentences for the fresh runs A-E written before their data, selected by the pre-reg |
| `2abe2d9` | dropped (held only removed files) | 2026-10-25 19:02:29 | Publication helpers |
| `b53a88f` | dropped (held only removed files) | 2026-10-25 19:03:50 | Draft notes (related work; fresh-run appendix) |
| `aab87b1` | dropped (held only removed files) | 2026-10-25 19:05:45 | Draft notes; note on the mixture inequality proof |
| `e363485` | dropped (held only removed files) | 2026-10-25 19:06:12 | Draft notes |
| `048fc70` | `00b612f` | 2026-10-25 19:10:14 | Theory moved to Appendix C.9 (corollary against the replay null, with the convexity step), |
| `a62b1fa` | `b71bf59` | 2026-10-25 19:16:09 | Report: sequential bet alone carries the anytime guarantee, conclusion scoped to the curre |
| `6d43612` | dropped (held only removed files) | 2026-10-25 19:17:05 | Insertion script for the fresh runs: appendix always, body sentences only where the pre-re |
| `9522305` | `d266c12` | 2026-10-25 19:20:19 | Conclusion scoped to the detection-time result, replication scoped to its positive control |
| `2ea1b23` | `3c3661d` | 2026-10-25 19:21:06 | Abstract trimmed |
| `11b9a80` | dropped (held only removed files) | 2026-10-25 19:27:47 | Notes |
| `6eca2ff` | `e72290a` | 2026-10-25 20:34:04 | Run D: vendor-advice compliance (descriptive, planned) in the analysis and its sentence |
| `1cfbb7c` | `363cdc2` | 2026-10-25 21:04:57 | Fresh pre-registered runs A-E inserted (Appendix C.11 and body sentences chosen by the ver |
| `039d1b4` | `d465e2f` | 2026-10-25 21:05:21 | README: auction study and fresh pre-registered runs in the results summary |
| `1d875fc` | `fed629f` | 2026-10-25 21:05:21 | audit(): warn when the action space is inferred from the log (sequential bet predictabilit |
| `1a438bf` | dropped (held only removed files) | 2026-10-25 21:05:41 | Progress note |
| `8467d0f` | `a3a765d` | 2026-10-25 21:06:02 | Dataset card: full auction study and fresh runs |
| `4a813e2` | `0aea5ab` | 2026-10-25 21:09:23 | Report: corollary stated against the pool law without a finite-k inequality, leader compli |
| `1e786b3` | dropped (held only removed files) | 2026-10-25 21:10:09 | Checklist updated |
| `6e07acf` | dropped (held only removed files) | 2026-10-25 21:10:56 | Check script: allow read-only shell reads explicitly |
| `184a4d2` | `c2d1441` | 2026-10-25 21:12:46 | The flag's interpretation made conditional on A1-A3 |
| `31f0530` | `8a34e79` | 2026-10-25 21:16:20 | Report: abstract reflects the confirmed co-departure series, fixed-horizon version named w |
| `cb11f68` | `07b6c8c` | 2026-10-25 21:18:13 | Report: abstract reflects the confirmed co-departure series, fixed-horizon version named w |
| `4ab6630` | `2a0bcf5` | 2026-10-25 21:20:23 | Body trimmed back to 8 pages |
| `bb12601` | `1168182` | 2026-10-25 21:23:22 | Conclusion states the measured false-alarm rate rather than removal |
| `d7b3a0a` | dropped (held only removed files) | 2026-10-25 21:25:23 | Progress note |
| `9cd1a94` | dropped (held only removed files) | 2026-10-25 21:26:05 | Report: text check |
| `9b09a6b` | dropped (held only removed files) | 2026-10-25 21:26:11 | Summary page rebuilt |
| `22362e4` | `985114b` | 2026-10-25 22:24:44 | Readability pass: judge table, n in captions, short lead-ins, lighter abstract, figure fix |
| `3da62bf` | dropped (held only removed files) | 2026-10-25 22:26:45 | Notes: gap summary |
| `022e1b6` | dropped (held only removed files) | 2026-10-25 22:27:57 | Report: text check |
| `be176c8` | dropped (held only removed files) | 2026-10-25 22:28:20 | Notes after the readability pass |
| `b6121da` | `a4347b9` | 2026-10-26 00:04:37 | Title block: AI assistance listed as co-author; author slot rendered as one centred cell |
| `6d21f41` | `e49af1a` | 2026-10-26 00:19:24 | Plain-language Sections 2-3, live example figure, Table 1 regrouped, appendix numbers move |
| `4b862f8` | `8cbfd9e` | 2026-10-26 00:30:32 | Plain-language body: abstract, intro, setup, results, discussion rewritten; appendix conde |
| `62cc1b4` | `3debbd2` | 2026-10-26 00:35:00 | Report: fixed-horizon labelled, jargon glossed, Figure 1 in words |
| `7b434ed` | `9a9554e` | 2026-10-26 00:37:01 | Figure 2: text fit |
| `343f3a8` | `ca7c81a` | 2026-10-26 00:38:07 | Figure 2: vendor text fits its card |
| `2d94e74` | `5f71531` | 2026-10-26 00:39:23 | Figure 2: vendor text fits its card |
| `77f842b` | `9b296a8` | 2026-10-26 00:40:48 | Figure 2: card heights fit three-line messages |
| `f35164d` | `c676a37` | 2026-10-26 00:54:18 | Deck-style figures: conditions schematic, bars and result cards, green palette; formulas i |
| `c0caa44` | `b7924b0` | 2026-10-26 00:58:38 | Report: flags as evidence at the chosen rate, honest-fleet wording, no colons or semicolon |
| `3abfd07` | `a9efd48` | 2026-10-26 01:00:17 | Report: wording fixes; colons and semicolons removed from body prose and captions |
| `377c7d6` | `c05af2e` | 2026-10-26 01:02:31 | Wording: no possessive plural apostrophes |
| `6242c63` | `f6ce63c` | 2026-10-26 01:04:25 | Results: the pooled honest sample is spelled out |
| `98fefb3` | `b6220ca` | 2026-10-26 01:11:15 | Publication: README, REPRODUCE, pre-sprint log, third_party note and dataset card tidied |
| `9b4c306` | dropped (held only removed files) | 2026-10-26 01:11:33 | publish_public.sh: zsh word splitting for the branch list |
| `041be3d` | dropped (held only removed files) | 2026-10-26 01:11:42 | publish_public.sh: the checked-out branch is already at its remote tip |
| `2265f1b` | dropped (held only removed files) | 2026-10-26 01:12:13 | Publication helper fix |
| `7962745` | dropped (held only removed files) | 2026-10-26 01:12:33 | publish_msgfilter: neutral subject for progress notes |
| `477e1e1` | dropped (held only removed files) | 2026-10-26 01:13:23 | publish_msgfilter: subjects for the publication commits |
| `2940cfa` | dropped (held only removed files) | 2026-10-26 01:14:38 | publish_msgfilter: plain subject for the table-consistency fixes |
| `806c05a` | dropped (held only removed files) | 2026-10-26 01:16:32 | publish_public.sh: skip peeled tag refs when deleting remote refs |
| `3c3ac50` | dropped (held only removed files) | 2026-10-26 01:17:41 | publish_public.sh: skip peeled tag refs when deleting remote refs |
| `9930c70` | dropped (held only removed files) | 2026-10-26 01:21:02 | Notes |
| `f6209d6` | dropped (held only removed files) | 2026-10-26 07:20:14 | Notes |
| `c105420` | `7c81ff9` | 2026-10-26 11:13:32 | Report: plainer body and appendix, proofs in words, redrawn figures, double-auction temper |
| `ccf0f29` | `ba38da3` | 2026-10-26 11:13:32 | README, REPRODUCE and dataset card rewritten; agent-round tables for the dataset viewer; d |
| `21ebb71` | `a785597` | 2026-10-26 11:18:00 | README: results as three short points, table removed |
| `ca85bbb` | `3e7eff9` | 2026-10-26 11:25:24 | Report: appendix tables A5 to A5g numbered without a gap; pilot excluded from hypothesis t |
