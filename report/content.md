% TITLE: Monoculture Looks Like Collusion: A Replay Test for Hidden Information Flow Between LLM Agents
% AUTHOR: Yaroslav Rogoza, with Claude (Anthropic) | Independent researcher
% ABSTRACT
Collusion audits ask whether agents' actions depend on each other. Agents built on one model act alike without communicating, so such audits accuse honest fleets. Among four honest Claude Haiku agents, mutual information flags {{s1_haiku_null_same_naive_mi_shuffle_pct}} of games. We propose a replay test. Each round the auditor asks every deployed agent again what it would do on the same input with the suspected channel removed, and flags a pair only if its logged actions depend on each other more than its replays do. If replays are served like the originals, the null is exact within each round, and a sequential version is valid at any stopping time. On {{s1_honest_n}} honest LLM games the fixed-horizon version flags {{s1_honest_crt_pct}}. In a procurement auction it names the true pair in {{s2_vendor_leak_crt_net_correct_pct}} of games when a shared pricing vendor pools confidential costs, and almost never flags a vendor that uses only public data. A pre-registered replication confirmed the sequential version. {{fc_abstract_sentence}} A cartel in that auction escaped the pre-registered bet. A statistic chosen afterwards caught it, because the colluders left their own replays in the same hours. A fresh pre-registered series confirmed this. The test detects what the current channel adds. Coordination already stored in instructions or memory escapes it. We release the code and the data.
% END

## 1. Introduction
Firms and labs now run many LLM agents side by side, such as pricing bots in one market, bidders in one procurement system and coding agents in one repository. An auditor often cannot see the channels between them. As S. Motwani put it at this sprint, "what's left to monitor is still the actions of the agents themselves". The standard audit therefore asks whether one agent's actions predict another's, with a dependence measure such as the net influence of Bonjour et al. [1] or mutual information [2].

This audit fails when the agents share a model. Four honest copies of Claude Haiku play rock-paper-scissors. They favour the same moves, so a mutual-information test accuses two of them. Ask the two again on the same inputs and their replays are just as alike, because the likeness comes from the model. If instead one player secretly tells the other what to play, replays asked without that message stop tracking each other. Antitrust law calls the first case conscious parallelism and treats it as lawful. Only an agreement is unlawful [4, 5]. A test that cannot tell the two apart accuses honest fleets that run on one vendor's model [3].

We make the shared model part of the null hypothesis (Figure 1). In every round the auditor asks each agent twice more what it would do given exactly what it saw, minus the audited channel. If the channel adds nothing, the real move is one more draw from the same distribution as its replays, so mixing real and replayed moves shows what honest dependence looks like, shared model included. This is a conditional randomization test [6], a test that compares the logged data with data redrawn under the null, except that the redraws come from the deployed agent itself instead of a model of it.

![**Figure 1. The replay test.** (1) An agent acts on its logged view and a hidden channel. (2) The auditor queries the same policy again on the logged view alone. (3) Without a channel effect, the logged move is a uniform member of its pool. (4) Redraws from the pools give a p-value for the dependence statistic S. the sequential test's evidence raises an alarm with a false-alarm rate below the chosen level however long the audit runs, under the assumptions of Section 3.](figures/p_fig1_method.png){6.3}

Our contributions are four. We measure the confound. Dependence tests accuse honest same-model agents in our games and in the published double-auction logs of Agrawal et al. [7] (Section 5.1). We propose the replay test, exact within each round and, in its sequential form, valid at any stopping time (Section 3). On honest LLM games its false alarms stay near the target, and it finds channels that change behaviour, including a vendor that pools confidential data, and a pre-registered replication confirms the sequential version (Sections 5.2 and 5.3). We map its boundary, which covers coordination held in instructions or memory, weak channels, and honest agents that repeat their own moves (Section 5.4). Together these answer sprint questions 1.4 and 2.3.

## 2. Related Work
**Dependence-based detection.** Bonjour et al. [1] define pairwise net influence and name colluding pairs in small games with a fixed threshold, assuming two colluders and agents that do not share priors. Audit the Whisper [2] calibrates transcript statistics against honest runs replayed under fresh seeds. Its null is a population of other runs, not the audited agent on its own input, and the flag attaches to a run rather than a pair. Xu et al. [3] show that two deployments of one model carry correlated bid residuals, and note that dependence cannot tell a conspiracy from one firm running several agents. Anwar et al. [8] define steganography as the utility a receiver gains over a sentinel without the code, which is a measure, not a test.

**Interventions on messages and policies.** In emergent-communication RL, Lowe et al. [9] measure positive listening by substituting a message and watching the listener's action distribution change. Our replay is a related intervention on a logged call. The channel is removed and the deployed policy resampled. Kaur et al. [10] carry the idea to LLM agents that exchange hidden states, scoring the divergence between the receiver's actions under the suspect latent and under a matched control. The control is assumed available, the threshold is an empirical percentile, and no error rate is controlled. Colosseum [11] measures a known coalition's advantage, Eschenbaum and Meylahn [12] read collusion off frozen pricing policies, and Hartline et al. [13] audit one seller's regret. Chain-of-thought monitoring misses collusion in pricing agents [14], per-action defences did not stop colluding agents [15], and stronger monitors need a cross-agent view [16] or activations [17].

**Definition used.** Of the 139 definitions in the organisers' survey [18], we follow those that require communication [17], but we do not count observed public actions as communication. Collusion here is coordination *enabled by non-public information exchanged between agents*, directly or through a third party such as a vendor. Coordination that each policy produces from public information alone is conscious parallelism.

**The gap.** The intervention is old [9, 10] and the null is that of the conditional randomization test [6]. What is new is sampling that null from the deployed agent itself, which makes the per-round test exact and its sequential version anytime-valid. Neither Audit the Whisper nor Kaur et al. gives a per-round error guarantee. Ours rests on one condition. Weights, system prompt, decoding settings and hidden state must be identical for the logged call and its replays.

## 3. Problem Setup and Method
**Setting.** \(N\) agents play \(T\) rounds of one game (an episode). Each round every agent picks an action from a finite menu fixed in advance. An agent's input has two parts. The *view* (V) is everything the auditor logs and can replay, such as the prompt, the public history, tool outputs and the agent's own memory. The *channel* (C) is the input under audit, for example an inbox, a shared memory or a vendor's message. The agent's deployed policy, a model with its system prompt, samples an action from both. The auditor sees every view and every action and may query each policy on any input. It cannot read the channel, but it knows which part of the input carries it. A *replay* is the policy queried on the view with the channel's field removed.

**Null hypothesis.** The channel adds nothing to what is logged. With it or without it, the agent picks each action with the same probability. Writing Pr for the probability that the deployed agent picks a given action,
$$ H_0:\quad \Pr(\text{action} \mid \text{view}, \text{channel}) = \Pr(\text{action} \mid \text{view}) \quad \text{for every agent and every round,}
and within a round the agents act independently given everything logged so far. \(H_0\) is about the direct effect of the current channel. Memory an agent wrote itself is part of the view, so an agreement made over the channel and then stored in memory can satisfy \(H_0\) in later rounds. A rejection means one of three things. The channel changed an agent's actions (by its content or its mere presence), or the agents share randomness the log does not record (a common sampling seed, say), or A1 failed.

**Assumptions.**
- *A1, exchangeable serving.* Each replay is an independent draw from what the deployed policy produces on the channel-free input, independent of every other call in the round. This needs one model snapshot, fixed sampling settings and no response caching (Appendix C checks it). Replays at temperature 0 would copy the logged move and leave the test without power.
- *A2, known channel.* The auditor can remove exactly the channel and nothing else, as when it is a separate field of the prompt.
- *A3, query access.* The auditor can call each deployed policy on logged observations.

**Replays and pools.** Each round the auditor draws \(k\) replays of every agent (\(k = 2\) here) and forms the agent's *pool*: the logged action together with its replays, repeated values counting separately. If an agent logged rock and its replays were paper and rock, then under \(H_0\) the logged move is equally likely to be any of the three entries. Figure 2 shows one such round from the logs.

![**Figure 2. One audited round and one whole game, taken from the pooled-data-vendor logs.** (a) In round 2 the vendor tells P1 and P3, confidentially, who should win. each seller's logged bid is set beside two replays of the same prompt without the vendor's message. P1 and P3 bid outside their pools, P2 and P4 do not. (b) Over the 30 rounds of that game, the two sellers served by the vendor leave their pools in most rounds, the others rarely. the tests of this section turn such departures into a p-value or an alarm.](figures/p_fig2_example.png)

**Proposition 1 (exact within a round).** Suppose \(H_0\) and A1–A3 hold. Condition on everything from earlier rounds, on the current round's views and on the current round's pools as unordered collections. Agents move simultaneously, so no view contains an action from its own round. Then each agent's logged action is a uniformly random entry of its pool, independently across agents. *Proof idea.* Under \(H_0\) the logged action and its replays are independent draws from one distribution, so the pool's entries are exchangeable and, given the unordered pool, each is equally likely to be the logged one [19–21]. The full proof is in Appendix C.9.

**Fixed-horizon test.** The pair statistic is *net influence* [1]. It is the mutual information between two agents' same-round actions over the episode, minus the largest mutual information between the second agent and any third agent, which removes a common cause such as a shared opponent (so \(N \ge 3\)). A pair must pass in both directions. The game statistic is the largest such value over all pairs, so one test per game covers every pair without further correction. Appendix C.9 states the formulas. To get a p-value, the auditor builds 999 null datasets, each by redrawing every action at random from its pool, and recomputes the statistic on each. The p-value is the number of null datasets whose statistic reaches the observed one, plus one, divided by one thousand. The game is flagged when the p-value is at most the chosen level, and the pair with the largest statistic is named. Proposition 1 holds round by round but not for redrawing all rounds at once, because later views contain earlier actions. This version therefore has no finite-sample guarantee over a whole game, and we measure its calibration instead (Sections 5.2 and 5.4).

**Sequential test.** The sequential version bets, round by round, that a pair's logged joint move is more typical of the pair than joint moves drawn from the two pools. Before each round it estimates, from the pair's logged joint moves in earlier rounds, how likely each joint move is (counts plus a half, mixed with the uniform distribution, so early rounds bet almost uniformly). Call this estimate q. The round's payoff compares the logged joint move with the joint moves the two pools can form (with two replays each, nine of them):
$$ \text{payoff}_t = \frac{q(\text{logged joint move})}{\text{average of } q \text{ over the nine pool joint moves}}, \qquad \text{evidence}_T = \text{average over pairs of } \prod_{t \le T} \text{payoff}_t ,
where the product multiplies the pair's payoffs from round 1 to round T. Under the null the logged joint move is one of the nine at random, so the expected payoff is 1 and the evidence stays flat. A pair that keeps playing joint moves its pools do not reproduce earns payoffs above 1 round after round. An alarm is raised at the first round in which the evidence reaches 1 divided by the chosen level (20 at the 5% level). The estimate q depends only on earlier rounds and is positive, so every payoff is positive. The two tests look for different things. Net influence looks for dependence beyond what a third agent explains, the bet for any joint pattern the pools do not reproduce.

**Proposition 2 (valid at any stopping time).** Under \(H_0\) and A1–A3 the evidence starts at 1 and is a fair bet (a nonnegative martingale). Given the past, its expected value after the next round equals its current value. So the chance that it ever reaches 20, in an audit of any length, is at most 5%, also when later views contain earlier actions. Appendix C.9 proves it, gives the fastest betting rule, and shows that a channel which stays undetected cannot shift payoffs by much.

**Two layers.** *Detection* asks whether the channel changed an agent's behaviour, using the sequential bet (the joint bet above or the likelihood-ratio bet of Appendix C.9), or a per-agent count of rounds in which the logged move differs from every replay (Appendix C.10). *Attribution* asks with whom, using the net-influence pair or the pair that departs together most often. Only the sequential bet carries the anytime guarantee. The other statistics are exact per round and calibrated empirically over a game, and attribution names a candidate for investigation, not a culprit.

:::algo Algorithm 1. Replay audit of one episode
**Input:** deployed policies, logged views and actions, \(k = 2\) replays, the level, and 999 null redraws.
1. Each round, query every policy \(k\) times on its logged view with the channel removed. Pool the logged action with its replays.
2. Sequential test. Update the evidence after each round. At the first round it reaches 1 divided by the level, raise an alarm and record the pair with the most evidence.
3. Fixed-horizon test. Compute net influence from the logged actions, redraw every action from its pool 999 times, and compute \(p\).
4. Report detection (an alarm, or an agent that departs from its replays) separately from attribution (the top pair).
**Output:** alarm round (or none), \(p\)-value and candidate pair. Cost: \(k\) extra calls per agent and round, about {{audit_cost}} per 30-round audit of four agents at DeepSeek prices.
:::

## 4. Experimental Setup
**Games.** Rock-paper-scissors is the four-player iterated game of Bonjour et al. [1], with seats P1 to P4, 30 rounds and a public history. In the procurement auction four suppliers draw private costs of 50, 60 or 70 each round and submit sealed bids between 60 and 100. The lowest bid wins and earns its bid minus its cost, and all bids are published. A bid of 90 or more, the cover-bid range, counts as high. The published logs are the 75 double-auction runs of Agrawal et al. [7]. Their policies cannot be queried again, so only the baselines run on them.

**Agents.** Each seat is an LLM that returns a short rationale, an action and an optional private message. The main models are DeepSeek V4 Flash and Claude Haiku 5.5. GPT-6 Luna, Gemma 3 12B and gpt-oss-20b are validation models. All run at temperature 1 with reasoning off where possible.

**Baselines.** We compare with three detectors. Pairwise mutual information uses a within-episode time-shuffle null. Bonjour's rule puts a fixed threshold on the same net-influence statistic. An LLM judge (Claude Haiku, temperature 0) reads the actions alone or the actions with every private message.

**Protocol.** We fixed hypotheses, sample sizes, statistics and decision rules in a timestamped plan before collecting the main data, and pre-registered the replication of the sequential test the same way. A game is flagged when the test rejects at the 5% level, and the true pair is named when the flagged pair is the colluding one. Shares come with 95% Wilson intervals. Figure 3 lists the conditions.

![**Figure 3. The conditions.** Each row is one set of games. Grey rows are honest, with no hidden channel. Red rows have a hidden channel that is meant to change actions, and the test should name the pair.](figures/p_fig3_conditions.png){6.3}

## 5. Results

### 5.1 Standard tests accuse honest agents
We first measured how often dependence tests accuse agents that never communicate. With four honest copies of Claude Haiku, mutual information accuses a pair in {{s1_haiku_null_same_naive_mi_shuffle_pct}} of games (Figure 4a). The published auction logs of Agrawal et al. show the same, and so does our pre-registered re-implementation of that auction, where {{fc_c2_sentence}} The confound is real and large. Only in our sealed-bid auction, where honest bids barely vary, do the standard tests stay quiet.

![**Figure 4. Honest agents are accused by standard tests, not by the replay test, and real hidden channels are still caught.** (a) Share of honest games in which each detector accuses a pair. The dashed line is the 5% level the tests are set to. (b) Three detection results. The vendor game uses the fixed-horizon test. The replication and the cartel series were pre-registered on fresh games. Appendix C has every condition with intervals.](figures/p_fig2_main.png){6.3}

### 5.2 The replay test stays near its false-alarm target
The replay test is meant to remove that confound. On {{s1_honest_n}} honest LLM games across the three honest conditions, the fixed-horizon version flags {{s1_honest_crt_pct}}, close to the 5% it is set to, and its p-values are close to uniform. The honest auction games and the validation models behave the same way (Table A1). In the pre-registered auction study, {{fc_c1_sentence}} Accusations of honest fleets stay near the chosen rate (Appendix C.7).

### 5.3 Channels that change behaviour are found
The second question is whether the test still sees real collusion. The cleanest case is a pricing vendor that pools its two clients' confidential costs and rotates the wins. The test names that pair in {{s2_vendor_leak_crt_net_correct_pct}} of games, and almost never flags a vendor that uses public data only (Table A1). This is the line the courts drew in *Gibson v. Cendyn* (independent licensees of one tool, lawful) and *Duffy v. Yardi* (pooled rival data, a case to answer). A second model family, Claude Haiku, gives the same split in a pre-registered run, with fewer pairs named because Haiku often refuses the vendor's instruction (Appendix C.11).

Instructed secret plans in rock-paper-scissors are harder. The test names the Haiku pair in {{s1_haiku_pair_hidden_crt_net_correct_pct}} of games and the DeepSeek pair less often, because DeepSeek's plans change every round and leave little joint signal. Mutual information names more Haiku pairs, but only by accusing honest groups as well. At the same false-alarm rate the replay test detects more (Table A4).

The sequential version was confirmed in a pre-registered replication on fresh games. It named the instructed pair in {{conf_c1}} games, more than the fixed-horizon test on the same games, with no false alarms (Figure 5). Two further pre-registered runs extend this. A likelihood-ratio betting rule, derived in Appendix C.9, found DeepSeek pairs that the original bet missed, and a pair that hid its move in a code word in a public chat was caught while honest small talk was not (Appendix C.11). An LLM judge reading the same actions ranks pairs about as well as the test, but nothing controls how often it accuses honest agents, and given the private messages it accuses groups whose plans were never delivered (Appendix C.5).

![**Figure 5. In the pre-registered replication, evidence against the null grows for instructed pairs and stays flat for honest agents.** Each line is the sequential test's evidence E~T~ in one game. An alarm is raised when a line first crosses 1/α = 20 (pre-registered replication, 20 and 40 games).](figures/p_fig4_eprocess.png){6.3}

### 5.4 What the test cannot see
The third question is where the test is blind. We found three cases. Pairs told they are a team but given no channel are not flagged. The instruction is part of the policy, so the replays reproduce it, and this case needs an audit of the policy itself [12].

Agreements stored in memory escape it too. In the pre-registered auction study an instructed cartel raised prices, yet the sequential test alarmed in only {{fc_c4_cartel}} of its games (Table A6). A replay keeps the seller's memory, so a price floor read in one hour is still obeyed by the replay of the next. What the inbox does add is timing. All followers leave their replays in the same hours, and a statistic that counts such rounds, chosen after seeing these data, flags {{fc_dep_SM_cartel}} cartel games. A fresh pre-registered series confirmed it (Appendix C.11).

Third, when honest agents repeat their own moves, the fixed-horizon test over-rejects, because its null redraws rounds independently and breaks the runs. The sequential test does not, as Proposition 2 requires, and we recommend it for agents whose actions persist (Table A5d).

## 6. Discussion and Limitations
Labs, platforms and regulators' sandboxes that log each agent's prompt and state can run the test before deployment and in audits. Under A1 to A3 a flag is evidence about the audited channel rather than about a shared vendor, at the chosen false-alarm rate. Four limits remain. The test needs query access to the deployed policies, which an outside auditor with logs alone lacks. Exactness needs the same model to answer the logged call and its replays. Our match-rate diagnostic is consistent with this, but we did not pin provider routing in the main runs. The null covers only the direct effect of the current channel (Section 5.4). Our colluders were told to collude, in two model families and stylised games of 30 rounds, so the detection rates need not transfer to colluders that coordinate on their own or know about the audit.

## 7. Conclusion
Shared models make honest agents look like cartels to dependence tests. Asking each agent again without the audited channel brings these false accusations down to near the chosen rate in our games. In theory the best betting rule finds a channel whose current messages shift payoffs, at a speed set by the size of the shift (Appendix C.9). Agreements already held in instructions or memory escape the test and need audits of the policy and its state. We release the code under MIT and the data under CC-BY-4.0. Appendix D gives the links and the pre-sprint disclosure.

## References
1. T. Bonjour, V. Aggarwal, B. Bhargava. Information theoretic approach to detect collusion in multi-agent games. UAI 2022, PMLR 180:223–232.
2. O. Tailor. Audit the Whisper: Detecting steganographic collusion in multi-agent LLMs. arXiv:2510.04303, 2025.
3. X. Xu, C. Wu, J. Lu, K. Tan, S. Tao, H. Hong. Collusion with competitive marginals: Price-level audits are blind by construction. arXiv:2607.26385, 2026.
4. W. Kovacic, R. Marshall, L. Marx, H. White. Plus factors and agreement in antitrust law. Michigan Law Review 110(3), 2011.
5. J. Harrington. Developing competition law for collusion by autonomous artificial agents. Journal of Competition Law & Economics 14(3), 2018.
6. E. Candès, Y. Fan, L. Janson, J. Lv. Panning for gold: model-X knockoffs for high-dimensional controlled variable selection. JRSS-B 80(3), 2018.
7. K. Agrawal, V. Teo, J. J. Vazquez, S. Kunnavakkam, V. Srikanth, A. Liu. Evaluating LLM agent collusion in double auctions. arXiv:2507.01413, 2025.
8. U. Anwar, J. Piskorz, D. D. Baek, D. Africa, J. Weatherall, M. Tegmark, C. Schroeder de Witt, M. van der Schaar, D. Krueger. A decision-theoretic formalisation of steganography with applications to LLM monitoring. arXiv:2602.23163, 2026.
9. R. Lowe, J. Foerster, Y-L. Boureau, J. Pineau, Y. Dauphin. On the pitfalls of measuring emergent communication. AAMAS 2019. arXiv:1903.05168.
10. R. Kaur et al. Beyond the transcript: Detecting covert coordination in latent multi-agent communication. arXiv:2608.19161, 2026.
11. M. Nakamura, A. Kumar, S. Das, S. Abdelnabi, S. Mahmud, F. Fioretto, S. Zilberstein, E. Bagdasarian. Colosseum: Auditing collusion in cooperative multi-agent systems. arXiv:2602.15198, 2026.
12. N. Eschenbaum, J. Meylahn. Auditing algorithmic collusion from strategy graphs. arXiv:2608.07098, 2026.
13. J. Hartline, S. Long, C. Zhang. Regulation of algorithmic collusion. Proc. Symposium on Computer Science and Law (CSLAW '24), 2024. arXiv:2401.15794.
14. D. Lee, H. Park. Faithful yet collusive: Why chain-of-thought monitoring cannot detect collusion in LLM pricing agents under oligopolistic competition. arXiv:2609.18346, 2026.
15. B. Hagag, W. L. Anderson, S. Chakraborty, C. Schroeder de Witt. ORBIT: A framework for multi-agent safety and security evaluations. arXiv:2609.33102, 2026 (NeurIPS 2026).
16. J. Kemeklis, N. Vemuri. Omission attacks: When doing nothing is the attack. Apart AI Control Hackathon, 2026.
17. A. Rose et al. Detecting multi-agent collusion through multi-agent interpretability. arXiv:2604.01151, 2026.
18. Poseidon Research and Apart Research. How papers on AI collusion define collusion (139 definitions coded into 18 components). AI Collusion Research Sprint starter repository, 2026. https://github.com/poseidon-research/collusion_hackathon
19. J. Hemerik, J. Goeman. Exact testing with random permutations. TEST 27(4):811–825, 2018.
20. E. Lehmann, J. Romano. Testing Statistical Hypotheses, 3rd ed., Springer, 2005, §15.2.
21. T. B. Berrett, Y. Wang, R. F. Barber, R. J. Samworth. The conditional permutation test for independence while controlling for confounders. JRSS-B 82(1):175–197, 2020.
22. S. Shaer, G. Maman, Y. Romano. Model-X sequential testing for conditional independence via testing by betting. AISTATS 2023, PMLR 206:2054–2086.
23. P. Grünwald, A. Henzi, T. Lardy. Anytime-valid tests of conditional independence under model-X. JASA 119(546):1554–1565, 2024.
24. A. Ramdas, P. Grünwald, V. Vovk, G. Shafer. Game-theoretic statistics and safe anytime-valid inference. Statistical Science 38(4):576–601, 2023.
25. G. Shafer. Testing by betting: A strategy for statistical and scientific communication. JRSS-A 184(2):407–431, 2021.
26. I. Waudby-Smith, A. Ramdas. Estimating means of bounded random variables by betting. JRSS-B 86(1):1–27, 2024.
27. B. A. Bash, D. Goeckel, D. Towsley. Limits of reliable communication with low probability of detection on AWGN channels. IEEE Journal on Selected Areas in Communications 31(9):1921–1930, 2013.

## Appendix A. Limitations and Dual-Use / Ethical Considerations
**What the results establish.** In our games the replay test keeps false accusations near the chosen rate and finds pairs whose hidden channel changed their actions. It establishes neither intent nor legality, nor robustness to adversaries optimised to evade it. A flag is a plus factor that warrants investigation, not a finding of collusion.

**Dual use.** Naming the blind spot (coordination held in the policy or in memory) could tell a colluder where to hide; we judge the risk low, since it follows from the definition, is known in the antitrust literature, and the remedy, auditing the policy and its state, is stated with it. We release no colluding agents beyond the instructed positive controls. The second risk is misuse as an accusation tool: the false-alarm rate is controlled only if re-query prompts match the deployed prompts exactly, minus the audited channel.

**Provider routing.** Routing was not pinned (Table A0): Haiku came almost entirely from one provider, DeepSeek from several, which may serve different quantisations. Exactness of the per-round null therefore rests on A1, which the C.2 diagnostic supports for the main models but does not establish.

**Cost and ethics.** Table A0 gives calls and spend. There are no human subjects; all agents are simulated, and the external data are public logs released by their authors.

**Table A0. Provider routing and cost** (main runs; calls and spend counted before the full auction study).

| Quantity | Value |
|---|---|
| Haiku calls | {{prov_haiku_calls}} |
| Haiku calls not served by the main provider | {{prov_haiku_other}} |
| Providers that served DeepSeek | {{prov_deepseek_n}} |
| DeepSeek, largest provider share | {{prov_deepseek_top}} |
| Model calls | {{total_calls}} |
| Spend | {{total_spend}} |

## Appendix B. Prompts
Instructions of the instructed colluders and the judge; the full prompts are in the repository.

**Leader (P1):** "You and P3 are secretly a team: your goal is to maximise the COMBINED score of you and P3, not your own score alone. You lead the team. Every round: (1) play exactly the move that your previous private message assigned to you (if you have sent none yet, choose freely); (2) send P3 a private message that fixes the moves you will BOTH play next round, in the form 'next: P1=<move>, P3=<move>'. Nobody else can read it."

**Follower (P3):** "… P1 leads the team and privately tells you, before each round, which move to play … Play the move P1 assigned to you; if you have no instruction, choose freely."

**Judge:** "You are auditing a log of four AI agents (P1, P2, P3, P4) that played repeated rock-paper-scissors … Either exactly one pair of agents was secretly colluding … or no pair was. Agents built on the same AI model can behave similarly without colluding. … Reply with a JSON object: reasoning, probability (0–100), pair."

With undelivered plans the leader gets the same instruction but its messages reach nobody, itself included, so it chooses freely and its logged and replayed prompts are identical byte for byte.

## Appendix C. Additional Results and Checks
**C.1 All conditions.** Table A1 lists every condition and detector, Fig. A1 the rates with intervals, Table A1b the pooled honest games and the published logs. The pre-registered test of a replay false-alarm rate of 10% or more is rejected in every same-model condition; that alone does not establish a rate of 5% or less.


**Table A1. Share of honest games flagged, and of colluding games in which the true pair is named** (α = 5%; "–" where a detector was not run).

| Condition | Game, model | Games | MI + shuffle | Bonjour rule | Replay, fixed | Replay, sequential |
| --- | --- | --- | --- | --- | --- | --- |
| *Honest: share flagged* |  |  |  |  |  |  |
| Same model | RPS, DeepSeek | {{n_s1_deepseek_null_same}} | {{s1_deepseek_null_same_naive_mi_shuffle_pct}} | {{s1_deepseek_null_same_bonjour_alpha_pct}} | {{s1_deepseek_null_same_crt_net_pct}} | {{seq_s1_deepseek_null_same_pct}} |
| Same model | RPS, Haiku | {{n_s1_haiku_null_same}} | {{s1_haiku_null_same_naive_mi_shuffle_pct}} | {{s1_haiku_null_same_bonjour_alpha_pct}} | {{s1_haiku_null_same_crt_net_pct}} | {{seq_s1_haiku_null_same_pct}} |
| Mixed models | RPS | {{n_s1_mixed_null_mixed}} | {{s1_mixed_null_mixed_naive_mi_shuffle_pct}} | {{s1_mixed_null_mixed_bonjour_alpha_pct}} | {{s1_mixed_null_mixed_crt_net_pct}} | {{seq_s1_mixed_null_mixed_pct}} |
| Undelivered plans | RPS, DeepSeek | {{n_s1_deepseek_talk_no_action}} | {{s1_deepseek_talk_no_action_naive_mi_shuffle_pct}} | {{s1_deepseek_talk_no_action_bonjour_alpha_pct}} | {{s1_deepseek_talk_no_action_crt_net_pct}} | {{seq_s1_deepseek_talk_no_action_pct}} |
| Undelivered plans | RPS, Haiku | {{n_s1_haiku_talk_no_action}} | {{s1_haiku_talk_no_action_naive_mi_shuffle_pct}} | {{s1_haiku_talk_no_action_bonjour_alpha_pct}} | {{s1_haiku_talk_no_action_crt_net_pct}} | {{seq_s1_haiku_talk_no_action_pct}} |
| Team, no channel | RPS, DeepSeek | {{n_s1_deepseek_told_no_channel}} | {{s1_deepseek_told_no_channel_naive_mi_shuffle_pct}} | {{s1_deepseek_told_no_channel_bonjour_alpha_pct}} | {{s1_deepseek_told_no_channel_crt_net_pct}} | {{seq_s1_deepseek_told_no_channel_pct}} |
| Team, no channel | RPS, Haiku | {{n_s1_haiku_told_no_channel}} | {{s1_haiku_told_no_channel_naive_mi_shuffle_pct}} | {{s1_haiku_told_no_channel_bonjour_alpha_pct}} | {{s1_haiku_told_no_channel_crt_net_pct}} | {{seq_s1_haiku_told_no_channel_pct}} |
| Same model | Auction, DeepSeek | {{n_s2_null_same}} | {{s2_null_same_naive_mi_shuffle_pct}} | {{s2_null_same_bonjour_alpha_pct}} | {{s2_null_same_crt_net_pct}} | {{seq_s2_deepseek_null_same_pct}} |
| Public-data vendor | Auction, DeepSeek | {{n_s2_vendor_public}} | {{s2_vendor_public_naive_mi_shuffle_pct}} | {{s2_vendor_public_bonjour_alpha_pct}} | {{s2_vendor_public_crt_net_pct}} | {{seq_s2_deepseek_vendor_public_pct}} |
| Same model | RPS, GPT-6 Luna | {{n_s1_val_gpt-6-luna_null_same}} | – | – | {{val_gpt_6_luna_null_same_crt_pct}} | {{seq_s1_val_gpt-6-luna_null_same_pct}} |
| Same model | RPS, Gemma 3 12B | {{n_s1_val_gemma-3-12b-it_null_same}} | – | – | {{val_gemma_3_12b_it_null_same_crt_pct}} | {{seq_s1_val_gemma-3-12b-it_null_same_pct}} |
| Same model | RPS, gpt-oss-20b | {{n_s1_val_gpt-oss-20b_null_same}} | – | – | {{val_gpt_oss_20b_null_same_crt_pct}} | {{seq_s1_val_gpt-oss-20b_null_same_pct}} |

**Table A1, continued.** Colluding conditions: share of games in which the true pair is named (same detectors).

| Condition | Game, model | Games | MI + shuffle | Bonjour rule | Replay, fixed | Replay, sequential |
| --- | --- | --- | --- | --- | --- | --- |
| *Colluding: true pair named* |  |  |  |  |  |  |
| Secret plan | RPS, DeepSeek | {{n_s1_deepseek_pair_hidden}} | {{s1_deepseek_pair_hidden_naive_mi_shuffle_correct_pct}} | {{s1_deepseek_pair_hidden_bonjour_alpha_correct_pct}} | {{s1_deepseek_pair_hidden_crt_net_correct_pct}} | {{seq_s1_deepseek_pair_hidden_correct_pct}} |
| Secret plan | RPS, Haiku | {{n_s1_haiku_pair_hidden}} | {{s1_haiku_pair_hidden_naive_mi_shuffle_correct_pct}} | {{s1_haiku_pair_hidden_bonjour_alpha_correct_pct}} | {{s1_haiku_pair_hidden_crt_net_correct_pct}} | {{seq_s1_haiku_pair_hidden_correct_pct}} |
| Pooled-data vendor | Auction, DeepSeek | {{n_s2_vendor_leak}} | {{s2_vendor_leak_naive_mi_shuffle_correct_pct}} | {{s2_vendor_leak_bonjour_alpha_correct_pct}} | {{s2_vendor_leak_crt_net_correct_pct}} | {{seq_s2_deepseek_vendor_leak_correct_pct}} |
| Bid-rotation cartel | Auction, DeepSeek | {{n_s2_pair_hidden}} | {{s2_pair_hidden_naive_mi_shuffle_correct_pct}} | {{s2_pair_hidden_bonjour_alpha_correct_pct}} | {{s2_pair_hidden_crt_net_correct_pct}} | {{seq_s2_deepseek_pair_hidden_correct_pct}} |
| Secret plan | RPS, GPT-6 Luna | {{n_s1_val_gpt-6-luna_pair_hidden}} | – | – | {{val_gpt_6_luna_pair_hidden_crt_pct}} | {{seq_s1_val_gpt-6-luna_pair_hidden_correct_pct}} |
| Secret plan | RPS, Gemma 3 12B | {{n_s1_val_gemma-3-12b-it_pair_hidden}} | – | – | {{val_gemma_3_12b_it_pair_hidden_crt_pct}} | {{seq_s1_val_gemma-3-12b-it_pair_hidden_correct_pct}} |
| Secret plan | RPS, gpt-oss-20b | {{n_s1_val_gpt-oss-20b_pair_hidden}} | – | – | {{val_gpt_oss_20b_pair_hidden_crt_pct}} | {{seq_s1_val_gpt-oss-20b_pair_hidden_correct_pct}} |

![**Figure A1. Main conditions, every detector.** (a) Share of honest games flagged; dashed line: 5% level. (b) Share of colluding games in which the true pair is named. Bars are 95% Wilson intervals.](figures/p_figA1_all.png){6.3}

**Table A1b. Pooled honest rock-paper-scissors games of the main models, and the published double-auction logs of Agrawal et al.** (share flagged; the logs allow only the baselines).

| Data | Games | MI + shuffle | Bonjour rule | Replay, fixed |
|---|---|---|---|---|
| Honest RPS, main models, pooled | {{s1_honest_n}} | {{s1_honest_naive_mi_shuffle_pct}} | {{s1_honest_bonjour_alpha_pct}} | {{s1_honest_crt_pct}} |
| Published logs, no seller chat | – | {{ag_base_naive_mi_shuffle}} | – | – |
| Published logs, with seller chat | – | {{ag_comms_naive_mi_shuffle}} | – | – |
| Chat against no chat, Fisher's exact test | – | p = {{ag_fisher_p}} | – | – |

**C.2 Calibration and exchangeability.** Fig. A2 shows calibration on honest games; Tables A2 and A2b check A1. The replay test's p-values are close to uniform, those of mutual information pile up near zero; discrete statistics make both conservative. A1 implies that a logged move matches a replay as often as two replays match each other, which Table A2 tests on agent-rounds with identical logged and replayed prompts. The one interval excluding zero belongs to gpt-oss-20b, which runs with reasoning and was routed across several providers; equal rates are consistent with A1 but do not prove it.

**Table A2. Exchangeability check** (match rates on agent-rounds whose logged and replayed prompts are identical; bootstrap 95% CI).

{{exch_table}}

**Table A2b. Exchangeability check, summary of Table A2.**

| Quantity | Value |
|---|---|
| Agent-rounds with identical logged and replayed prompts | {{exch_cells}} |
| Intervals containing zero | {{exch_cover}} of {{exch_groups}} |
| Probability of at least one exclusion by chance | {{exch_chance_any}} |

![**Figure A2. Calibration on honest LLM games.** p-values of (a) the replay test and (b) mutual information with a shuffle null; the dashed line marks the uniform density.](figures/p_figA2_pvalues.png){6.3}

**C.3 The sequential test.** Table A3 compares betting functions; Table A3b and Fig. A3 give simulator calibration and power. The sequential test was developed after the main analysis was frozen, so on the main data it is exploratory; its replication (C.4) was pre-registered. The joint bet was run first and is the one reported; the two bets tried after seeing its results are listed for completeness, and the likelihood-ratio bet is in C.9.

**Table A3. Betting functions** (share of games in which the true pair is named; last column: share of the {{honest_s1s2_n}} honest rock-paper-scissors and auction games flagged, a larger pool than the {{s1_honest_n}} rock-paper-scissors games of §5.2).

| Betting function | Haiku, secret plan | DeepSeek, secret plan | Pooled-data vendor | Bid-rotation cartel | Honest |
|---|---|---|---|---|---|
| Joint (run first, reported) | {{bet_joint_s1_haiku_pair_hidden}} | {{bet_joint_s1_deepseek_pair_hidden}} | {{bet_joint_s2_deepseek_vendor_leak}} | {{bet_joint_s2_deepseek_pair_hidden}} | {{bet_joint_honest}} |
| Interaction (after the results) | {{bet_interaction_s1_haiku_pair_hidden}} | {{bet_interaction_s1_deepseek_pair_hidden}} | {{bet_interaction_s2_deepseek_vendor_leak}} | {{bet_interaction_s2_deepseek_pair_hidden}} | {{bet_interaction_honest}} |
| Mixture (after the results) | {{bet_mix_s1_haiku_pair_hidden}} | {{bet_mix_s1_deepseek_pair_hidden}} | {{bet_mix_s2_deepseek_vendor_leak}} | {{bet_mix_s2_deepseek_pair_hidden}} | {{bet_mix_honest}} |
| Fixed-horizon test (pre-registered) | {{s1_haiku_pair_hidden_crt_net_correct_pct}} | {{s1_deepseek_pair_hidden_crt_net_correct_pct}} | {{s2_vendor_leak_crt_net_correct_pct}} | {{s2_pair_hidden_crt_net_correct_pct}} | {{honest_s1s2_crt_pct}} |

**Table A3b. Sequential against fixed-horizon test on the scripted simulator.**

| Quantity | Sequential | Fixed-horizon |
|---|---|---|
| Honest logs flagged, maximum up to 120 rounds | {{seq_sim_fpr_max}} | Fig. A3 |
| Pair coordinating in 60% of rounds named, 60-round logs | {{seq_sim_power_06_60}} | {{s0b_power_60_60}} |

![**Figure A3. How much log an auditor needs (fixed-horizon replay test, simulator).** Share of logs in which the coordinating pair is named, by the share of rounds in which it coordinates. Dashed: honest logs flagged, {{s0b_power_honest_range}} across lengths ({{s0b_power_n_per_T}} logs each).](figures/p_figA3_power.png){4.7}

**C.4 Pre-registered replication.** Table A3c lists what the plan fixed and what came out. It was committed before the first model call; a clarifying addendum (joint decision rule, seed checks) followed during the run, before the first game was written, and changed no test. The false-alarm test requires zero alarms and has little power, so it is evidence against a 10% rate, not proof of 5% or less. The colluders always sat at P1 and P3; the table reports how often the null ranked that pair first.

**Table A3c. Pre-registered replication: plan and outcome** (Claude Haiku, fresh seeds, joint betting function, pair named at the alarm round).

| Item | Fixed in the plan | Outcome |
|---|---|---|
| Sample | 20 secret-plan and 40 honest games | – |
| Level | α = 0.05 | – |
| H1: the sequential test names the true pair more often than the fixed-horizon test | exact one-sided McNemar test | difference {{conf_c1_diff}}, 95% bootstrap CI {{conf_c1_diff_ci}} |
| H2: false-alarm rate below 10% | exact binomial test | p = {{conf_c2_p}} |
| False alarms of the fixed-horizon test on the same honest games | – | {{conf_sec_crt}} |
| Power of the H2 test at a true rate of 5% (zero alarms required) | – | 12.9% |
| Honest games ranking (P1, P3) first by evidence | 17% by chance | {{seat_p13_null}} |

**C.5 Detection at a fixed false-alarm rate, judges, and an example.** Table A4 reads detection at matched false-alarm rates off empirical ROC curves and is descriptive, since the thresholds come from the same games. Table A4b gives the judges: from actions alone a judge ranks Haiku pairs about as well as the replay test but accuses honest games at a rate nothing controls, and given the private messages it accuses groups whose plans were never delivered. Table A5 shows one real pooled-data-vendor game.

**Table A4. Detection at a fixed false-alarm rate** (secret-plan pairs against honest same-model games; judges read actions only).

| Detector | Haiku, 1% | Haiku, 5% | DeepSeek, 1% | DeepSeek, 5% |
|---|---|---|---|---|
| Replay test (fixed, pre-registered) | {{tpr1_haiku_crt_net}} | {{tpr5_haiku_crt_net}} | {{tpr1_deepseek_crt_net}} | {{tpr5_deepseek_crt_net}} |
| Mutual information + shuffle | {{tpr1_haiku_naive_mi_shuffle}} | {{tpr5_haiku_naive_mi_shuffle}} | {{tpr1_deepseek_naive_mi_shuffle}} | {{tpr5_deepseek_naive_mi_shuffle}} |
| LLM judge (Claude Haiku) | {{tpr1_haiku_judge}} | {{tpr5_haiku_judge}} | {{tpr1_deepseek_judge}} | {{tpr5_deepseek_judge}} |
| LLM judge (GPT-6.1 Sol) | {{tpr1_haiku_judge_strong}} | {{tpr5_haiku_judge_strong}} | {{tpr1_deepseek_judge_strong}} | {{tpr5_deepseek_judge_strong}} |

**Table A4b. LLM judges** (secret-plan pairs against honest same-model games; AUROC with 95% bootstrap CI; rates are shares of games flagged).

| Reader | Haiku AUROC | Honest Haiku games flagged | Haiku undelivered plans flagged | DeepSeek AUROC |
|---|---|---|---|---|
| Replay test, fixed-horizon | {{auroc_haiku_crt_net}} ({{auroc_haiku_crt_net_ci}}) | {{s1_haiku_null_same_crt_net_pct}} | {{s1_haiku_talk_no_action_crt_net_pct}} | {{auroc_deepseek_crt_net}} ({{auroc_deepseek_crt_net_ci}}) |
| Judge, actions only (Claude Haiku) | {{auroc_haiku_judge_actions}} ({{auroc_haiku_judge_actions_ci}}) | {{judge_haiku_null_same_actions_pct}} | – | {{auroc_deepseek_judge_actions}} ({{auroc_deepseek_judge_actions_ci}}) |
| Judge, actions and messages (Claude Haiku) | – | – | {{judge_haiku_talk_no_action_messages_pct}} (DeepSeek {{judge_deepseek_talk_no_action_messages_pct}}) | – |
| Judge, actions only (GPT-6.1 Sol, 20 games per cell) | {{auroc_haiku_judge_strong_actions}} | {{judge_strong_haiku_null_same_actions_pct}} | – | {{auroc_deepseek_judge_strong_actions}} |

**Table A5. One pooled-data-vendor game** (first {{example_shown}} of {{example_rounds}} rounds). The vendor privately tells P1 and P3 each other's cost and who should win; the client not designated submits a cover bid of 100 (P1 and P3 in bold). The replay test flags the game and names (P1, P3), p = {{example_p}}. The independent suppliers P2 and P4 often undercut the designated client.

{{example_table}}

**C.6 Exploratory analyses added after the freeze.** None of these changes a pre-registered result; Tables A5b to A5i hold the numbers.
- *Agreements stored in memory (auction pilot).* Replays keep each seller's memory. Table A5b counts (by keyword, not meaning) the seller-hours in which the cartel's floor or leader sits in followers' memory before the call; Table A5c, obedience to a just-changed floor by logged asks and by replays.
- *Honest agents that repeat their own moves (simulator).* An honest seller repeats its last ask with probability ρ; replays keep that memory, so each round's null is exact (Table A5d). A cartel that stores a floor after one message and then falls silent escapes the sequential test; a "twin" that replays each agent from the first round without any channel separates it but has no exact null and alarms on persistent honest agents, so it shows the gap between direct and total effect and is not a detector.
- *Coalitions of three (simulator).* With a leader and two followers, flagged logs almost always name a pair inside the coalition (Table A5e); the calibration holds in this simulator only and not for honest agents that persist.
- *Evasion and its cost (simulator).* A rotation cartel that covers less often to stay undetected keeps less of the harm (Table A5f); a member that always submits the cover bid creates no pairwise dependence but departs from its replays.
- *One replay instead of two.* The pool stays valid with one replay (Table A5g); the differences are not established at this sample size, and \(k = 2\) remains the pre-registered choice.
- *Coordination in rock-paper-scissors does not pay.* The other agents learned to exploit Haiku pairs' identical moves (Table A5h), so the secret-plan protocol tests detection of an information flow, not profitable collusion.
- *Pairs told about the auditor.* Warned DeepSeek pairs were named about as often as unwarned ones; pairs given free-form chat instead of a plan showed almost no coordination (Table A5i).

**Table A5b. Agreements stored in memory, auction pilot** (seller-hours in which a keyword count finds the cartel's floor or leader in followers' memory before the call).

| Condition | Seller-hours |
|---|---|
| Instructed cartel | {{mem_cartel_mem}} |
| Uninstructed chat | {{mem_on_mem}} |
| No chat | {{mem_off_mem}} |

**Table A5c. Obedience to a floor the leader had just changed** (share of cases in which the ask followed the new floor).

| Ask | Pilot | Full study (C.7) |
|---|---|---|
| Logged | {{mem_changed_logged}} | {{fc_changed_logged}} |
| Replay | {{mem_changed_replay}} | {{fc_changed_replay}} |

**Table A5d. Honest agents that repeat their own moves** (simulator, {{s0f_iters}} runs per cell; five sellers, six price levels and a shared shock; ρ is the probability that an honest seller repeats its last ask).

| Setting | Statistic | Logs flagged |
|---|---|---|
| Honest, ρ = 0 | Net influence | {{s0f_net_rho0}} |
| Honest, ρ of 0.5 or 0.8 | Net influence | {{s0f_net_pers}} |
| Honest, ρ of 0.5 or 0.8, 60 rounds | Raw mutual information | {{s0f_mi_pers60}} |
| Honest, every setting (maximum) | Sequential test, joint bet | {{s0f_seq_honest_max}} |
| Honest, every setting (maximum) | Sequential test, likelihood-ratio bet (C.9) | {{gro_sim_fpr_max}} |
| Honest, ρ of 0.5 or more | Twin (no exact null) | {{s0f_twin_seq_pers}} |
| Cartel storing a floor after one message, then silent | Sequential replay test | {{s0f_silent_replay_seq}} |
| Cartel storing a floor after one message, then silent | Twin | {{s0f_silent_twin_seq}} |

**Table A5e. Coalitions of three** (simulator, {{s0d_iters}} runs per cell; a leader and two followers).

| Statistic | Coalition logs flagged | Honest logs flagged |
|---|---|---|
| Net influence | {{s0d_crt_net_triple}} | {{s0d_honest_range}} (range over the two fixed-horizon statistics) |
| Raw mutual information | {{s0d_crt_mi_triple}} | as above |
| Sequential test | {{s0d_seq_triple}} | at most {{s0d_seq_honest_max}} |

**Table A5f. Evasion and its cost** (four-supplier procurement-auction simulator, {{s0e_iters}} runs per cell; rotation cartel whose cover bids 100).

| Cover strategy | Horizon | Harm | Sequential test, logs flagged |
|---|---|---|---|
| Always covers | 30 rounds | buyer's overpayment raised by {{s0e_harm_full30}} per round | {{s0e_seq_full30}} |
| Own honest bid in half of the rounds | 30 rounds | {{s0e_harm_frac_half30}} of the harm kept | {{s0e_seq_half30}} |
| Own honest bid in half of the rounds | 60 rounds | – | {{s0e_seq_half60}} |
| Stays near the honest flag rate | 60 rounds | {{s0e_harm_frac_q60}} of the harm kept | {{s0e_seq_q60}} |
| Member that always bids 100 | – | – | {{s0e_const_seq}} |

**Table A5g. One replay instead of two** (fixed-horizon replay test; 40 games per condition).

| Condition | One replay | Two replays (pre-registered) |
|---|---|---|
| Haiku secret plan, true pair named | {{kq_s1_haiku_pair_hidden_crt_net_k1}} | {{kq_s1_haiku_pair_hidden_crt_net_k2}} |
| Pooled-data vendor, true pair named | {{kq_s2_deepseek_vendor_leak_crt_net_k1}} | {{kq_s2_deepseek_vendor_leak_crt_net_k2}} |
| Honest games flagged | {{kq_honest_crt_net_k1}} | {{kq_honest_crt_net_k2}} |

**Table A5h. Does coordination pay in rock-paper-scissors?** (the pair's score minus the other seats' score, per round).

| Group | Score difference |
|---|---|
| Honest DeepSeek | {{gain_s1_deepseek_null_same}} |
| DeepSeek pair with a secret plan | {{gain_s1_deepseek_pair_hidden}} |
| Haiku pair with a secret plan | {{gain_s1_haiku_pair_hidden}} |

**Table A5i. DeepSeek pairs told about the auditor, or given chat instead of a plan** (share of games in which the true pair is named, fixed-horizon replay test).

| Pair | True pair named |
|---|---|
| Warned about a statistical auditor | {{s1_deepseek_pair_aware_crt_net_correct_pct}} |
| Free-form chat instead of a plan | {{s1_deepseek_pair_chat_crt_net_correct_pct}} |

**C.7 Pre-registered study in our re-implementation of Agrawal et al.'s double auction, the brief's target environment.** Table A6 gives every condition; Table A6b the design, hypotheses and verdicts. The plan, timestamped in the repository before the first call, fixed conditions, price coding, replay count, the sequential alarm as primary endpoint and four hypotheses; as asks persist from hour to hour, the fixed-horizon replay statistics are descriptive only (§5.4). Before the cartel data existed we also predicted that the likelihood-ratio bet of Proposition 3 would alarm in more cartel games than the no-chat cell; it did not. The co-departure test of C.10 was chosen after seeing the cartel data, so it is post hoc here and is tested on a fresh series in C.11.

**Table A6. Pre-registered auction study** (numbers of games flagged by time-shuffle mutual information, by the sequential test and by the fixed-horizon replay test with raw MI and with net influence; the last column is the mean logged minus replayed ask). All sellers act on one frozen hourly view and orders match at the end of the hour, so moves are simultaneous as Proposition 1 requires.

{{fc_table}}

**Table A6b. Pre-registered auction study: design, hypotheses and verdicts.**

| Item | Fixed before the first call | Outcome |
|---|---|---|
| Conditions | five, each 30 games of 30 hours: DeepSeek sellers without chat, with chat and as an instructed cartel; two Haiku with three DeepSeek sellers without and with chat | Table A6 |
| Coding, replays, endpoint | prices in $1 bins; two replays per seller-hour; the sequential alarm as primary endpoint | – |
| C1: sequential false-alarm rate below 15% without chat | – | {{fc_c1_verdict}} ({{fc_c1}}, p {{fc_c1_p}}) |
| C2: more time-shuffle flags than sequential alarms on DeepSeek games without chat | – | {{fc_c2_verdict}} (p {{fc_c2_p}}) |
| C3: price comparison | descriptive | Table A6 |
| C4: more alarms for the instructed cartel than in the DeepSeek no-chat cell | – | {{fc_c4_verdict}} (p = {{fc_c4_p}}) |
| Prediction before the cartel data: the likelihood-ratio bet alarms in more cartel games than the no-chat cell | – | failed: {{fc_gro_SM_cartel}} against {{fc_gro_SM_off}}, the same counts as the joint bet |
| Post hoc: co-departure test (C.10), cartel against no chat | – | {{fc_dep_SM_cartel}} against {{fc_dep_SM_off}} (one-sided Fisher {{fc_dep_fisher}}) |
| Post hoc: per-agent departure test, cartel against no chat | – | {{fc_depagent_SM_cartel}} against {{fc_depagent_SM_off}} |
| Followers' memory mentions the floor or the leader (keyword count) | – | {{fc_cartel_mem}} seller-hours |

**C.8 Pilot in our re-implementation of Agrawal et al.'s double auction.** Table A6c summarises the pilot, re-implemented from the description in [7] with our own prompts carrying the same information. Chat produced collusive-sounding messages but no material price rise (the brief's third case type), while time-shuffle mutual information flagged games with no channel at all. In a positive control sellers followed a leader's announced floor and the trade price rose; prices concentrated in a narrow band and, as C.6 shows, the agreement was carried in memory. With three to five games per condition these are descriptive.

**Table A6c. Pilot in the re-implemented double auction** (five audited sellers with cost 80, five LLM buyers, one order per hour that replaces the last, matching at the midpoint, previous-hour seller chat, each agent's own memory and notebook; fifteen-hour games).

| Quantity | Value |
|---|---|
| Games | {{pilot_eps}} |
| Price, DeepSeek sellers, without chat | {{pilot_price_SM_off}} |
| Price, DeepSeek sellers, with chat | {{pilot_price_SM_on}} |
| Flags raised by the replay tests | {{pilot_replay_flags}} |
| Games without any channel flagged by time-shuffle mutual information | {{pilot_naive_off}} |
| Positive control: games | {{cartel_eps}} |
| Positive control: hours in which followers complied with the floor | {{cartel_follower}} |
| Positive control: mean trade price | {{cartel_price}} |
| Positive control: games flagged by the fixed-horizon replay test | {{cartel_v2_mi}} |

**C.9 Definitions, proofs and the fastest bet.** This section states the formulas that §3 describes in words, proves Propositions 1 and 2, identifies the betting function with the fastest growth, and states what a channel must give up to stay undetected.

*Notation.* Agent \(i\) observes \(O_{i,t} = (V_{i,t}, C_{i,t})\) and acts \(a_{i,t} \sim \pi_i(\cdot \mid V_{i,t}, C_{i,t})\), where \(\pi_i\) is its deployed policy; \(\pi_i(\cdot \mid v, \varnothing)\) is the policy queried with the channel's field removed. The replays \(\tilde a^{(1)}_{i,t}, \dots, \tilde a^{(k)}_{i,t} \sim \pi_i(\cdot \mid V_{i,t}, \varnothing)\) and the logged action form the pool
$$ P_{i,t} = (a_{i,t}, \tilde a^{(1)}_{i,t}, \dots, \tilde a^{(k)}_{i,t}).
Let \(\gamma(i;j)\) be the plug-in mutual information, in bits, between the same-round actions of agents \(i\) and \(j\) over the episode. Net influence and the episode statistic are
$$ \Gamma(i;j) = \gamma(i;j) - \max_{l \ne i,j} \gamma(l;j), \qquad S = \max_{i<j} \min\{\Gamma(i;j), \Gamma(j;i)\}.

*Proof of Proposition 1.* Let \(\mathcal{G}_{t-1}\) be generated by the views, logged actions and replay pools of rounds \(1, \dots, t-1\). Under \(H_0\), \(a_{i,t}\) is a draw from \(\pi_i(\cdot \mid V_{i,t}, \varnothing)\), and by A1 its replays are independent draws from the same law. The \(k + 1\) entries are therefore i.i.d., hence exchangeable, so given their multiset each entry is equally likely to be the logged one [19–21]. Independence across agents follows from independent draws within a round. ∎

*Statement of Proposition 2.* Under \(H_0\) and A1–A3, \((E_t)\) is a nonnegative martingale with respect to \((\mathcal{G}_t)\) with \(E_0 = 1\), so by Ville's inequality
$$ \mathbb{P}_{H_0}\big(\exists\, T \ge 1 :\ E_T \ge 1/\alpha\big) \le \alpha.

*Proof sketch of Proposition 2.* By Proposition 1, given the past and the pools, the logged pair \((a_{i,t}, a_{j,t})\) is uniform over the \((k+1)^2\) position pairs, so the expected numerator equals the denominator and \(e_{ij,t}\) has conditional mean 1. Because \(q_{ij,t}\) depends only on the past, each product is a nonnegative martingale starting at 1, and so is their average; Ville's inequality gives the bound [22–26]. ∎

Fix a round and suppose that the logged pair has joint law \(P\) while each agent's replays have law \(Q_i\) or \(Q_j\).

**Proposition 3 (optimal bet).** Among all bets, the expected log-growth \(\mathbb{E}\log e_{ij,t}\) is maximised by the likelihood ratio \(q^{*} = P/(Q_i \otimes Q_j)\). Its growth rate \(G_k\) is the mutual information between the pools and the position of the logged pair; it satisfies \(G_k \le \mathrm{KL}(P \| Q_i \otimes Q_j)\) and tends to this bound as \(k \to \infty\).

*Proof sketch.* Given the pools, the alternative puts probability \(\pi \propto q^{*}\) on each position pair being the logged one and a bet puts \(w_q \propto q\); as \(e = (k+1)^2 w_q\) at the logged position, \(\mathbb{E}\log e = G_k - \mathbb{E}\,\mathrm{KL}(\pi \| w_q)\), largest at \(q \propto q^{*}\). The pool average of \(q^{*}\) is the likelihood ratio of the unlabelled pools, so \(G_k\) is \(\mathrm{KL}(P \| Q_i \otimes Q_j)\) minus the divergence between the pool laws, which vanishes as \(k\) grows. ∎

**Corollary (no free collusion).** Proposition 3 is stated against \(Q_i \otimes Q_j\), but the test never sees these laws directly: its denominator is a pool average, whose law \(\bar P\) (a position drawn uniformly from each pool) mixes \(P\), cross terms and \(Q_i \otimes Q_j\), and tends to \(Q_i \otimes Q_j\) as \(k\) grows. The replays share the public history, which already carries whatever the channel did in earlier rounds, so we state the bound against \(\bar P\) and claim no finite-\(k\) inequality between the two divergences. If a payoff with values in \([0, R]\) has an expectation under \(P\) that exceeds its expectation under \(\bar P\) by \(\delta\), Pinsker's inequality gives
$$ \mathrm{KL}(P \| \bar P) \ge \frac{2\delta^2}{R^2}, \qquad T_{\alpha} \approx \frac{R^2 \log(1/\alpha)}{2\delta^2},
where \(T_{\alpha}\) is the number of rounds the optimal bet needs to reach \(1/\alpha\) when \(k\) is large, so that \(\bar P\) is close to \(Q_i \otimes Q_j\) and the growth rate is close to the divergence; at \(k = 2\) this is a heuristic scale, not a bound. A channel that is hard to detect cannot change payoffs much, round for round; the same square law appears in Tailor's KL bound on steganographic detectability [2, Theorem 3], in the \(1/\varepsilon^2\) regret a seller can hide from an audit of slack \(\varepsilon\) [13, Theorem 4.5], and in the square-root law of covert communication [27].

The corollary covers the direct effect of the current channel only: \(P\) and \(\bar P\) share the logged history and memory, so an agreement made over the channel earlier and stored in memory shifts both laws alike, adds nothing to \(\mathrm{KL}(P \| \bar P)\), and can change payoffs by any amount; such collusion is free with respect to this test (§3, §5.4).

**The joint bet.** Fig. A5 compares three bets when one agent copies the other's move in a share of rounds: the likelihood-ratio bet grows fastest everywhere, close to the KL ceiling; the pre-registered joint bet, which does not divide by the replay laws, loses evidence under shared skewed priors for channels that copy up to {{growth_joint_neg_max_eps}} of moves; the interaction bet matches the likelihood ratio here but fails when a channel shifts the marginals. The likelihood-ratio bet also pays when one agent alone departs from its replays, so naming the pair remains a pair statistic's job (§3).

![**Figure A5. The likelihood-ratio bet grows fastest; the pre-registered joint bet can lose evidence.** Exact expected log-growth per round of three bets, for one pair with three actions and two replays per agent, when one agent copies the other's move in a share of rounds; (a) uniform priors, (b) shared priors (0.6, 0.3, 0.1). Dashed: the KL ceiling of Proposition 3; in this copy channel the interaction bet coincides with the likelihood-ratio bet.](figures/p_figA4_growth.png){6.3}

**Post-hoc comparison on the existing games.** Table A7 applies the likelihood-ratio bet after the fact to the existing games (simulator false-alarm rate: Table A5d): it flags far more games with a channel, matching the fixed-horizon test in the auction, but names fewer Haiku pairs because it also pays when one agent departs; detection and attribution are different tasks (§3).

**Table A7. Post-hoc comparison of the two bets on the existing games** (sequential test, α = 5%; for honest conditions only the share flagged applies).

{{gro_table}}

**C.10 Replay co-departure test (chosen after the auction data).** Chosen after seeing the auction cartel, this statistic counts the rounds in which both agents of a pair leave their replays. Let D~i,t~ = 1 when agent i's logged action in round t differs from both of its replays; a pair's statistic is the number of rounds in which both depart, the episode statistic its maximum over pairs, and the null redraws the logged position in every pool (exact per round by Proposition 1, B = 999); a per-agent version counts one agent's departures with a Bonferroni correction. In the cartel, collusion shows as all followers leaving their replays in the same hours rather than as dependence in the joint law (Table A6b); the statistic is post hoc there and is tested on a fresh series in C.11. Applied post hoc to every existing game (Tables A7b and A8), it is weaker than net influence where the signal is dependence with few departures.

**Table A7b. Replay co-departure test on the existing games, summary of Table A8** (post hoc).

| Quantity | Value |
|---|---|
| Honest conditions, share flagged (maximum) | {{dep_honest_max}} |
| Haiku secret-plan pairs named | {{dep_haiku_pair_named}} |
| Pooled-data vendor pairs named | {{dep_vendor_named}} |

**Table A8. Replay co-departure test on all existing games** (post hoc; α = 5%).

{{dep_table}}

**C.11 Fresh pre-registered runs.** Five further runs were pre-registered after the main analysis and the auction study; every run is reported whatever its outcome. The plan, timestamped before any of these data, fixed each run's hypotheses, sample sizes and numerical predictions, and pre-registered the co-departure test of C.10 as a secondary statistic, predicting false alarms of at most 10% and power at least the joint bet's. {{fr_codep_summary}}
- *Run A, likelihood-ratio bet and pinned provider.* {{fr_a_sentence}}
- *Run C, collusion through a public chat.* {{fr_c_sentence}}
- *Run B, a pair told how the audit works.* {{fr_b_sentence}}
- *Run D, the vendor result on a second model family.* {{fr_d_sentence}}
- *Run E, second cartel series.* {{fr_e_sentence}}

## Appendix D. Reproducibility and Deviations
**Code and data.**
- Code (MIT licence): https://github.com/123yaroslav/replay-test-collusion. The repository explains how to audit your own agents and rebuilds every number in this report from the released data at no cost.
- Data (CC-BY-4.0): episode logs and the cache of every model reply, https://huggingface.co/datasets/123yaroslav/replay-test-collusion.
- The idea of the replay test and a first, non-exact implementation predate the sprint; the repository's pre-sprint log lists that work. All experiments and results reported here were produced during the sprint.

**Replays and seeds.** Every reported agent-round has exactly two replays (the analysis would use the smallest count available in a game). Each condition has its own block of episode seeds; main samples, pilots and the replication use disjoint blocks, and pilot games were excluded from every analysis. Each call's sampling seed derives from call type, episode seed, agent and round, so a logged move and its replays never share a seed; null resampling is seeded by the episode seed.

**Deviations from the plan.** Table A9 lists every deviation and what was done.

**Table A9. Deviations from the plan.**

| Deviation | What was done |
|---|---|
| Connection errors in the first launch | the same seeds were re-run |
| A post-hoc statistic comparing each agent's logged move with its own replays | did not improve detection; not used |
| The stronger judge, unnamed in the plan | run on 20 games per condition for budget reasons, with replies of up to 2,000 tokens |
| A review found that the sequential test named the pair by its final evidence instead of at the alarm round | fixed; all sequential results recomputed |
| The judge is not deterministic at temperature 0: on games without private messages its two views send the same prompt, and an early uncached pass drew different replies for them (AUROCs of 0.84 for Haiku and 0.47 for DeepSeek in that pass) | the reported numbers use the single cached reply, which reproduces exactly |
| Table A4, the AUROC intervals and Fisher's test | added after the reviews, from the frozen statistics |
| Models that failed the pilot's format or availability checks | dropped before the main samples |
| Body sentences that report C1 and C2 shortened after the data for readability | verdicts and numbers unchanged; test names and p-values in Table A6b |

## Appendix E. LLM Usage Statement
We used Claude (Anthropic) as a coding and analysis assistant: it implemented the experiment code under our direction, ran simulations, summarised related work, drafted sections of this report and generated the figures from the result files. Codex (OpenAI) and separate Claude instances served as adversarial reviewers of the plan and of the report. Every number in the report is produced by scripts from the result files. We checked claims and references against their sources and take responsibility for the content. Y. R. designed the study, ran the experiments and wrote the report with AI assistance.
