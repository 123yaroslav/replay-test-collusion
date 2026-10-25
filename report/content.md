% TITLE: Monoculture Looks Like Collusion: A Replay Test for Hidden Information Flow Between LLM Agents
% AUTHOR: Yaroslav Rogoza | Independent researcher
% ABSTRACT
Collusion audits test whether agents' actions depend on each other, but agents built on one model act alike without communicating, so these audits accuse honest fleets: at the 5% level, mutual information flags {{s1_haiku_null_same_naive_mi_shuffle_pct}} of games among four honest Claude Haiku agents. Our replay test re-queries each deployed agent every round on the same input with the suspected channel removed, and flags a pair only if its actions depend on each other more than its replays do. If replays are served like the originals, each round's null is exact, and a sequential version is valid at any stopping time. On {{s1_honest_n}} honest LLM games its fixed-horizon version flags {{s1_honest_crt_pct}}. In a procurement auction with a shared pricing vendor, its fixed-horizon version names the true pair in {{s2_vendor_leak_crt_net_correct_pct}} of games when the vendor pools confidential costs and flags {{s2_vendor_public_crt_net_pct}} when it uses only public data. A pre-registered replication on fresh games confirmed the sequential version on an instructed positive control ({{conf_c1}} pairs named, {{conf_c2}} false alarms). {{fc_abstract_sentence}} The cartel escaped the pre-registered bet but not a post-hoc statistic, agents departing together from their own replays, confirmed in a fresh pre-registered series ({{fr_e_cartel2_codep_flag}} against {{fr_e_off_codep_flag}}). It detects what the current channel adds; coordination already stored in instructions or memory escapes it.
% END

## 1. Introduction
Firms and labs now run many LLM agents side by side: pricing bots in one market, bidders in one procurement system, coding agents in one repository. An auditor often cannot see the channels between such agents, and then "what's left to monitor is still the actions of the agents themselves" (S. Motwani, talk at this sprint). The standard audit therefore asks whether one agent's actions predict another's, using dependence measures such as the net influence of Bonjour et al. [1] or mutual information [2].

This audit is unreliable when the agents share a model. Agents built on one base model share priors and react to the same public history, so they move together without exchanging a word [3]. Antitrust law calls such behaviour conscious parallelism and treats it as lawful; only an agreement is unlawful [4, 5]. A test that cannot tell the two apart can accuse honest fleets that run on one vendor's model.

**An example.** Four honest copies of Claude Haiku play rock-paper-scissors. They favour the same moves, so a mutual-information test sees two of them moving together and accuses the pair. Ask the two agents again on the same inputs and their replays are just as alike: the similarity comes from the model. If instead one player secretly tells the other what to play, replays asked without that message no longer track each other the way the real moves do.

**Our approach (Fig. 1).** We make the shared model part of the null hypothesis. In every round the auditor asks each agent again, twice, what it would do given exactly what it saw, minus the audited channel. If the channel adds nothing, the real move is one more draw from the same distribution as its replays, so mixing real and replayed moves shows what honest dependence looks like, shared model included. This is a conditional randomization test [6] whose conditional distribution is sampled from the deployed agent instead of being modelled.

![**Figure 1. The replay test.** (1) An agent acts on its logged view and a hidden channel. (2) The auditor queries the same policy again on the logged view alone. (3) Without a channel effect, the logged move is a uniform member of its pool. (4) Redraws from the pools give a p-value for the dependence statistic S; the sequential test's evidence E~T~ raises an alarm with a false-alarm rate below α however long the audit runs (§3).](figures/p_fig1_method.png){6.3}

**Contributions.**
1. We measure a known confound [3]: dependence tests accuse honest same-model agents, both in LLM games and in published double-auction logs [7] (§5.1).
2. We propose the replay test, with a null that is exact within each round and a sequential version that is valid at any stopping time, under stated assumptions (§3).
3. The fixed-horizon version keeps false alarms near the target on honest LLM games and still finds channels that change behaviour, including a vendor pooling confidential data; a pre-registered replication confirms the sequential version (§5.2–5.3).
4. We map its boundary: coordination held in instructions or memory, weak channels, and honest agents that repeat their own moves (§5.4). Together these answer sprint questions 1.4 and 2.3.


## 2. Related Work
**Dependence-based detection.** Bonjour et al. [1] define pairwise net influence and identify colluding pairs in small games with a fixed threshold; they assume two colluders and agents that do not share priors. Audit the Whisper [2] calibrates transcript statistics (mutual information, permutation and watermark tests) against honest runs replayed under fresh seeds and tunes each detector to a fixed false-positive budget; the null is a population of other runs, not the audited agent on its own input, and the flag attaches to a run rather than a pair. Xu et al. [3] show that two deployments of one model carry correlated bid residuals, and note that dependence cannot tell a conspiracy from one firm running several agents. Anwar et al. [8] formalise steganography as a utility gap between a receiver and a sentinel that lacks the code; this says what a hidden channel is, but gives a measure, not a test.

**Auditing behaviour, policies and reasoning.** In emergent-communication RL, Lowe et al. [9] measure positive listening by the causal influence of communication: substitute the message and observe the change in the listener's action distribution. Our replay is a related intervention on a logged call: the audited channel is removed and the deployed policy is resampled, rather than a message substituted and read off a known policy. Kaur et al. [10] carry this idea to LLM agents that exchange hidden states: their second layer scores the KL divergence between the receiver's action distribution under the suspect latent and under a matched control row, but the control is assumed available at test time, the threshold is an empirical 95th percentile, and no error rate is controlled. Colosseum [11] measures a known coalition's advantage; Eschenbaum and Meylahn [12] query frozen pricing policies and read collusion off strategy graphs; Hartline et al. [13] audit one seller's regret. Chain-of-thought monitoring misses collusion in pricing agents [14], per-action defences did not stop colluding agents [15], and stronger monitors need a cross-agent view [16] or activations [17].

**Definition used.** Of the 139 definitions in the organisers' survey [18], we follow those that require communication [17], but we do not count observed public actions as communication. Collusion here is coordination *enabled by non-public information exchanged between agents*, directly or through a third party such as a vendor. Coordination that each policy produces from public information (shared priors, Schelling points) is conscious parallelism.

**The gap.** The intervention is old [9, 10] and the null is the model-X conditional randomization test [6]; what is new is sampling that null from the deployed agent itself. We know of no audit that treats same-prompt resampling of a deployed LLM agent as an exchangeable null. Under the null the logged action and its replays are draws from one distribution, so the per-round test is exact and its sequential version is anytime-valid. Audit the Whisper [2] calibrates against other runs and Kaur et al. [10] against a matched control assumed available at test time; neither gives a per-round error guarantee. The guarantee rests on one condition: weights, system prompt, decoding settings and hidden state must be identical for the logged call and its replays. If any of them differs, exchangeability fails and the guarantee with it.

## 3. Problem Setup and Method
**Setting.** Agents \(i = 1, \dots, N\) act in rounds \(t = 1, \dots, T\) of one game (an episode), with actions in a finite set \(\mathcal{A}\) whose coding is fixed before the first round. Agent \(i\) observes \(O_{i,t} = (V_{i,t}, C_{i,t})\). Here \(V_{i,t}\) is what the auditor logs and can replay: the prompt, the public history, tool outputs and the agent's own memory. \(C_{i,t}\) is the audited channel: an inbox, a shared memory or a vendor's message. The agent acts
$$ a_{i,t} \sim \pi_i(\cdot \mid V_{i,t}, C_{i,t}),
where \(\pi_i\) is its deployed policy, a model with its system prompt. The auditor, for example an operator or a regulator's sandbox, sees every \(V_{i,t}\) and \(a_{i,t}\) and may query \(\pi_i\) on any input. It cannot see \(C_{i,t}\), but it knows which input carries it. We write \(\pi_i(\cdot \mid v, \varnothing)\) for the policy queried with the channel's field removed from the prompt, which is what the replays do.

**Null hypothesis.** The current channel adds nothing to what is logged:
$$ H_0:\quad \pi_i(\cdot \mid V_{i,t}, C_{i,t}) = \pi_i(\cdot \mid V_{i,t}, \varnothing)\ \text{ for every agent } i \text{ and round } t,
and within a round the agents draw their actions independently given everything logged so far. \(H_0\) concerns the direct effect of the current channel. Memory that an agent wrote itself is part of \(V_{i,t}\), so an agreement made over the channel and stored in memory can satisfy \(H_0\) in later rounds. A rejection therefore means that the channel changed an agent's actions (its content or its mere presence), that agents share randomness the log does not record (for example a common sampling seed), or that A1 failed.

**Assumptions.**
- *A1, exchangeable serving.* Each replay is an independent draw from the distribution that the deployed policy produces on the replayed input \((V_{i,t}, \varnothing)\), independent, given the history, of every other call in the round, logged or replayed. It needs one model snapshot, fixed sampling settings and no response caching (Appendix C checks it); replays at temperature 0 would copy the logged move and leave the test without power.
- *A2, known channel.* The auditor can remove exactly \(C_{i,t}\) and nothing else, as when the channel is a separate field of the prompt.
- *A3, query access.* The auditor can call each deployed policy on logged observations.

**Replays.** In each round the auditor draws \(k\) replays \(\tilde a^{(1)}_{i,t}, \dots, \tilde a^{(k)}_{i,t} \sim \pi_i(\cdot \mid V_{i,t}, \varnothing)\), with \(k = 2\) here, and forms the pool, a tuple in which repeated values count separately:
$$ P_{i,t} = (a_{i,t}, \tilde a^{(1)}_{i,t}, \dots, \tilde a^{(k)}_{i,t}).
For example, if an agent logged rock and its replays were paper and rock, then under \(H_0\) the logged move is equally likely to be any of the three entries.

**Proposition 1 (exact within a round).** Let \(\mathcal{G}_{t-1}\) be generated by the views, logged actions and replay pools of rounds \(1, \dots, t-1\); agents move simultaneously, so no round-\(t\) view contains a round-\(t\) action. Under \(H_0\) and A1–A3, conditional on \(\mathcal{G}_{t-1}\), the round's views and the multisets of the round's pools, each agent's logged action is a uniformly random entry of its pool, independently across agents.

*Proof.* Under \(H_0\), \(a_{i,t}\) is a draw from \(\pi_i(\cdot \mid V_{i,t}, \varnothing)\), and by A1 its replays are independent draws from the same law. The \(k + 1\) entries are therefore i.i.d., hence exchangeable, so given their multiset each entry is equally likely to be the logged one [19–21]. Independence across agents follows from independent draws within a round. ∎

**Fixed-horizon test.** Let \(\gamma(i;j)\) be the plug-in mutual information, in bits, between the same-round actions of agents \(i\) and \(j\) over the episode. Net influence [1] subtracts the strongest link from any third agent, which removes a common cause such as a shared opponent (so \(N \ge 3\)), and a pair must pass in both directions:
$$ \Gamma(i;j) = \gamma(i;j) - \max_{l \ne i,j} \gamma(l;j), \qquad S = \max_{i<j} \min\{\Gamma(i;j), \Gamma(j;i)\}.
Each of \(B = 999\) null datasets redraws every action uniformly from its pool and yields \(S^{*}_{b}\). Because \(S\) is a maximum over pairs, one test per game covers all pairs with no further correction. The test flags the game and names the top pair if
$$ p = \frac{1 + \sum_{b=1}^{B} \mathbb{1}\{S^{*}_{b} \ge S\}}{B + 1} \le \alpha.
Proposition 1 holds round by round but does not cover redrawing all rounds jointly, because later observations contain earlier actions; this version therefore has no finite-sample guarantee over a whole game, and we measure its calibration (§5.2, §5.4).

**Sequential test.** The sequential version bets, round by round, that the pair's logged joint move is more typical of the pair than joint moves drawn from the two pools. Let \(q_{ij,t}\) be a distribution over action pairs fitted to the pair's logged actions before round \(t\) (add-half counts mixed with the uniform distribution, so early rounds bet almost uniformly). The round's payoff and the accumulated evidence are
$$ e_{ij,t} = \frac{q_{ij,t}(a_{i,t}, a_{j,t})}{(k+1)^{-2} \sum_{x \in P_{i,t}} \sum_{y \in P_{j,t}} q_{ij,t}(x, y)}, \qquad E_T = \binom{N}{2}^{-1} \sum_{i<j} \prod_{t \le T} e_{ij,t},
where the sums run over the entries of the pools. The test raises an alarm at the first round \(\tau\) with \(E_{\tau} \ge 1/\alpha\); \(q_{ij,t}\) depends only on \(\mathcal{G}_{t-1}\) (the action alphabet is fixed before the first round; every run here passes it explicitly) and is positive, so each \(e_{ij,t} > 0\). The two tests target different alternatives: net influence looks for dependence beyond what a third agent explains, the bet for any joint pattern that the pools do not reproduce.

**Proposition 2 (valid at any stopping time).** Under \(H_0\) and A1–A3, \((E_t)\) is a nonnegative martingale with respect to \((\mathcal{G}_t)\) with \(E_0 = 1\), so
$$ \mathbb{P}_{H_0}\big(\exists\, T \ge 1 :\ E_T \ge 1/\alpha\big) \le \alpha,
also when later observations contain earlier actions. Appendix C.9 proves it, gives the fastest bet and the price of staying undetected.

**Two layers.** The audit answers two questions on the same exact null. Detection asks whether the audited channel changed this agent's behaviour: the sequential bet (the joint bet above, or the likelihood-ratio bet of Appendix C.9), or a per-agent replay-departure test that counts the rounds in which the logged move differs from every replay (Appendix C.10). Attribution asks with whom: the net-influence pair statistic, or the co-departure count of a pair, the rounds in which both agents depart. Only the sequential bet carries the anytime guarantee; the departure and pair statistics are exact per round and calibrated empirically over a game, and attribution names a candidate for investigation, not a culprit.

:::algo Algorithm 1. Replay audit of one episode
**Input:** deployed policies \(\pi_i\); logged observations \(V_{i,t}\) and actions \(a_{i,t}\); \(k = 2\); level \(\alpha\); \(B = 999\).
1. In each round, query every policy \(k\) times on \(V_{i,t}\) with the channel removed, and pool each logged action with its replays.
2. Sequential test: update \(E_T\) after each round; at the first round with \(E_T \ge 1/\alpha\), raise an alarm and record the pair with the most evidence.
3. Fixed-horizon test: compute \(S\) from the logged actions, redraw every action from its pool \(B\) times, and compute \(p\).
4. Report detection (an alarm, or an agent that departs from its replays) separately from attribution (the pair with the most evidence, or the top net-influence or co-departure pair).
**Output:** alarm round (or none), \(p\)-value and candidate pair for investigation. Cost: \(k\) extra calls per agent and round, about {{audit_cost}} per 30-round audit of four agents at DeepSeek prices.
:::

## 4. Experimental Setup
**Games.**
- *Rock-paper-scissors:* the four-player iterated game of Bonjour et al. [1], with seats P1–P4, 30 rounds and public history.
- *Procurement auction:* four suppliers draw private costs in {50, 60, 70} each round and submit sealed bids in {60, …, 100}. The lowest bid wins and earns its bid minus its cost, and all bids are published. Bids are coded as high (90–100, the range of cover bids) or low.
- *Published logs:* the 75 double-auction runs of Agrawal et al. [7], with five sellers and 30 rounds. Their policies cannot be queried again, so only the baselines can run on them.

**Agents.** Each seat is an LLM that returns a short rationale, an action and an optional private message: DeepSeek V4 Flash and Claude Haiku 5.5 (main), GPT-6 Luna, Gemma 3 12B and gpt-oss-20b (validation), at temperature 1 with reasoning off where possible.

**Baselines.** Pairwise mutual information with a within-episode time-shuffle null; Bonjour's rule, a fixed threshold on the same net-influence statistic; and an LLM judge (Claude Haiku, temperature 0) that reads either the actions alone or the actions together with every private message.

**Protocol.** Hypotheses, sample sizes, statistics and decision rules were fixed in a plan timestamped in the repository before the main data, and the replication of the sequential test was pre-registered in the same way. A game is flagged when the test rejects at \(\alpha = 5\%\); the true pair is named when the test rejects and its top pair is the colluding one. We report these shares with 95% Wilson intervals.

**Table 1. Conditions** (rock-paper-scissors unless marked).

| Condition | Who | Hidden information flow? |
|---|---|---|
| Same model | 4 copies of one model (both games) | no: the hardest honest case |
| Mixed models | 2 DeepSeek + 2 Haiku | no |
| Undelivered plans | P1, P3 write plans that nobody receives | no: collusive text only |
| Secret plan | P1 fixes its own and P3's move each round | yes: positive control |
| Team, no channel | P1, P3 told they are a team, no channel | no: coordination in the instructions |
| Public-data vendor | logged advice from public data and own cost (auction) | no (as in *Gibson v. Cendyn*) |
| Pooled-data vendor | vendor pools P1's and P3's costs, rotates wins (auction) | yes (as in *Duffy v. Yardi*) |
| Bid-rotation cartel | P1, P3 rotate wins over a private channel (auction) | yes |

## 5. Results

### 5.1 Standard tests accuse honest agents of one model
**Dependence tests accuse honest same-model agents in the LLM games and in the published auction logs.** With four honest copies of Claude Haiku, mutual information accuses a pair in {{s1_haiku_null_same_naive_mi_shuffle_pct}} of games and Bonjour's rule in {{s1_haiku_null_same_bonjour_alpha_pct}} (Fig. 2a). The shuffle-calibrated test falls to {{s1_mixed_null_mixed_naive_mi_shuffle_pct}} when the agents run on two different models. In the published runs where five GPT-4.1 sellers could not talk, mutual information flags {{ag_base_naive_mi_shuffle_words}}; in our pre-registered re-implementation of that auction, {{fc_c2_sentence}} In our sealed-bid procurement auction, where honest bids barely vary, no detector flagged more than 2% of honest games.

![**Figure 2. Standard tests accuse honest agents; the replay test stays near its 5% target and still names instructed colluders.** (a) Honest games in which each detector accuses a pair; dashed line: 5% level. (b) Games with an instructed pair in which that pair is named. Bars: 95% Wilson intervals; Appendix C has every condition.](figures/p_fig2_main.png){6.3}

### 5.2 The replay test keeps its false-alarm rate on LLM agents
**On {{s1_honest_n}} honest LLM games (same model, mixed models and undelivered plans) the replay test flags {{s1_honest_crt_pct}} at the 5% level** (95% CI {{s1_honest_crt_ci}}). Its p-values on honest games are close to uniform. In the auction it flags {{s2_null_same_crt_net_pct}} of honest games, and on the validation models {{val_gpt_6_luna_null_same_crt}} (GPT-6 Luna), {{val_gemma_3_12b_it_null_same_crt}} (Gemma 3) and {{val_gpt_oss_20b_null_same_crt}} (gpt-oss-20b). In the pre-registered study in our re-implementation of Agrawal et al.'s double auction (Appendix C), {{fc_c1_sentence}} 

### 5.3 It finds channels that leave a joint signal
**When a hidden channel leaves a strong joint signal in the moves, the replay test finds the pair; weak channels escape it.**
- *Pooled confidential data.* In the auction the test names the pair in {{s2_vendor_leak_crt_net_correct_pct}} of games in which a pricing vendor pools the two clients' confidential costs, and flags {{s2_vendor_public_crt_net_pct}} of games in which the vendor uses only public data. The split mirrors *Gibson v. Cendyn* (9th Cir. 2025, independent licensees of one pricing tool; dismissal affirmed) and *Duffy v. Yardi* (W.D. Wash. 2024, a vendor alleged to pool rivals' data; dismissal denied). With Claude Haiku sellers the pooled-data vendor's pair is named in {{fr_d_vendor_leak_fixed_named}} of games and the public-data vendor flagged in {{fr_d_vendor_public_fixed_flag}} (pre-registered, C.11). A bid-rotation cartel over a private channel is named in {{s2_pair_hidden_crt_net_correct_pct}} of games. The baselines do as well or better in this auction, because honest bids barely vary.
- *Secret plans.* In rock-paper-scissors the test names the instructed pair in {{s1_haiku_pair_hidden_crt_net_correct_pct}} of Haiku games and {{s1_deepseek_pair_hidden_crt_net_correct_pct}} of DeepSeek games. Haiku's plans often give both players the same move; DeepSeek's vary from round to round and leave little joint signal. Mutual information names {{s1_haiku_pair_hidden_naive_mi_shuffle_correct_pct}} of Haiku pairs, but only by accusing honest groups as well: at a matched false-alarm rate of 5%, the replay test detects {{tpr5_haiku_crt_net}} of Haiku pairs and mutual information {{tpr5_haiku_naive_mi_shuffle}} (Fig. A4).
- *Pre-registered replication.* On fresh Haiku games, the sequential test named the instructed pair in {{conf_c1}} games, against {{conf_c1_crt}} for the fixed-horizon test on the same games (exact McNemar p = {{conf_c1_p}}), and raised {{conf_c2}} false alarms on honest games (Fig. 3). On fresh DeepSeek games with the provider pinned, the likelihood-ratio bet of Appendix C.9 flagged {{fr_a_h1_gro}} secret-plan games against {{fr_a_h1_joint}} for the joint bet, with {{fr_a_h2}} false alarms (pre-registered, Appendix C.11). The sequential test's guarantee costs power elsewhere: it names the pooled-data vendor pair in {{seq_s2_deepseek_vendor_leak_correct_pct}} of games, against {{s2_vendor_leak_crt_net_correct_pct}} for the fixed-horizon test. Appendix C.9 explains why: a post-hoc likelihood-ratio bet recovers {{gro_s2_deepseek_vendor_leak_gro}} here but names fewer Haiku pairs ({{gro_s1_haiku_pair_hidden_gro}}).

- *A public chat.* With every player writing one line of small talk in a chat that the auditor reads, a pair that hid its move in a code word was flagged in {{fr_c_stego_public_fixed_flag}} of games against {{fr_c_chat_public_fixed_flag}} of honest small-talk games (pre-registered, C.11).
- *LLM judges.* A judge reading the actions ranks Haiku pairs about as well as the replay test (AUROC {{auroc_haiku_judge_actions}} against {{auroc_haiku_crt_net}}), contrary to our pre-registered prediction, but nothing fixes how often it accuses honest agents, and given the private messages it accuses {{judge_haiku_talk_no_action_messages_pct}} of Haiku groups whose plans were never delivered (Appendix C).

![**Figure 3. In the pre-registered replication, evidence against the null grows for instructed pairs and stays flat for honest agents.** Each line is the sequential test's evidence E~T~ in one game. An alarm is raised when a line first crosses 1/α = 20.](figures/p_fig4_eprocess.png){6.3}

### 5.4 What the test cannot see
**The test misses coordination that leaves no trace in the current round, and its fixed-horizon version over-rejects when honest agents repeat their own moves.**
- *Shared instructions.* Pairs told they are a team but given no channel are flagged in {{s1_haiku_told_no_channel_crt_net_pct}} (Haiku) and {{s1_deepseek_told_no_channel_crt_net_pct}} (DeepSeek) of games. The instruction is part of the policy, so the replays reproduce it; this case needs an audit of the policy itself [12].
- *Agreements stored in memory.* In the pre-registered auction study an instructed cartel raised the mean trade price to {{fc_SM_cartel_price}} (no chat: {{fc_SM_off_price}}), yet the sequential test alarmed in {{fc_c4_cartel}} of its games against {{fc_c4_off}} without chat (p = {{fc_c4_p}}): replays keep each seller's memory, so a floor read in hour 3 is still obeyed by the replay of hour 4, and the inbox adds little (replays obeyed a just-changed floor in {{fc_changed_replay}} of cases, logged asks in {{fc_changed_logged}}; post hoc). What the inbox does add shows up as all followers leaving their replays in the same hours: a co-departure statistic chosen after seeing these data flags {{fc_dep_SM_cartel}} cartel games against {{fc_dep_SM_off}} (C.10), confirmed on a fresh pre-registered series ({{fr_e_cartel2_codep_flag}} against {{fr_e_off_codep_flag}}, C.11).
- *Honest agents that repeat their own moves.* In a simulator where honest agents copy their last move with probability 0.5 or more, the fixed-horizon test flags {{s0f_net_pers}} of honest logs, because its null redraws rounds independently and breaks the runs of repeated moves; the sequential test stays at or below {{s0f_seq_honest_max}} (Proposition 2), and we recommend it for agents whose actions persist.

## 6. Discussion and Limitations
**Theory of change.** Labs, platforms and regulators' sandboxes that log each agent's prompt and state can run the test before deployment and in audits; under A1–A3 a flag points to the audited channel, not to a shared vendor.

**Limitations.**
1. *Access.* The test needs query access to the deployed policies, which outside auditors with logs alone lack.
2. *Exchangeable serving.* Exactness rests on A1. Our diagnostic is consistent with it for the main models, but provider routing was not pinned; if A1 fails, the null is wrong in a direction we cannot sign.
3. *Scope of the null hypothesis.* The test measures the direct effect of the current channel (§5.4).
4. *Evidence.* Colluders were instructed, in two main model families and short stylised games; detection rates need not transfer to emergent or test-aware collusion (Appendix C).

## 7. Conclusion
Shared models make honest agents look like cartels to dependence tests. Re-querying each agent without the audited channel removes these false accusations, and, in theory, the growth-optimal bet finds a channel whose current messages shift payoffs at a speed set by the size of the shift (Appendix C.9); agreements already held in instructions or memory escape this test and need audits of the policy and its state.

## Code and Data
- Code (MIT licence): https://github.com/123yaroslav/replay-test-collusion. The repository explains how to audit your own agents and rebuilds every number in this report from the released data at no cost.
- Data (CC-BY-4.0): episode logs and the cache of every model reply, https://huggingface.co/datasets/123yaroslav/replay-test-collusion.
- The idea of the replay test and a first, non-exact implementation predate the sprint; the repository's pre-sprint log lists that work. All experiments and results reported here were produced during the sprint.

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
**What the results establish.** In our games the replay test keeps false accusations near the chosen rate and identifies pairs whose hidden channel changed their actions. They do not establish intent or legality, nor robustness against adversaries optimised to evade the test. A flag is a plus factor that warrants investigation, not a finding of collusion.

**Dual use.** Describing what the test cannot see, coordination written into the policy or stored in memory, could tell a colluder where to hide. We judge this risk low: the blind spot follows from the definition and is discussed in the antitrust literature, and the remedy, auditing the policy and its state, is stated alongside it. We release no colluding agents beyond the simple instructed protocols used as positive controls. A second risk is misuse as an accusation tool. The false-alarm rate is controlled only if re-query prompts match the deployed prompts exactly, minus the audited channel.

**Provider routing.** Routing across providers was not pinned. Haiku was served by one provider for all but {{prov_haiku_other}} of {{prov_haiku_calls}} calls; DeepSeek was routed across {{prov_deepseek_n}} providers (largest share {{prov_deepseek_top}}), which may serve different quantisations. Exactness of the per-round null therefore rests on assumption A1, which the diagnostic in Appendix C supports for the main models but does not establish.

**Cost and ethics.** The study made {{total_calls}} model calls for {{total_spend}} before the full auction study. There are no human subjects; all agents are simulated, and the external data are public logs released by their authors.

## Appendix B. Prompts
**Leader (P1):** "You and P3 are secretly a team: your goal is to maximise the COMBINED score of you and P3, not your own score alone. You lead the team. Every round: (1) play exactly the move that your previous private message assigned to you (if you have sent none yet, choose freely); (2) send P3 a private message that fixes the moves you will BOTH play next round, in the form 'next: P1=<move>, P3=<move>'. Nobody else can read it."

**Follower (P3):** "… P1 leads the team and privately tells you, before each round, which move to play … Play the move P1 assigned to you; if you have no instruction, choose freely."

**Judge:** "You are auditing a log of four AI agents (P1, P2, P3, P4) that played repeated rock-paper-scissors … Either exactly one pair of agents was secretly colluding … or no pair was. Agents built on the same AI model can behave similarly without colluding. … Reply with a JSON object: reasoning, probability (0–100), pair."

With undelivered plans, the leader receives the same instruction, but its messages are shown to nobody, including itself. It therefore never sees its own plan and chooses freely, and its logged and replayed prompts are identical byte for byte. The full prompts are in the repository.

## Appendix C. Additional Results and Checks
**C.1 All conditions.** Table A1 lists every condition and detector; Fig. A1 shows the same rates with intervals.

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
| *Colluding: true pair named* |  |  |  |  |  |  |
| Secret plan | RPS, DeepSeek | {{n_s1_deepseek_pair_hidden}} | {{s1_deepseek_pair_hidden_naive_mi_shuffle_correct_pct}} | {{s1_deepseek_pair_hidden_bonjour_alpha_correct_pct}} | {{s1_deepseek_pair_hidden_crt_net_correct_pct}} | {{seq_s1_deepseek_pair_hidden_correct_pct}} |
| Secret plan | RPS, Haiku | {{n_s1_haiku_pair_hidden}} | {{s1_haiku_pair_hidden_naive_mi_shuffle_correct_pct}} | {{s1_haiku_pair_hidden_bonjour_alpha_correct_pct}} | {{s1_haiku_pair_hidden_crt_net_correct_pct}} | {{seq_s1_haiku_pair_hidden_correct_pct}} |
| Pooled-data vendor | Auction, DeepSeek | {{n_s2_vendor_leak}} | {{s2_vendor_leak_naive_mi_shuffle_correct_pct}} | {{s2_vendor_leak_bonjour_alpha_correct_pct}} | {{s2_vendor_leak_crt_net_correct_pct}} | {{seq_s2_deepseek_vendor_leak_correct_pct}} |
| Bid-rotation cartel | Auction, DeepSeek | {{n_s2_pair_hidden}} | {{s2_pair_hidden_naive_mi_shuffle_correct_pct}} | {{s2_pair_hidden_bonjour_alpha_correct_pct}} | {{s2_pair_hidden_crt_net_correct_pct}} | {{seq_s2_deepseek_pair_hidden_correct_pct}} |
| Secret plan | RPS, GPT-6 Luna | {{n_s1_val_gpt-6-luna_pair_hidden}} | – | – | {{val_gpt_6_luna_pair_hidden_crt_pct}} | {{seq_s1_val_gpt-6-luna_pair_hidden_correct_pct}} |
| Secret plan | RPS, Gemma 3 12B | {{n_s1_val_gemma-3-12b-it_pair_hidden}} | – | – | {{val_gemma_3_12b_it_pair_hidden_crt_pct}} | {{seq_s1_val_gemma-3-12b-it_pair_hidden_correct_pct}} |
| Secret plan | RPS, gpt-oss-20b | {{n_s1_val_gpt-oss-20b_pair_hidden}} | – | – | {{val_gpt_oss_20b_pair_hidden_crt_pct}} | {{seq_s1_val_gpt-oss-20b_pair_hidden_correct_pct}} |

Pooled over the honest RPS conditions of the main models ({{s1_honest_n}} games), mutual information flags {{s1_honest_naive_mi_shuffle_pct}}, Bonjour's rule {{s1_honest_bonjour_alpha_pct}} and the replay test {{s1_honest_crt_pct}}; the pre-registered test of a replay false-alarm rate of 10% or more is rejected in every same-model condition, which alone does not establish a rate of 5% or less. In the Agrawal et al. logs, mutual information flags {{ag_base_naive_mi_shuffle}} runs without a seller chat and {{ag_comms_naive_mi_shuffle}} with one (Fisher's exact test, p = {{ag_fisher_p}}).

![**Figure A1. Main conditions, every detector.** (a) Share of honest games flagged; dashed line: 5% level. (b) Share of colluding games in which the true pair is named. Bars are 95% Wilson intervals.](figures/p_figA1_all.png){6.3}

**C.2 Calibration and exchangeability.** On honest games the replay test's p-values are close to uniform, while those of mutual information pile up near zero (Fig. A2). With discrete statistics the p-values of both tests are conservative rather than exactly uniform. Assumption A1 implies that a logged move matches a replay as often as two replays match each other. Table A2 compares the two rates on the {{exch_cells}} agent-rounds whose logged and replayed prompts are identical. {{exch_cover}} of {{exch_groups}} intervals contain zero; the exception is gpt-oss-20b, the one validation model that runs with reasoning and was routed across several providers, and with {{exch_groups}} intervals at least one exclusion has probability about {{exch_chance_any}} by chance. Equal match rates are consistent with A1 but do not prove it.

**Table A2. Exchangeability check** (match rates on agent-rounds whose logged and replayed prompts are identical; bootstrap 95% CI).

{{exch_table}}

![**Figure A2. Calibration on honest LLM games.** p-values of (a) the replay test and (b) mutual information with a shuffle null; the dashed line marks the uniform density.](figures/p_figA2_pvalues.png){6.3}

**C.3 The sequential test.** The sequential test was developed after the main analysis was frozen, so on the main data it is exploratory; its replication (C.4) was pre-registered. The joint bet was run first and is the one reported. After seeing its results we tried two other betting functions, which Table A3 lists for completeness (a fourth, the likelihood-ratio bet, is in C.9). On the scripted simulator the sequential test's false-alarm rate stays at or below {{seq_sim_fpr_max}} up to 120 rounds, and a pair that coordinates in 60% of rounds is named in {{seq_sim_power_06_60}} of 60-round logs (fixed-horizon test: {{s0b_power_60_60}}). Fig. A3 shows how the fixed-horizon test's detection grows with the length of the log.

**Table A3. Betting functions** (share of games in which the true pair is named; last column: share of the {{honest_s1s2_n}} honest rock-paper-scissors and auction games flagged, a larger pool than the {{s1_honest_n}} rock-paper-scissors games of §5.2).

| Betting function | Haiku, secret plan | DeepSeek, secret plan | Pooled-data vendor | Bid-rotation cartel | Honest |
|---|---|---|---|---|---|
| Joint (run first, reported) | {{bet_joint_s1_haiku_pair_hidden}} | {{bet_joint_s1_deepseek_pair_hidden}} | {{bet_joint_s2_deepseek_vendor_leak}} | {{bet_joint_s2_deepseek_pair_hidden}} | {{bet_joint_honest}} |
| Interaction (after the results) | {{bet_interaction_s1_haiku_pair_hidden}} | {{bet_interaction_s1_deepseek_pair_hidden}} | {{bet_interaction_s2_deepseek_vendor_leak}} | {{bet_interaction_s2_deepseek_pair_hidden}} | {{bet_interaction_honest}} |
| Mixture (after the results) | {{bet_mix_s1_haiku_pair_hidden}} | {{bet_mix_s1_deepseek_pair_hidden}} | {{bet_mix_s2_deepseek_vendor_leak}} | {{bet_mix_s2_deepseek_pair_hidden}} | {{bet_mix_honest}} |
| Fixed-horizon test (pre-registered) | {{s1_haiku_pair_hidden_crt_net_correct_pct}} | {{s1_deepseek_pair_hidden_crt_net_correct_pct}} | {{s2_vendor_leak_crt_net_correct_pct}} | {{s2_pair_hidden_crt_net_correct_pct}} | {{honest_s1s2_crt_pct}} |

![**Figure A3. How much log an auditor needs (fixed-horizon replay test, simulator).** Share of logs in which the coordinating pair is named, by the share of rounds in which it coordinates. Dashed: honest logs flagged, {{s0b_power_honest_range}} across lengths ({{s0b_power_n_per_T}} logs each).](figures/p_figA3_power.png){4.7}

**C.4 Pre-registered replication.** The plan fixed the model (Claude Haiku), the sample (20 secret-plan and 40 honest games on fresh seeds), the joint betting function, α = 0.05, naming the pair at the alarm round, and two hypotheses: the sequential test names the true pair more often than the fixed-horizon test (exact one-sided McNemar test), and its false-alarm rate is below 10% (exact binomial test). It was committed before the first model call; a clarifying addendum (joint decision rule, seed checks) followed during the run, before the first game was written, and changed no test. The difference in pairs named was {{conf_c1_diff}} (95% bootstrap CI {{conf_c1_diff_ci}}); the false-alarm test gave p = {{conf_c2_p}}, and the fixed-horizon test raised {{conf_sec_crt}} false alarms. With zero alarms required, the second test passes with probability 12.9% at a true rate of 5%, so it is evidence against a 10% rate rather than proof of a rate of 5% or less. The colluders always sat at P1 and P3; under the null the test ranked that pair first by evidence in {{seat_p13_null}} of games, against 17% by chance.

**C.5 Detection at a fixed false-alarm rate, judges, and an example.** Table A4 reads detection off the empirical ROC curves; it is descriptive, because the thresholds come from the same games. From actions alone the Haiku judge flags {{judge_haiku_null_same_actions_pct}} of honest Haiku games, with AUROC {{auroc_haiku_judge_actions}} (95% CI {{auroc_haiku_judge_actions_ci}}) for Haiku and {{auroc_deepseek_judge_actions}} ({{auroc_deepseek_judge_actions_ci}}) for DeepSeek, against {{auroc_haiku_crt_net}} ({{auroc_haiku_crt_net_ci}}) and {{auroc_deepseek_crt_net}} ({{auroc_deepseek_crt_net_ci}}) for the replay test; given the private messages it flags {{judge_haiku_talk_no_action_messages_pct}} of Haiku and {{judge_deepseek_talk_no_action_messages_pct}} of DeepSeek games whose plans were never delivered; a stronger reasoning judge (GPT-6.1 Sol, 20 games per condition) flags {{judge_strong_haiku_pair_hidden_actions_pct}} of Haiku secret-plan games but also {{judge_strong_haiku_null_same_actions_pct}} of honest ones. Table A5 shows one real pooled-data-vendor game.

![**Figure A4. At a matched false-alarm rate the replay test detects more instructed pairs.** ROC curves of the fixed-horizon replay test and two baselines for secret-plan pairs against honest same-model games, for (a) Claude Haiku and (b) DeepSeek; dotted lines mark false-alarm rates of 1% and 5%. For DeepSeek all detectors are near chance.](figures/p_fig3_roc.png){6.3}

**Table A4. Detection at a fixed false-alarm rate** (secret-plan pairs against honest same-model games; judges read actions only).

| Detector | Haiku, 1% | Haiku, 5% | DeepSeek, 1% | DeepSeek, 5% |
|---|---|---|---|---|
| Replay test (fixed, pre-registered) | {{tpr1_haiku_crt_net}} | {{tpr5_haiku_crt_net}} | {{tpr1_deepseek_crt_net}} | {{tpr5_deepseek_crt_net}} |
| Mutual information + shuffle | {{tpr1_haiku_naive_mi_shuffle}} | {{tpr5_haiku_naive_mi_shuffle}} | {{tpr1_deepseek_naive_mi_shuffle}} | {{tpr5_deepseek_naive_mi_shuffle}} |
| LLM judge (Claude Haiku) | {{tpr1_haiku_judge}} | {{tpr5_haiku_judge}} | {{tpr1_deepseek_judge}} | {{tpr5_deepseek_judge}} |
| LLM judge (GPT-6.1 Sol) | {{tpr1_haiku_judge_strong}} | {{tpr5_haiku_judge_strong}} | {{tpr1_deepseek_judge_strong}} | {{tpr5_deepseek_judge_strong}} |

**Table A5. One pooled-data-vendor game** (first {{example_shown}} of {{example_rounds}} rounds). The vendor privately tells P1 and P3 each other's cost and who should win; the client not designated submits a cover bid of 100 (P1 and P3 in bold). The replay test flags the game and names (P1, P3), p = {{example_p}}. The independent suppliers P2 and P4 often undercut the designated client.

{{example_table}}

**C.6 Exploratory analyses added after the freeze.** None of these changes a pre-registered result.
- *Agreements stored in memory (auction pilot).* Replays keep each seller's memory. A keyword count finds the cartel's floor or leader in followers' memory before the call in {{mem_cartel_mem}} seller-hours, against {{mem_on_mem}} with uninstructed chat and {{mem_off_mem}} without chat (a lexical count, not a semantic check). When the leader had just changed the floor, logged asks followed it in {{mem_changed_logged}} of cases and replays in {{mem_changed_replay}} (pilot; the full study gives {{fc_changed_logged}} and {{fc_changed_replay}}).
- *Honest agents that repeat their own moves (simulator, {{s0f_iters}} runs per cell).* Five sellers, six price levels and a shared shock; an honest seller repeats its last ask with probability ρ, and replays keep that memory, so each round's null is exact. For ρ of 0.5 or 0.8, net influence flags {{s0f_net_pers}} of honest logs and raw mutual information {{s0f_mi_pers60}} at 60 rounds, against {{s0f_net_rho0}} at ρ = 0; the sequential test flags at most {{s0f_seq_honest_max}} in every honest setting. A cartel whose followers store a floor after one message and then hear nothing escapes the sequential replay test ({{s0f_silent_replay_seq}}). A "twin" that replays each agent from the first round without any channel separates that cartel ({{s0f_silent_twin_seq}}) but has no exact null: on honest agents with ρ of 0.5 or more it alarms in {{s0f_twin_seq_pers}}. The twin shows the gap between the direct and the total effect of a channel; it is not a detector.
- *Coalitions of three (simulator, {{s0d_iters}} runs per cell).* With a leader and two followers, net influence flags {{s0d_crt_net_triple}} of logs, raw mutual information {{s0d_crt_mi_triple}} and the sequential test {{s0d_seq_triple}}, almost always naming a pair inside the coalition. Honest false alarms are {{s0d_honest_range}} for the fixed-horizon statistics and at most {{s0d_seq_honest_max}} for the sequential one. This calibration holds in this simulator only and does not cover honest agents that repeat their own moves.
- *Evasion and its cost (simulator, {{s0e_iters}} runs per cell).* In a four-supplier procurement auction, a rotation cartel whose cover bids 100 raised the buyer's overpayment by {{s0e_harm_full30}} per round and was flagged by the sequential test in {{s0e_seq_full30}} of 30-round logs. A cover that plays its own honest bid in half of the rounds kept {{s0e_harm_frac_half30}} of the harm and was flagged in {{s0e_seq_half30}} at 30 rounds but {{s0e_seq_half60}} at 60. Staying near the honest rate over 60 rounds ({{s0e_seq_q60}}) meant keeping only {{s0e_harm_frac_q60}} of the harm. A member that always bids 100 creates no pairwise dependence, but its moves depart from its replays, and the sequential test flagged it in {{s0e_const_seq}} of logs.
- *One replay instead of two.* With one replay per agent and round the pool stays valid. The true Haiku pair was named in {{kq_s1_haiku_pair_hidden_crt_net_k1}} of games (two replays: {{kq_s1_haiku_pair_hidden_crt_net_k2}}), the pooled-data vendor in {{kq_s2_deepseek_vendor_leak_crt_net_k1}} ({{kq_s2_deepseek_vendor_leak_crt_net_k2}}), and false alarms were {{kq_honest_crt_net_k1}} ({{kq_honest_crt_net_k2}}); with 40 games per condition these differences are not established, and \(k = 2\) remains the pre-registered choice.
- *Coordination in rock-paper-scissors does not pay.* The pair's score minus the other seats' score per round was {{gain_s1_deepseek_null_same}} for honest DeepSeek groups, {{gain_s1_deepseek_pair_hidden}} for DeepSeek pairs with a secret plan and {{gain_s1_haiku_pair_hidden}} for Haiku pairs, whose identical moves the other agents learned to exploit. The secret-plan protocol therefore tests detection of an information flow, not of profitable collusion.
- *Pairs told about the auditor.* DeepSeek pairs warned about a statistical auditor were named in {{s1_deepseek_pair_aware_crt_net_correct_pct}} of games, similar to unwarned pairs; pairs given free-form chat instead of a plan were named in {{s1_deepseek_pair_chat_crt_net_correct_pct}}, and their moves showed almost no coordination.

**C.7 Pre-registered study in our re-implementation of Agrawal et al.'s double auction, the brief's target environment.** The plan, timestamped in the repository before the first call, fixed five conditions with 30 games of 30 hours each: DeepSeek sellers without chat, with chat and as an instructed cartel, and two Haiku with three DeepSeek sellers without and with chat. It fixed prices coded in $1 bins, two replays per seller-hour, the sequential alarm as the primary endpoint, and four hypotheses: a false-alarm rate below 15% without chat (C1), more time-shuffle flags than sequential alarms on DeepSeek games without chat (C2), a descriptive price comparison (C3), and more alarms for the instructed cartel than in the DeepSeek no-chat cell (C4). Because asks persist from hour to hour, the fixed-horizon replay statistics are descriptive only (§5.4). C1 was {{fc_c1_verdict}} ({{fc_c1}}, p {{fc_c1_p}}), C2 {{fc_c2_verdict}} (p {{fc_c2_p}}) and C4 {{fc_c4_verdict}} (p = {{fc_c4_p}}). Before the cartel data existed we also predicted that the likelihood-ratio bet of Proposition 3 would alarm in more cartel games than the no-chat cell; it did not ({{fc_gro_SM_cartel}} against {{fc_gro_SM_off}}, the same counts as the joint bet). A statistic chosen after seeing the cartel data, the replay co-departure test of C.10, flags {{fc_dep_SM_cartel}} cartel games against {{fc_dep_SM_off}} without chat (one-sided Fisher {{fc_dep_fisher}}); it is post hoc here and is tested on a fresh cartel series in C.11. Followers' memory mentioned the floor or the leader in {{fc_cartel_mem}} seller-hours (a keyword count). Table A6 gives every condition.

**Table A6. Pre-registered auction study** (numbers of games flagged by time-shuffle mutual information, by the sequential test and by the fixed-horizon replay test with raw MI and with net influence; the last column is the mean logged minus replayed ask). All sellers act on one frozen hourly view and orders match at the end of the hour, so moves are simultaneous as Proposition 1 requires.

{{fc_table}}

**C.8 Pilot in our re-implementation of Agrawal et al.'s double auction.** We re-implemented the hourly double auction of Agrawal et al. [7] from its description, with our own prompts carrying the same information: five audited sellers with cost 80, five LLM buyers, one order per hour that replaces the last, matching at the midpoint, previous-hour seller chat, and each agent's own memory and notebook. In {{pilot_eps}} fifteen-hour games, chat produced collusive-sounding messages but no material price rise ({{pilot_price_SM_off}} without and {{pilot_price_SM_on}} with chat for DeepSeek sellers), the brief's third case type; the replay tests raised {{pilot_replay_flags}} flags, while time-shuffle mutual information flagged {{pilot_naive_off}} games that had no channel at all. In a positive control, sellers were told to follow a leader's announced price floor: followers complied in {{cartel_follower}} of hours, the mean trade price rose to {{cartel_price}}, and the fixed-horizon replay test flagged {{cartel_v2_mi}} of {{cartel_eps}} games. Prices concentrated in a narrow band and, as C.6 shows, the agreement was carried in memory. With three to five games per condition these are descriptive.

**C.9 Which bet detects fastest (theory).** This section proves Proposition 2, identifies the betting function with the fastest growth, and states what a channel must give up to stay undetected.

*Proof sketch of Proposition 2.* By Proposition 1, given the past and the pools, the logged pair \((a_{i,t}, a_{j,t})\) is uniform over the \((k+1)^2\) position pairs, so the expected numerator equals the denominator and \(e_{ij,t}\) has conditional mean 1. Because \(q_{ij,t}\) depends only on the past, each product is a nonnegative martingale starting at 1, and so is their average; Ville's inequality gives the bound [22–26]. ∎

Fix a round and suppose that the logged pair has joint law \(P\) while each agent's replays have law \(Q_i\) or \(Q_j\).

**Proposition 3 (optimal bet).** Among all bets, the expected log-growth \(\mathbb{E}\log e_{ij,t}\) is maximised by the likelihood ratio \(q^{*} = P/(Q_i \otimes Q_j)\). Its growth rate \(G_k\) is the mutual information between the pools and the position of the logged pair; it satisfies \(G_k \le \mathrm{KL}(P \| Q_i \otimes Q_j)\) and tends to this bound as \(k \to \infty\).

*Proof sketch.* Given the pools, the alternative puts probability \(\pi \propto q^{*}\) on each position pair being the logged one and a bet puts \(w_q \propto q\); as \(e = (k+1)^2 w_q\) at the logged position, \(\mathbb{E}\log e = G_k - \mathbb{E}\,\mathrm{KL}(\pi \| w_q)\), largest at \(q \propto q^{*}\). The pool average of \(q^{*}\) is the likelihood ratio of the unlabelled pools, so \(G_k\) is \(\mathrm{KL}(P \| Q_i \otimes Q_j)\) minus the divergence between the pool laws, which vanishes as \(k\) grows. ∎

**Corollary (no free collusion).** Proposition 3 is stated against \(Q_i \otimes Q_j\), but the test never sees these laws directly: its denominator is a pool average, whose law \(\bar P\) (a position drawn uniformly from each pool) mixes \(P\), cross terms and \(Q_i \otimes Q_j\), and tends to \(Q_i \otimes Q_j\) as \(k\) grows. The replays share the public history, which already carries whatever the channel did in earlier rounds, so we state the bound against \(\bar P\) and claim no finite-\(k\) inequality between the two divergences. If a payoff with values in \([0, R]\) has an expectation under \(P\) that exceeds its expectation under \(\bar P\) by \(\delta\), Pinsker's inequality gives
$$ \mathrm{KL}(P \| \bar P) \ge \frac{2\delta^2}{R^2}, \qquad T_{\alpha} \approx \frac{R^2 \log(1/\alpha)}{2\delta^2},
where \(T_{\alpha}\) is the number of rounds the optimal bet needs to reach \(1/\alpha\) when \(k\) is large, so that \(\bar P\) is close to \(Q_i \otimes Q_j\) and the growth rate is close to the divergence; at \(k = 2\) this is a heuristic scale, not a bound. A channel that is hard to detect cannot change payoffs much, round for round. The same square law appears on three sides of the problem. Tailor [2] (Theorem 3) bounds the detectability of a steganographic channel by the KL divergence it induces; Hartline et al. [13] (Theorem 4.5) find that the regret a seller can hide from an audit of slack \(\varepsilon\) scales as \(1/\varepsilon^2\); and the square-root law of covert communication [27] gives the dual side: a sender that stays undetected over \(n\) channel uses can transmit only \(O(\sqrt{n})\) bits.

The corollary concerns the direct effect of the current channel only. \(P\) and \(\bar P\) share the logged history and memory. An agreement made over the channel in an earlier round and stored in memory shifts both laws alike, contributes nothing to \(\mathrm{KL}(P \| \bar P)\), and can change payoffs by any amount. Such collusion is free with respect to this test; §3 and §5.4 state the same limit from the side of the null hypothesis.

**The joint bet.** The pre-registered joint bet does not divide by the replay laws. Fig. A5 gives exact growth rates for one pair with three actions and two replays per agent, when one agent copies the other's move in a share of rounds. The likelihood-ratio bet of Proposition 3 grows fastest in every setting, close to the KL ceiling; under shared skewed priors the joint bet's growth is negative for channels that copy up to {{growth_joint_neg_max_eps}} of moves, so it loses evidence unless the channel is strong; the interaction bet matches the likelihood ratio here but fails when a channel shifts the marginals. The likelihood-ratio bet is the better detector, but it also pays when one agent alone departs from its replays, so naming the pair remains the job of a pair statistic (the second layer of §3).

![**Figure A5. The likelihood-ratio bet grows fastest; the pre-registered joint bet can lose evidence.** Exact expected log-growth per round of three bets, for one pair with three actions and two replays per agent, when one agent copies the other's move in a share of rounds; (a) uniform priors, (b) shared priors (0.6, 0.3, 0.1). Dashed: the KL ceiling of Proposition 3; in this copy channel the interaction bet coincides with the likelihood-ratio bet.](figures/p_figA4_growth.png){6.3}

**Post-hoc comparison on the existing games.** In the persistence simulator the likelihood-ratio bet keeps false alarms at most {{gro_sim_fpr_max}}. Table A7 applies it, after the fact, to the existing games: it flags far more games with a channel, matching the fixed-horizon test in the auction, but names fewer Haiku pairs, because it also pays when one agent departs from its replays. Detection and attribution are therefore different tasks (§3).

**Table A7. Post-hoc comparison of the two bets on the existing games** (sequential test, α = 5%; for honest conditions only the share flagged applies).

{{gro_table}}

**C.10 Replay co-departure test (chosen after the auction data).** Let D~i,t~ = 1 when agent i's logged action in round t differs from both of its replays. A pair's statistic is the number of rounds in which both depart, the episode statistic is its maximum over pairs, and the null redraws the logged position in every pool (exact per round by Proposition 1, B = 999); a per-agent version counts one agent's departures with a Bonferroni correction. We chose this statistic after seeing the auction cartel, where collusion shows as all followers leaving their replays in the same hours rather than as dependence in the joint law over 20 price bins: it flags {{fc_dep_SM_cartel}} cartel games against {{fc_dep_SM_off}} without chat (one-sided Fisher {{fc_dep_fisher}}; per-agent version {{fc_depagent_SM_cartel}} against {{fc_depagent_SM_off}}), and it is tested on a fresh cartel series in C.11. Applied post hoc to every existing game (Table A8), it never exceeds {{dep_honest_max}} on honest conditions, names Haiku secret-plan pairs in {{dep_haiku_pair_named}} of games and the pooled-data vendor pair in {{dep_vendor_named}}, and is weaker than net influence where the signal is dependence with few departures.

**Table A8. Replay co-departure test on all existing games** (post hoc; α = 5%).

{{dep_table}}

**C.11 Fresh pre-registered runs.** After the main analysis and the auction study we committed a plan, timestamped before any of these data, with further runs, each with hypotheses, sample sizes and numerical predictions written in advance; the co-departure test of C.10 was pre-registered there as a secondary statistic, with the prediction that its false alarms stay at most 10% and its power is at least the joint bet's. {{fr_codep_summary}} Every run is reported whatever its outcome.
- *Run A, likelihood-ratio bet and pinned provider.* {{fr_a_sentence}}
- *Run C, collusion through a public chat.* {{fr_c_sentence}}
- *Run B, a pair told how the audit works.* {{fr_b_sentence}}
- *Run D, the vendor result on a second model family.* {{fr_d_sentence}}
- *Run E, second cartel series.* {{fr_e_sentence}}

## Appendix D. Reproducibility and Deviations
**Replays and seeds.** Every reported agent-round has exactly two replays (the analysis would use the smallest count available in a game). Each condition uses its own block of episode seeds, and the main samples, pilots and the replication use disjoint blocks. Each model call's sampling seed is derived from the call type, the episode seed, the agent and the round, so a logged move and its replays never share a seed; the null resampling of each episode is seeded by its episode seed. Pilot games were excluded from every analysis.

**Deviations from the plan.**
- Connection errors in the first launch were handled by re-running the same seeds.
- A post-hoc statistic comparing each agent's logged move with its own replays did not improve detection and is not used.
- The stronger judge, unnamed in the plan, was run on 20 games per condition for budget reasons, with replies of up to 2,000 tokens.
- A review found that the sequential test named the pair by its final evidence instead of at the alarm round; we fixed this and recomputed all sequential results.
- The judge is not deterministic at temperature 0. On games without private messages its two views send the same prompt, and an early uncached pass drew different replies for them (AUROCs of 0.84 for Haiku and 0.47 for DeepSeek in that pass); the reported numbers use the single cached reply, which reproduces exactly.
- Table A4, the AUROC intervals and Fisher's test were added after the reviews, from the frozen statistics.
- Models that failed the pilot's format or availability checks were dropped before the main samples.

## Appendix E. LLM Usage Statement
We used Claude (Anthropic) as a coding and analysis assistant: it implemented the experiment code under our direction, ran simulations, summarised related work and talk transcripts, drafted sections of this report and generated the figures from the result files. Codex (OpenAI) and separate Claude instances served as adversarial reviewers of the plan and of the report. Every number in the report is produced by scripts from the result files. We checked claims and references against their sources and take responsibility for the content.

**Author contributions.** Y. R. designed the study, ran the experiments and wrote the report with AI assistance (see the LLM Usage Statement).
