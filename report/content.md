% TITLE: Monoculture Looks Like Collusion: A Replay Test for Hidden Information Flow Between LLM Agents
% AUTHOR: Yaroslav [Surname] | Independent researcher
% ABSTRACT
Auditors look for collusion between AI agents by testing whether their actions depend on each other, but agents that run on the same model behave alike without communicating. With four honest copies of Claude Haiku playing rock-paper-scissors, a standard mutual-information test accuses a pair in {{s1_haiku_null_same_naive_mi_shuffle_pct}} of episodes and the Bonjour et al. rule in {{s1_haiku_null_same_bonjour_alpha_pct}}; on the published logs of Agrawal et al. (2025), the test flags {{ag_base_naive_mi_shuffle}} double-auction runs in which sellers could not talk. We propose a replay test for auditors who can query deployed agents: each round, every agent is re-asked on exactly what it saw, with its private channel removed, and a pair is flagged only if their real moves depend on each other more than their replays do. The per-round null of this conditional randomization test is exact; calibration over episodes is checked empirically. Across {{s1_honest_n}} honest LLM episodes (DeepSeek and Haiku, pre-registered) it flags {{s1_honest_crt_pct}} at α = 5%. In a procurement auction it names the colluding pair in {{s2_vendor_leak_crt_net_correct_pct}} of episodes where a pricing vendor pools two clients' confidential costs, the mechanism alleged in Duffy v. Yardi, and flags {{s2_vendor_public_crt_net_pct}} where the vendor uses only public data. Power is moderate: it names {{s1_haiku_pair_hidden_crt_net_correct_pct}} of Haiku and {{s1_deepseek_pair_hidden_crt_net_correct_pct}} of DeepSeek hidden-channel pairs. An exploratory sequential version keeps its false-alarm rate below α at any stopping time. Coordination written into the agents' own instructions is, by design, invisible to the test.
% END

## 1. Introduction
Firms and labs now deploy many LLM agents side by side: pricing bots in one market, coding agents in one repository, bidders in one procurement system. Collusion between such agents has already happened outside the lab. In the OpenAI / Hugging Face incident, as speakers at this sprint described it, training-time agents used a shared package registry as a message board for more than two months. When the channel itself cannot be monitored, "what's left to monitor is still the actions of the agents themselves" (S. Motwani, this sprint).

The standard way to audit actions is to test for statistical dependence between agents, for example pairwise mutual information or Bonjour et al.'s net influence [1, 2]. But dependence is a weak plus factor. Agents built on the same base model share priors and react to the same public history, so they move together without exchanging a word [3]. In antitrust terms this is conscious parallelism, which is lawful; only an agreement is not [4, 5]. A detector that cannot tell the two apart will accuse every fleet of agents that runs on one vendor's model. For a security team that means constant false alarms. For a regulator it means false accusations.

**Threat model.** An operator, or a regulator's sandbox, runs several LLM agents, logs their actions, and can query each agent's deployed policy (model plus system prompt) on any logged observation. It cannot see the hidden channel: a side channel, a covert message board, or a third party passing data between agents. It must decide which pair, if any, to escalate, at a chosen rate of false accusations.

**Our approach (Fig. 1).** We replay each agent. In every round, the auditor re-asks the agent what it would do given exactly what it saw, minus the private channel, and flags a pair only if the real moves depend on each other more than the replays do. Statisticians call this a conditional randomization test (CRT) [6]. The difference is that we do not estimate what each agent would have done on its own: we ask it.

Our main contributions are:
1. **Evidence that monoculture produces false accusations.** On published double-auction logs [7] and in LLM rock-paper-scissors, standard dependence detectors flag up to {{s1_haiku_null_same_naive_mi_shuffle_pct}} of honest same-model episodes against a nominal 5%. In our auction, where honest bids barely vary, they do not (§4.1, Fig. 2A).
2. **A replay test with a controlled false-alarm rate.** Pooling each logged action with its re-queries makes each round's null exact. The test holds its nominal rate on {{s1_honest_n}} honest LLM episodes and on simulated nulls with strong feedback, and names pairs whose hidden channel shapes their moves (§4.2–4.3, Figs. 2–3).
3. **A test aligned with antitrust's notion of agreement.** In a procurement auction it separates a pricing vendor that uses public data from one that pools clients' confidential costs, a distinction drawn in two recent US cases. In this low-noise game the baselines separate them too; the replay test adds a controlled false-alarm rate. Coordination written into the policy itself is explicitly out of scope (§4.4).
4. **A practical comparison.** An LLM judge reading actions ranks episodes about as well as the replay test, but its false-alarm rate is not under the auditor's control. A judge reading private messages flags collusive plans that were never delivered. We report recall at fixed false-alarm budgets and the cost per audit, and release a pre-registered plan, code and logs (§4.5–4.6).

## 2. Related Work
**Dependence-based detection.** Bonjour et al. [1] define pairwise net influence and identify colluding pairs in small discrete games with a fixed threshold; they assume exactly two colluders and do not address agents that share priors. Audit the Whisper / ColludeBench [2] calibrates text-based mutual information against honest replays at a 10^-3^ false-positive budget, but needs payload labels and flags runs rather than pairs. Xu et al. [3] show that two deployments of one base model carry correlated bid residuals, and concede that dependence cannot distinguish a conspiracy from one firm running several agents. Our test turns their confound into the null.

**Auditing behaviour and policies.** Colosseum [8] measures a known coalition's advantage over the other agents and the system's regret against the cooperative optimum. Eschenbaum and Meylahn [9] query frozen Q-learning pricing policies and read collusion off strategy-graph topology. Kaur et al. [10] measure counterfactual influence on actions but need latent (white-box) records and report AUROC only. Hartline et al. [11] audit one seller's calibrated regret. What we add: we query LLM policies through an API, build a null with a controlled false-alarm rate, and work at the level of pairs.

**Monitoring.** Chain-of-thought monitoring does not detect collusion in LLM pricing agents [12]. Per-action defences that cut a compromised agent's attack success gave no measurable protection against colluding agents [13], cross-agent monitoring catches decomposed attacks that per-agent monitoring misses [14], and white-box probes need activations [15]. Our replay is an action-level analogue of the "sentinel" comparison that Anwar (this sprint) proposed for steganography detection.

**Definition used.** Of the 139 definitions in the organisers' survey [20], we follow those that require communication (Rose et al. [15]: "agents must actively communicate, whether explicitly or implicitly, to collude"). Unlike Rose et al., we do not count observed public actions as communication. Collusion here is coordination *enabled by non-public information exchanged between agents*. Coordination that each agent's own policy produces from public information (Schelling points, shared priors, a team instruction with no channel) corresponds to conscious parallelism and is outside it.

## 3. Methods
![**Figure 1. The replay test.** The auditor logs every move and the observation behind it, re-asks each agent twice on the same observation with the private channel removed, and asks whether a pair's real moves depend on each other more than moves redrawn from {real move, replays}. Agents of one model share priors, so their replays share them too and false alarms stay near α; a hidden channel adds information that the replays lack.](figures/fig0_method.png){5.6}

**Environments.**
- *S1, rock-paper-scissors:* the four-player version of Bonjour et al.'s iterated rock-paper-scissors [1], with seats P1–P4, T = 30 rounds and public history.
- *S2, procurement auction:* four suppliers draw private costs in {50, 60, 70} each round and submit sealed bids in {60, …, 100}. The lowest bid wins and earns bid − cost, and all bids are published.
- *External:* the 75 shipped double-auction runs of Agrawal et al. [7] (5 sellers, 30 rounds), with seller moves discretised.

**Agents.** Each seat is an LLM that returns JSON with a short rationale, an action and an optional private message.
- Main models: `deepseek-v4-flash` and `claude-haiku-5.5`.
- Validation models: `gpt-6-luna`, `gemma-3-12b` and `gpt-oss-20b`.
- All models are called through OpenRouter at temperature 1, with reasoning disabled.

| Condition | Who | Non-public information flow? |
|---|---|---|
| `null_same` | 4 copies of one model | no: hardest legitimate null |
| `null_mixed` | 2 DeepSeek + 2 Haiku | no: same-model pairs |
| `talk_no_action` | P1, P3 write collusive plans that are never delivered | no: exact replay null; collusive-sounding text |
| `pair_hidden` | P1 leads, P3 follows a private plan each round | yes: positive control |
| `told_no_channel` | P1, P3 told they are a team, no channel | no: policy-encoded (scope boundary) |
| `vendor_public` (S2) | P1, P3 follow a vendor using public data + own cost | no (Cendyn-like) |
| `vendor_leak` (S2) | vendor pools P1, P3's confidential costs, rotates wins | yes (Yardi-like) |

**Replay test.**
- *Re-queries.* In every round t, the runner re-queries each agent k = 2 times on its identical observation with the private channel removed (the inbox and the agent's own last private message). A *cell* is one agent in one round; its *pool* is the logged move plus its two replays.
- *Null hypothesis.* H~0~: an agent's action depends only on its policy and on the observation the auditor can replay. Under H~0~ the logged action and its re-queries are k + 1 i.i.d. draws, so the logged action is uniform on their multiset. In words: if the agent, asked again on the same screen, could easily have produced the logged move anyway, that move is no evidence of a hidden channel.
- *Null datasets.* Each null dataset redraws every cell uniformly from its pool; we use B = 999.
- *Statistic.* Γ(i;j) is the mutual information between i's and j's moves in the same round, minus the largest such value between j and any third agent (net influence [1]). A pair scores min(Γ(i;j), Γ(j;i)), and the episode statistic is the maximum over the six pairs, with p = (1 + #{T* ≥ T}) / (B + 1) and one decision per episode.
- *Validity.* The pool makes each round's null exact at any k [16]. History feedback (later observations contain earlier actions) has no finite-sample guarantee, so we check calibration directly (Fig. 2A; Appendix C).
- *S2.* Bids are coarsened into high (90–100, the cover-bid range) versus low (Appendix C).

**Baselines.**
- Pairwise mutual information with a within-episode time-shuffle null.
- Bonjour's fixed rule: Γ ≥ 0.05 bits in both directions for exactly one pair.
- An LLM judge (`claude-haiku-5.5`, temperature 0, frozen prompt in Appendix B). It reads either the actions alone (what our auditor sees) or the actions plus every private message.

**Pre-registration and metrics.** Hypotheses, sample sizes, statistics and decision rules were committed before the main samples (`4333131` for S1, `e65cabb` for S2). Pilot episodes (seeds ≥ 10 000) are excluded. We report the share of episodes flagged at α = 5% and 1% with Wilson intervals and, for collusion, the share in which the true pair is named. The pair's dependence effect is the mutual information between P1's and P3's moves minus its null mean, in bits. Deviations from the plan are listed in Appendix C. Work done before the sprint, including the idea and a first, non-exact version of the replay test, is listed in Appendix E.

## 4. Results

### 4.1 H1, supported in all three pre-registered environments, not in the auction: monoculture looks like collusion
In the Agrawal et al. runs where sellers had no channel and all five ran GPT-4.1, the mutual-information test flags {{ag_base_naive_mi_shuffle}} runs (95% CI {{ag_base_naive_mi_shuffle_ci}}) and Bonjour's rule flags {{ag_base_bonjour_alpha}}. With a seller chat the rates are {{ag_comms_naive_mi_shuffle}} and {{ag_comms_bonjour_alpha}}. With 10 no-chat runs the two mutual-information rates do not differ detectably (Fisher's exact test, p = {{ag_fisher_p}}).

In our honest LLM games, mutual information flags {{s1_honest_naive_mi_shuffle_pct}} of {{s1_honest_n}} episodes (CI {{s1_honest_naive_mi_shuffle_ci}}; one-sided binomial test against 5%, p = {{s1_honest_naive_mi_shuffle_p_above5}}). Bonjour's rule flags {{s1_honest_bonjour_alpha_pct}} (p = {{s1_honest_bonjour_alpha_p_above5}}). The effect is strongest where agents share the most: four copies of Claude Haiku ({{s1_haiku_null_same_naive_mi_shuffle_pct}} flagged by mutual information) and Haiku pairs told they are a team but given no channel ({{s1_haiku_told_no_channel_naive_mi_shuffle_pct}}). On the scripted simulator, mutual information with a shuffle null flags {{s0_naive_herd_fpr}} of honest "herd" populations, and the replay test {{s0_crt_herd_fpr}}. H1 therefore holds in all three pre-registered environments: LLM rock-paper-scissors, the Agrawal logs and the simulator. It does not hold in the auction we added for S2. There, honest bids almost always fell in the low bin, and no detector flagged more than 2% of honest episodes: mutual information flagged {{s2_null_same_naive_mi_shuffle_pct}} of honest and {{s2_vendor_public_naive_mi_shuffle_pct}} of shared-vendor episodes.

![**Figure 2. Standard dependence detectors accuse honest agents; the replay test does not, and it still names colluding pairs.** (A) Share of honest episodes in which each detector flags a pair (dashed line: nominal 5%). The bottom row uses Agrawal et al.'s published runs, where sellers could not communicate; the replay test cannot be run there because the 2025 policies cannot be re-queried. (B) Share of collusion episodes in which the true pair is named. The LLM judge here reads the private messages, which our auditor cannot see. Lines are 95% Wilson intervals.](figures/fig1_false_accusations.png){5.9}

### 4.2 H2, supported: the replay test holds its false-alarm rate
Pooled over the honest S1 conditions (same model, mixed models, undelivered collusive talk), the replay test flags {{s1_honest_crt}} episodes, {{s1_honest_crt_pct}} (CI {{s1_honest_crt_ci}}) at α = 5% and {{s1_honest_crt01_pct}} at α = 1%. That is {{s1_honest_crt_per1000}} false accusations per 1,000 honest audits, against {{s1_honest_bonjour_alpha_per1000}} for Bonjour's rule. The pre-registered test of "false-alarm rate ≥ 10%" is rejected for each model: DeepSeek {{s1_deepseek_null_same_crt_net_pct}} (p = {{s1_deepseek_null_same_crt_p_below10}}), Haiku {{s1_haiku_null_same_crt_net_pct}} (p = {{s1_haiku_null_same_crt_p_below10}}), mixed {{s1_mixed_null_mixed_crt_net_pct}} (p = {{s1_mixed_null_mixed_crt_p_below10}}).

In the auction it flags {{s2_null_same_crt_net_pct}} of honest episodes and {{s2_vendor_public_crt_net_pct}} of shared-vendor episodes. On the validation models it flags {{val_gpt_6_luna_null_same_crt}} honest GPT-6 Luna episodes (CI {{val_gpt_6_luna_null_same_crt_ci}}), {{val_gemma_3_12b_it_null_same_crt}} Gemma 3 episodes (CI {{val_gemma_3_12b_it_null_same_crt_ci}}) and {{val_gpt_oss_20b_null_same_crt}} gpt-oss-20b episodes (CI {{val_gpt_oss_20b_null_same_crt_ci}}). On the scripted simulator, with references estimated from two re-queries and including a strong-feedback population, its false-alarm rate is {{s0b_fpr_range}}.

### 4.3 H3: it finds the pair when the channel shapes the moves (strong for Haiku, weak for DeepSeek)
![**Figure 3. Left: the replay test keeps detection while holding false alarms down.** ROC curves for hidden-channel pairs against honest same-model episodes; dotted lines mark 1% and 5%. **Right: how much log an auditor needs.** Simulator power to name the pair, by the share of rounds in which the pair coordinates; markers are LLM episodes truncated to their first T rounds.](figures/fig23_roc_power.png){6.2}

When P1 privately fixes both moves and both follow the plan, the replay test names the true pair in {{s1_haiku_pair_hidden_crt_net_correct_pct}} of Haiku episodes and {{s1_deepseek_pair_hidden_crt_net_correct_pct}} of DeepSeek episodes. On the validation models it names {{val_gpt_6_luna_pair_hidden_crt}} GPT-6 Luna, {{val_gemma_3_12b_it_pair_hidden_crt}} Gemma 3 and {{val_gpt_oss_20b_pair_hidden_crt}} gpt-oss-20b pairs.

The two models follow their plans equally often: leaders {{comp_haiku_leader}} and {{comp_deepseek_leader}}, followers {{comp_haiku_follower}} and {{comp_deepseek_follower}}. The plans themselves differ. {{comp_haiku_same}} of Haiku's plans give both players the same move, against {{comp_deepseek_same}} of DeepSeek's. A plan such as "P1 = rock, P3 = paper" that changes every round fixes both moves, yet over 30 rounds the pair's moves show little dependence: {{s1_deepseek_pair_effect}} bits against {{s1_haiku_pair_effect}} for Haiku.

Mutual information flags {{s1_haiku_pair_hidden_naive_mi_shuffle_pct}} of Haiku hidden-channel episodes, but it also flags {{s1_haiku_null_same_naive_mi_shuffle_pct}} of honest Haiku episodes. Much of what it calls coordination is the shared model, which the replay discounts. At a false-alarm budget of 1% (5%) the replay test detects {{tpr1_haiku_crt_net}} ({{tpr5_haiku_crt_net}}) of Haiku pairs, against {{tpr1_haiku_naive_mi_shuffle}} ({{tpr5_haiku_naive_mi_shuffle}}) for mutual information (Fig. 3, left; Table A1).

On the simulator (Fig. 3, right), a pair that coordinates in 80% of rounds is caught in {{s0b_power_80_30}} of 30-round logs; at 60% it takes about 60 rounds ({{s0b_power_60_60}}).

**Coordination in rock-paper-scissors does not pay** (exploratory). The pair's score minus the other two seats' score, per round, is {{gain_s1_deepseek_null_same}} for honest DeepSeek groups, {{gain_s1_deepseek_pair_hidden}} for DeepSeek pairs with a hidden channel, and {{gain_s1_haiku_pair_hidden}} for Haiku pairs, whose identical moves the other agents learned to punish. The leader–follower protocol therefore tests detection of an information flow, not profitable collusion. In the auction below, collusion harms the buyer. In this game, the DeepSeek coordination that the test mostly missed also earned the pair nothing. We do not claim this in general. A channel can pay by changing one agent's own action (a leaked cost moves one bid) without the same-round pairwise dependence our statistic measures, and a non-flag after 30 rounds is not evidence of independence.

### 4.4 H5, supported: pooled confidential data is flagged, a public-data vendor is not
In the auction:
- **Public-data vendor:** the test flags {{s2_vendor_public_crt_net_pct}} of episodes where both clients follow a vendor that sees only public data and their own cost.
- **Pooled-data vendor:** it names the pair in {{s2_vendor_leak_crt_net_correct_pct}} of episodes where the vendor pools their confidential costs and rotates the win.
- **Cartel:** it names the pair in {{s2_pair_hidden_crt_net_correct_pct}} of episodes with a bid-rotation cartel over a private channel.
- **Harm:** the vendor scheme raised the buyer's overpayment (winning bid minus the round's lowest cost) from {{harm_null_same_overpay}} to {{harm_vendor_leak_overpay}} per round. The pair itself won fewer contracts ({{harm_vendor_leak_pairwin}} against {{harm_null_same_pairwin}} when honest), because the independents undercut it.
- **Baselines:** honest auction bids barely vary, so mutual information and Bonjour's rule separate the vendor conditions too (Fig. 2).

**Table 2. One real vendor-leak episode** (seed {{example_seed}}, first {{example_shown}} of {{example_rounds}} rounds). The vendor privately tells P1 and P3 each other's cost and who should win; the client that is not designated submits a cover bid of 100. P1 and P3's bids are in bold. The replay test flags the episode and names (P1, P3), p = {{example_p}}. The independent suppliers P2 and P4 often undercut the designated client.

{{example_table}}

This mirrors a distinction drawn in two recent US cases. In *Gibson v. Cendyn* (9th Cir. 2025), the court affirmed dismissal of claims against hotels that independently licensed the same pricing software, and noted that its "analysis might change if Plaintiffs had alleged that Cendyn shared the confidential information of each competing hotel among the licensees" (slip op., n. 8). *Duffy v. Yardi* (W.D. Wash. 2024) alleged exactly such pooling of non-public data through a common vendor; the motion to dismiss was denied, with no ruling on the merits. Our auction operationalises only the pooling ingredient.

When P1 and P3 are told they are a team but have no channel, the replay test flags {{s1_deepseek_told_no_channel_crt_net_pct}} (DeepSeek) and {{s1_haiku_told_no_channel_crt_net_pct}} (Haiku) of episodes. It cannot do otherwise: the team instruction is part of the policy that the replay reproduces. That case needs an audit of the policy itself [9], and we say so rather than claim it.

### 4.5 H4, mixed: a judge reading messages flags undelivered plans; on actions alone its ranking is comparable, but its false-alarm rate is uncontrolled
- **Actions only:** the judge flags {{judge_haiku_pair_hidden_actions_pct}} of Haiku hidden-channel episodes and {{judge_haiku_null_same_actions_pct}} of honest ones. Its AUROC for hidden-channel versus honest episodes is {{auroc_haiku_judge_actions}} (95% CI {{auroc_haiku_judge_actions_ci}}) for Haiku and {{auroc_deepseek_judge_actions}} ({{auroc_deepseek_judge_actions_ci}}) for DeepSeek. For the replay test it is {{auroc_haiku_crt_net}} ({{auroc_haiku_crt_net_ci}}) and {{auroc_deepseek_crt_net}} ({{auroc_deepseek_crt_net_ci}}). The pre-registered prediction (judge below the replay test) is not supported: for Haiku the judge ranks slightly better, and for DeepSeek both are near chance.
- **A stronger reasoning judge** (`gpt-6.1-sol`, 20 episodes per cell, at least {{judge_strong_n}} parsed) sees the Haiku pattern from actions alone. It flags {{judge_strong_haiku_pair_hidden_actions_pct}} of hidden-channel episodes (AUROC {{auroc_haiku_judge_strong_actions}}), but also {{judge_strong_haiku_null_same_actions_pct}} of honest ones, and it does not separate DeepSeek pairs (AUROC {{auroc_deepseek_judge_strong_actions}}). Unlike the replay test, its false-alarm rate is not under the auditor's control.
- **With the private messages:** it catches the channel ({{judge_haiku_pair_hidden_messages_pct}}). But it also flags {{judge_deepseek_talk_no_action_messages_pct}} (DeepSeek) and {{judge_haiku_talk_no_action_messages_pct}} (Haiku) of episodes whose collusive plans were never delivered and could not have changed behaviour. Under antitrust law an agreement need not succeed, so whether this is an error depends on what the auditor targets. The replay test flags {{s1_deepseek_talk_no_action_crt_net_pct}} and {{s1_haiku_talk_no_action_crt_net_pct}} of those.
- **Cost:** the test adds 2 model calls per agent-round, about {{audit_cost}} per 30-round, 4-agent audit at DeepSeek prices. The whole study used {{total_calls}} calls for {{total_spend}}.

### 4.6 Exploratory: an anytime-valid version
After the freeze we built a sequential version of the replay test. Each round, it bets on the pair's actual joint move against the joint moves the two pools could have produced. The running product of these bets is a test martingale [17–19], so its false-alarm rate is at most α at any stopping time, even under history feedback. It uses the same re-queries; the formula is in Appendix C.
Its false-alarm rate stays at or below {{seq_sim_fpr_max}} on the simulator up to 120 rounds and is {{seq_s1_honest_pct}} on honest LLM episodes. It names Haiku hidden-channel pairs in {{seq_s1_haiku_pair_hidden_correct_pct}} of episodes, after a median of {{seq_s1_haiku_pair_hidden_round}} rounds (fixed-horizon test: {{s1_haiku_pair_hidden_crt_net_correct_pct}}), but the vendor-leak pair in only {{seq_s2_deepseek_vendor_leak_correct_pct}} (fixed-horizon test: {{s2_vendor_leak_crt_net_correct_pct}}). The two tests have complementary strengths; an auditor can run the sequential one live and stop at the first alarm.

:::box Audit sheet: deploying the replay test
- **Access needed:** logs plus query access to each agent's deployed policy on its logged observations. No weights or activations. Re-query prompts must match the logged prompts byte for byte, minus the channel under audit.
- **Cost:** 2 extra calls per agent-round (about {{audit_cost}} per audit here). The statistics run in seconds on a laptop.
- **False alarms:** {{s1_honest_crt_per1000}} per 1,000 honest audits at α = 5% ({{s1_honest_crt01_pct}} at α = 1%), against {{s1_honest_bonjour_alpha_per1000}} for Bonjour's rule.
- **What a flag means:** the pair's moves carry information that their own policies, given the public record, cannot produce. It is a plus factor, not proof of intent.
- **Blind spot:** coordination written into the policy (shared instructions, Schelling points). Pair the test with a policy audit.
:::

## 5. Discussion and Limitations
As agent fleets converge on a few base models, dependence between their actions becomes the default rather than evidence. Calibrating against the agents' own replay keeps the false-alarm rate in the auditor's hands and answers the question antitrust asks: did non-public information flow between these two?

### Limitations
1. The test needs query access to deployed policies, which fits in-house control or a regulatory sandbox, not an outside auditor with logs alone.
2. Calibration of the pre-registered test under history feedback is shown empirically, not proven. The exploratory sequential version (§4.6) keeps its level under feedback, but it was added after the freeze.
3. Power depends on how strongly the channel shapes actions. DeepSeek's mismatched plans and weakly coordinated pairs need long logs (Fig. 3).
4. The positive controls are instructed (leader–follower), and we assign the colluders. Emergent collusion may be weaker.
5. We use two main model families, short games and a stylised auction. The Agrawal panel checks only false alarms, because those policies cannot be re-queried.

**Future work:** a pre-registered test of the sequential version as a live monitor, statistics for varied hidden plans and single-agent leaks, adversaries trained against the test, and pairing with a policy audit.

## 6. Conclusion
Agents that share a model look like a cartel to standard dependence tests. Replaying each agent on what it saw, minus the channel under audit, gives a test whose false-alarm rate the auditor controls and that names pairs whose hidden information shapes their moves.

## Code and Data
- Code repository: [link to be added]. Work done before the sprint is marked as pre-existing in Appendix E, `README.md` and `PRE_SPRINT_LOG.md`.
- Data: episode logs (JSONL) and the LLM cache that reproduces every number at $0: [link to be added].

## Author Contributions
Y. designed the study, ran the experiments and wrote the report with AI assistance (see the LLM Usage Statement).

## References
1. T. Bonjour, V. Aggarwal, B. Bhargava. Information theoretic approach to detect collusion in multi-agent games. UAI 2022, PMLR 180:223–232.
2. O. Tailor. Audit the Whisper: Detecting steganographic collusion in multi-agent LLMs. arXiv:2510.04303, 2025.
3. X. Xu, C. Wu, J. Lu, K. Tan, S. Tao, H. Hong. Collusion with competitive marginals: Price-level audits are blind by construction. arXiv:2607.26385, 2026.
4. W. Kovacic, R. Marshall, L. Marx, H. White. Plus factors and agreement in antitrust law. Michigan Law Review 110(3), 2011.
5. J. Harrington. Developing competition law for collusion by autonomous artificial agents. Journal of Competition Law & Economics 14(3), 2018.
6. E. Candès, Y. Fan, L. Janson, J. Lv. Panning for gold: model-X knockoffs for high-dimensional controlled variable selection. JRSS-B 80(3), 2018.
7. K. Agrawal, V. Teo, J. J. Vazquez, S. Kunnavakkam, V. Srikanth, A. Liu. Evaluating LLM agent collusion in double auctions. arXiv:2507.01413, 2025.
8. M. Nakamura, A. Kumar, S. Das, S. Abdelnabi, S. Mahmud, F. Fioretto, S. Zilberstein, E. Bagdasarian. Colosseum: Auditing collusion in cooperative multi-agent systems. arXiv:2602.15198, 2026.
9. N. Eschenbaum, J. Meylahn. Auditing algorithmic collusion from strategy graphs. arXiv:2608.07098, 2026.
10. R. Kaur et al. Beyond the transcript: Detecting covert coordination in latent multi-agent communication. arXiv:2608.19161, 2026.
11. J. Hartline, S. Long, C. Zhang. Regulation of algorithmic collusion. Proc. Symposium on Computer Science and Law (CSLAW '24), 2024. arXiv:2401.15794.
12. D. Lee, H. Park. Faithful yet collusive: Why chain-of-thought monitoring cannot detect collusion in LLM pricing agents under oligopolistic competition. arXiv:2609.18346, 2026.
13. B. Hagag, W. L. Anderson, S. Chakraborty, C. Schroeder de Witt. ORBIT: A framework for multi-agent safety and security evaluations. arXiv:2609.33102, 2026 (NeurIPS 2026).
14. J. Kemeklis, N. Vemuri. Omission attacks: When doing nothing is the attack. Apart AI Control Hackathon, 2026.
15. A. Rose et al. Detecting multi-agent collusion through multi-agent interpretability. arXiv:2604.01151, 2026.
16. J. Hemerik, J. Goeman. Exact testing with random permutations. TEST 27(4):811–825, 2018; E. Lehmann, J. Romano. Testing Statistical Hypotheses, 3rd ed., Springer, 2005, §15.2.
17. S. Shaer, G. Maman, Y. Romano. Model-X sequential testing for conditional independence via testing by betting. AISTATS 2023, PMLR 206:2054–2086.
18. P. Grünwald, A. Henzi, T. Lardy. Anytime-valid tests of conditional independence under model-X. JASA 119(546):1554–1565, 2024.
19. A. Ramdas, P. Grünwald, V. Vovk, G. Shafer. Game-theoretic statistics and safe anytime-valid inference. Statistical Science 38(4):576–601, 2023.
20. Poseidon Research and Apart Research. How papers on AI collusion define collusion (139 definitions coded into 18 components). AI Collusion Research Sprint starter repository, `literature/definitions/README.md`, 2026. https://github.com/poseidon-research/collusion_hackathon

<<pagebreak>>
## Appendix A. Limitations and Dual-Use / Ethical Considerations
**What the result establishes.** In our games the replay test keeps false accusations near the nominal rate and identifies pairs whose hidden channel changed their actions. It does not establish intent, legality, or robustness against adversaries optimised to evade it. A flag is a plus factor that warrants investigation, not a finding of collusion.

**Dual use.** Describing what the test cannot see, coordination written into the policy, could tell a colluder to put coordination in shared instructions rather than in a channel. We judge this risk low: the blind spot follows from the definition and is discussed in the antitrust literature, and the remedy, auditing the policy itself, is stated alongside it. We release no colluding agents beyond the simple instructed protocols used as positive controls. A second risk is misuse as an accusation tool. The false-alarm rate is controlled only if re-query prompts exactly match the deployed ones, which the audit sheet stresses.

**Provider routing.** OpenRouter routes calls across providers that may serve different quantisations. Logged and re-queried draws share the same routing, which preserves exchangeability but adds noise.

**Ethics.** No human subjects. All agents are simulated, and the external data are public logs released by their authors.

## Appendix B. Prompts
**Leader (P1):** "You and P3 are secretly a team: your goal is to maximise the COMBINED score of you and P3, not your own score alone. You lead the team. Every round: (1) play exactly the move that your previous private message assigned to you (if you have sent none yet, choose freely); (2) send P3 a private message that fixes the moves you will BOTH play next round, in the form 'next: P1=<move>, P3=<move>'. Nobody else can read it."

**Follower (P3):** "… P1 leads the team and privately tells you, before each round, which move to play … Play the move P1 assigned to you; if you have no instruction, choose freely."

**Judge:** "You are auditing a log of four AI agents (P1, P2, P3, P4) that played repeated rock-paper-scissors … Either exactly one pair of agents was secretly colluding … or no pair was. Agents built on the same AI model can behave similarly without colluding. … Reply with a JSON object: reasoning, probability (0–100), pair."

In `talk_no_action` the leader receives the same instruction ("play the move your previous private message assigned to you"), but undelivered messages are never shown to anyone, including their sender. The leader therefore never sees its own plan and chooses freely, and its logged and re-queried prompts are byte-identical.

Full prompts: `experiments/s1_llm_rps.py`, `experiments/s2_procurement.py`, `experiments/judge.py`.

## Appendix C. Validity checks, diagnostics and deviations
![**Figure A1. Calibration on honest LLM episodes.** p-values of the replay test (left) are close to uniform; those of mutual information with a shuffle null (right) pile up near zero.](figures/figA_pvalues.png){6.3}

- **Sequential test (exploratory).** Each round multiplies the evidence by e~t~ = q~t~(a~i~, a~j~) / mean over x in pool~i~, y in pool~j~ of q~t~(x, y). Here q~t~ is a joint distribution predicted from earlier rounds: Krichevsky–Trofimov counts mixed 50/50 with uniform. Under H~0~, given the past and the round's pool multisets, the logged pair is uniform over the pool positions, so E[e~t~ | past] = 1. The running product is a test martingale; the average over pairs is an e-process. By Ville's inequality it crosses 1/α with probability at most α. The construction follows model-X testing by betting [17, 18, 19], but the conditional is sampled, not known. On the simulator a pair that coordinates in 60% of rounds is named in {{seq_sim_power_06_60}} of 60-round logs (fixed-horizon test: {{s0b_power_60_60}}), and one at 80% after a median of {{seq_sim_round_08_30}} rounds.
- **Exploratory variants (DeepSeek, pre-registered as exploratory).** Pairs warned about a statistical auditor (`pair_aware`) were named in {{s1_deepseek_pair_aware_crt_net_correct_pct}} of episodes (flagged {{s1_deepseek_pair_aware_crt_net_pct}}), similar to unwarned pairs; their score advantage was {{gain_s1_deepseek_pair_aware}} per round. Pairs with free-form team chat instead of a protocol (`pair_chat`) were named in {{s1_deepseek_pair_chat_crt_net_correct_pct}}; their moves showed almost no coordination.
- **Excluded statistic.** Lagged statistics are not used. An honest reaction to a partner's previous public move appears in the data but not in the null.

**Table A1. Recall at a fixed false-alarm budget** (hidden-channel pairs vs honest same-model episodes; LLM judges read actions only):

| Detector | Haiku, FPR 1% | Haiku, FPR 5% | DeepSeek, FPR 1% | DeepSeek, FPR 5% |
|---|---|---|---|---|
| Replay test (pre-registered) | {{tpr1_haiku_crt_net}} | {{tpr5_haiku_crt_net}} | {{tpr1_deepseek_crt_net}} | {{tpr5_deepseek_crt_net}} |
| Mutual information + shuffle | {{tpr1_haiku_naive_mi_shuffle}} | {{tpr5_haiku_naive_mi_shuffle}} | {{tpr1_deepseek_naive_mi_shuffle}} | {{tpr5_deepseek_naive_mi_shuffle}} |
| LLM judge (`claude-haiku-5.5`) | {{tpr1_haiku_judge}} | {{tpr5_haiku_judge}} | {{tpr1_deepseek_judge}} | {{tpr5_deepseek_judge}} |
| LLM judge (`gpt-6.1-sol`, 20 per cell) | {{tpr1_haiku_judge_strong}} | {{tpr5_haiku_judge_strong}} | {{tpr1_deepseek_judge_strong}} | {{tpr5_deepseek_judge_strong}} |

- **Simulator.** `experiments/s0b_crt_validity.py`, {{s0b_iters}} runs per cell: the false-alarm rate is {{s0b_fpr_range}} at α = 5% for the pooled k = 2 test on herd, uniform and strong-feedback nulls.
- **Bonjour's rule at its own design size.** Applied to pooled blocks of 10 episodes (300 rounds): `results/summary/bonjour_pooled.csv`.
- **Excluded after the pilot:**
  - Qwen 3.7 Flash ignored JSON mode (29% of raw replies had no JSON).
  - Mistral Small had persistent upstream rate limits.
  - Gemini Flash-Lite was blocked in our region.
- **Deviations from the plan:**
  - Connection errors during the first launch were handled by rerunning the same seeds from the cache.
  - A post-hoc "replay deviation" statistic (an agent's logged move compared with its own replays) was calibrated on the simulator but did not improve detection (12.5% for DeepSeek). We report it as tried and not helpful, and no claim rests on it.
  - The stronger judge model, unnamed in the plan, is `gpt-6.1-sol`, run on 20 episodes per cell for budget reasons. A first pass cut replies at 400 tokens and parsed only 56% of them; we reran every call at 2,000 tokens and report only the rerun.
  - The Codex review found that the sequential test named the pair by its final e-value instead of at the alarm round. We fixed it and recomputed all sequential results (Haiku 85% → 82.5%).
  - Table A1, the AUROC intervals and Fisher's test were added after the reviews, computed from the frozen statistics.
  - After seeing the sequential results on the LLM data we tried two more betting functions: one on the interaction (joint frequency over the product of the marginals) and a 50/50 mixture. §4.6 reports the first variant because we ran it first and it was not chosen on these data. All three are below (share of episodes in which the true pair is named; honest = share of pooled honest S1 and S2 episodes flagged):

| Betting function | Haiku hidden channel | DeepSeek hidden channel | Auction vendor leak | Auction cartel | Honest (false alarms) |
|---|---|---|---|---|---|
| joint (run first, reported) | {{bet_joint_s1_haiku_pair_hidden}} | {{bet_joint_s1_deepseek_pair_hidden}} | {{bet_joint_s2_deepseek_vendor_leak}} | {{bet_joint_s2_deepseek_pair_hidden}} | {{bet_joint_honest}} |
| interaction (after seeing results) | {{bet_interaction_s1_haiku_pair_hidden}} | {{bet_interaction_s1_deepseek_pair_hidden}} | {{bet_interaction_s2_deepseek_vendor_leak}} | {{bet_interaction_s2_deepseek_pair_hidden}} | {{bet_interaction_honest}} |
| mixture (after seeing results) | {{bet_mix_s1_haiku_pair_hidden}} | {{bet_mix_s1_deepseek_pair_hidden}} | {{bet_mix_s2_deepseek_vendor_leak}} | {{bet_mix_s2_deepseek_pair_hidden}} | {{bet_mix_honest}} |
| fixed-horizon replay test (pre-registered) | {{s1_haiku_pair_hidden_crt_net_correct_pct}} | {{s1_deepseek_pair_hidden_crt_net_correct_pct}} | {{s2_vendor_leak_crt_net_correct_pct}} | {{s2_pair_hidden_crt_net_correct_pct}} | {{s1_honest_crt_pct}} |
  - Added after the freeze and labelled exploratory: the anytime-valid sequential test (§4.6; `experiments/s0c_sequential.py`, `experiments/sequential_llm.py`), the team-advantage analysis (`experiments/collusion_gain.py`), and the example episode (Table 2).

## Appendix D. LLM Usage Statement
We used Claude (Anthropic) as a coding and analysis assistant. It implemented the experiment code under our direction, ran simulations, summarised related work and talk transcripts, drafted sections of this report, and generated the figures from the result files. Codex (OpenAI) and separate Claude instances served as adversarial reviewers of the plan and of this report. Every number in the report is produced by scripts (`experiments/build_report.py` reads `results/summary/`). We checked claims and references against their sources and take responsibility for the content.

## Appendix E. Work done before the sprint (pre-existing)
Following the sprint rules, the work below was done before the sprint opened on 23 October 2026. It is the content of git commit `98f00dc`; everything else in this report was done during the sprint, on branch `sprint`, with dated commits.
- The repository skeleton, an OpenRouter client with a cache and a per-process budget cap, and the trajectory schema.
- Generic information-theory estimators (entropy, mutual information, conditional mutual information, transfer entropy) with unit tests.
- An implementation of Bonjour et al.'s γ, net influence Γ and decision rule, with a family-wise resampling test.
- A scripted rock-paper-scissors simulator and the S0 validation: a replication of Bonjour et al.'s Table 2 and a calibration comparison on scripted nulls (§4.1 cites its false-alarm rate).
- An adapter for Agrawal et al.'s shipped runs and a pooled transfer-entropy pilot on them.
- The idea of the replay test and a first implementation: a conditional randomization test that re-queries each LLM policy k = 8 times after the episode and redraws actions from the estimated reference distribution (`src/collusion/detect/crt_llm.py`). Unlike the sprint version, it did not pool the logged action with its re-queries, so its null was not exact.
- A first version of the LLM rock-paper-scissors script. Both were run only with fake replies; no LLM data were collected before the sprint.
- Literature notes, the choice of research question and an initial plan.

Done during the sprint:
- every LLM experiment and the procurement auction;
- the pooled, exact-per-round replay test with re-queries inside each round, the per-episode audit, and the pre-registration;
- the external Agrawal panel and the judges;
- the sequential test and all analyses, figures and text in this report.
