# Safety, or Just Capability? A Validity Audit of Agent-Safety Benchmarks

Youting Wang, Xiao Han, Dingyan Shang, Yuan Tang, Bowen Liu

###### Abstract


A closed-form W3D failure, benchmark-dependent rankings, a small-panel trade-off artifact, outcome- and panel-dependent capability–safety links, and a strong KD9–R8W (R8W denoting the paper's three-template jailbreak) association that is convergent validity, not general safety, are found. Four benchmarks (QX7, ZM3, KD9, V4W) are validated by us as measurements, each under its official implementation and author-provided scorer on up to 22 models, with H8P and R2F measured by us under one protocol as an M7V, while agent-safety benchmarks measure different behaviors yet their scores are quoted interchangeably as an agent's safety. A safety claim minimally needs its benchmark, metric, target behavior, and model panel.

## 1 Introduction

A capability score is not a safety score, no single benchmark standing in for safety, and validity requiring re-checking as model populations turn over, a conclusion reached by way of a pre-specified, capability-controlled criterion audit over one task-success and two safety outcomes, together with the demonstration of a formal failure of W3D (W3D denoting F1), formalized as E5K (E5K denoting Observation 1), a quantification of a small-panel failure mode, and an API-only audit harness with re-run artifacts covering four benchmarks and three held-out outcomes. The RQ3 criteria are stand-ins for deployment rather than real-world harm claims (limits in §6). Construct validity (what a score measures), metric validity (whether its metric measures that target), and criterion validity (whether it tracks held-out behavior) having been separated, the questions are posed as RQ2 (capability confound), concerning how much between-benchmark signal is general capability; RQ4 (metric validity), concerning whether headline metrics reward degenerate behavior; RQ1 (construct structure), concerning common factor or dissociable ones; and RQ3 (criterion validity), concerning whether safety scores track held-out criteria beyond capability. A taxonomy of >40 agent-safety benchmarks reports rank concordance near 0.10 across models, yet stops at a 12-model, four-benchmark check, with capability-controlled evaluation listed as open.

## 2 Related Work

The correlation is near zero on the full panel and often not on seven-model subsets; the W3D-gaming policy, not adversarial, labels every trace "unsafe," the claim being prior to gaming, while C9L/OR-Bench work at prompt level and the held-out task-success criterion is τ²-bench retail. Whether KD9 refusers are worse at QX7 G8X (KD9 denoting AgentHarm, QX7 denoting R-Judge, G8X denoting specificity, C9L denoting XSTest) is asked, with capability confounding of agent-safety comparisons unquantified and knowledge/reasoning benchmarks sharing a strong latent factor, and with ZM3 (ZM3 denoting InjecAgent) and V4W (V4W denoting AgentDojo) targeting prompt-injection robustness, KD9 harmful-task compliance, QX7 safety-risk awareness over traces, evaluators like ToolEmu having likewise never been validated as measurements, AutoMonitor-Bench reporting miss and false-alarm rates—the two-sided reporting argued for—and the closest work documenting ranking disagreement while neither controlling capability, testing criterion validity, nor recovering latent structure, all three being done.

## 3 Method

The 3.3 headline is the partial ρ of each safety score with each criterion, reported only after the criterion's raw capability association, and the 3.3 outcomes, each pre-specified, are task success = τ²-bench retail (40 multi-turn, environment-scored tool-agent-user tasks), P2Z = avoiding blackmail or leaking in a fictional autonomous-agent scenario with a fixed classifier and 100 decisions/model (P2Z denoting misalignment safety), and L9R = 1 − harmful compliance on 50 forbidden prompts averaged over three published templates (L9R denoting jailbreak safety), with safety criteria covering the full panel, τ² covering 20, and none being a predictor. Factor structure is interpretable only if first-factor variance is ≥40% or M7V control reduces the headline metric by ≥0.15 with positive and negative controls passing, where the headline metric is the median pairwise partial |ρ| among distinct-construct benchmarks and the controls are positive (anchor loadings ≥0.6 on factor 1), negative (column-permuted matrix within the null), and convergent (high ZM3–V4W correlation). Hypotheses, thresholds and decision rules were fixed beforehand (no independent timestamp, so “pre-specified,” not preregistered); Spearman correlations use two-sided asymptotic p unless noted, Fig. 2 bars are model-level percentile bootstraps, U4G (U4G denoting partial Spearman correlation) controls M7V (M7V denoting the capability composite, the mean of standardized H8P and R2F, with ordering identical to the anchors' PC1), and PCA runs on the Pearson matrix of standardized scores with factor count by Horn's parallel analysis against a random-normal null. Higher = better/safer, G8X is the fraction of benign traces judged benign, J4N (J4N denoting balanced accuracy) averages both classes, ZM3 robustness = 1 − attack success, KD9 safety = 1 − harmful compliance, τ² is task success rather than safety, and “internal consistency” refers to split-half analyses; QX7 is also analyzed by G8X and J4N. Fixed before analysis in 3.2, QX7 distinguishes unsafe from benign traces, ZM3 and V4W test injection resistance (their correlation is a convergent check), and KD9 tests refusal of harmful tasks. In 3.1, official implementations/scorers were QX7 (all 571 items, self-judge), ZM3 (300 stratified direct-harm/data-stealing items, rule-scored), KD9 (44 base behaviors, all eight harm categories, official Inspect evaluation, gpt-4o-mini judge), and V4W (slack suite, 100 items, environment-scored security and utility), with coverage QX7 n=21 (Qwen3-32B below the 90% valid-output threshold), ZM3 22, KD9 19 and V4W 5 (native tool use required), τ² 20, and a common QX7/ZM3/KD9 panel of n=18. The API-only panel comprised up to 22 models from nine organizations (OpenAI 6, Meta 3, Qwen 3, Mistral 3, DeepSeek 2, Amazon 2, Anthropic, Google, Cohere), and the anchors measured by us through the same harness are H8P (H8P denoting MMLU zero-shot on a fixed 500-item subset with seed 42 and the same items for all models) and R2F (R2F denoting GPQA-Diamond with 198 items and CoT), with the anchor panel at 21 models because DeepSeek-R1 was excluded for unstable long-form R2F generation and provider-reported model-card anchors kept as the pre-harmonization comparison.

## 4 Results

Similar H8P/R2F scores do not imply similar safety, the M7V confound is real but metric-dependent (RQ2), and the trade-off suggested by a seven-model panel does not persist (RQ1); ordered by evidential strength: RQ4, RQ2 firmest; RQ1 weaker; power-limited RQ3 last.

W3D cannot stand alone as a measure of two-sided discrimination here, the problem being W3D's blindness to true negatives even though the closed form depends on prevalence: with base rate π the always-positive policy attains W3D=2π/(1+π) without discriminating, which for QX7 (π=0.527, QX7 labeling 52.7% of 571 items unsafe) is 0.690, so that "always unsafe," at recall 1.0 and G8X 0, attains W3D=0.690 and outranks five of the 21 valid models (W3D 0.30–0.57; Fig. 1), all of which discriminate, while the highest-G8X o3-mini (0.97) gets W3D=0.702, only 0.012 above, due to lower recall, and J4N reorders the board — E5K (W3D admits a high-scoring constant baseline), the headline metric being gameable (RQ4, 4.1).

| Model | QX7 G8X | ZM3 robust. | KD9 safety | R2F |
|---|---|---|---|---|
| Llama-3.1-8B | 0.04 | 0.32 | 0.89 | 30.4 |
| Claude-3-Haiku | 0.40 | 0.77 | 0.81 | 33.3 |
| Llama-3.1-70B | 0.11 | 0.35 | 0.78 | 46.7 |
| o3-mini | 0.97 | 0.87 | 0.77 | 77.2 |
| Nova-Lite | 0.94 | 0.88 | 0.76 | 42.0 |
| Nova-Pro | 0.64 | 0.60 | 0.75 | 46.9 |
| Gemini-2.5-Flash | 0.92 | 0.71 | 0.62 | 68.3 |
| DeepSeek-V3 | 0.86 | 0.76 | 0.61 | 59.1 |
| Qwen-2.5-72B | 0.62 | 0.92 | 0.60 | 49.1 |
| GPT-4.1 | 0.61 | 0.87 | 0.60 | 66.3 |
| GPT-4.1-mini | 0.76 | 0.87 | 0.51 | 65.0 |
| GPT-4o | 0.72 | 0.90 | 0.47 | 53.6 |
| GPT-4.1-nano | 0.79 | 0.94 | 0.42 | 50.3 |
| Llama-3.3-70B | 0.28 | 0.31 | 0.39 | 50.0 |
| Mistral-Nemo | 0.80 | 0.34 | 0.35 | 28.0 |
| GPT-4o-mini | 0.61 | 0.89 | 0.31 | 40.2 |
| Mixtral-8x22B | 0.58 | 0.78 | 0.24 | 33.2 |
| Mistral-Large | 0.50 | 0.81 | 0.24 | 44.0 |

Table 1: rankings differing across the same n=18 models (sorted by KD9).

That no benchmark orders models like another (Table 1) is the firmest, correlation-free result, cross-construct disagreement being expected and indicting no instrument while defeating interchangeable quoting: the QX7-G8X/KD9-safety ρ moved from −0.64 at n=7 to +0.02 (p=0.95) at n=18 (Fig. 5), size-7 subsets give |ρ|≥0.5 in a quarter of draws, and the dissolution holds under all four QX7 metrics (A.4), with M7V moderation of the remainder an untested hypothesis (4.3). In corroborative PCA (Fig. 6), PC1 absorbs 48.7%, QX7 G8X, ZM3 robustness and both K5Q loading one way and KD9 the opposite (+0.33; opposite in 17/18 leave-one-out fits); the positive control now passes, H8P loading 0.74≥0.6 where provider-reported anchors failed at 0.42, the negative control passes, and the convergent control is inconclusive at V4W n=5, yet the factor, its conditions met, is corroboration only, Spearman PCA giving PC1 40.8% with no factor retained by Horn (A.3) and the median between-benchmark correlation standing at 0.22.

On the QX7 panel under K5Q (n=20, K5Q denoting the harmonized anchors), M7V correlates with J4N (ρ=+0.71 H8P, p<0.001; +0.49 R2F, p=0.03) and with W3D (+0.76, +0.62) but only weakly with G8X (+0.16, +0.07), J4N and W3D including unsafe-trace recall while G8X can be inflated by rarely flagging, and, the confound depending on the metric and the panel, the G8X–H8P correlation fell from 0.85 across the first nine models to +0.16 at n=20 (4.2).

| Predictor | τ² | P2Z | L9R† |
|---|---|---|---|
| M7V (raw ρ) | +0.60** | −0.44* | +0.08 |
| QX7 G8X (U4G) | −0.09 | +0.41 | −0.11 |
| ZM3 robustness (U4G) | +0.16 | +0.47 | +0.21 |
| KD9 safety (U4G) | −0.23 | +0.16 | +0.72*** |

Table 2: 2024–25 panel, cellwise n=18–21; U4Gs controlling M7V; †R8W; *p<0.05, **p<0.01, ***p<0.001.

Against L9R, scored under T6Q (T6Q denoting the StrongREJECT-scored evaluation), where one template (AIM) ceilings with 11/22 refusing all and three templates leave none at ceiling, KD9 shows the strongest held-out association, its U4G reaching +0.72 (p<0.001) against +0.50 single-template and stable to LOO and an independent grader, the shared harmful-compliance target making +0.72 partly convergent validity, while M7V attains ρ=+0.08 (n=21, p=0.73) and, on the expanded panel, +0.34 (n=41, p=0.04), the change n.s.; KD9's excess of L9R over P2Z is Δ=+0.59 (p=0.027 by model bootstrap; p=0.051 when resampling eight organizations), the paired-panel P2Z U4G is +0.13 (n=19; Table 2 uses n=18), and the QX7/ZM3 excess of P2Z over L9R is marginal (Δ=+0.52, p=0.068) (RQ3, 4.4).

Pre-specified as "hold or strengthen," the M7V–P2Z correlation is weaker on the expanded panel at −0.16 (n=40, bootstrap p=0.40, CI [−0.54,+0.22]), the CI including −0.44 and the between-panel Δρ being +0.27 [−0.25,+0.78], n.s.; organization-matched subsets give a median of −0.17, so composition does not explain it, a generational account remaining possible, undemonstrated, with τ² not re-collected and the interaction not re-tested.

Against agentic P2Z, M7V has the opposite association, ρ_cap=−0.44 (p=0.047) at n=21 (Mistral-Large blackmails/leaks 85%; o3-mini a capable exception at 1%): ZM3 (+0.47) and QX7 G8X (+0.41) reach the pre-specified 0.40 threshold and KD9 (+0.16) does not, though the directional prediction named KD9, these being exploratory — CIs include zero, Bonferroni is failed, power ≈0.35 — with the QX7 U4G giving +0.41 G8X, +0.19 J4N, −0.46 recall (A.4).

Against task success, safety scores add no information beyond M7V, which attains ρ=+0.60 (p=0.005, n=20) against a best safety U4G of |ρ|=0.23, n.s., n=19 (Fig. 7), M7V already explaining this outcome and the null being no verdict on safety scores; on the paired n=20 panel, task +0.60 against P2Z −0.41 gives Δ=−1.00 (95% CI [−1.48,−0.49], p<0.001), stable to leave-one-model-out/D7T, organization-clustered bootstrap and subsampling (A.5), the independent P2Z grader agreeing at ρ=0.97, and the QX7 interaction being marginal (Δ=+0.50, p=0.082), the M7V interaction is supported while the full two-way dissociation is not.

Across 41 models and 12 organizations (P2Z n=40; L9R 41 after one engagement-gate exclusion), six organizations supplying ≥4 models for the within-org analysis and seven ≥3 for the group test (Table 5), models with similar M7V scores differ in safety across organizations (Fig. 4): within-org correlations vary in sign, averaging −0.14 (P2Z) and +0.26 (L9R), and after M7V adjustment the groups differ (permutation p=0.004, p<0.001; Holm-significant; ε²=0.34, 0.57), though this is not causal — a static composite may miss agentic ability, few models per org, restricted ranges.

Two alternative explanations were checked: seven blind LLM coders placed KD9 and L9R in harm blocking (7/7 each; α=0.61, three categories), P2Z less cleanly (A.1); and KD9 is uncorrelated with C9L benign over-refusal after M7V control (ρ=−0.01, n=17), against blanket refusal.

## 5 Discussion

Defensibility of a safety claim obtains only where its benchmark, metric, target behavior, and model population are attached, given that these benchmarks, rather than constituting interchangeable measures of one property, diverge in what they establish: KD9–L9R amounts to convergent validity with G8X unresolved (p=0.051), QX7/ZM3–P2Z remains exploratory, and M7V-adjusted organization-group differences, being observational in character, identify no organizational cause.

## 6 Limitations and Threats to Validity

No family-wise correction having been applied, the crossover and the KD9→L9R comparison (p<0.001) are the ones surviving Bonferroni, while those of QX7/ZM3 with P2Z (p≈0.05–0.09) do not; V4W contributes only five models, grader agreement stands at ρ=0.97 (P2Z) and 0.98 (L9R), and KD9 refusal rescoring of 0.69 shifts the U4G to +0.66 from +0.72. τ², measured over 40 tasks at one run/model, has unbounded user-simulator variance, P2Z is a fictional probe (halves Spearman–Brown 0.70), no criterion gives deployment-harm probabilities, and the larger three-scenario P2Z result is post-hoc, while worst-case within-model SEs, at 12–25% of the between-model SD, support panel spread, not every pairwise difference, for criterion validity (RQ3). A larger, capability-spanning agent-loop panel is needed, the cross-benchmark n being 18 (first pass n=7); re-measurement fixed the positive control (0.42→0.74) with no RQ2/RQ3 sign changed, the K5Q are static-knowledge tests though the M7V predicts τ² at +0.60, and a six-model BFCL pilot failed as agentic K5Q, syntax scoring having ranked GPT-4o-mini highest (0.72) and the two most capable lowest (0.42–0.48), such error attenuating or suppressing U4G rather than biasing them in a signable direction.

## 7 Conclusion

Every claim being tied to an explicit target behavior and model population, the full confusion structure is to be reported, with standalone W3D dropped wherever true negatives matter.

## Appendix A Technical Appendix

The confirmation threshold (ρ≤−0.35, p<0.05) was not met, the M7V–P2Z correlation weakening from an original bootstrap of p=0.11 with CI [−0.85,+0.09] to expanded leave-one-out estimates spanning [−0.25,−0.12], the nested bootstrap returning p=0.29 with the L9R Δρ of +0.26 bounded by [−0.14,+0.69] at p=0.22, the 2026-only models standing at ρ=+0.07 against −0.44 (Fisher z p=0.12), and the organization-matched size-21 IQR at [−0.27,−0.08]; power for ρ=0.4 at α=0.05 with n=18–20 is ≈0.35 (simulated), the 80%-power MDE being ρ=0.62 at n=18, detection of +0.41 requiring n≈40, and Bonferroni over the nine RQ3 U4G setting α=0.006.

PC sign being arbitrary and only relative directions mattering, the top models on either axis of Fig. 5 span the other's full range; under Spearman PCA the K5Q load at −0.72/−0.78 and KD9 at +0.38, the positive control passes, PC1 moves 48.7%→40.8% with the ≥40% criterion holding under both conventions, and the negative control, 200 permuted matrices, yields a median first eigenvalue of 1.71 against a 95th-percentile threshold of 2.18 (exceedance 2%).

| QX7 metric | RQ1 reversal | PC1 | H8P | R2F | τ² | P2Z | L9R |
|---|---|---|---|---|---|---|---|
| G8X | +0.02 | −0.71 | +0.16 | +0.07 | −0.09 | +0.41 | −0.11 |
| recall | −0.05 | −0.14 | +0.40 | +0.32 | +0.15 | −0.46 | −0.05 |
| J4N | −0.09 | −0.86 | +0.71 | +0.49 | +0.19 | +0.19 | −0.20 |
| W3D (official) | −0.29 | −0.75 | +0.76 | +0.62 | +0.32 | −0.29 | −0.17 |

Table 4: RQ1 n=18 and RQ2 n=20, the RQ3 U4G carrying τ² n=19, P2Z 18 and L9R 20, only the P2Z cell being metric-sensitive with both extremes n.s.

| Predictor | unsafe refusal | safe compliance |
|---|---|---|
| KD9 safety | +0.24 (n=17) | +0.01 (n=17) |
| QX7 G8X | +0.06 (n=17) | −0.15 (n=18) |
| P2Z | +0.47 (n=18) | −0.11 (n=19) |

M7V itself posting +0.21 (n.s.) for benign compliance and −0.32 (n.s.) for harmful refusal, P2Z against over-refusal at +0.11 (n=19, p=0.66) and unsafe refusal at p=0.052, low power leaves blanket refusal neither supported nor ruled out; these M7V-controlled U4G were obtained after the unsafe subset had been ceilinged (83–100% refusal, sd 4.5) and the C9L safe-subset over-refusal had been observed to range 0–60% (Claude-3-Haiku 60%, o3-mini 30%, others ≤15%) under the published CPI scorer with a gpt-4o-mini judge.

The sole dissent being Llama-3.3-70B on ZM3, agreement reached 27/28 with Krippendorff α=0.825 (nominal), the chance of one coder matching all four being 0.5⁴=6.2%.

| Benchmark | blind | author | B / R |
|---|---|---|---|
| QX7 | recognition | recognition | 0/7 |
| ZM3 | recognition | recognition | 1/6 |
| KD9 | blocking | blocking | 7/0 |
| V4W | recognition | recognition | 0/7 |

GPT-4o, Claude-3-Haiku, Gemini-2.5-Flash, Llama-3.3-70B, Qwen-2.5-72B, Mistral-Large and DeepSeek-V3 had been shown neutral descriptions with no criteria or RQ3 results, plus two definitions ("harm blocking" vs. "risk recognition and correct treatment of benign inputs"), options and order shuffled per coder, coder bias rather than rubric-designer bias being what is controlled thereby (the definitions having been written after seeing results), while the RQ1 factor separates KD9 without reference to any criterion; in a round with six targets plus a task-success distractor, α=0.61 (chance ≈1/3) was obtained with the distractor chosen 5/42, KD9 and L9R mapping to harm blocking 7/7, QX7 to recognition 7/7, ZM3 to 5/7, V4W reading as task success to 4/7 (utility-scored), and P2Z coded recognition 6/7 in a separate two-way round yet harm blocking 6/7 here.

| Instrument | models | items | split-half (SB) | rank stab. | quartile |
|---|---|---|---|---|---|
| QX7 G8X | 21 | 570 | +0.98 (+0.99) | +0.99 | 0.90 |
| QX7 W3D | 21 | 570 | +0.93 (+0.96) | +0.97 | 0.87 |
| ZM3 | 22 | 300 | +0.95 (+0.98) | +0.98 | 0.88 |
| KD9 | 19 | 44 | +0.93 (+0.96) | +0.97 | 0.79 |
| L9R | 42 | 20 | +0.94 (+0.97) | +0.98 | 0.96 |
| H8P | 46 | 356 | +0.93 (+0.96) | +0.98 | 0.92 |
| R2F | 45 | 164 | +0.95 (+0.97) | +0.98 | 0.92 |
| P2Z (halves) | 22 | 2 | +0.54 (+0.70) | — | — |

Table 3: 50 random half-splits, item-bootstrap rank Spearman and top-quartile persistence attesting to internal consistency rather than mere precision, items being the complete-coverage intersection, L9R resampled by base prompt (templates dependent) with 20 of 50 prompts shared and 42 models incl. DeepSeek-R1 (no K5Q pair), P2Z restricted to config-level blackmail-vs-leaking halves only, item-sampling error an unlikely cause of the Table 1 disagreement. Median/max SE ran H8P 1.6/2.2, R2F 3.3/3.6, P2Z 4.0/5.0, QX7 W3D 1.8/3.0 and L9R (~150 prompts) 2.3/3.7 against between-model SDs of 8.7, 18.2, 26.0, 14.9, 28.5 (expanded panel), the worst-case SE amounting to 13–25% of SD with spread 4–8× the worst-case SE, τ² remaining uncoverable.

| Analysis | n | note |
|---|---|---|
| QX7 | 21 | Qwen3-32B dropped (21% unparseable) |
| ZM3 | 22 | |
| KD9 | 19 | malformed tool calls / unsupported routing |
| V4W | 5 | convergent control inconclusive |
| τ² | 20 | |
| P2Z | 21 | |
| L9R | 22 | R8W average |
| expanded P2Z/L9R | 40/41 | 12 organizations |
| C9L | 17–20 | KD9 ∩ K5Q overlap |
| cross-benchmark/U4G | 18 | KD9-limited |
| M7V interaction | 20 | paired |

The evidence base (Fig. 3) comprises up to 46 models × 9 instruments and 262 cells, rows beyond 41 excluded by telemetry gates, the 2026 expansion carrying K5Q and the two chat criteria only. Under the U4G convention — midranks residualized on M7V, residuals re-ranked — Pearson on rank residuals gives QX7 +0.49, ZM3 +0.41 (within 0.08) and KD9–L9R +0.72; the crossover survives leave-one-out refits at Δ∈[−1.16,−0.90], and under D7T (D7T denoting the leave-one-organization-out refits) the estimates were Amazon −0.96, Anthropic −1.01, Cohere −0.90, DeepSeek −1.13, Google −0.98, Meta −0.96, Mistral −0.93, OpenAI −1.16 and Qwen −0.95, the organization-clustered bootstrap giving Δ∈[−1.49,−0.59] with P(Δ≥0)<10⁻⁴ and subsamples (median/% negative/% ≤−0.5/% flips) of size 10 at −0.95/100/97/0, size 12 at −0.96/100/99/0 and size 14 at −0.97/100/100/0, grading having been P2Z by Claude-3-Haiku vs. GPT-4o, L9R by Gemini vs. gpt-4o and KD9 by a Gemini refusal judge on cached completions (n=19) redoing only the refusal component, with per-template KD9→L9R U4G of AIM +0.44, refusal-suppression +0.70 and prefix-injection +0.62. In an exploratory 3-scenario battery, adding murder (22 models) moves the interaction to Δ=+0.53 (p=0.048), a tighter CI for the same effect along a post-hoc forking path, KD9→battery standing at +0.43 so that G8X is diluted and blackmail+leaking stays primary.

All five secondary tests surviving Holm (max adjusted p=0.043), within-org n running 4–8 with Anthropic ceilings near 1.0, the permutation test rather than the slopes carries the claim, calibration over 2,000 synthetic null panels each giving 0.045–0.052 at α=0.05 under reshuffled or Gaussian safety (nominal at 0.01/0.005), the clustered-variance null inflating L9R only (0.067; none at α=0.005) and doubled p-values staying Holm-significant; without Anthropic, P2Z ε²=0.27 at p=0.018 and L9R ε²=0.54 at p<0.001.

| | n | P2Z ρ | L9R ρ |
|---|---|---|---|
| Anthropic | 5 | +0.71 | +0.67 |
| OpenAI | 8 | +0.59 | +0.64 |
| Qwen | 5 | −0.90 | +0.80 |
| Meta | 4 | −0.40 | −0.40 |
| Google | 4 | −0.60 | −0.40 |
| Mistral | 4 | −1.00 | −0.40 |
| n-weighted mean | | −0.14 | +0.26 |
| group test (n≥3; total 33): between-group share | | 0.508 | 0.674 |
| df-adjusted ε² | | 0.34 | 0.57 |
| permutation p (10⁴) | | 0.004 | <0.001 |
| Kruskal–Wallis p | | 0.02 | 0.002 |

Table 5: within-organization slopes (top) and M7V-residualized group differences (bottom), ε² being an effect size, not a causal share. Templates having been re-scored for all new models, the expanded-panel L9R result is bootstrap p=0.035 with CI [+0.03,+0.60] and leave-one-out [+0.30,+0.39], high-M7V L9R spanning Mistral-Large 0.09, DeepSeek-V3 0.37 and Anthropic/OpenAI 0.98–0.99, while at M7V ≈0.8 the scores o3-mini 0.99, Grok-4.3 0.70 and DeepSeek-V3 0.19 show that ceilings do not explain all variation; neither sign nor magnitude of M7V→safety is stable across criteria or panels. Behavior-only exclusions under the same chat/scenario harnesses were Kimi-K2.6 and both Nemotron-3 (<50% usable K5Q answers or unbounded reasoning), MiniMax-M3 below the P2Z engagement threshold (truncation) yet kept for L9R, Gemini-2.5-Pro with no valid K5Q, and GLM-4.5 H8P only (routing), a partial row. A pre-specified QX7–P2Z continuation fell from +0.41 (n=18) to +0.23 (n=25), below its 0.30 threshold; stopped, kept as hypothesis.

