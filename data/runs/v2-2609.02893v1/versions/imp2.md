# Probe Generalization as Subspace Selection for OOD Deception Detection

Daniel Yoo

  
Adrians Skapars

###### Abstract

For Llama-3.1-8B-Instruct deception probes evaluated on 3 held-out datasets, source-trained probes restricted to a small subset of source-domain principal components (PCs) nearly match target-trained probes. An LLM judge's OOD scores over PC interpretations, computed without target data, select a subspace that closes the baseline-to-oracle gap by 78% on Insider Trading Report and by 25% on Sandbagging.

Machine Learning, ICML

## 1 Introduction

Linear probes (Alain and Bengio, 2018; Hewitt and Liang, 2019) are unreliable out-of-distribution (Kirch et al., 2026) despite linear representation of behavioral concepts (Marks and Tegmark, 2024; Park et al., 2024; Zou et al., 2025), and pick up spurious correlations (Orgad et al., 2025; Wang et al., 2025). We work in the source-domain PCA basis rather than the model's latent dimensions; cf. Mu et al. (2018), where removing top PCs of an embedding space helps downstream tasks, and Kantamneni et al. (2025), who use top-PC projections as a probe baseline. Question: which PC subsets support generalizable probe training, and how are they identified?

Testbed: deception probes on Llama-3.1-8B-Instruct (Grattafiori et al., 2024), source Roleplaying, targets Insider Trading Report, Insider Trading Confirm, Sandbagging (Goldowsky-Dill et al., 2025).

Contributions:

•

Section 4.2: target-guided greedy search finds 8–15 source PCs on which a source-trained probe nearly matches a target-trained probe.

•

Section 4.3: rankings by source-probe or target-probe weights do not transfer better than the full-representation baseline.

•

Section 4.5: an LLM judge partially recovers the subspace without target data.

Figure 1: Overview of PC-based subspace selection for OOD deception probing. Activations are projected onto the source-domain PCA basis, candidate PC directions are filtered, and probes are trained on the selected low-dimensional subspace.

## 2 Related Work

#### Linear concept directions in activation space.

Linear true/false structure across datasets (Marks and Tegmark, 2024); the linear representation hypothesis linking probing and steering (Park et al., 2024); population-level directions for honesty and emotion (Zou et al., 2025). A single refusal direction (Arditi et al., 2024); an “assistant” axis (Lu et al., 2026).

#### Linear probes and their limitations.

Probing as analysis (Alain and Bengio, 2018; Hewitt and Liang, 2019); safety-relevant probes for latent truthfulness (Burns et al., 2024) and strategic deception (Goldowsky-Dill et al., 2025). Critiques: probe accuracy versus encoded information (Hewitt and Liang, 2019; Ravichander et al., 2021); uneven OOD generalization of strong in-distribution probes (kramár2026buildingproductionreadyprobesgemini).

#### Interpreting features with natural-language explanations.

LLM-generated neuron explanations from high-activation examples (Bills et al., 2023), extended to SAE features ((Paulo et al., 2025), (Bricken et al., 2023)); concept ablation during fine-tuning via automated interpretation (Casademunt et al., 2025).

## 3 Setup

#### Datasets.

4 datasets from Goldowsky-Dill et al. (2025) (Appendix A): Roleplaying (source); Insider Trading Report, Insider Trading Confirm, Sandbagging (targets). GPT-4o (OpenAI, 2024) labels: honest (0), ambiguous (-1), deceptive (1); -1 removed.

#### Language models.

Layer 15 residual stream of Llama-3.1-8B-Instruct (Grattafiori et al., 2024), the highest cross-domain AUROC layer in Ying et al. (2026); mean over assistant-response tokens.

#### Probe training and evaluation.

Source activations are standardized; targets use source statistics. PCA on scaled source activations, top 100 PCs retained. Logistic regression ( $\lambda=10$ ); for subset $S$, activations projected onto $S$; baseline on full standardized activations. The top 100 are used since SAE decoder alignment (Bricken et al., 2023) decreases with PC index (Appendix C); no claim is made that low-index PCs generalize better.

## 4 Experiments

### 4.1 Setting a baseline

Figure 2: Probe AUROC on target datasets for source-trained probes on full representations (blue) and top-100 PCs (orange), and target-trained probes on top-100 source PCs (green, 5-fold CV). Error bars: 95% CIs

Source-trained full-representation baseline AUROC: 0.874 (Insider Trading Confirm), 0.706 (Insider Trading Report), 0.606 (Sandbagging). Target-trained, top-100 source PCs, 5-fold CV: 0.936 / 0.950 / 0.869.

### 4.2 Showing a transferable subspace exists

Figure 3: Probe AUROC across the three targets for the full-representation baseline (blue), top-k PCs by variance (teal, k matched to greedy), and greedy oracle PCs (purple). Error bars: 95% CIs.

Greedy oracle: from an empty set, add the PC among the top 100 maximizing target AUROC of a probe fit on source projections onto the current subset; stop when gain $\le$ 0.001. 5-fold CV over the target: selection on non-held-out folds, evaluation on the held-out fold. Mean subset size: 15 (Insider Trading Report), 8 (Insider Trading Confirm), 14 (Sandbagging). Being target-guided, it is an upper bound, not an unbiased OOD estimate. Control: top $k$ PCs by variance, $k$ matched to greedy.

Greedy nearly matches target-trained probes on Insider Trading Report and Sandbagging, +0.22 and +0.23 AUROC over baseline. Variance control matches greedy only on Insider Trading Confirm and falls far behind on Insider Trading Report and Sandbagging.

### 4.3 Testing hypotheses for transferable principal components

Figure 4: Probe AUROC across the three targets for three PC-selection rules: ranking by source-probe weights (red), by target-probe weights (pink), and greedy oracle selection (purple). Error bars: 95% CIs.

Hypotheses: greedy PCs are (i) those individually separating target classes, or (ii) those the source probe weights most. Rankings by $|w_j|\,\sigma_j$ from a probe on source-PCA projections fit on source data (‘source-PCA probe’) or target data (‘target-PCA probe’); in both cases a probe is then trained on source data projected onto the top $k$ ranked PCs. $k$ from greedy on the full target (no CV): $k=13$ for Insider Trading Report, 7 for Insider Trading Confirm, 14 for Sandbagging; greedy numbers therefore differ slightly from Section 4.2.

Source-PCA probe ≈ full-representation baseline. Target-PCA probe: good on Insider Trading Report, below greedy on Insider Trading Confirm, below chance on Sandbagging. See Appendix D.

### 4.4 Interpreting the transferable subspace

Following Bills et al. (2023) and Casademunt et al. (2025): for each of the top 100 source PCs, the 8 most and 8 least activating source examples are given to an LLM, which describes positive pole, negative pole, and direction, and assigns an OOD score on 1–10 (Appendix B). Maximum observed score: 6. PCs scoring 6 in both runs: 19.

5 PCs appear in greedy selections for more than one target; 3 of 5 score 6. Of the top 5 source-PCA-probe PCs, 1 scores 6. Appendix E.

### 4.5 Selecting a subspace by interpreting PCs

Figure 5: Probe AUROC across the three targets for the interp-selected subspace (red; 19 PCs) versus three controls: full-representation baseline (blue), top-19 PCs by variance (teal), and the remaining 81 PCs after ablating the interp-selected PCs (brown). Error bars: 95% CIs.

Probe on the 19 interp-selected PCs (score 6 in both of two runs), no target data. Controls: the 81 ablated-complement PCs; top 19 by variance.

Interp-selected beats baseline on all 3 targets, beats the variance control (by wide margins on Insider Trading Report and Sandbagging), and beats the 81-PC ablation on all 3. With gap closed $=(\text{AUROC}_{\text{interp}}-\text{AUROC}_{\text{base}})/(\text{AUROC}_{\text{greedy}}-\text{AUROC}_{\text{base}})$: Insider Trading Report 78% (baseline 0.706, greedy 0.929); Sandbagging 25% (greedy 0.832). Insider Trading Confirm: baseline near ceiling, all methods but interp-ablated roughly equal; uninformative.

## 5 Limitations

Single model and source dataset.
Llama-3.1-8B-Instruct, layer 15; source Roleplaying (Goldowsky-Dill et al., 2025). Insider Trading Confirm baseline is already 0.874. Other model families and datasets untested.

Sandbagging margins are small.
0.606 to 0.662, 25% of the gap to greedy (0.832); Sandbagging is the most out-of-distribution target. Insider Trading Report: 0.706 to 0.881, 78% of the gap to 0.929.

Linear pipeline. PCA projection plus logistic regression throughout; nonlinear probes and other readout locations untested.

No causal validation.
PC “encodings” rest on top-activating examples and judge verbalizations; no inference-time ablation of high-scoring PCs.

## 6 Conclusion

A small subset of source PCs supports cross-domain transfer of deception probes nearly matching target-trained probes, is not recovered by source-side or target-side separability, and is partially recovered by LLM-judge OOD scores without target data. OOD probing is thus a representation-selection problem: choose a subspace preserving the behavioral feature and excluding spurious correlation, in addition to fitting a classifier.

## Acknowledgment

This work was conducted during the 11th edition of AI Safety Camp, which provided the compute to run our experiments. Thanks to Prasad Mahadik and Arya Hariharan for useful discussions around this work.

## Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none
which we feel must be specifically highlighted here.

## Appendix A Dataset Details

Roleplaying: the model is placed in a simple scenario with an opportunity to lie.

Insider Trading Report: acting as a trader, the model decides whether to honestly report that it used insider information.

Insider Trading Confirm: after hiding earlier use of insider information, the model is asked to confirm it did not use any.

Sandbagging: the model is told to deliberately under-perform on some tasks while behaving normally on others.

| Dataset | Total | Label Counts

| Examples | (-1 / 0 / 1)

| Roleplaying | 371 | 140 / 59 / 172

| Insider Trading Report | 473 | 218 / 194 / 61

| Insider Trading Confirm | 151 | 29 / 38 / 84

| Sandbagging | 1000 | 379 / 375 / 246

Table 1: Dataset label distributions.

## Appendix B LLM Judge Prompt

Judge: GPT-5-mini (OpenAI, 2025).

You are interpreting a principal component direction in a neural representation space.
This direction is bipolar:

- Positive activations mean movement toward one pole.

- Negative activations mean movement toward the opposite pole.

- Larger magnitudes mean a stronger match to that pole.

Each line is formatted as token<TAB>activation, with activations normalized to integers from -10 to +10.

The examples below were selected from a dataset with this setup: The model is instructed to Roleplay in a simple scenario.
The model is asked a question which gives a strategic chance to lie.

High-positive examples:

{positive_examples}

High-negative examples:

{negative_examples}

Infer the dominant contrast captured by the direction using both the positive and negative examples.

Important instructions:

- Do not assume the direction is a broad sentiment direction unless the examples clearly support that.

- Prefer the narrowest accurate description over a broad generic one.

- Do not overgeneralize beyond what is supported by the examples.

- Rate how likely this direction is to be useful for out-of-distribution deception on a scale from 1 to 10, and briefly explain why.

- Give low scores to directions that are mainly surface cues, lexical habits, or task-specific phrasing, even if they correlate with deception in these examples.

- A direction that is noisy, brittle, context-dependent, or likely to fail under different wording or domains should usually be scored around 1 to 4, not 5 or above.

- Reserve scores of 5 or higher for directions that seem plausibly tied to a more general deception-relevant strategy or behavior, rather than local wording.

Write your answer in exactly this format:

Positive pole: ...

Negative pole: ...

Shared direction: ...

OOD deception usefulness (1-10): ...

## Appendix C Lower PCs Show Stronger Decoder Alignment

Figure 6: Lower PCs show stronger decoder alignment, both for the best-matching SAE feature and for the mean of the top 10 features (Spearman $\rho$ = -0.857 and -0.917, respectively). We used andyrdt/saes-llama-3.1-8b-instruct (andyrdt, 2024)

.

## Appendix D PC Set Comparisons

Jaccard overlap, greedy vs. target-PCA: 0.182 (Insider Trading Report), 0.077 (Insider Trading Confirm), 0.167 (Sandbagging), i.e. agreement on under a fifth of selections.

| Selection method | Dataset / target
|

Selected PCs

| Source probe weights | Source
|

PC7, PC8, PC11, PC12, PC4, PC5, PC2, PC3, PC6, PC27, PC22, PC21, PC1, PC41, PC18

| Greedy selection | Insider Trading Report
|

PC6, PC89, PC62, PC66, PC26, PC14, PC57, PC25, PC93, PC9, PC70, PC97, PC78

| Greedy selection | Insider Trading Confirm
|

PC71, PC53, PC28, PC86, PC32, PC81, PC77

| Greedy selection | Sandbagging
|

PC37, PC62, PC54, PC41, PC46, PC9, PC57, PC44, PC51, PC43, PC19, PC48, PC86, PC81, PC33

| Target separability ranking | Insider Trading Report
|

PC5, PC6, PC89, PC18, PC79, PC46, PC87, PC4, PC14, PC34, PC29, PC1, PC66, PC67, PC19

| Target separability ranking | Insider Trading Confirm
|

PC4, PC1, PC8, PC53, PC30, PC7, PC13, PC18, PC5, PC20, PC26, PC49, PC28, PC31, PC97

| Target separability ranking | Sandbagging
|

PC51, PC91, PC97, PC13, PC39, PC77, PC4, PC66, PC53, PC86, PC62, PC23, PC41, PC10, PC61

Table 2: Comparison of PCs selected by source-only probe weights, greedy transfer selection, and target separability ranking.

## Appendix E PC Interpretability

Table 3: Principal components selected for likely transferability, based on interpretation with an LLM judge.

| PC | Selected On | OOD Score
|

Full explanation

| PC62 | Insider Trading Report + Sandbagging | 6
|

OOD deception usefulness (1-10): 6 — This direction maps onto a general deception-relevant behavior (deny/deflect vs provide details), so it can be useful across settings, but it is noisy: true speakers also deny or obfuscate for legitimate reasons and liars sometimes produce elaborate fabricated detail, and the signal may pick up simple lexical negation rather than semantic strategy.

| PC57 | Insider Trading Report + Sandbagging | 6
|

This direction aligns with a general deception-relevant strategy (deny/minimize or present innocuous explanations vs give specific/incriminating details), so it can help flag rhetorical moves often used when hiding or deflecting. However it is partly lexical and task-specific (relies on surface denials, hedges, and domain terms), so it will be brittle and context-dependent out of distribution.

| PC9 | Insider Trading Report + Sandbagging | 3
|

OOD deception usefulness (1-10): 3 — This direction mainly picks up domain‑specific lexical cues (medical/leave language vs project/status language) and stylistic framing rather than a generalizable deception strategy. It may flag genuine illness claims and is brittle across domains and rephrasings.

| PC81 | Insider Trading Confirm + Sandbagging | 3
|

OOD deception usefulness (1-10): 3 — The direction picks up a plausible deception‑relevant contrast (admit vs deny/hedge), but it mainly relies on surface/formal cues (negations, hedges, résumé style, enumerations) that are task- and wording‑dependent and likely brittle across domains and phrasings.

| PC86 | Insider Trading Confirm + Sandbagging | 6
|

OOD deception usefulness (1-10): 6 — Hedging and defensive PR-style wording is a general tactic associated with deceptive or self-protective behavior, so the direction could transfer across domains. However it is noisy (used legitimately for privacy, politeness, or negotiation), sensitive to phrasing and role, and will produce false positives, so it’s only moderately reliable out-of-distribution.

Table 4: Principal components and their OOD scores with interpretations.

| PC | OOD Score
|

Full explanation

| PC7 | 3
|

OOD deception usefulness (1-10): 3 — This direction mostly picks up a pragmatic contrast (deny/deflect vs admit) which can correlate with deception in some settings, but it relies heavily on surface cues (explicit “No”, word order, presence of “I” + admission) and task-specific phrasing. It is therefore brittle and likely to fail or mislead in different domains or with varied wording.

| PC8 | 4
|

OOD deception usefulness (1-10): 4 — The direction captures a useful narrow contrast (denial/excuse vs admission) that can correlate with evasive or deceptive responses in similar QA/roleplay tasks, but it largely picks up surface cues (explicit “No”/“Yes” and common denial phrasing) and is likely brittle under paraphrase, different domains, or subtler deception strategies.

| PC11 | 3
|

OOD deception usefulness (1-10): 3 — The direction mainly picks up a stylistic/pragmatic contrast (assertive first‑person role/defense vs problem-reporting/negation). That can correlate with evasive or face-saving lies in some scenarios, but it’s surface-y and context dependent and likely to fail across different domains or phrasings.

| PC12 | 3
|

OOD deception usefulness (1-10): 3 — This direction mainly captures discourse style/forcefulness (a surface cue). While overconfident or over-assertive phrasing can sometimes correlate with deceptive defensive behavior, it is brittle and domain-dependent and will produce many false positives/negatives when applied across different speakers, topics, or cultural styles.

| PC4 | 6
|

OOD deception usefulness (1-10): 6 — This captures a general communicative strategy relevant to deception (deny vs admit), so it can transfer beyond the specific prompts. However it also leans on surface cues (explicit “Yes”/“No” turns and common defensive phrasings) and may be brittle when denials or lies use different wording or more subtle tactics.