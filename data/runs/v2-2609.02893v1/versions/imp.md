# Probe Generalization as Subspace Selection for OOD Deception Detection

Daniel Yoo

  
Adrians Skapars

###### Abstract

Linear probes can be used to detect behaviors and concepts inside language model activations, but may fail to transfer to out-of-distribution examples. We study the generalization of Llama-3.1-8B-Instruct deception probes trained on a single source dataset and evaluated on 3 held-out deception detection datasets, working in the basis of the top 100 source-domain principal components (PCs). Source-trained probes restricted to a target-selected subset of 8–15 PCs nearly match probes trained directly on the test distribution, whereas subsets chosen by explained variance, by source-probe weight, or by target-probe weight do not. Scoring each PC with an LLM judge on whether its most and least activating source examples imply a transferable deception direction, and probing only on the highest-scoring PCs, closes the baseline-to-oracle gap by 78% on Insider Trading Report and by 25% on Sandbagging, with no target data used for selection.

Machine Learning, ICML

## 1 Introduction

Linear probes are standard tooling for reading information from language model activations (Alain and Bengio, 2018; Hewitt and Liang, 2019).
They are cheap to train and recover semantically meaningful information from the residual stream.
Probes that look strong in-distribution are, however, sometimes unreliable out-of-distribution (Kirch et al., 2026). Key behavioral concepts are linearly represented in model activations (Marks and Tegmark, 2024; Park et al., 2024; Zou et al., 2025), yet probes trained on them appear to pick up spurious correlations in practice (Orgad et al., 2025; Wang et al., 2025), a concern for probe-based monitoring of harmful interactions.

We work in the basis of source-domain principal components rather than raw activations, so as not to treat the model’s latent dimensions as a privileged basis. Prior work indicates that the choice of PCs affects downstream behavior: Mu et al. (2018) show that removing the top PCs of an embedding space improves downstream performance, and Kantamneni et al. (2025) use top-PC projections as a probe baseline. We ask which subset of principal components supports generalizable probe training and how that subset can be identified.

The testbed is deception probing on Llama-3.1-8B-Instruct (Grattafiori et al., 2024), with probes trained on a single source dataset (Roleplaying deception scenarios) and evaluated on 3 held-out target datasets (Insider Trading Report, Insider Trading Confirm, and Sandbagging) (Goldowsky-Dill et al., 2025). Within the source-domain PCA basis, we compare strategies for selecting the training dimensions of a source-trained probe.

Contributions:

•

A small transferable subspace exists (Section 4.2).
A target-supervised greedy search over source PCs finds 8–15
components on which a source-trained probe nearly matches a
target-trained probe.

•

Source-side and target-side rankings both fail to
recover the subspace
(Section 4.3). Ranking PCs by the source or target probe’s weights does not produce a basis that transfers better than the full-representation baseline.

•

An LLM judge partially recovers the subspace without
target data (Section 4.5).
PCs selected by an LLM judge’s OOD score over natural-language interpretations outperform variance-matched and ablation controls on all targets.

Figure 1: Overview of PC-based subspace selection for OOD deception probing. Activations are projected onto the source-domain PCA basis, candidate PC directions are filtered, and probes are trained on the selected low-dimensional subspace.

## 2 Related Work

#### Linear concept directions in activation space.

A related line of work studies whether high-level behavioral or semantic properties are represented as directions or low-dimensional subspaces in activation space. Marks and Tegmark (2024) report linear structure in true/false representations across datasets, while Park et al. (2024) formalize the linear representation hypothesis and connect it to both probing and steering. Zou et al. (2025) model high-level properties such as honesty and emotion as population-level directions that can be read from and intervened on in model activations.

This perspective has also been used to explain and control model behavior. Arditi et al. (2024) show that a single direction can mediate refusal behavior, suggesting that some high-level behaviors may be localized in simple linear features. Lu et al. (2026) identify an “assistant” axis corresponding to assistant-like behavior. These works support both mechanistic understanding and activation-space interventions, while leaving open when such directions capture stable concepts rather than context-dependent correlations.

#### Linear probes and their limitations.

Linear and lightweight probes have a long history in interpretability as tools for analyzing what information is represented in neural networks (Alain and Bengio, 2018; Hewitt and Liang, 2019). Recent work has applied such probes to safety-relevant signals in LLM activations, including latent truthfulness without direct supervision (Burns et al., 2024) and strategic deception (Goldowsky-Dill et al., 2025).

Methodological skepticism is central to the probing literature. Hewitt and Liang (2019) argue that high probe accuracy alone is insufficient evidence that a model already represents the target property, since the probe may itself learn the task or exploit surface-level correlates. Ravichander et al. (2021) question whether probing reliably establishes what information is encoded or merely recoverable. Probes with strong in-distribution performance can also generalize unevenly out of distribution (kramár2026buildingproductionreadyprobesgemini), motivating evaluation of whether safety probes identify robust internal signals rather than dataset-specific artifacts.

#### Interpreting features with natural-language explanations.

Another approach interprets internal representations by assigning natural-language descriptions to neurons or learned features. Bills et al. (2023) introduced the use of LLMs to generate explanations of neurons from high-activation examples. This has been extended to the automatic labeling of features in sparse autoencoders ((Paulo et al., 2025), (Bricken et al., 2023)), enabling large-scale interpretation of learned feature dictionaries.

Natural-language feature descriptions have also been used for intervention: Casademunt et al. (2025) identify concepts through automated natural-language interpretation and ablate them during fine-tuning to reduce misaligned behavior.

## 3 Setup

#### Datasets.

Following Goldowsky-Dill et al. (2025), we use 4 of their deception detection datasets (Roleplaying, Insider Trading Report, Insider Trading Confirm, Sandbagging), described in Appendix A.

Roleplaying is the source domain and the other 3 are target test sets. Each response is labeled honest (0), ambiguous (-1), or deceptive (1) by GPT-4o (OpenAI, 2024); ambiguous responses are removed throughout.

#### Language models.

Following Ying et al. (2026), we use layer 15 residual stream activations of Llama-3.1-8B-Instruct (Grattafiori et al., 2024), the layer with the highest cross-domain AUROC in their analysis. Each prompt is passed through the model and token activations are averaged over the assistant’s response only.

#### Probe training and evaluation.

Source activations are standardized to zero mean and unit variance, and the same source statistics are applied to all target activations. PCA is run on the scaled source activations and the top 100 PCs are retained. All probes are logistic regression classifiers ( $\lambda=10$ ) evaluated by AUROC; for a subset $S$ of the 100 PCs, activations are projected onto $S$ before training and evaluation, while baseline probes train directly on the full standardized activations.

The top 100 PCs are used because alignment with SAE decoder features (Bricken et al., 2023) decreases with PC index (Appendix C), so lower-index PCs are better suited to natural-language interpretation and comparable with interpretability-driven selection. No claim is made that low-index PCs generalize better; Section 4.1 checks the information content of the 100-PC subspace on every target.

## 4 Experiments

### 4.1 Setting a baseline

Figure 2: Probe AUROC on target datasets for source-trained probes on full representations (blue) and top-100 PCs (orange), and target-trained probes on top-100 source PCs (green, 5-fold CV). Error bars: 95% CIs

Probes trained on the full source representations and evaluated on each target (Figure 2) reach 0.874 AUROC on Insider Trading Confirm, 0.706 on Insider Trading Report, and 0.606 on Sandbagging.

A probe trained directly on target data within the 100-PC source subspace, using 5-fold cross-validation, reaches 0.936 / 0.950 / 0.869 AUROC across the three targets.

### 4.2 Showing a transferable subspace exists

Figure 3: Probe AUROC across the three targets for the full-representation baseline (blue), top-k PCs by variance (teal, k matched to greedy), and greedy oracle PCs (purple). Error bars: 95% CIs.

We next consider greedy PC selection. Starting from an empty set, PCs are added one at a time: at each step every remaining candidate among the top 100 PCs is evaluated by projecting source activations onto the current subset plus the candidate, fitting a logistic regression probe on those source projections, and measuring AUROC on the target; the candidate yielding the largest improvement is added.

Probe weights are always fit on source data, while subset choice is guided by labeled target examples. Under 5-fold cross-validation over the target dataset, greedy selection is run on the non-held-out target folds and the resulting source-trained probe is evaluated on the held-out fold. The search terminates when no candidate improves AUROC by more than 0.001, converging on 15 PCs on average for Insider Trading Report, 8 for Insider Trading Confirm, and 14 for Sandbagging.

Since subset selection consumes target examples, the greedy probe is an upper bound on transfer performance rather than an unbiased OOD estimate. As a control we select the top $k$ PCs by variance, with $k$ matched to the greedy subset size (Figure 3).

The greedy probe approaches the target-trained probe on Insider Trading Report and Sandbagging, improving on the baseline by 0.22 and 0.23 points of AUROC, respectively. The variance control matches greedy only on Insider Trading Confirm and falls well behind on Insider Trading Report and Sandbagging.

### 4.3 Testing hypotheses for transferable principal components

Figure 4: Probe AUROC across the three targets for three PC-selection rules: ranking by source-probe weights (red), by target-probe weights (pink), and greedy oracle selection (purple). Error bars: 95% CIs.

We test 2 hypotheses about the greedy subset. Under the first, the transferable PCs are those that individually separate the target classes well across the test set, which greedy recovers through its access to target data. Under the second, they are the PCs the source probe already weights most heavily, so that greedy merely compresses the source signal.

Both are tested via the standardized coefficient given to each PC by a probe on activations projected onto the source-PCA basis. The ‘source-PCA probe’ ranks PCs by the weights of such a probe fit on source data; the ‘target-PCA probe’ ranks PCs by the weights of such a probe fit on target data. In either case, a probe is then trained on source data projected onto the top $k$ PCs under the respective ranking, and either hypothesis predicts recovery of greedy performance.

$k$ is set by running greedy on the full target dataset rather than per fold, giving a single stable PC set: $k=13$ for Insider Trading Report, 7 for Insider Trading Confirm, and 14 for Sandbagging. The greedy figures here therefore differ slightly from the cross-validated figures of Section 4.2 (Figure 4).

The source-PCA probe transfers about as well as the full-representation baseline. The target-PCA probe does well on Insider Trading Report, is worse than greedy on Insider Trading Confirm, and falls below chance on Sandbagging. Set overlaps are given in Appendix D.

### 4.4 Interpreting the transferable subspace

Following Bills et al. (2023) and Casademunt et al. (2025), for each of the top 100 source PCs we retrieve the 8 most and 8 least PC-activating source examples and prompt an LLM to describe the positive pole, the negative pole, and the shared direction. The judge then rates the usefulness of the direction for out-of-distribution deception detection on a 1–10 scale (OOD score); the prompt and model are given in Appendix B.
No PC received an OOD score above 6 on the judge’s 1–10 scale, so 6 is treated as the maximum observed OOD score, and PCs reaching it in both of two scoring runs are retained, yielding 19 PCs.

5 PCs appear in the greedy selections of more than one target; 3 of these 5 received the maximum OOD score. Of the top 5 PCs selected by the source-PCA probes, only 1 received the maximum OOD score. Per-PC interpretations for both groups are listed in Appendix E.

### 4.5 Selecting a subspace by interpreting PCs

Figure 5: Probe AUROC across the three targets for the interp-selected subspace (red; 19 PCs) versus three controls: full-representation baseline (blue), top-19 PCs by variance (teal), and the remaining 81 PCs after ablating the interp-selected PCs (brown). Error bars: 95% CIs.

The 19 PCs scoring the maximum OOD score in both runs form the interp-selected subspace; the two-run requirement accounts for noise in the judge scores. No target data enter this selection. Controls are a probe on the ablated complement (the 81 PCs not selected) and a probe on the top 19 source PCs by variance (Figure 5).

The interp-selected probe outperforms the baseline and the variance control on all 3 targets, by wide margins on Insider Trading Report and Sandbagging, and outperforms the interp-ablated probe on all 3 targets. We report gap closed as (interp AUROC − baseline) / (greedy AUROC − baseline).
On Insider Trading Report, interp-selection closes 78% of the gap between baseline (0.706) and the greedy oracle (0.929).
On Insider Trading Confirm, baseline transfer is already near ceiling and all methods other than the interp-ablated probe perform comparably.
On Sandbagging, interp-selection closes 25% of the gap to the greedy oracle (0.832).

## 5 Limitations

Our strongest claims are narrow, and several limitations apply.

Single model and source dataset.
All experiments use a single model (Llama-3.1-8B-Instruct, layer 15) and a single source dataset (Roleplaying (Goldowsky-Dill et al., 2025)). Insider Trading Confirm results are uninformative, as baseline transfer there already reaches 0.874. Generalization to other model families or datasets is left to future work.

Sandbagging margins are small.
On Sandbagging, interpretability selection raises AUROC from 0.606 to 0.662, recovering only 25% of the gap to the greedy upper bound (0.832). Sandbagging is also the most out-of-distribution target, and the modest gain likely reflects this distribution shift.
The Insider Trading Report result (0.706 to 0.881, closing 78% of the gap to greedy at 0.929) is the more robust.

Linear pipeline. The probe pipeline is fully linear (PCA projection followed by logistic regression), so the existence of a transferable subspace is perhaps unsurprising. Extension to nonlinear probes or other readout locations remains open.

No causal validation.
Claims about what each PC “encodes” rest on top-activating examples and an LLM judge’s verbalization of the contrast; they are descriptive, not causal. High-scoring PCs have not been ablated during inference to verify that they carry the contrast attributed to them.

## 6 Conclusion

In probe-based deception monitoring, a small subset of source-domain principal components (PCs) supports cross-domain transfer nearly matching target-trained probes, and this subset is not recovered by source-side or target-side separability rankings.
An LLM judge’s interpretability score over each PC’s top-activating examples, computed without target data, partially recovers it.

Robust probing thus depends not only on whether a behavioral signal is linearly accessible but on where in activation space the probe reads it. Full-representation probes may mix general behavioral features with dataset-specific cues that raise in-distribution accuracy while hurting transfer, whereas sparse PC subspaces may isolate more general behavioral structure. OOD probing is accordingly a representation-selection problem: beyond fitting a classifier, one must choose a subspace that preserves the intended behavioral feature and excludes spurious correlation.

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

We use GPT-5-mini (OpenAI, 2025) to interpret directions.

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

Jaccard overlap between the PC sets of 2 methods (greedy and target-pca) is 0.182 for Insider Trading Report, 0.077 for Insider Trading Confirm, and 0.167 for Sandbagging, i.e. agreement on less than a fifth of selections.

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