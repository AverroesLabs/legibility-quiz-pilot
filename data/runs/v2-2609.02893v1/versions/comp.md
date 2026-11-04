# Probe Generalization as Subspace Selection for OOD Deception Detection

Daniel Yoo, Adrians Skapars

###### Abstract

For Llama-3.1-8B-Instruct deception probes on 3 held-out datasets, projecting inputs onto a few training-distribution principal components (PCs) gives transfer nearly matching probes trained on the test distribution. An LLM judge scoring each PC on whether its extreme examples imply a transferable deception direction selects PCs that close the baseline-to-oracle gap by 78% (Insider Trading Report) and 25% (Sandbagging). Heavily weighted source directions encode surface features; transferring ones encode the contrast abstractly. OOD robustness is largely subspace selection.

## 1 Introduction

Probes strong in-distribution can be unreliable OOD, learning spurious correlations even though behavioral concepts are linearly represented. We work in the source-domain PCA basis (not assuming a privileged neuron basis) and ask which PC subset supports generalizable probes and how to identify it. Testbed: deception probes on Llama-3.1-8B-Instruct, trained on Roleplaying, evaluated on Insider Trading Report, Insider Trading Confirm, Sandbagging.

Contributions:
- **A small transferable subspace exists** (4.2): target-supervised greedy search finds 8–15 PCs on which a source-trained probe nearly matches a target-trained probe.
- **Source- and target-side rankings fail to recover it** (4.3): ranking PCs by source or target probe weights does not beat the full-representation baseline.
- **An LLM judge partially recovers it without target data** (4.5), selecting PCs likely encoding transferable behavioral contrast rather than surface patterns.

Figure 1: Activations projected onto source PCA basis, PCs filtered, probes trained on selected subspace.

## 2 Related Work

#### Linear concept directions in activation space.

Truth, honesty, emotion, refusal and an "assistant" axis appear as linear directions usable for reading and steering, raising the question of when such directions are stable concepts versus context-dependent correlations.

#### Linear probes and their limitations.

Probes have been applied to latent truthfulness and strategic deception. High probe accuracy may reflect the probe learning the task or surface correlates, or information merely recoverable rather than encoded; probes strong in-distribution can generalize unevenly OOD.

#### Interpreting features with natural-language explanations.

LLMs can explain neurons from high-activation examples, extended to automatic SAE feature labeling; natural-language concept identification has also been used to ablate concepts during fine-tuning to reduce misalignment.

## 3 Setup

#### Datasets.

4 deception datasets (Appendix A): Roleplaying (source), Insider Trading Report, Insider Trading Confirm, Sandbagging (targets). GPT-4o labels: honest (0), ambiguous (-1, removed), deceptive (1).

#### Language models.

Llama-3.1-8B-Instruct layer 15 residual stream (highest cross-domain AUROC in prior work), averaged over assistant-response tokens.

#### Probe training and evaluation.

Standardize with source statistics (applied to targets); PCA on source, keep top 100 PCs. Logistic regression ($\lambda=10$); subset-$S$ probes project onto $S$; baselines use full standardized activations. Top 100 chosen because SAE decoder alignment decreases with PC index (Appendix C), aiding interpretation; not a claim that low-index PCs generalize better.

## 4 Experiments

### 4.1 Setting a baseline

Figure 2: target AUROC for source-trained full-representation and top-100-PC probes, and target-trained top-100-PC probes (5-fold CV); 95% CIs.

Full-representation source probe: Insider Trading Confirm 0.874, Insider Trading Report 0.706, Sandbagging 0.606. Target-trained probes in the 100-PC subspace: 0.936 / 0.950 / 0.869. So the information is present; the source probe learns a non-generalizing direction.

### 4.2 Showing a transferable subspace exists

Figure 3: AUROC for baseline, top-k PCs by variance (k matched to greedy), greedy oracle PCs; 95% CIs.

Greedy: from empty set, add the PC (of top 100) that most improves target AUROC of a probe fit on source projections. Weights always fit on source; selection uses target labels on non-held-out folds (5-fold CV), evaluated on held-out fold; stop when no gain >0.001. Mean subset sizes: 15 (Report), 8 (Confirm), 14 (Sandbagging). This is an upper bound, not an unbiased OOD estimate.

Greedy nearly matches target-trained probes on Report and Sandbagging, +0.22 and +0.23 AUROC over baseline. Variance control matches greedy only on Confirm and falls far behind on the harder tasks: which PCs matter, and variance explained does not predict usefulness.

### 4.3 Testing hypotheses for transferable principal components

Figure 4: AUROC for PCs ranked by source-probe weights, target-probe weights, and greedy; 95% CIs.

Hypotheses: transferable PCs (i) individually separate target classes, or (ii) are those the source probe already relies on. Rank PCs by standardized coefficient (|coef| × PC std) of a probe on source-PCA-projected activations, trained on source ('source-PCA probe') or target ('target-PCA probe'); then train on source projected onto the top $k$. $k$ from greedy on the full target set (no CV): 13 (Report), 7 (Confirm), 14 (Sandbagging); greedy numbers thus differ slightly from 4.2.

Both fail. Source-PCA ≈ full baseline: PCs the source probe relies on don't generalize. Target-PCA does well on Report, worse than greedy on Confirm, below random on Sandbagging: target-separating PCs are not those that transfer when weighted on source. See Appendix D.

### 4.4 Interpreting the transferable subspace

For each of the 100 PCs, the 8 most and 8 least activating source examples are given to an LLM, which describes positive pole, negative pole and direction, and rates generalizability 1–10 (OOD score; Appendix B). No PC scored above 6, so 6 is treated as maximum; PCs scoring 6 in both runs: 19.

5 PCs appeared in greedy selections for >1 target; 3 of 5 got the maximum score, vs 1 of the top 5 source-PCA-probe PCs. Interpretations of greedy PCs describe the contrast at a domain-surviving level of abstraction (Appendix E).

### 4.5 Selecting a subspace by interpreting PCs

Figure 5: AUROC for interp-selected (19 PCs), baseline, top-19 by variance, and remaining 81 PCs (interp-ablated); 95% CIs.

Using the 19 interp-selected PCs (no target data), the probe beats baseline on all 3 targets, beats variance control by wide margins on Report and Sandbagging, and beats interp-ablated on all 3. Report: closes 78% of the gap from baseline (0.706) to greedy (0.929). Confirm: near ceiling, uninformative; all methods except interp-ablated similar. Sandbagging: closes 25% of the gap to greedy (0.832).

## 5 Limitations

- Single model (layer 15) and source dataset; Confirm uninformative (baseline 0.874); generality to other models/datasets open.
- Sandbagging gains small: 0.606→0.662, 25% of gap to 0.832; most OOD target. Report (0.706→0.881, 78% of gap to 0.929) is more robust.
- Fully linear pipeline, so a transferable subspace is perhaps unsurprising; nonlinear probes and other readout locations untested.
- No causal validation: PC "encodings" are descriptive; high-scoring PCs not ablated at inference.

## 6 Conclusion

Robust probing requires finding not just whether a signal is linearly accessible but where it is represented most robustly. Full-representation probes may mix general behavioral features with dataset-specific cues that aid in-distribution accuracy but hurt transfer; sparse PC subspaces may isolate more general structure. OOD probing is thus a representation-selection problem: choose a subspace preserving the intended feature and filtering spurious correlation.

## Acknowledgment

Conducted during the 11th AI Safety Camp, which provided compute. Thanks to Prasad Mahadik and Arya Hariharan for discussions.

## Impact Statement

This work aims to advance Machine Learning; we see no societal consequences requiring specific highlighting.

## Appendix A Dataset Details

- Roleplaying: simple scenario with opportunity to lie.
- Insider Trading Report: trader decides whether to honestly report using insider information.
- Insider Trading Confirm: after hiding it, asked to confirm no insider information used.
- Sandbagging: told to under-perform on some tasks, behave normally on others.

| Dataset | Total | -1 / 0 / 1 |
|---|---|---|
| Roleplaying | 371 | 140 / 59 / 172 |
| Insider Trading Report | 473 | 218 / 194 / 61 |
| Insider Trading Confirm | 151 | 29 / 38 / 84 |
| Sandbagging | 1000 | 379 / 375 / 246 |

Table 1: Label distributions.

## Appendix B LLM Judge Prompt

Judge: GPT-5-mini. Prompt content:
- Interpret a bipolar PC direction (positive/negative activations toward opposite poles; larger magnitude = stronger match); lines are token<TAB>activation, normalized to integers −10 to +10.
- Setup: Roleplay in a simple scenario with a strategic chance to lie; high-positive and high-negative examples inserted.
- Infer the dominant contrast; don't assume broad sentiment; prefer narrowest accurate description; don't overgeneralize.
- Rate OOD deception usefulness 1–10 with brief reason; low scores for surface cues, lexical habits, task-specific phrasing; noisy/brittle/context-dependent directions ~1–4; ≥5 only for general deception-relevant strategy.
- Output: Positive pole, Negative pole, Shared direction, OOD deception usefulness (1-10).

## Appendix C Lower PCs Show Stronger Decoder Alignment

Figure 6: Alignment with best-matching SAE feature and mean of top 10 decreases with PC index (Spearman $\rho$ = -0.857, -0.917); SAE: andyrdt/saes-llama-3.1-8b-instruct.

## Appendix D PC Set Comparisons

Greedy vs target-PCA Jaccard: 0.182 (Report), 0.077 (Confirm), 0.167 (Sandbagging). The transferable subspace is neither most source- nor most target-discriminative.

| Method | Target | PCs |
|---|---|---|
| Source probe weights | Source | 7,8,11,12,4,5,2,3,6,27,22,21,1,41,18 |
| Greedy | Report | 6,89,62,66,26,14,57,25,93,9,70,97,78 |
| Greedy | Confirm | 71,53,28,86,32,81,77 |
| Greedy | Sandbagging | 37,62,54,41,46,9,57,44,51,43,19,48,86,81,33 |
| Target separability | Report | 5,6,89,18,79,46,87,4,14,34,29,1,66,67,19 |
| Target separability | Confirm | 4,1,8,53,30,7,13,18,5,20,26,49,28,31,97 |
| Target separability | Sandbagging | 51,91,97,13,39,77,4,66,53,86,62,23,41,10,61 |

Table 2: Selected PCs by method.

## Appendix E PC Interpretability

Table 3: Greedy PCs selected for >1 target.

| PC | Selected On | Score | Explanation |
|---|---|---|---|
| PC62 | Report + Sandbagging | 6 | Deny/deflect vs provide details; general but noisy, may pick up lexical negation |
| PC57 | Report + Sandbagging | 6 | Deny/minimize/innocuous explanation vs incriminating details; partly lexical, brittle |
| PC9 | Report + Sandbagging | 3 | Domain lexical cues (medical/leave vs project/status) |
| PC81 | Confirm + Sandbagging | 3 | Admit vs deny/hedge via surface cues (negations, hedges, enumerations) |
| PC86 | Confirm + Sandbagging | 6 | Hedging/defensive PR wording; transferable but false positives |

Table 4: Top source-probe-weight PCs.

| PC | Score | Explanation |
|---|---|---|
| PC7 | 3 | Deny/deflect vs admit; explicit "No", "I" + admission |
| PC8 | 4 | Denial/excuse vs admission; "No"/"Yes", denial phrasing |
| PC11 | 3 | Assertive first-person defense vs problem-reporting/negation |
| PC12 | 3 | Discourse style/forcefulness |
| PC4 | 6 | Deny vs admit; general but leans on "Yes"/"No" |