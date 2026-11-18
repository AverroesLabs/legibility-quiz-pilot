# Probe Generalization as Subspace Selection for OOD Deception Detection

Daniel Yoo

  
Adrians Skapars

###### Abstract

A linear probe is a classifier that reads a weighted combination of a language model’s internal activations, the numerical representations computed inside the model. Such probes can detect behaviors and concepts, but may fail on out-of-distribution (OOD) examples: examples from a different setting than the probe’s training data.

We study how deception probes for Llama-3.1-8B-Instruct generalize to 3 held-out datasets. We find that projecting activations onto a small subset of principal components (PCs) from the training distribution enables cross-domain transfer that nearly matches probes trained directly on the test distribution (Section 4.2, Figure 3). PCs are mutually perpendicular directions, ranked by how much variation they capture in the training activations. A selected subset defines a subspace: the part of the representation retained for probing.

We also find that PC interpretations can identify some of these transferable PCs. A large language model (LLM) judge scores each PC using its most- and least-activating examples. The judge assesses whether the contrast between those examples suggests a transferable deception direction. Probing on the highest-scoring PCs closes the baseline-to-oracle gap by 78% on Insider Trading Report and by 25% on Sandbagging (Section 4.5, Figure 5). Here, the baseline uses full representations, and the oracle uses target-guided PC selection.

Directions heavily weighted by a source-trained probe appear to encode source-specific surface features. Directions that transfer appear to encode the same contrast more abstractly, in a way natural-language descriptions can capture. Broadly, our results suggest that OOD probe robustness is largely determined by subspace selection.

Machine Learning, ICML (International Conference on Machine Learning)

## 1 Introduction

Linear probes are standard tools for reading information from language model activations (Alain and Bengio, 2018; Hewitt and Liang, 2019). They are cheap to train and expose some semantically meaningful information in an otherwise difficult-to-interpret representation space.

However, probes that perform well in-distribution—that is, on examples from the same setting as their training data—can be unreliable out of distribution (Kirch et al., 2026). Existing work shows that key behavioral concepts are represented linearly in model activations (Marks and Tegmark, 2024; Park et al., 2024; Zou et al., 2025). Probes should therefore have the capacity to detect these concepts. Yet, in practice, they appear to learn spurious correlations: cues associated with the labels in one dataset that do not reliably transfer elsewhere (Orgad et al., 2025; Wang et al., 2025). This is unacceptable for probe-based monitoring of harmful interactions. It poses a clear challenge: how can we build probes that learn only generalizing features?

We study this question using source-domain PCs rather than raw activation coordinates. The source domain is the setting used to train the probe. This choice avoids assuming that the model’s original activation coordinates form a privileged basis, or set of directions for expressing the representations. Principal component analysis (PCA) provides a simple, data-driven basis for studying representation structure.

Prior work suggests that the choice of PCs matters. Mu et al. (2018) show that removing the top PCs of an embedding space, a vector representation space, improves downstream performance. Kantamneni et al. (2025) use projections onto top PCs as a probe baseline. Our question is therefore: which subset of PCs supports generalizable probe training, and how can that subset be identified?

We use deception probes on Llama-3.1-8B-Instruct (Grattafiori et al., 2024) as our testbed. We train them on one source dataset, Roleplaying deception scenarios. We evaluate them on 3 held-out target datasets: Insider Trading Report, Insider Trading Confirm, and Sandbagging (Goldowsky-Dill et al., 2025). A target domain is an evaluation setting different from the source. Working in the source-domain PCA basis, we test ways to select training dimensions that avoid spurious correlations and transfer to these targets.

This work contributes the following findings:

- **A small transferable subspace exists (Section 4.2).** A target-supervised greedy search over source PCs finds 8–15 components on which a source-trained probe nearly matches a target-trained probe. “Target-supervised” means that labeled target examples guide PC selection. “Greedy” means that each step adds the PC that gives the largest immediate improvement.

- **Source-side and target-side rankings both fail to recover the subspace (Section 4.3).** Ranking PCs by source- or target-probe weights does not produce a basis that transfers better than the full-representation baseline.

- **An LLM judge partially recovers the subspace without target data (Section 4.5).** The judge can interpret and select PCs likely to encode a transferable behavioral contrast rather than surface patterns.

Figure 1: Overview of PC-based subspace selection for OOD deception probing. Activations are projected onto the source-domain PCA basis, candidate PC directions are filtered, and probes are trained on the selected low-dimensional subspace.

## 2 Related Work

#### Linear concept directions in activation space.

Related work studies whether high-level behavioral or semantic properties are represented as directions or low-dimensional subspaces in activation space. Marks and Tegmark (2024) report linear structure in true/false representations across datasets. Park et al. (2024) formalize the linear representation hypothesis: the idea that concepts are represented by linear directions. They connect it to both probing and steering, where steering means changing activations to influence model behavior. Similarly, Zou et al. (2025) model properties such as honesty and emotion as population-level directions, estimated across examples, that can be read from and intervened on in model activations.

This perspective has also been used to explain and control model behavior. Arditi et al. (2024) show that a single direction can mediate refusal behavior. This suggests that some high-level behaviors may be localized in simple linear features. Lu et al. (2026) similarly identify an “assistant” axis corresponding to assistant-like behavior.

Together, these works suggest that linear structure can support understanding of how a model works and interventions on its activations. They also raise questions about when such directions capture stable concepts rather than context-dependent correlations.

#### Linear probes and their limitations.

Linear and lightweight probes have a long history in interpretability as tools for analyzing information represented in neural networks (Alain and Bengio, 2018; Hewitt and Liang, 2019). Recent work applies such probes to safety-relevant signals in LLM activations. These include latent truthfulness without direct supervision (Burns et al., 2024) and strategic deception (Goldowsky-Dill et al., 2025).

Methodological skepticism is also central to the probing literature. Hewitt and Liang (2019) argue that high probe accuracy alone does not establish that a model already represents the target property. The probe may itself learn the task or exploit surface-level correlates. Ravichander et al. (2021) similarly question whether probing establishes what information is encoded, or only what information can be recovered. Recent work also finds uneven OOD generalization among probes with strong in-distribution performance (kramár2026buildingproductionreadyprobesgemini). These concerns motivate careful tests of whether safety probes identify robust internal signals rather than dataset-specific artifacts.

#### Interpreting features with natural-language explanations.

Another approach assigns natural-language descriptions to neurons, the individual computational units in a network, or to learned features. Bills et al. (2023) introduced the use of LLMs to explain neurons from high-activation examples.

This approach has since been extended to automatically label features in sparse autoencoders (SAEs) ((Paulo et al., 2025), (Bricken et al., 2023)). An SAE reconstructs activations using a learned set of features, with only a small number active at once. Automatic labeling enables large-scale interpretation of these feature dictionaries, or collections of learned features.

Recent work also asks whether natural-language feature descriptions can guide interventions. For example, Casademunt et al. (2025) identify concepts through automated natural-language interpretation. They then ablate, or remove, those concepts during fine-tuning—further model training—to reduce misaligned behavior.

## 3 Setup

#### Datasets.

Following Goldowsky-Dill et al. (2025), we use 4 deception detection datasets: Roleplaying, Insider Trading Report, Insider Trading Confirm, and Sandbagging. Appendix A describes them and gives their label counts in Table 1.

Roleplaying is the source domain. The other 3 datasets are target test sets. GPT-4o (OpenAI, 2024) labels each response as honest (0), ambiguous (-1), or deceptive (1). We remove ambiguous responses before running the experiments.

#### Language models.

Following Ying et al. (2026), we use layer 15 residual stream activations from Llama-3.1-8B-Instruct (Grattafiori et al., 2024). The residual stream is the internal representation passed through and updated by the model’s layers. Layer 15 had the highest cross-domain AUROC in their analysis. AUROC means area under the receiver operating characteristic curve. It summarizes how well a classifier separates the two labels across decision thresholds.

To extract activations, we pass each prompt through the model. We average token activations over only the assistant’s response. A token is a unit of text processed by the model.

#### Probe training and evaluation.

We standardize each source activation coordinate to zero mean and unit variance: we subtract its source mean and divide by its source standard deviation, a measure of spread. We apply those same source statistics to target activations.

We then run PCA on the standardized source activations and retain the top 100 PCs. PCs are ordered by decreasing captured variance, so a lower PC index denotes a higher-variance direction. Projection onto selected PCs means retaining only the activation coordinates along those directions.

All probes are logistic regression classifiers. A logistic regression probe maps a linear score to a class probability. We use $\lambda=10$, where $\lambda$ is the regularization parameter controlling the penalty on fitted coefficients. For experiments using a subset $S$ of the 100 PCs, $S$ denotes the selected components. We project activations onto $S$ before both training and evaluation. Baseline probes instead train directly on the full standardized activations.

Where specified below, we use 5-fold cross-validation (CV). We divide target data into 5 parts, or folds. Each fold is held out for evaluation in turn. The other folds are used for target-dependent fitting or selection, as specified for each experiment. Figure error bars are 95% confidence intervals (CIs), which report statistical uncertainty.

We retain the top 100 PCs because alignment with SAE decoder features decreases with PC index (Appendix C, Figure 6). SAE decoder features are the learned directions used to reconstruct activations. Alignment describes how closely those directions match the PCs. Lower-index PCs are therefore more amenable to natural-language interpretation and comparable with interpretability-driven selection methods.

This is not a claim that lower-index PCs generalize better. Section 4.1 shows that the 100-PC subspace retains enough information to separate deceptive from honest responses on every target.

## 4 Experiments

### 4.1 Setting a baseline

Figure 2: Probe AUROC on target datasets for source-trained probes on full representations (blue) and top-100 PCs (orange), and target-trained probes on top-100 source PCs (green, 5-fold CV). Error bars: 95% CIs

We first evaluate how well probes trained on full source representations transfer to each target. Insider Trading Confirm transfers reasonably well, with 0.874 AUROC. Insider Trading Report performs moderately, with 0.706 AUROC. Sandbagging performs poorly, with 0.606 AUROC (Figure 2, blue).

We also train probes directly on target data within the 100-PC source subspace, using 5-fold cross-validation. This checks whether the subspace itself limits performance. These probes achieve 0.936 / 0.950 / 0.869 AUROC on Insider Trading Report / Insider Trading Confirm / Sandbagging, respectively (Figure 2, green).

These results confirm that the 100-PC subspace contains enough information to separate deceptive from honest responses on every target. Thus, the difficulty is not that the relevant information is missing. It is that the source-trained probe learns a direction in the subspace that does not generalize.

### 4.2 Showing a transferable subspace exists

Figure 3: Probe AUROC across the three targets for the full-representation baseline (blue), top-k PCs by variance (teal, k matched to greedy), and greedy oracle PCs (purple). Error bars: 95% CIs.

Can a probe trained on source data close the performance gap between the baseline and target-trained probes? We test this with a greedy algorithm for selecting PCs.

We start with an empty set. At each step, we consider adding each remaining PC among the top 100 source PCs. For each candidate, we project source activations onto the current subset plus that candidate and train a logistic regression probe on those projections. We add the PC that most improves target AUROC.

Probe weights are always fitted on source data. However, labeled target examples guide the choice of PC subset. We use 5-fold cross-validation over the target dataset. For each held-out target fold, we select PCs using the other target folds, then evaluate the selected source-trained probe on the held-out fold.

We stop when no candidate improves AUROC by more than 0.001. The search converges on small subsets: on average, 15 PCs for Insider Trading Report, 8 for Insider Trading Confirm, and 14 for Sandbagging.

The greedy probe provides an upper bound on transfer performance, not an unbiased OOD estimate, because subset selection uses target examples. We therefore call this target-guided selection an oracle. As a control, we select the top $k$ PCs by source variance, where $k$ is the number of PCs selected by the greedy algorithm. Figure 3 compares these methods.

The greedy method comes close to a probe trained directly on target data for Insider Trading Report and Sandbagging. It improves on the full-representation baseline by 0.22 and 0.23 points of AUROC, respectively (Figure 3, purple versus blue; target-trained comparison in Figure 2). Thus, a transferable subspace exists in principle.

The variance control matches greedy selection only for Insider Trading Confirm, the easy case. It falls far behind for Insider Trading Report and Sandbagging, the harder generalization tasks (Figure 3, teal versus purple). These findings suggest that PC choice matters, and that the variance explained by a PC does not predict its usefulness.

### 4.3 Testing hypotheses for transferable principal components

Figure 4: Probe AUROC across the three targets for three PC-selection rules: ranking by source-probe weights (red), by target-probe weights (pink), and greedy oracle selection (purple). Error bars: 95% CIs.

What makes these PCs transferable? Can that property be detected using source data alone? We test 2 hypotheses about the greedy subset.

The first hypothesis is that transferable PCs are simply those that separate target classes well across the test set. Greedy selection would find them because it has access to target data. The second is that they are the PCs the source probe already weights most heavily. Greedy selection would then be recovering the source signal in a more compact form.

We compare 2 PC rankings. Both use standardized coefficients from a probe fitted to activations projected onto the source-PCA basis. For each PC, its standardized coefficient is the absolute value of its probe coefficient multiplied by the standard deviation of that PC’s projected values.

The **source-PCA probe** ranks PCs by coefficients learned from source data. The **target-PCA probe** ranks PCs by coefficients learned from target data. Both use the source-PCA basis.

After either ranking is constructed, we train the evaluated probe on source data projected onto the top $k$ PCs from that ranking. Thus, target data determine the target-PCA ranking, but the final evaluated probe’s weights are still fitted on source data. If either hypothesis were correct, its ranking should recover greedy performance.

To set $k$, we run greedy selection on the full target dataset rather than separately within folds. This gives a single stable PC set: $k=13$ for Insider Trading Report, 7 for Insider Trading Confirm, and 14 for Sandbagging. Because this uses the full dataset rather than CV, the greedy results differ slightly from those in Section 4.2. Figure 4 shows the results.

Both hypotheses fail, but in different ways. The source-PCA probe transfers about as well as the full-representation baseline: the PCs relied on by the source probe are those that do not generalize (Figure 4, red; baseline in Figure 2).

The target-PCA probe does well on Insider Trading Report. However, it performs worse than greedy selection on Insider Trading Confirm and has below-random transfer performance on Sandbagging (Figure 4, pink versus purple). PCs that best separate target classes are not the PCs that transfer best when their weights are fitted on source data.

Appendix D provides further PC-set comparisons.

### 4.4 Interpreting the transferable subspace

We have shown that a small subset of source PCs supports cross-domain transfer. It differs from the directions relied on by either source or target classification. Does this transferable subset have a human-interpretable meaning?

We test this using the interpretability method of Bills et al. (2023) and Casademunt et al. (2025). For each of the top 100 source PCs, we find the 8 most-activating and 8 least-activating source examples. We prompt an LLM to describe the positive pole, the negative pole, and the direction as a whole. The poles are the two opposite ends of the PC direction.

The LLM judge then rates each direction’s generalizability on a 1–10 scale, called its OOD score. Appendix B gives the full prompt and model details.

No PC receives an OOD score above 6 on this 1–10 scale. We therefore treat 6 as the maximum observed score. We retain PCs that receive this maximum in both scoring runs, yielding 19 PCs.

There are 5 PCs selected by greedy search for more than one target. Of these, 3 receive the maximum OOD score: PC62, PC57, and PC86. PC9 and PC81 score 3 (Appendix E, Table 3). This suggests that LLMs can partially recover transferable directions.

For comparison, only 1 of the top 5 PCs selected by source-PCA probes receives the maximum OOD score. PC4 scores 6; PC7, PC8, PC11, and PC12 score 3, 4, 3, and 3, respectively (Appendix E, Table 4).

Interpretations of greedy-selected PCs describe the contrast at a level of abstraction that survives transfer to other domains. Appendix E provides a full breakdown of the PCs selected by each method.

### 4.5 Selecting a subspace by interpreting PCs

Figure 5: Probe AUROC across the three targets for the interp-selected subspace (red; 19 PCs) versus three controls: full-representation baseline (blue), top-19 PCs by variance (teal), and the remaining 81 PCs after ablating the interp-selected PCs (brown). Error bars: 95% CIs.

Can interpretation-based OOD scores recover a meaningful fraction of greedy performance without target data?

We select PCs that receive the judge’s maximum observed OOD score and use them as the probe subspace. Because LLM scores are noisy, we score twice and retain only PCs that receive the maximum in both runs. This produces 19 **interp-selected**, or interpretation-selected, PCs.

We compare this probe with two controls. One uses the top 19 source PCs by variance. The other uses the 81 unselected PCs, thereby ablating the selected PCs from the retained 100-PC representation. We call this second control **interp-ablated**. Figure 5 shows the results.

The interp-selected probe outperforms the full-representation baseline on all 3 targets. It also outperforms the variance-selected control, by wide margins on Insider Trading Report and Sandbagging. As expected, it outperforms the interp-ablated probe on all 3 targets (Figure 5, red versus blue, teal, and brown).

The amount recovered varies sharply by dataset:

- **Insider Trading Report:** Interp-selection closes 78% of the gap between the baseline, 0.706, and the greedy oracle, 0.929. The interp-selected AUROC is 0.881 (Figure 5; Section 5).
- **Insider Trading Confirm:** Baseline transfer is already near the ceiling, leaving little headroom for comparison. The result is therefore uninformative. All selection methods except interp-ablation perform roughly alike (Figure 5).
- **Sandbagging:** Interp-selection closes 25% of the gap to the greedy oracle, 0.832. AUROC rises from the baseline’s 0.606 to 0.662 (Figure 5; Section 5).

Overall, selecting PCs solely from natural-language descriptions of what they encode, without target data, meaningfully improves baseline performance on the generalization tasks.

## 5 Limitations

Our strongest claims remain narrow. Readers should consider several limitations.

**Single model and source dataset.**  
All experiments use one model, Llama-3.1-8B-Instruct at layer 15, and one source dataset, Roleplaying (Goldowsky-Dill et al., 2025). Insider Trading Confirm results are also uninformative because baseline transfer already reaches 0.874 AUROC (Figure 2). Whether these results generalize to other model families or datasets remains open for future work.

**Sandbagging margins are small.**  
The weakest results are for Sandbagging. Interpretation-based selection raises AUROC from 0.606 to 0.662, recovering only 25% of the gap to the greedy upper bound of 0.832 (Figure 5). Sandbagging is also the most out-of-distribution target. The modest gain likely reflects this distribution shift, or difference between source and target data.

The Insider Trading Report result is more robust: AUROC rises from 0.706 to 0.881, closing 78% of the gap to greedy at 0.929 (Figure 5).

**Linear pipeline.**  
Our probe pipeline is fully linear: PCA projection followed by logistic regression. The existence of a transferable subspace is therefore perhaps unsurprising. Whether these findings extend to nonlinear probes or other readout locations—places in the model from which activations are extracted—remains open.

**No causal validation.**  
We make structural claims about what each PC “encodes” based on top-activating examples and an LLM judge’s verbal description of the contrast. These claims are descriptive, not causal. We have not ablated high-scoring PCs during inference, when the model generates responses, to verify that they carry the contrast they appear to encode.

## 6 Conclusion

For probe-based deception monitoring, a small subset of source-domain PCs enables cross-domain transfer that nearly matches target-trained probes (Section 4.2). Source-side or target-side separability alone does not identify this subset (Section 4.3). An LLM judge’s interpretability scores, based on each PC’s top-activating examples and without target data, partially recover it (Sections 4.4–4.5).

A key implication is that robust probing requires more than determining whether a behavioral signal is linearly accessible. It also requires identifying where in activation space that signal is represented most robustly.

Linear probes trained on full representations may combine general behavioral features with dataset-specific surface cues. These heuristics can support high in-distribution accuracy while hurting transfer. In contrast, sparse PC subspaces—those retaining only a small number of PCs—may better isolate general behavioral structure.

This reframes OOD probing as a representation-selection problem. The goal is not only to fit a classifier. It is also to choose a subspace that preserves the intended behavioral feature and filters out spurious correlation.

## Acknowledgment

This work was conducted during the 11th edition of AI Safety Camp, which provided the compute to run our experiments. Thanks to Prasad Mahadik and Arya Hariharan for useful discussions around this work.

## Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none
which we feel must be specifically highlighted here.

## Appendix A Dataset Details

**Roleplaying:** The model is placed in a simple scenario with an opportunity to lie.

**Insider Trading Report:** Acting as a trader, the model decides whether to honestly report that it used insider information.

**Insider Trading Confirm:** After hiding earlier use of insider information, the model is asked to confirm that it did not use any.

**Sandbagging:** The model is told to deliberately under-perform on some tasks while behaving normally on others.

Table 1 gives total examples and label counts before ambiguous responses are removed. Labels are ambiguous (-1), honest (0), and deceptive (1), in that order.

| Dataset | Total Examples | Label Counts (-1 / 0 / 1) |
|---|---|---|
| Roleplaying | 371 | 140 / 59 / 172 |
| Insider Trading Report | 473 | 218 / 194 / 61 |
| Insider Trading Confirm | 151 | 29 / 38 / 84 |
| Sandbagging | 1000 | 379 / 375 / 246 |

Table 1: Dataset label distributions.

## Appendix B LLM Judge Prompt

We use GPT-5-mini (OpenAI, 2025) to interpret directions.

The prompt below uses “bipolar” to mean that a direction has two opposite poles. In `token<TAB>activation`, `<TAB>` denotes a tab separator. The activation is the signed value associated with the token; its sign identifies the pole and its magnitude indicates strength. `{positive_examples}` and `{negative_examples}` are placeholders for the selected high-positive and high-negative examples.

```text
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
```

## Appendix C Lower PCs Show Stronger Decoder Alignment

Here, “lower PCs” means lower-index PCs. Spearman $\rho$ is a rank-correlation measure: it measures association between the orderings of two quantities.

Figure 6: Lower PCs show stronger decoder alignment, both for the best-matching SAE feature and for the mean of the top 10 features (Spearman $\rho$ = -0.857 and -0.917, respectively). We used andyrdt/saes-llama-3.1-8b-instruct (andyrdt, 2024).

## Appendix D PC Set Comparisons

Jaccard overlap is the number of PCs shared by two sets divided by the number in their union. Between greedy and target-pca selections, it is 0.182 for Insider Trading Report, 0.077 for Insider Trading Confirm, and 0.167 for Sandbagging. The 2 methods agree on less than a fifth of their selections.

Together, these results suggest that the transferable subspace is neither the part of representation space most discriminative on the source nor the part most discriminative on the target. “Discriminative” here means useful for separating honest and deceptive labels.

Table 2 lists selected PCs. A label such as PC7 means the seventh component in the source-PCA ordering.

| Selection method | Dataset / target | Selected PCs |
|---|---|---|
| Source probe weights | Source | PC7, PC8, PC11, PC12, PC4, PC5, PC2, PC3, PC6, PC27, PC22, PC21, PC1, PC41, PC18 |
| Greedy selection | Insider Trading Report | PC6, PC89, PC62, PC66, PC26, PC14, PC57, PC25, PC93, PC9, PC70, PC97, PC78 |
| Greedy selection | Insider Trading Confirm | PC71, PC53, PC28, PC86, PC32, PC81, PC77 |
| Greedy selection | Sandbagging | PC37, PC62, PC54, PC41, PC46, PC9, PC57, PC44, PC51, PC43, PC19, PC48, PC86, PC81, PC33 |
| Target separability ranking | Insider Trading Report | PC5, PC6, PC89, PC18, PC79, PC46, PC87, PC4, PC14, PC34, PC29, PC1, PC66, PC67, PC19 |
| Target separability ranking | Insider Trading Confirm | PC4, PC1, PC8, PC53, PC30, PC7, PC13, PC18, PC5, PC20, PC26, PC49, PC28, PC31, PC97 |
| Target separability ranking | Sandbagging | PC51, PC91, PC97, PC13, PC39, PC77, PC4, PC66, PC53, PC86, PC62, PC23, PC41, PC10, PC61 |

Table 2: Comparison of PCs selected by source-only probe weights, greedy transfer selection, and target separability ranking.

## Appendix E PC Interpretability

The tables below reproduce the judge’s explanations. “Lexical” refers to particular words or word choices. “Hedging” means wording that qualifies or weakens a statement. PR means public relations; QA means question answering. False positives are honest responses flagged as deceptive; false negatives are deceptive responses not flagged as deceptive.

Table 3: Principal components selected for likely transferability, based on interpretation with an LLM judge.

| PC | Selected On | OOD Score | Full explanation |
|---|---|---|---|
| PC62 | Insider Trading Report + Sandbagging | 6 | OOD deception usefulness (1-10): 6 — This direction maps onto a general deception-relevant behavior (deny/deflect vs provide details), so it can be useful across settings, but it is noisy: true speakers also deny or obfuscate for legitimate reasons and liars sometimes produce elaborate fabricated detail, and the signal may pick up simple lexical negation rather than semantic strategy. |
| PC57 | Insider Trading Report + Sandbagging | 6 | This direction aligns with a general deception-relevant strategy (deny/minimize or present innocuous explanations vs give specific/incriminating details), so it can help flag rhetorical moves often used when hiding or deflecting. However it is partly lexical and task-specific (relies on surface denials, hedges, and domain terms), so it will be brittle and context-dependent out of distribution. |
| PC9 | Insider Trading Report + Sandbagging | 3 | OOD deception usefulness (1-10): 3 — This direction mainly picks up domain‑specific lexical cues (medical/leave language vs project/status language) and stylistic framing rather than a generalizable deception strategy. It may flag genuine illness claims and is brittle across domains and rephrasings. |
| PC81 | Insider Trading Confirm + Sandbagging | 3 | OOD deception usefulness (1-10): 3 — The direction picks up a plausible deception‑relevant contrast (admit vs deny/hedge), but it mainly relies on surface/formal cues (negations, hedges, résumé style, enumerations) that are task- and wording‑dependent and likely brittle across domains and phrasings. |
| PC86 | Insider Trading Confirm + Sandbagging | 6 | OOD deception usefulness (1-10): 6 — Hedging and defensive PR-style wording is a general tactic associated with deceptive or self-protective behavior, so the direction could transfer across domains. However it is noisy (used legitimately for privacy, politeness, or negotiation), sensitive to phrasing and role, and will produce false positives, so it’s only moderately reliable out-of-distribution. |

Table 4: Principal components and their OOD scores with interpretations.

| PC | OOD Score | Full explanation |
|---|---|---|
| PC7 | 3 | OOD deception usefulness (1-10): 3 — This direction mostly picks up a pragmatic contrast (deny/deflect vs admit) which can correlate with deception in some settings, but it relies heavily on surface cues (explicit “No”, word order, presence of “I” + admission) and task-specific phrasing. It is therefore brittle and likely to fail or mislead in different domains or with varied wording. |
| PC8 | 4 | OOD deception usefulness (1-10): 4 — The direction captures a useful narrow contrast (denial/excuse vs admission) that can correlate with evasive or deceptive responses in similar QA/roleplay tasks, but it largely picks up surface cues (explicit “No”/“Yes” and common denial phrasing) and is likely brittle under paraphrase, different domains, or subtler deception strategies. |
| PC11 | 3 | OOD deception usefulness (1-10): 3 — The direction mainly picks up a stylistic/pragmatic contrast (assertive first‑person role/defense vs problem-reporting/negation). That can correlate with evasive or face-saving lies in some scenarios, but it’s surface-y and context dependent and likely to fail across different domains or phrasings. |
| PC12 | 3 | OOD deception usefulness (1-10): 3 — This direction mainly captures discourse style/forcefulness (a surface cue). While overconfident or over-assertive phrasing can sometimes correlate with deceptive defensive behavior, it is brittle and domain-dependent and will produce many false positives/negatives when applied across different speakers, topics, or cultural styles. |
| PC4 | 6 | OOD deception usefulness (1-10): 6 — This captures a general communicative strategy relevant to deception (deny vs admit), so it can transfer beyond the specific prompts. However it also leans on surface cues (explicit “Yes”/“No” turns and common defensive phrasings) and may be brittle when denials or lies use different wording or more subtle tactics. |
