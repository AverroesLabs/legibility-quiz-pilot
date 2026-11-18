# Probe Generalization as Subspace Selection for OOD Deception Detection

Daniel Yoo

  
Adrians Skapars

###### Abstract

Linear probes can detect behaviors and concepts inside language model activations, but these probes may fail to transfer to out-of-distribution (OOD) examples. We study the generalization performance of probes for Llama-3.1-8B-Instruct across 3 held-out deception detection datasets. We find that projecting activations onto a small subset of principal components (PCs) from the training distribution enables cross-domain transfer that nearly matches the performance of probes trained directly on the test distribution.

Furthermore, natural-language interpretations of PCs can identify a subset of these transferable PCs. We use a large language model (LLM) judge to score whether each PC’s most- and least-activating examples imply a transferable deception direction, and then train probes on the highest-scoring PCs. This selection closes the gap between the full-representation baseline and the target-supervised greedy-selection oracle by 78% on Insider Trading Report and by 25% on Sandbagging. The directions that a source-trained probe weights heavily appear to encode source-specific surface features. In contrast, the directions that transfer appear to encode the same contrast more abstractly, in a way that natural-language descriptions can capture. Broadly, these results suggest that the OOD robustness of probes is largely determined by subspace selection.

Machine Learning, ICML

## 1 Introduction

Linear probes are standard tooling for reading information from language model activations (Alain and Bengio, 2018; Hewitt and Liang, 2019). Linear probes are cheap to train and expose some semantically meaningful information in an otherwise uninterpretable representation space.

However, probes that look strong in-distribution are sometimes unreliable out-of-distribution (Kirch et al., 2026). Existing work shows that key behavioral concepts are linearly represented in model activations (Marks and Tegmark, 2024; Park et al., 2024; Zou et al., 2025). Therefore, probes should have the capacity to detect these concepts. Yet, in practice, probes appear to learn spurious correlations instead (Orgad et al., 2025; Wang et al., 2025). This unreliability is unacceptable for probe-based monitoring of harmful interactions, setting a clear challenge: how can we build probes that only learn generalizing features?

To tackle this challenge, we work in the basis of source-domain principal components rather than raw activations. Principal components are directions obtained through principal component analysis (PCA). Using this data-driven basis avoids assuming that the model’s latent dimensions provide a privileged basis.

Prior work suggests that the choice of PCs matters. Mu et al. (2018) show that removing the top PCs of an embedding space improves downstream performance, while Kantamneni et al. (2025) use top-PC projections as a probe baseline. Therefore, our investigation asks: which subset of principal components supports generalizable probe training, and how can that subset be identified?

We use deception probes on Llama-3.1-8B-Instruct (Grattafiori et al., 2024) as our testbed. We train the probes on a single source dataset, Roleplaying deception scenarios, and evaluate the probes on 3 held-out target datasets: Insider Trading Report, Insider Trading Confirm, and Sandbagging (Goldowsky-Dill et al., 2025). Working in the source-domain PCA basis, we explore strategies for selecting training dimensions that avoid spurious correlations and transfer to target domains.

This work contributes the following findings:

- **A small transferable subspace exists (Section 4.2).** A target-supervised greedy search over source PCs finds 8–15 components on which a source-trained probe nearly matches a target-trained probe.

- **Source-side and target-side rankings both fail to recover the subspace (Section 4.3).** Ranking PCs by the source or target probe’s weights does not produce a basis that transfers better than the full-representation baseline.

- **An LLM judge partially recovers the subspace without target data (Section 4.5).** An LLM judge can interpret and select PCs that are likely to encode transferable behavioral contrast rather than surface patterns.

**Figure 1:** Overview of PC-based subspace selection for OOD deception probing. Activations are projected onto the source-domain PCA basis, candidate PC directions are filtered, and probes are trained on the selected low-dimensional subspace.

## 2 Related Work

#### Linear concept directions in activation space.

A related line of work studies whether high-level behavioral or semantic properties are represented as directions or low-dimensional subspaces in activation space. Marks and Tegmark (2024) report linear structure in true/false representations across datasets, while Park et al. (2024) formalize the linear representation hypothesis and connect that hypothesis to both probing and steering. Similarly, Zou et al. (2025) model high-level properties, such as honesty and emotion, as population-level directions that can be read from and intervened on in model activations.

This perspective has also been used to explain and control model behavior. Arditi et al. (2024) show that a single direction can mediate refusal behavior, suggesting that some high-level behaviors may be localized in simple linear features. Lu et al. (2026) similarly identify an “assistant” axis corresponding to assistant-like behavior. Together, these works suggest that linear structure can support both mechanistic understanding and activation-space interventions. However, these works also raise questions about when such directions capture stable concepts rather than context-dependent correlations.

#### Linear probes and their limitations.

Linear and lightweight probes have a long history in interpretability as tools for analyzing what information is represented in neural networks (Alain and Bengio, 2018; Hewitt and Liang, 2019). Recent work has applied such probes to safety-relevant signals in LLM activations, including latent truthfulness without direct supervision (Burns et al., 2024) and strategic deception (Goldowsky-Dill et al., 2025).

At the same time, methodological skepticism is central to the probing literature. Hewitt and Liang (2019) argue that high probe accuracy alone is insufficient evidence that a model already represents the target property, because the probe may itself learn the task or exploit surface-level correlates. Similarly, Ravichander et al. (2021) question whether probing reliably establishes what information is encoded or merely recoverable. Recent work also finds that probes with strong in-distribution performance can generalize unevenly out of distribution (kramár2026buildingproductionreadyprobesgemini). These concerns motivate careful evaluation of whether safety probes identify robust internal signals rather than dataset-specific artifacts.

#### Interpreting features with natural-language explanations.

Another approach aims to interpret internal representations by assigning natural-language descriptions to neurons or learned features. Bills et al. (2023) introduced the use of LLMs to generate explanations of neurons from high-activation examples. This approach has since been extended to automatic labeling of features in sparse autoencoders (SAEs) ((Paulo et al., 2025), (Bricken et al., 2023)), enabling large-scale interpretation of learned feature dictionaries.

Recent work has further explored whether natural-language feature descriptions can be used not only for interpretation but also for intervention. For example, Casademunt et al. (2025) identify concepts through automated natural-language interpretation and ablate those concepts during fine-tuning to reduce misaligned behavior.

## 3 Setup

#### Datasets.

Following Goldowsky-Dill et al. (2025), we use 4 of their deception detection datasets:

- **Roleplaying:** the model is placed in a simple scenario with an opportunity to lie.
- **Insider Trading Report:** acting as a trader, the model decides whether to honestly report that it used insider information.
- **Insider Trading Confirm:** after hiding earlier use of insider information, the model is asked to confirm that it did not use any.
- **Sandbagging:** the model is told to deliberately under-perform on some tasks while behaving normally on others.

Appendix A provides these descriptions and the dataset label distributions. Roleplaying is the source domain; the other 3 datasets are target test sets. GPT-4o (OpenAI, 2024) labels each response as honest (0), ambiguous (-1), or deceptive (1). We remove ambiguous responses from all experiments.

#### Language models.

Following Ying et al. (2026), we use layer 15 residual stream activations of Llama-3.1-8B-Instruct (Grattafiori et al., 2024). Layer 15 had the highest cross-domain area under the receiver operating characteristic curve (AUROC) in their analysis. To extract activations, we pass each prompt through the model and average token activations over only the assistant’s response.

#### Probe training and evaluation.

We first standardize source activations to zero mean and unit variance. We then use the same source statistics to standardize all target activations. Next, we run PCA on the standardized source activations and retain the top 100 PCs.

All probes are logistic regression classifiers with parameter $\lambda=10$. For experiments using a subset $S$ of the retained 100 PCs, where $S$ denotes the selected set of components, we project activations onto $S$ before both training and evaluation. In contrast, baseline probes train directly on the full standardized activations rather than on a PC subset.

We use the top 100 PCs because alignment with SAE decoder features (Bricken et al., 2023) decreases with PC index, as shown in Appendix C. Lower-index PCs are therefore more amenable to natural-language interpretation, making these PCs suitable for comparison with interpretability-driven selection methods. This choice is not a claim that low-index PCs generalize better. Instead, Section 4.1 checks that the 100-PC subspace retains enough information to separate deceptive from honest responses on every target.

## 4 Experiments

### 4.1 Setting a baseline

**Figure 2:** Probe AUROC on target datasets for source-trained probes on full representations (blue) and top-100 PCs (orange), and target-trained probes on top-100 source PCs (green, using 5-fold cross-validation, or CV). Error bars: 95% confidence intervals (CIs).

We first test how well probes trained on full source representations transfer to each target. As shown in Figure 2, Insider Trading Confirm transfers reasonably well, with 0.874 AUROC; Insider Trading Report performs moderately, with 0.706 AUROC; and Sandbagging performs poorly, with 0.606 AUROC.

To check whether the 100-PC subspace itself is the bottleneck, we also train a probe directly on target data within that subspace, using 5-fold cross-validation. The target-trained probes achieve 0.936 / 0.950 / 0.869 AUROC across the three targets. These results confirm that the 100-PC subspace encodes enough information to separate deceptive from honest responses on every target.

Thus, the transfer difficulty does not stem from the relevant information being missing from the subspace. Rather, the source-trained probe learns a direction in the subspace that does not generalize.

### 4.2 Showing a transferable subspace exists

**Figure 3:** Probe AUROC across the three targets for the full-representation baseline (blue), top-$k$ PCs by variance (teal, where $k$ is the number of PCs selected by the greedy method), and greedy oracle PCs (purple). Error bars: 95% CIs.

Is there a method for training probes on source data that closes the performance gap between the baseline and target-trained probes? To test this possibility, we introduce a greedy algorithm for PC selection.

The greedy algorithm starts from an empty set and iteratively adds PCs from the source dataset. At each step, the algorithm searches over the top 100 PCs and adds the PC that most improves AUROC on the target. To evaluate each candidate addition, we project source activations onto the current subset with that candidate included, train a logistic regression probe on the source projections, and evaluate the probe on target examples.

Therefore, probe weights are always fit on source data, while labeled examples from the target dataset guide the choice of PC subset. Using 5-fold cross-validation over the target dataset, we perform greedy PC selection on the non-held-out target folds and evaluate the selected source-trained probe on the held-out target fold. Selection stops when no candidate PC improves AUROC by more than 0.001.

The search converges on small subsets: on average, 15 PCs for Insider Trading Report, 8 for Insider Trading Confirm, and 14 for Sandbagging. Because subset selection uses some target examples, the greedy probe provides an upper bound on transfer performance, not an unbiased OOD estimate. We therefore refer to this target-supervised selection method as the greedy oracle.

To ground these results, we compare greedy selection against simply selecting the top $k$ PCs by variance, with $k$ matched to the number of PCs selected by the greedy algorithm. Figure 3 shows the comparison.

The greedy method comes close to a probe trained directly on the target dataset for Insider Trading Report and Sandbagging, improving on the full-representation baseline by 0.22 and 0.23 points of AUROC, respectively. Thus, a transferable subspace exists in principle.

However, the variance-based control matches greedy selection only for Insider Trading Confirm, the easy case. The variance-based control falls far behind greedy selection for Insider Trading Report and Sandbagging, the harder generalization tasks. These comparisons suggest that the choice of PCs matters and that the variance explained by those PCs does not predict their usefulness.

### 4.3 Testing hypotheses for transferable principal components

**Figure 4:** Probe AUROC across the three targets for three PC-selection rules: ranking by source-probe weights (red), ranking by target-probe weights (pink), and greedy oracle selection (purple). Error bars: 95% CIs.

What property makes the greedy-selected PCs transferable, and can that property be detected from the source side alone? We investigate 2 simple hypotheses about what the greedy subset contains.

The first hypothesis is that transferable PCs are simply the PCs that separate the target classes well by themselves across the test set. Under this hypothesis, greedy selection finds those PCs because greedy selection has access to target data.

The second hypothesis is that transferable PCs are the PCs on which the source probe already relies most heavily. Under this hypothesis, greedy selection merely rediscovers the source signal in a more compact form.

To test these hypotheses, we compare 2 PC rankings. Both rankings use standardized coefficients from a probe trained on activations projected onto the source-PCA basis. A PC’s standardized coefficient is the absolute value of the probe coefficient for that PC multiplied by the standard deviation of that PC.

- The **source-PCA probe** ranking uses weights learned by a probe trained on source data.
- The **target-PCA probe** ranking uses weights learned by a probe trained on target data.

In both ranking procedures, activations are projected onto the source-PCA basis. After obtaining each ranking, we train the probe used for transfer evaluation on source data projected onto the top $k$ PCs from that ranking. Thus, target data determine the target-PCA ranking, but the resulting transfer probe’s weights are still trained on source data. If either hypothesis were correct, the corresponding selected-subspace probe should recover the greedy method’s performance.

To set $k$, we run the greedy method on the full target dataset rather than separately within each fold. This produces a single stable PC set, with $k=13$ for Insider Trading Report, 7 for Insider Trading Confirm, and 14 for Sandbagging. Because this comparison uses the full target dataset rather than cross-validation, the greedy numbers differ slightly from the cross-validation numbers in Section 4.2. Figure 4 shows the results.

Both hypotheses fail, but in different ways. The source-PCA probe transfers about as well as the full-representation baseline. Thus, the PCs on which the source probe relies are those that do not generalize.

In contrast, the target-PCA probe does well on Insider Trading Report, but performs worse than the greedy probe on Insider Trading Confirm and has below-random transfer performance on Sandbagging. Therefore, the PCs that best separate target classes are not the PCs that transfer best when their probe weights are learned using source data. Appendix D provides further comparisons of the selected PC sets.

### 4.4 Interpreting the transferable subspace

So far, we have shown that a small subset of source PCs supports cross-domain transfer, and that this subset differs from the directions on which source classification or target classification relies. We next ask whether the transferable subset corresponds to anything human-interpretable. To investigate this question, we adopt the interpretability method of Bills et al. (2023) and Casademunt et al. (2025).

For each of the top 100 source PCs, we find the 8 most-activating and 8 least-activating examples from the source data. We then prompt an LLM to generate a natural-language description of the positive pole, the negative pole, and the PC direction as a whole.

The LLM judge rates the generalizability of each direction on a 1–10 scale. We call this rating the **OOD score**. Appendix B provides the full judge prompt and model details.

No PC received an OOD score above 6 on the judge’s 1–10 scale, so we treat 6 as the maximum observed OOD score. We run the scoring twice and retain all PCs that achieve this maximum score in both runs, yielding 19 PCs.

There were 5 PCs selected by the greedy method for more than one target: PC62, PC57, PC9, PC81, and PC86. Here, a code such as PC62 denotes principal component 62 in the source-PCA basis. Of these 5 shared greedy-selected PCs, 3—PC62, PC57, and PC86—received the maximum OOD score of 6 (Appendix E, Table 3). This agreement suggests that LLMs can partially recover transferable directions. For comparison, only 1 of the top 5 PCs selected by the source-PCA probes, PC4, received the maximum OOD score (Appendix E, Table 4).

Interpretations of the greedy-selected PCs describe the contrast at a level of abstraction that survives the move to other domains. Appendix E provides a full breakdown of PCs selected by each method.

### 4.5 Selecting a subspace by interpreting PCs

**Figure 5:** Probe AUROC across the three targets for the interpretation-selected, or interp-selected, subspace (red; 19 PCs) versus three controls: full-representation baseline (blue), top-19 PCs by variance (teal), and the remaining 81 PCs after ablating the interp-selected PCs (brown). Error bars: 95% CIs.

Does selecting PCs based on interpretability-based OOD scores recover a meaningful fraction of greedy probe performance without using target data? To test this possibility, we use the PCs given the maximum observed OOD score by the judge model as the probe subspace.

Because LLM judge scores are noisy, we score each PC twice and retain only PCs that receive the maximum score in both runs. As described in Section 4.4, this procedure produces 19 interp-selected PCs.

We compare this subspace against the full-representation baseline and two additional controls. The first additional control trains on the ablation of the selected PCs: the 81 retained source PCs not included in the interp-selected set. We call this the **interp-ablated** subspace. The second additional control trains on the top 19 source PCs by variance. Figure 5 shows the results.

The interp-selected PC probe outperforms the baseline on all 3 targets. The interp-selected probe also outperforms the source-variance-selected control, with wide margins on Insider Trading Report and Sandbagging. As expected, the interp-selected probe also outperforms the interp-ablated probe across all 3 targets.

However, the amount of performance recovered varies sharply by dataset:

- **Insider Trading Report:** interp-selection raises AUROC from the baseline’s 0.706 to 0.881, closing 78% of the gap to the greedy oracle at 0.929.
- **Insider Trading Confirm:** baseline transfer performance is already near the ceiling, so there is little headroom for comparison. Consequently, this result is uninformative: each selection method aside from interp-ablation performs roughly as well as the others.
- **Sandbagging:** interp-selection raises AUROC from the baseline’s 0.606 to 0.662, closing 25% of the gap to the greedy oracle at 0.832.

Overall, selecting PCs purely from natural-language descriptions of what those PCs encode, with no use of target data, provides a meaningful improvement over baseline performance on the generalization tasks.

## 5 Limitations

Our strongest claims remain quite narrow, and readers should consider several limitations.

**Single model and source dataset.** All experiments use a single model, Llama-3.1-8B-Instruct at layer 15, and a single source dataset, Roleplaying (Goldowsky-Dill et al., 2025). Moreover, the Insider Trading Confirm results are uninformative because baseline transfer on that dataset already reaches 0.874 AUROC. Whether these results generalize to other model families or other datasets remains an open question for future work.

**Sandbagging margins are small.** The weakest results are for Sandbagging. Interpretability selection raises AUROC from 0.606 to 0.662, recovering only 25% of the gap to the greedy upper bound of 0.832. Sandbagging is also the most out-of-distribution target, and the modest gain likely reflects this distribution shift. In contrast, the Insider Trading Report result is more robust: interpretability selection raises AUROC from 0.706 to 0.881, closing 78% of the gap to greedy selection at 0.929.

**Linear pipeline.** Our probe pipeline is fully linear: PCA projection is followed by logistic regression. Therefore, the existence of a transferable subspace is perhaps unsurprising. Whether these findings extend to nonlinear probes or other readout locations remains an open question.

**No causal validation.** We make structural claims about what each PC “encodes” based on top-activating examples and an LLM judge’s verbalization of the contrast. These claims are descriptive, not causal. We have not ablated the high-scoring PCs during inference to verify that those PCs carry the contrast that they appear to encode.

## 6 Conclusion

In probe-based deception monitoring, a small subset of source-domain principal components enables cross-domain transfer that nearly matches target-trained probes. However, this subset cannot be found through source-side or target-side separability alone. An interpretability score over each PC’s top-activating examples, produced by an LLM judge with no access to target data, partially recovers the transferable subspace.

A key implication is that robust probing requires identifying not only whether a behavioral signal is linearly accessible, but also where in activation space that signal is represented most robustly. Linear probes trained on full representations may combine general behavioral features with dataset-specific surface cues. These heuristics can support high in-distribution accuracy while hurting transfer.

In contrast, sparse subspaces of PCs may better isolate directions corresponding to more general behavioral structure. This contrast reframes OOD probing as a representation-selection problem: the goal is not only to fit a classifier, but also to choose a subspace that preserves the intended behavioral feature and filters out spurious correlation.

## Acknowledgment

This work was conducted during the 11th edition of AI Safety Camp, which provided the compute to run our experiments. Thanks to Prasad Mahadik and Arya Hariharan for useful discussions around this work.

## Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

## Appendix A Dataset Details

**Roleplaying:** the model is placed in a simple scenario with an opportunity to lie.

**Insider Trading Report:** acting as a trader, the model decides whether to honestly report that it used insider information.

**Insider Trading Confirm:** after hiding earlier use of insider information, the model is asked to confirm it did not use any.

**Sandbagging:** the model is told to deliberately under-perform on some tasks while behaving normally on others.

Table 1 reports total examples and label counts before ambiguous responses are removed. The label-count order is ambiguous (-1), honest (0), and deceptive (1).

| Dataset | Total Examples | Label Counts (-1 / 0 / 1) |
|---|---|---|
| Roleplaying | 371 | 140 / 59 / 172 |
| Insider Trading Report | 473 | 218 / 194 / 61 |
| Insider Trading Confirm | 151 | 29 / 38 / 84 |
| Sandbagging | 1000 | 379 / 375 / 246 |

**Table 1:** Dataset label distributions.

## Appendix B LLM Judge Prompt

We use GPT-5-mini (OpenAI, 2025) to interpret directions. The judge prompt is reproduced below.

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

**Figure 6:** Lower PCs show stronger decoder alignment, both for the best-matching SAE feature and for the mean of the top 10 features. The Spearman rank correlation coefficient, $\rho$, between PC index and alignment is -0.857 for the best-matching feature and -0.917 for the mean of the top 10 features. We used andyrdt/saes-llama-3.1-8b-instruct (andyrdt, 2024).

## Appendix D PC Set Comparisons

We compare the PC sets selected by greedy selection and target-PCA ranking using Jaccard overlap: the number of PCs shared by the two sets divided by the number of PCs in their union. Jaccard overlap is 0.182 for Insider Trading Report, 0.077 for Insider Trading Confirm, and 0.167 for Sandbagging. Thus, the 2 methods agree on less than a fifth of their selections.

Together, these results suggest that the transferable subspace is neither the part of representation space most discriminative on the source nor the part most discriminative on the target.

| Selection method | Dataset / target | Selected PCs |
|---|---|---|
| Source probe weights | Source | PC7, PC8, PC11, PC12, PC4, PC5, PC2, PC3, PC6, PC27, PC22, PC21, PC1, PC41, PC18 |
| Greedy selection | Insider Trading Report | PC6, PC89, PC62, PC66, PC26, PC14, PC57, PC25, PC93, PC9, PC70, PC97, PC78 |
| Greedy selection | Insider Trading Confirm | PC71, PC53, PC28, PC86, PC32, PC81, PC77 |
| Greedy selection | Sandbagging | PC37, PC62, PC54, PC41, PC46, PC9, PC57, PC44, PC51, PC43, PC19, PC48, PC86, PC81, PC33 |
| Target separability ranking | Insider Trading Report | PC5, PC6, PC89, PC18, PC79, PC46, PC87, PC4, PC14, PC34, PC29, PC1, PC66, PC67, PC19 |
| Target separability ranking | Insider Trading Confirm | PC4, PC1, PC8, PC53, PC30, PC7, PC13, PC18, PC5, PC20, PC26, PC49, PC28, PC31, PC97 |
| Target separability ranking | Sandbagging | PC51, PC91, PC97, PC13, PC39, PC77, PC4, PC66, PC53, PC86, PC62, PC23, PC41, PC10, PC61 |

**Table 2:** Comparison of PCs selected by source-only probe weights, greedy transfer selection, and target separability ranking.

## Appendix E PC Interpretability

Table 3 gives the judge’s scores and explanations for the 5 PCs selected by greedy selection for more than one target. Table 4 gives the scores and explanations for the top 5 PCs selected by source-probe weights.

**Table 3:** Principal components selected for likely transferability, based on interpretation with an LLM judge.

| PC | Selected On | OOD Score | Full explanation |
|---|---|---|---|
| PC62 | Insider Trading Report + Sandbagging | 6 | OOD deception usefulness (1-10): 6 — This direction maps onto a general deception-relevant behavior (deny/deflect vs provide details), so it can be useful across settings, but it is noisy: true speakers also deny or obfuscate for legitimate reasons and liars sometimes produce elaborate fabricated detail, and the signal may pick up simple lexical negation rather than semantic strategy. |
| PC57 | Insider Trading Report + Sandbagging | 6 | This direction aligns with a general deception-relevant strategy (deny/minimize or present innocuous explanations vs give specific/incriminating details), so it can help flag rhetorical moves often used when hiding or deflecting. However it is partly lexical and task-specific (relies on surface denials, hedges, and domain terms), so it will be brittle and context-dependent out of distribution. |
| PC9 | Insider Trading Report + Sandbagging | 3 | OOD deception usefulness (1-10): 3 — This direction mainly picks up domain‑specific lexical cues (medical/leave language vs project/status language) and stylistic framing rather than a generalizable deception strategy. It may flag genuine illness claims and is brittle across domains and rephrasings. |
| PC81 | Insider Trading Confirm + Sandbagging | 3 | OOD deception usefulness (1-10): 3 — The direction picks up a plausible deception‑relevant contrast (admit vs deny/hedge), but it mainly relies on surface/formal cues (negations, hedges, résumé style, enumerations) that are task- and wording‑dependent and likely brittle across domains and phrasings. |
| PC86 | Insider Trading Confirm + Sandbagging | 6 | OOD deception usefulness (1-10): 6 — Hedging and defensive PR-style wording is a general tactic associated with deceptive or self-protective behavior, so the direction could transfer across domains. However it is noisy (used legitimately for privacy, politeness, or negotiation), sensitive to phrasing and role, and will produce false positives, so it’s only moderately reliable out-of-distribution. |

**Table 4:** Principal components and their OOD scores with interpretations.

| PC | OOD Score | Full explanation |
|---|---|---|
| PC7 | 3 | OOD deception usefulness (1-10): 3 — This direction mostly picks up a pragmatic contrast (deny/deflect vs admit) which can correlate with deception in some settings, but it relies heavily on surface cues (explicit “No”, word order, presence of “I” + admission) and task-specific phrasing. It is therefore brittle and likely to fail or mislead in different domains or with varied wording. |
| PC8 | 4 | OOD deception usefulness (1-10): 4 — The direction captures a useful narrow contrast (denial/excuse vs admission) that can correlate with evasive or deceptive responses in similar QA/roleplay tasks, but it largely picks up surface cues (explicit “No”/“Yes” and common denial phrasing) and is likely brittle under paraphrase, different domains, or subtler deception strategies. |
| PC11 | 3 | OOD deception usefulness (1-10): 3 — The direction mainly picks up a stylistic/pragmatic contrast (assertive first‑person role/defense vs problem-reporting/negation). That can correlate with evasive or face-saving lies in some scenarios, but it’s surface-y and context dependent and likely to fail across different domains or phrasings. |
| PC12 | 3 | OOD deception usefulness (1-10): 3 — This direction mainly captures discourse style/forcefulness (a surface cue). While overconfident or over-assertive phrasing can sometimes correlate with deceptive defensive behavior, it is brittle and domain-dependent and will produce many false positives/negatives when applied across different speakers, topics, or cultural styles. |
| PC4 | 6 | OOD deception usefulness (1-10): 6 — This captures a general communicative strategy relevant to deception (deny vs admit), so it can transfer beyond the specific prompts. However it also leans on surface cues (explicit “Yes”/“No” turns and common defensive phrasings) and may be brittle when denials or lies use different wording or more subtle tactics. |
