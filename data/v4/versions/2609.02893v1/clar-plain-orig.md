# Probe Generalization as Subspace Selection for OOD Deception Detection

Daniel Yoo

  
Adrians Skapars

###### Abstract

Linear probes are classifiers that read information from language model activations, the model’s internal numerical signals. They can detect behaviors and concepts, but they may fail on out-of-distribution (OOD) examples: examples that differ from the training distribution. We study how well Llama-3.1-8B-Instruct probes generalize to 3 held-out deception detection datasets. We find that projecting inputs onto a small set of principal components (PCs) from the training distribution enables transfer across domains. PCs are mutually perpendicular directions that capture variation in the activations. The selected directions form a subspace: a restricted set of directions within the full representation. Probes trained in this subspace nearly match probes trained directly on the test distribution.

We also find that interpretations of PCs can identify some of these transferable directions. We use a large language model (LLM) judge to score each PC. The judge assesses whether the examples with the highest and lowest activations suggest a transferable deception direction. We then train probes on the highest-scoring PCs. This closes the baseline-to-oracle gap by 78% on Insider Trading Report and by 25% on Sandbagging. Here, the baseline uses full activations, while the oracle uses target data to select PCs.

The directions that a source probe weights heavily appear to encode surface features specific to the source dataset. In contrast, the directions that transfer appear to encode the same contrast more abstractly, in ways that natural-language descriptions can capture. Overall, our results suggest that subspace selection largely determines probes’ OOD robustness.

Machine Learning, ICML

## 1 Introduction

Linear probes are standard tools for reading information from language model activations (Alain and Bengio, 2018; Hewitt and Liang, 2019). They are cheap to train and reveal some meaningful information in an otherwise hard-to-interpret representation space. However, probes that perform well in-distribution—on data like their training data—can prove unreliable out of distribution (Kirch et al., 2026).

Existing work shows that model activations represent key behavioral concepts linearly, meaning that directions in activation space capture these concepts (Marks and Tegmark, 2024; Park et al., 2024; Zou et al., 2025). Probes should therefore have the capacity to detect them. Yet, in practice, probes appear to learn spurious correlations: patterns that predict labels in the training data but may not hold elsewhere (Orgad et al., 2025; Wang et al., 2025). This is unacceptable for probe-based monitoring of harmful interactions. It raises a clear challenge: how can we build probes that learn only features that generalize?

We address this challenge by working in the basis of source-domain PCs rather than raw activations. A basis is a set of directions used to express the activations. This approach avoids assuming that the model’s own hidden dimensions provide a privileged basis. PCs offer a simple, data-driven way to study structure in representation spaces. Prior work also suggests that the choice of PCs matters. Mu et al. (2018) show that removing the top PCs of an embedding space, a numerical representation space, improves downstream performance. Kantamneni et al. (2025) use projections onto top PCs as a probe baseline. Our question is therefore: which subset of PCs supports probe training that generalizes, and how can we identify it?

We study deception probes on Llama-3.1-8B-Instruct (Grattafiori et al., 2024). We train them on one source dataset, Roleplaying deception scenarios. We then evaluate them on 3 held-out target datasets: Insider Trading Report, Insider Trading Confirm, and Sandbagging (Goldowsky-Dill et al., 2025). The source is the training domain; the targets are the test domains. We use principal component analysis (PCA), which finds directions ordered by the variation they capture, to construct the source-domain basis. Within this basis, we test ways to select training dimensions that avoid spurious correlations and transfer to target domains.

This work contributes the following findings:

- **A small transferable subspace exists (Section 4.2).** A greedy search, which adds one PC at a time using target labels to guide selection, finds 8–15 source PCs. A source-trained probe on these components nearly matches a target-trained probe.

- **Source-side and target-side rankings both fail to recover the subspace (Section 4.3).** Ranking PCs by the source or target probe’s weights does not produce a basis that transfers better than the full-representation baseline.

- **An LLM judge partially recovers the subspace without target data (Section 4.5).** An LLM judge can interpret and select PCs that likely encode transferable behavioral contrasts rather than surface patterns.

Figure 1: Overview of PC-based subspace selection for OOD deception probing. We project activations onto the source-domain PCA basis, filter candidate PC directions, and train probes on the selected low-dimensional subspace.

## 2 Related Work

#### Linear concept directions in activation space.

Related work studies whether directions or low-dimensional subspaces in activation space represent high-level behavioral or semantic properties. Marks and Tegmark (2024) report linear structure in true/false representations across datasets. Park et al. (2024) formalize the linear representation hypothesis, which proposes that directions represent concepts, and connect it to probing and steering. Steering means changing activations to influence model behavior. Similarly, Zou et al. (2025) model high-level properties, such as honesty and emotion, as population-level directions. These directions allow researchers to both read properties and intervene on them in model activations.

Researchers have also used this perspective to explain and control model behavior. Arditi et al. (2024) show that a single direction can mediate refusal behavior. This suggests that simple linear features may capture some high-level behaviors. Lu et al. (2026) similarly identify an “assistant” axis that corresponds to assistant-like behavior. Together, these studies suggest that linear structure can support both an understanding of how models work and interventions on their activations. They also raise questions about when such directions capture stable concepts rather than context-dependent correlations.

#### Linear probes and their limitations.

Researchers have long used linear and lightweight probes to study what information neural networks represent (Alain and Bengio, 2018; Hewitt and Liang, 2019). Recent work applies these probes to safety-related signals in LLM activations. Examples include latent truthfulness, or internal truthfulness signals, without direct supervision (Burns et al., 2024), and strategic deception (Goldowsky-Dill et al., 2025).

At the same time, the probing literature emphasizes methodological caution. Hewitt and Liang (2019) argue that high probe accuracy alone does not establish that a model already represents the target property. The probe may learn the task itself or exploit surface-level correlations. Ravichander et al. (2021) similarly question whether probing establishes what information a model encodes or only what information a probe can recover. Recent work also finds that probes with strong in-distribution performance can generalize unevenly out of distribution (kramár2026buildingproductionreadyprobesgemini). These concerns motivate careful tests of whether safety probes identify robust internal signals rather than dataset-specific artifacts.

#### Interpreting features with natural-language explanations.

Another approach interprets internal representations by assigning natural-language descriptions to neurons or learned features. Bills et al. (2023) introduced the use of LLMs to explain neurons from examples with high activations. Researchers have since extended this approach to automatically label features in sparse autoencoders ((Paulo et al., 2025), (Bricken et al., 2023)). A sparse autoencoder (SAE) reconstructs activations using a small number of active learned features. Automatic labeling enables researchers to interpret large collections of these features.

Recent work also tests whether natural-language feature descriptions can guide interventions, not just interpretation. For example, Casademunt et al. (2025) identify concepts through automated natural-language interpretation. They then ablate, or remove, those concepts during fine-tuning to reduce misaligned behavior.

## 3 Setup

#### Datasets.

Following Goldowsky-Dill et al. (2025), we use 4 of their deception detection datasets: Roleplaying, Insider Trading Report, Insider Trading Confirm, and Sandbagging. Appendix A describes them.

Roleplaying serves as our source domain. The other 3 datasets serve as target test sets. GPT-4o (OpenAI, 2024) labels each response as honest (0), ambiguous (-1), or deceptive (1). We remove ambiguous responses from our experiments.

#### Language models.

Following Ying et al. (2026), we use layer 15 residual stream activations from Llama-3.1-8B-Instruct (Grattafiori et al., 2024). The residual stream is the internal state passed between model layers. Layer 15 had the highest cross-domain area under the receiver operating characteristic curve (AUROC) in their analysis. AUROC measures how well a classifier separates classes across decision thresholds.

To extract activations, we pass each prompt through the model. We then average token activations over only the assistant’s response. Tokens are the pieces of text that the model processes.

#### Probe training and evaluation.

We standardize source activations to have zero mean and unit variance. We use the same source statistics to standardize all target activations. We then run PCA on the scaled source activations and retain the top 100 PCs.

All probes are logistic regression classifiers, which use a weighted linear score to predict class probabilities. We set the regularization parameter, lambda, to 10. When an experiment uses a subset, denoted by \(S\), of the 100 PCs, we project activations onto \(S\) before training and evaluation. This projection keeps only the activations’ coordinates along the selected directions. Baseline probes instead train directly on the full standardized activations.

We use the top 100 PCs because their alignment with SAE decoder features decreases as the PC index rises (Bricken et al., 2023; Appendix C). Decoder features are the directions that the SAE uses to reconstruct activations; alignment measures how closely these directions match the PCs. Lower-index PCs are therefore easier to interpret in natural language. This makes them suitable for comparison with interpretation-based selection methods.

We do not claim that lower-index PCs generalize better. Section 4.1 confirms that the 100-PC subspace retains enough information to separate deceptive from honest responses on every target.

## 4 Experiments

### 4.1 Setting a baseline

Figure 2: Probe AUROC on target datasets for source-trained probes on full representations (blue) and top-100 PCs (orange), and target-trained probes on top-100 source PCs (green, 5-fold cross-validation). Error bars show 95% confidence intervals.

We first test how well probes trained on full source representations transfer to each target. Figure 2 shows the results. Transfer works reasonably well for Insider Trading Confirm, with an AUROC of 0.874. Performance is moderate for Insider Trading Report, at 0.706 AUROC, and poor for Sandbagging, at 0.606 AUROC.

We also test whether the 100-PC subspace itself limits performance. We train a probe directly on target data within this subspace using 5-fold cross-validation. Cross-validation divides the data into folds and evaluates on held-out folds. The probe achieves 0.936 / 0.950 / 0.869 AUROC across the three targets. These results confirm that the 100-PC subspace contains enough information to separate deceptive from honest responses on every target.

The problem is therefore not missing information. Rather, the source probe learns a direction in the subspace that does not generalize.

### 4.2 Showing a transferable subspace exists

Figure 3: Probe AUROC across the three targets for the full-representation baseline (blue), top-\(k\) PCs by variance (teal, with \(k\) matched to greedy selection), and greedy oracle PCs (purple). Error bars show 95% confidence intervals.

Can any method train probes on source data while closing the performance gap between the baseline and target-trained probes? We test this with a greedy algorithm for PC selection.

We start with an empty set and add source PCs one at a time. At each step, we search the top 100 PCs and add the one that most improves target AUROC. To evaluate a candidate, we project source activations onto the current subset and train a logistic regression probe on those projections.

We always fit probe weights on source data. However, performance on labeled target examples guides the choice of PC subsets. We use 5-fold cross-validation over the target dataset. For each held-out fold, we perform greedy selection on the other target folds. We then evaluate the selected source-trained probe on the held-out target fold.

We stop when no candidate PC improves AUROC by more than 0.001. The search converges on small subsets. On average, it selects 15 PCs for Insider Trading Report, 8 for Insider Trading Confirm, and 14 for Sandbagging.

The greedy probe provides an upper bound on transfer performance, not an unbiased OOD estimate, because selection uses some target examples. As a comparison, we also select the top \(k\) PCs by variance, where \(k\) equals the number selected by the greedy algorithm. Figure 3 shows the results.

The greedy method nearly matches a probe trained directly on target data for Insider Trading Report and Sandbagging. It improves over the baseline by 0.22 and 0.23 AUROC points, respectively. Thus, a transferable subspace exists in principle.

Selecting by variance matches greedy selection only for Insider Trading Confirm, the easy case. It falls far behind on Insider Trading Report and Sandbagging, the harder generalization tasks. These findings suggest that the choice of PCs matters. The variance those PCs explain does not predict their usefulness.

### 4.3 Testing hypotheses for transferable principal components

Figure 4: Probe AUROC across the three targets for three PC-selection rules: ranking by source-probe weights (red), ranking by target-probe weights (pink), and greedy oracle selection (purple). Error bars show 95% confidence intervals.

What makes these PCs transferable, and can we detect that property using only the source domain? We test 2 simple hypotheses about the greedy subset.

The first hypothesis says that transferable PCs are simply those that individually separate the target classes well across the test set. Under this account, greedy selection finds them because it has access to target data.

The second hypothesis says that transferable PCs are those on which the source probe already relies most heavily. Under this account, greedy selection merely recovers the source signal in a more compact form.

We compare 2 PC rankings based on standardized probe coefficients. For each PC, the standardized coefficient is the absolute value of its probe coefficient multiplied by that PC’s standard deviation. We obtain the coefficients from probes trained on activations projected onto the source-PCA basis.

The “source-PCA probe” ranking uses weights learned by a probe trained on source data. The “target-PCA probe” ranking uses weights learned by a probe trained on target data. Both ranking probes use activations expressed in the source-PCA basis.

For each ranking, we select the top \(k\) PCs and project source data onto them. We then train the evaluated probes on this projected source data in both cases. If either hypothesis were correct, the corresponding probe should recover the greedy method’s performance.

To set \(k\), we run greedy selection on the full target dataset rather than separately within each fold. This gives one stable PC set: \(k=13\) for Insider Trading Report, 7 for Insider Trading Confirm, and 14 for Sandbagging. These greedy results differ slightly from those in the previous section because we use the full dataset rather than cross-validation. Figure 4 shows the results.

Both hypotheses fail, but in different ways. The source-PCA probe transfers about as well as the full-representation baseline. The PCs on which the source probe relies are those that do not generalize.

The target-PCA probe performs well on Insider Trading Report. However, it performs worse than the greedy probe on Insider Trading Confirm and below random on Sandbagging. The PCs that best separate target classes are not those that transfer best when source data determines their weights. Appendix D provides further comparisons of the PC sets.

### 4.4 Interpreting the transferable subspace

So far, we have shown that a small subset of source PCs supports cross-domain transfer. This subset differs from the directions on which source or target classification relies. Does the transferable subset have a meaning that people can interpret? We test this using the interpretation methods of Bills et al. (2023) and Casademunt et al. (2025).

For each of the top 100 source PCs, we find the 8 source examples with the highest activations and the 8 with the lowest activations. We prompt an LLM to describe the positive pole, the negative pole, and the overall direction in natural language. The poles are the two opposing ends of the PC direction.

The LLM judge then rates each direction’s generalizability on a 1–10 scale, which we call the OOD score. Appendix B gives the full judge prompt and model details. No PC receives a score above 6, so we treat 6 as the maximum observed OOD score. We retain all PCs that receive this maximum in both runs, yielding 19 PCs.

Five PCs appear in greedy selections for more than one target. Of these, 3 out of 5 receive the maximum OOD score. This suggests that LLMs can recover some transferable directions. By comparison, only 1 of the top 5 PCs selected by the source-PCA probes receives the maximum OOD score.

Interpretations of the greedy-selected PCs describe contrasts at an abstract enough level to survive the move to other domains. Appendix E provides a full breakdown of the PCs selected by each method.

### 4.5 Selecting a subspace by interpreting PCs

Figure 5: Probe AUROC across the three targets for the interp-selected subspace (red; 19 PCs) versus three controls: full-representation baseline (blue), top-19 PCs by variance (teal), and the remaining 81 PCs after removing the interp-selected PCs (brown). Error bars show 95% confidence intervals.

Can interpretation-based OOD scores recover a meaningful share of the greedy probe’s performance without target data? To test this, we use the PCs that receive the judge model’s maximum OOD score as the probe subspace. We call this method “interp-selection.”

LLM judge scores are noisy, so we score each PC twice. We retain only PCs that receive the maximum score in both runs. This yields 19 interp-selected PCs.

We compare this probe with two controls. One trains on the remaining 81 PCs after we remove the selected PCs. The other trains on the top 19 source PCs by variance. Figure 5 shows the results.

The interp-selected probe outperforms the baseline on all 3 targets. It also outperforms the variance-selected control, with wide margins on Insider Trading Report and Sandbagging. As expected, it outperforms the interp-ablated probe—the probe using the remaining PCs—on all 3 targets.

The amount of recovery varies sharply by dataset. On Insider Trading Report, interp-selection closes 78% of the gap between the baseline (0.706) and the greedy oracle (0.929).

On Insider Trading Confirm, baseline transfer is already near the ceiling, leaving little room for comparison. The result is therefore uninformative: all selection methods except interp-ablation perform roughly alike.

On Sandbagging, interp-selection closes 25% of the gap to the greedy oracle (0.832).

Overall, selecting PCs solely from natural-language descriptions of what they encode meaningfully improves baseline performance on these generalization tasks. This selection uses no target data.

## 5 Limitations

Our strongest claims remain narrow. Readers should consider several limitations.

**Single model and source dataset.** All experiments use one model, Llama-3.1-8B-Instruct at layer 15, and one source dataset, Roleplaying (Goldowsky-Dill et al., 2025). Moreover, the Insider Trading Confirm results are uninformative because baseline transfer already reaches 0.874. Whether these results generalize to other model families or datasets remains an open question for future work.

**Sandbagging margins are small.** Sandbagging produces our weakest results. Interpretation-based selection raises AUROC from 0.606 to 0.662, recovering only 25% of the gap to the greedy upper bound (0.832). Sandbagging is also the most out-of-distribution target. The modest gain likely reflects this distribution shift, or difference between source and target data. The Insider Trading Report result is more robust: AUROC rises from 0.706 to 0.881, closing 78% of the gap to greedy selection at 0.929.

**Linear pipeline.** Our probe pipeline is fully linear: a PCA projection followed by logistic regression. The existence of a transferable subspace is therefore perhaps unsurprising. Whether these findings extend to nonlinear probes or to other locations from which we read model activations remains open.

**No causal validation.** We make structural claims about what each PC “encodes” based on highly activating examples and an LLM judge’s description of their contrast. These claims describe patterns; they do not establish causes. We have not removed the high-scoring PCs during inference to verify that they carry the contrast they appear to encode.

## 6 Conclusion

For probe-based deception monitoring, a small subset of source-domain PCs enables cross-domain transfer that nearly matches target-trained probes. Source-side or target-side class separation alone does not identify this subset. An LLM judge’s interpretation score, based on each PC’s top-activating examples and no target data, partially recovers the subspace.

A key implication is that robust probing requires more than checking whether a linear probe can access a behavioral signal. We must also identify where activation space represents that signal most robustly. Linear probes trained on full representations may combine general behavioral features with dataset-specific surface cues. These shortcuts can support high in-distribution accuracy while hurting transfer.

In contrast, subspaces containing only a few PCs may better isolate directions that represent more general behavioral structure. This reframes OOD probing as a representation-selection problem. The goal is not only to fit a classifier. It is also to choose a subspace that preserves the intended behavioral feature and filters out spurious correlation.

## Acknowledgment

We conducted this work during the 11th edition of AI Safety Camp, which provided the computing resources for our experiments. We thank Prasad Mahadik and Arya Hariharan for useful discussions about this work.

## Impact Statement

This paper presents work that aims to advance the field of Machine Learning. Our work has many potential societal consequences, but we do not feel that any require specific attention here.

## Appendix A Dataset Details

**Roleplaying:** The model enters a simple scenario with an opportunity to lie.

**Insider Trading Report:** Acting as a trader, the model decides whether to report honestly that it used insider information.

**Insider Trading Confirm:** After hiding earlier use of insider information, the model is asked to confirm that it did not use any.

**Sandbagging:** The model receives instructions to deliberately under-perform on some tasks while behaving normally on others.

| Dataset | Total Examples | Label Counts (-1 / 0 / 1) |
|---|---:|---|
| Roleplaying | 371 | 140 / 59 / 172 |
| Insider Trading Report | 473 | 218 / 194 / 61 |
| Insider Trading Confirm | 151 | 29 / 38 / 84 |
| Sandbagging | 1000 | 379 / 375 / 246 |

Table 1: Dataset label distributions.

## Appendix B LLM Judge Prompt

We use GPT-5-mini (OpenAI, 2025) to interpret directions.

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

Figure 6: Lower PCs show stronger decoder alignment, both for the best-matching SAE feature and for the mean of the top 10 features. The Spearman rank correlations are -0.857 and -0.917, respectively. We used andyrdt/saes-llama-3.1-8b-instruct (andyrdt, 2024).

## Appendix D PC Set Comparisons

Jaccard overlap measures the number of PCs shared by two sets divided by the number selected by either set. The overlap between greedy and target-pca selection is 0.182 for Insider Trading Report, 0.077 for Insider Trading Confirm, and 0.167 for Sandbagging. The 2 methods agree on less than a fifth of their selections.

Together, these results suggest that the transferable subspace is neither the part of representation space that best separates source classes nor the part that best separates target classes.

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

In the explanations below, PR means public relations and QA means question answering.

Table 3: Principal components selected for likely transferability, based on interpretation with an LLM judge.

| PC | Selected On | OOD Score | Full explanation |
|---|---|---:|---|
| PC62 | Insider Trading Report + Sandbagging | 6 | OOD deception usefulness (1-10): 6 — This direction maps onto a general deception-relevant behavior (deny/deflect vs provide details), so it can be useful across settings, but it is noisy: true speakers also deny or obfuscate for legitimate reasons and liars sometimes produce elaborate fabricated detail, and the signal may pick up simple lexical negation rather than semantic strategy. |
| PC57 | Insider Trading Report + Sandbagging | 6 | This direction aligns with a general deception-relevant strategy (deny/minimize or present innocuous explanations vs give specific/incriminating details), so it can help flag rhetorical moves often used when hiding or deflecting. However it is partly lexical and task-specific (relies on surface denials, hedges, and domain terms), so it will be brittle and context-dependent out of distribution. |
| PC9 | Insider Trading Report + Sandbagging | 3 | OOD deception usefulness (1-10): 3 — This direction mainly picks up domain‑specific lexical cues (medical/leave language vs project/status language) and stylistic framing rather than a generalizable deception strategy. It may flag genuine illness claims and is brittle across domains and rephrasings. |
| PC81 | Insider Trading Confirm + Sandbagging | 3 | OOD deception usefulness (1-10): 3 — The direction picks up a plausible deception‑relevant contrast (admit vs deny/hedge), but it mainly relies on surface/formal cues (negations, hedges, résumé style, enumerations) that are task- and wording‑dependent and likely brittle across domains and phrasings. |
| PC86 | Insider Trading Confirm + Sandbagging | 6 | OOD deception usefulness (1-10): 6 — Hedging and defensive PR-style wording is a general tactic associated with deceptive or self-protective behavior, so the direction could transfer across domains. However it is noisy (used legitimately for privacy, politeness, or negotiation), sensitive to phrasing and role, and will produce false positives, so it’s only moderately reliable out-of-distribution. |

Table 4: Principal components and their OOD scores with interpretations.

| PC | OOD Score | Full explanation |
|---|---:|---|
| PC7 | 3 | OOD deception usefulness (1-10): 3 — This direction mostly picks up a pragmatic contrast (deny/deflect vs admit) which can correlate with deception in some settings, but it relies heavily on surface cues (explicit “No”, word order, presence of “I” + admission) and task-specific phrasing. It is therefore brittle and likely to fail or mislead in different domains or with varied wording. |
| PC8 | 4 | OOD deception usefulness (1-10): 4 — The direction captures a useful narrow contrast (denial/excuse vs admission) that can correlate with evasive or deceptive responses in similar QA/roleplay tasks, but it largely picks up surface cues (explicit “No”/“Yes” and common denial phrasing) and is likely brittle under paraphrase, different domains, or subtler deception strategies. |
| PC11 | 3 | OOD deception usefulness (1-10): 3 — The direction mainly picks up a stylistic/pragmatic contrast (assertive first‑person role/defense vs problem-reporting/negation). That can correlate with evasive or face-saving lies in some scenarios, but it’s surface-y and context dependent and likely to fail across different domains or phrasings. |
| PC12 | 3 | OOD deception usefulness (1-10): 3 — This direction mainly captures discourse style/forcefulness (a surface cue). While overconfident or over-assertive phrasing can sometimes correlate with deceptive defensive behavior, it is brittle and domain-dependent and will produce many false positives/negatives when applied across different speakers, topics, or cultural styles. |
| PC4 | 6 | OOD deception usefulness (1-10): 6 — This captures a general communicative strategy relevant to deception (deny vs admit), so it can transfer beyond the specific prompts. However it also leans on surface cues (explicit “Yes”/“No” turns and common defensive phrasings) and may be brittle when denials or lies use different wording or more subtle tactics. |
