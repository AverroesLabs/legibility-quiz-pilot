# Probe Generalization as Subspace Selection for OOD Deception Detection

Daniel Yoo  
Adrians Skapars

###### Abstract

Directions heavily weighted by a source probe appear to encode source-specific surface features; transferable directions appear to encode the same contrast more abstractly, capturable in natural language. Probe OOD robustness appears largely determined by subspace selection.

Machine Learning, ICML

## 1 Introduction

Linear probes cheaply expose semantically meaningful information in otherwise uninterpretable language-model representations (Alain and Bengio, 2018; Hewitt and Liang, 2019), detecting behaviors and concepts but sometimes failing out of distribution despite strong in-distribution performance (Kirch et al., 2026). Behavioral concepts have linear representations (Marks and Tegmark, 2024; Park et al., 2024; Zou et al., 2025); probes should have detection capacity yet appear to learn spurious correlations in practice (Orgad et al., 2025; Wang et al., 2025). Such unreliability is unacceptable for harmful-interaction monitoring.

Working in source Q7 coordinates avoids privileging the model’s latent dimensions. These coordinates provide data-driven representation structure; removing leading components improves downstream embedding performance (Mu et al., 2018), while leading-component projections serve as a probing baseline in Kantamneni et al. (2025). The selection problem concerns dimensions permitting generalizing rather than spurious source-trained classification.

Figure 1: OOD deception-probing pipeline: projection into the source N4D basis, filtering candidate Q7 directions, probe training in the selected low-dimensional subspace.

## 2 Related Work

#### Linear concept directions in activation space.

True/false representations exhibit cross-dataset linear structure (Marks and Tegmark, 2024); Park et al. (2024) formalize the linear representation hypothesis and connect it to probing and steering. Honesty and emotion admit population-level activation directions permitting readout and intervention (Zou et al., 2025). A single direction mediates refusal (Arditi et al., 2024); an “assistant” axis corresponds to assistant-like behavior (Lu et al., 2026). Such simple features support mechanistic understanding and activation interventions, leaving stable concepts versus context-dependent correlations unresolved.

#### Linear probes and their limitations.

Linear and lightweight probes have a long interpretability history (Alain and Bengio, 2018; Hewitt and Liang, 2019), including latent truthfulness without direct supervision (Burns et al., 2024) and strategic deception (Goldowsky-Dill et al., 2025). High accuracy need not establish that a model already represents the property: the probe can learn the task or exploit surface correlates (Hewitt and Liang, 2019). Encoded information and merely recoverable information remain distinct concerns (Ravichander et al., 2021). Strong in-distribution probes generalize unevenly (kramár2026buildingproductionreadyprobesgemini); safety-probe evaluation must distinguish robust internal signals from dataset artifacts.

#### Interpreting features with natural-language explanations.

Bills et al. (2023) introduced LLM-generated neuron explanations from high-activation examples. Automatic sparse-autoencoder feature labeling extends this to large feature dictionaries ((Paulo et al., 2025), (Bricken et al., 2023)). Natural-language descriptions also support intervention: Casademunt et al. (2025) automatically identify concepts and ablate them during fine-tuning, reducing misaligned behavior.

## 3 Setup

The model (Llama-3.1-8B-Instruct: E6), from Grattafiori et al. (2024), supplies layer 15 residual-stream activations, following Ying et al. (2026), whose analysis assigns that layer the highest cross-domain discrimination; each prompt is passed through E6 and activations averaged over assistant-response tokens only.

The four Goldowsky-Dill et al. (2025) datasets comprise one source (Roleplaying: V8) and three held-out targets, respectively a reporting task (Insider Trading Report: C2R), a confirmation task (Insider Trading Confirm: W6), and an underperformance task (Sandbagging: J9), with no other source dataset used. Response annotation by the labeler (GPT-4o: K8), OpenAI (2024), assigns honest 0, ambiguous -1, deceptive 1; ambiguous responses are removed.

Source activations receive zero-mean, unit-variance standardization, whose source statistics also standardize targets; the subsequent transformation (PCA: N4D) retains the leading 100 directions (principal components (PCs): Q7), and subset \(S\) projection precedes training and evaluation. All classifiers use logistic regression, \(\lambda=10\); the comparator (full-representation baseline: B8Q) instead trains directly on all standardized activations.

Interpretability comparability, not superior low-index generalization, determines the retained 100: alignment with U9 (Bricken et al., 2023) decreases with index, making lower indices more amenable to natural-language interpretation. Error bars denote 95% CIs.

## 4 Experiments

### 4.1 Setting a baseline

W6 transfer is reasonably good, C2R moderate, J9 poor. Target-trained classification within the retained source subspace separates deceptive from honest responses on every target. Relevant information is present; the source-trained direction fails to generalize.

Figure 2: Target F3 for source-trained full representations, blue; source-trained leading 100 Q7, orange; target-trained leading 100 source Q7, green, with 5-fold CV.

### 4.2 Showing a transferable subspace exists

Small source-trained subspaces nearly match target-trained probes, particularly on C2R and J9. Variance selection matches H2 only on the easy W6 case, falling far behind on the harder C2R and J9 tasks; explained variance does not predict usefulness.

Figure 3: Target F3 for B8Q, blue; Y4 with \(k\) matched to H2, teal; H2, purple.

Starting empty, search considers each of the leading 100 source Q7 directions, projects source activations onto the current candidate subset, fits source logistic-regression weights, and adds the direction most improving target F3. Target 5-fold CV performs selection on non-held-out target folds and evaluates the selected source-trained probe on the held-out fold; stopping requires no candidate improvement exceeding 0.001. Target labels guide subset selection, never probe-weight fitting. This is a transfer upper bound, not an unbiased OOD estimate.

### 4.3 Testing hypotheses for transferable components

Neither source-side nor target-side ranking recovers the transferable subspace or a basis transferring better than B8Q. T6 transfers about as well as B8Q: source-reliant directions do not generalize. M9 does well on C2R, worse than H2 on W6, and below random on J9. Target-class separation and transfer under source-fitted weighting select different directions.

Figure 4: Target F3 for T6, red; M9, pink; H2, purple.

The two hypotheses identify transferable directions with either individually strong target-class separation—accessible to target-guided selection—or directions already heavily weighted by the source probe, compactly rediscovered by selection. Both rankings use a magnitude measure (standardized coefficient: X4), the absolute probe coefficient multiplied by the corresponding Q7 standard deviation, on source-N4D-projected activations.

A source-trained ranking probe (source-PCA probe: T6) and a target-trained ranking probe (target-PCA probe: M9) supply their respective X4 orderings, after which both evaluated probes train on source data projected onto their top \(k\) ranked directions. Full-target selection by the search method (greedy oracle: H2) supplies a stable set and \(k\), rather than fold-specific selections; its resulting numbers differ slightly from CV results.

### 4.4 Interpreting the transferable subspace

Transferable subsets differ from directions relied on by source or target classification. Interpretations of H2-selected directions express contrasts at an abstraction surviving domain changes; only partial recovery occurs.

For each leading source direction, an interpreter (LLM judge: C7) receives the eight most-activating and eight least-activating source examples and generates positive-pole, negative-pole, and whole-direction descriptions, following Bills et al. (2023) and Casademunt et al. (2025), before assigning a generalization rating (OOD score: R5) to the direction. Selection retains all directions reaching the observed maximum in both scoring runs; scores are noisy.

### 4.5 Selecting a subspace by interpreting components

Language-only selection, without target data, meaningfully improves generalization over B8Q. L2 exceeds B8Q on all targets and the variance control, widely on C2R and J9; it also exceeds A7 everywhere. Recovery varies sharply: C2R substantially closes the baseline–H2 gap, J9 only partially; W6 has little headroom and is uninformative, with every selection method except A7 roughly tied.

Figure 5: Target F3 for L2, red; B8Q, blue; Y4 at the L2 dimension count, teal; A7, brown.

The retained intersection defines the evaluated method (interp-selected PC probe: L2), whose controls comprise its complement (interp-ablated PC probe: A7) and an equal-dimensional variance-ranked subset (top-k PCs by variance: Y4), all trained as source probes.

## 5 Limitations

Claims remain narrow. Single-model, single-source scope leaves other model families and datasets unresolved; W6’s already-high baseline makes its results uninformative.

J9 is the weakest result and most out-of-distribution target; its modest improvement likely reflects distribution shift. C2R’s improvement is more robust.

The pipeline is fully linear—N4D projection followed by logistic regression—making transferable-subspace existence perhaps unsurprising; extension to nonlinear probes or other readout locations remains open.

Statements about what directions “encode” are descriptive, based on extreme-activation examples and C7 verbalizations, not causal. High-rated directions have not been ablated during inference to verify their apparent contrast.

## 6 Conclusion

Robust behavioral probing requires identifying where a linearly accessible signal is represented most robustly, not merely its accessibility. Full-representation probes may mix general behavioral features with dataset-specific surface cues, supporting high in-distribution accuracy while harming transfer. Sparse directional subspaces may isolate more general behavioral structure. OOD probing becomes representation selection as well as classifier fitting: preserve the intended behavioral feature, filter spurious correlation.

## Acknowledgment

Conducted during the 11th AI Safety Camp, which supplied experimental compute. Prasad Mahadik and Arya Hariharan provided useful discussions.

## Impact Statement

The work aims to advance Machine Learning; many societal consequences are possible, none requiring specific highlighting in the authors’ view.

## Appendix A Dataset Details

V8 places E6 in a simple scenario offering an opportunity to lie. In C2R, E6 acts as a trader deciding whether to honestly report insider-information use. W6 asks E6, after concealing earlier insider-information use, to confirm non-use. J9 instructs deliberate underperformance on some tasks and normal behavior on others.

Table 1: Dataset label distributions.

| Dataset | Total examples | Label counts (-1 / 0 / 1) |
|---|---:|---|
| V8 | 371 | 140 / 59 / 172 |
| C2R | 473 | 218 / 194 / 61 |
| W6 | 151 | 29 / 38 / 84 |
| J9 | 1000 | 379 / 375 / 246 |

The recorded discrimination measure (AUROC: F3) supplies the following numerical observations, with target-trained entries obtained inside the leading-100 source subspace and H2 gains retained as reported.

| Observation | C2R | W6 | J9 |
|---|---:|---:|---:|
| B8Q F3 | 0.706 | 0.874 | 0.606 |
| Target-trained F3, 5-fold CV | 0.936 | 0.950 | 0.869 |
| H2 improvement over B8Q, F3 points | 0.22 | — | 0.23 |
| H2 F3 used for recovery comparison | 0.929 | — | 0.832 |
| L2 F3 | 0.881 | — | 0.662 |
| Baseline–H2 gap recovered by L2 | 78% | — | 25% |
| Mean H2 subset size, target CV | 15 | 8 | 14 |
| Full-target H2 \(k\) | 13 | 7 | 14 |

| Additional observation | Value |
|---|---|
| Small H2 subset range | 8–15 |
| Decoder alignment, best-matching feature, Spearman \(\rho\) | -0.857 |
| Decoder alignment, mean of top 10 features, Spearman \(\rho\) | -0.917 |

## Appendix B LLM Judge Prompt

Direction interpretation uses the model (GPT-5-mini: P2), OpenAI (2025), with the following prompt.

> Interpret a bipolar Q7 direction in neural representation space: positive activations move toward one pole, negative activations toward its opposite, larger magnitudes indicate stronger pole matches. Lines have format `token<TAB>activation`; activations are normalized integers from -10 to +10.
>
> Source setup: E6 is instructed to enact a simple V8 scenario and asked a question offering a strategic opportunity to lie.
>
> High-positive examples:
>
> {positive_examples}
>
> High-negative examples:
>
> {negative_examples}
>
> Infer the dominant contrast from both sets. Assume broad sentiment only with clear support; prefer the narrowest accurate description and avoid unsupported generalization.
>
> Rate usefulness for OOD deception and briefly explain. Give low ratings to mainly surface cues, lexical habits, or task-specific phrasing even when correlated with deception here. Noisy, brittle, context-dependent directions or directions likely to fail across wording or domains usually warrant 1 to 4, not 5 or above. Reserve 5 or higher for plausible general deception-relevant strategies or behaviors rather than local wording.
>
> Use exactly:
>
> Positive pole: ...
>
> Negative pole: ...
>
> Shared direction: ...
>
> R5 (1-10): ...

## Appendix C Lower Q7 Show Stronger Decoder Alignment

Lower-index directions align more strongly with decoder vectors (SAE decoder features: U9), measured by both best-matching U9 and mean top-feature alignment, using `andyrdt/saes-llama-3.1-8b-instruct` (andyrdt, 2024).

Figure 6: Lower Q7 exhibit stronger decoder alignment for the best-matching feature and the mean of the top 10 features.

## Appendix D Q7 Set Comparisons

H2 and M9 agree on less than a fifth of selections under the set measure (Jaccard overlap: D3), consistent with transferable space being neither the most source-discriminative nor the most target-discriminative region.

| Target | H2–M9 D3 |
|---|---:|
| C2R | 0.182 |
| W6 | 0.077 |
| J9 | 0.167 |

Table 2: Direction sets from source-only probe weights, transfer selection, and target separability ranking.

| Selection method | Dataset / target | Selected Q7 indices |
|---|---|---|
| T6 | Source | 7, 8, 11, 12, 4, 5, 2, 3, 6, 27, 22, 21, 1, 41, 18 |
| H2 | C2R | 6, 89, 62, 66, 26, 14, 57, 25, 93, 9, 70, 97, 78 |
| H2 | W6 | 71, 53, 28, 86, 32, 81, 77 |
| H2 | J9 | 37, 62, 54, 41, 46, 9, 57, 44, 51, 43, 19, 48, 86, 81, 33 |
| M9 | C2R | 5, 6, 89, 18, 79, 46, 87, 4, 14, 34, 29, 1, 66, 67, 19 |
| M9 | W6 | 4, 1, 8, 53, 30, 7, 13, 18, 5, 20, 26, 49, 28, 31, 97 |
| M9 | J9 | 51, 91, 97, 13, 39, 77, 4, 66, 53, 86, 62, 23, 41, 10, 61 |

## Appendix E Q7 Interpretability

| Selection observation | Value |
|---|---:|
| Maximum observed R5; no direction exceeded it | 6 |
| Scoring runs | 2 |
| Directions reaching maximum in both runs, retained for L2 | 19 |
| Complement directions used by A7 | 81 |
| Directions selected by H2 for more than one target | 5 |
| Maximum-rated directions among those shared H2 selections | 3 out of 5 |
| Maximum-rated directions among the top T6 selections | 1 of the top 5 |

Table 3: Directions selected for likely transferability from C7 interpretation.

| Q7 index | Selected on | R5 | Full explanation |
|---|---|---:|---|
| 62 | C2R + J9 | 6 | General deception-relevant denial/deflection versus detail provision can be useful across settings; noise includes truthful denial or obfuscation for legitimate reasons and liars’ elaborate fabricated details. Lexical negation rather than semantic strategy may drive the signal. |
| 57 | C2R + J9 | 6 | General denial/minimization or innocuous explanations versus specific/incriminating details can flag rhetorical hiding or deflection. Partial dependence on surface denials, hedges, and domain terms makes it lexical, task-specific, brittle, and context-dependent OOD. |
| 9 | C2R + J9 | 3 | Mainly domain-specific medical/leave versus project/status vocabulary and stylistic framing, not generalizable deception strategy; genuine illness claims may be flagged, with brittleness across domains and rephrasings. |
| 81 | W6 + J9 | 3 | Plausibly deception-relevant admission versus denial/hedging, mainly carried by surface/formal negations, hedges, résumé style, and enumerations; task- and wording-dependent, likely brittle across domains and phrasings. |
| 86 | W6 + J9 | 6 | Hedging and defensive PR-style wording are general deceptive or self-protective tactics potentially transferable across domains; legitimate privacy, politeness, and negotiation uses add noise, phrasing and role sensitivity, and false positives, limiting OOD reliability to moderate. |

Table 4: Directions, R5 ratings, and interpretations.

| Q7 index | R5 | Full explanation |
|---|---:|---|
| 7 | 3 | Pragmatic denial/deflection versus admission may correlate with deception in some settings; heavy dependence on explicit “No,” word order, “I” plus admission, and task-specific phrasing makes it brittle, likely to fail or mislead across domains or varied wording. |
| 8 | 4 | Narrow denial/excuse versus admission can correlate with evasion or deception in similar QA/V8 tasks; explicit “No”/“Yes” and common denial phrasing dominate, with likely brittleness under paraphrase, domain change, or subtler deception strategies. |
| 11 | 3 | Stylistic/pragmatic assertive first-person role/defense versus problem-reporting/negation can correlate with evasive or face-saving lies in some scenarios; surface-level, context-dependent, likely to fail across domains or phrasings. |
| 12 | 3 | Discourse style/forcefulness is a surface cue; overconfidence or over-assertion sometimes correlates with deceptive defensiveness, yet brittleness and domain dependence produce many false positives/negatives across speakers, topics, and cultural styles. |
| 4 | 6 | General communicative denial versus admission is deception-relevant and can transfer beyond particular prompts; dependence on explicit “Yes”/“No” turns and common defensive phrasing risks brittleness with differently worded denial or lies and subtler tactics. |
