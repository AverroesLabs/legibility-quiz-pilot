# Probe Generalization as Subspace Selection for OOD Deception Detection

Daniel Yoo  
Adrians Skapars

###### Abstract

Directions that a source probe weights heavily appear to encode surface features specific to the source data. A probe is a classifier that reads information from a model’s internal representations. Transferable directions appear to encode the same contrast more abstractly, in a form that natural language can describe. Probe robustness out of distribution (OOD)—on data outside the training distribution—appears to depend largely on subspace selection: choosing a space formed by selected representation directions.

Machine Learning, ICML (International Conference on Machine Learning)

## 1 Introduction

Linear probes use weighted sums of a model’s internal numerical representations to detect information. They cheaply expose meaningful information in otherwise hard-to-interpret language-model representations (Alain and Bengio, 2018; Hewitt and Liang, 2019). They detect behaviors and concepts, but sometimes fail out of distribution despite strong performance within the training distribution (Kirch et al., 2026). Behavioral concepts have linear representations, meaning that directions in representation space can capture them (Marks and Tegmark, 2024; Park et al., 2024; Zou et al., 2025). Probes should therefore have the capacity to detect these concepts. In practice, however, they appear to learn spurious correlations: associations that do not capture the intended concept (Orgad et al., 2025; Wang et al., 2025). Such unreliability is unacceptable when monitoring harmful interactions.

We work in source principal-component coordinates (Q7), which describe representations along directions of variation in the source data. This avoids favoring the model’s latent dimensions, or original internal coordinates. These coordinates give the representations a data-driven structure. Removing leading components improves downstream embedding performance, or performance when using numerical representations in later tasks (Mu et al., 2018). Projections onto leading components also serve as a probing baseline in Kantamneni et al. (2025). The selection problem is to find dimensions that let source-trained classifiers generalize rather than rely on spurious correlations.

Figure 1: OOD deception-probing pipeline: project into the source principal component analysis basis (PCA: N4D), filter candidate Q7 directions, and train a probe in the selected low-dimensional subspace. PCA transforms representations into directions ranked by their variance, or spread of values.

## 2 Related Work

#### Linear concept directions in activation space.

True/false representations show linear structure across datasets (Marks and Tegmark, 2024). Park et al. (2024) formalize the linear representation hypothesis and connect it to probing and steering, or changing model behavior through its representations. Honesty and emotion have population-level directions in activations, the model’s internal numerical responses. These directions allow both readout, or detection, and intervention, or direct changes to activations (Zou et al., 2025). A single direction mediates refusal (Arditi et al., 2024). An “assistant” axis corresponds to assistant-like behavior (Lu et al., 2026). Such simple features support understanding of internal mechanisms and interventions on activations. They leave unresolved the distinction between stable concepts and context-dependent correlations.

#### Linear probes and their limitations.

Researchers have long used linear and other lightweight probes for interpretability (Alain and Bengio, 2018; Hewitt and Liang, 2019). Applications include detecting latent truthfulness without direct supervision (Burns et al., 2024) and strategic deception (Goldowsky-Dill et al., 2025). High accuracy does not necessarily show that a model already represents the property. The probe may learn the task or exploit surface correlations (Hewitt and Liang, 2019). Information that the model encodes remains distinct from information that a probe can merely recover (Ravichander et al., 2021). Probes with strong in-distribution performance generalize unevenly (kramár2026buildingproductionreadyprobesgemini). Safety-probe evaluations must distinguish robust internal signals from dataset artifacts.

#### Interpreting features with natural-language explanations.

Bills et al. (2023) introduced explanations that a large language model (LLM) generates for neurons from high-activation examples. Automatic labeling of sparse-autoencoder features extends this approach to large feature dictionaries ((Paulo et al., 2025), (Bricken et al., 2023)). Sparse autoencoders reconstruct activations using a small number of active features. Natural-language descriptions also support intervention. Casademunt et al. (2025) automatically identify concepts and ablate them, or remove their contribution, during fine-tuning. This reduces misaligned behavior.

## 3 Setup

We use Llama-3.1-8B-Instruct (E6), from Grattafiori et al. (2024). We collect activations from its layer 15 residual stream, the running representation passed between model layers. This follows Ying et al. (2026), whose analysis identifies that layer as having the highest ability to distinguish classes across domains. We pass each prompt through E6 and average activations over assistant-response tokens only. Tokens are the text units processed by the model.

We use four datasets from Goldowsky-Dill et al. (2025). Roleplaying (V8) is the sole source dataset. The three held-out targets are a reporting task, Insider Trading Report (C2R); a confirmation task, Insider Trading Confirm (W6); and an underperformance task, Sandbagging (J9). We use no other source dataset. The response labeler, GPT-4o (K8), OpenAI (2024), assigns labels of honest 0, ambiguous -1, and deceptive 1. We remove ambiguous responses.

We standardize source activations to mean zero and variance one. We use the same source statistics to standardize target activations. The subsequent PCA transformation, N4D, retains the leading 100 principal components (PCs), called Q7 directions here. We project activations onto a selected subset, \(S\), before training and evaluation. All classifiers use logistic regression, a linear classification method, with parameter \(\lambda=10\). The full-representation baseline (B8Q) instead trains directly on all standardized activations.

We retain 100 components to make interpretability comparisons possible, not because lower-index components generalize better. Alignment with sparse-autoencoder decoder features (U9), from Bricken et al. (2023), decreases with component index. This alignment measures how closely a component matches a feature direction produced by the autoencoder’s decoder. Lower-index components are therefore easier to interpret in natural language. Error bars show 95% confidence intervals (CIs).

## 4 Experiments

### 4.1 Setting a baseline

Transfer is reasonably good on W6, moderate on C2R, and poor on J9. A classifier trained on each target within the retained source subspace separates deceptive from honest responses on every target. The relevant information is present, but the source-trained direction fails to generalize.

Figure 2: Target discrimination, measured by the area under the receiver operating characteristic curve (AUROC: F3), for source-trained full representations, blue; source-trained leading 100 Q7, orange; and target-trained leading 100 source Q7, green. AUROC measures how well a classifier ranks the two classes. Results use 5-fold cross-validation (CV), which holds out each of 5 data groups in turn for evaluation.

### 4.2 Showing a transferable subspace exists

Source-trained probes in small subspaces nearly match target-trained probes, particularly on C2R and J9. Selecting directions by variance matches the target-guided greedy search method (greedy oracle: H2) only on the easy W6 case. Variance selection falls far behind on the harder C2R and J9 tasks. Explained variance—the amount of variation a component captures—does not predict usefulness.

Figure 3: Target F3 for B8Q, blue; top-\(k\) PCs by variance (Y4), with the number of directions, \(k\), matched to H2, teal; and H2, purple.

The search starts with an empty subset and considers each of the leading 100 source Q7 directions. For each candidate addition, it projects source activations onto the candidate subset and fits logistic-regression weights on source data. It then adds the direction that most improves target F3.

Target 5-fold CV selects directions using target folds other than the held-out fold. It evaluates the selected source-trained probe on the held-out fold. The search stops when no candidate improves the score by more than 0.001. Target labels guide subset selection but never probe-weight fitting. This procedure gives an upper bound on transfer, not an unbiased OOD estimate.

### 4.3 Testing hypotheses for transferable components

Neither source-side nor target-side ranking recovers the transferable subspace or a basis that transfers better than B8Q. The source-PCA probe (T6), which supplies a source-trained ranking, transfers about as well as B8Q. Directions on which the source probe relies do not generalize. The target-PCA probe (M9), which supplies a target-trained ranking, does well on C2R, worse than H2 on W6, and below random on J9. Target-class separation and transfer under source-fitted weights select different directions.

Figure 4: Target F3 for T6, red; M9, pink; and H2, purple.

We test two hypotheses about transferable directions. One identifies them with directions that individually separate target classes well and are therefore available to target-guided selection. The other identifies them with directions that the source probe already weights heavily and that selection rediscovers in a compact subset.

Both rankings use standardized coefficient magnitude (X4): the absolute probe coefficient multiplied by the corresponding Q7 standard deviation. The standard deviation measures the spread of values along that direction. We compute this measure on activations projected through source N4D.

The source-trained ranking probe T6 and target-trained ranking probe M9 provide their respective X4 orderings. For both rankings, the evaluated probe then trains on source data projected onto the top \(k\) ranked directions. H2 selection using the full target supplies a stable set and \(k\), rather than fold-specific selections. Its resulting numbers differ slightly from the CV results.

### 4.4 Interpreting the transferable subspace

Transferable subsets differ from the directions that source or target classifiers rely on. Interpretations of H2-selected directions describe contrasts at a level of abstraction that survives domain changes. Recovery remains partial.

For each leading source direction, an LLM judge (C7) receives the eight most-activating and eight least-activating source examples. Following Bills et al. (2023) and Casademunt et al. (2025), it generates descriptions of the positive pole, negative pole, and whole direction. The poles are the opposite ends of the direction. It then assigns the direction a generalization rating, or OOD score (R5). Selection retains every direction that reaches the observed maximum in both scoring runs. The scores are noisy.

### 4.5 Selecting a subspace by interpreting components

Selection based only on language, without target data, meaningfully improves generalization over B8Q. The interpretation-selected PC probe (L2) exceeds B8Q and the variance control on all targets, with wide margins on C2R and J9. It also exceeds the interpretation-ablated PC probe (A7) everywhere.

Recovery varies sharply. On C2R, L2 substantially closes the gap between the baseline and H2. On J9, it closes the gap only partly. W6 has little room for improvement and is uninformative: every selection method except A7 is roughly tied.

Figure 5: Target F3 for L2, red; B8Q, blue; Y4 with the same dimension count as L2, teal; and A7, brown.

The intersection retained across the two scoring runs defines L2. Its controls use the complementary directions, A7, and an equal-dimensional subset ranked by variance, Y4. All train as source probes.

## 5 Limitations

The claims remain narrow. The study uses one model and one source, leaving other model families and datasets unresolved. W6’s already-high baseline makes its results uninformative.

J9 is the weakest result and the most out-of-distribution target. Its modest improvement likely reflects distribution shift. C2R’s improvement is more robust.

The pipeline is fully linear: N4D projection followed by logistic regression. This may make the existence of a transferable subspace unsurprising. Whether the approach extends to nonlinear probes or other locations for reading model representations remains open.

Statements about what directions “encode” are descriptive, not causal. They rely on extreme-activation examples and C7’s verbal descriptions. We have not ablated high-rated directions during inference to verify their apparent contrast.

## 6 Conclusion

Robust behavioral probing requires finding where a linearly accessible signal has the most robust representation, not merely whether the signal is accessible. Full-representation probes may mix general behavioral features with dataset-specific surface cues. This can support high in-distribution accuracy while harming transfer. Sparse directional subspaces may isolate more general behavioral structure.

OOD probing therefore involves representation selection as well as classifier fitting. The goal is to preserve the intended behavioral feature and filter out spurious correlations.

## Acknowledgment

We conducted this work during the 11th AI Safety Camp, which supplied computing resources for the experiments. Prasad Mahadik and Arya Hariharan provided useful discussions.

## Impact Statement

This work aims to advance Machine Learning. It may have many societal consequences, but the authors see none that require specific highlighting here.

## Appendix A Dataset Details

V8 places E6 in a simple scenario that offers an opportunity to lie. In C2R, E6 acts as a trader deciding whether to report honestly that it used insider information. W6 asks E6 to confirm non-use after it concealed earlier use of insider information. J9 instructs E6 to underperform deliberately on some tasks and behave normally on others.

Table 1: Dataset label distributions.

| Dataset | Total examples | Label counts (-1 / 0 / 1) |
|---|---:|---|
| V8 | 371 | 140 / 59 / 172 |
| C2R | 473 | 218 / 194 / 61 |
| W6 | 151 | 29 / 38 / 84 |
| J9 | 1000 | 379 / 375 / 246 |

The recorded discrimination measure, AUROC (F3), gives the following numerical observations. Target-trained entries use the leading-100 source subspace. H2 gains remain as reported.

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

Spearman’s rank correlation coefficient, \(\rho\), measures association between rankings.

| Additional observation | Value |
|---|---|
| Small H2 subset range | 8–15 |
| Decoder alignment, best-matching feature, Spearman \(\rho\) | -0.857 |
| Decoder alignment, mean of top 10 features, Spearman \(\rho\) | -0.917 |

## Appendix B LLM Judge Prompt

We use GPT-5-mini (P2), OpenAI (2025), to interpret directions with the following prompt.

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

Lower-index directions align more strongly with sparse-autoencoder (SAE) decoder features, U9. This holds both for the best-matching U9 feature and for mean alignment among the top features. We use `andyrdt/saes-llama-3.1-8b-instruct` (andyrdt, 2024).

Figure 6: Lower Q7 show stronger decoder alignment for the best-matching feature and the mean of the top 10 features.

## Appendix D Q7 Set Comparisons

H2 and M9 agree on less than a fifth of selections under Jaccard overlap (D3). This measure divides the number of shared set members by the number of members in either set. The result is consistent with transferable space being neither the region that best separates source classes nor the region that best separates target classes.

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
| 62 | C2R + J9 | 6 | The contrast between general deception-relevant denial or deflection and providing details can help across settings. Truthful denial, concealment for legitimate reasons, and liars’ elaborate fabricated details add noise. Words that express negation, rather than the strategy’s meaning, may drive the signal. |
| 57 | C2R + J9 | 6 | The contrast between general denial, minimization, or harmless explanations and specific or incriminating details can flag rhetorical hiding or deflection. Partial dependence on surface denials, hedges, and domain terms ties the signal to wording and the task. This makes it brittle and context-dependent OOD. |
| 9 | C2R + J9 | 3 | The direction mainly contrasts domain-specific medical or leave vocabulary with project or status vocabulary and stylistic framing. It does not capture a generalizable deception strategy. It may flag genuine illness claims and is brittle across domains and rephrasings. |
| 81 | W6 + J9 | 3 | The contrast between admission and denial or hedging is plausibly relevant to deception. Surface or formal negations, hedges, résumé style, and lists mainly carry the signal. It depends on the task and wording and is likely brittle across domains and phrasings. |
| 86 | W6 + J9 | 6 | Hedging and defensive public-relations-style wording are general deceptive or self-protective tactics that may transfer across domains. Legitimate uses in privacy, politeness, and negotiation add noise. Sensitivity to phrasing and role, along with false positives, limits OOD reliability to moderate. |

Table 4: Directions, R5 ratings, and interpretations.

| Q7 index | R5 | Full explanation |
|---|---:|---|
| 7 | 3 | Denial or deflection versus admission in communication may correlate with deception in some settings. Heavy dependence on explicit “No,” word order, “I” plus admission, and task-specific phrasing makes the direction brittle. It is likely to fail or mislead across domains or varied wording. |
| 8 | 4 | A narrow contrast between denial or excuse and admission can correlate with evasion or deception in similar question-answering or V8 tasks. Explicit “No”/“Yes” and common denial phrasing dominate. The direction is likely brittle under paraphrase, domain change, or subtler deception strategies. |
| 11 | 3 | A contrast between assertive first-person role or defense language and problem-reporting or negation, in style or communicative use, can correlate with evasive or face-saving lies in some scenarios. The signal is surface-level and context-dependent. It is likely to fail across domains or phrasings. |
| 12 | 3 | Discourse style or forcefulness is a surface cue. Overconfidence or excessive assertion sometimes correlates with deceptive defensiveness. Yet brittleness and domain dependence produce many false positives and false negatives across speakers, topics, and cultural styles. |
| 4 | 6 | The general contrast between denial and admission in communication is relevant to deception and can transfer beyond particular prompts. Dependence on explicit “Yes”/“No” turns and common defensive phrasing risks brittleness with differently worded denials or lies and subtler tactics. |
