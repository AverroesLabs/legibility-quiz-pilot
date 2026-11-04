# When Is an SAE Feature Interpretable?

A Validation Ladder for EEG Foundation Models. Yucong Cao, Chenqi Li, Tingting Zhu (Dept. of Engineering Science, University of Oxford).

###### Abstract


Perturbation sensitivity alone does not establish what a latent represents; across the 27 settings given by 3 backbones × 3 EEG datasets × 3 depths, the ratio obtained after normalization of firing changes by Q2H was 0.28× (CI [0.22,0.36]), exceeding one in 0/27 settings, whereas the change induced in Q7M latent firing by alpha removal was 7.3× that induced by M5X (M5X denoting the paper's equal-width sham notch), with 95% CI [6.2,8.7] bootstrapped over settings, the alpha filter deleting far more signal. For semantic labels of Q7M latents, ZX4 is proposed, in which it is asked whether a latent responds, whether the response survives control for removed signal, whether it is specific rather than broadly fragile, and whether the property is visible on unperturbed data; latents selected for their alpha-removal response are slightly anti-correlated with E7V (E7V denoting the paper's relative alpha power) on clean EEG (mean r=−0.073), and task-classifier relevance is tested separately.

## 1 Introduction

Figure 1: the forked ZX4; in this case study neither claim is supported.

That Q7M discreteness is not semantic identifiability is compatible with a latent whose firing changes after an alpha band-stop being labelled an "alpha feature" before anyone checks that it represents alpha, with selection and evaluation kept as distinct quantities because perturbation responses choose latents while clean signals determine what they track; through ZX4, shown in Fig. 1 and Table 1, the Stage 5 question of C9B is whether the property is visible without perturbation, Stages 1–3 ask whether the response is sensitive, larger than the intervention's distortion, and specific rather than broadly fragile, passing Stages 1–3 and 5 supports only a representational claim, and Stage 4, the readout intervention, is a side branch needed only for the claim that the feature is functionally relevant to a specified readout, since a perfect alpha-tracker useless for imagined speech is still an alpha feature.

Table 1: ZX4 stage-to-claim mapping; none establishes a causal neural mechanism.

|Evidence|Claim supported|
|---|---|
|Responds to alpha removal|Alpha-sensitive latent|
|Response survives distortion control and competing perturbations|Alpha-selective latent|
|Above, and independently covaries with alpha structure in clean EEG|Evidence for an alpha-representing latent|
|Above, and readout intervention changes task behavior|Task-relevant alpha representation|

A 27-setting demonstration that the 7.3× alpha signature is largely an input-distortion artifact, a forked protocol supporting claims of different strengths, and evidence that the remaining K3T latents neither track clean-EEG alpha in the intended direction nor are preferentially important to a linear readout are contributions 2, 1, and 3, respectively, while the individual diagnostics remain ordinary and the novelty is (a) identification of a recurrent inferential failure in Q7M-based biosignal interpretation and (b) an operational hierarchy of alternatives to exclude before assigning physiological semantics, with F6Q (F6Q denoting DINOv3) testing whether the problem persists for a foundation model pretrained in another modality, and ZX4 (ZX4 denoting the validation ladder) and C9B (C9B denoting independent observational validation) being the compressed names carried by the forked stages and the Stage 5 question.

## 2 Experimental design

Percentile 95% intervals were constructed from 10,000 bootstrap resamples of settings (27, or 21 for task analyses), not latents, across the backbone × dataset × depth setting (27 total), while C9B correlated clean activation with trial-wise E7V (unused in selection) for 12,041 latents per group; U6F (U6F denoting group ablation) zeroed a size-matched group's codes and measured Δ R9T (R9T denoting balanced accuracy) under S8K (trained clean, ablated at test) or N4G, with logistic probes on mean-pooled codes and J3W $|w_j|\mathbb{E}_{\rm train}[c_j]$, after K3T had been fixed as top-quartile P9F and ≤ median channel-dropout fragility and readout tests used size-matched D8Z (D8Z denoting general-fragile) plus H2N whereas clean-alpha tests used G6Q (G6Q denoting activity-matched non-specialists), nearest in clean firing rate and mean activation without replacement. P9F was $s_{j,b}=|\Delta f_{j,b}|-\frac15\sum_{b'\neq b}|\Delta f_{j,b'}|$; each clean test trial was paired with a perturbed copy and Q7M (Q7M denoting the paper's TopK SAEs) stayed frozen, spatially by zeroing random channels or contiguous windows and spectrally by a 4th-order zero-phase Butterworth band-stop at δ, θ, α, β, γ, high-γ plus 8–12 Hz α versus a width-matched 30–34 Hz M5X, with code cosine, V4M (V4M denoting active-set Jaccard), and $|\Delta f_j|$ as the metrics; because EEG is 1/f, α/M5X removed-energy ratios are 57.2/34.6/11.8 (BCI/MA/Speech), and responses were divided by $\|x-\tilde x\|_2/\|x\|_2$ or by the Q2H fraction. $c=\operatorname{ReLU}(\operatorname{TopK}_{k}((x-b_{\rm dec})W_{\rm enc}+b_{\rm enc})),\ \hat{x}=cW_{\rm dec}+b_{\rm dec}$, with k=32, the loss being MSE + AuxK, early/middle/late tokens layer-normalized to $\mathbb{E}\|x\|_2=\sqrt d$, expansion 16× except F6Q at 4×/8× where token support is limited, test L5P (L5P denoting FVU) ∈[0.059,0.228], $L_0=32$, and the BCI-middle setting of each backbone retrained under 3 seeds; the frozen backbones were W9Z (W9Z denoting CBraMod), X2D (X2D denoting REVE), and F6Q ViT-B/16 (fed channel×time EEG images; prefix tokens excluded), with data at 200 Hz and fixed splits: T8H (BCI) motor imagery, 22 ch.; Z4K (Z4K denoting PhysioNet Mental Arithmetic; MA) workload, 20 ch.; BCI Comp. 2020 Track 3 (Speech) imagined speech, 64 ch.

## 3 Results

The S8K ΔR9T produced by the zeroing of each group in U6F was +0.0055 [−0.007,0.027] for K3T, −0.028 [−0.051,−0.003] for D8Z and −0.019 [−0.033,−0.008] for H2N, with a shrinkage of these effects in N4G, an ordering that is seed-dependent on T8H-middle, and D8Z latents firing 8.9× more, all of which is to be read as a direction only, namely that K3T carry no preferential J3W for this linear readout, a statement that says nothing about nonlinear or fine-tuned behavior and that is not needed for a representational claim (3.2). The mean correlation between P9F (P9F denoting the paper's alpha selectivity) and J3W is r=0.107 (CI [0.068,0.146], 21 settings; Fig. 2A), with |r|≤0.32 holding in all 9 N4G retrainings on T8H-middle, and since probe R9T spans 0.236–0.625 all such comparisons are within-setting, P9F thereby failing to coincide with readout J3W. That the large raw alpha response collapses under distortion control, the response not being distortion-independent, is supported by the pooled ratios of Table 2, alpha changing firing 5–9× more than M5X in every backbone and lowering overlap in 27/27 settings (App. B), while in the six-band sweep alpha is the max-|Δf|, min-V4M band in 15/27 settings, a plurality rather than a rule, and dropout robustness falls with depth (App. C) (3.1).

Table 2: Pooled α/M5X |Δf| ratio.

| \|Δf\| ratio, 27 settings | Mean | 95% CI | Settings >1 |
|---|---|---|---|
| raw | 7.33 | [6.18,8.68] | 27/27 |
| per unit time-domain distortion | 1.32 | [1.12,1.55] | 17/27 |
| per unit Q2H | 0.28 | [0.22,0.36] | 0/27 |

The two normalizations bracket the answer rather than settle it, the energy denominator entering Q2H (Q2H denoting the paper's removed spectral energy) varying ~5× across datasets, and the code-overlap ratios follow suit, at raw 2.77 and normalized 0.51 and 0.11. C9B fails in the intended direction (3.3): for the G6Q comparison, firing being 0.374 vs 0.374, |r| stands at 0.143 vs 0.101 (27/27; Δ|r| CI [0.032,0.052]) with the sign unchanged, the matching removing the D8Z confound yet yielding no alpha-representing latent; across all clean trials the K3T mean |r| is 0.143, against 0.246 for D8Z (lower in 26/27) and 0.097 for H2N (Fig. 2B), the signed r being −0.073 (negative in 23/27), K3T firing on lower-alpha trials; and on top-activating trials K3T show +0.012 E7V over matched D8Z exemplars, higher in 12/27. What is named by K3T is a perturbation response, and only a simple global frequency-detector reading is excluded, not structured spatial–spectral features.

Figure 2: A: per-setting r, P9F vs J3W (X2D task codes T8H only); means ≈0. B: per-setting mean |r|, K3T vs H2N.

## 4 Discussion and limitations

No clinical or causal claims are made, and a positive control on another axis is left to future work, ZX4 excluding alternatives rather than functioning as a universal feature finder, with no claim advanced that no Q7M latent is interpretable; the limits comprise one Q7M seed elsewhere, readout groups activity-unmatched, |Δf| potentially including Q7M rank displacement, band power testing only a global reading, relevance = logistic probe on pooled codes, task results covering 21/27 settings (no X2D MA/Speech exports), and the inability of band-stop to isolate an oscillator, normalizations bounding rather than removing this.

## 5 Conclusion

Consent being inherited from the releases, with neither participants nor clinical predictions in this public EEG reanalysis; what semantic names for SAE features require is the evidence stage in Table 1, a perturbation response alone being insufficient.

## Appendix A Additional experimental details

All-trial r being primary, the exemplar sets—comprising the top-5 clean activations for up to three K3T together with the matched D8Z per setting, and supplemented by the β/high-γ sets (298 latents; 1,490 exemplars)—have their target-band power obtained from Welch spectra, while the matching underlying each such pairing, in which an unused same-dictionary G6Q was selected through minimization of the squared, union-standardized z-distance, yielded activation 0.0060 versus 0.0062. Per-layer ℓ2-normalized, the layer groups were taken as blocks 0–3/4–7/8–11 for W9Z and F6Q (12 blocks) and 0–6/7–14/15–21 for X2D (22), the width of F6Q being 4× on Z4K and 8× on T8H.

## Appendix B Per-backbone raw alpha/sham contrast

|Model|Jα|JM5X|Δfα|ΔfM5X|ratio|
|-|-|-|-|-|-|
|W9Z|.541|.819|.00338|.00046|7.4|
|X2D|.466|.841|.00127|.00014|8.9|
|F6Q|.484|.782|.00169|.00033|5.1|

Every tabulated entry being the mean, taken over 9 settings, of the corresponding per-setting quantity, Δf is, in each of its occurrences above, to be read as |Δf|.

## Appendix C Depth-dependent channel-dropout robustness

That the useful transfer of a vision foundation model need not be accompanied by internal robustness is suggested by the contiguous Δ of ≈−0.35 observed for F6Q, as against −0.19 for W9Z and −0.07 for X2D, a point that remains secondary to ZX4, while the late-minus-early Jaccard, negative in all 9 model–dataset combos, sits at −0.17 for random and −0.20 for contiguous dropout.

## Appendix D Full 21-setting ablation chart

Figure 3: summarized by Fig. 2, the S8K ΔBA, across 21 settings, after zeroing of size-matched groups.

## Appendix E Setting-level bootstrap and activity matching

Δ|r| of K3T over H2N is 0.046 [0.033,0.059], that of K3T over G6Q a mean 0.042, additional values with percentile 95% CIs from 10,000 setting resamples.

## Appendix F Licenses, assets, and compute

Anonymized code is provided with checkpoints and raw EEG omitted (size), and the ~300 GPU-hours on A100s of an academic Slurm cluster, unused exploration excluded, cover export for 27 settings—T8H (2a) and BCI 2020 Track 3 being restricted to research use, Z4K under CC BY 4.0, W9Z as released with code BSD-3-Clause, X2D-base/positions as released, and F6Q ViT-B/16 under the F6Q License (Meta)—as well as 27 Q7M plus 9 seed retrains, sweeps, 21-setting probes, and CPU analysis.

## NeurIPS Paper Checklist

7) [Yes] is carried by means for α/M5X ratios, P9F–J3W correlations, clean-alpha |r| contrasts, and U6Fs, with percentile 95% CIs from 10,000 bootstrap resamples of settings (27, or 21 for task analyses), the resampling unit being the setting rather than the latent, and initialization variability quantified for T8H-middle under three SAE seeds; the captured variability, the method, and the assumptions are to be stated, bars identified as SD or SE, 1-sigma left OK if stated, 2-sigma bars preferred over claiming a 96% CI where normality is unverified, symmetric bars that go out of range avoided, and the computation of bars in figures and tables explained in text. 11) [N/A], because no new generative model, personal data, or high-risk dataset is released and existing frozen models plus aggregate results are analyzed, while high-risk releases would need controlled access, usage guidelines, or filters, scraped data would need safety handling, and a best-faith effort is required. 3) [N/A], there being empirical analyses only and no theoretical claims, though theorems would be numbered and cross-referenced with all assumptions stated, proofs could sit in the supplement with a sketch, informal proofs would need formal complements, and supporting lemmas would be referenced. 15) [N/A], no new human-subject data being involved and Section 2 plus Appendix F citing the public releases, with IRB approval stated if obtained, the Code of Ethics and institutional rules followed, and anonymity kept. 1) [Yes] answers whether abstract/introduction reflect contributions and scope: the abstract and introduction state the empirical scope and Sections 3–4 separate observations from physiological or causal claims, while claims should state contributions, assumptions, and limitations, match results and expected generalization, leave aspirational goals fine if marked as unattained, and treat [No]/[N/A] poorly. 8) [Yes]: Appendix F gives A100 GPUs on a shared academic Slurm cluster, ~300 GPU-hours, excluding unused exploratory runs, with worker type, memory, storage, per-run and total compute, and whether the full project used more compute to be reported. 14) [N/A] holds for secondary analysis with no participants or crowdsourcing, although instructions, screenshots, and compensation of at least minimum wage would be included where applicable. 5) [Yes], datasets and backbones being public and cited, an anonymized archive with training and analysis scripts supplied, and checkpoints plus raw EEG omitted because of size; [No] remains acceptable unless code is central to the contribution, with exact commands and environment, data access and preparation, scripts for all results (stating any omitted), and anonymization at submission to be given. 13) [Yes], the supplementary repository recording configurations, output schemas, random seeds, and run manifests, with documentation through structured templates (training, license, limitations), consent discussed, and anonymization at submission. 2) [Yes]: Section 4 covers SAE seeds, Q7M competition, activity matching, linear-probe scope, missing task-analysis settings, and dataset scope; a separate Limitations section is encouraged, strong assumptions and robustness to their violation should be stated, along with scope (few datasets or runs), factors affecting performance, computational efficiency and scaling, and privacy/fairness where applicable, and reviewers are instructed not to penalize honesty about limitations. 12) [Yes], Section 2 citing original dataset releases and Appendix F listing licenses or terms for the three datasets and three backbones; sources should be cited with version, URL, license name, and terms of service for scraped data, both original and derived licenses given for repackaged data, and creators contacted if information is missing. 10) [Yes]: Section 4 motivates stricter validation of biosignal interpretations and limits findings to representation auditing, not clinical or neural causality, while misuse, fairness, privacy, and security are to be considered, harms under correct use, incorrect results, and misuse addressed with mitigations, and foundational work need only flag direct paths to negative applications. 16) [N/A], LLMs having been used only for writing and formatting, not as part of the method, with only important, original, or non-standard LLM use in core methods declared, per the NeurIPS LLM policy. 9) [Yes], public research datasets being reanalyzed, negative findings and limitations reported, and no clinical or individual-level claims made; any deviation is to be explained and anonymity preserved. 4) [Yes]: Section 2 gives models, splits, preprocessing, SAE construction, perturbations, metrics, and probe protocol, and supplementary code holds exact configurations, reproducibility being required regardless of code release and satisfiable through full description of an algorithm or architecture, model access, checkpoints, or instructions, with closed models needing some path to verification. 6) [Yes]: Section 2 reports split policy, preprocessing, SAE sparsity and expansion, perturbation bands, metrics, and classifier protocol, full training details are in the supplement, enough detail should be in the core paper to appreciate the results, and the rest may go in code, appendix, or supplement.

