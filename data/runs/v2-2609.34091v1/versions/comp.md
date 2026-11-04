# When Is an SAE Feature Interpretable?

A Validation Ladder for EEG Foundation Models. Yucong Cao, Chenqi Li, Tingting Zhu (Dept. of Engineering Science, University of Oxford).

###### Abstract

We test 27 settings: 3 backbones × 3 EEG datasets × 3 depths. Alpha removal changes SAE latent firing 7.3× more than an equal-width sham notch (95% CI [6.2,8.7], bootstrapped over settings). However, the alpha filter deletes far more signal. Normalized by removed spectral energy, the ratio is 0.28× (CI [0.22,0.36]) and exceeds one in 0/27 settings. Latents selected for their alpha-removal response are slightly anti-correlated with relative alpha power on clean EEG (mean r=−0.073). We propose a validation ladder for semantic labels of SAE latents. It asks whether a latent responds, whether the response survives control for removed signal, whether it is specific rather than broadly fragile, and whether the property is visible on unperturbed data. Task-classifier relevance is tested separately. Perturbation sensitivity alone does not establish what a latent represents.

## 1 Introduction

SAE discreteness is not semantic identifiability. A latent whose firing changes after an alpha band-stop can be labelled an "alpha feature" before anyone checks that it represents alpha. Here, selection and evaluation use distinct quantities: perturbation responses choose latents, and clean signals determine what they track.

The ladder (Fig. 1, Table 1):
- **Stages 1–3:** is the response sensitive, larger than the intervention's distortion, and specific rather than broadly fragile?
- **Stage 5 (independent observational validation):** is the property visible without perturbation?
- Passing stages 1–3 and 5 supports only a representational claim.
- **Stage 4 (readout intervention)** is a side branch. It is needed only for the claim that the feature is functionally relevant to a specified readout. A perfect alpha-tracker useless for imagined speech is still an alpha feature.

The individual diagnostics are ordinary. What is new is (a) identifying a recurrent inferential failure in SAE-based biosignal interpretation and (b) an operational hierarchy of alternatives to exclude before assigning physiological semantics. DINOv3 tests whether the problem persists for a foundation model pretrained in another modality.

Contributions:
1. A forked protocol supporting claims of different strengths.
2. A 27-setting demonstration that the 7.3× alpha signature is largely an input-distortion artifact.
3. Evidence that the remaining specialist-like latents neither track clean-EEG alpha in the intended direction nor are preferentially important to a linear readout.

Table 1: Claim supported by each stage. None establishes a causal neural mechanism.

|Evidence|Claim supported|
|---|---|
|Responds to alpha removal|Alpha-sensitive latent|
|Response survives distortion control and competing perturbations|Alpha-selective latent|
|Above, and independently covaries with alpha structure in clean EEG|Evidence for an alpha-representing latent|
|Above, and readout intervention changes task behavior|Task-relevant alpha representation|

Figure 1: Forked validation ladder. In this case study neither claim is supported.

## 2 Experimental design

#### Backbones and data.
- Frozen backbones: CBraMod, REVE, and DINOv3 ViT-B/16 (fed channel×time EEG images; prefix tokens excluded).
- Data at 200 Hz with fixed splits:
  - BCI IV 2a (BCI): motor imagery, 22 ch.
  - PhysioNet Mental Arithmetic (MA): workload, 20 ch.
  - BCI Comp. 2020 Track 3 (Speech): imagined speech, 64 ch.
- Setting = backbone × dataset × depth (27 total).

#### TopK SAEs.
$c=\operatorname{ReLU}(\operatorname{TopK}_{k}((x-b_{\rm dec})W_{\rm enc}+b_{\rm enc})),\ \hat{x}=cW_{\rm dec}+b_{\rm dec}$
- k=32; loss is MSE + AuxK.
- Early/middle/late tokens layer-normalized to $\mathbb{E}\|x\|_2=\sqrt d$.
- Expansion 16×; DINOv3 uses 4×/8× where token support is limited.
- Test FVU ∈[0.059,0.228]; $L_0=32$.
- BCI-middle setting of each backbone retrained under 3 seeds.

#### Perturbations and distortion control.
- Each clean test trial is paired with a perturbed copy; SAEs stay frozen.
- Spatial: zero random channels or contiguous windows.
- Spectral: 4th-order zero-phase Butterworth band-stop at δ, θ, α, β, γ, high-γ; plus 8–12 Hz α vs width-matched 30–34 Hz sham.
- Metrics: code cosine, active-set Jaccard, $|\Delta f_j|$.
- EEG is 1/f, so α/sham removed-energy ratios are 57.2/34.6/11.8 (BCI/MA/Speech). Responses are divided by $\|x-\tilde x\|_2/\|x\|_2$ or by the removed spectral-energy fraction.

#### Selectivity, probes, and independent validation.
- Selectivity: $s_{j,b}=|\Delta f_{j,b}|-\frac15\sum_{b'\neq b}|\Delta f_{j,b'}|$.
- Specialist-like α latent: top-quartile α selectivity and ≤ median channel-dropout fragility.
- Controls: size-matched general-fragile and random (readout tests); activity-matched non-specialists, nearest in clean firing rate and mean activation without replacement (clean-alpha tests).
- Probes: logistic, on mean-pooled codes; importance $|w_j|\mathbb{E}_{\rm train}[c_j]$.
- Group ablation: zero a size-matched group's codes and measure Δ balanced accuracy, with a frozen probe (trained clean, ablated at test) or a retrained probe.
- Validation: correlate clean activation with trial-wise relative α power (unused in selection), 12,041 latents per group.
- CIs: percentile 95%, 10,000 bootstrap resamples of settings (27, or 21 for task analyses), not latents.

## 3 Results

### 3.1 A large raw alpha response collapses under distortion control

Alpha changes firing 5–9× more than sham in every backbone and lowers overlap in 27/27 settings (App. B). In the six-band sweep, alpha is the max-|Δf|, min-Jaccard band in 15/27 settings (a plurality, not a rule). Dropout robustness falls with depth (App. C).

Table 2: Pooled α/sham |Δf| ratio.

| \|Δf\| ratio, 27 settings | Mean | 95% CI | Settings >1 |
|---|---|---|---|
| raw | 7.33 | [6.18,8.68] | 27/27 |
| per unit time-domain distortion | 1.32 | [1.12,1.55] | 17/27 |
| per unit removed spectral energy | 0.28 | [0.22,0.36] | 0/27 |

The two normalizations bracket the answer rather than settle it: the energy denominator varies ~5× across datasets. They still show the alpha response is not distortion-independent. Code-overlap ratios follow suit: raw 2.77, normalized 0.51 and 0.11.

### 3.2 Alpha selectivity is not readout importance

Probe BA spans 0.236–0.625, so comparisons are within-setting. Selectivity–importance mean r=0.107 (CI [0.068,0.146], 21 settings; Fig. 2A); |r|≤0.32 in all 9 BCI-middle retrainings.

Frozen-probe ΔBA when zeroing each group:
- Specialists: +0.0055 [−0.007,0.027].
- Fragile: −0.028 [−0.051,−0.003].
- Random: −0.019 [−0.033,−0.008].

Effects shrink after retraining. The ordering is seed-dependent on BCI-middle, and fragile latents fire 8.9× more. Read as a direction only: specialists are not preferentially important to this linear readout. This says nothing about nonlinear or fine-tuned behavior and is not needed for a representational claim.

### 3.3 Independent observational validation fails in the intended direction

- **Top-activating trials:** specialists show +0.012 relative α power over matched fragile exemplars (higher in 12/27).
- **All clean trials:** specialist mean |r|=0.143, vs fragile 0.246 (lower in 26/27) and random 0.097 (Fig. 2B). Signed r is −0.073 (negative in 23/27): specialists fire on lower-alpha trials.
- **Activity-matched (firing 0.374 vs 0.374):** |r| 0.143 vs 0.101 (27/27; Δ|r| CI [0.032,0.052]); sign unchanged. Matching removes the fragile confound but yields no alpha-representing latent.

"Specialist-like" names a perturbation response. Only a simple global frequency-detector reading is excluded, not structured spatial–spectral features.

Figure 2: A: per-setting r, selectivity vs importance (REVE task codes BCI only); means ≈0. B: per-setting mean |r|, specialists vs random.

## 4 Discussion and limitations

The ladder excludes alternatives; it is not a universal feature finder. We do not claim no SAE latent is interpretable. A positive control on another axis is left to future work.

Limits:
- one SAE seed elsewhere;
- readout groups activity-unmatched;
- |Δf| may include TopK rank displacement;
- band power tests only a global reading;
- relevance = logistic probe on pooled codes;
- task results cover 21/27 settings (no REVE MA/Speech exports);
- band-stop can't isolate an oscillator (normalizations bound, not remove, this).

No clinical or causal claims.

## 5 Conclusion

Semantic names for SAE features need the evidence stage in Table 1, not a perturbation response alone.

#### Ethics statement.
Public EEG reanalysis; no participants or clinical predictions; consent inherited from releases.

## Appendix A Additional experimental details

- Layer groups (per-layer ℓ2-normalized): blocks 0–3/4–7/8–11 for CBraMod and DINOv3 (12 blocks); 0–6/7–14/15–21 for REVE (22).
- DINOv3 width: 4× on MA, 8× on BCI.
- Exemplars: top-5 clean activations for ≤3 α specialists + matched fragile per setting, plus β/high-γ sets (298 latents; 1,490 exemplars). Welch spectra give target-band power. All-trial r is primary.
- Matching: unused same-dictionary non-specialist minimizing squared z-distance (union-standardized); activation 0.0060 vs 0.0062.

## Appendix B Per-backbone raw alpha/sham contrast

Means over 9 settings; Δf=|Δf|.

|Model|Jα|Jsham|Δfα|Δfsham|ratio|
|-|-|-|-|-|-|
|CBraMod|.541|.819|.00338|.00046|7.4|
|REVE|.466|.841|.00127|.00014|8.9|
|DINOv3|.484|.782|.00169|.00033|5.1|

## Appendix C Depth-dependent channel-dropout robustness

In all 9 model–dataset combos, late-minus-early Jaccard is negative: −0.17 for random and −0.20 for contiguous dropout. Contiguous Δ ≈−0.35 for DINOv3, −0.19 for CBraMod, −0.07 for REVE. A vision foundation model's useful transfer need not bring internal robustness. This is secondary to the ladder.

## Appendix D Full 21-setting ablation chart

Figure 3: Frozen-probe ΔBA after zeroing size-matched groups, 21 settings; summarized by Fig. 2.

## Appendix E Setting-level bootstrap and activity matching

Percentile 95% CIs, 10,000 setting resamples. Additional values:
- Δ|r| specialist − random: 0.046 [0.033,0.059].
- Δ|r| specialist − activity-matched: mean 0.042.

## Appendix F Licenses, assets, and compute

Assets:
- BCI IV 2a, BCI 2020 Track 3: research use.
- EEGMAT (PhysioNet): CC BY 4.0.
- CBraMod: as released; code BSD-3-Clause.
- REVE-base/positions: as released.
- DINOv3 ViT-B/16: DINOv3 License (Meta).

Compute: A100s on an academic Slurm cluster, ~300 GPU-hours. Covers export for 27 settings, 27 SAEs + 9 seed retrains, sweeps, 21-setting probes, and CPU analysis; unused exploration excluded. Anonymized code provided; checkpoints/raw EEG omitted (size).

## NeurIPS Paper Checklist

1. **Claims.** Do abstract/introduction reflect contributions and scope? [Yes]. The abstract and introduction state the empirical scope; Sections 3–4 separate observations from physiological or causal claims. *Guidelines:* claims should state contributions, assumptions, and limitations; match results and expected generalization; aspirational goals are fine if marked as unattained; [No]/[N/A] is viewed poorly.

2. **Limitations.** Discussed? [Yes]. Section 4 covers SAE seeds, TopK competition, activity matching, linear-probe scope, missing task-analysis settings, and dataset scope. *Guidelines:* a separate Limitations section is encouraged. State strong assumptions and robustness to their violation, scope (few datasets or runs), factors affecting performance, computational efficiency and scaling, and privacy/fairness where applicable. Reviewers are instructed not to penalize honesty about limitations.

3. **Theory assumptions and proofs.** [N/A]. Empirical analyses only; no theoretical claims. *Guidelines:* theorems should be numbered and cross-referenced with all assumptions stated; proofs may sit in the supplement with a sketch; informal proofs need formal complements; supporting lemmas should be referenced.

4. **Experimental result reproducibility.** [Yes]. Section 2 gives models, splits, preprocessing, SAE construction, perturbations, metrics, and probe protocol; supplementary code holds exact configurations. *Guidelines:* reproducibility is required regardless of code release. It may come via full description of an algorithm or architecture, model access, checkpoints, or instructions; closed models need some path to verification.

5. **Open access to data and code.** [Yes]. Datasets and backbones are public and cited. An anonymized archive with training and analysis scripts is supplied; checkpoints and raw EEG are omitted because of size. *Guidelines:* [No] is acceptable unless code is central to the contribution. Give exact commands and environment, data access and preparation, scripts for all results (stating any omitted), anonymized at submission.

6. **Experimental setting/details.** [Yes]. Section 2 reports split policy, preprocessing, SAE sparsity and expansion, perturbation bands, metrics, and classifier protocol; full training details are in the supplement. *Guidelines:* give enough detail in the core paper to appreciate the results; the rest may go in code, appendix, or supplement.

7. **Experiment statistical significance.** [Yes]. Means for α/sham ratios, selectivity–importance correlations, clean-alpha |r| contrasts, and group ablations carry percentile 95% CIs from 10,000 bootstrap resamples of settings (27, or 21 for task analyses). The resampling unit is the setting, not the latent. Initialization variability is quantified for BCI-middle under three SAE seeds. *Guidelines:* state the variability captured, the method, and the assumptions; say whether bars are SD or SE. 1-sigma is OK if stated. If normality is unverified, prefer 2-sigma bars over claiming a 96% CI. Avoid symmetric bars that go out of range. Explain in text how bars in figures and tables were computed.

8. **Experiments compute resources.** [Yes]. Appendix F: A100 GPUs on a shared academic Slurm cluster, ~300 GPU-hours, excluding unused exploratory runs. *Guidelines:* report worker type, memory, storage, per-run and total compute, and whether the full project used more compute.

9. **Code of ethics.** [Yes]. Reanalyzes public research datasets, reports negative findings and limitations, and makes no clinical or individual-level claims. *Guidelines:* explain any deviation; preserve anonymity.

10. **Broader impacts.** [Yes]. Section 4 motivates stricter validation of biosignal interpretations and limits findings to representation auditing, not clinical or neural causality. *Guidelines:* consider misuse, fairness, privacy, and security; harms under correct use, incorrect results, and misuse; mitigations. Foundational work need only flag direct paths to negative applications.

11. **Safeguards.** [N/A]. Releases no new generative model, personal data, or high-risk dataset; analyzes existing frozen models and aggregate results. *Guidelines:* high-risk releases need controlled access, usage guidelines, or filters; scraped data needs safety handling; make a best-faith effort.

12. **Licenses for existing assets.** [Yes]. Section 2 cites original dataset releases; Appendix F lists licenses or terms for the three datasets and three backbones. *Guidelines:* cite sources and give version, URL, license name, and terms of service for scraped data. Give both original and derived licenses for repackaged data; contact creators if information is missing.

13. **New assets.** [Yes]. The supplementary repository records configurations, output schemas, random seeds, and run manifests. *Guidelines:* document via structured templates (training, license, limitations), discuss consent, and anonymize at submission.

14. **Crowdsourcing and research with human subjects.** [N/A]. Secondary analysis with no participants or crowdsourcing. *Guidelines:* include instructions, screenshots, and compensation (at least minimum wage) where applicable.

15. **IRB approvals or equivalent.** [N/A]. No new human-subject data; Section 2 and Appendix F cite the public releases. *Guidelines:* state IRB approval if obtained; follow the Code of Ethics and institutional rules; keep anonymity.

16. **Declaration of LLM usage.** [N/A]. LLMs were used only for writing and formatting, not as part of the method. *Guidelines:* declare only important, original, or non-standard LLM use in core methods, per the NeurIPS LLM policy.