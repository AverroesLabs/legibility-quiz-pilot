# When Is an SAE Feature Interpretable?

A Validation Ladder for EEG Foundation Models

Yucong Cao

Affiliation: Department of Engineering Science

Affiliation: University of Oxford

  
Chenqi Li

Affiliation: Department of Engineering Science

Affiliation: University of Oxford

  
Tingting Zhu

Affiliation: Department of Engineering Science

Affiliation: University of Oxford

###### Abstract

Sparse autoencoders (SAEs) decompose dense model activations into discrete
latents, which makes individual features easy to inspect and easy to name.
For EEG foundation models, this raises a practical question that precedes
any naming: under which conditions are the fitted latents stable enough,
and reported carefully enough, to be worth interpreting at all? We study
TopK SAEs across 27 settings spanning three backbones (two EEG-pretrained
models and one vision foundation model), three public EEG datasets, and
three network depths. All 27 dictionaries reconstruct held-out activations
well (fraction of variance unexplained between 0.059 and 0.228 at
$L_{0}=32$ ). Paired spatial and spectral perturbations, however, reveal
that sparse codes become progressively less stable with depth: late-layer
active-set overlap under channel dropout is lower than early-layer overlap
in all nine backbone–dataset combinations, most strongly for the vision
backbone. We organize these measurements into a validation ladder for SAE
analyses of biosignal models: reconstruction fidelity, perturbation
stability of the code, consistency across seeds and settings, and
descriptive response profiles under band-limited interventions. The ladder
does not certify that a latent has a particular meaning; it specifies what
should be measured and reported before semantic interpretation is
attempted, and it makes clear that reconstruction quality alone does not
imply a stable or interpretable code.

## 1 Introduction

Foundation models (FMs) for brain and body signals increasingly raise a
mechanistic question: what, exactly, do the learned representations encode?
Sparse autoencoders (SAEs) make that question unusually tempting. By
decomposing dense activations into discrete latents, they invite
human-readable names for individual units (Gao et al., 2025; Cunningham et al., 2023; Bricken et al., 2023).
Discreteness, however, is not semantic identifiability. A dictionary can
reconstruct its inputs well while assigning codes that shift substantially
under small, plausible changes to the recording, and a latent whose
behavior depends on which channels happen to be present is a poor
candidate for a physiological label regardless of how its activations
look on a handful of exemplars.

EEG makes these concerns concrete. Recordings vary in montage, channel
count, and noise; electrodes drop out; spectral content differs across
subjects, tasks, and sessions. An SAE fitted to a frozen backbone inherits
all of this variability, and any interpretive claim about its latents is
implicitly a claim that the latent behaves consistently across it. In
language models, SAE interpretability work typically asks whether
features are sparse, reconstructive, monosemantic, or causally
editable (Cunningham et al., 2023; Bricken et al., 2023; Marks et al., 2025).
EEG explainability typically asks which input structure influences a
prediction (Schirrmeister et al., 2017). Between these traditions sits a
more basic question for biosignal FMs: how should an SAE analysis of an
EEG backbone be evaluated and reported so that later interpretation rests
on stable ground?

We approach that question empirically across 27 settings spanning frozen
CBraMod (Wang et al., 2025), REVE (El Ouahidi et al., 2025), and
DINOv3 (Siméoni et al., 2025) at early, middle, and late depth on three
public EEG tasks. We include DINOv3 to ask whether the behavior of SAE
codes is peculiar to EEG-pretrained representations or persists when EEG
is encoded through a foundation model pretrained in another modality. In
each setting we train a TopK SAE, pair every clean test trial with
spatially and spectrally perturbed copies, and record how the sparse code
changes. We also fit linear probes on pooled codes to relate the
dictionaries to downstream tasks.

The organizing device is a validation ladder
(Figure 1; Table 1). Its lower rungs are
properties that every SAE analysis can and should report: whether the
dictionary reconstructs held-out activations, whether its codes are stable
under perturbations that should not change the underlying signal content
much, and whether those stability properties are consistent across seeds
and settings. Its upper rung is a descriptive response profile under
band-limited interventions, which characterizes how latents react to
specific spectral changes. Each rung licenses a modest, explicitly scoped
statement; none of them, alone or together, establishes a causal neural
mechanism.

The novelty is not any individual diagnostic. Reconstruction error,
perturbation stability, seed variability, and band-wise response profiles
are ordinary measurements. What we contribute is (i) an operational
hierarchy for reporting SAE analyses of biosignal foundation models, with
each level tied to the scope of claim it supports; (ii) a 27-setting study
showing that SAE reconstruction quality is uniformly good while code
stability degrades systematically with depth; and (iii) a cross-modal
comparison indicating that a vision FM that transfers usefully to EEG need
not bring matching internal robustness to spatial perturbation.

Table 1: Evidence available after each stage, and the claim that evidence
supports. None of these establish a causal neural mechanism.

|

Evidence

|

Claim supported

|

Low held-out reconstruction error at fixed sparsity

|

Faithful sparse decomposition

|

Codes remain stable under spatial and spectral perturbation

|

Stable latent code

|

Above, and stability patterns hold across seeds and settings

|

Reproducible latent code

|

Above, and band-wise responses are characterized and reported

|

Descriptively profiled latent

Figure 1: Validation ladder for SAE analyses of EEG foundation models.
Each rung adds a measurement and a corresponding, explicitly scoped claim.
In this case study, every dictionary passes the first rung, while
stability under spatial perturbation degrades with depth, so claims at
higher rungs must be qualified by layer and backbone.

## 2 Experimental design

#### Backbones and data.

We compare EEG-pretrained CBraMod and REVE with DINOv3 ViT-B/16, which
receives channel-by-time EEG images (prefix tokens excluded). Three public
classification sets are sampled at 200 Hz: BCI Competition IV
2a (Tangermann et al., 2012; Brunner et al., 2008) (BCI; motor imagery, 22
channels), PhysioNet Mental Arithmetic (Zyma et al., 2019; Goldberger et al., 2000)
(MA; workload, 20 channels), and BCI Competition 2020 Track 3 imagined
speech (Jeong et al., 2022; Lee et al., 2020) (Speech; 64 channels).
Splits are fixed; backbones are frozen. We refer to each combination of
backbone, dataset, and depth as a setting, giving 27 settings in
total. The datasets were chosen to cover distinct paradigms, channel
counts, and signal-to-noise regimes, so that conclusions about code
stability are not tied to a single montage or task. Licenses and compute
are listed in Appendix F.

#### TopK SAEs.

For $x\in\mathbb{R}^{d}$ ,

| | $c=\operatorname{ReLU}\!\left(\operatorname{TopK}_{k}\left((x-b_{\rm dec})W_{\rm enc}+b_{\rm enc}\right)\right),\qquad\hat{x}=cW_{\rm dec}+b_{\rm dec}.$
|

We set $k=32$ and train with reconstruction MSE plus AuxK dead-latent
recovery (Gao et al., 2025). Token activations from early, middle, or late
blocks are layer-normalized to $\mathbb{E}\|x\|_{2}=\sqrt{d}$ before mixing.
Dictionary expansion is $16\times$ except where token support motivates
$4\times$ or $8\times$ for DINOv3, giving 27 dictionaries. All converge to
test FVU in $[0.059,0.228]$ with $L_{0}=32$ . The BCI middle setting of each
backbone is retrained under three seeds. Further training details are in
Appendix A. The TopK formulation fixes sparsity exactly, which removes one
common source of variation between dictionaries and makes stability
comparisons across settings more direct than with an $\ell_{1}$ penalty
whose effective sparsity drifts during training.

#### Perturbations and distortion control.

Each clean test trial is paired with a perturbed copy. Spatial
interventions zero random channels or contiguous windows. Spectral
interventions apply a fourth-order zero-phase Butterworth band-stop at
$\delta,\theta,\alpha,\beta,\gamma$ , or high- $\gamma$ , plus an 8–12 Hz
alpha test against a width-matched 30–34 Hz sham. SAEs stay frozen. We
record sparse-code cosine, active-set Jaccard, and $|\Delta f_{j}|$ . All
perturbations are applied to the raw signal before the backbone, so the
same clean trial serves as its own reference and differences in code
reflect only the intervention. We also record the time-domain magnitude
$\|x-\tilde{x}\|_{2}/\|x\|_{2}$ of every perturbation so that responses can
be placed alongside the size of the input change that produced them.

#### Selectivity, probes, and independent validation.

Selectivity is
$s_{j,b}=|\Delta f_{j,b}|-\frac{1}{5}\sum_{b^{\prime}\neq b}|\Delta f_{j,b^{\prime}}|$ ,
a descriptive summary of how much more latent $j$ moves under band $b$
than under the other five bands. Channel-dropout fragility is the
analogous quantity for spatial interventions. Mean-pooled SAE codes feed
logistic probes on the fixed split. Importance is
$|w_{j}|\mathbb{E}_{\rm train}[c_{j}]$ . A group ablation sets the codes of
a selected set of latents to zero and measures the resulting change in
probe balanced accuracy, with groups matched in size. In the frozen-probe
variant the probe is trained on clean codes and the ablation is applied
only at test time; in the retrained variant the probe is refit after
ablation. For clean-data inspection we compute each latent’s activation
across unperturbed test trials and the trial-wise Welch spectrum, so that
perturbation-derived summaries can be read alongside the latent’s ordinary
behavior.

Confidence intervals in the text are percentile 95% intervals from 10,000
bootstrap resamples of settings (27 settings, or 21 for task analyses), not
of individual latents.

## 3 Results

### 3.1 A large raw alpha response collapses under distortion control

The first rung of the ladder is passed everywhere: all 27 dictionaries
reach test FVU between 0.059 and 0.228 at $L_{0}=32$ , with no backbone or
dataset standing out as poorly reconstructed. Reconstruction, however, says
nothing about how codes behave when the input changes. Spectral band-stops,
including the alpha notch, alter sparse codes in every backbone, but these
responses sit within a broader pattern of code instability that is most
visible under spatial perturbation and grows with depth. We therefore
treat individual spectral responses as one component of a code’s general
sensitivity to input change rather than as an isolated signature, and
place them alongside the magnitude of the corresponding input change
before reading them further.

Table 2: Late-minus-early change in mean active-set Jaccard under channel
dropout, pooled across the nine backbone–dataset combinations and by
backbone. Negative values indicate that deeper SAE codes are less stable
to spatial perturbation.

| Late $-$ early Jaccard | Random dropout | Contiguous dropout | Combinations with decrease

| all backbones (mean) | $-0.17$ | $-0.20$ | 9/9

| DINOv3 | — | $\approx-0.35$ | 3/3

| CBraMod | — | $\approx-0.19$ | 3/3

| REVE | — | $\approx-0.07$ | 3/3

Channel-dropout robustness decreases with depth across all nine
backbone–dataset combinations (Table 2). The effect is strongest for
DINOv3, intermediate for CBraMod, and mild for REVE. Deeper SAE codes are
thus less stable under the kind of perturbation that routinely occurs in
practice when electrodes are noisy or missing, and this holds for both
EEG-native and cross-modal backbones. A stability statement made at one
depth therefore does not transfer automatically to another, and any later
interpretive claim must be indexed by the layer at which it was made. The
cross-modal contrast is also informative: DINOv3 can encode EEG images
usefully for downstream probes, yet its deeper sparse codes are the least
robust to spatial perturbation, so transfer performance and internal
robustness should be reported as separate properties.

### 3.2 Alpha selectivity is not readout importance

Probe balanced accuracy spans 0.236–0.625 across the 21 settings with
task-analysis codes, so ablations are compared within settings rather
than across them. Selectivity and importance answer different questions.
Selectivity summarizes how a latent’s firing changes when a particular
band is removed from the input; importance summarizes how much a fitted
linear readout weights that latent on clean data. A latent can be strongly
affected by a band-stop and receive little weight from a probe, or the
reverse, and nothing in the construction of either quantity forces them
to agree. We therefore report them as separate columns of the analysis
rather than using one as a proxy for the other.

The group-ablation protocol is designed accordingly. Zeroing a size-matched
group of latents and measuring the change in balanced accuracy, with the
probe either frozen or retrained, tests whether a readout depends on that
group; it does not test what the group encodes. Effects shrink after
probe retraining, as expected when the readout can redistribute weight to
correlated latents, and the ordering of groups on the BCI middle setting
depends on the SAE seed. Groups chosen by perturbation criteria also
differ in baseline activity, which complicates direct comparison. For
these reasons the ablation results are presented descriptively in
Appendix D, and we draw no conclusion about task relevance from them.
Relevance in any case is defined here only for a logistic probe on
mean-pooled codes and does not speak to nonlinear or fine-tuned behavior.

### 3.3 Independent observational validation fails in the intended direction

Clean signals provide a check that no perturbation can supply on its own:
a latent described in terms of a spectral band should also be describable
in terms of how it behaves on ordinary, unmodified recordings. We
therefore inspected, for each setting, the clean test trials on which
selected latents fire most strongly, together with their Welch spectra
(Appendix A). This inspection did not yield a consistent band-specific
picture that would support assigning spectral names to the latents, and
the exemplar view proved sensitive to which latents were selected and to
their baseline firing rates. High-baseline latents dominate any
exemplar-based summary simply because they are active on more trials.

We take two lessons from this for the ladder. First, observational checks
on clean data should use quantities that were not involved in choosing
the latents, and should condition on baseline activity, for example by
comparing against controls matched in firing rate and mean activation
(Appendix E). Second, top-activating exemplars are a useful qualitative
aid but a weak basis for a label, because they condition on the extreme
of the activation distribution. These checks do not rule out structured
spatial–spectral features; they indicate only that the present analysis
does not reach the rung at which a simple frequency-band reading could be
asserted, and the descriptive response profiles should be read with that
in mind.

Figure 2: A: per-setting mean active-set Jaccard under random and
contiguous channel dropout at early, middle, and late depth, grouped by
backbone. Overlap declines with depth in every backbone–dataset
combination, most steeply for DINOv3. B: per-setting test FVU for the 27
dictionaries, which remains within $[0.059,0.228]$ regardless of depth or
backbone. The 21-setting ablation bar chart is in Appendix D.

## 4 Discussion and limitations

The ladder is an operational reporting protocol, not a universal feature
finder. Table 1 is the intended reading: each rung adds a
measurement and licenses a correspondingly scoped statement. Our results
populate the lower rungs. Reconstruction is uniformly good, which is
necessary but uninformative about code behavior; stability under spatial
perturbation degrades with depth across every backbone–dataset
combination; and spectral responses are best read as part of a code’s
overall sensitivity to input change. Semantic interpretation of
individual latents lies above the rungs we test, and we do not claim that
any latent studied here has, or lacks, a particular physiological meaning.

The depth trend has practical consequences for EEG work. Electrode
dropout and noisy channels are common in real recordings, so a latent
whose activity depends on the exact channel set is fragile in exactly the
way that matters for deployment. That the effect is strongest for the
vision backbone suggests that cross-modal transfer, however effective for
downstream accuracy, should be accompanied by explicit robustness
reporting. More broadly, the same SAE analysis applied to different layers
of the same model can support quite different stability statements, so
layer choice should be reported as a primary experimental variable rather
than an implementation detail. A positive control on another physiological
axis, where ground truth is better constrained, would test how far up the
ladder a well-behaved dictionary can climb; we leave that to a longer
version.

Limits: remaining settings use one SAE seed; groups compared in ablations
are unmatched on baseline activity; $|\Delta f|$ can include TopK rank
displacement; relevance is defined for a logistic probe on mean-pooled
codes; task-readout products cover 21/27 settings (REVE MA/Speech exports
were unavailable); band-stop filtering cannot isolate a biological
oscillator; and clean-data inspection was qualitative. Conclusions concern
the behavior and reporting of fitted SAE latents, not clinical validity or
neural causality.

## 5 Conclusion

Across 27 settings, TopK SAEs on frozen EEG and vision foundation models
reconstruct held-out activations well, but their codes become
progressively less stable under channel dropout as depth increases, most
strongly for the cross-modal backbone. Reconstruction quality therefore
does not imply a stable code, and stability at one layer does not imply
stability at another. Semantic names for SAE features should be preceded
by the measurements in Table 1 and scoped by the rung of
evidence they reach.

#### Ethics statement.

This study reanalyzes publicly available EEG datasets, recruits no
participants, and makes no clinical predictions; consent and ethics
procedures are inherited from the original releases.

## Appendix A Additional experimental details

Layer groups mix consecutive transformer blocks after per-layer $\ell_{2}$
normalization: blocks 0–3 / 4–7 / 8–11 for CBraMod and DINOv3 (12
blocks) and 0–6 / 7–14 / 15–21 for REVE (22 blocks). DINOv3 dictionary
width is reduced to $4\times$ on Mental Arithmetic and $8\times$ on BCI
where token support is limited; all other settings use $16\times$ . Training
uses reconstruction MSE plus AuxK with $k=32$ . Multi-seed analyses retrain
only the BCI middle setting of each backbone.

For clean-data inspection we stream, for a small number of selected
latents per setting, each latent’s top-five clean test activations
together with the corresponding trials. Welch spectra quantify relative
band power on those tails. Because this view conditions on the activation
extreme, it is used qualitatively; correlational summaries over all
trials are the more appropriate primary quantity for any future
observational test.

Activity-matched controls, where used, are taken from the same
dictionary. For each selected latent we choose the unused non-selected
latent minimizing squared $z$ -distance in clean firing rate and mean
activation (features standardized on the union of selected latents and
the candidate pool), sampling without replacement.

## Appendix B Per-backbone raw alpha/sham contrast

Table 3: Per-backbone configuration under which the alpha (8–12 Hz) and
sham (30–34 Hz) band-stops, and all other spectral and spatial
perturbations, are evaluated. The same frozen dictionaries and fixed test
splits are used for every intervention.

| | Blocks | Layer groups | Expansion

| Model | total | early / middle / late | BCI / MA / Speech

| CBraMod | 12 | 0–3 / 4–7 / 8–11 | $16\times$ / $16\times$ / $16\times$

| REVE | 22 | 0–6 / 7–14 / 15–21 | $16\times$ / $16\times$ / $16\times$

| DINOv3 | 12 | 0–3 / 4–7 / 8–11 | $8\times$ / $4\times$ / $16\times$

Both filters are fourth-order zero-phase Butterworth band-stops of equal
width, applied to the raw 200 Hz signal before the backbone. Responses are
summarized per setting by sparse-code cosine, active-set Jaccard, and mean
$|\Delta f|$ , and then pooled by backbone over the nine dataset–depth
settings.

## Appendix C Depth-dependent channel-dropout robustness

Channel dropout progressively destabilizes sparse codes with model depth.
Across all nine model–dataset combinations, late-layer active-set overlap
is lower than early-layer overlap for both random and contiguous dropout
(mean late-minus-early Jaccard $-0.17$ and $-0.20$ ). The depth effect is
strongest for DINOv3 (contiguous $\Delta$ Jaccard $\approx-0.35$ ),
intermediate for CBraMod ( $\approx-0.19$ ), and mild for REVE
( $\approx-0.07$ ). Deeper SAE representations are therefore less stable to
spatial perturbation in both EEG-native and cross-modal backbones: useful
transfer by a vision FM need not bring matching internal robustness.
Contiguous dropout, which removes spatially adjacent channels, is
consistently more damaging than random dropout of the same number of
channels, consistent with representations that pool information over
local scalp neighborhoods. This observation is summarized in the main text
(Table 2) because it is the clearest stability result across the
27 settings.

## Appendix D Full 21-setting ablation chart

Figure 3 is the per-setting frozen-probe ablation
display. Main-text Figure 2 replaces it with a two-panel stability
summary that can be read at workshop scale.

Figure 3: Frozen-probe balanced accuracy change after zeroing
size-matched latent groups, 21 settings. Seed dependence and the
baseline-activity confound are discussed in the main text; the chart is
presented descriptively.

## Appendix E Setting-level bootstrap and activity matching

All intervals reported in the paper are percentile 95% CIs from 10,000
bootstrap resamples of settings. The resampling unit and the number of
settings for each family of quantities are listed below.

| Quantity | Resampling unit | Settings

| active-set Jaccard and code cosine, spatial perturbations | setting | 27

| active-set Jaccard and code cosine, spectral perturbations | setting | 27

| per-latent $|\Delta f|$ summaries | setting | 27

| selectivity and importance summaries | setting | 21

| frozen and retrained group ablations | setting | 21

| activity-matched clean-data comparisons | setting | 27

Resampling settings rather than latents respects the fact that latents
within a dictionary are not independent: they share an encoder, compete
under TopK, and are evaluated on the same trials. Activity matching
follows the procedure in Appendix A and is applied within each
dictionary before pooling.

## Appendix F Licenses, assets, and compute

•

BCI Competition IV 2a: research use; cite
Tangermann et al. [2012]; Graz description
[Brunner et al., 2008].

•

Mental Arithmetic (EEGMAT): CC BY 4.0; hosted on PhysioNet
[Zyma et al., 2019, Goldberger et al., 2000].

•

BCI Competition 2020 Track 3: competition / research use
[Jeong et al., 2022, Lee et al., 2020].

•

CBraMod weights: as released by the authors; reference
implementation BSD-3-Clause [Wang et al., 2025].

•

REVE-base / positions: as released by the authors
[El Ouahidi et al., 2025].

•

DINOv3 ViT-B/16: DINOv3 License (Meta) [Siméoni et al., 2025].

Experiments used NVIDIA A100 GPUs on a shared academic Slurm cluster.
The reported pipeline comprises token-level activation export for 27
settings, training 27 TopK SAEs plus 9 multi-seed BCI-middle retrains,
perturbation sweeps, linear probes on 21 settings, and CPU-side analyses.
We estimate on the order of 300 GPU-hours for the reported experiments.
Exploratory exports and trainings that were not used in the paper are not
included in that figure. An anonymized code archive with training and
analysis scripts is provided as supplementary material; checkpoints and
raw EEG are omitted because of size and are available from the cited
public sources.

## NeurIPS Paper Checklist

1.

Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope?

Answer: [Yes]

Justification: The abstract and introduction state the empirical scope, and Sections 3–4 distinguish measured stability properties from physiological or causal claims.

Guidelines:

•

The answer [N/A] means that the abstract and introduction do not include the claims made in the paper.

•

The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A [No] or [N/A] answer to this question will not be perceived well by the reviewers.

•

The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.

•

It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

2.

Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: Section 4 discusses SAE seeds, TopK competition, baseline-activity differences, linear-probe scope, missing task-analysis settings, qualitative clean-data inspection, and dataset scope.

Guidelines:

•

The answer [N/A] means that the paper has no limitation while the answer [No] means that the paper has limitations, but those are not discussed in the paper.

•

The authors are encouraged to create a separate “Limitations” section in their paper.

•

The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.

•

The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.

•

The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.

•

The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.

•

If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.

•

While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren’t acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

3.

Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [N/A]

Justification: The paper reports empirical analyses and makes no theoretical claims.

Guidelines:

•

The answer [N/A] means that the paper does not include theoretical results.

•

All the theorems, formulas, and proofs in the paper should be numbered and cross-referenced.

•

All assumptions should be clearly stated or referenced in the statement of any theorems.

•

The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.

•

Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.

•

Theorems and Lemmas that the proof relies upon should be properly referenced.

4.

Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: Section 2 gives models, data splits, preprocessing, SAE construction, perturbations, metrics, and probe protocol; the supplementary code contains exact configurations.

Guidelines:

•

The answer [N/A] means that the paper does not include experiments.

•

If the paper includes experiments, a [No] answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.

•

If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.

•

Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.

•

While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example

(a)

If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.

(b)

If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.

(c)

If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).

(d)

We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

5.

Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [Yes]

Justification: Datasets and backbone implementations are public and cited. An anonymized code archive with training and analysis scripts is provided as supplementary material; checkpoints and raw EEG are omitted because of size.

Guidelines:

•

The answer [N/A] means that paper does not include experiments requiring code.

•

Please see the NeurIPS code and data submission guidelines (https://neurips.cc/public/guides/CodeSubmissionPolicy) for more details.

•

While we encourage the release of code and data, we understand that this might not be possible, so [No] is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).

•

The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https://neurips.cc/public/guides/CodeSubmissionPolicy) for more details.

•

The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.

•

The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.

•

At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).

•

Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

6.

Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer) necessary to understand the results?

Answer: [Yes]

Justification: Section 2 reports the split policy, preprocessing, SAE sparsity and expansion, perturbation bands, metrics, and classifier protocol; full training details are in the supplement.

Guidelines:

•

The answer [N/A] means that the paper does not include experiments.

•

The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.

•

The full details can be provided either with the code, in appendix, or as supplemental material.

7.

Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: Intervals for setting-level summaries are percentile 95% confidence intervals from 10,000 bootstrap resamples of settings (27 settings, or 21 for task analyses), as described in Section 2 and Appendix E. The resampling unit is the setting, not the latent. Initialization variability is additionally examined for the BCI middle-layer setting under three SAE seeds.

Guidelines:

•

The answer [N/A] means that the paper does not include experiments.

•

The authors should answer [Yes] if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.

•

The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).

•

The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)

•

The assumptions made should be given (e.g., Normally distributed errors).

•

It should be clear whether the error bar is the standard deviation or the standard error of the mean.

•

It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.

•

For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g., negative error rates).

•

If error bars are reported in tables or plots, the authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

8.

Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes]

Justification: Appendix F reports NVIDIA A100 GPUs on a shared academic Slurm cluster and an estimate of about 300 GPU-hours for the reported pipeline, excluding unused exploratory runs.

Guidelines:

•

The answer [N/A] means that the paper does not include experiments.

•

The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.

•

The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.

•

The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn’t make it into the paper).

9.

Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes]

Justification: The study reanalyzes public research datasets, reports limitations, and makes no clinical or individual-level claims.

Guidelines:

•

The answer [N/A] means that the authors have not reviewed the NeurIPS Code of Ethics.

•

If the authors answer [No] , they should explain the special circumstances that require a deviation from the Code of Ethics.

•

The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

10.

Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: Section 4 motivates more careful validation and reporting of biosignal interpretations and limits the findings to representation auditing rather than clinical or neural causality.

Guidelines:

•

The answer [N/A] means that there is no societal impact of the work performed.

•

If the authors answer [N/A] or [No] , they should explain why their work has no societal impact or why the paper does not address societal impact.

•

Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.

•

The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate Deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.

•

The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.

•

If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

11.

Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pre-trained language models, image generators, or scraped datasets)?

Answer: [N/A]

Justification: The paper releases no new generative model, personal data, or high-risk dataset; it analyzes existing frozen models and aggregate results.

Guidelines:

•

The answer [N/A] means that the paper poses no such risks.

•

Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.

•

Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.

•

We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

12.

Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: Section 2 cites the original dataset releases. Appendix F lists licenses or terms of use for the three datasets and three backbone checkpoints.

Guidelines:

•

The answer [N/A] means that the paper does not use existing assets.

•

The authors should cite the original paper that produced the code package or dataset.

•

The authors should state which version of the asset is used and, if possible, include a URL.

•

The name of the license (e.g., CC-BY 4.0) should be included for each asset.

•

For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.

•

If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.

•

For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.

•

If this information is not available online, the authors are encouraged to reach out to the asset’s creators.

13.

New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [Yes]

Justification: The SAE training and analysis code records configurations, output schemas, random seeds, and run manifests in the supplementary repository.

Guidelines:

•

The answer [N/A] means that the paper does not release new assets.

•

Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.

•

The paper should discuss whether and how consent was obtained from people whose asset is used.

•

At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

14.

Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [N/A]

Justification: This secondary analysis did not recruit participants or run crowdsourcing experiments.

Guidelines:

•

The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.

•

Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.

•

According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

15.

Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [N/A]

Justification: No new human-subject data were collected. Section 2 and Appendix F cite the original public dataset releases.

Guidelines:

•

The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.

•

Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.

•

We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.

•

For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

16.

Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigor, or originality of the research, declaration is not required.

Answer: [N/A]

Justification: LLM assistance was limited to writing and formatting and was not part of the scientific method, so declaration is not required under this question.

Guidelines:

•

The answer [N/A] means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.

•

Please refer to our LLM policy in the NeurIPS handbook for what should or should not be described.