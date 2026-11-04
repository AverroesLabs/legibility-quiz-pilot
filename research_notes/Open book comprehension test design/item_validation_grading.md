# Item validation and reliable grading of free-response answers (small samples, LLM respondents and graders)

Scope note: about 20 tool calls. Several primary sources (Polikoff 2010, Williamson et al. 2012, Brennan 1972, Masters 1982) were not reachable in full text. Their content here comes from secondary sources that cite them, and is flagged that way. Anything taken from background knowledge rather than a fetched page sits under Inferences or Gaps, never under Cited Findings.

## 1. CTT item statistics and IRT for small samples, and IRT with LLMs as respondents

### Takeaway
Classical item statistics (p-value, upper-lower discrimination D, point-biserial) can be computed with tens of respondents and have well-known rule-of-thumb bands. IRT calibration needs hundreds to about 1,000 respondents for stable 2PL parameters, far more than a "ladder" of a few LLM readers can supply. LLM-IRT work such as tinyBenchmarks and metabench fits IRT on hundreds to thousands of models to pick small item subsets. That does not transfer to a bespoke quiz answered by about 5 to 10 readers, so CTT-style statistics plus explicit condition contrasts are the realistic tool.

### Cited Findings
**CTT thresholds**
- Ebel's discrimination-index bands, widely reproduced: D ≥ 0.40 very good; 0.30–0.39 reasonably good but could be improved; 0.20–0.29 marginal, needs revision; < 0.19 poor. Attributed to Ebel (1986) and Hetzel (1997). These are defined on the upper-minus-lower-group index (D = proportion correct in the high group minus proportion correct in the low group), not on point-biserial. — [Metricgate guide](https://metricgate.com/blogs/item-difficulty-vs-discrimination/); [Item analysis slides](https://www.slideshare.net/slideshow/item-analysis-16082143/16082143)
- Difficulty (proportion correct, p): the University of Florida guidance says most items should fall between about 0.25 and 0.75, and calls these "general guidelines, not hard and fast rules". Other sources give 0.30–0.70 as acceptable because D is more likely to be high in that range. A literature review found no agreement on difficulty bands but closer agreement on discrimination bands. — [UF Module 4c](https://teach.ufl.edu/wp-content/uploads/2019/05/Module-4c_-Analyzing-Item-Difficulty-and-Discrimination.pdf); [Relationship between difficulty and discrimination indices](https://revistas.upc.edu.pe/index.php/docencia/article/view/614)
- A common practitioner keep-threshold is point-biserial ≥ 0.20. It comes from a blog, not a validated mapping. — [Metricgate](https://metricgate.com/blogs/item-difficulty-vs-discrimination/)

**IRT sample size**
- No single authoritative minimum N exists. Current practice is simulation-based planning (the Schroeders & Gnambs 2025 tutorial, implemented in the R package `irtsim`). The vignette uses an MSE threshold of 0.05 for difficulty recovery as a default and says that 100 respondents is small for a 10-item 2PL while 1,000 should be ample. — [irtsim vignette: Planning a Sample Size for an IRT Study](https://cloud.r-project.org/web/packages/irtsim/vignettes/irtsim.html); [irtsim: choosing item parameters](https://cloud.r-project.org/web/packages/irtsim/vignettes/choosing-item-parameters.html)
- A 2024 empirical study (REID journal) found that 1,000 participants was the minimum for stable 2PL parameter estimates with EAP estimation. — [REID vol10 iss1 art8](https://scholarhub.uny.ac.id/reid/vol10/iss1/8)

**IRT with LLM respondents**
- tinyBenchmarks (Polo et al., 2024) uses a 2-parameter *multidimensional* IRT model, P(Y_il=1) = 1/(1+exp(−α_iᵀθ_l + β_i)), with the dimension chosen from {2, 5, 10, 15}, fitted by variational inference. It was fitted on 395 LLMs (about 75% used for training) for the Open LLM Leaderboard and MMLU, and on 100 LLMs for AlpacaEval. Anchor items are chosen by K-means on IRT item parameters (or on correctness vectors). The "gp-IRT" estimator blends the sample mean with the IRT prediction: λ = b̂²/(σ̂²/|I| + b̂²). With 100 items per scenario, average error is about 2%, and MMLU stays within 1.9% for recent LLMs. — [tinyBenchmarks HTML](https://arxiv.org/html/2402.14992); [abstract](https://arxiv.org/abs/2402.14992)
- tinyBenchmarks caveats: if the IRT model is misspecified, the bias does not vanish with more data. Severe distribution shift, such as new architectures or rapid capability gains, causes larger errors. Correctness-based anchors degrade on specialized LLMs. The authors recommend refitting periodically with newer models. — [tinyBenchmarks HTML](https://arxiv.org/html/2402.14992)
- metabench (Kipnis et al., 2024) used data from more than 5,000 LLMs over six benchmarks (28,632 items) and distilled them to less than 3% of the items. It reconstructs individual benchmark scores with about 1.24% RMSE and the total score with about 0.58% RMSE. A single common factor correlates r = 0.94 (Spearman) with the total score. — [metabench abstract](https://arxiv.org/abs/2407.12844)
- Fluid Benchmarking (COLM 2025) fits IRT on existing LM evaluation results and selects items adaptively per model, as in computerized adaptive testing. On MMLU it achieves higher validity and lower variance with 50× fewer items. The authors credit the validity gain mostly to IRT and the variance reduction mostly to adaptive selection. — [Fluid Benchmarking abstract](https://arxiv.org/abs/2509.11106)

### Inferences
- With a ladder of about 4 to 8 reader models × a few samples × 2 conditions, every LLM-IRT method above is out of reach, since they calibrate on hundreds to thousands of models. The usable CTT analogue is a per-item "discrimination across the ladder": the correlation between item score and reader strength (rank or an external benchmark score), or a D-like contrast between the top and bottom rungs. Treat each reader as a respondent and average its repeated samples to an item-level proportion.
- The Ebel bands assume large human cohorts. With about 6 readers, a point-biserial has a huge standard error (roughly 1/√(n−1) ≈ 0.45 near r = 0), so it should be a screening heuristic rather than a decision rule. Pooling across repeated samples reduces sampling noise but does not add independent respondents.
- A 1PL/Rasch model is the least data-hungry IRT model, but the fetched sources give no 1PL-specific minimum. Rasch fits on a handful of respondents would mainly regularize item difficulty and say little about discrimination.
- DIF (differential item functioning) maps naturally onto "condition" as the grouping variable. For a reader of the same ability, does the item behave differently when the paper is legible versus illegible? See section 2.

### Gaps
- No fetched source gives a 1PL-specific minimum N, or the classic rules of thumb from de Ayala or Linacre (often quoted as about 100–200 for Rasch and 200–500+ for 2PL). These come from background knowledge only and are unverified.
- No fetched study applies IRT to a ladder of fewer than about 30 LLMs.

## 2. Measuring item sensitivity to an intended manipulation (instructional or treatment sensitivity)

### Takeaway
The instructional-sensitivity literature applies directly. The simplest index is the pretest-posttest difference index, PPDI = p_post − p_pre (Cox & Vargas, 1966). In the legibility setting, "pre/post" becomes "hard-to-read/clear" condition. Polikoff (2010) and Haladyna recommend it because it is simple. Its known weakness is that it cannot separate item quality from treatment strength. DIF-based approaches give relative sensitivity, and the two approaches often disagree.

### Cited Findings
- Polikoff (2010, *EM:IP*) groups instructional-sensitivity methods into three families: (1) expert judgments, (2) instructional measures (e.g., D'Agostino et al., 2007), and (3) item statistics (e.g., Cox & Vargas, 1966; Linn & Harnisch, 1981). This is as summarized by Naumann, Hochweber & Hartig (2014, *JEM* 51(4):381–399). — [Naumann et al. 2014 (pedocs full text)](https://www.pedocs.de/volltexte/2020/18997/pdf/Naumann_et_al_2014_Modeling_instructional_sensitivity_using_LML-DIF_A.pdf)
- PPDI = Difficulty_post − Difficulty_pre (Eq. 1). It is "the difference in item difficulty between instructed and uninstructed students. High absolute PPDI values indicate high instructional sensitivity." Haladyna & Roid (1981) and Polikoff (2010) favour PPDI as "technically easy to implement and conceptually easy to understand". Ruiz-Primo et al. (2012) found experimentally that PPDI was proportional to instructional effects. Selecting items on PPDI does not hurt reliability (Crehan, 1974). — [Naumann et al. 2014](https://www.pedocs.de/volltexte/2020/18997/pdf/Naumann_et_al_2014_Modeling_instructional_sensitivity_using_LML-DIF_A.pdf)
- Naumann et al. computed PPDI on IRT (1PL) difficulty estimates. An item counted as sensitive if |PPDI| exceeded twice the SD of the posttest item-difficulty estimate, which is a concrete significance rule. — [Naumann et al. 2014](https://www.pedocs.de/volltexte/2020/18997/pdf/Naumann_et_al_2014_Modeling_instructional_sensitivity_using_LML-DIF_A.pdf)
- PPDI and DIF disagree in practice. In one data set, PPDI flagged every item as sensitive, but only six items showed non-negligible multilevel DIF variance, and the items with the highest PPDI looked insensitive under DIF. PPDI is an absolute measure and DIF a relative one. PPDI cannot rule out maturation without an untreated control group (Polikoff, 2010). — [Naumann et al. 2014](https://www.pedocs.de/volltexte/2020/18997/pdf/Naumann_et_al_2014_Modeling_instructional_sensitivity_using_LML-DIF_A.pdf)
- A low PPDI is ambiguous: "it is often impossible to say whether low PPDIs are due to poor-quality test question or poor-quality instruction" (citing Polikoff, 2010). — [Journal of Extension 56(1) tt3](https://archives.joe.org/joe/2018february/tt3.php)

### Inferences
- Mapping to the legibility quiz:
  - Per item, per reader: PPDI_item = p(correct | clear version) − p(correct | hard version). Average over the ladder.
  - A good item has PPDI well above zero and a small "ability-only" slope inside the clear condition, or at least a PPDI that does not shrink to zero for weak readers.
  - An item that is easy in both conditions (ceiling) or hard in both (floor) has PPDI ≈ 0 and is uninformative.
  - Because the "treatment" (legibility) is controlled experimentally, the maturation confound that troubles educational PPDI is absent. That makes PPDI cleaner here than in classrooms.
- A DIF-style formulation fits the same data: a logistic model score ~ reader_ability + condition + condition×item. The item-specific condition effect (the interaction) is the sensitivity index, and the ability slope is the ability-sensitivity. That separates "sensitive to manipulation" from "sensitive to ability".
- A control is also needed: a "no-document" (closed-book) condition. It shows items answerable from prior knowledge. Those items will have low PPDI and high baseline p, and should be cut. This plays the role of the untreated control group in Polikoff's critique.
- Brennan's B index, from background knowledge and unverified: B = p(masters) − p(non-masters), where mastery is defined by total-score cut. It is the criterion-referenced analogue of D. The fetched sources gave no definition.

### Gaps
- Brennan's B index definition, and Polikoff's (2010) own recommended thresholds, could not be retrieved from a primary source.
- No source was found that applies instructional-sensitivity indices to LLM respondents.

## 3. Rubric design for short constructed responses and inter-rater reliability

### Takeaway
For automated or LLM grading, analytic point-based rubrics broken into binary criteria give markedly higher agreement than graded or holistic scales. In one benchmark, moving from ternary to binary criteria raised human-judge agreement by about 20 percentage points. The operational agreement standard from automated scoring (Williamson, Xi & Breyer, 2012) is QWK/kappa ≥ 0.70, automated-human agreement no more than 0.10 below human-human agreement, and a standardized mean difference ≤ 0.15.

### Cited Findings
- Williamson, Xi & Breyer (2012), "A Framework for Evaluation and Use of Automated Scoring", *EM:IP* 31(1):2–13, as cited by secondary sources:
  - kappa/QWK ≥ 0.70 for high-stakes use.
  - Automated-human agreement should not degrade by more than 0.10 (kappa or Pearson r) relative to human-human agreement.
  - Standardized mean score difference ≤ 0.15.
  - The framework also asks for evidence on construct fit, associations with external measures, generalizability across tasks and forms, and subgroup effects.
  - ACT applies the rule as "human-CRASE QWK within .10 of human-human QWK."
  - — [ACT CRASE report R2304](https://www.act.org/content/dam/act/unsecured/documents/R2304-CRASE-Essay-Scoring-Model-Performance-02-2023.pdf); [ERIC EJ1109680](https://files.eric.ed.gov/fulltext/EJ1109680.pdf); [Wiley DOI record](https://onlinelibrary-wiley-com-443.webvpn.zafu.edu.cn/doi/10.1111/j.1745-3992.2011.00223.x)
- Binary versus graded criteria: in ResearchRubrics (2025), switching rubric items from ternary to binary grading raised human-LLM-judge agreement by about 20 percentage points, which the authors attribute to partial credit adding ambiguity. — [ResearchRubrics arXiv 2511.07685](https://arxiv.org/pdf/2511.07685)
- A practitioner (non-peer-reviewed) claim: judge-human kappa rises from about 0.4 with a one-line rubric to about 0.78 with anchored 1–5 levels. Practitioners also advise measuring human-human agreement first, because a judge cannot align better than its labelers. — [Koji docs (vendor)](https://www.koji.so/docs/llm-as-a-judge-vs-human-evaluation); [MLflow judge calibration](https://mlflow.org/articles/llm-judge-calibration)

### Inferences
These come from background knowledge and standard formulas, not verified on fetched pages.
- Cohen's kappa: κ = (p_o − p_e)/(1 − p_e).
- Quadratic weighted kappa: QWK = 1 − Σw_ij O_ij / Σw_ij E_ij with w_ij = (i−j)²/(k−1)². Use it for 0–3 total scores.
- ICC(2,1) or ICC(3,1) suits continuous averaged scores.
- Landis & Koch bands: 0.61–0.80 substantial, > 0.80 almost perfect. This scale is referenced in [Step-wise Rubric Rewards](https://arxiv.org/pdf/2605.17291).
- Cohen's kappa is unstable when a rubric point's base rate is extreme (the kappa paradox). Report raw percentage agreement and a prevalence-robust index (Gwet's AC1 or PABAK) next to it. This matters because many rubric points will be passed by nearly all strong readers.
- Rubric points that capture reasoning rather than keywords:
  - Phrase each point as a claim the answer must *assert or entail*, for example "states that X causes Y because Z", not "mentions X".
  - List common acceptable paraphrases and named non-credit near-misses, such as stating the conclusion without the mechanism.
  - Require that the credited claim be consistent with the rest of the answer, so a contradicting answer scores 0.
  - Make points independent so they can be scored as separate binary checks.

### Gaps
- No primary rubric-design study (e.g., Brookhart; Jonsson & Svingby 2007) was fetched. The analytic-versus-holistic reliability evidence for human raters is therefore not cited here.

## 4. Automated short-answer scoring and LLM-as-a-judge for rubric grading

### Takeaway
Strong LLM judges reach about 80% agreement with humans on pairwise preference, matching human-human agreement. On binary rubric items they reach kappa of about 0.64–0.76, versus about 0.82 between humans. Documented biases include position, verbosity (with recent evidence of a *conciseness* preference instead), self-preference linked to self-recognition, and leniency or strictness drift. Mitigations: binary per-criterion checks, reference answers, cross-family graders, position swapping, and calibration against a human-labelled set.

### Cited Findings
- Zheng et al. (2023, MT-Bench / Chatbot Arena): GPT-4 as judge achieves "over 80% agreement, the same level of agreement between humans". It identifies position, verbosity and self-enhancement biases plus limited reasoning ability. — [arXiv 2306.05685](https://arxiv.org/abs/2306.05685)
- Self-preference (Panickssery et al., 2024): GPT-4 and Llama 2 recognize their own outputs with non-trivial accuracy and score their own outputs higher, even when humans rate them equal. Self-recognition ability correlates linearly with the strength of self-preference, and the causal link holds against simple confounders. — [arXiv 2404.13076](https://arxiv.org/abs/2404.13076). Further work: [Self-Preference Bias in LLM-as-a-Judge, arXiv 2410.21819](https://arxiv.org/pdf/2410.21819)
- Surveys catalogue position, verbosity and self-enhancement as core biases, with metrics such as position consistency and preference fairness. Position swapping is the most common mitigation. Length-controlled win rates (Dubois et al., 2024) correct verbosity post hoc. — [LLMs-as-Judges survey, arXiv 2412.05579](https://arxiv.org/pdf/2412.05579); [A Survey on LLM-as-a-Judge, arXiv 2411.15594](https://arxiv.org/pdf/2411.15594)
- "Judging the Judges" (April 2026, single-author preprint):
  - LLM judges show a *conciseness* preference rather than a simple preference for longer answers.
  - Position swapping reduces style bias but can *increase* verbosity bias, so mitigations interact.
  - — [arXiv 2604.23178](https://arxiv.org/html/2604.23178)
- BLUEX v2 (2026), open-ended exam answers graded with a binary rubric-item protocol:
  - Human-human κ = 0.82, which the authors treat as the practical ceiling.
  - The LLM judge reached κ = 0.64–0.69 against humans and was lenient, marking criteria satisfied more often than humans did.
  - — [arXiv 2606.22723](https://arxiv.org/pdf/2606.22723)
- Step-wise rubric study: an LLM judge and two human authors agreed on 89.3% of 664 binary rubric items, κ = 0.76 ("substantial", Landis–Koch). — [arXiv 2605.17291](https://arxiv.org/pdf/2605.17291)
- Direction of bias varies by setting. Another study found LLM judges systematically *stricter* than experts on rubric satisfaction, the opposite of BLUEX's leniency. So graders must be calibrated per task. — summarized in search results ([ResearchRubrics arXiv 2511.07685](https://arxiv.org/pdf/2511.07685); [BLUEX v2](https://arxiv.org/pdf/2606.22723))

### Inferences
- For 2–3 rubric points per item:
  - Score each point as a separate binary call, with the paper excerpt and a reference answer in context.
  - Ask for a quoted span from the response as evidence before the verdict. This reduces both keyword-matching and leniency.
  - Sum the binary calls into the item score.
- Self-preference is a live risk when a ladder reader and the grader come from the same model family. Use a grader from a different family, or average two graders from different families, and check for a family × grader interaction.
- Calibration set:
  - Hand-grade about 50–100 responses per item type, spread across ladder rungs and both conditions. Compute grader-human κ per rubric point and compare it with human-human κ, applying the Williamson rule of degradation ≤ 0.10.
  - Also check that the grader's error is *not correlated with condition*. A grader that is more lenient on answers written from the clear version would inflate the apparent legibility effect. This differential-bias check matters more for the experiment than overall κ does.
- Strip or normalize length, for example by capping answer length in the instructions. Verbosity and conciseness biases go both ways, and a manipulation may change answer length.

### Gaps
- No fetched 2024–2026 study reports LLM-grader agreement specifically on the SemEval-2013 / ASAP-SAS short-answer datasets. Numbers there would be the closest analogue to this use case.
- The mitigation numbers from Zheng et al. (swap-and-average, few-shot, chain-of-thought effects) were not extracted.

## 5. Two-tier items and partial-credit models

### Takeaway
Two-tier items, an answer tier plus a reason tier (Treagust, 1988), are conventionally scored correct only if both tiers are correct. The response combinations also support a four-category diagnosis. Polytomous IRT models (Masters' Partial Credit Model, Graded Response Model) are used to give partial credit. Rasch analyses suggest the reason tier measures a related but distinct trait and is often harder.

### Cited Findings
- Standard scoring: an item counts as correct only if both the content choice and the reason are correct. — [core.ac.uk study](https://core.ac.uk/download/195657903.pdf); [Springer, Two-tier MCQs](https://link.springer.com/article/10.1186/s41029-015-0005-x)
- Treagust's (1988) two-tier structure pairs a content question with a reason set. The combination of responses sorts students into four diagnostic categories. — [JPPIPA article](https://jppipa.unram.ac.id/index.php/jppipa/article/download/12305/9077/78836)
- For two-tier diagnostic tests, PCM, GRM and a modified GRM are commonly applied scoring models. — [JPPIPA (PhyTestApp 2025)](https://jppipa.unram.ac.id/index.php/jppipa/article/download/12305/9077/78836)
- "Is it harder to know or to reason?" applies Rasch to each tier separately, treating them as related but distinct traits (knowing versus reasoning). — [Curtin espace record](https://espace.curtin.edu.au/handle/20.500.11937/43349?show=full)

### Inferences
- In the quiz, the reason tier is where comprehension of the *argument* shows, as opposed to retrieval of a fact. Scoring answer and reason as separate rubric points (0/1 each), plus an optional consistency point, is a free-response two-tier design.
- Four diagnostic categories:
  - Right answer, right reason: understood.
  - Right answer, wrong reason: guessing or surface retrieval.
  - Wrong answer, right reason: slip.
  - Wrong answer, wrong reason: misconception or did not understand.
- The "right answer, wrong reason" rate per condition may itself be the most legibility-sensitive statistic.
- Partial credit gives more score variance per item, and so more power per item, than a both-correct rule. The ResearchRubrics finding, though, suggests keeping each *grading decision* binary and getting partial credit by summing binary points, rather than asking the grader for graded judgments.

### Gaps
- Masters' (1982) PCM formulation was not retrieved, so no source here covers how PCM category thresholds map onto tier combinations.

## 6. Items and samples needed to detect a mean-score difference, and handling LLM sampling noise

### Takeaway
Miller (2024) gives closed-form tools: CLT standard errors, clustered SEs for grouped items, paired differences, variance reduction from K resamples per question, and a sample-size formula n = (z_{α/2}+z_β)²(ω² + σ_A²/K_A + σ_B²/K_B)/δ². Repeated samples only remove the within-item sampling term. The between-item term ω² is fixed per item, so adding items matters more than adding samples once K is a few.

### Cited Findings
All from Evan Miller, "Adding Error Bars to Evals" (2024) — [arXiv HTML 2411.00640v1](https://arxiv.org/html/2411.00640v1)
- SE_CLT = √(Var(s)/n). Bernoulli case: √(s̄(1−s̄)/n). 95% CI = s̄ ± 1.96·SE.
- Clustered SE, for items grouped by paper or section: SE_clustered = (SE_CLT² + (1/n²)Σ_c Σ_i Σ_{j≠i}(s_ic − s̄)(s_jc − s̄))^{1/2}.
- Variance decomposition: Var(μ̂) = (Var(x) + E[σ_i²])/n, where x is the item's true mean score and σ_i² the within-item sampling variance.
- Resampling K times gives Var(s_i) = σ_i²/K. In the uniform-difficulty example, Var(μ̂ | K) = Var(μ̂ | K=1)·(1 + 2/K)/3. The gain therefore saturates at 1/3 of the K=1 variance, and most of it comes from K up to about 4 to 8.
- Paired comparison, the same items under both conditions: SE_paired = √(Var(s_A − s_B)/n) = √(SE_A² + SE_B² − 2·SE_A·SE_B·Corr(s_A,s_B)). Paired SEs can also be clustered.
- Sample size: n = (z_{α/2} + z_β)²(ω² + σ_A²/K_A + σ_B²/K_B)/δ², with ω² = Var(x_A) + Var(x_B) − 2Cov(x_A,x_B).
- Minimum detectable effect: δ = (z_{α/2}+z_β)·√((ω² + σ_A²/K_A + σ_B²/K_B)/n).
- Worked example: α = 0.05 and power 0.80 (z sum ≈ 2.80), δ = 0.03, ω² = 1/9, no sampling noise, gives n ≈ 969 items.
- Recommendations: use CLT SEs rather than bootstrap; cluster SEs when questions are grouped; reduce variance by resampling or by using token probabilities; analyse paired question-level differences; run a power analysis before building the eval.

### Inferences
- Worked numbers for this design, using Miller's Eq. 10 rearranged:
  - Item scores are on a 0–1 scale (rubric points / max).
  - The legibility effect for a given reader is δ = 0.10–0.15.
  - The between-item SD of the condition *difference* is ω ≈ 0.25–0.35 (ω² ≈ 0.06–0.12).
  - Then n ≈ 7.84 × 0.09/0.01 ≈ 70 items for δ = 0.10, or about 30 items for δ = 0.15, per reader, before sampling noise.
  - K = 3–5 samples per item per condition shrinks the σ²/K terms.
  - These ω values are assumptions. Estimate ω² from a pilot.
- Pooling readers in a mixed model (score ~ condition + (1|item) + (1|reader) + (condition|reader)) uses all rungs, and lets the analysis ask whether the legibility effect grows or shrinks with reader strength.
- Temperature:
  - At temperature 0, or with greedy decoding, σ_i² ≈ 0 but item-level variance stays. Determinism also means "repeated samples" add nothing.
  - At temperature > 0, average K samples into an item-level proportion. Treat the item, not the sample, as the unit of analysis to avoid pseudo-replication; samples of the same item are clustered.
  - Grader noise adds another variance component. Re-grading each response 2–3 times (or with 2 graders) and averaging reduces it in the same σ²/K way.
- Counterbalancing: if the same reader sees both versions of a paper, they are separate API calls with no memory, so carryover is not an issue. Order effects within the prompt still apply.

### Gaps
- No source was found that estimates typical ω² (between-item variance in condition effects) for reading-comprehension items answered by LLMs. It must come from a pilot.
- No fetched work gives power analysis specifically for ladders of LLM readers, or for multilevel designs with readers as a random factor.
