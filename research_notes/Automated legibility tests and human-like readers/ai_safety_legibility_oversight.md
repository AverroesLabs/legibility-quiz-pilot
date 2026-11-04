# AI safety and scalable oversight work on the legibility of AI outputs to humans

Scope note: written 2026-11-04. These notes come from about 22 search and fetch calls. Pages marked "(opened)" were fetched. Pages marked "(search snippet only)" were seen only through search-result summaries and should be checked before anyone relies on them. Empirical results and proposals are labelled separately.

## Q1. Prover-verifier games and checkability: how was legibility measured, did small verifiers stand in for humans, and how well did they track human time-limited checkers?

### Takeaway
Kirchner et al. (2024) remains the only PVG work found that validates legibility against humans. It measured legibility as the accuracy and speed of contractors judging grade-school math (GSM) solutions under a 45-second limit. It shows that legibility to small verifiers transferred to these humans as a direction of change (accuracy rose over rounds). It does not report a quantitative correlation between verifier scores and human judgments. The main 2026 follow-up, Kim and Lee's decoupled PVG (the "translator" paper), removes the legibility tax on paper but ran no human evaluation at all.

### Cited Findings
**Empirical: Kirchner et al. 2024, "Prover-Verifier Games improve legibility of LLM outputs" (OpenAI; arXiv 2407.13692)**
- Training setup: small verifiers learn to predict solution correctness. A "helpful" prover is rewarded for correct solutions the verifier accepts, and a "sneaky" prover for incorrect solutions that fool the verifier. Task: GSM math. — [arXiv abstract (opened)](https://arxiv.org/abs/2407.13692)
- Human measure of legibility: contractors had a strict 45-second limit to judge each solution correct or incorrect. Legibility = their accuracy and time. The authors call this "one of many possible measures." — [arXiv HTML v2 (opened)](https://arxiv.org/html/2407.13692v2)
- Main human campaign: 75 trainers were randomized to samples, and 4 were excluded as indistinguishable from chance, leaving 71. In total 15,983 datapoints, a 45-second timer, payment per completed item, no correctness feedback, at most 4 hours per trainer. Each prover contributed at most one correct and one incorrect solution per problem. Problems lacking both were dropped, about one-third of problems per round. — [arXiv HTML v2, Appendix I (opened)](https://arxiv.org/html/2407.13692v2)
- A pilot campaign (10 trainers, 1-minute limit, solutions shown in round order over about a month) is flagged by the authors as possibly confounded by learning or fatigue. The sneaky-prover human results and the "goodharting" human results come from this flawed pilot. — [arXiv HTML v2 (opened)](https://arxiv.org/html/2407.13692v2)
- Qualitative results:
  - Correctness-only RL raised accuracy, but human accuracy and speed "deteriorated noticeably" on its solutions, most of all on hard problems.
  - On helpful-prover solutions, human accuracy and checking time improved across rounds, ending roughly at the legibility of the initialization.
  - On sneaky-prover solutions, human accuracy started "comparatively high" and fell over rounds.
  - With goodharting training (verifier about 10^-3 of prover size), human accuracy showed no significant increase over rounds 0-2.
  - Exact values appear only in figures (Figs 4, 7, 8, S5c, S6). — [arXiv HTML v2 (opened)](https://arxiv.org/html/2407.13692v2)
- Verifier size: the main runs use a verifier with about 10^-3 of the prover's pretraining compute.
  - The stable range was 10^-3 to 10^-1.8.
  - At 10^-4, training was unstable and provers were weak and poorly legible.
  - At equal size (10^0), verifiers did not become more robust, and provers were strong but poorly legible. — [arXiv HTML v2 (opened)](https://arxiv.org/html/2407.13692v2)
- Legibility tax: correctness-only training gives the highest accuracy and the lowest legibility, and checkability training lands in between. Kim and Lee 2026 summarise the size of Kirchner's gap as about 60% vs about 80% accuracy. — [Kirchner HTML (opened)](https://arxiv.org/html/2407.13692v2); [Kim & Lee HTML (opened)](https://arxiv.org/html/2602.23248v1)
- Authors' caveats:
  - The transfer of small-verifier legibility to humans is "a surprising empirical finding," not a guarantee.
  - Only one dataset (GSM) was used.
  - Bad results at extreme verifier sizes might come from hyperparameters.
  - The human studies have design flaws.
  - No direct comparison of verifier accuracy with human accuracy, and no correlation between them, appears in the text I read. — [arXiv HTML v2 (opened)](https://arxiv.org/html/2407.13692v2)

**Empirical: Kim & Lee 2026, "Mitigating Legibility Tax with Decoupled Prover-Verifier Games" (KAIST; arXiv 2602.23248; ICLR 2026)**
- Method: a fixed solver handles correctness, and a separate "translator" rewrites the solver's solution into checkable form. A verifier and a sneaky translator play the game. Theorem 1: equilibria correspond to faithful and checkable translators. — [arXiv HTML (opened)](https://arxiv.org/html/2602.23248v1); [ICLR page (search snippet only)](https://iclr.cc/virtual/2026/10019174)
- Setup:
  - Dataset: GSM8K-Aug, 8K problems for the verifier and 8K for the solver and translators, with a 1,024-problem test set.
  - Verifier: Qwen2-1.5B. A 0.5B verifier gave "limited qualitative improvements."
  - Solver and translators: Llama-2-7b-chat. — [arXiv HTML (opened)](https://arxiv.org/html/2602.23248v1)
- Test accuracy:
  - Base model: 9.6%
  - Trained solver: 57.0%
  - Decoupled PVG: 56.9%
  - Reproduced PVG baseline: 22.3%, a 34.7-point tax
- "Faithfulness" means only that the translator's final answer matches the solver's: 99.8% on the test set. — [arXiv HTML (opened)](https://arxiv.org/html/2602.23248v1)
- No direct checkability metric was reported, only the average verifier logit. The authors state they "did not conduct human experiments that measure legibility." The legibility evidence is qualitative samples from a single problem. The PVG baseline departs from the original in several ways (RLOO instead of PPO, cross-entropy loss, the sneaky prover is given the answer, among others). — [arXiv HTML (opened)](https://arxiv.org/html/2602.23248v1)

### Inferences
- The only human validation in the PVG line is a single GSM study with 45-second binary judgments. Quick binary checking of short arithmetic is a very different construct from comprehending a research paper. Nothing found validates small-verifier legibility as a proxy for humans on long, technical, open-ended text.
- Kim and Lee's notion of faithfulness (same final answer) is far weaker than the faithfulness a legibility audit of research needs, where every claim should be preserved.
- The weak-verifier sweet spot (10^-3 to 10^-1.8 of prover compute) hints that a reader model must be much weaker than the writer to put pressure on legibility. A matched-size verifier gave strong but illegible provers. This fits the pilot finding that capable LLM readers are insensitive to prose that is hard for humans.

### Gaps
- No per-round human accuracy numbers could be pulled from the text; they exist only in figures. No verifier-vs-human agreement statistic was found.
- I found no 2025-2026 PVG follow-up with a new human checkability study. A search for "prover-verifier games legibility follow-up 2025 human evaluation" turned up only Kim and Lee.

## Q2. Translators, legible chain of thought, monitorability, and legibility metrics for reasoning traces: what metrics exist and how were they validated against humans?

### Takeaway
The leading metric of legibility to models is Roytburg, Sridhar and Ippolito's "transfer utility": how much a weak model gains from prefixes of a strong model's trace. It was validated against downstream weak-model monitoring, not against humans. The authors state that human legibility was not measured. They find that efficiency or brevity metrics conflict with transfer utility: longer, more thorough traces transfer better. CoT-illegibility work (arXiv 2510.27338) finds that most RL-trained reasoning models produce partly illegible CoT. Elsewhere in software engineering, adding verifier signals did not improve prediction of human-rated code comprehensibility.

### Cited Findings
**Empirical: Roytburg, Sridhar, Ippolito 2026, "Measuring Weak-to-Strong Legibility of Reasoning Models" (v1 title: "Measuring Reasoning Trace Legibility: Can Those Who Understand Teach?"; arXiv 2603.20508; listed at ICML 2026)**
- Metrics. Transfer utility (TU) gives a weak model a 2%-100% prefix of a strong model's trace, in bins:
  - FOTU = mean weak-model accuracy across bins.
  - SOTU = normalised entropy of where the weak model first answers correctly.
  - Regression rate = how often more steps lower weak-model correctness.
  - Efficiency metrics: token and step length, semantic redundancy (sentence-embedding cosine above 0.8), and backtracking counted by an LLM judge. — [arXiv HTML v2 (opened)](https://arxiv.org/html/2603.20508v2); [ICML 2026 notes (search snippet only)](https://en.papernotes.org/ICML2026/llm_reasoning/measuring_weak-to-strong_legibility_of_reasoning_models/)
- Scale: about 84-85K traces from 12 open-weight reasoning models.
  - Datasets: MATH (5,000 problems), GPQA (448), LSAT (1,598).
  - Weak readers: Phi-3-Mini and Llama-3.2-1B.
  - Proprietary models were excluded because their APIs return summaries. — [arXiv HTML v2 (opened)](https://arxiv.org/html/2603.20508v2)
- Results:
  - Accuracy vs transfer: Spearman ρ = −0.35. GPT-OSS-120B (81% accuracy) ranks 11th of 12 on FOTU.
  - Length vs transfer: ρ = +0.68. Concise models (e.g. Gemma-3-12B, mean 686 tokens) rank low. Verbose models (DeepSeek-R1, Kimi-K2, QwQ-32B) take the top 3.
  - FOTU vs length-residualised FOTU: ρ = +0.728.
  - Rank stability: ρ ≈ 0.91 across weak models and ≈ 0.81 across tasks.
  - No model is top-3 on all six metrics. — [arXiv HTML v2 (opened)](https://arxiv.org/html/2603.20508v2)
- Brevity finding: existing efficiency metrics reward conciseness and miss "thoroughness." The paper describes a legibility-efficiency Pareto frontier. — [arXiv HTML v2 (opened)](https://arxiv.org/html/2603.20508v2); [search summary of v2 abstract](https://arxiv.org/pdf/2603.20508v2)
- Downstream validation is weak-model monitoring, not humans:
  - On MATH, teacher FOTU predicts a Phi-3-Mini monitor's precision at flagging incorrect traces: ρ = +0.900 with sabotage, +0.873 with natural errors, and the signal survives length control.
  - On LSAT: ρ = +0.818 for precision under sabotage; recall is not predicted.
  - On GPQA: no signal survives length control. The authors attribute this to factual knowledge that small models lack. — [arXiv HTML v2 (opened)](https://arxiv.org/html/2603.20508v2)
- Reward models: FOTU vs RM score is about 0 at trace level. At model level ρ ≈ −0.26 to −0.30, shrinking to about −0.08 once accuracy is controlled. So current RMs do not reward legibility. — [arXiv HTML v2 (opened)](https://arxiv.org/html/2603.20508v2)
- Human validation is limited to sub-metrics: 2 annotators labelled 60 traces for backtracking (ρ = 0.717 with the LLM judge, κ = 0.712), and the redundancy threshold was checked by hand on 100 traces per dataset. The authors state that legibility to humans was not measured and "may differ from legibility to models." — [arXiv HTML v2 (opened)](https://arxiv.org/html/2603.20508v2)

**Empirical: "Reasoning Models Sometimes Output Illegible Chains of Thought" (arXiv 2510.27338, Oct 2025)**
- 14 reasoning models trained with outcome-based RL. All except Claude produced illegible reasoning.
- Accuracy fell 53% when models were restricted to the legible parts of their CoT; the abstract does not say whether this is relative or absolute.
- Under resampling, legibility showed no correlation with performance, and legibility degrades on harder questions.
- The abstract does not describe the legibility grader or any human validation. — [arXiv abstract (opened)](https://arxiv.org/abs/2510.27338); [LessWrong/GreaterWrong discussion (search snippet only)](https://www.greaterwrong.com/posts/GKyyYCs8n2goDcAe2/reasoning-models-sometimes-output-illegible-chains-of/comment/ocdBNdi85P7SYGk28)

**Empirical, adjacent (software engineering): De Silva, Kellogg, Chaparro 2026, "Verifier Warnings Do Not Improve Comprehensibility Prediction" (arXiv 2604.22653)**
- Adding the sum of formal-verifier warnings to ML predictors of human-judged code comprehensibility gave "no significant difference" in performance.
- Syntactic and developer features alone predict about as well, even though the warnings correlate with comprehensibility. — [arXiv abstract (opened)](https://arxiv.org/abs/2604.22653)

**Proposal or norm: OpenAI Model Spec (2026-10-29)** says the model should make its actions, uncertainty and mistakes legible to authorized overseers. This is a norm with no metric. — [Model Spec (search snippet only)](https://model-spec.openai.com/2026-10-29.html)

### Inferences
- Transfer utility is the closest existing analogue to the pilot's "weak-model reader" design, and it shares the same open problem: no human validation. Its length correlation (ρ = +0.68) is the reverse of the brevity bias in efficiency metrics. Either metric family can be gamed through length, so a legibility test should report length-residualised scores, as Roytburg et al. do.
- The GPQA null result (no signal after length control) is directly relevant. When reading needs domain knowledge the weak reader lacks, weak-reader metrics stop tracking anything. Research papers sit closer to GPQA than to MATH.
- The code-comprehensibility result echoes the pilot finding that predictors of human understandability tend to collapse onto surface features.

### Gaps
- I found no reasoning-trace legibility metric that has been validated against human comprehension or human checking time, apart from Kirchner's GSM study.
- I did not open Korbak et al. 2025 ("CoT monitorability") or OpenAI and Anthropic work on CoT faithfulness, so no numbers from them are included.
- The grader used by 2510.27338 is unknown because only the abstract was read.

## Q3. Oversight of automated AI research (AI scientists, aiXiv, Agents4Science, AI reviewers, automated alignment researchers): do any measure or require human understandability?

### Takeaway
Two 2026 works treat demonstrated human understanding as the object of measurement, and both test humans, not the documents. Comprehension Audits (Bodkin, Sokhansanj, Hadfield; arXiv 2610.10064) is a governance proposal: Socratic oral audits of lab staff, with no LLM-generated quiz. greCAPTCHA (Payan, Gyevnár, Kasirzadeh, Shah; arXiv 2609.20481) is the closest existing system to an LLM-written quiz. GPT-5.6 Sol writes 8 questions per manuscript, and an automated grader scores them. It separated authors from non-authors with AUC 0.90, but non-authors with five minutes of reading could sometimes match authors. AI-scientist venues (Agents4Science) used LLM reviewers but did not measure human legibility.

### Cited Findings
**Proposal with supporting data: Comprehension Audits (Bodkin [Independent], Sokhansanj [Institute for Law and AI], Hadfield [Johns Hopkins]; arXiv 2610.10064, 7 Oct 2026)**
- Mechanism:
  - Independent auditors embedded at labs, or an internal audit team, with an audit review board overseeing them.
  - Auditors pick contributions by importance, novelty, risk metrics or random sampling.
  - Audits are unannounced: an approximately 90-minute, non-adversarial Socratic oral review.
  - Contributors may use tools to navigate but not to generate answers.
  - Grading: three tiers (deep, familiarity, limited) across five areas: architecture and design, implementation, process, key findings, analysis of results.
  - Outcomes: Pass, Conditional Pass or Failure. Failure pauses dependent releases until remediated, and repeated failures escalate to suspended release authority. — [arXiv HTML (opened)](https://arxiv.org/html/2610.10064v1); [abstract (search snippet)](https://arxiv.org/abs/2610.10064)
- Rehearsed or AI-generated answers do not count. Understanding that existed when the work was approved counts more than understanding remediated for the audit. Auditors may use AI for discovery and navigation, but grading must rest on their own understanding. — [arXiv HTML (opened)](https://arxiv.org/html/2610.10064v1)
- Sampling math (Table 3): at a 25% failure rate, 17 audits per quarter give 99% detection odds. — [arXiv HTML (opened)](https://arxiv.org/html/2610.10064v1)
- Empirical repo analysis:
  - Scope: 224 organisations, 1,278 repos from 179 AI developers plus 857 AI-topic repos from 45 larger organisations; 54K-162K merged PRs per quarter, from 2024 through Q2 2026.
  - Traceable AI use appears in 17% of PRs, treated as a floor.
  - Median code submitted per regular contributor rose 183%, and 202% for those using AI.
  - Comments per kLOC fell about 50% from the Q1 2024 baseline. By June 2026 AI PRs were at 36% of baseline and non-AI PRs at 46%.
  - Review rate for "fleet operator" accounts (300+ PRs per quarter, 42 accounts) fell below 30%, from about 50%.
  - Only 3-7% of fleet PRs had inline comments, vs 20-26% of other PRs. — [arXiv HTML (opened)](https://arxiv.org/html/2610.10064v1)
- The paper does not measure legibility of AI output directly. It proposes five attention-proxy metrics (specification, interaction depth, inspection, documentation, review) and collected only the review metrics. It notes that no frontier lab publishes documentation-fidelity metrics. — [arXiv HTML (opened)](https://arxiv.org/html/2610.10064v1)
- Stated open problems:
  - Rubrics are immature and untested.
  - Few independent auditors exist.
  - There is no threshold for "degraded comprehension."
  - A pilot could not by itself separate high understanding from an ineffective instrument. — [arXiv HTML (opened)](https://arxiv.org/html/2610.10064v1)

**Empirical system: greCAPTCHA (Justin Payan, Bálint Gyevnár, Atoosa Kasirzadeh, Nihar B. Shah; arXiv 2609.20481, 17 Sep 2026; cs.DL/AI/HC)**
- A proctored assessment of the "capacity to verify" a manuscript.
  - An LLM (GPT-5.6 Sol) generates questions and rubrics from the manuscript and the stated contributions.
  - Each exam has 8 questions in 4 families: planted-error (multiple choice), unstated rationale, background knowledge, failure mode. The families are mapped to Bloom's revised taxonomy.
  - An automated grader scores answers against the rubrics. — [arXiv HTML (opened)](https://arxiv.org/html/2609.20481); [abstract (opened)](https://arxiv.org/abs/2609.20481)
- Study design:
  - 31 researchers, each in a one-hour in-person session plus a 20-minute interview.
  - Within subjects: their own paper vs an unfamiliar arXiv paper, order counterbalanced.
  - Between subjects: the unfamiliar paper was in-field or out-of-field.
  - 15 minutes per exam. — [arXiv HTML (opened)](https://arxiv.org/html/2609.20481)
- Results:
  - Mean score 49.7 on own paper vs 27.2 on the unfamiliar paper.
  - Authorship AUC 0.90.
  - Planted-error family alone: AUC 0.595, so it was mostly searchable with Ctrl-F. Excluding it gave AUC 0.934.
  - Best false-positive rate about 20%.
  - In-field vs out-of-field unfamiliar papers did not differ (p = 0.437).
  - Background knowledge was the only family where field expertise helped. — [arXiv HTML (opened)](https://arxiv.org/html/2609.20481)
- Open-book behaviour:
  - Participants evidently could search the paper.
  - One non-author scored 50.6 on an unfamiliar paper after about 5 minutes of preparation, vs 53.1 on their own paper.
  - One participant who used an AI summary of the unfamiliar paper scored 1.3.
  - Participants called the grader overly rigid. No grader-human agreement figure was seen in the text read. — [arXiv HTML (opened)](https://arxiv.org/html/2609.20481)
- Interviews: 16 participants favoured deployment, 12 favoured deployment after significant changes, and 7 opposed. — [search snippet of PDF](https://arxiv.org/pdf/2609.20481)

**Empirical: Agents4Science 2025 (arXiv 2511.15534)**
- The first venue with AI agents as primary authors and reviewers.
- 315 submissions and 48 accepted. Press reports differ: 48 of 314 (Science News), 47 (SCMP).
- Three LLM reviewers were calibrated on ICLR 2022 and 2025 decisions, with a mean pairwise Pearson r of 0.48.
- Reported failure modes: sycophancy, hallucination, fabricated sources.
- No human-legibility or human-comprehension measure was reported in the material seen. — [arXiv PDF (search snippet only)](https://arxiv.org/pdf/2511.15534v1); [Moonlight review (search snippet only)](https://www.themoonlight.io/review/exploring-the-use-of-ai-authors-and-reviewers-at-agents4science); [Science News (search snippet only)](https://www.sciencenews.org/article/science-conference-test-ai-agents)

**Position paper (proposal): "Human-Centric AI Requires a Minimum Viable Level of Human Understanding" (arXiv 2602.00854, Jan 2026)**
- Defines a "Capability-Comprehension Gap": assisted performance improves while users' mental models deteriorate.
- Proposes a "Cognitive Integrity Threshold" as the minimum comprehension needed to preserve oversight. — [search snippet only](https://www.arxiv.org/abs/2602.00854)

### Inferences
- greCAPTCHA is structurally almost the same as the planned test: a strong LLM writes the quiz, and the readers are humans. Its lessons transfer directly:
  1. Multiple-choice "planted-error" items are retrievable open-book and do not discriminate (AUC 0.595).
  2. Questions on unstated rationale and failure modes discriminate best.
  3. Rubric-based auto-grading is seen as rigid.
  4. Five minutes of open-book reading can sometimes close the gap with authors.
- Point 4 suggests that open-book quizzes partly measure retrieval, not comprehension.
- Neither Comprehension Audits nor greCAPTCHA scores the legibility of the document itself. Both test whether a person understands it. The pilot's construct (is the document legible to a non-author reader?) is the reverse direction. greCAPTCHA's non-author condition is the closest human data point to it.
- I found nothing showing AI-scientist systems or AI-review venues measuring or requiring human understandability of outputs.

### Gaps
- aiXiv, automated alignment researcher proposals (OpenAI superalignment, Anthropic, Redwood), and Sakana AI Scientist v2 were not opened, for lack of budget. No claims are made about them.
- No grader-human agreement figure for greCAPTCHA, and no per-family distributions (Appendix D was not in the text read).

## Q4. Weak-to-strong and sandwiching experiments using weaker models or non-experts as proxies for human understanding: when are weak models good proxies for humans?

### Takeaway
Direct evidence that weak models proxy human comprehension is thin. Kirchner shows that improvements in legibility to small verifiers moved human checking accuracy in the same direction on GSM. Roytburg et al. show weak-model rankings are stable across weak models (ρ ≈ 0.91) but break down when the task needs knowledge the weak model lacks (GPQA). Sandwiching (Bowman et al. 2022) used non-expert humans, not weak models. No found study measures the agreement between weak-model readers and human readers on the same documents.

### Cited Findings
- Kirchner et al.: legibility to a verifier about 10^-3 of prover compute transferred to time-limited humans on GSM. Matched-size verifiers did not induce legibility. The authors call the transfer "surprising" and limit it to GSM. — [arXiv HTML v2 (opened)](https://arxiv.org/html/2407.13692v2)
- Roytburg et al.:
  - TU rankings agree across the two weak readers (ρ ≈ 0.91) and across tasks (≈ 0.81).
  - TU predicts weak-monitor precision on MATH (ρ ≈ 0.9) but shows no length-controlled signal on knowledge-heavy GPQA.
  - Humans were not tested. — [arXiv HTML v2 (opened)](https://arxiv.org/html/2603.20508v2)
- Sandwiching (Bowman et al. 2022, arXiv 2211.03540): non-experts chatting with an unreliable LLM assistant beat both the model alone and their own unaided performance.
  - Figures from a secondary aggregator: MMLU 75.4% (team) vs 57.2% (human) vs 57.2% (model); QuALITY 76.8% vs 48.6% vs 59.2%.
  - These figures are unverified against the original. — [search summary of arXiv 2211.03540 (not opened)](https://arxiv.org/pdf/2211.03540)
- "Debate Helps Weak Judges Reward Stronger Models" (arXiv 2605.27483, 2026): the authors note that every protocol's rates decline as the overseen system grows stronger. — [search snippet only](https://arxiv.org/pdf/2605.27483)

### Inferences
- From what was found, weak models look like plausible proxies when:
  - the content is self-contained (math, logic), where the reader needs no outside knowledge, and
  - the measure is directional (did legibility improve?), not absolute agreement with humans.
- They look poor proxies when understanding needs domain knowledge, as on GPQA.
- This partly explains the pilot result. LLM readers bring their own knowledge to bear and so look past prose problems that block humans. Plausible fixes include knowledge-restricted readers (much smaller models, or readers told to use only the text) and calibrating weak-reader scores against a human panel.
- I found no theory predicting when weak-model and human comprehension diverge. This is an open problem.

### Gaps
- Burns et al. 2023 (weak-to-strong generalization, arXiv 2312.09390) and later W2S papers were not opened. Their PGR results concern supervision, not human comprehension, and were not checked.
- No study found correlating weak-LLM reader accuracy with human reader accuracy on the same long technical documents.

## Q5. Gradual disempowerment and epistemics literature on keeping science comprehensible to humans: any concrete measurement proposals?

### Takeaway
The gradual-disempowerment literature argues conceptually that loss of human understanding is a route to losing control. I found no concrete metric for whether science stays comprehensible to humans. The most concrete measurement-like proposals are Comprehension Audits' attention-proxy metrics and audit pass rates, plus greCAPTCHA scores.

### Cited Findings
- Kulveit, Douglas, Ammann, Turan, Krueger, Duvenaud, "Gradual Disempowerment" (arXiv 2501.16946, Jan 2025). It argues that incremental replacement of human labour and cognition erodes both explicit control and implicit alignment, possibly irreversibly. The abstract calls for research and governance in general terms. No comprehension metrics appear in the abstract. — [arXiv abstract (opened)](https://arxiv.org/abs/2501.16946)
- EA Forum, "Epistemic erosion" (proposal or opinion): argues that gradual disempowerment acts on individual minds too, and that available information does not equal understanding. Conceptual, with no metric. — [EA Forum (search snippet only)](https://forum.effectivealtruism.org/posts/otHLeLgER9eDjzEfB/epistemic-erosion-what-gradual-disempowerment-does-to-the)
- "Delegation Without Comprehension: AI Mediated Discovery and the Erosion of Human Epistemic Agency" (PhilArchive) introduces "interpretive latency" between machine discovery and human understanding. Whether it proposes metrics was not confirmed. — [PhilArchive (search snippet only)](https://philarchive.org/rec/ATHDWC)
- Forethought newsletter: notes that in many areas of epistemics it is hard to say what counts as "good." — [Forethought (search snippet only)](https://newsletter.forethought.org/i/193454919/the-bad)
- Comprehension Audits proposes measurable proxies: review rates, comments per kLOC, and audit pass or fail rates by unit, reported as trends. It finds comments per kLOC down about 50% since 2024 in open-source AI repos. — [arXiv HTML (opened)](https://arxiv.org/html/2610.10064v1)

### Inferences
- The field has a clear motivation (human comprehension as a safety-relevant quantity) but no validated instrument. A scalable, automated legibility test for AI-written research would fill a gap that both the Comprehension Audits authors and the gradual-disempowerment authors name but do not address.

### Gaps
- The full text of the gradual-disempowerment paper was not read, so indicator proposals inside it, if any, are unconfirmed.
- UK AISI, Redwood and DeepMind publications on this specific topic were not found in the searches run.
