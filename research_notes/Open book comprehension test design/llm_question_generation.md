# LLM Generation and Validation of Deep Comprehension Questions from Documents (2023-2026)

Scope note: this is a time-boxed sweep (about 20 tool calls). The primary sources I opened and read are Savaal (arXiv 2502.12477v2), greCAPTCHA (arXiv 2609.20481v1), KnowledgeGain (arXiv 2605.31099v1), Comprehension Audits (alphaXiv 2610.10064v1), Scaria et al. 2024 (arXiv 2408.04394v1), the Acquaye et al. 2026 abstract, a Pith review of Zhou/Liu/Tang/Wilson 2026, and the NeurIPS 2025 abstract for "No Question, No Passage, No Problem." Items marked "(search snippet only)" were not opened, so their numbers need checking before anyone relies on them.

## 1. Which generation methods give deeper questions, and what is the evidence?

### Takeaway
The strongest human-validated evidence for "deeper" questions on research documents comes from Savaal's concept-first pipeline. It works by extracting and ranking main ideas, retrieving passages for each idea, and then generating questions. Experts (authors of the papers) rated its questions better on depth of understanding than direct whole-document prompting. The gain was large on long dissertations (about 2.4x fewer "doesn't test understanding" questions) and modest on conference papers (about 1.5x). Bloom-conditioned prompting with chain-of-thought plus one example per level gives good overall quality. However, the intended Bloom level is hit only about two-thirds of the time, and "Create"-level questions are weak. Adding more prompt detail eventually hurts.

### Cited Findings
**Savaal (Noorbakhsh, Chandler, Karimi, Alizadeh, Balakrishnan; MIT; arXiv Feb 2025)**
- Pipeline:
  - GROBID parses the document into sections. An LLM extracts main ideas per section (map), merges and deduplicates them (combine), condenses the list if it is too long for context (reduce), and then ranks the ideas by importance.
  - ColBERT retrieves the top 3 passages per idea.
  - The LLM writes N/M four-option MCQs for each of the top M ideas, using only those passages.
  - [Savaal](https://arxiv.org/html/2502.12477v2)
- Practical details:
  - A separate choice-refinement stage was dropped because it added ambiguity and experts disliked it.
  - Answer positions had to be randomized because GPT-4o put the correct answer at position B 73.3% of the time.
  - Main model: GPT-4o at temperature 0. Llama-3.3-70B was also tested.
  - [Savaal](https://arxiv.org/html/2502.12477v2)
- Human evaluation:
  - 76 experts, mostly primary authors of the documents, with a 38% response rate.
  - Documents: 21 PhD dissertations (about 142 pages on average) and 50 conference papers (about 19 pages).
  - Each document got 10 questions per method, rated blind on 4-point scales for clarity, depth of understanding and choice quality, plus a usability rating.
  - Totals: 420 dissertation questions and 1,100 paper questions.
  - [Savaal](https://arxiv.org/html/2502.12477v2)
- Dissertation results (Savaal vs. direct prompting):
  - Questions that failed to test understanding: 11.9% vs. 29.0%.
  - Unusable questions: 21.4% vs. 32.9%.
  - Poor choices: 29.0% vs. 39.0%.
  - Among experts with a preference on depth: 61.9% preferred Savaal vs. 9.5% Direct (the headline "6.5x").
  - [Savaal](https://arxiv.org/html/2502.12477v2)
- Conference paper results:
  - Questions that failed to test understanding: 10.9% vs. 16.7% (the headline "1.5x").
  - Unusable questions: 13.8% vs. 15.3%.
  - No clear preference on choice quality or usability.
  - Over 90% of questions from both methods were rated clear.
  - Implication: for 10-20 page papers, direct prompting is nearly as good. Concept extraction matters most for long documents.
  - [Savaal](https://arxiv.org/html/2502.12477v2)
- Agreement and difficulty effects:
  - Agreement between human raters was weak on raw scores (ρ = 0.32 for Savaal's understanding scores) but rose to ρ = 0.76 once scores were binarized.
  - Experts rated questions they themselves got wrong lower on clarity, choice quality and usability, but not on depth.
  - [Savaal](https://arxiv.org/html/2502.12477v2)
- Limitations:
  - Only single-answer MCQs. The system does not produce questions that need mathematical or logical reasoning.
  - Figures were excluded.
  - Only individual questions were evaluated, not quiz-level coverage or difficulty.
  - Evaluation covered only CS and aeronautics.
  - [Savaal](https://arxiv.org/html/2502.12477v2)
- Cost: Savaal pays a fixed extraction cost per document, and its marginal cost per question does not depend on document length. It beats direct prompting on cost at roughly 60-80 questions per dissertation. The paper's $0.47 vs. $0.77 per-document figures appear to be swapped in the text. [Savaal](https://arxiv.org/html/2502.12477v2)

**Bloom-conditioned prompting (Scaria et al., AIED 2024, arXiv 2408.04394)**
- Design:
  - Five generators (Mistral-7B, Llama-2-70B, PaLM 2, GPT-3.5, GPT-4) and five prompt strategies, from a simple prompt to CoT plus skill definitions, examples and explanations.
  - 2,550 questions for a graduate data science course across six Bloom levels.
  - Two experts rated them with a hierarchical nine-item rubric.
  - [Scaria et al.](https://arxiv.org/html/2408.04394v1)
- Results:
  - 78% of questions were rated high quality, but only 65.56% of those matched the intended Bloom level.
  - Quality by prompt: simple 70.2%; CoT + skill 81.2%; CoT + one example per level 82.6%; CoT + skill + examples 81.4%; everything included 74.7%. More prompt detail eventually hurt both quality and level adherence.
  - Quality by model: GPT-4 89.0%, GPT-3.5 86.3%, others about 71-73%.
  - Most models struggled with "Create"-level questions.
  - [Scaria et al.](https://arxiv.org/html/2408.04394v1)
- Other GPT-4 Bloom-level studies (search snippet only): a biology MCQ study found the questions relevant and usable but Bloom-level alignment weak. A 2024 GPT-4 Turbo study reported human-machine quality disagreement that grew with Bloom level. [Search summary](https://arxiv.org/html/2408.04394v1)

**Question families aimed at understanding of research (greCAPTCHA, 2026; details in Section 6)**
- greCAPTCHA uses four families: planted-error detection (MCQ), unstated rationale (explain an unjustified design choice), background knowledge the paper presupposes, and failure mode (a realistic condition under which the method degrades, and why). [greCAPTCHA](https://arxiv.org/html/2609.20481v1)

### Inferences
- For 10-30 page AI safety papers, the concept-first step should be used mainly to guarantee coverage of the paper's main claims, methods and conclusions (rank ideas, then allocate questions). On paper-length documents it buys only a modest depth gain over a well-prompted strong model.
- A few-shot prompt with one example per question type beats a long specification. Over-long generator prompts reduced quality in Scaria et al.
- I found no head-to-head human-rated study (2024-2026) comparing multi-agent writer-critic loops or adversarial solver filtering against single-pass generation for comprehension questions. Claims for those methods should be treated as untested here.

### Gaps
- I did not locate or verify human-rated evidence for multi-agent writer-critic QG, or for adversarial filtering against a solver as a way to get deeper questions (as opposed to harder benchmark items). This is not covered.
- The later sections of Savaal (characters after 100k) were not read. There may be extra appendix results, for example on arXiv post-cutoff papers.

## 2. Automatic vs. human measurement of question quality; AI-judge agreement with experts

### Takeaway
In every study I opened, LLM judges of question quality agreed poorly with human experts. Savaal's GPT-4o judge rarely gave negative ratings, and DSPy prompt tuning did not fix it. Scaria et al.'s Gemini Pro judge favored a different model than the experts did and over-labeled Bloom levels. Even agreement between human experts is weak on fine-grained scales and acceptable only when scores are binarized. Pipelines validated only by AI judges should be treated as unvalidated.

### Cited Findings
- **Savaal's AI judge:**
  - GPT-4o rated each question 1-4 on seven criteria and was "misaligned" with experts. It rarely gave negative ratings and rated choice quality poorly for both methods.
  - It only slightly favored Savaal on depth: 28.6% vs. 14.3% "Agree" on dissertations.
  - DSPy prompt optimization did not improve alignment. The authors conclude that AI judges alone are unreliable for evaluating QG.
  - [Savaal](https://arxiv.org/html/2502.12477v2)
- **Scaria et al.'s judge:**
  - The Gemini Pro judge favored PaLM 2, which the experts did not, and tended to label questions as "Apply" or "Analyze."
  - Experts agreed with each other 82-93% on Bloom level, depending on prompt.
  - No kappa between the judge and the experts is reported, only a "significant discrepancy."
  - [Scaria et al.](https://arxiv.org/html/2408.04394v1)
- **Human-human agreement:** in Savaal, raw-score agreement was ρ = 0.32 and binarized agreement was ρ = 0.76. Binary judgments ("does this test understanding? yes/no") are much more reliable than Likert scales. [Savaal](https://arxiv.org/html/2502.12477v2)
- **A 2026 preprint comparing small and large models** (search snippet only) reports that model-based evaluations show "systematic inconsistencies and bias relative to expert ratings." [arXiv 2605.15015](https://arxiv.org/html/2605.15015v1)
- **Simulated readers as a behavioral judge:** KnowledgeGain's calibrated LLM reader simulator matches the aggregate distribution of human answers (KL 0.007-0.119 by condition). However, its per-topic gain correlates with human gain at only r ≈ 0.263, and the authors call it "a development proxy, not a replacement for human evaluation." [KnowledgeGain](https://arxiv.org/html/2605.31099v1)

### Inferences
- If an AI judge is used to screen questions, it should make binary, specific checks: answerable from the document, single correct answer, tests a main idea rather than a detail, no leak. Holistic Likert quality ratings are what failed to agree with experts.
- A small human-expert audit is still necessary. Savaal's evidence suggests that asking authors or domain experts for binary "tests understanding / usable" labels on a sample is a feasible gold standard.

### Gaps
- I found no study reporting a kappa or correlation between an LLM judge and humans specifically for depth of comprehension questions on research papers.
- greCAPTCHA's excerpt did not report agreement between the AI grader and humans.

## 3. Detecting leaky or shortcut questions and hardening them without raising the intelligence required

### Takeaway
The standard check is ablation. Answer the question without the document, and with only the answer options. Benchmark work shows that LLMs beat chance on MCQ reading comprehension even with both passage and question removed. MCQs are therefore inherently leaky. greCAPTCHA found that MCQs were the least discriminating question type (AUC 0.595) and recommends dropping them. KnowledgeGain measures leakage directly with a pre-reading test and normalized gain.

### Cited Findings
- **Choices-only shortcuts:**
  - "No Question, No Passage, No Problem" (NeurIPS 2025): with the passage and question both removed, models "consistently exceed majority baselines across five benchmarks." The authors attribute this to distractor elimination and inferring a plausible context, not only to artifacts.
  - The abstract page gives no numbers.
  - [NeurIPS 2025](https://neurips.cc/virtual/2025/126714)
- **Balepur, Ravichander and Rudinger 2024, "Artifacts or Abduction"** (search snippet only): LLMs can often answer MCQs from the choices alone, sometimes by reconstructing the missing question. [Search summary](https://arxiv.org/html/2402.12483v1)
- **Pre/post design (KnowledgeGain):**
  - Raw gain = post-reading accuracy - pre-reading accuracy. Normalized gain = (Post - Pre)/(1 - Pre).
  - Each question includes an "I do not know" option, which is scored as incorrect but tracked separately.
  - Questions are replaced if the answer cannot be recovered from the source, if more than one option is plausible, or if they rely on unstated assumptions.
  - [KnowledgeGain](https://arxiv.org/html/2605.31099v1)
- **Open-book searchability:** in greCAPTCHA, participants could search the paper for MCQ answers. Planted-error MCQs had an AUC of only 0.595 for telling authors from non-authors, versus 0.934 with MCQs excluded. [greCAPTCHA](https://arxiv.org/html/2609.20481v1)
- **Savaal position bias:** GPT-4o placed the correct answer at B 73.3% of the time, which is a trivially exploitable leak unless positions are randomized. [Savaal](https://arxiv.org/html/2502.12477v2)
- **RepLiQA (2024)** (search snippet only): uses unseen, fabricated reference content so that questions cannot be answered from memory, and reports that models rely on internal memory over reading. [Search summary](https://neurips.cc/virtual/2025/126714)

### Inferences
- A leakage filter for the quiz pipeline should:
  - Run the reader models closed-book (question only) and choices-only, and drop or rewrite any item they pass above chance.
  - Report gain over a no-document baseline, as in KnowledgeGain, rather than raw accuracy.
  - Randomize option positions.
- To harden a leaky question without making it require more intelligence:
  - Anchor it to a paper-specific number, condition or choice that a model cannot know from priors.
  - Switch from MCQ to short free response graded with a rubric.
  - Avoid adding reasoning steps. The difficulty should come from needing the document, not from extra inference.
- Open-book readers make "locate the sentence" questions trivial. Questions about rationale and failure modes resist lookup better, based on greCAPTCHA's discrimination results.

### Gaps
- I found no published method that explicitly controls difficulty, meaning one that makes a question less leaky while keeping its reasoning demand fixed. This is open.

## 4. Counterfactual, what-if, transfer and edge-case questions; integration across long documents

### Takeaway
Little 2024-2026 work generates counterfactual or edge-case comprehension questions from a document for assessment. The closest are greCAPTCHA's "failure mode" and "unstated rationale" families, which discriminated well when combined with the other free-response families. Counterfactual QA benchmarks such as CofCA and CRiT-QA show that LLMs over-rely on parametric knowledge when premises conflict with it. That is a risk when weaker readers answer counterfactual questions.

### Cited Findings
- **greCAPTCHA's free-response families:**
  - "Failure mode" asks for a realistic condition under which the method or finding would degrade, and the mechanism. "Unstated rationale" asks the reader to explain a design choice the paper does not justify.
  - Together with background knowledge, the free-response families reached AUC 0.934 for authors vs. non-authors.
  - Participants asked for one aspect per question and found some detail questions irrelevant, even though they discriminated well.
  - [greCAPTCHA](https://arxiv.org/html/2609.20481v1)
- **CofCA (ICLR 2024):** a step-wise counterfactual multi-hop QA benchmark built from edited Wikipedia passages. LLMs' multi-hop reasoning is "exaggerated" once memorization is removed. (Search snippet only.) [CofCA](https://arxiv.org/html/2402.11924v5)
- **CRiT-QA (arXiv 2607.10562):** an LLM pipeline turns a factual reasoning chain into a counterfactual one and rewrites the supporting paragraphs. (Search snippet only.) [CRiT-QA](https://arxiv.org/pdf/2607.10562)
- **"LLMs Struggle to Perform Counterfactual Reasoning with Parametric Knowledge" (ICML 2025):** LLMs often fall back on parametric knowledge instead of in-context counterfactual premises. (Search snippet only.) [arXiv 2506.15732](https://arxiv.org/html/2506.15732v1)
- **Long documents:** Savaal's per-section main-idea extraction and ranking is the only long-document QG method with human expert validation that I found. Its gain was largest on 142-page dissertations. [Savaal](https://arxiv.org/html/2502.12477v2)

### Inferences
- Counterfactual and edge-case questions about a paper should come with a rubric answer derived from the paper's stated mechanisms or ablations. Otherwise they become open research questions, which raises the intelligence required and makes them ungradeable.
- Questions that integrate across a long document can be built by pairing two ranked main ideas, for example a method choice and the result it explains. No study I found validated this specifically.

### Gaps
- No human-validated work found on generating "what if" or transfer questions from scientific papers for assessment.
- No validated method found for generating questions that require integration across sections.

## 5. LLMs simulating test takers, estimating difficulty and calibrating items

### Takeaway
LLMs are poor direct judges of difficulty. Simulating a diverse population of students and fitting IRT to their responses gives difficulty estimates that correlate moderately to strongly with human data: 0.75-0.82 with NAEP correctness rates (Acquaye et al. 2026), and up to Spearman 0.86-0.90 with cognitive-diagnostic profiles (Zhou et al. 2026). A recurring finding is that strong models are "too accurate and too uniform." Weaker models or explicit mastery profiles track humans better. For the quiz use case, a single per-topic simulator agreed with humans poorly (KnowledgeGain r ≈ 0.26).

### Cited Findings
- **Acquaye, Huang, Carpuat and Rudinger, Findings of ACL 2026, "Take Out Your Calculators" (arXiv 2601.09953):**
  - LLMs are "poor direct judges of problem difficulty."
  - Role-played classrooms of 4th/8th/12th graders plus IRT gave correlations with NAEP item correctness of 0.75 (grade 4), 0.76 (grade 8) and 0.82 (grade 12).
  - Giving students diverse names helped compared with plain IDs, and stratifying names by gender and race helped more.
  - Gemma, which is weaker at math, predicted difficulty better than the stronger Llama and Qwen models.
  - Larger classrooms showed a cost vs. accuracy tradeoff.
  - [arXiv abstract](https://arxiv.org/abs/2601.09953v2); [ACL Anthology](https://aclanthology.org/2026.findings-acl.1807/)
- **Zhou, Liu, Tang and Wilson 2026, "Aligning LLM-Simulated and Human Examinees… Cognitive Diagnostic Profiling" (arXiv 2607.26317):**
  - Data: the Tatsuoka fraction-subtraction set (536 humans, 15 items, 5 attributes). Each simulated examinee gets a binary mastery profile.
  - Best case, Gemini 3.0 Flash (Thinking): 1PL difficulty Spearman rose from 0.24 with no profile to 0.86 (uniform prior) and 0.90 (informative prior). RMSE fell from 6.31 to 1.30 and 0.90.
  - Baseline LLMs were "too accurate and too uniform." One non-thinking baseline produced 98% identical response patterns.
  - Profiled simulations produced at most 133 distinct response patterns, versus 267 for humans.
  - Cost: under $0.03 to about $14 per 100 examinees.
  - Caveats: the informative prior was fit on the same humans (in-sample), each cell was run once with no error bars, only 15 items were used, and contamination is possible.
  - Authors: CDP "complements human calibration."
  - [Pith review](https://pith.science/paper/2607.26317); [arXiv](https://arxiv.org/pdf/2607.26317)
- **Liu et al. 2024/2025, "Leveraging LLM-Respondents for Item Evaluation"** (search snippet only):
  - Item parameters calibrated from LLM respondents correlate above 0.8 with human calibrations (e.g., GPT-3.5).
  - A review notes that mixing human and synthetic responses 1:1 improved calibration.
  - [arXiv 2407.10899](https://arxiv.org/pdf/2407.10899)
- **A 2026 medical item-difficulty paper** (search snippet only) also reports that weaker models reproduce real student item performance better than stronger ones. [arXiv 2610.38402](https://arxiv.org/html/2610.38402)
- **KnowledgeGain's LLMSim:** calibrated by minimizing KL divergence over Correct/Incorrect/IDK distributions. It fits aggregate distributions but correlates with human per-topic KGain at only 0.263. [KnowledgeGain](https://arxiv.org/html/2605.31099v1)

### Inferences
- In the quiz setup, weaker LLM readers are themselves the "test takers," so IRT on a panel of diverse weaker readers can estimate item difficulty and discrimination and flag items that are too hard or that no reader solves. Population diversity (several models or personas) is needed, because uniform panels make the IRT parameters poorly identified.
- Simulated difficulty validity is shown mainly for math items with large human norm datasets. Transfer to free-response comprehension of research papers is unproven.

### Gaps
- No study found that calibrates LLM-simulated difficulty for open-book comprehension questions on research papers.

## 6. KnowledgeGain, greCAPTCHA and Comprehension Audits on question design

### Takeaway
KnowledgeGain offers a pre/post, Bloom-tiered, "I don't know"-enabled quiz design and a cautionary simulator result. greCAPTCHA offers four question families for understanding of research and strong evidence that free-response rationale and failure-mode questions discriminate understanding while MCQs do not. Comprehension Audits is a policy proposal that defines what to understand but specifies no question types and has no empirical validation.

### Cited Findings
**KnowledgeGain (Soós, Jiang, Wu; EMNLP 2026; arXiv 2605.31099)**
- Question design:
  - Questions are generated from the source abstract by an LLM generator (QGen) and manually verified.
  - Each abstract gets 6 questions: 2 extractive true/false, 2 extractive MCQ and 2 inferential MCQ, mapped to Bloom's knowledge and comprehension levels.
  - Each question has an "I do not know" option.
  - Questions must be answerable from the source without outside knowledge.
  - [KnowledgeGain](https://arxiv.org/html/2605.31099v1)
- Study 1:
  - 30 STEM undergraduates, 30 topics, each with an abstract, a news article and a tweet. Crossover design.
  - Median gain: news about 60, abstract about 50. Tweets were lower (p < 0.001).
  - [KnowledgeGain](https://arxiv.org/html/2605.31099v1)
- Study 2:
  - Articles chosen by the simulator gave higher human normalized gain than a strong agentic baseline: 0.777 vs. 0.719 (Wilcoxon p = 0.048; mixed-effects p = 0.015).
  - Limitations: results depend on the generated question set, and the simulator does not reliably predict human gain per topic.
  - [KnowledgeGain](https://arxiv.org/html/2605.31099v1); [ACL Anthology](https://aclanthology.org/2026.emnlp-main.1672/)

**greCAPTCHA (Payan, Gyevnár, Kasirzadeh, Shah; arXiv 2609.20481, Sept 2026)**
- Design:
  - Revised Bloom's taxonomy (factual, conceptual and procedural knowledge) mapped to four families: planted-error detection (MCQ), unstated rationale, background knowledge and failure mode.
  - 8 questions per paper (2 per family), generated with GPT-5.6 Sol along with grading rubrics, then graded automatically with partial credit.
  - [greCAPTCHA](https://arxiv.org/html/2609.20481v1)
- Results (n = 31 researchers, 15 minutes per paper, proctored):
  - Mean score 49.7 on their own paper vs. 27.2 on an unfamiliar paper.
  - AUC 0.90 overall, 0.595 for planted-error MCQs alone, 0.934 without MCQs. Best false-positive rate about 20%.
  - No significant difference between in-field and out-of-field participants (p = 0.437), though in-field participants scored higher on background knowledge.
  - [greCAPTCHA](https://arxiv.org/html/2609.20481v1)
- Lessons:
  - Remove MCQs, which were too easy and searchable.
  - Ask about one aspect per question.
  - State the expected detail and audience, and signal rubric items in the question.
  - Accept equivalent correct answers. Participants found the grader too rigid.
  - Avoid irrelevant detail questions.
  - Time limits and typing load may measure things other than understanding.
  - No systematic checks for answerability, leakage or difficulty are reported in the part of the paper I read.
  - [greCAPTCHA](https://arxiv.org/html/2609.20481v1)
- Code and anonymized responses are on GitHub. [GitHub](https://github.com/justinpayan/greCAPTCHA)

**Comprehension Audits (Bodkin, Sokhansanj, Hadfield; Institute for Law & AI and JHU; Oct 2026)**
- The audited unit is a "contribution," such as an experiment and its conclusions, or a training-recipe change.
- Required comprehension: deep understanding of design, implementation behavior and process; familiarity with key findings and analysis; only limited understanding of model internals.
- Format: Socratic auditor meetings, graded against a reportable rubric, with outcomes of pass, conditional pass or fail. A fail gates dependent work. No AI may be used to generate answers.
- It specifies no question types and no LLM-generated questions.
- It is a proposal with no completed audits. Its empirical proxy is a decline in PR review comments per kLOC: 36% and 46% of the Q1 2024 baseline. The authors call this suggestive, not proof.
- [alphaXiv 2610.10064](https://alphaxiv.org/abs/2610.10064v1); [Law-AI listing](https://law-ai.org/research/)

### Inferences
- A good template for quizzes on AI safety papers combines greCAPTCHA's free-response families (rationale, failure mode, plus a "key result / conclusion" family) with KnowledgeGain's pre/post gain and "I don't know" option.
- The scope of "understanding" in Comprehension Audits (design, process, findings and their analysis, with limited demand for internals) maps onto quiz coverage of methods, results and conclusions.
- greCAPTCHA's AUC was measured on humans who wrote the papers versus strangers, not on LLM readers. Its discrimination may not transfer to weaker LLM readers taking the quiz open-book.

### Gaps
- KnowledgeGain's question verification prompts (Appendix E.2) and greCAPTCHA's generation prompts (Appendix A) were not read.
- The full Comprehension Audits rubric was not read.
- Neither KnowledgeGain nor greCAPTCHA reports agreement between the AI grader and humans in the parts I read.
