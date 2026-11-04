# Making Language Models Fail Like Human Readers (human-like model readers for legibility testing)

Scope note: research done 2026-11-04 under a 15-tool-call budget. Most sources were read as arXiv/ACL abstract pages, not full PDFs, so many papers have no headline numbers here. Where a number comes only from a search-engine snippet and was not confirmed on the page, it is labelled UNVERIFIED.

## Simulated students and test-takers (profiles, role-play, IRT with LLM respondents)

### Takeaway
Simulating a "classroom" of role-played students and fitting IRT to their answers is the best-supported way to get human-like item difficulty: r = 0.75-0.82 on NAEP math (Acquaye et al. 2026). Two things that matter for a legibility test: weaker models simulate humans better than stronger ones, and proficiency prompts mostly do not make a strong model fail like a weaker human. For reading comprehension specifically, correlations are weak (rank correlation about 0.23 for discrimination; "small but positive" for difficulty).

### Cited Findings
- **Acquaye, Huang, Carpuat and Rudinger (UMD), "Take Out Your Calculators" (arXiv 2601.09953, Findings of ACL 2026).** Open-source LLMs role-play 4th, 8th and 12th graders of varying proficiency. An IRT model fit to the simulated responses estimates item difficulty, which is compared with NAEP item statistics. The estimates correlate with real item correctness rates at 0.75 (grade 4), 0.76 (grade 8) and 0.82 (grade 12). Asking LLMs directly to judge difficulty works poorly. Named students beat student IDs, and stratifying names by gender and race helps further. There is a cost-accuracy tradeoff in classroom size. Gemma, which is weaker at math, gave better real-world difficulty predictions than the stronger Llama and Qwen models. Domain: math, not reading. — [arXiv abstract](https://arxiv.org/abs/2601.09953); [ACL Anthology](https://aclanthology.org/2026.findings-acl.1807/)
- **"Can LLMs Estimate Student Struggles? Human-AI Difficulty Alignment with Proficiency Simulation" (arXiv 2512.18880).** Over 20 models were tested on medical-knowledge and math items. Models "struggle to simulate the capability limitations of students even when being explicitly prompted to adopt specific proficiency levels". "High performance often impedes accurate difficulty estimation". Scaling is "not reliably helpful". Models converge on a shared "machine consensus" rather than human difficulty patterns, and they fail to predict their own limitations. The abstract gives no correlation numbers. — [arXiv](https://arxiv.org/abs/2512.18880)
- **Chen, Li, Wang, Liang, Zhou, Jiao and Zhou, "LLMs Struggle to Measure What Distinguishes Students of Different Proficiency Levels" (arXiv 2606.18709, 2026).** 42 proprietary and open-weight LLMs were tested on reading comprehension items. Direct prediction aligns only weakly with human item discrimination. A response-based proxy (models actually answering) reaches a CEFR-stratified rank correlation of 0.231, "a stronger but still limited ranking signal". That signal comes mainly from differences between models, not from proficiency prompts reliably simulating different ability levels. The authors conclude that current LLMs do not model ability-conditioned human response behavior. — [arXiv](https://arxiv.org/abs/2606.18709)
- **Benedetto, Aradelli, Donvito, Lucchetti, Cappelli and Buttery, "Using LLMs to simulate students' responses to exam questions" (Findings of EMNLP 2024).** Three public datasets were used: one science and two English reading comprehension. From the search snippet: there is a "small but positive correlation" between difficulty from LLM virtual pretesting and difficulty from human pretesting. The prompt generalised to new questions but not to other LLMs. The abstract page shows no coefficient. — [ACL Anthology](https://aclanthology.org/2024.findings-emnlp.663/)
- UNVERIFIED (search snippet only): a zero-shot Rasch-based simulation with GPT-3.5/GPT-4-class models on reading and science exams reached r = 0.35 with IRT-calibrated difficulty. Likely the IRT-aligned simulated-students line (SMART, EMNLP 2025, which includes a 49-item short-answer reading comprehension dataset). — [SMART, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.1274.pdf)
- **Zelikman, Ma, Tran, Yang, Yeatman and Haber (arXiv 2310.06837).** LLMs fine-tuned on previous students' responses simulate answers to unseen items, giving estimates of item difficulty and ambiguity, which were used to filter GPT-4-generated sentence-reading-efficiency items. Scores on the generated test correlated with scores on the expert-made test at r = 0.93 (234 students, grades 2-8). This is a test-level score correlation, not a per-item difficulty correlation. Fine-tuning on real response data is the key ingredient. — [arXiv](https://arxiv.org/abs/2310.06837)
- **LLMs directly rating reading item difficulty.** One study (2502.17785) says o1 and GPT-4o classify reading question difficulty well, even better than human participants on several subtests. Another source reports that models systematically overestimate item ease and agree only modestly with human raters. Comparative (pairwise) prompting aligned better than absolute ratings. These are secondary snippets; check before citing. — [arXiv 2502.17785](https://arxiv.org/pdf/2502.17785); [ResearchGate: AI tools vs human raters](https://www.researchgate.net/publication/397037970_Comparing_AI_tools_and_human_raters_in_predicting_reading_item_difficulty)
- Profile-based simulators (knowledge-graph "cognitive prototypes", history-aware profiles trained with RL, SimLearner with hierarchical memory) report better simulation accuracy than baselines in tutoring settings. Expert raters, however, found agent behavior drifting from the assigned profile even when an LLM scorer did not notice. — [Embracing Imperfection](https://arxiv.org/abs/2505.19997); [Who Am I?](https://arxiv.org/pdf/2605.30051); [Which Type of Students can LLMs Act?](https://arxiv.org/html/2502.11678v4)

### Inferences
- The pattern "weaker model simulates better" and "high performance impedes estimation" matches the pilot: strong-ish models are not limited the way humans are. Telling a strong model to act like a novice does not reliably remove its competence. Using genuinely weaker models, or a population of models of different strengths, plus IRT is the better-supported lever.
- All the high correlations (0.75-0.82, 0.93) come from math or decoding items with large real response datasets. Nothing comparable was found for long-document reading comprehension.
- Item-difficulty simulation measures whether a question is hard. It does not measure whether a text is hard to read. None of these studies manipulated the text while holding the questions fixed.

### Gaps
- I could not find the "Zhou, Liu, Wilson and Tang" paper. One author-name search found nothing; it may need the exact title.
- The full-text coefficients of 2512.18880 and Benedetto et al. 2024 were not retrieved.

## Cognitively constrained models (limited memory, decay, chunked reading, resource-rational readers)

### Takeaway
Memory-limited LMs (fixed attention windows, decay, interference, dual-task load) reproduce human sentence-level difficulty and reading times better than unconstrained LMs. A 2026 hierarchical resource-rational reader claims to reproduce findings up to comprehension outcomes. I found no study that applies such a constrained reader to question-answering on whole documents and checks it against human comprehension of incoherent or complex text.

### Cited Findings
- **Restricted memory and reading costs.** An ACL 2026 study reports that LMs with restricted memory resources better approximate human reading costs such as reading times. A companion dual-task paradigm paper (ACL 2026 long; arXiv 2604.26351) reports that LMs adopt human-like comprehension strategies when cognitive resources are constrained. — [Dual-task paradigm, arXiv](https://arxiv.org/html/2604.26351v1); [ACL Anthology PDF](https://aclanthology.org/2026.acl-long.552.pdf)
- **Working-memory constraints built into GPT-2.** Fixed-width attention windows, decay mechanisms, and primacy and recency biases were tested. Fixed-width attention significantly improves grammatical accuracy when training data is scarce. — [Findings ACL 2026](https://aclanthology.org/2026.findings-acl.2133.pdf); [arXiv 2604.20789](https://arxiv.org/html/2604.20789)
- An LM with limited memory capacity captures similarity-based interference in human sentence processing. — [arXiv 2310.16142](https://arxiv.org/pdf/2310.16142)
- "Beyond Static and Linear: What Attention Constraints Best Fit Human Reading Times?" (arXiv 2608.23818, 2026) compares attention-constraint forms against reading times. Only the title and snippet were seen; no numbers. — [arXiv](https://arxiv.org/pdf/2608.23818)
- **Bai, Jin, Zhao and Oulasvirta, "Hierarchical Resource Rationality Explains Human Reading Behavior" (arXiv 2603.06006, 2026).** Eye movements are chosen to maximise expected comprehension minus cognitive and time costs, at nested levels: fixations serve word recognition, sentence integration drives skipping and regressions, and text-level goals drive memory construction and rereading. The authors claim it replicates "an unprecedented range of findings in human reading, from lexical effects to comprehension outcomes". The abstract gives no numbers. — [arXiv](https://arxiv.org/abs/2603.06006)
- Unconstrained modern LMs "often rely excessively on context information, resulting in superhuman predictions". This is the stated motivation for constrained or less contextualised variants. — [Kuribayashi et al. 2025, arXiv 2502.01615](https://arxiv.org/pdf/2502.01615)

### Inferences
- The evidence that constraints help is at the level of words and sentences (reading times, agreement, interference). Extending it to document-level QA (a sliding window, a decaying or summarised memory, a limited note budget, then answering questions) is plausible, but I found no validation.
- One practical design follows from this: make the weak reader read the paper in a single streaming pass with a bounded scratchpad and no ability to re-attend to the full text at answer time. That attacks directly the "whole paper in context" advantage that probably cancels out scrambling.

### Gaps
- No paper found with quantitative human-vs-constrained-model agreement on comprehension questions over multi-paragraph texts.
- No numbers retrieved for the ACL 2026 restricted-memory or attention-constraint papers.

## Psycholinguistic models of reading difficulty (surprisal, lossy-context surprisal, DLT)

### Takeaway
Surprisal from LMs predicts word-by-word reading times. But surprisal from the final layer of larger models fits worse ("inverse scaling"), and internal layers of large models restore the fit. Lossy-context surprisal (an LM plus a resource-rational memory limit) explains nested-structure difficulty that neither surprisal nor memory locality explains alone. None of these has been validated as a predictor of document-level comprehension or QA accuracy in what I found.

### Cited Findings
- **Oh and Schuler, TACL 2023.** Across 5 GPT-Neo and 8 OPT variants and 2 reading-time datasets, they find a "strictly monotonic, positive log-linear relationship between perplexity and fit to reading times": lower-perplexity (larger) models fit human reading times worse. The larger models underpredict reading times on named entities and overpredict them on function words, attributed to memorisation. This replicates earlier GPT-2 findings. — [ACL Anthology](https://aclanthology.org/2023.tacl-1.20/)
- **Kuribayashi, Oseki, Ben Taieb, Inui and Baldwin (arXiv 2502.01615, 2025), "Large Language Models Are Human-Like Internally".** Next-word probabilities from internal layers of larger models fit human data as well as or better than smaller models. This holds for self-paced reading, gaze durations, MAZE and N400. Earlier layers match fast gaze durations; later layers match N400 and MAZE. — [arXiv](https://arxiv.org/abs/2502.01615)
- **Hahn, Futrell, Levy and Gibson, PNAS 2022 ("A computational model of language comprehension unites diverse perspectives").** GPT-2 combined with a resource-rational lossy memory unifies expectation-based (surprisal) and memory-based (dependency locality) accounts. It predicts nested-structure difficulty that neither predicts alone, and predicts that rarer phrasing is harder to remember (e.g., "The fact that" is easier than "The report that"). Confirmed in two English comprehension experiments and in completions in English, Spanish and German. It builds on Futrell's 2020 lossy-context surprisal. No effect sizes in the secondary report; the PNAS page returned 403. — [PNAS](https://www.pnas.org/doi/10.1073/pnas.2217108119); [Neuroscience News summary](https://neurosciencenews.com/language-comprehension-model-22133/)
- Dependency locality theory ties comprehension difficulty to verbal working-memory limits, but whether working-memory capacity alone explains comprehension is still debated. — [PNAS](https://www.pnas.org/doi/10.1073/pnas.2217108119); [Dual-task paper](https://arxiv.org/html/2604.26351v1)
- **Clark, Martin and Mahowald, "Why is AI-generated language so hard to understand?" (arXiv 2610.29162, Oct 2026).** They measure features linked to human processing difficulty (object relative clauses, long dependencies) in AI-generated text from 9 frontier and near-frontier models, against paraphrases and human writing, and fine-tune an LM under a length penalty. This is a production-side analysis, not a comprehension test. — [arXiv](https://arxiv.org/abs/2610.29162)
- A study comparing human and LM sentence-processing difficulty on complex structures exists (arXiv 2510.07141); it was not opened. — [arXiv](https://arxiv.org/pdf/2510.07141)

### Inferences
- For a legibility metric, aggregated surprisal from a small or internal-layer LM, or lossy-context surprisal, combined with dependency-length statistics, is the best-grounded predictor of local human processing cost. It could act as a "human-cost" signal alongside the quiz, since it responds to long nested sentences that the quiz-takers ignore.
- These measures say little about scrambled discourse order or opaque abbreviations across a whole paper. Those are discourse- and memory-level effects.

### Gaps
- No study found that uses surprisal or DLT aggregates to predict whole-document comprehension scores.
- No effect sizes retrieved for Hahn et al. 2022.

## Does LLM comprehension depend on coherence and organization like human comprehension?

### Takeaway
Largely no. Reading comprehension models keep about 95% of their performance when sentences are shuffled. LLMs detect incoherence using world knowledge rather than human-like processing. Evidence on human-like position effects ("lost in the middle") is mixed. This matches the pilot's finding of at most a 9-point drop.

### Cited Findings
- **Sugawara, Stenetorp, Inui and Aizawa, AAAI 2020.** Ablations on 10 MRC datasets (including CoQA, SQuAD 2.0 and RACE) with a strong baseline. Shuffling sentence order kept 95.4% of original performance; content words only kept 89.2%; shuffling words within sentences kept 78.5%. Conclusion: most questions the model answers correctly do not require grammatical or complex reasoning. — [arXiv 1911.09241](https://arxiv.org/pdf/1911.09241); [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/6422)
- **Acceptability judgments with word-shuffled paradigms.** Models scored 71.6% on these versus 87.9% on other paradigms; humans scored 83.1% versus 89.7%. The authors read this as LLMs being insensitive to word order. — [arXiv 2408.09639](https://arxiv.org/pdf/2408.09639)
- Shuffling input order lowers LLM output accuracy measurably, and few-shot prompting only partly mitigates it. There is no human baseline. — [arXiv 2502.04134](https://arxiv.org/html/2502.04134v2)
- **"Mary, the Cheeseburger-Eating Vegetarian" (arXiv 2512.07777).** Uses the human paired-narrative inconsistency paradigm on LLMs. Internal representations show an asymmetry humans lack, suggesting LLMs rely more on background world knowledge to detect incoherence. — [arXiv](https://arxiv.org/pdf/2512.07777)
- LLM-derived coherence scores were significantly associated with lay readers' multiple-choice comprehension of health texts. Simplifying edits raised both. Less knowledgeable readers gained more from coherence, consistent with comprehension theory. This uses LLMs as raters, not as readers. — [ScienceDirect (J. Biomed. Inform.)](https://www.sciencedirect.com/science/article/abs/pii/S153204642400176X); [Experts@Minnesota](https://experts.umn.edu/en/publications/coherence-and-comprehensibility-large-language-models-predict-lay/)
- **Position effects, mixed.** Guo and Vosoughi report serial-position (primacy and recency) effects consistently across LLMs (arXiv 2406.15981). A 2025 cognitive-psychology study found position-invariant LLM behavior, unlike humans (arXiv 2509.17138). Another paper argues lost-in-the-middle emerges from retrieval demands in small models trained on human memory tasks (arXiv 2510.10276). A 2026 preprint finds human recall is recency-dominant while a dense retriever is primacy-dominant (arXiv 2606.28327). — [2406.15981](https://arxiv.org/html/2406.15981v1); [2509.17138](https://arxiv.org/pdf/2509.17138); [2510.10276](https://arxiv.org/html/2510.10276v1); [2606.28327](https://arxiv.org/pdf/2606.28327)
- Randomly shuffled documents do not resemble naturally incoherent human writing, so shuffling is an unnatural perturbation. — [Discourse Coherence in the Wild, arXiv 1805.04993](https://arxiv.org/pdf/1805.04993)

### Inferences
- With the whole paper in context, a model can attend directly to the relevant sentences wherever they sit, so reordering costs little. Humans pay for reordering through memory and integration. To make models sensitive to it, the reading process has to be sequential and memory-bounded (see the constrained-models section). Changing the prompt persona will not do it.
- LLM-judged readability tracks human comprehension better than LLM QA accuracy does. That matches the pilot, where models rated the text far less readable while barely losing accuracy. A hybrid metric (model judgment plus QA) is supported by the health-text study.

### Gaps
- I found no direct study putting humans and LLMs through identical QA on the same passages with sentence order or cohesion manipulated.

## Human-model alignment of per-item and per-text difficulty

### Takeaway
Direct per-item correlations between LLM answer accuracy and human accuracy on reading questions are rarely reported and appear weak. Textual features barely explain human difficulty or the human-model gap. The strongest alignments come from IRT over simulated populations (math) or from models trained on item data.

### Cited Findings
- **Sugawara, Nangia, Warstadt and Bowman, ACL 2022 ("What Makes Reading Comprehension Questions Difficult?").** "To our surprise", passage source, passage length and readability measures did not significantly affect question difficulty. Per the search snippet: no strong correlations between textual features and the human-model performance gap, except a weak negative correlation with answering time. — [arXiv 2203.06342](https://arxiv.org/abs/2203.06342)
- A model trained on embeddings of annotated items predicted reading item difficulty at r = 0.77. It is supervised on item data, not a model answering the questions. — [arXiv 2502.20663](https://arxiv.org/html/2502.20663)
- Reading comprehension discrimination: rank correlation 0.231 from simulated responses across 42 LLMs. — [arXiv 2606.18709](https://arxiv.org/abs/2606.18709)
- Reading pretesting: "small but positive" correlation. — [Benedetto et al. 2024](https://aclanthology.org/2024.findings-emnlp.663/)
- Math (NAEP): r = 0.75-0.82 via role-played classroom plus IRT. — [Acquaye et al. 2026](https://arxiv.org/abs/2601.09953)

### Inferences
- Treat any LLM-reader legibility score as uncalibrated until it is checked against a small human sample on the same papers and quizzes. The literature gives no reason to expect strong item-level agreement for long-document comprehension without such calibration.
- Since readability features do not predict human question difficulty well (Sugawara 2022), quiz difficulty mixes text legibility with question difficulty. The test should compare scores on the same questions across text versions (a within-item design) rather than raw accuracy levels.

### Gaps
- I found no dataset with humans and LLMs answering the same questions on long expository or scientific documents with per-item correlations reported. Candidates that were not checked: QuALITY (it has time-limited human annotators) and the OneStop eye-tracking comprehension corpus.
