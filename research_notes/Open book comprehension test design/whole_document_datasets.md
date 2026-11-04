# Datasets and methods that require whole-document understanding, and how their builders validated it

Scope note: The primary sources for QuALITY, ROPES and Contrast Sets were read in full (PDF text). ELAIPBench, RPC-Bench and NovelQA were read through their arXiv HTML. QASPER, SciDQA, LooGLE, Kaushik & Lipton, FlenQA ("Same Task, More Tokens"), CRASS and IfQA were read at abstract or search-snippet level only. Those items are flagged and carry fewer numbers. TimeTravel, BABILong, LongBench, PeerQA and SPIQA were not opened (see Gaps).

## 1. QuALITY (Pang et al. 2022): building and speed-validating "hard" questions

### Takeaway
QuALITY gives writers bonuses for questions that time-limited skimmers get wrong but careful untimed readers get right. Under this protocol, 49.9% of the retained questions qualified as HARD. A question-only baseline still reached 43.3% (chance is 25%), so speed validation does not remove answer-option artifacts by itself. You also need partial-input baselines.

### Cited Findings
- Writers: 22 hired from Upwork, most with degrees or professional experience in literature or teaching. Each writes 10 four-option questions per passage, and each passage goes to two writers (20 questions per passage before filtering). Writers earned an average of $21.05/hr after bonuses. Bonuses depended on how many of a writer's questions met the HARD criteria, and writers got detailed feedback from both validation tasks each round. — [Pang et al. 2022, QuALITY](https://arxiv.org/abs/2112.08608)
- Rationale: "If a person can quickly identify the answer to a question, such as through skimming or ctrl-F-style in-browser search, then the question does not require broader understanding of the passage, and a model is likely to be able to identify the correct answer via extractive methods." — [QuALITY §2.3.1](https://arxiv.org/abs/2112.08608)
- Speed validation protocol: 5 annotators per question. The worker reads the question and the 4 options without the passage, then reveals the passage and gets 40 seconds to skim or ctrl+F. The passage then disappears, and the worker has 5 more seconds to answer. Each task holds 10 questions from different passages with option order randomized, including 1 author-written catch question. Pay was $2.25 per task plus $0.20 per correct answer (about $17/hr). — [QuALITY §2.3.1](https://arxiv.org/abs/2112.08608)
- HARD is defined as a question the majority of speed annotators answer incorrectly, while untimed annotators answer it correctly and judge it unambiguous. — [QuALITY](https://arxiv.org/abs/2112.08608)
- Why they did not use model-in-the-loop adversarial collection: models were so weak that their behavior "would not provide enough signal," and adversarial collection risks limiting the test set's usefulness across models. Writers therefore reason about what is hard for a human skimmer, not for a model. — [QuALITY §2.3.1](https://arxiv.org/abs/2112.08608)
- Untimed validation: 3 annotators per item for train and 5 for dev/test. Each read the whole passage (about 50 to 60 minutes per task, 20 questions), with a bonus when their answer and their answerability judgment matched the majority. Three meta-questions followed each item: (Q1) answerable and unambiguous? (Q2) how much of the text is needed: a sentence or two, a long paragraph or two, at least a third, or most or all? (Q3) which unchosen option was the best distractor? Q1 decided inclusion. Q2 and Q3 were feedback for writers. — [QuALITY §2.3.2, Table 2](https://arxiv.org/abs/2112.08608)
- Attrition: 6,737 of 7,620 questions (88.4%) passed inclusion, meaning a strict majority gold label existed and a majority rated the item answerable and unambiguous. Of these, 49.9% are HARD (3,360). The writer's label and the gold label differ for about 4% (274/7620). — [QuALITY §3](https://arxiv.org/abs/2112.08608)
- Self-reported context needed was only weakly separated by HARD status. 81.3% of HARD items vs 73.9% of EASY items were rated as needing "at least a long paragraph or two," and 26.1% vs 21.7% as needing at least a third of the passage. Raters judged HARD items answerable and unambiguous slightly less often (92.8% vs 95.1%). — [QuALITY §2.3.2](https://arxiv.org/abs/2112.08608)
- Agreement: median Krippendorff's alpha was 0.71 on answers and individual agreement with the majority on Q1 was 92.6%. The subjective Q2 and Q3 were noisy (alpha 0.12 and 0.21). Workers below 75% accuracy in a round were removed. — [QuALITY](https://arxiv.org/abs/2112.08608)
- Human accuracy, from fresh annotators taking a majority vote on 367 questions: 93.5% overall, 89.1% HARD, 97.0% EASY. — [QuALITY §3.1](https://arxiv.org/abs/2112.08608)
- Baselines: the best model (DeBERTaV3-large, RACE→QuALITY, with DPR-based extraction of passage excerpts) reached 55.4% overall and 46.7% on HARD. The question-only baseline (no passage) reached 43.3%, only 12.1 points below. Every model, including the question-only ones, scored lower on HARD, "suggest[ing] that speed-validation filtering yields more challenging questions for human annotators and models." Extraction using the oracle answer as the query topped out at 78.3%. — [QuALITY §4.2](https://arxiv.org/abs/2112.08608)
- Other levers: splits were made with minimal writer overlap across train/dev/test, so models cannot overfit one writer's style. Reasoning-type annotation of 500 questions found most items concern description, why/reason, or symbolism/interpretation, not entities. HARD and EASY did not differ by question word. — [QuALITY §3.2, 3.5, 3.6](https://arxiv.org/abs/2112.08608)

### Inferences
- The speed-validation pattern can be reused for LLM readers. Run a "skim proxy" (a model given only retrieved top-k chunks, or a 40-second-equivalent excerpt, or only the abstract) and keep items the proxy fails but a full-context careful reader passes. Writer-facing feedback (Q2 "how much of the text is needed") was cheap but noisy, so it should not be the only criterion.
- The 43.3% question-only score shows that difficulty filtering and artifact filtering are separate checks. Run a no-document baseline on every item.

### Gaps
- QuALITY does not report what fraction of HARD items still fall to retrieval-based LLM pipelines today. Later work on this was not checked.

## 2. ROPES (Lin et al. 2019): applying a passage to new situations

### Takeaway
ROPES splits each item into (a) a real expository background stating a causal or qualitative relation, (b) a crowd-written novel situation that instantiates the cause or effect for named entities, and (c) questions written as minimal pairs (e.g. "more" vs "less") whose answers flip. This makes lexical lookup useless and forces application of the relation. A model without the background still did fairly well, which shows the need for ablations.

### Cited Findings
- Backgrounds: passages scraped from science textbooks (ck12, OpenStax) and Wikipedia that contained causal connectives ("causes," "leads to") or qualitative cues ("increases," "decreases"). They were manually filtered to keep those with at least one relation, giving about 1,000 passages (49% physical science, 45% life science, 5% economics). — [Lin et al. 2019](https://aclanthology.org/D19-5808/)
- Crowdworkers (AMT) chose from 5 backgrounds per HIT, wrote situations involving the relation, and wrote 10 questions requiring both background and situation. "To mitigate the potential for easy lexical shortcuts... the workers were encouraged via instructions to write questions in minimal pairs, where a very small change in the question results in a different answer." Answers are spans of the situation or question. Total: 14,322 questions. — [Lin et al. 2019](https://aclanthology.org/D19-5808/)
- Example: the background says animal pollination is more efficient than wind. The situation has category A (wind) and category B (animal) flowers. Q: "Would category B have more or less efficient fertilization than A?" → more. The flipped version asks about A → less. Further questions ask which category is more or less likely to have bright petals. — [Lin et al. 2019, Fig. 1](https://aclanthology.org/D19-5808/)
- Analysis of 100 questions. Grounding the relation in the situation was explicit in 67%, needed common sense in 13%, and crossed a lexical gap in 20%. Question reasoning types: effect comparison, cause comparison, effect prediction, cause prediction. Comparisons dominate and are hardest because they need two grounded instances. — [Lin et al. 2019, Tables 3-4](https://aclanthology.org/D19-5808/)
- Results: RoBERTa-large +RACE scored 61.6% F1 on test against expert humans at 89.0%. Most questions have two sensible choices, so the model is "little better than randomly picking one of the choices." Table 5 includes "−background" ablations, and by our reading of the extracted table RoBERTa-large without background was within a few points of the full model. — [Lin et al. 2019, Table 5](https://aclanthology.org/D19-5808/)
- Train/dev/test used disjoint annotators (7/2/2). — [Lin et al. 2019, Table 1](https://aclanthology.org/D19-5808/)

### Inferences
- Direct template for AI-safety papers: background = the paper's claimed mechanism or result; situation = a new setup (a different model, training regime or threat model) that instantiates the cause; question = the direction of the effect, written as a minimal pair whose answer flips with one changed condition. Grade on pair consistency (both members right), not per item.
- The ROPES "−background" ablation corresponds to a "paper-removed" baseline. If the model can answer the situation question from priors alone, the item tests world knowledge, not the paper.

### Gaps
- The exact "−background" numbers could not be mapped reliably from the PDF text extraction. Check Table 5 in the paper before quoting them.

## 3. Contrast sets (Gardner et al. 2020) and minimally edited examples

### Takeaway
Dataset authors hand-perturb test instances in small, meaningful ways that usually change the gold label, then score "contrast consistency": the fraction of whole sets (original plus all perturbations) answered entirely correctly. Models dropped 15 to 26 points on reading-comprehension contrast sets, and consistency fell much further, while humans were essentially unaffected.

### Cited Findings
- Recommendation: "after a dataset is constructed... the dataset authors manually perturb the test instances in small but meaningful ways that (typically) change the gold label, creating contrast sets." The goal is to probe a model's local decision boundary. Contrast sets were built for 10 datasets, including DROP, Quoref, ROPES, BoolQ, MC-TACO, IMDb and UD parsing. — [Gardner et al. 2020](https://aclanthology.org/2020.findings-emnlp.117/)
- Cost: perturbing took 1 to 3 minutes per example on average (about 17 to 50 hours per 1,000 examples), and about 15 minutes for some harder datasets. "A person-week of work can yield high-quality perturbed test sets of approximately 1000 instances." — [Gardner et al. 2020](https://aclanthology.org/2020.findings-emnlp.117/)
- Results (original → contrast, then consistency): DROP (MTMSN) 79.9 → 54.2 F1 (−25.7), consistency 39.0. Quoref (XLNet-QA) 70.5 → 55.4 (−15.1), consistency 29.9. ROPES (RoBERTa) 47.7 → 32.5 EM (−15.2), consistency 17.6. BoolQ 86.1 → 71.1 (−15.0), consistency 59.0. MC-TACO 38.0 → 14.0 (−24.0), consistency 8.0. The UD parser had consistency 17.3 despite 95.7% UAS on the PTB. — [Gardner et al. 2020, Table 2](https://aclanthology.org/2020.findings-emnlp.117/)
- Human check: authors who did not write the examples answered at least 100 original/contrast pairs. Human performance was comparable (e.g. IMDb 94.3 vs 93.9, PERSPECTRUM 91.5 vs 90.3). — [Gardner et al. 2020, Table 3](https://aclanthology.org/2020.findings-emnlp.117/)
- DROP perturbations added compositional steps, inverted superlatives ("shortest" → "longest"), and changed event order and similar features. The authors recommend categorizing perturbation types up front so failures can be analyzed by type. — [Gardner et al. 2020](https://aclanthology.org/2020.findings-emnlp.117/)
- They deliberately kept models out of the loop because this "can bias the data to the failures of a particular model" instead of characterizing the decision boundary. Annotator time should go to perturbations that automation (e.g. paraphrasing) cannot produce. — [Gardner et al. 2020](https://aclanthology.org/2020.findings-emnlp.117/)

### Inferences
- For paper quizzes, a contrast set can perturb the question rather than the paper: "what if the authors had used X instead of Y," or "what if the effect had been measured on a model with property Z." The gold answer must follow from the paper's stated mechanism. A further option is to perturb the paper itself, editing one key result in context and checking whether the reader's conclusions update. This separates reading from recalling a memorized paper.
- Contrast consistency (all-or-nothing per set) is a stricter and more diagnostic score than per-item accuracy for rubric-graded quizzes.

### Gaps
- Contrast sets for long documents or full papers were not found in this search.

## 4. Counterfactual reasoning and reading datasets (CRASS, IfQA, TimeTravel)

### Takeaway
Counterfactual datasets build items around an explicit "if" premise that contradicts known facts, so the answer cannot be retrieved and must be derived. IfQA showed strong models (2023) scoring low (EM 27 to 36). Construction details beyond the basic pipeline were thinly covered in the sources opened.

### Cited Findings
- IfQA (Yu et al., EMNLP 2023): about 3,800 open-domain questions, each with an "if" clause counterfactual premise (e.g. the LA–Paris time difference if LA were on the US east coast). Pipeline: (1) automatically select Wikipedia passages suited to counterfactual questions, which substantially reduced annotation difficulty compared with random passages; (2) workers given 20 passages pick at least 10 and write Q/A, after seeing 5 example questions because pilots without examples were poor; (3) workers give answers and may add alternative valid answers; (4) additional workers validate. Results: supervised retrieve-then-read EM 36.2, GPT-3 chain-of-thought EM 27.4. — [Yu et al. 2023, IfQA](https://arxiv.org/abs/2305.14010) (search snippet level; numbers from the abstract and summary)
- CRASS (Frohberg & Binder): uses "questionized counterfactual conditionals" and scores models against a crowd-validated human baseline. Six state-of-the-art models (at the time) left considerable room for improvement. — [CRASS](https://arxiv.org/abs/2112.11941) (abstract only)

### Inferences
- IfQA's lesson for paper quizzes is to pre-select passages that support counterfactuals, meaning sections stating a mechanism or a causal claim. Writers produce better counterfactual items there than on arbitrary sections, and showing them a few worked examples raises quality (at some risk of templating).

### Gaps
- TimeTravel (Qin et al. 2019, counterfactual story rewriting) was not opened. Its construction (crowdworkers given an original story and an altered premise, rewriting the ending minimally) is from memory and unverified.
- CRASS item counts and human vs model scores were not retrieved.

## 5. Scientific-paper QA aimed at deep understanding

### Takeaway
The main ways to separate deep from shallow questions in this literature are: (a) the writer sees only the title and abstract (QASPER); (b) questions mined from peer reviews or rebuttals, with author-written answers (SciDQA, RPC-Bench); (c) expert-written MCQs with a rule against keyword-retrievable answers, filtered by an LLM panel and an evidence-only check (ELAIPBench); (d) what/how/why taxonomies (RPC-Bench). Headline finding: on ELAIPBench (2025) the best LLM scored about 40% and time-limited humans 48%. Thinking modes, CoT and intra-paper RAG often hurt.

### Cited Findings
- QASPER: 5,049 questions over 1,585 NLP papers. Questions were written by NLP practitioners who read only the title and abstract, seeking information in the full text. A separate group answered and provided evidence. Models trail humans by at least 27 F1 when answering from whole papers. — [Dasigi et al. 2021, QASPER](https://arxiv.org/abs/2105.03011) (abstract)
- SciDQA: 2,937 QA pairs. Questions come from expert peer reviews and answers from the papers' authors. Filtering, decontextualization, version tracking and a bibliography were added for multi-document QA. Questions require reasoning across figures, tables, equations, appendices and supplements. Scoring used surface similarity plus LLM judges. — [SciDQA](https://arxiv.org/abs/2411.05338) (abstract)
- ELAIPBench (Oct 2025): 403 MCQs (88 single-answer, 315 multi-answer) on 137 ML/CV/NLP papers. They were written by 20 annotators with AI publication experience in three roles: 10 writers, 4 evidence verifiers, 6 answer verifiers. Writers avoided highly cited papers. Guidelines bar answers obtainable by keyword matching or direct retrieval, so answers must require reasoning, summarizing, or combining sections. Distractors are partial truths. — [ELAIPBench](https://arxiv.org/abs/2510.10549)
- ELAIPBench validation pipeline, in order: (1) automatic review, where three small LLMs (GPT-4o-mini, Qwen2.5-14B, GLM-4-Flash) answer with the paper, and if any gets it right the question is revised as too easy; (2) evidence verification, where a verifier must reach the answer from the evidence excerpts alone; (3) answer verification, where a verifier answers from the paper within 20 minutes and the outcome sets difficulty (85 Easy, 109 Moderate, 209 Hard). No no-paper baseline was reported. — [ELAIPBench](https://arxiv.org/html/2510.10549)
- ELAIPBench results: best was DeepSeek-V3 (base) at 39.95%, GPT-5 38.71%, and Claude-3.7-Sonnet-thinking 38.46%. Humans (answer verifiers, 20-minute limit) scored 48.14%. Reasoning models often underperformed their base models ("reasoning paralysis," e.g. DeepSeek-R1 25.81% vs V3 39.95%), with over half of reasoning errors coming from "harmful verification" that overturned correct answers. CoT lowered accuracy for most models. Intra-paper RAG (BGE-m3, BM25) reduced accuracy, and web retrieval gave marginal gains. — [ELAIPBench](https://arxiv.org/html/2510.10549)
- RPC-Bench (Jan 2026): 4,150 papers (3,521 accepted with 50+ citations plus 722 rejected) and 61.3K QA pairs mined from OpenReview review–rebuttal exchanges. GPT-4o splits reviews into comment–response pairs, other LLMs rewrite them as QA or claim verification, and an LLM filters items not answerable from the paper alone (editorial issues, external resources, "empty promises"). Only val/test were human-annotated, and humans removed 8.87% as low-quality or unanswerable. Cohen's kappa was 0.72/0.78 for category and 0.81/0.85 for retention. Taxonomy: Concept Understanding; Methods (what/how/why/comparison); Experiments (exposition/setup/analysis); Claim Verification (true/false, about 25%). Depth is ordered what → how → why. — [RPC-Bench](https://arxiv.org/html/2601.14289)
- RPC-Bench results: of 28 models, GPT-5 was best at 68.2% "F1-like" (correctness–completeness), falling to 37.46% on informativeness, which penalizes verbosity. Page-image input cost multimodal models 4.74 to 36.1 points. Claim verification was weak. For LLM-judge scoring, decimal scores aligned better with humans than integer scores. No closed-book baseline was reported. — [RPC-Bench](https://arxiv.org/html/2601.14289)
- Other 2025-26 entries found but not opened: PaperMind (2604.21304, agentic critique and experimental interpretation); FrontierScience Bench (88 AI papers with methods redacted, where the model predicts the redacted methodological details); AcademicEval (live arXiv-based, leakage-resistant writing tasks). — [search results: PaperMind](https://arxiv.org/pdf/2604.21304), [FrontierScience Bench](https://aclanthology.org/2025.realm-1.31.pdf), [AcademicEval](https://arxiv.org/html/2510.17725v1)

### Inferences
- ELAIPBench's "if any of three weak LLMs gets it right, revise" filter is a cheap reusable stand-in for QuALITY's speed validation. Its evidence-only check confirms answerability and records the needed evidence span(s). Recording evidence spans also lets you measure how dispersed the needed evidence is, a proxy for whole-paper reasoning.
- Reviewer and rebuttal questions are a natural source of "edge case where the claim fails" and "why does this result follow" items for AI safety papers on OpenReview. RPC-Bench shows they need filtering for answerability from the paper alone.
- The finding that intra-paper RAG hurts on ELAIPBench is positive validity evidence: questions written against retrieval do resist chunk retrieval. Reporting a RAG-only baseline is worth copying.
- FrontierScience Bench's redaction approach (hide the method, ask the reader to predict it) is a ready form of a "counterfactual or extrapolation" item.

### Gaps
- PeerQA and SPIQA were not opened. Their construction details and numbers are missing.
- QASPER's exact human F1 and model F1 and its unanswerable rate were not retrieved, only the at-least-27 F1 gap.
- None of the opened science-QA papers report a closed-book (no-paper) baseline. Its absence is notable given memorization risk for published papers.

## 6. Long-document QA designed against retrieval, and the standard validation checks

### Takeaway
Long-document benchmarks distinguish short- or local-dependency questions (needle, single-hop) from long-dependency, multi-hop or global ones, and models consistently do much worse on the latter. Validation checks used across the papers opened: closed-book (no-text) baselines (NovelQA, QuALITY), retrieval or extraction baselines (QuALITY, ELAIPBench, LooGLE), evidence-position analysis (NovelQA), length-controlled padding (FlenQA), multi-annotator agreement (QuALITY, NovelQA), and human baselines.

### Cited Findings
- NovelQA: 2,305 QA pairs kept out of collected items (79.4% retained after three authors double-checked). Annotators were English literature students or enthusiasts who knew the novels well, writing 20 to 30 questions per novel. About half came from templates (mostly multi-hop) and half were free-form. Mix: multi-hop 35.0%, single-hop 42.8%, detail 22.2%. Other annotators answering MCQs on books they had read gave 94.6% kappa. Contexts average over 200K tokens. — [NovelQA](https://arxiv.org/html/2403.12766v3)
- NovelQA results: Claude-3.5-Sonnet scored 77.92 MC and 62.30 generative, GPT-4 71.80 MC and 46.88 generative, humans 97 MC and 90 generative. Multi-hop was hardest (about 30 to 33% generative for GPT-4 and Claude 2.1) and single-hop easiest (about 64 to 65%). With evidence past 100K tokens, GPT-4 fell from 53.64% to 32.01% and Claude 2.1 from 61.96% to 30.57%. The GPT-4 judge agreed with humans at 89.25% kappa. — [NovelQA](https://arxiv.org/html/2403.12766v3)
- NovelQA closed-book baseline (no novel): GPT-4 scored 60.94 MC and 34.30 generative, Claude 2.1 51.77 MC and 22.36 generative. Famous novels are likely in training data, so closed-book accuracy can be high. Test answers are withheld to limit leakage. — [NovelQA](https://arxiv.org/html/2403.12766v3)
- LooGLE: documents of more than 24K tokens published after 2022 (to reduce contamination), about 6,000 generated questions plus more than 1,100 human-written long-dependency QA pairs with "thorough cross-validation." LLMs did well on short-dependency tasks but struggled on long-dependency ones. Retrieval substantially helped short QA, while ICL, CoT and context-extension methods helped little. — [LooGLE](https://arxiv.org/abs/2311.04939) (abstract)
- FlenQA ("Same Task, More Tokens"): creates "multiple versions of the same sample, each being extended with padding of different lengths, types and locations," which isolates input length from task difficulty. Reasoning degrades at lengths far below the technical maximum. — [Levy, Jacoby & Goldberg 2024](https://arxiv.org/abs/2402.14848) (abstract)

### Inferences
- A validation battery to reuse for paper quizzes: (1) no-paper baseline (critical for well-known papers, see NovelQA's 61% closed-book MC); (2) abstract-only or intro-only baseline (the analogue of truncated context and QASPER's writer view); (3) top-k retrieval baseline (QuALITY DPR, ELAIPBench RAG); (4) full-paper human or expert answer with evidence spans and agreement; (5) contrast or minimal-pair consistency; (6) optionally an edited-paper check, changing a result and seeing whether answers follow the text rather than memory.
- Post-cutoff documents (LooGLE) or lesser-cited papers (ELAIPBench) reduce memorization confounds. The same applies to choosing AI safety papers.

### Gaps
- BABILong, LongBench (v1/v2) and NarrativeQA were not opened, so their construction and validation details (e.g. LongBench v2's expert time-limited verification) are not documented here.
- Shuffled-context baselines were not found in the sources opened.

## 7. Question-only and partial-input baselines as a validity check

### Takeaway
Partial-input baselines often do surprisingly well. Question-only reached 43.3% on 4-way QuALITY (vs 25% chance). Passage-only exceeded 50% on 14 of 20 bAbI tasks. For CBT, the last sentence alone suffices. Closed-book LLMs get 52 to 61% MC on NovelQA. Every quiz item should be checked against no-document and partial-document inputs.

### Cited Findings
- Kaushik & Lipton (EMNLP 2018), testing bAbI, SQuAD, CBT, CNN and Who-did-What: "question- and passage-only models often perform surprisingly well." On 14 of 20 bAbI tasks passage-only models exceed 50%. On CBT the last sentence alone gives similar accuracy to the full 20-sentence story. SQuAD and CNN appeared better constructed. — [Kaushik & Lipton 2018](https://arxiv.org/abs/1808.04926) (abstract)
- QuALITY question-only (options plus question, no passage) scored 43.3%, just 12.1 points below the best passage-using model, and lower on HARD. — [QuALITY](https://arxiv.org/abs/2112.08608)
- NovelQA closed-book: GPT-4 60.94% MC. — [NovelQA](https://arxiv.org/html/2403.12766v3)
- ROPES reports a "−background" ablation (Table 5). — [Lin et al. 2019](https://aclanthology.org/D19-5808/)

### Inferences
- For MCQs, the question-only signal comes mostly from option artifacts (implausible distractors, the longest option being correct). ELAIPBench's "partial-truth distractors" and QuALITY's writer feedback on the best distractor (Q3) are direct countermeasures. For rubric-graded open answers, a no-paper LLM answer graded against the rubric gives the equivalent baseline. Items it passes should be revised or dropped.

### Gaps
- The well-known hypothesis-only NLI results (Poliak et al. 2018; Gururangan et al. 2018) were not opened in this session, so no figures are given.
- No systematic 2024-26 survey of the fraction of benchmark items answerable without the source was located within the tool budget.
