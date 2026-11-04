# Quiz design: prior work and proposal

Written 2026-11-02. The proposal is untested. Figures were read through a web-fetch summary of the first 100,000 characters of each paper. On 2026-11-03 a subagent re-checked them through the same kind of fetch, and the corrections are below. Sedlmayr was read in full; Savaal, KnowledgeGain and greCAPTCHA only in part (about 100,000 characters each). Kintsch, Long 2006 and McNamara and Kintsch 1996 could not be opened (403, unreadable PDF), so treat those as unverified hypotheses.

## Prior work

No one we found has built a quiz to audit whether a paper is legible. The parts exist.

| Work | What it does | Useful for us | Warning for us |
| --- | --- | --- | --- |
| [KnowledgeGain](https://arxiv.org/abs/2605.31099) | Six questions from a paper's abstract (2 true/false, 2 extractive multiple choice, 2 inferential), asked before and after reading science news | Pre/post scoring, an "I do not know" option, and keeping only questions answerable from the source | Simulator gain correlated only 0.263 with humans (Table 3, full simulator; which level of aggregation was not confirmed), and the authors say it does not reliably predict topic-level human gain and is a proxy for relative comparison. It covers abstracts, not whole papers |
| [greCAPTCHA](https://arxiv.org/abs/2609.20481) | Eight questions testing authors on their own paper, across four families, graded by rubric | Free-response failure-mode questions; rubric grading | The one multiple-choice family (planted errors) was too easy to search: AUC 0.595, against a mean of 0.934 for the free-response families and 0.90 for all four together. 31 participants. Participants complained of minor-detail questions, unclear wording and rigid grading |
| [Savaal](https://arxiv.org/html/2502.12477v2) | Extracts and ranks main ideas by section, retrieves passages, writes questions per idea | The pipeline, which gives coverage by design | Experts preferred it by 6.5x on dissertations (61.9% vs 9.5%) but 1.5x on conference papers (34.5% vs 21.8%): preference rates, not performance. GPT-4o as judge was misaligned with human experts |
| [ELAIPBench](https://arxiv.org/pdf/2510.10549) | 403 expert-written multiple-choice questions on 137 AI papers | Possible benchmark for quiz quality | Tests models, not papers |
| [SciDQA](https://arxiv.org/html/2411.05338v1) | 2,937 question-answer pairs from reviewer-author discussions | Examples of questions that test deep reading | Tests models, not papers |

## What to test

Test what a reader must rebuild to have followed the paper:

| Aspect | Example question | Why |
| --- | --- | --- |
| Main claims and their exact scope | On which benchmarks did the method work, and under what conditions? | Translations lose scope, as in "three of four" becoming "benchmarks" |
| The argument | Why does this evidence support this conclusion? | Separates following from memorizing |
| Method and its purpose | How does this design rule out the main alternative explanation? | Shows the reader knows what was done and why |
| Reading the results | If this baseline had scored higher, what would change in the conclusion? | Tests transfer, and cannot be found by searching |
| Limits | What does the paper not show? When would the result fail? | Stops over-generalizing |
| The paper's own terms | Apply this paper's defined term to a new case | Catches jargon the reader can only repeat |

Do not test:

- **Background knowledge not in the paper.** That measures the reader, not the paper. greCAPTCHA uses this family because it tests authors.
- **Minor details.** Failing one says little about whether the reader followed the paper.
- **Anything a good paper might omit.** A paper that does not state a limitation may be weak, but not illegible. Whether that should count is an open question.

## Proposed pipeline

1. **Extract claims.** Pull key claims by section, and record which claims depend on which.
2. **Write questions.** Use free response with a rubric, one aspect per question, with at least one question per key claim.
3. **Filter questions.** Keep a question only if it passes all of these:
   - It passes the four-condition test below, which catches bad questions and questions answerable from prior knowledge.
   - A second writer's rubric agrees on the answer.
   - A lookup baseline cannot answer it. See "Open book" below.
   - It is not harder than necessary. See "Difficulty" below.
4. **Take the quiz open-book.** Jason's decision: the reader has the paper while answering. The questions must be what requires understanding.
5. **Score per claim.** Report coverage and where understanding breaks down, with gain as a supporting number. See `INBOX.md`.

## Core design: four conditions

Jason's decision: base question writing on existing open-book research, and test both a smart and a dumb model, before and after seeing the paper.

| Condition | Expected | If not |
| --- | --- | --- |
| Smart, before | Low | High: the question tests general knowledge, not the paper. Drop it |
| Smart, after | High | Low: the question or answer key is wrong, or the paper's content is not there to be understood. Drop or review it |
| Dumb, before | Low | High: the question does not measure the paper for this reader. Drop it |
| Dumb, after | The audit result | Low on a valid question: the paper may be illegible to this reader, or the question may be too hard. See "Difficulty" |

Rules:

- **The smart model taking the quiz must not be the one that wrote it.** Otherwise a high score may only show that it agrees with itself.
- **Never drop questions based on the dumb model's after-reading score on the paper being audited.** Dropping the questions it fails would inflate the score.

What this cannot show: a before-to-after rise shows an answer came from the paper, not that it matters. Key-information coverage needs its own checks. Proposed, untested:

- **Reconstruction.** Give a model only the questions and answers, and ask it to state the paper's main claims. Compare against the paper.
- **Section removal.** Remove a section from the paper. Questions on that section should fail for the smart model. A key section with no affected questions is a coverage gap.
- **Independent claim lists.** Two smart models list key claims separately, and each claim on either list needs a question.

## Difficulty

The quiz should need understanding of the paper and little more. If it asks too much, a dumb model can fail on a legible paper. Questions must therefore sit in a narrow band: too easy and lookup answers them, too hard and they measure intelligence.

Proposed, untested:

- **One step beyond the paper.** Each question should need one inference from what the paper says, using only the paper's own reasoning and basic arithmetic. No chains of inference, outside knowledge or trick wording.
- **Explanation check.** Give the dumb model a very clear explanation of the paper's point, written by the smart model, then ask the question. If it still fails, the question exceeds its ability regardless of the writing, so drop it. Keep the answer itself out of the explanation, or the check only tests copying. This check uses a different text from the one being audited, so it does not break the rule above.
- **Calibrate on a clear text.** Run the quiz method on papers or explainers that are widely considered clear. If the dumb model fails most of them, the method is too hard.

Without human data this can only show that a question is within the dumb model's reach. It cannot show that the dumb model matches a person.

## Open book

A correct answer must require understanding the paper, not just finding its text. Reading alone should not be enough.

Ask questions whose answers are not written in the paper but follow from it:

- **Change a condition.** If this assumption failed, which results would no longer hold?
- **Connect sections.** Which part of the method makes the limit described in the discussion necessary?
- **Apply it to a new case.** Given these new numbers, what would the paper's method conclude?
- **Spot a wrong version.** Why would this altered argument fail, given the paper's reasoning?

Proposed lookup baselines, to test whether questions meet this bar (untested):

- **Scrambled paper.** Give a reader the paper with its sentences or paragraphs shuffled, which keeps the facts but breaks the argument. If it can still answer, the question tests lookup.
- **Retrieved passages.** Give a reader only the passages a keyword search returns for the question. If that is enough, the question tests lookup.

A question can also be too hard: it should stay answerable from the paper alone, with no outside knowledge. The strong-model check tests this, but a question that only a strong model can answer may reflect the reader's ability rather than the paper's legibility. The clear and obfuscated versions of the same paper separate these: the score should fall when only the writing gets worse.

## Research on open-book testing

Searched 2026-11-02. A short search, not a systematic review. All of this is about human readers.

| Source | Finding | Use for us | How far we checked it |
| --- | --- | --- | --- |
| Kintsch's comprehension model ([summary](https://aura-lab.siue.edu/theories/model-of-text-comprehension/), [Long 2006](https://sites.uw.edu/ccdl/files//2013/12/Long_Assessing-text_2006.pdf)) | Separates the text's content (textbase) from a mental model of what it describes (situation model). Recall and text-based questions mainly reflect the first; inference and problem-solving questions rely on the second | The frame for the quiz: target the situation model | Search summaries only. Could not open on 2026-11-03 |
| [McNamara and Kintsch 1996](https://doi.org/10.1080/01638539609544975) | High-knowledge readers can do better on deep measures with low-coherence text, while low-knowledge readers benefit from coherent text. Later studies are mixed | Warning: the reader's prior knowledge can change or reverse the effect of clear writing | Search summaries only. Publisher returned 403 on 2026-11-03 |
| [Sedlmayr and Weissenbacher 2025](https://frontiersin.org/journals/education/articles/10.3389/feduc.2025.1524561/full) | Having the text available changes what the test measures. Conditions correlated r = 0.34 (95% CI 0.10 to 0.54) after item selection. Only the text-available test correlated significantly with verbal intelligence (0.24) and language grades (-0.26; lower grades are better, so this is a positive relationship) | Some support for open book | Opened. N = 107, with condition confounded with order |
| Medical school study, [cited here](https://scholarhub.vn/publication/Assessment-of-factual-recall-and-higher-order-cognitive-domains-in-an-open-book-medical-school-examination/cc89df5f-45ba-445a-ac47-ce1f03bd3335) | Open-book access helped factual recall more than higher-order questions | Supports asking higher-order questions | Could not open the study |
| Teaching guides: [Waterloo](https://uwaterloo.ca/keep-learning/node/36), [ECU](https://intranet.ecu.edu.au/__data/assets/pdf_file/0011/868511/Designing-open-book-questions-for-an-exam-alternative.pdf), [Indiana](https://citl.indiana.edu/fall2020/guides/openbook.html) | Use apply, analyze and evaluate questions, "what if" extensions, and explained reasoning | Practical patterns | Search summaries. Advice, not controlled evidence |
| [Two-tier questions](https://peer.asee.org/24516), [concept inventories](https://arxiv.org/html/2505.02850v1) | Ask for a prediction, then a justification. Build wrong answers from real misconceptions | Stops right answers from wrong reasoning. Reader failures could supply wrong options | Search summaries only |

Limits for us:

- **Models are not students.** Open-book research assumes readers are limited by time and memory. A model has the whole paper in context, so these findings may not transfer. Test them on models.
- **Models can answer from general knowledge.** A what-if question may be answerable without the paper. The no-paper filter must catch this.
- **The reader's knowledge matters.** Clear and obfuscated versions may not separate for a reader who already knows the field, so test this at each reader level.

Proposed additions, untested: label each question by level (textbase or situation model), and require a prediction plus a justification in free-response questions.

## Known risks

- **Reader validity.** KnowledgeGain's weak per-topic correlation is direct evidence that simulated readers may not track people paper by paper. This is the largest risk to the project.
- **Validating quiz quality.** Savaal suggests AI judges alone cannot do it. It will likely need people at some point, for example paper authors rating questions, as Savaal did.
- **Grader leniency or rigidity.** Grading can dominate the score, so test it separately.
- **Length.** Six to eight questions may be too few for per-claim coverage of a whole paper.
