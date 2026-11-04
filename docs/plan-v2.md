# Plan v2: whole-paper legibility test

Agreed with Jason on 2026-11-03. This is the design for the next run, and it supersedes `docs/experiment-plan.md`, `docs/measure-comparison-plan.md` and `docs/preregistration.md` wherever they conflict. Those stay as the record of the pilot. Reasons and sources are in `docs/decisions.md`, `docs/pilot-report.md` and `reports/Open book comprehension test design.md`.

## The question

**The unit that matters is one paper.** The final tool gives a verdict on a single paper, so every paper needs enough kept questions for its own result to be reliable. The 12 papers show whether the signal holds across papers; they do not replace per-paper accuracy.

If a whole paper is made harder to read, do weaker LLM readers' open-book quiz scores fall, while a smart reader's do not? The quiz must measure understanding of the whole paper (methods, key results and conclusions), not the ability to find passages.

Per-section legibility is a follow-up. Do not test or score sections in this run.

## Papers

The 12 core papers in `data/selection/selection.json` (4 per stratum). The 6 pilot papers are among them. Develop on reserve paper 2607.05615v2 first, so tuning cannot bias the core papers.

## Versions of each paper

| Version | What it is |
|---|---|
| No paper | The floor: readers answer with no paper |
| Original | The paper as published. Assumed legible, since it was written for people |
| Obfuscated | The whole paper, same content and length, made hard to follow: ideas out of logical order, long nested sentences, no signposting, named terms replaced by abbreviations defined once in a distant parenthesis. Stronger than the pilot's rewrite, which readers read straight through |
| Compressed | The whole paper at about a third to half the length: every claim, number and result kept; explanation, motivation, examples and repetition removed |
| Compressed + obfuscated | The compressed version, then obfuscated |

No section removal and no section-level versions. Versions do not depend on the quiz, so make them in parallel with quiz writing.

**Information check:** every version must keep all facts. A script checks that every number in the original survives. The smart reader with each version must score at least 0.9 of its score with the original on kept questions. A version that fails does not count, and is reported.

## Questions

The core rule: **each answer is implied by the paper but stated nowhere in it**, so no retrieved passage contains it. Each answer must follow from mechanisms, ablations or numbers the paper states, so no extra intelligence or outside knowledge is needed.

1. **Claim map first.** The writer (`anthropic/claude-opus-5.5`) maps the paper's main claims to the methods, results and stated conditions behind each one.
2. **Six question types,** covering methods, key results and conclusions, with most questions linking two or more parts of the map:
   - **Failure mode or edge case:** when would this result stop holding, and why?
   - **Why a result follows:** only where the paper does not state the reason.
   - **Counterfactual pair:** two near-identical questions with opposite answers. Credit only if both are right.
   - **Transfer:** apply the paper's findings to a new model, setting or case.
   - **Connecting distant parts:** which result would be undermined if a different finding had come out the other way?
   - **Scope:** is this plausible over-generalization supported, contradicted or not settled by the paper?
3. **Short prompt, one worked example per type, one aspect per question.** Free response only, no multiple choice.
4. **Target at least 40 kept questions per paper.** By the error-bar formula in the literature review, about 30 detect a 15-point drop and about 70 a 10-point drop, for one paper. Write about 120 candidates per paper, since filtering should reject many, and more if survival is low in development. If a paper cannot yield 40 genuinely different implied-not-stated questions without filler, record that: it limits the tool.
5. **Rubric:** 2 to 3 binary points per question: the conclusion, then the reason, optionally the condition. Each point says what the answer must assert, with acceptable paraphrases.

## Filtering questions

Run on the original version only. Keep a question only if it passes all of these:

| Check | Reader and input | Keep if |
|---|---|---|
| Without the paper | Every reader, question only | Smart reader at most 0.5; weaker readers' average at most 0.5 |
| With the paper | Smart reader, full original | At least 0.5 |
| Retrieval | `google/gemma-4-31b-it` (the stronger audited reader), given only the 3 paragraphs a keyword search returns | At most 1 of 3 points |
| Counterfactual pairs | Smart reader, full original | Both twins right |

Never drop a question based on a weaker reader's with-paper score: that is the audit result. Do not filter on an AI judge's opinion of question quality. Leak screening during writing (a model answers without the paper, and leaked questions are rewritten) may continue, using `moonshotai/kimi-k3`.

## Readers

- **Smart:** `openai/gpt-5.6-sol`.
- **Weaker:** `google/gemma-4-31b-it`, `mistralai/ministral-8b-2512`, `meta-llama/llama-3.2-3b-instruct`.

Each reader answers the quiz per version: no paper, then each of the 4 versions. Open book, paper text with figure captions. Split long quizzes into sittings of at most 25 questions, each with the full paper, since weak readers broke down on 27 questions in one sitting. **3 samples per condition**: more questions per paper help more than more samples.

## Grading

- `z-ai/glm-5.3` at reasoning effort `minimal` (already in `scripts/quiz/quiz.py`). Checked against full reasoning on 616 pilot answers: same score 88.6%, kappa 0.90, no version shifted by more than 0.03.
- The grader gets the question, rubric, model answer and reader's answer, not the paper. Blind to reader and version.
- The grader quotes the part of the answer that earns each point.
- Score "I don't know" as 0 without a grader call.
- Cap answer length, since graders favour length.
- Report "right conclusion, wrong reason" rates per reader and version.
- **Grader bias check:** the grader must not be more lenient on one version than another. Report agreement by version.
- **Human calibration:** prepare 60 answers spread across readers and versions as a grading sheet for Jason (`docs/human-grading-sheet.md`), and add it to "Waiting on Jason". Do not block the run on it.

## Analysis

- **Main score:** per reader and version, the mean rubric score over kept questions. Show it as a position between no paper (0) and the original (1).
- **Main figure:** one line per reader across the five conditions, with 95% intervals.
- **Model:** score ~ version + (1 | question) + (1 | paper), per reader, plus a pooled model with version × reader. Intervals by bootstrap over questions within papers.
- **Per-paper result first:** for each paper and reader, the drop from original to each version with a 95% interval over that paper's questions. Report how often a single paper's result is conclusive.
- **Questions needed:** report kept questions per paper, the spread between questions within a paper, and how many kept questions one paper needs for a 10-point and a 15-point drop.
- **Bars:** draft them in `docs/preregistration-v2.md` and commit before any core-paper result exists. Label them "proposed, awaiting Jason's sign-off". The run may proceed against them, with results reported as provisional.

## Baseline: just ask a model (Jason, 2026-11-04)

If a model can simply judge how legible a paper is, the quiz protocol may be unnecessary. Exploratory, added after the preregistration, so label it as such.

- **Direct judgment:** each of the 4 readers rates each of the 4 versions of every paper, one version per call, never side by side: "How easy is this paper for a reader at your level to understand, from 1 to 10?" with anchors for 1, 5 and 10, plus a short reason. 3 samples each.
- **Readability formulas:** Flesch-Kincaid grade and similar, computed by script, at no cost.
- **Compare with the quiz, per paper:** does each method rank the original above the obfuscated, compressed, and compressed + obfuscated versions, and how reliably across samples and papers? Also, does each method agree with the quiz on which papers are hardest?
- **Interpretation to report:** the obfuscation is stylistically obvious, so a judge may easily spot it. That would show the judge detects visible difficulty, not that it measures understanding. A judge can also be fooled by fluent prose that hides gaps, which is the gaming risk an audit must resist. Without human ratings, neither method can be shown to track people.
- **Figure:** judge rating and quiz score side by side for each reader and version, per paper.

## Speed

Run all papers at once; make versions in parallel with quiz writing; start each reader on each version as soon as quiz and version exist; grade each answer as it arrives; screen questions in parallel with capped tokens and timeouts; limit concurrency per model provider; use shorter timeouts on short calls; log time per stage per paper and report the total. None of this may change what is measured.

## Budget and report

- **Budget:** $500 (raised from $300 by Jason on 2026-11-04), in `STATE.md`. Claude's rough estimate for 12 papers at about 120 candidates and 40 kept questions each is $200 to $280, so it is tight. **If the budget binds, cut papers, keeping strata balanced, never questions per paper.** Measure the cost per paper on the first papers and decide the paper count before running the rest.
- **Report:** `docs/report-v2.md`, started at the first real result and kept current. Same structure and rules as the pilot report: a short "How the experiment works" section with diagrams first, and plain statements of what failed.

## Figures (Jason: show all the data, in a clear, straightforward series of figures)

Every result is a figure, in light and dark versions, generated from the raw data by one script. Tables only add detail. Lead with absolute scores on a fixed 0 to 1 axis, with drops shown alongside, so the reader can see whether a model was near its floor or the quiz near ceiling (Jason, 2026-11-04). At least:

1. **Question funnel:** questions written, then how many each filter removed, in order, and how many were kept. One overall, and one per paper (small multiples).
2. **Kept questions by type:** how many of each of the six types survived, and the survival rate per type.
3. **Main result:** score for each reader across the five conditions (no paper, original, obfuscated, compressed, compressed + obfuscated), with 95% intervals, all papers pooled.
4. **Every paper:** the same chart for each of the 12 papers (small multiples), so no paper is hidden in an average.
5. **Position between floor and original:** each reader's score in each version on a 0 (no paper) to 1 (original) scale.
6. **Information check:** the smart reader's score on each version relative to the original, per paper, with the 0.9 line, and which versions failed.
7. **Version lengths:** each version's length relative to the original, per paper.
8. **Score by question type:** per reader and version.
9. **Right conclusion, wrong reason:** rate per reader and version.
10. **Grader checks:** agreement by version, and the human-calibration agreement once Jason has graded the sheet.
11. **Per-paper reliability:** kept questions per paper, the 95% interval width of each paper's result against its number of questions, and how many questions one paper needs for a 10-point and a 15-point drop.
13. **Every run, paper by reader (Jason):** a grid with one row per paper and one column per reader; each run's score (one sample's mean over kept questions) as a point on a 0 to 1 axis, colored and shaped by condition, all five conditions, lightly jittered. Spec in `INBOX.md` (2026-11-04).
14. **Quiz vs. just asking a model (Jason, required):** for each reader, the direct legibility rating (1 to 10) and the quiz score in each version, side by side, pooled and per paper; plus how often each method ranks the original above each hard version. A required report section, "Quiz vs. asking a model", near the main results.
12. **Spend and time:** cost per stage, and wall-clock time per stage per paper.

## Follow-up: fluent versions with a hidden gap (Jason, 2026-11-04)

The plan v2 rewrites were visibly awkward, so a model's legibility rating caught them as well as the quiz did. The case for a quiz is a paper that reads fluently but leaves the reader without what they need. Two new versions per core paper, both fluent, polished prose of about the original length:

| Version | What changes | What it tests |
|---|---|---|
| **Fluent, insight removed** | The paper's crucial insight and the key results that depend on it are removed, and the text is smoothed over so nothing looks missing | Completeness. A positive control: the quiz should catch it, and a judge rating the prose should not |
| **Fluent, insight implicit** | Every fact stays derivable, but the crucial insight, key definitions and the reasoning steps that connect method to result are left implicit, as an expert writing for experts might | Legibility in the sense that matters for humans. The strong reader should still pass (information check); weaker readers should not |

Rules:

- Reuse each paper's existing kept quiz (`data/runs/v2-<paper>/kept-v2.json`); do not write or filter new questions. Readers, samples (3), grading and the judge (both prompts) as in plan v2.
- Version writer: Claude Opus 5.5 identifies the crucial insight and writes both versions, with the identified insight recorded in `versions/meta.json`. Check fluency: the human-prompt judge should rate both versions close to the original; report it either way.
- **Information check applies to "insight implicit" only.** "Insight removed" is expected to fail it, and that is the point.
- **Commit predictions before running** (in `docs/preregistration-v2.md`, a new section): judge ratings for both versions within 1 point of the original; quiz drop on "insight removed" for every reader, including the strong one; quiz drop on "insight implicit" for weaker readers larger than for the strong reader.
- **Budget:** within the existing $500 (stop at $639 account usage). Estimated about $50.
- **Report:** add a section and figures to `docs/report-v2.md`. Put the result next to the earlier rewrites so the four kinds of difficulty can be compared directly. Then notify Jason.
