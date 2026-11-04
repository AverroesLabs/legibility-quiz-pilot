# Plan: comparing candidate legibility measures

Status: draft, committed before any quiz, reader or obfuscation output exists. Jason has not signed off on it. It compares measures only and sets no final pass/fail bar. Thresholds below are proposals for that comparison, and anything decided after seeing results is exploratory.

## Why

Jason's note of 2026-11-02 (`INBOX.md`): a reader can gain knowledge from a paper while following only small parts of it. The audit should say "this paper is legible", not "this paper teaches the model something". So knowledge gain becomes a supporting measure unless this comparison shows otherwise.

This test cannot show that any measure tracks human understanding. "Partly obfuscated" is a label we create with a model, not a human judgment.

## Question and hypotheses

Which measure best answers "could the reader follow the whole paper?"

- **H1:** on papers with some sections obfuscated, M3 and M4 fall further below the clear original than M1 and M2 do, relative to their noise across quizzes.
- **H2:** the lowest-scoring section or claim points at an obfuscated section more often than chance.
- **H3:** on material the reader already knows, M1 is low even though the material is clear, while M2 to M4 stay high.

## Quiz structure

- The quiz is open book (Jason's decision): after reading, the reader has the paper while answering. See `docs/quiz-design.md`.
- The quiz writer works only from the clear original, without knowing which sections will be obfuscated, and sees the whole paper. It lists the paper's key claims and tags each with the section(s) that support it.
- Target 5 to 8 key claims, with at least 3 items each, plus items that need several sections together. Items need one inference beyond the text, not lookup.
- Items are filtered with Jason's four conditions (smart and dumb, with and without the paper) on the clear original only. Never drop an item because of the dumb reader's score with the paper. Items that a lookup baseline can answer are flagged, and results are reported with and without them.
- The same quiz is used across all versions of a paper, so differences come from the text.
- Each paper gets 3 quizzes from different seeds. Each reader condition is sampled at least 3 times per item.
- Items are free response with a rubric, graded by a model from a family other than the writer's. Grading gets its own check, since it can dominate the score.

## Candidate measures

| Id | Measure | Definition |
| --- | --- | --- |
| M1 | Mean gain | Mean of (with paper minus without paper) across items, also reported normalized as gain / (1 minus pre) |
| M2 | Mean post score | Mean post-reading score across items |
| M3 | Claim coverage | Share of key claims with a claim-level post score of at least 0.7 |
| M4 | Weakest section | Lowest post score among the paper's sections, for sections with at least 3 items |
| M5 | Integration | Mean post score on items that need several sections together |

The 0.7 in M3 is a starting value. Report M3 across 0.5 to 0.9 so the comparison does not depend on it.

## Test material

Each paper appears in these versions:

1. **Clear:** the original.
2. **One section obfuscated:** one section chosen by a seeded hash, rewritten to be illegible, with the rest verbatim.
3. **Half obfuscated:** half of the eligible sections, chosen the same way.
4. **Section removed:** the same section as version 2, deleted and marked as omitted. This separates a measure that responds to illegibility from one that only responds to missing information.

Only sections with at least one tagged claim are eligible, and the abstract and introduction are never chosen. They stay clear, so a summary alone could reveal claims. That is the realistic case and the one average scores are expected to miss.

**Controls:**

- **Placebo:** the reader takes the quiz with an unrelated paper of similar length. This measures score changes that do not come from the paper.
- **Information check:** a stronger model takes the quiz on each obfuscated version. If it also fails, the rewrite probably destroyed information. Such a version is reported, and excluded from the main analysis only if its score falls below 0.8 of the same model's score on the clear version.
- **Familiar material (H3):** 2 to 3 well-known, clearly written older papers, chosen by a rule to be committed first. The reader's pretest must be high for them, or they do not test H3.

## Analysis

Papers are the unit. Report results per paper, per stratum and per reader model, since prior knowledge and reader level may change whether clear and obfuscated versions separate.

1. **Separation:** for each measure, the paired difference between clear and each obfuscated version, divided by that measure's between-quiz standard deviation on clear papers. Also AUROC for clear against obfuscated, and the detection rate when at most 1 in 6 clear papers would be failed.
2. **Localization (H2):** how often the lowest section is an obfuscated one in version 2, compared with chance (1 divided by the number of eligible sections).
3. **Illegibility or missing information:** compare versions 2 and 4 for each measure.
4. **Reliability:** intraclass correlation across quizzes on clear papers. A measure that detects well but does not repeat is not usable.

Bootstrap over papers for confidence intervals, with 10,000 resamples and seed 20261102.

## Decision rule (proposed)

Prefer the measure with the best separation on version 2, provided its between-quiz reliability is at least 0.5. If the intervals of two measures overlap, prefer the simpler or report both.

With 12 papers, the intervals will be wide. This comparison picks a measure to carry forward. It does not validate it. Choosing the winner on these papers also inflates its apparent performance, so the chosen measure needs confirming on papers not used here. It is not a substitute for bars or a human study.

## Weaknesses

- A weak rewriter may produce obfuscation that is easy to read, and a strong one may delete content. The information check and the section-removed version only partly address this.
- Obfuscation style is a model's idea of illegible and may not match what makes real papers hard for people.
- Section-level scores rest on few items, so M4 may be noisy. Reliability should show this.
- Claim tags come from the quiz writer, and can be wrong. Localization inherits those errors.
