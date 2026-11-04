# Preregistration: plan v2 whole-paper run

**Status: proposed, awaiting Jason's sign-off.** Drafted 2026-11-03 (pass 16), before any plan v2 result on a core paper exists. Only the reserve dev paper 2607.05615v2 has been run, and its takes were still running when this was committed. The run may proceed against these bars; results judged by them are provisional until Jason signs off. Anything decided after core-paper results exist is exploratory and labelled so. These bars judge whether this run shows signal worth scaling, not whether the method is valid: there is no human baseline.

Design, readers and filters are in `docs/plan-v2.md` and `docs/decisions.md`. Code: `scripts/quiz/run_v2.py` (pipeline), `filter_v2.py` (filters), `takes_v2.py` (takes, grading, per-paper summary).

## Fixed analysis choices

- **Unit:** one paper. Each paper's result is computed on its own kept questions.
- **Score:** a question's score in a condition is the mean over 3 samples of its rubric score (points earned / points). A counterfactual twin scores 0 in a sample unless both twins earn their conclusion point (rubric point 1) in that sample.
- **Conditions:** none (no paper), orig, obf, comp, compobf. Sittings of at most 25 questions; the split depends on reader and sample only, so all conditions use identical sittings.
- **Grading:** GLM-5.3 at minimal reasoning with `GRADE_V2` (quotes per point), answers cut to 100 words, "I don't know" scored 0 without a call. Failed or unanswered questions are counted and reported; an unanswered question scores 0.
- **Per-paper drop:** for reader R and version V, drop = mean over questions of (orig − V), with a 95% bootstrap interval over questions (10,000 resamples, seed 20261103).
- **Position:** (V − none) / (orig − none), reported only where orig > none.
- **Information check:** a version counts for a paper only if the smart reader (`openai/gpt-5.6-sol`) scores at least 0.9 of its original score on it. Versions that fail are reported and excluded from the bars below for that paper.
- **Pooled model:** per reader, score ~ version + (1 | question) + (1 | paper); and pooled, version × reader. Intervals by bootstrap over questions within papers. This is secondary to the per-paper results.

## Paper gate

A paper enters the bars only if it keeps at least 30 questions (the plan's target is 40; 30 detects a 15-point drop for one paper) and the smart reader scores at least 0.6 on the original. Papers below the gate are reported, not dropped silently.

## Proposed bars

Let "weaker readers" be Gemma-4-31B, Ministral-8B and Llama-3.2-3B.

1. **Per-paper detection.** A paper detects obfuscation for weaker reader R if R's drop orig − obf is at least 0.10 and its 95% interval excludes 0. Same rule for compobf.
2. **Dissociation.** On the same paper, the smart reader's drop orig − obf is smaller than R's, and the smart reader passes the information check on obf.
3. **Signal across papers.** "Initial signal" if, for at least one weaker reader, bars 1 and 2 hold for obf in at least half of the gated papers with a valid obf version.
4. **Brevity is not penalised as illegibility (descriptive).** Report orig − comp for each weaker reader. If comp drops as much as obf for a reader on most papers, the quiz is likely sensitive to length or missing explanation rather than to the confusing rewrite, and that is reported as a concern. No pass/fail.
5. **Per-paper reliability (descriptive).** Report kept questions per paper, the interval width of each paper's drop, the fraction of paper × reader results that are conclusive (interval excludes 0), and the questions one paper needs for a 10-point and a 15-point drop given the observed spread.

## Outcomes

- Bar 3 holds: "initial signal for whole-paper obfuscation", with the readers and papers it holds for.
- Papers pass the gate but bar 3 fails: a negative for this obfuscation, these readers and this quiz design.
- Fewer than half the papers pass the gate: "question writing is still the bottleneck".
- Every outcome is a pilot result on at most 12 papers, not a test of whether scores track human understanding.

## Follow-up: fluent versions with a hidden gap (predictions committed before any run)

Committed 2026-11-04 (pass 21) before any "insight removed" or "insight implicit" version, take or rating exists. Design in `docs/plan-v2.md` ("Follow-up: fluent versions with a hidden gap"); code `scripts/quiz/gap_v2.py`. These are predictions, not pass/fail bars (final bars need Jason). Anything decided after the first result is exploratory.

**Versions** (per core paper, Claude Opus 5.5): `rem`, the crucial insight and the results that depend on it removed and smoothed over; `imp`, every fact kept but the insight, key definitions and connecting reasoning left implicit. Both fluent, 0.8x to 1.25x the original's length, same section headings. `imp` must keep every number of the original; `rem`'s lost numbers are recorded, not enforced.

**Fixed analysis:** as for plan v2 above (kept questions reused unchanged, same sittings, 3 samples, `GRADE_V2`, pair rule, per-paper drop with bootstrap over questions, pooled drop with bootstrap over questions within papers, same paper gate). Original and no-paper scores come from the existing `takes-v2.jsonl`; new conditions go to `takes-gap.jsonl`. Judge: both prompts (`JUDGE_V2`, `JUDGE_HUMAN_V2`), 4 readers x 3 samples, compared with the existing original ratings.

**Predictions:**

1. **Fluency (judge).** For each reader and each judge prompt, the mean rating of `rem` and of `imp` over papers and samples is within 1 point of the original's. Primary: the human-reader prompt. Also reported: the share of papers where each reader's paper mean is within 1 point.
2. **Insight removed lowers every reader's quiz score.** Pooled drop orig − rem is above 0 with a 95% interval excluding 0 for each of the four readers, including the smart reader (`openai/gpt-5.6-sol`). Expected caveat, stated now: Llama-3.2-3B scores near its no-paper floor on the original, so its drop may be too small to detect; a null for Llama alone is reported as uninformative, not as a failed prediction. Per paper: the count of papers where the drop's interval excludes 0, per reader.
3. **Insight implicit separates readers.** The smart reader passes the information check (score on `imp` at least 0.9 of its original score) on most gated papers, and for Gemma-4-31B and Ministral-8B the pooled drop orig − imp minus the smart reader's drop is above 0 with a 95% interval excluding 0. Per paper: the count of papers where it is above 0.
4. **The contrast that matters (follows from 1 to 3).** On `rem`, the quiz shows a conclusive drop where the judge shows less than a 1-point drop. If the judge also drops by more than 1 point on `rem`, the follow-up does not separate the quiz from asking a model, and that is reported as the result.

**If the version writer fails** (length, headings or numbers), the paper's failure is reported and the paper is excluded from that version's analysis; no paper is swapped in.

## Follow-up 2: a quiz written on the hidden-gap version (predictions committed before any run)

Committed 2026-11-04 (laptop session, Jason's request) before any call of this run. Jason's point: in real use the quiz is written from the paper being audited, so the follow-up above (quiz written on the original, taken on `rem`) gave the quiz an advantage it would not have. This run writes and filters a new quiz on each paper's existing `rem` version, exactly as plan v2 did on the original (`write_v2`, `filter_v2`, same models, prompts, filters and seeds; the `rem` text plays the role of the paper), then all four readers take the kept questions with the `rem` text and with no paper (3 samples, `GRADE_V2`). Code `scripts/quiz/gap_requiz.py`; run directories `data/runs/v2rem-<paper>/`. Budget: stop at $839 account usage (Jason raised it by $200 on 2026-11-04).

**Compared with:** the plan v2 quiz written on the original and taken on the original (`data/runs/v2-<paper>/`).

**Predictions (descriptive, no pass/fail bars):**

1. **Yield.** The `rem` quiz keeps fewer questions than the original quiz on most of the 12 papers, because the version has less content to ask about.
2. **No detection.** For Gemma 4 31B and Ministral 8B, the pooled with-paper score on the `rem` quiz is within 0.05 of their score on the original quiz (both taken with their own paper), and per-paper differences have no consistent sign. If so, a quiz written from a fluent paper missing its key results does not reveal the gap. A clearly lower score on the `rem` quiz would be a signal we did not expect, and would be reported as such.
3. **Strong reader.** GPT-5.6 Sol scores about as high on the `rem` quiz as on the original quiz, since both are filtered to questions it passes with the paper.

**Also reported:** the no-paper score on each quiz (how much of each quiz is answerable from general knowledge) and the position of the with-paper score between no-paper and the strong reader. Paper 2607.10112v1 had only 11 original kept questions (writer content filter); it is reported but flagged.

## Follow-up 3: a stronger "insight implicit" version (predictions committed before any run)

Committed 2026-11-04 (laptop session, Jason's request) before any call. Why: the first `imp` version moved nothing, most likely because it was too mild; its 0.8x length floor forced the writer to keep explanation, and 5 of 12 papers failed it. This version (`imp2`, prompt p10 `GAP_IMPLICIT_STRONG`) drops the floor (allowed 0.4x to 1.25x, only to catch broken output), forbids defining the key terms in the insight note, forbids explaining why results follow, and removes intuition, examples, signposting and summaries. It must keep the section headings and every number (script-checked; up to 3 tries; a paper whose version fails all 3 is excluded and reported). Writer Claude Opus 5.5, from the insight note already recorded for each paper (no new insight call). Code `scripts/quiz/gap_imp2.py`.

**Measured:** all four readers take each paper's existing kept quiz (written on the original; valid because every fact is kept) with `imp2`, 3 samples, same sittings, `GRADE_V2` (`takes-imp2.jsonl`). The four readers rate `imp2` with the human-reader judge prompt, 3 samples (`judge-human-imp2.jsonl`). GLM-5.3 checks, for each key term in the insight note, whether `imp2` (and, for comparison, the original) defines it in words (`DEFINED_CHECK`). Analysis as for the first hidden-gap follow-up: per-paper and pooled drop orig − imp2 with a bootstrap over questions within papers, the same paper gate, and the information check.

**Predictions:**
1. **Information check:** GPT-5.6 Sol keeps at least 0.9 of its original score on most gated papers.
2. **Definitions are really gone:** on average at least two thirds of the key terms that the original defines are not defined in `imp2`.
3. **Ratings:** GPT-5.6 Sol's and Gemma's human-prompt ratings fall by at least 1 point from the original.
4. **The legibility test:** for Gemma and Ministral, the pooled drop orig − imp2 minus GPT-5.6 Sol's drop is above 0 with a 95% interval excluding 0. Our expectation, stated now, is that any such effect is small (under 0.05), as with obfuscation; a null is reported as the result.

Budget: within the $839 stop line (usage $625.24 before this run); estimated $50 to $60.
