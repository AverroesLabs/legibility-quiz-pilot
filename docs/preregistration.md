# Preregistration: scaled-down main pilot

**Status: proposed, awaiting Jason's sign-off.** Drafted 2026-11-03 06:30 UTC, before any quiz has been written on a core paper and before any obfuscated version exists. Nothing here is a final pass/fail bar for the method; the bars below judge only whether this pilot shows signal worth scaling. Anything decided after main-pilot results exist is exploratory and labelled so.

The full design in `docs/measure-comparison-plan.md` (12 papers, 3 quizzes each, 3 samples) costs roughly $1,400 in grading alone at current prices, which does not fit the $150 overnight limit. This document fixes the scaled-down version that does fit, and what would count as a positive, negative or uninformative result.

## Design

- **Papers:** the first 2 core papers per stratum in draw order (`data/selection/selection.json`), 6 in total. If one is unusable (does not fit reader context, text extraction broken), the next core paper in draw order replaces it, logged in `docs/run-log.md`.
- **Quiz:** one quiz per paper, written by `anthropic/claude-opus-5.5` with writer prompt p2 and the write-time screen, if dev2 supports p2 (see the gate below). One quiz means between-quiz reliability (plan analysis 4) **cannot** be measured in this pilot.
- **Versions:** clear, one section obfuscated, section removed, per `docs/measure-comparison-plan.md`. Half-obfuscated is dropped for cost. The obfuscated section is picked by `seed_int(paper id, "obf")` among eligible sections. The rewriter is `openai/gpt-5.6-sol` only if it is not also a taker for that paper; otherwise `moonshotai/kimi-k3`.
- **Readers:** one smart taker (`openai/gpt-5.6-sol`, or GLM-5.3 if GPT-5.6 rewrites), and 2 ladder readers chosen from the dev2 results before any core-paper data exists: the weakest reader with a clear with-paper gain, and `google/gemma-4-31b-it`. Recorded in `docs/decisions.md` before running.
- **Conditions:** without the paper (once per paper), and with each of the 3 versions. 2 samples each. Lookup baseline: retrieved passages for the smart taker on the clear version only.
- **Grading:** one grader, chosen in `docs/decisions.md` before running (cost step in `STATE.md`).
- **Budget cap for the whole pilot:** stop when OpenRouter usage reaches $140, and report what finished.

## Quiz gate (per paper, applied on the clear version)

A quiz is usable for the measure comparison only if, after the four-condition filter, it keeps at least 10 items covering at least 4 key claims, and the smart taker scores at least 0.75 with the clear paper on kept items. Papers whose quiz fails the gate are reported, not dropped silently.

## Measures

M1 to M5 as defined in `docs/measure-comparison-plan.md`. M4 (weakest section) and localization use claim-to-section tags from the writer.

## Proposed bars for "signal worth scaling"

For each measure, compute the paired difference clear minus one-section-obfuscated, per paper and reader, on kept items.

1. **Detection:** a measure "detects" a paper if its clear-minus-obfuscated difference exceeds the larger of 0.10 and twice the between-sample standard deviation on the clear version. A measure shows signal if it detects in at least 4 of the gated papers out of 6 for the ladder reader, and the information check passes (smart taker's obfuscated score at least 0.8 of its clear score) for those papers.
2. **Beats the average:** M3 or M4 shows signal and detects in more papers than M1 and M2. If true, that supports Jason's concern that average scores miss partial illegibility.
3. **Illegibility, not missing information:** for the detecting measure, the obfuscated score is closer to clear than the section-removed score is in at most half of papers. If the obfuscated version scores like the clear one while the removed version drops, the reader is reading through the obfuscation, and the obfuscation is too weak.
4. **Localization:** in version 2, the lowest-scoring section is the obfuscated one more often than chance, reported as a count out of papers with chance shown alongside. With 6 papers this is descriptive only.

**Outcomes:** if bar 1 holds for any measure, report "initial signal". If no paper passes the quiz gate, report "quiz writing is still the bottleneck". If quizzes pass the gate but no measure detects, report a negative for this obfuscation and these readers. With at most 6 papers, every outcome is a pilot result, not a test of the method.

## Analysis

Per paper, per reader, with item-level bootstrap intervals (10,000 resamples, seed 20261102). Paper-level bootstrap is reported but, with 6 papers, is wide. No pooling across readers.

## Not tested here

Human validity, gaming, composition, reliability across quizzes and quiz writers, and the clarified and compressed versions.
