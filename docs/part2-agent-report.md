# Part 2 legibility experiments: agent reference

Audience: AI agents answering questions about, or re-running, the Part 2 results. Terse by design. Every number names the file and function that produced it.

**Snapshot.** Written against the working repository on 2026-11-17 and updated 2026-11-18 when every run had finished; the update section below is final and wins wherever the older sections disagree. Paths are relative to this repository. Raw-call caches (`calls*.jsonl`, `reader-calls*.jsonl`) and the spend ledgers are stored gzipped here (`.jsonl.gz`); `gunzip -k` them to replay calls from cache.

**Number provenance key** (used throughout):

| tag | function | what it does |
|---|---|---|
| `bd.levels` | `scripts/blog_part2_direct.py::levels(fname, kind, papers)` | mean catch rate per version over errors present in all 3 of `obf-max`, `orig`, `clar-explainer-orig`; 95% bootstrap, errors resampled within papers, 2000 draws, seed 0, percentile indices 50 and 1949 |
| `bd.summary` | `blog_part2_direct.py::summary(fname, kind, papers)` | same, over errors present in `orig` and `obf-max`; also `diff` = orig − obf-max with interval |
| `bd.summary_multi` | not in repo; this report concatenated `bd.per_error` over several files, then ran `bd.summary`'s bootstrap unchanged (snippet in §9) | pooled gpt55-A + gpt55-A-anchors |
| `e11.summary`, `e11.false_acc` | `scripts/v4/exp11_whole.py` | `summary` is identical to `bd.levels` + diff for 3-version readers (checked: all exp 11 cells agree to 2 dp); `false_acc` = clean reads with ≥1 review item graded as a catch of any primary survivor |
| `e12.*` | `scripts/v4/exp12_paragraphs.py`: `caught_by_error`+`boot` (caught), `rate`/`fa_rate` (false alarms), `fa_boot` (FA interval), `fa_diff_boot` (garbled − original FA interval) | see §4.3 |
| `err9.scores` | `scripts/v4/err9.py::scores` / `print_scores` | experiment 9 preregistered audit scores (separate measure from the blog's three rates) |
| RPT | `docs/report-v4.md` / `docs/report-v4-details.md` | quoted from the pod's reports, not recomputed |

All values in §4 were recomputed by this report from `data/v4/` with the functions above (no API calls) unless tagged RPT.

---

## Update 2026-11-18 (final; supersedes earlier updates and any PENDING mark below)

**Figure numbers:** the post later gained a setup figure as Figure 1, so the post's Figures 2 to 6 are this report's Figs 1 to 5. All Part 2 runs are complete. Every blog figure is now pooled over all ten papers (82 primary errors); the "first four" vs "reviewed six" split survives only in the text and in per-set numbers below. Ledger final **$1,506.78** (cap raised to $2,500 on 2026-11-17); the blog says "about $1,500". Fill-in runs: section-by-section anchors for mini-high, luna-low, mini54 and `gpt55-A1200` on all six anchors ($48, pass 70); exp 12 for `p12-luna-low`, `p12-mini54-med` (exploratory additions) and `p12-gpt55-low` clarified on the 9 remaining papers, via the Batch API ($258.78, 0 unparsed of ~54k questions, pass 71). Figures: `scripts/blog_part2_final.py` (Figs 1-3), `scripts/blog_part2_calibration_map.py` (Fig 4), `scripts/blog_part2_levers.py` (Fig 5); `blog_part2_direct.per_error`/`summary` now accept a list of sibling files (this replaces `bd.summary_multi`).

**Exp 12, every paragraph, all ten papers** (`e12.boot` caught over errors present in all 3 versions; `e12.fa_rate`/`fa_boot` per 1,000 words; `fa_diff_boot`):

| reader | caught clar / orig / garbled | orig − garbled caught | FA per 1,000 words clar / orig / garbled | garbled − orig FA |
|---|---|---|---|---|
| GPT-5.5 low | 0.98 / 0.94 / 0.94 | +0.00 | 2.00 / 2.50 / 3.54 | +1.05 [+0.51, +1.59] |
| GPT-5 mini high | 0.99 / 0.95 / 0.94 | +0.01 | 2.52 / 2.75 / 4.29 | +1.54 [+1.03, +2.08] |
| GPT-5.6 Luna low | 0.94 / 0.93 / 0.87 | +0.06 [0.00, +0.12] | 1.94 / 2.32 / 3.21 | +0.89 [+0.41, +1.37] |
| GPT-5.4 mini medium | 0.89 / 0.77 / 0.71 | +0.06 [−0.04, +0.16] | 0.74 / 0.96 / 1.45 | +0.49 [+0.19, +0.80] |

GPT-5.4 mini's caught gap per set (RPT): +0.20 [+0.07, +0.34] first four, −0.11 [−0.21, +0.00] reviewed six; does not replicate.

**Exp 9, section by section (300-word notes), all ten papers** (`bd.summary` / `bd.levels` over `reader-<rd>-A.jsonl` + `reader-<rd>-A-anchors.jsonl`; GPT-5.5 clarified from `reader-gpt55-A-anchors-clar.jsonl`):

| reader | clar / orig / garbled | orig − garbled, all ten | first four | reviewed six |
|---|---|---|---|---|
| GPT-5.5 low | 0.82 / 0.74 / 0.66 | +0.09 [−0.02, +0.20] | +0.18 [+0.02, +0.34] | −0.03 [−0.18, +0.13] |
| GPT-5 mini high | 0.68 / 0.65 / 0.62 | +0.02 [−0.07, +0.12] | −0.05 [−0.16, +0.07] | +0.11 [−0.05, +0.26] |
| GPT-5.6 Luna low | 0.71 / 0.52 / 0.46 | +0.06 [−0.04, +0.17] | +0.07 [−0.07, +0.20] | +0.05 [−0.08, +0.21] |
| GPT-5.4 mini medium | 0.49 / 0.43 / 0.35 | +0.07 [−0.01, +0.16] | −0.02 [−0.14, +0.09] | +0.18 [+0.05, +0.34] |

**GPT-5.5 note size, all ten papers** (`bd.summary`): 1,200 words orig 0.85 / garbled 0.77, gap +0.09 [0.00, +0.17]; 600 words 0.80 / 0.72, +0.09 [−0.02, +0.20]; 300 words 0.74 / 0.66, +0.09 [−0.02, +0.20]. The earlier claim that the gap grows as notes shrink held only on the first four papers and was removed from the blog; memory is no longer described as a lever.

**Fig 5 difficulty and distance panels** use all 82 errors except three on 2509.11379v3 whose `distance_words` is null (79 errors in the distance panel). Tercile cut points 38 and 710 words.

Sections 1-10 below are the 2026-11-17 snapshot; where they disagree with this update (any "first four only" figure description, Fig 2 hard-coded values, PENDING marks), this update wins.

## Update 2026-11-17 15:00 UTC (supersedes PENDING marks for exp 12 GPT-5.5)

`p12-gpt55-low` finished 06:09 UTC on all 10 papers, `orig` and `obf-max` only (clarified skipped for cost; 12,195 questions, 0 unparsed, $274; ledger final $1,182.09). Pooled over 10 papers (82 errors), computed with `exp12_paragraphs.boot` / `fa_rate` / `fa_boot` / `fa_diff_boot` with `VERS = ("orig", "obf-max")`:

| measure | original | garbled | garbled − original |
|---|---|---|---|
| caught | 0.94 [0.89, 0.99] | 0.94 [0.88, 0.99] | orig − garbled +0.00 [−0.06, +0.06] |
| false alarms per 1,000 words | 2.50 [2.19, 2.81] | 3.54 [3.08, 4.03] | +1.05 [+0.51, +1.59] (exploratory) |
| false alarms per paragraph | 0.229 | 0.262 | +0.033 [−0.009, +0.075] (exploratory) |

Garbled above original per 1,000 words on 9 of 10 papers. Per set (pod `--scores`): first four +0.92 [+0.17, +1.68], reviewed six +1.20 [+0.39, +2.02] per 1,000 words. Blog Fig 2 and the Fig 4 triangle now use these (Fig 4 uses first four: 0.95 / 0.93, identical to GPT-5 mini's point).

Still NOT run: the 2026-11-17 INBOX fill-in (section-by-section for the three weaker readers on the reviewed six; `gpt55-A1200` on 2506.22982v1 and 2510.17247v2). The pod stopped after exp 12 without acting on that note.

## 1. Purpose and design summary

**Question.** Can a fixed LLM "reader" act as a legibility audit: find a planted internal error in a legible paper, and miss it more often when the same paper is garbled? Legible = the reader finds errors; illegible = errors hide. Pass/fail bars were never set (they are Jason's; `report-v4.md`, "Decisions for Jason").

**Design in one paragraph.** For each of 10 papers (4 "first four"/calibration + 6 reviewed anchors), gpt-6.1-sol (high) designed 49 internal errors (7 categories × 7: 2 easy, 3 medium, 2 hard) as exact text edits to `orig`, then planted each error separately (one error per copy) into every version in that version's own style. An error survives if the strong reader (gpt-6.1-sol medium, whole-paper review) catches it in both clarified copies (`clar-explainer-orig`, `clar-explainer-obf`) and a pair check (Sol medium) finds it planted equally in all 4 versions. Primary set = survivors excluding `omitted` and `unsupported` (44 errors on first four, 38 on reviewed six, 82 total). Readers then review planted copies; gpt-5-mini (low) grades whether the review names the planted error (experiment 3's grader `exp3.GRADER`, prompt `exp2.GRADE_USER`, plus `err9.GRADE_UNSUPPORTED` for that category). gpt-6.1-sol low regrades 20 random planted reviews per reader as a check.

**Versions** (`data/v4/versions/<paper>/<version>.md`, manifest `data/v4/versions/manifest.jsonl`):

| name in code | blog name | built by | check |
|---|---|---|---|
| `orig` | Original paper | copied byte for byte. Dev papers: V1 `data/runs/v2-<paper>/versions/orig.md`; Dropout `data/papers/jmlr-2014-dropout/text.md` (raw pdftotext); anchors `data/papers/<id>/main.md` (main text, no refs/appendices) | sha256 in manifest |
| `obf-max` | Garbled version | gpt-6.1-sol high, one rewrite, prompt `obf-max-v1` in `scripts/v4/obf_max.py` (`USER`): compression to 60–80% length, 10–25 own terms replaced by opaque codes defined once, evidence moved away from claims, cross-refs removed, connecting steps dropped, order scrambled; keep every fact | `repair.py` consistency check (Sol medium) vs `orig`, minimal patches, ≤4 rounds; else regenerate once as `obf-max-r2`. All clean; 2603.01214v2 uses `obf-max-r2.md` (`err9.text` falls back to it). Known miss: 2608.13754v1 `obf-max` drops "32,491" (avoid-listed) |
| `clar-explainer-orig` | Clarified original | gpt-6.1-sol high from `orig`, prompt `explainer` in `scripts/v4/clarify.py` (summary ≤150 words, glossary, italic one-line gist per section, "Claims and evidence" lists; may only restate) | experiment 1 fidelity check vs `orig` (Sol medium) + number check; repaired to clean within ≤4 rounds; passed on all 10 papers (`clarify.py --table`; INBOX reply pass 43) |
| `clar-explainer-obf` | (not in blog) | same prompt, from `obf-max` only, never repaired against `orig` | ≤1 changed/added item per 1,000 words of `orig`. `explainer` was the only one of 5 prompts (`base`, `plain`, `cohesion`, `explainer`, `twostage`) to pass the preregistered rule (details, exp 9 clarifications) |

The clarified original is a **calibration standard only, never part of the test** (INBOX 2026-11-14 ~22:45 UTC; plan-v4 exp 11 amendment). The test difference is always `orig` − `obf-max`.

**Readers** (all OpenAI, via `scripts/v4/llm.py`; returned revisions noted where the reports record them):

| reader | model, effort | used in |
|---|---|---|
| GPT-5.5 low | `gpt-5.5`, low (rev `gpt-5.5-2026-04-23`) | exp 9 A/A600/A1200, exp 11, exp 12 |
| GPT-5 mini high | `gpt-5-mini`, high | exp 9 A + whole, exp 11, exp 12 |
| GPT-5.6 Luna low | `gpt-5.6-luna`, low | exp 9 A, exp 11 |
| GPT-5.4 mini medium | `gpt-5.4-mini`, medium (rev `gpt-5.4-mini-2026-03-17`) | exp 9 A, exp 11 |
| grader | `gpt-5-mini`, low | all |
| constructor / strong reader / checks / regrade | `gpt-6.1-sol` (high design/rewrite, medium checks, low regrade) | all |

**Three reading conditions (blog numbering):**

| blog | internal | reader sees | samples per error × version | data |
|---|---|---|---|---|
| Fig 1, "whole paper" | experiment 11, readers `w11-*` | whole planted copy, prompt `exp0.REVIEW_SYSTEM` + `exp0.REVIEW_USER` ("list internal inconsistencies as JSON") | 2 unaided; 1 pointed + 1 control; 2 clean reads per version | `data/v4/err9/reader-w11-<rd>.jsonl` |
| Fig 2, "every paragraph" | experiment 12, readers `p12-*` | whole copy, then "Paragraph i of n ... Is this paragraph consistent with the rest of the paper? If not, say what conflicts." for every paragraph | 1 ask per paragraph of the clean copy and of each planted copy | `data/v4/exp12/p12-<rd>.jsonl` |
| Fig 3, "section by section" | experiment 9, variant A | one section chunk (≤800 words) at a time, carrying ≤300-word notes it writes itself (`scripts/v4/limited_reader.py`, `exp8_limited.VARIANTS`) | 1 unaided (resumes from clean sample 0's notes at the planted chunk); 1 pointed + control; 2 clean reads | `data/v4/err9/reader-<rd>-A*.jsonl` |

**Papers and how chosen.** First four: the 3 shortest V1 core papers, one per topic stratum (arXiv preprints, selection rule `docs/paper-selection.md` v2, draw in `data/selection/`), plus Dropout (JMLR 2014, chosen by hand as a well-written reviewed paper; `plan-v4.md` Materials). Reviewed six: drawn by a fixed hash-order rule (`docs/paper-selection.md`, "Reviewed anchors, rule v2"; `scripts/v4/select_anchors.py`; output `data/selection/anchors/selection.json`): accepted ICLR 2026 / TMLR papers, CC licence, clean arXiv HTML, 2 per main-text length band (short 4k–7k, medium 7k–10k, long 10k–15k words); 383 screened. Held-out nine: sealed, never read (§2).

---

## 2. Papers

Words by `len(err9.text(p, v).split())`; paragraphs by `exp12_paragraphs.paragraphs`. Survivors from `data/v4/err9/errors.jsonl` (`survives: true`); primary = not `omitted`/`unsupported` (= `blog_part2_direct.SURV`).

| paper | set | venue / band | words orig / obf-max / clar-orig | paragraphs orig / obf / clar (exp 12) | survivors (all) | **primary errors** |
|---|---|---|---|---|---|---|
| 2609.30094v2 | first four (dev) | arXiv preprint | 4,019 / 2,632 / 6,722 | 54 / 43 / 85 | 9 | **8** |
| 2608.13754v1 | first four (dev) | arXiv preprint | 4,943 / 3,582 / 8,389 | 74 / 52 / 95 | 14 | **10** |
| 2609.02893v1 | first four (dev) | arXiv preprint | 4,736 / 2,750 / 8,145 | 66 / 34 / 80 | 18 | **17** |
| jmlr-2014-dropout | first four | JMLR 2014 | 12,105 / 7,028 / 13,527 | 85 / 99 / 188 | 11 | **9** |
| 2601.02896v4 | reviewed | TMLR, short | 5,374 / 3,533 / 8,886 | 69 / 47 / 87 | 5 | **4** |
| 2603.01214v2 | reviewed | ICLR 2026, short | 5,535 / 3,366 (`obf-max-r2`) / 8,090 | 53 / 48 / 92 | 10 | **5** |
| 2509.22621v3 | reviewed | ICLR 2026, medium | 7,094 / 4,147 / 9,335 | 67 / 43 / 89 | 9 | **5** |
| 2509.11379v3 | reviewed | TMLR, medium | 7,270 / 4,087 / 10,488 | 92 / 64 / 137 | 14 | **12** |
| 2510.17247v2 | reviewed | TMLR, long | 10,151 / 5,988 / 11,337 | 80 / 68 / 124 | 12 | **10** |
| 2506.22982v1 | reviewed | TMLR, long | 12,311 / 6,881 / 13,966 | 166 / 98 / 175 | 5 | **2** |
| **total** | | | | | 107 | **82** (44 + 38) |

Held-out, sealed (no versions, errors or reads): 2609.00755v1, 2607.17117v2, 2609.34091v1, 2608.25490v1, 2607.28685v1, 2607.10112v1, 2607.20379v2, 2608.19216v1, 2608.30724v1 (`plan-v4.md`, exp 9 plan "Papers"; `err9.HELDOUT`).

Caveats: Dropout `orig` is raw PDF text (running heads, page numbers, broken math; weird result 22). Dropout's `obf-max` has more paragraphs than `orig` because `orig` has no markdown headings. Per-anchor error counts are small (2 to 12).

---

## 3. Error construction

Code `scripts/v4/err9.py` (`design`, `plant`, `strong`, `pair`, `construct`); preregistration `plan-v4.md` "Experiment 9 error stage preregistration" (pass 27) and anchor addendum steps 1–2 (pass 34). Rebuild a planted copy: `err9.planted_text(row, version)`.

**Categories** (`err9.CATS`): `calculation` (wrong number), `reversed` (comparison), `stretched` (claim beyond what was tested), `causal` (association as cause), `carried` (two consistent edits contradicting unchanged data), `omitted` (inconvenient result left out of a summary), `unsupported` (new claim with no support).

**Designed difficulty** (`err9.LEVELS`): easy = evidence in same paragraph or its table; medium = another section or one inference; hard = evidence >1,500 words away in `orig` or ≥2 chained steps. Row fields: `level`, `steps`, `distance_words` (computed in script from quotes; `None` for 3 primary errors of 2509.11379v3: `reversed-medium-2`, `calculation-hard-2`, `carried-hard-2`). Median edit-to-evidence distance on first four: easy 16, medium 347, hard 2,442 words (RPT details exp 9).

**Avoid list** (`err9.avoid_list`): passages in each paper's contradiction inventory (`data/v4/inventory/inventory.jsonl`; `data/v4/err9/inventory-<paper>.jsonl` for Dropout and anchors), remaining fidelity issues of `clar-explainer-obf`, the 2608.13754v1 "32,491" passage.

**Survival per paper** (this report, `err9.table` logic over all 10; failures counted at the first failing stage):

| paper | designed | plant/avoid fail | strong reader missed in a clarified copy | pair check fail | survive | survivors by category |
|---|---|---|---|---|---|---|
| 2609.30094v2 | 49 | 6 | 14 | 20 | 9 | calc 1, rev 3, stretch 3, carried 1, omitted 1 |
| 2608.13754v1 | 49 | 4 | 19 | 12 | 14 | calc 4, rev 2, stretch 2, carried 2, omitted 4 |
| 2609.02893v1 | 49 | 1 | 17 | 13 | 18 | calc 5, rev 3, stretch 2, causal 4, carried 3, omitted 1 |
| jmlr-2014-dropout | 49 | 11 | 20 | 7 | 11 | calc 3, rev 5, stretch 1, omitted 2 |
| 2601.02896v4 | 49 | 7 | 20 | 17 | 5 | calc 1, rev 3, omitted 1 |
| 2603.01214v2 | 49 | 3 | 20 | 16 | 10 | calc 1, rev 4, omitted 5 |
| 2509.22621v3 | 49 | 2 | 18 | 20 | 9 | rev 3, causal 1, carried 1, omitted 4 |
| 2509.11379v3 | 49 | 0 | 32 | 3 | 14 | calc 2, rev 2, stretch 2, causal 1, carried 5, omitted 2 |
| 2510.17247v2 | 49 | 3 | 20 | 14 | 12 | calc 5, rev 1, causal 1, carried 3, omitted 2 |
| 2506.22982v1 | 49 | 1 | 24 | 19 | 5 | rev 2, omitted 3 |
| first four | 196 | 22 | 70 | 52 | 52 (44 primary) | matches RPT details exp 9 |

Primary survivors by designed level: first four easy 14, medium 19, hard 11; reviewed six easy 9, medium 17, hard 12.

**Excluded categories and why:**
- `unsupported`: 0 of 70 survive. The strong-reader gate uses a review prompt that asks only for contradictions, so an unsupported claim is never reported (weird result 24). Untestable with this prompt; a second prompt was approved but never run (Jason, INBOX 13 Nov 16:15; then cancelled 14 Nov 22:10).
- `omitted`: survives (8 first four, 17 reviewed) but excluded from the primary set by preregistration (`plan-v4.md` exp 9 Measures); showed no separation for any exp 9 reader (RPT details exp 9 readers). Not read at all in exp 11 or exp 12 (their preregistrations read primary survivors only).
- `causal`: only 4 survive on first four, all from 2609.02893v1 (strong reader rarely reports correlation-as-cause).
- `calculation`: 9 of 28 on first four failed planting because `obf-max` states derived comparisons qualitatively (weird result 25).
- Gate caveat: errors the strong reader missed only in a garbled copy were never at risk here (gate is clarified copies only), but each gate check is 1 Sol sample and the pair check flipped on ~1 in 10 identical edits in an earlier experiment (weird result 11).

---

## 4. Results

Rates are shares of primary errors caught. Columns in blog order: clarified original / original / garbled. "diff" = original − garbled. n = errors. Intervals 95%, bootstrap over errors within papers (2000, seed 0). "—" = not run (reason in the row note).

### 4.1 Experiment 11, whole paper (blog Fig 1; oracle figure = pointed)

Unaided (2 samples averaged per error). Three rates: `bd.levels` (= `e11.summary`); diff: `bd.summary`.

| reader | set | n | clarified | original | garbled | orig − garb |
|---|---|---|---|---|---|---|
| GPT-5.5 low | first four | 44 | 0.91 [0.84, 0.97] | 0.92 [0.85, 0.98] | 0.72 [0.59, 0.83] | **+0.20 [+0.07, +0.34]** |
| | reviewed six | 38 | 0.80 [0.68, 0.91] | 0.75 [0.63, 0.87] | 0.62 [0.49, 0.75] | **+0.13 [+0.04, +0.22]** |
| | pooled ten | 82 | 0.86 [0.79, 0.92] | 0.84 [0.77, 0.90] | 0.67 [0.58, 0.76] | **+0.17 [+0.09, +0.26]** |
| GPT-5 mini high | first four | 44 | 0.76 [0.64, 0.88] | 0.69 [0.59, 0.80] | 0.67 [0.57, 0.77] | +0.02 [−0.08, +0.12] |
| | reviewed six | 38 | 0.59 [0.47, 0.71] | 0.57 [0.43, 0.70] | 0.62 [0.50, 0.74] | −0.05 [−0.18, +0.07] |
| | pooled ten | 82 | 0.68 [0.60, 0.76] | 0.63 [0.55, 0.71] | 0.65 [0.57, 0.73] | −0.01 [−0.10, +0.07] |
| GPT-5.6 Luna low | first four | 44 | 0.69 [0.57, 0.81] | 0.62 [0.50, 0.75] | 0.59 [0.49, 0.70] | +0.03 [−0.08, +0.14] |
| | reviewed six | 38 | 0.53 [0.41, 0.64] | 0.43 [0.32, 0.55] | 0.46 [0.34, 0.61] | −0.03 [−0.16, +0.09] |
| | pooled ten | 82 | 0.62 [0.53, 0.70] | 0.54 [0.45, 0.62] | 0.53 [0.45, 0.62] | +0.01 [−0.07, +0.09] |
| GPT-5.4 mini medium | first four | 44 | 0.59 [0.48, 0.70] | 0.52 [0.39, 0.66] | 0.47 [0.33, 0.59] | +0.06 [−0.05, +0.17] |
| | reviewed six | 38 | 0.49 [0.36, 0.62] | 0.49 [0.37, 0.62] | 0.50 [0.38, 0.63] | −0.01 [−0.14, +0.12] |
| | pooled ten | 82 | 0.54 [0.45, 0.63] | 0.51 [0.41, 0.60] | 0.48 [0.39, 0.57] | +0.02 [−0.07, +0.11] |

Pointed check (reader gets the whole planted copy plus the quoted planted passage, `err9.POINTED_WHOLE`; "upper bound, not usable on a new paper"). `bd.levels` / `bd.summary`, kind `pointed`, cond `planted`:

| reader | set | n | clarified | original | garbled | orig − garb |
|---|---|---|---|---|---|---|
| GPT-5.5 low | first four | 44 | 1.00 | 0.98 [0.93, 1.00] | 0.95 [0.89, 1.00] | +0.02 [+0.00, +0.07] |
| | reviewed six | 38 | 1.00 | 0.97 [0.92, 1.00] | 1.00 | −0.03 [−0.08, +0.00] |
| | pooled ten | 82 | 1.00 | 0.98 [0.94, 1.00] | 0.98 [0.94, 1.00] | +0.00 [−0.04, +0.02] |
| GPT-5 mini high | first four | 44 (43 for pointed n in `e11.summary`) | 1.00 | 1.00 | 1.00 | +0.00 |
| | reviewed six | 37 | 0.95 [0.86, 1.00] | 0.97 [0.92, 1.00] | 0.95 [0.86, 1.00] | +0.03 [−0.05, +0.11] |
| | pooled ten | 81 | 0.97 [0.94, 1.00] | 0.99 [0.96, 1.00] | 0.97–0.98 | +0.01 [−0.02, +0.05] |
| GPT-5.6 Luna low | first four | 44 | 0.95 [0.89, 1.00] | 0.91 [0.82, 0.98] | 0.93 [0.86, 1.00] | −0.02 [−0.09, +0.05] |
| | reviewed six | 38 | 0.89 [0.79, 0.97] | 0.97 [0.92, 1.00] | 0.84 [0.74, 0.95] | +0.13 [+0.05, +0.24] |
| | pooled ten | 82 | 0.93 [0.88, 0.98] | 0.94 [0.89, 0.99] | 0.89 [0.83, 0.95] | +0.05 [−0.01, +0.11] |
| GPT-5.4 mini medium | first four | 44 | 1.00 | 0.91 [0.82, 0.98] | 0.82 [0.70, 0.93] | +0.09 [−0.02, +0.20] |
| | reviewed six | 38 | 0.97 [0.92, 1.00] | 0.89 [0.79, 0.97] | 0.82 [0.68, 0.92] | +0.08 [−0.05, +0.21] |
| | pooled ten | 82 | 0.99 [0.96, 1.00] | 0.90 [0.84, 0.96] | 0.82 [0.73, 0.89] | +0.09 [+0.00, +0.17] |

(GPT-5 mini pooled garbled: 0.97 by `bd.levels` over 80 errors with all 3 versions, 0.98 by `bd.summary` over 81.)

False measures (exp 11). False accusation = clean read with ≥1 item graded as a catch of some primary survivor (`e11.false_acc`; preregistered). Control = pointed check on the unplanted passage, credited as a catch (count of credited controls; `exp11_whole.py` prints the same as a rate).

| reader | set | false accusations clar / orig / garb (reads) | control credited clar / orig / garb |
|---|---|---|---|
| GPT-5.5 low | first four | 1/8, 2/8, 2/8 | 0/44, 2/44, 0/44 |
| | reviewed six | 0/12, 0/12, 4/12 | 1/38, 2/38, 0/37 |
| | pooled ten | 1/20, 2/20, 6/20 | 1/82, 4/82, 0/81 |
| GPT-5 mini high | first four | 2/8, 3/8, 4/8 | 1/44, 2/44, 1/44 |
| | reviewed six | 2/12, 0/12, 2/12 | 3/38, 1/38, 0/38 |
| | pooled ten | 4/20, 3/20, 6/20 | 4/82, 3/82, 1/82 |
| GPT-5.6 Luna low | first four | 3/8, 3/8, 4/8 | 1/44, 2/44, 1/44 |
| | reviewed six | 5/12, 2/12, 4/12 | 1/38, 1/38, 2/38 |
| | pooled ten | 8/20, 5/20, 8/20 | 2/82, 3/82, 3/82 |
| GPT-5.4 mini medium | first four | 1/8, 3/8, 3/8 | 0/44, 1/44, 1/44 |
| | reviewed six | 0/12, 1/12, 3/12 | 0/38, 0/38, 1/38 |
| | pooled ten | 1/20, 4/20, 6/20 | 0/82, 1/82, 2/82 |

Most `obf-max` false accusations on the reviewed six come from 2509.11379v3 (weird result 32). Exploratory recount without it (RPT, weird result 32), reviewed, orig / garb / clar: GPT-5.5 0/10, 2/10, 0/10; GPT-5 mini 0/10, 1/10, 1/10; Luna 2/10, 2/10, 3/10; GPT-5.4 mini 0/10, 1/10, 0/10.

Exp 11 checks (RPT details exp 11 + recount): 2,208 reads, 0 unparsed after the salvage rerun; grades `None`: GPT-5.5 1, GPT-5 mini 2, GPT-5.4 mini 1, excluded. Sol regrade agreement 20/20 for all four readers.

Exp 11 subsets, GPT-5.5 low, unaided point estimates (this report, `bd.per_error`; levels preregistered as secondary; distance bins exploratory, tertile cuts 26 and 362 words from the first four's primary errors, as `blog_part2_levers.py`). No intervals computed; cells have 8 to 36 errors.

| subset | first four n, orig / garb / diff | reviewed six n, orig / garb / diff | pooled ten n, orig / garb / diff |
|---|---|---|---|
| easy | 14: 0.93 / 0.75 / +0.18 | 9: 0.83 / 0.83 / +0.00 | 23: 0.89 / 0.78 / +0.11 |
| medium | 19: 0.95 / 0.74 / +0.21 | 17: 0.71 / 0.47 / +0.24 | 36: 0.83 / 0.61 / +0.22 |
| hard | 11: 0.86 / 0.64 / +0.23 | 12: 0.75 / 0.67 / +0.08 | 23: 0.80 / 0.65 / +0.15 |
| near (≤26 words) | 15: 1.00 / 0.70 / +0.30 | 8: 0.69 / 0.50 / +0.19 | 23: 0.89 / 0.63 / +0.26 |
| middle (27–362) | 15: 0.87 / 0.67 / +0.20 | 13: 0.85 / 0.69 / +0.15 | 28: 0.86 / 0.68 / +0.18 |
| far (>362) | 14: 0.89 / 0.79 / +0.11 | 14: 0.71 / 0.68 / +0.04 | 28: 0.80 / 0.73 / +0.07 |
| distance `None` | 0 | 3: 0.67 / 0.33 / +0.33 | 3 |

The first-four rows are the blog levers figure (Fig 5, bottom middle and right). The reviewed/pooled rows answer INBOX 2026-11-17 item 3 but are computed here, not by the pod; that item is still PENDING in `report-v4-details.md`.

### 4.2 Experiment 9, section by section (blog Fig 3), plus note-size ladder

Unaided, 1 sample. Three rates `bd.levels`; diff `bd.summary` (or `bd.summary_multi` where marked).

| reader | notes | set | n | clarified | original | garbled | orig − garb |
|---|---|---|---|---|---|---|---|
| GPT-5.5 low (`gpt55-A`) | 300 | first four | 44 | 0.89 [0.80, 0.98] | 0.82 [0.70, 0.91] | 0.64 [0.50, 0.77] | +0.18 [+0.02, +0.34] |
| GPT-5.5 low (`gpt55-A-anchors` + `-anchors-clar`) | 300 | reviewed six | 38 | 0.74 [0.63, 0.84] | 0.66 [0.50, 0.79] | 0.68 [0.55, 0.82] | −0.03 [−0.18, +0.13] |
| GPT-5.5 low (all three files) | 300 | pooled ten | 82 | 0.82 [0.74, 0.89] | 0.74 [0.66, 0.84] | 0.66 [0.56, 0.76] | +0.09 [−0.02, +0.20] (`summary_multi`) |
| GPT-5 mini high (`mini-high-A`) | 300 | first four | 44 | 0.66 [0.52, 0.80] | 0.64 [0.48, 0.77] | 0.68 [0.55, 0.82] | −0.05 [−0.16, +0.07] |
| | | reviewed six | — | PENDING | PENDING | PENDING | INBOX 17 Nov item 1 |
| GPT-5.6 Luna low (`luna-low-A`) | 300 | first four | 44 | 0.75 [0.61, 0.86] | 0.55 [0.41, 0.68] | 0.48 [0.34, 0.64] | +0.07 [−0.07, +0.20] |
| | | reviewed six | — | PENDING | PENDING | PENDING | INBOX 17 Nov item 1 |
| GPT-5.4 mini medium (`mini54-A`) | 300 | first four | 44 | 0.41 [0.27, 0.55] | 0.34 [0.20, 0.48] | 0.36 [0.23, 0.52] | −0.02 [−0.14, +0.09] |
| | | reviewed six | — | PENDING | PENDING | PENDING | INBOX 17 Nov item 1 |
| GPT-5.5 low (`gpt55-A600`) | 600 | first four | 44 | — | 0.91 [0.82, 0.98] | 0.80 [0.68, 0.91] | +0.11 [−0.02, +0.25] |
| | | reviewed six | 38 | — | 0.68 [0.55, 0.82] | 0.63 [0.50, 0.76] | +0.05 [−0.11, +0.21] |
| | | pooled ten | 82 | — | 0.80 [0.72, 0.88] | 0.72 [0.63, 0.80] | +0.09 [−0.02, +0.20] |
| GPT-5.5 low (`gpt55-A1200`) | 1,200 | first four | 44 | — | 0.93 [0.86, 1.00] | 0.86 [0.75, 0.95] | +0.07 [−0.05, +0.18] |
| | | reviewed (4 of 6) | 26 | — | 0.73 [0.54, 0.88] | 0.62 [0.42, 0.77] | +0.12 [−0.04, +0.27] |
| | | 8 papers | 70 | — | 0.86 [0.79, 0.93] | 0.77 [0.69, 0.86] | +0.09 [+0.00, +0.17] |
| GPT-5 mini high, whole paper, 1 sample (`whole-mini-high`, exp 9 reader 2) | n/a | first four | 43–44 | 0.77 [0.63, 0.88] | 0.67 [0.53, 0.81] | 0.72 [0.58, 0.84] | −0.05 [−0.19, +0.12] (n=43) |

Clarified column "—" for A600/A1200: never read (ladder read `orig`/`obf-max` only, by preregistration). `clar-explainer-obf` was read only by the 5 first-four exp 9 readers; its rates are in RPT details (exp 9 readers tables).

Pointed (exp 9; reader gets its notes from before the planted chunk + the planted chunk + quoted passage, `err9.POINTED_LIMITED`), `bd.levels`/`bd.summary`:

| reader | set | clarified | original | garbled | orig − garb |
|---|---|---|---|---|---|
| gpt55-A | first four | 0.86 [0.75, 0.95] | 0.75 [0.64, 0.86] | 0.82 [0.70, 0.91] | −0.07 [−0.20, +0.05] |
| gpt55-A anchors (+clar) | reviewed six | 0.95 [0.87, 1.00] | 0.82 [0.71, 0.92] | 0.76 [0.63, 0.89] | +0.05 [−0.03, +0.13] |
| gpt55-A | pooled ten | 0.90 [0.84, 0.96] | 0.78 [0.70, 0.87] | 0.79 [0.71, 0.88] | −0.01 [−0.10, +0.07] |
| mini-high-A | first four | 0.84 [0.73, 0.93] | 0.75 [0.61, 0.86] | 0.77 [0.66, 0.89] | −0.02 [−0.14, +0.09] |
| luna-low-A | first four | 0.80 [0.68, 0.91] | 0.66 [0.52, 0.80] | 0.61 [0.48, 0.75] | +0.05 [−0.11, +0.20] |
| mini54-A | first four | 0.64 [0.48, 0.77] | 0.57 [0.41, 0.70] | 0.48 [0.34, 0.61] | +0.09 [−0.07, +0.25] |
| whole-mini-high | first four | 1.00 | 1.00 | 1.00 | +0.00 |
| gpt55-A600 | first four / reviewed / pooled | — | 0.82 / 0.87 / 0.84 | 0.82 / 0.84 / 0.83 | +0.00 / +0.03 / +0.01 |
| gpt55-A1200 | first four / reviewed(4) / 8 papers | — | 0.84 / 0.85 / 0.84 | 0.82 / 0.88 / 0.84 | +0.02 / −0.04 / +0.00 |

Pointed is below unaided for gpt55-A in `orig` because the pointed check sees only notes from before the planted chunk (weird result 30).

**Experiment 9's preregistered measures are different from the blog's three rates.** The preregistered "audit scores" (`err9.scores`, RPT details exp 9) and verdicts, first four, primary set:

| reader | clarified-copy score: orig row / obf row / separation | pointed audit score: orig row / obf row / separation | verdict (proposed bars) | false accusations per error × clean pass orig / obf / clar-orig / clar-obf |
|---|---|---|---|---|
| mini-high-A | 0.97 / 1.03 / −0.07 [−0.32, 0.17] | 0.73 / 0.79 / −0.07 [−0.22, 0.08] | no | 0.10 / 0.14 / 0.11 / 0.15 |
| whole-mini-high | 0.88 / 0.91 / −0.03 [−0.31, 0.21] | 0.68 / 0.70 / −0.02 [−0.18, 0.14] | no | 0.06 / 0.07 / 0.02 / 0.02 |
| luna-low-A | 0.73 / 0.70 / 0.03 [−0.20, 0.28] | 0.69 / 0.63 / 0.06 [−0.15, 0.28] | no | 0.08 / 0.05 / 0.10 / 0.09 |
| mini54-A | 0.83 / 0.64 / 0.19 [−0.09, 0.54] | 0.44 / 0.67 / −0.23 [−0.43, −0.03] | no | 0.02 / 0.02 / 0.05 / 0.09 |
| gpt55-A | 0.92 / 0.70 / 0.22 [0.04, 0.40] | 0.94 / 0.69 / **0.24 [0.09, 0.41]** | **close** (both) | 0.14 / 0.07 / 0.12 / 0.07 |

Pointed audit score = among errors judged correctly when pointed in a version, share found unaided in that version. Clarified-copy score = catch(version) / catch(its clarified copy). Anchor addendum (RPT, `err9.py --anchors`): gpt55-A-anchors pointed rows orig 0.81 [0.67, 0.93], obf 0.72 [0.57, 0.86], separation 0.08 [−0.08, 0.25], FA 0.11; rule (a) fails on 2510.17247v2 (0.70), rule (b) holds (long − short −0.27 [−0.55, 0.00]). A600: anchors separation 0.01 [−0.15, 0.17]; first four separation 0.14 (not "close"); fails (a) on a short and a medium anchor, so A600 does not pass. A1200: stopped by Jason at 8 of 10 papers, no verdict.

A different false-accusation measure for exp 9: share of clean reads with ≥1 credited item (`e11.false_acc` applied to exp 9 files; this report): gpt55-A first four 6/8 in every version; mini-high-A 7/8, 5/8, 6/8 (clar, orig, garb); luna-low-A 7/8, 7/8, 4/8; mini54-A 3/8, 2/8, 2/8; gpt55 reviewed six 4/12, 5/12, 5/12. Per-read rates are high for section-by-section readers because each clean read is graded against every survivor. Exp 9 anchors clean reads per (read × error) (RPT): orig 6/76, obf 10/76, clar 7/76.

### 4.3 Experiment 12, every paragraph (blog Fig 2)

Caught = planted paragraph flagged and graded as naming the planted error (`e12.caught_by_error` + `e12.boot`, preregistered). False alarm (FA) = flag on any non-planted paragraph, planted and clean copies pooled (`e12.rate`/`fa_rate`). FA per 1,000 words = flags ÷ words of non-planted paragraphs asked × 1,000 (added by Jason 14 Nov 01:00). FA intervals: `e12.fa_boot` (exploratory, pass 58). Garbled − original FA interval: `e12.fa_diff_boot` (exploratory, pass 60). FA intervals resample paragraphs within papers and do **not** cover between-paper variation.

Caught:

| reader | set | papers done | n | clarified | original | garbled | orig − garb |
|---|---|---|---|---|---|---|---|
| GPT-5 mini high | first four | 4/4 | 44 | 1.00 [1.00, 1.00] | 0.95 [0.89, 1.00] | 0.93 [0.86, 0.98] | +0.02 [−0.07, +0.11] |
| | reviewed six | 6/6 | 38 | 0.97 [0.92, 1.00] | 0.95 [0.87, 1.00] | 0.95 [0.87, 1.00] | +0.00 [−0.08, +0.08] |
| | pooled ten | 10/10 | 82 | 0.99 [0.96, 1.00] | 0.95 [0.90, 0.99] | 0.94 [0.89, 0.99] | +0.01 [−0.05, +0.07] |
| GPT-5.5 low | first four | 4/4 | 44 | not read (cost) | 0.95 [0.89, 1.00] | 0.93 [0.86, 1.00] | +0.02 [−0.05, +0.09] |
| | reviewed six | **1/6, PENDING** | 4 | not read | 1.00 | 1.00 | +0.00 |
| | pooled ten | **5/10, PENDING** | 48 | not read | 0.96 [0.90, 1.00] | 0.94 [0.85, 1.00] | +0.02 [−0.04, +0.08] |

False alarms:

| reader | set | version | FA / paragraph | FA / 1,000 words | clean copies only: flags/questions, /para, /1kw |
|---|---|---|---|---|---|
| GPT-5 mini high | first four | clarified | 0.165 [0.139, 0.194] (837/5,083) | 1.94 [1.53, 2.44] | 67/448, 0.150, 1.82 |
| | | original | 0.212 [0.176, 0.249] (697/3,285) | 2.39 [1.96, 2.85] | 49/279, 0.176, 1.90 |
| | | garbled | 0.278 [0.234, 0.327] (698/2,510) | 3.90 [3.25, 4.62] | 64/228, 0.281, 4.00 |
| | reviewed six | clarified | 0.269 [0.240, 0.299] (1,385/5,143) | 3.09 [2.50, 3.71] | 191/704, 0.271, 3.08 |
| | | original | 0.289 [0.258, 0.323] (1,040/3,594) | 3.06 [2.71, 3.43] | 169/527, 0.321, 3.54 |
| | | garbled | 0.354 [0.313, 0.396] (925/2,610) | 4.65 [4.07, 5.24] | 131/368, 0.356, 4.68 |
| | pooled ten | clarified | 0.217 [0.198, 0.238] (2,222/10,226) | 2.52 [2.17, 2.89] | 258/1,152, 0.224, 2.61 |
| | | original | 0.253 [0.228, 0.277] (1,737/6,879) | 2.75 [2.47, 3.04] | 218/806, 0.270, 2.96 |
| | | garbled | 0.317 [0.286, 0.348] (1,623/5,120) | 4.29 [3.84, 4.75] | 195/596, 0.327, 4.43 |
| GPT-5.5 low | first four | original | 0.144 [0.109, 0.180] (473/3,285) | 1.62 [1.21, 2.05] | 34/279, 0.122, 1.32 |
| | | garbled | 0.181 [0.136, 0.228] (455/2,510) | 2.54 [1.93, 3.19] | 38/228, 0.167, 2.38 |
| | reviewed (1/6, PENDING) | original | 0.372 [0.279, 0.478] (127/341) | 4.78 [3.46, 6.23] | 26/69, 0.377, 4.84 |
| | | garbled | 0.365 [0.248, 0.480] (84/230) | 4.84 [3.34, 6.53] | 15/47, 0.319, 4.25 |
| | pooled (5/10, PENDING) | original | 0.165 [0.133, 0.200] (600/3,626) | 1.89 [1.50, 2.32] | 60/348, 0.172, 1.92 |
| | | garbled | 0.197 [0.156, 0.241] (539/2,740) | 2.74 [2.19, 3.37] | 53/275, 0.193, 2.71 |

Garbled − original false alarms (`e12.fa_diff_boot`) and papers where garbled > original:

| reader | set | per paragraph | per 1,000 words | papers garbled > original (para / 1kw) |
|---|---|---|---|---|
| GPT-5 mini high | first four | +0.066 [+0.006, +0.124] | +1.50 [+0.72, +2.29] | 3/4, 3/4 |
| | reviewed six | +0.065 [+0.011, +0.118] | +1.59 [+0.88, +2.32] | 6/6, 6/6 |
| | pooled ten | +0.064 [+0.026, +0.103] | +1.54 [+1.03, +2.08] | 9/10, 9/10 (exception 2608.13754v1) |
| GPT-5.5 low | first four | +0.037 [−0.021, +0.094] | +0.92 [+0.17, +1.68] | 4/4, 3/4 |
| | pooled (5/10) | +0.031 [−0.025, +0.085] | +0.86 [+0.13, +1.57] | 4/5, 4/5 |

Exp 12 counts: GPT-5 mini high 22,516 questions (11,028 + 11,488), 0 unparsed, $175.36 by row cost. GPT-5.5 low at snapshot 6,478 questions, 0 unparsed, $126.05 by row cost (first four 5,898, $115.07). Planted paragraphs flagged for a different reason are counted separately (`--scores` prints them; first four mini-high: 3 orig, 2 garbled, 0 clarified, RPT).

Exploratory exp 12 add-ons, GPT-5 mini high only (RPT details exp 12, weird results 33–37): found-elsewhere flags (planted error named from another paragraph) 0.026 / 0.024 / 0.026 per non-planted paragraph of planted copies (garb / orig / clar), no version difference; FA without them, pooled ten, per 1,000 words 4.10 / 2.57 / 2.33. Without "stable" paragraphs (flagged in ≥80% of a version's copies) garbled is highest on 8/10 papers, tied on 2. Heading-split recheck: all 3 Dropout garbled misses caught on a second sample (sampling noise).

### 4.4 Blog figure values

| figure | script | data | measured or mock |
|---|---|---|---|
| Fig 1 whole paper | `blog_part2_final.py::fig1` | exp 11 unaided, **pooled ten**, `bd.levels` | measured; values = §4.1 pooled rows. GPT-5.5 0.86 / 0.84 / 0.67 |
| Fig 2 every paragraph | `blog_part2_final.py::fig2` | **hard-coded** numbers, pooled ten | GPT-5 mini bars measured and verified (caught 0.99 [0.96, 1.00] / 0.95 [0.90, 0.99] / 0.94 [0.89, 0.99]; FA/1kw 2.52 [2.17, 2.89] / 2.75 [2.47, 3.04] / 4.29 [3.84, 4.75]). **GPT-5.5 bars are MOCK** (caught 0.97/0.97/1.00, FA 1.9/1.2/1.0 placeholders, light red, no intervals) |
| Fig 3 section by section | `blog_part2_final.py::fig3` | exp 9 A, first four for 4 readers; GPT-5.5 reviewed six from `gpt55-A-anchors` + `-anchors-clar` | measured; = §4.2 rows |
| oracle figure | `blog_part2_final.py::fig_oracle` | exp 11 pointed, first four | measured: GPT-5.5 1.00/0.98/0.95, GPT-5 mini 1.00/1.00/1.00, Luna 0.95/0.91/0.93, GPT-5.4 mini 1.00/0.91/0.82 |
| Fig 4 calibration map | `blog_part2_calibration_map.py` | (x, y) = (orig, garbled) unaided, `bd.summary`. Whole paper first four: GPT-5 mini (0.693, 0.670), GPT-5.4 mini (0.523, 0.466), Luna (0.625, 0.591), GPT-5.5 (0.920, 0.716); GPT-5.5 reviewed (0.750, 0.618). Sections first four: GPT-5 mini (0.636, 0.682), GPT-5.4 mini (0.341, 0.364), Luna (0.545, 0.477), GPT-5.5 (0.818, 0.636); GPT-5.5 reviewed sections (0.658, 0.684). Every paragraph: GPT-5 mini (0.95, 0.93) hard-coded = first four | all measured except **GPT-5.5 every-paragraph triangle (0.97, 0.97) = MOCK**; measured first-four value is now (0.95, 0.93). Target box x ≥ 0.70, y ≤ 0.35 is a proposal, not a preregistered bar |
| Fig 5 levers | `blog_part2_levers.py` | first four only, `bd.summary` / per-error subsets | measured. Panels: which reader (w11 points); whole → sections (w11 → A); GPT-5.5 notes 1,200 (0.93, 0.86) → 600 (0.91, 0.80) → 300 (0.82, 0.64); told where (w11 unaided → pointed); difficulty and distance (§4.1 subset table, first-four column) |

Fig 2 and Fig 4 mix sets: Fig 2 pools ten papers, Fig 4's every-paragraph point uses the first four.

---

## 5. Coverage matrix

Versions: C = `clar-explainer-orig`, O = `orig`, G = `obf-max`, X = `clar-explainer-obf`. "done" = rows present at the snapshot commit.

| reader | condition | first four | reviewed six | file |
|---|---|---|---|---|
| GPT-5.5 low | whole paper (exp 11) | done C O G | done C O G | `reader-w11-gpt55-low.jsonl` |
| GPT-5 mini high | whole paper (exp 11) | done C O G | done C O G | `reader-w11-mini-high.jsonl` |
| GPT-5.6 Luna low | whole paper (exp 11) | done C O G | done C O G | `reader-w11-luna-low.jsonl` |
| GPT-5.4 mini medium | whole paper (exp 11) | done C O G | done C O G | `reader-w11-mini54-med.jsonl` |
| GPT-5 mini high | whole paper, 1 sample (exp 9 reader 2) | done C O G X | not run | `reader-whole-mini-high.jsonl` |
| GPT-5.5 low | sections, 300-word notes | done C O G X | done O G (`gpt55-A-anchors`), C (`gpt55-A-anchors-clar`); X not run | `reader-gpt55-A*.jsonl` |
| GPT-5 mini high | sections, 300 | done C O G X | **PENDING** C O G + clean (INBOX 17 Nov item 1) | will write a sibling file, e.g. `data/v4/err9/reader-mini-high-A-anchors.jsonl` (exact name chosen by the pod) |
| GPT-5.6 Luna low | sections, 300 | done C O G X | **PENDING** (item 1) | e.g. `reader-luna-low-A-anchors.jsonl` |
| GPT-5.4 mini medium | sections, 300 | done C O G X | **PENDING** (item 1) | e.g. `reader-mini54-A-anchors.jsonl` |
| GPT-5.5 low | sections, 600 (A600) | done O G | done O G | `reader-gpt55-A600.jsonl` |
| GPT-5.5 low | sections, 1,200 (A1200) | done O G | 4 of 6 done O G; **PENDING** 2506.22982v1, 2510.17247v2 (INBOX 17 Nov item 2) | `reader-gpt55-A1200.jsonl` (resumes from `data/v4/err9/reader-calls.jsonl`) |
| GPT-5 mini high | every paragraph (exp 12) | done C O G | done C O G | `data/v4/exp12/p12-mini-high.jsonl` |
| GPT-5.5 low | every paragraph (exp 12) | done O G; C skipped for cost | 2601.02896v4 done O G; **PENDING (running at snapshot)** the other 5 (INBOX 16 Nov item 1) | `data/v4/exp12/p12-gpt55-low.jsonl`, calls `calls-p12-gpt55-low.jsonl`, done file `logs/exp12-gpt55.done` (not in repo) |
| GPT-5.5 low | subsets for all 10 papers (no API) | — | **PENDING** in `report-v4-details.md` (INBOX 17 Nov item 3; values computed here in §4.1) | — |
| any | held-out nine | sealed | sealed | none |

Pending approvals: 16 Nov note, exp 12 GPT-5.5 ≤$420 (run guard: no new paper after $340 spent by the run) and anchors clarified ≤$60 (done, $21.84). 17 Nov note, items 1 + 2 ≤$80 together, exploratory fill-in, "frozen settings, no tuning", one slice first, then one table row per reader in `report-v4.md`, `NOTIFY_JASON.md`, `STOP`. Both notes say nothing else runs.

Not run, by Jason's instruction (INBOX 14 Nov 00:00 and 22:10): held-out run, anchor addendum's remaining steps, ladder restart, experiment 10 (unrestrained clarification), `unsupported` second prompt. `omitted` errors were not read in exp 11/12.

---

## 6. Preregistration status

Preregistrations live in `docs/plan-v4.md`; section names below.

| result | status | source |
|---|---|---|
| Exp 11 unaided orig and garbled rates, orig − garb with interval, per set | preregistered | "Experiment 11 preregistration", Measures (committed pass 43 before data) |
| Exp 11 pooled ten | preregistered as "also pooled", but never reported by the pod; first appears in the blog Fig 1 | same |
| Exp 11 clarified-original rate | amendment (pass 43, before any clarified read) | "Amendment (pass 43 ...): third version" |
| Exp 11 pointed, control, false accusations | preregistered | Measures |
| Exp 11 per-level rates | preregistered secondary | Measures, "Secondary, reported" |
| Exp 11 evidence-distance bins | exploratory (tertile cuts chosen in `blog_part2_levers.py` after data) | none |
| Exp 11 recount without 2509.11379v3 | exploratory | weird result 32 |
| Held-out choice rule (largest first-four diff among settings with reviewed diff > 0) | preregistered; picks GPT-5.5 low whole paper; not acted on | exp 11 preregistration, "Feeds the held-out choice" |
| Exp 12 caught per version and orig − garb, first four and reviewed separately | preregistered | "Experiment 12 preregistration" |
| Exp 12 pooled ten (blog Fig 2) | exploratory (preregistration says "calibration and anchors separately") | — |
| Exp 12 FA per paragraph, clean copies | preregistered | same |
| Exp 12 FA per 1,000 words | amendment by Jason (INBOX 14 Nov 01:00), "primary reading" | "Measure added by Jason" |
| Exp 12 FA intervals (`fa_boot`) | exploratory, defined pass 58 after 7 of 10 papers' point rates | "Exploratory interval for false-alarm rates" |
| Exp 12 garbled − original FA interval (`fa_diff_boot`) | exploratory, defined pass 60 after the 10-paper rates | "Exploratory interval on the difference" |
| Exp 12 inherited/new split, found elsewhere, heading split, stable, regroup | exploratory, each defined in plan-v4 before its computation | plan-v4 exp 12 bullets (passes 47, 48, 52); `--stable` and `--regroup` documented in report only |
| Exp 12 clarified version | amendment (pass 43) for GPT-5 mini; skipped for GPT-5.5 | amendment; INBOX 16 Nov |
| Exp 9 per-version unaided and pointed rates (first four) | preregistered | "Experiment 9 error stage preregistration", Measures |
| Exp 9 audit scores, separations, verdicts, selection (gpt55-A pointed frozen) | preregistered | same, "Proposed bars and selection rule" |
| Exp 9 orig − garb unaided difference (blog Fig 3, map, levers) | exploratory reframing (Jason's direct test, INBOX 14 Nov 00:00) | — |
| Exp 9 anchors orig/garbled pointed rows, length rule | preregistered | "Experiment 9 anchor addendum" |
| Exp 9 anchors clarified original | exploratory, calibration only (INBOX 16 Nov item 2) | — |
| A600 | preregistered ladder step; verdict "does not pass" | addendum, "memory-budget ladder" |
| A1200 | preregistered, stopped incomplete; no verdict | same; run-log pass 43 |
| Exp 9 pooled ten, three rates for reviewed six | exploratory (this report / blog) | — |
| 17 Nov fill-ins (section readers on anchors, A1200 last 2, subsets) | exploratory fill-in, explicitly "not calibration" | INBOX 17 Nov |

**Deviations from preregistration:**
1. Held-out run preregistered (pass 35) and the choice rule applied, but the run is paused (Jason, 14 Nov); 9 papers sealed, no confirmatory test exists.
2. Anchor addendum: A1200 stopped at 8/10 papers, no verdict; the ladder never completed, so the frozen setting it was meant to choose was never fixed.
3. Exp 12 GPT-5.5: preregistered condition was a projection under $250 (amended to $420 with the third version); the 54-question check projected $550–750, so it waited for Jason; he then approved ≤$420, and since the re-projection ($482–500) exceeded it, only `orig`/`obf-max` run. No clarified rate for GPT-5.5 in exp 12.
4. Exp 12 paragraph splitter changed before any planted question (list-merge fix, run-log pass 43); clean-orig check rows superseded (`data/v4/exp12/superseded/`).
5. Exp 11 salvage parser added mid-run (pass 43) before rates were read; readers rerun from cache (pass 44).
6. Exp 9 `whole_review` reparse fallback for exp 11 readers only (pass 43).
7. The blog's framing (three rates, orig − garb) is not the exp 9 preregistered audit score; exp 9 verdicts were on audit scores.
8. GPT-5.5 as "the reader that works" was identified after seeing first-four and reviewed data (report "Why this is still weak": "1 of 4 readers, picked after the fact"). The held-out run was the planned guard and has not run.
9. Exp 12 FA treats real inconsistencies in the original papers as false alarms (preregistered definition); weird results 33/36 show this inflates per-paper rates.

---

## 7. Weird results (`docs/report-v4-details.md#weird-results`)

Status words from the source. 1–20 are from experiments 0–8 (V1 copies), before Part 2's design.

| # | one line | status | implication |
|---|---|---|---|
| 1 | V1 illegible copies had rewrite-introduced contradictions | fixed (42 patches, 3r rerun moved rates <0.1) | copies may differ in content, not only legibility; motivates `obf-max` consistency check |
| 2 | gpt-5-mini accuses `compobf` more than luna (9/9 vs 6/9) | open | none for Part 2 |
| 3 | gpt-4.1-mini near-empty reviews | probably weak reader | weak readers may not search |
| 4 | no reader caught a major rewrite contradiction in 2608.13754v1 `compobf` | open | garbling can hide real errors; here a defect |
| 5 | Sol inventory listed a false contradiction (Jaccard 0.182 is correct) | explained, Sol error | inventory is not ground truth |
| 6 | strong reader missed 2 errors only in `compobf` | open; gate later set to clarified copies only | old gate may remove the very errors the audit wants |
| 7 | one error missed by strong reader in all 4 copies | probably weak error | gate works as intended |
| 8 | stronger readers sometimes catch more in illegible copies | open, within noise | small cells are noisy |
| 9 | clarified copy does not beat original for strong readers | open | originals may already be at ceiling (also in exp 11 for GPT-5.5) |
| 10 | 11/880 planted reviews unparsed | excluded | small |
| 11 | pair check flips verdict on identical edits (~1 in 10) | partly explained, 1 Sol sample | survival rule is noisy |
| 12 | rates slightly lower on repaired copies | open, noise | none |
| 13 | luna-med catches fewer in `obf` than `compobf` | open | noise |
| 14 | a control found a real overclaim in 2609.02893v1 | explained | real tensions can be credited as catches; avoid-list it |
| 15 | number check flagged formatting as missing numbers in `scatter` | explained | number check needs normalisation |
| 16 | mini-high accused clean `scatter` copies of 3 planted errors | explained for 2 (real inconsistency in original) | grader location leniency |
| 17 | luna-low finds fewer in clarified when several errors planted | partly explained (stops after one find) | multi-error designs reward early stopping |
| 18 | 9 limited-view steps failed to parse (LaTeX in JSON) | fixed, rerun | parser fallbacks matter; luna's per-copy noise is large |
| 19 | limited-view mini-high falsely accuses `compobf` 21% | partly explained | chunked readers over-accuse dense text |
| 20 | variant B gave a smaller gap than A | probably noise | A's interval excluding 0 is not a stable effect |
| 21 | most "added" items in clarified-from-`obf-max` copies are kept codes | explained | prompt rule partly measured checker grouping; only `explainer` passes as preregistered |
| 22 | Dropout clarifications shorter than `orig` | explained (raw PDF noise) | Dropout length ratios not comparable |
| 23 | one Dropout fidelity reply failed to parse (control chars) | fixed | none |
| 24 | strong reader flagged 0/25 `unsupported` | explained (contradiction-only prompt) | category untestable; excluded |
| 25 | 9/28 `calculation` errors could not be planted in `obf-max` | explained (qualitative prose) | `obf-max` biases which errors survive |
| 26 | exp 9 mini-high-A catches more in `obf-max` than `orig` | explained: noise + shorter copy puts evidence closer | chunked reader benefits from compression; length confound |
| 27 | 14/44 whole-paper reads empty (16k token cap) | fixed, rerun at 32k | output allowance must be 32k for high effort |
| 28 | 10 luna pointed replies ungraded (LaTeX) | fixed, replay | verdict unchanged |
| 29 | mini54-A pointed audit score reversed | explained: selection effect of the score's definition | pointed audit score is biased for readers that misjudge when pointed |
| 30 | gpt55-A pointed catch in `orig` (0.75) below unaided (0.82) | explained: pointed check sees only notes before the planted chunk | pointed rates in exp 9 are a lower bound when evidence comes later |
| 31 | two anchors (2603.01214v2, 2509.11379v3) show no separation at A | open (theory-paper guess untested) | anchor separation is noisy; see 32 |
| 32 | every whole-paper reader "falsely accuses" clean `obf-max` of 2509.11379v3 | explained: real quirks of the original (α = 1, `A_n`, Cor 4.1), grader location leniency | reviewed-six garbled false accusations are inflated by one paper; recount in §4.1 |
| 33 | exp 12 flags ~25% of paragraphs on 2608.13754v1 | partly explained: real inconsistencies of the original, most not in inventory | FA rates mix real flaws with noise; compare versions within paper, not papers |
| 34 | found-elsewhere hand-label agreement fell to 16/30 | fixed (labels keyed by identity), 30/30 | none |
| 35 | exp 12 garbled misses on Dropout sit late in merged blocks | explained: sampling noise (caught on second sample) | single-ask caught rates carry a few points of per-copy noise |
| 36 | 2601.02896v4 has ~2× FA rate | explained: stable paragraphs naming real weaknesses | FA rate depends on how flawed a paper is |
| 37 | exp 12 misses causal-hard-2 on 2509.22621v3 in O and G, not C | explained: inherited overclaim in same paragraph takes the one answer | one-conflict-per-paragraph format; clarification smoothed a real flaw |

---

## 8. Spend

Ledger `data/v4/spend.jsonl` (172,989 rows at snapshot: 86,564 `reserve`, 86,420 `settle`, 5 `reprice`). Grouping: `settle` rows, sum of `actual`, by tag prefix (`err9:<stage>:<reader>:...`; pointed tags are `err9:pointed:<cond>:<reader>:...` and grouped by reader). "List price" recomputes every settle from its `usage` with current `llm.PRICES` (×0.5 for Batch rows); it differs from `actual` only for calls before the 2026-11-13 reprice. Script used: this report's scratch script; snippet in §9.

| experiment / item | settles | actual USD | list-price USD |
|---|---|---|---|
| Exp 0–8 and their checks (exp0, exp2, exp3, exp3r, exp4–7, exp8A/B/C, inventory, repair, fidelity, clarify, lrcheckA) | — | 306.25 | 92.63 |
| Exp 9 clean versions (`exp9:*`: obf-max, 5 clarification prompts, checks; 10 papers) | 338 | 41.06 | 41.06 |
| Exp 9 error construction (`err9` design 3.35, plant 10.04, strong 60.45, pair 2.88, grade 3.16, inventory 0.41) | 3,387 | 80.29 | 80.29 |
| Exp 9 readers, first four: mini-high-A 24.63, whole-mini-high 12.36, luna-low-A 3.33, mini54-A 15.85, gpt55-A 79.78 | — | 135.95 | 135.95 |
| Exp 9 anchors: gpt55-A-anchors 36.24, gpt55-A-anchors-clar 21.84 | — | 58.08 | 58.08 |
| Ladder: gpt55-A600 89.56, gpt55-A1200 104.17 | — | 193.73 | 193.73 |
| Exp 11: w11-gpt55-low 52.23, w11-mini-high 16.06, w11-luna-low 3.47, w11-mini54-med 23.55 | — | 95.31 | 95.31 |
| Exp 12: p12-mini-high 177.43, p12-gpt55-low 133.62 (settled at snapshot), elsewhere 2.26, headsplit 0.82, grade 0.18 | — | 314.31 | 314.31 |
| **all settles** | 86,420 | **1,224.98** | **1,011.36** |

Reconciliation to the ledger total `python3 scripts/v4/llm.py` → **$1,058.93 of $2,000**: settles at actual 1,224.98 + 5 reprice rows −213.62 (gpt-5-mini −79.52, luna −108.25, Sol −24.68, gpt-4.1-mini −0.89, nano −0.28) = 1,011.36; + 189 unsettled reservations $53.57 (charged in full because a lost request may still be billed: exp12 GPT-5.5 in flight at snapshot $17.23, err9 reads/pointed from killed runs ~$30.06, exp8 $6.06, small others); − $6.00 from 45 exp8 settle rows whose reserve rows are absent from the ledger (settle `usd` −5.11 with actual 0.89).

Differences from report figures: A1200 ledger $104.17 vs report/STATE "about $94" (run-log pass 43 notes calls in flight settled after the stop); A600 $89.56 vs "about $87"; exp 12 mini-high $177.43 ledger vs $175.36 row cost (ledger includes the 56/54-question checks and superseded rows); exp 11 $95.31 vs "about $97" (report figure is row cost incl. grades). `report-v4.md`'s "Spend: $886" line is stale (pass 59).

Pending budgets: ≤$420 exp 12 GPT-5.5 (projected $245–285 for 10 papers × 2 versions), ≤$80 for INBOX 17 Nov items 1 + 2. Cap $2,000 (`V4_SPEND_CAP`).

---

## 9. Reproduction

All analysis below is local, no API calls (call caches are read, never extended, unless a `read`/`construct`/`regrade`/`--elsewhere`/`--headsplit` command finds an uncached call). Run from the repo root. Never print `.env`.

**Blog figures** (each writes a tracked PNG; use a scratch checkout or `git checkout -- docs/blog` afterwards if you only want numbers):

| figure | command | writes | prints |
|---|---|---|---|
| Fig 1, Fig 2, Fig 3, oracle | `uv run -q --with matplotlib --with openai --with requests python scripts/blog_part2_final.py` | `docs/blog/part-2/fig-2-whole-paper.png`, `fig-3-every-paragraph.png`, `fig-4-section-by-section.png` | only "wrote ..."; Fig 2 is hard-coded (edit `fig2()` to replace the GPT-5.5 MOCK) |
| Fig 4 map | `uv run -q --with matplotlib --with openai --with requests python scripts/blog_part2_calibration_map.py` | `docs/blog/part-2/fig-5-map.png` | nothing; points in `points()` |
| Fig 5 levers | `uv run -q --with matplotlib python scripts/blog_part2_levers.py` | `docs/blog/part-2/fig-6-levers.png` | point coordinates per panel |
| older direct-test figures | `uv run -q --with matplotlib python scripts/blog_part2_direct.py` | `docs/blog/part2-direct-fig-*.png` | per-reader orig/garbled/diff lines |

**Tables:**

| table | command | expected key output |
|---|---|---|
| Exp 11 (§4.1 per-set rows, not pooled) | `uv run -q --with matplotlib python scripts/v4/exp11_whole.py` (writes `docs/figures/v4-exp11-whole.png`) | e.g. "GPT-5.5 low [4 calibration papers]: 44 errors ... orig-obf +0.20 [+0.07, +0.34]" |
| Exp 12 | `uv run -q --with matplotlib python scripts/v4/exp12_paragraphs.py --scores p12-mini-high p12-gpt55-low` (writes `docs/figures/v4-exp12-paragraphs.png`) | per set: caught line, FA per version with intervals, garbled − original FA, papers count, per-paper FA |
| Exp 12 exploratory | `python3 scripts/v4/exp12_paragraphs.py --stable p12-mini-high`, `--regroup p12-mini-high` (no API); `--elsewhere`, `--headsplit` (cached; may call API if uncached) | weird results 35–36 numbers |
| Exp 9 readers, audit scores | `uv run -q --with matplotlib python scripts/v4/err9.py --scores mini-high-A whole-mini-high luna-low-A mini54-A gpt55-A` (writes `docs/figures/v4-exp9-readers.png`) | tables in §4.2 "audit scores" |
| Exp 9 anchors / ladder | `... err9.py --anchors`, `--anchors gpt55-A600`, `--anchors-clar` (writes anchor figures) | anchor table, rule (a)/(b) |
| Exp 9 survival | `... err9.py --table` (writes `docs/figures/v4-exp9-survival.png`) | per-paper failure counts (iterates first four only: `err9.PAPERS`) |
| Spend | `uv run -q --with matplotlib --with openai python scripts/v4/llm.py` | "spent $1058.9305 of $2000.00" at snapshot |

**Pooled-ten and per-set values without writing files** (what this report ran):

```python
# run from repo root: uv run -q --with matplotlib --with openai python this.py
import sys; sys.path[:0] = ["scripts", "scripts/v4"]
import blog_part2_direct as bd, exp12_paragraphs as e12, exp11_whole as e11
ALL = tuple(bd.CAL) + tuple(bd.ANCHORS)
print(bd.levels("reader-w11-gpt55-low.jsonl", "planted", ALL))           # 3 rates, pooled ten
print(bd.summary("reader-w11-gpt55-low.jsonl", "planted", ALL)["diff:ci"]) # orig - garbled interval
rows = [r for r in e12.load("p12-mini-high") if r["paper"] in ALL]
print(e12.boot(e12.caught_by_error(rows)))                                  # caught, pooled ten
print(e12.fa_rate(rows, "obf-max", True), e12.fa_boot(rows, "obf-max", True), e12.fa_diff_boot(rows, True))
# pooled exp 9 GPT-5.5 (bd.summary_multi): merge bd.per_error over files, then bd.summary's bootstrap
pe = {**bd.per_error("reader-gpt55-A.jsonl", "planted", ALL), **bd.per_error("reader-gpt55-A-anchors.jsonl", "planted", ALL)}
```

The exp 12 bootstrap (`fa_clusters`) rebuilds every planted text; a full run over both readers takes ~2.5 min.

**Raw data:**

| what | where |
|---|---|
| errors (design, edits per version, gate results) | `data/v4/err9/errors.jsonl` |
| exp 9 / exp 11 rows | `data/v4/err9/reader-<rd>.jsonl`; `kind` ∈ `clean`, `planted`, `cgrade` (clean read graded vs one error), `pointed` (`cond` `planted`/`control`) |
| exp 9 call cache (all non-w11 readers incl. ladder and anchors) | `data/v4/err9/reader-calls.jsonl` (>50 MB) |
| exp 11 call caches | `data/v4/err9/reader-calls-w11-<rd>.jsonl` |
| construction calls | `data/v4/err9/calls.jsonl` |
| exp 12 rows / calls | `data/v4/exp12/p12-<rd>.jsonl` (one row per paragraph question: `copy`, `para`, `planted`, `flag`, `caught`, `words`), `calls-p12-<rd>.jsonl`; exploratory `elsewhere-*`, `headsplit-*`, `new-fa-labels-*`; `superseded/` |
| versions | `data/v4/versions/<paper>/*.md`, `manifest.jsonl`, `obf-max-log.jsonl`, `clarify-log.jsonl` |
| failed / superseded rows | `data/v4/err9/failed/`, `data/v4/exp12/superseded/` |
| spend ledger | `data/v4/spend.jsonl` |

**Caching and resume (`scripts/v4/llm.py`):** `Calls(path).call(model, system, user, tag=, sample=, max_tokens=, effort=)` keys each call by sha256 of `[model, system, user, params, sample]`; a key present in the calls file (with `content`) returns the logged reply with no API call, so rerunning any `read` command replays from cache and only makes calls for missing keys. Failed calls are logged with `error` and not cached, so a rerun retries them. Up to 4 attempts, exponential backoff, no retry on HTTP 4xx except 429. Reader row files are rewritten per paper at the end of that paper (`err9.py read`, `exp12_paragraphs.py read`), so a killed run loses only the in-progress paper's rows, not its calls. Before every call `_reserve` appends a worst-case `reserve` row (prompt chars / 2.5 × input price + full output allowance × output price) under a file lock and refuses (`CapReached`) if ledger total + worst case > cap; `_settle` appends the correction. Env vars: `V4_SPEND_CAP` (default 2000), `V4_SPEND_FILE` (default `data/v4/spend.jsonl`), `V4_CONCURRENCY` (default 12 in `llm.py`; exp 12 `read` uses its own default 16), `OPENAI_API_KEY` (else read from `.env`). Batch API helpers exist (`submit_batch`, `collect_batch`, half price) but Part 2 reader runs were synchronous. Repo rule (STATE): while a job writes `spend.jsonl` or a calls file, never `git stash`, `git pull --rebase` or check out.

---

## 10. Glossary

| term | meaning |
|---|---|
| first four / CAL / calibration papers | 2609.30094v2, 2608.13754v1, 2609.02893v1, jmlr-2014-dropout (`bd.CAL`, `err9.PAPERS`); settings were chosen on these |
| reviewed six / ANCHORS / anchors | 6 accepted TMLR/ICLR 2026 papers drawn by length band (`bd.ANCHORS`, `exp12_paragraphs.ANCHORS`) |
| held-out nine | sealed core papers (`err9.HELDOUT`), never read |
| development papers | the 3 arXiv papers of the first four (used since experiment 0) |
| `orig` | original paper text |
| `obf-max` | maximally garbled rewrite ("garbled version"); `obf-max-r2` = second sample when the first failed its check |
| `clar-explainer-orig` | clarified original ("clarified original"), `explainer` prompt from `orig`; calibration standard only |
| `clar-explainer-obf` | `explainer` clarification written from `obf-max`; used in exp 9 gate and exp 9 clarified-copy score |
| `explainer`, `base`, `plain`, `cohesion`, `twostage` | the 5 clarification prompts in `clarify.py`; only `explainer` passed |
| `obf`, `compobf`, `comp`, `imp2`, `scatter` | V1-era illegible copies used in experiments 0–8, not in Part 2 |
| survivor | designed error that passed planting, strong-reader gate and pair check (`survives: true`) |
| SURV / primary set | survivors excluding `omitted` and `unsupported` (`blog_part2_direct.SURV`; `err9.SETS["primary"]`) |
| strong reader / gate | gpt-6.1-sol medium whole-paper review on both clarified copies; must catch |
| pair check | Sol medium compares all 4 planted versions for equal planting |
| inventory / avoid list | Sol-listed real contradictions in `orig`; errors may not touch them |
| A | exp 8/9 limited-view variant: section chunks ≤800 words, notes ≤300 words (`exp8_limited.VARIANTS`) |
| A600 / A1200 | same with notes ≤600 / ≤1,200 words (ladder steps) |
| B, C | exp 8 variants (notes ≤150; paragraph chunks ≤250), not in Part 2 |
| `gpt55-A`, `mini-high-A`, `luna-low-A`, `mini54-A` | exp 9 section-by-section readers, first four |
| `whole-mini-high` | exp 9 reader 2, whole paper, 1 sample; its sample 0 is reused as sample 0 of `w11-mini-high` |
| `gpt55-A-anchors` | frozen exp 9 reader on the 6 anchors, `orig`/`obf-max` |
| `gpt55-A-anchors-clar` | same reader, `clar-explainer-orig` only (16 Nov) |
| `gpt55-A600`, `gpt55-A1200` | ladder readers on first four + anchors |
| `*-heldout` | reader names defined for the held-out run; never run |
| FROZEN, EXP11 | sets in `err9.py`: frozen readers read `orig`/`obf-max` primary only; EXP11 readers have own call caches and 2 samples |
| w11, `w11-<rd>` | experiment 11 whole-paper readers (`w11-gpt55-low`, `w11-mini-high`, `w11-luna-low`, `w11-mini54-med`) |
| p12, `p12-<rd>` | experiment 12 every-paragraph readers (`p12-mini-high`, `p12-gpt55-low`) |
| mini-high, mini54, luna, Sol | gpt-5-mini high; gpt-5.4-mini; gpt-5.6-luna; gpt-6.1-sol |
| planted (kind) | unaided review of a planted copy |
| clean (kind) | review of an unplanted version |
| cgrade | grade of a clean read against one primary survivor; caught = false accusation of that error |
| pointed | reader shown the planted passage (cond `planted`) or the same unplanted passage (cond `control`) and asked if it is consistent |
| pgrade | grade of a pointed reply |
| control accusation | control pointed check graded as a catch |
| false accusation (exp 9/11) | clean read with an item graded as a catch of some survivor |
| false alarm (exp 12) | flag on a non-planted paragraph |
| inherited / new | exp 12 planted-copy FA on a paragraph also flagged / not flagged in the clean copy |
| found elsewhere | exp 12 flag on a non-planted paragraph that the grader credits as naming the planted error |
| stable paragraph | flagged in ≥80% of a version's copies (`--stable`) |
| headsplit, regroup | exp 12 exploratory checks (weird result 35; planted-by-merge paragraphs) |
| separation | exp 9: `orig` row − `obf-max` row of an audit score |
| clarified-copy audit score | catch(version) / catch(its clarified copy) |
| pointed audit score | share found unaided among errors judged correctly when pointed |
| meets / close / no | exp 9 proposed bars (`plan-v4.md`); close = orig row ≥0.75, separation ≥0.15 with interval excluding 0, FA ≤0.15 |
| rule (a), (b) | anchor addendum length rule: every anchor pointed orig row ≥0.75; long − short ≥ −0.15 or interval includes 0 |
| ladder | A → A600 → A1200 notes budget increase if the length rule fails on long anchors |
| held-out choice rule | largest first-four orig − garb among settings with reviewed diff > 0; selects GPT-5.5 low whole paper |
| MOCK | light-red placeholder bar or point in blog scripts, not measured |
| reprice | 5 ledger rows (2026-11-13) correcting pre-list-price costs |
| pass N | pod agent loop iteration; `docs/run-log.md` and `STATE.md` are keyed by pass |
| weird result N | numbered anomaly in `report-v4-details.md` |
| V1 | tag of the earlier quiz-based pilot; source of the dev papers' `orig` |

---

## Known gaps and discrepancies (for agents)

- Exp 9 anchors-clar intervals: `err9.py --anchors-clar` (RPT) gives unaided orig 0.66 [0.50, 0.82], clarified 0.74 [0.61, 0.87]; `bd.levels` gives 0.66 [0.50, 0.79], 0.74 [0.63, 0.84]. Point values agree; interval code differs.
- Exp 9 `whole-mini-high` unaided: RPT orig 0.68, obf 0.70 (`err9.scores`); `bd.levels`/`bd.summary` give 0.67 and 0.72 (n = 43–44; the 2 unparsed rows are handled differently).
- Blog Fig 2 and the Fig 4 GPT-5.5 triangle are MOCK. Measured GPT-5.5 exp 12 values (first four) now exist: caught orig 0.95, garbled 0.93; FA/1kw 1.62 and 2.54.
- `blog_part2_final.py` docstring still says experiments 11 and 12 and clarified anchors are mock; only Fig 2's GPT-5.5 bars are.
- `report-v4.md` spend line ($886) and parts of `STATE.md` are stale relative to the ledger ($1,058.93).
- Superseded by the 2026-11-18 update: every run marked PENDING below has finished, and its numbers are in that update.
