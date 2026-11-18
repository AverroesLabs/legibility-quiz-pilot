# Decisions

Newest first. Add a row for each decision that changes the plan.

| Date | Decision | Made by | Why |
| --- | --- | --- | --- |
| 2026-11-04 | Publish as the blog post plus a new, separate public repo with a clean history: code, data, reports, plans, preregistrations, decisions, literature review and figures. No agent or operations files, no internal Averroes material. This repo stays private. Licences: MIT for code, CC BY-SA 4.0 for data and text | Jason | The private history holds internal docs, colleagues' comments, account details and a personal email. CC BY-SA covers the two share-alike papers' rewritten versions |
| 2026-11-04 (pass 20) | **Exploratory direct-judgment baseline, fixed before any rating was collected** (Jason's note; `docs/plan-v2.md`). Prompt `JUDGE_V2` (j1): "How easy is this paper for a reader at your level to understand, from 1 to 10?", anchors at 1 (cannot follow method, results or conclusions), 5 (follows parts) and 10 (follows all, could explain them), JSON reply with a reason of at most 50 words. The 4 quiz readers x 4 versions x 12 core papers x 3 samples = 576 calls, one version per call, temperature 0.7, seed from the sample; an unparseable reply is asked once more. Readability formulas (textstat: Flesch-Kincaid grade, Flesch reading ease, Gunning fog) on each version with LaTeX and markup stripped. **Comparison, all descriptive, no bars:** per reader and hard version, (a) share of papers where the original's mean rating is above the version's (ties half), against the same for the quiz's paper-level mean; (b) share of the 9 sample pairs per paper where the original is above, against the same for the quiz's per-sample run scores (figure 13's points); (c) Spearman over the 12 papers of original rating vs original quiz score, and of rating drop vs quiz drop. Figure 14: rating and quiz score side by side, pooled and per paper. Code `scripts/quiz/judge_v2.py` | Jason (baseline), Claude (details) | Plan v2 requires the baseline before the report is final. Fixing the comparison first keeps it from being chosen after seeing ratings. Exploratory because it was added after the preregistration |
| 2026-11-03 (pass 16) | Takes details not fixed by plan v2: sittings split by reader and sample but identical across the five conditions; temperature 0.7, seed from the sample; grading prompt `GRADE_V2` (p7), the old `GRADE` plus a verbatim quote per awarded point, with quotes not found in the answer flagged but not changed; answers cut to 100 words before grading; a counterfactual twin scores 0 in a sample unless both twins earn their conclusion point; information check uses the smart reader's takes on kept questions (no extra calls). The filters keep the old `GRADE`. First core batch is one paper per stratum, to measure cost before fixing the paper count | Claude, before any core-paper result | Identical sittings remove sitting composition as a source of version differences. 100 words is above every reader's 99th percentile in the dev filters (51 to 92), so it only stops outliers. Quotes are plan v2's requirement; the filter decisions were already made with `GRADE`, and re-grading them would cost money without changing what the takes measure. Plan v2 says to measure cost on the first papers before running the rest |
| 2026-11-03 (pass 15) | The smart reader's no-paper filter takes 2 samples, each with its own seeded sitting order and seed, and a question passes if the **mean** of the 2 scores is at most 0.5. The weaker readers keep 1 sample. Writer fixed at 5 batches. Every paper now runs through `scripts/quiz/run_v2.py` (versions in parallel with writing, then the 4 filters in parallel, then combine; all papers in one process, so per-provider limits hold across papers; no stage starts at or above the $439 account-usage stop line) | Claude, in development on the reserve paper before any core-paper result | One sample's pass/fail agreed with a second on only 80% of 111 questions (pass 14), and on 73% of 160 here, so with 1 sample the kept set depends on chance. The smart reader binds the filter (the weaker readers' mean exceeds 0.5 on only 4 questions), so only its samples matter. Mean of 2, not "both at most 0.5": a question at 0.67 and 0.33 is borderline, and the stricter rule would drop more questions without a clearer case for it. Cost about $0.50 per paper. Dev paper: 49 kept, against 57 with 1 sample (`docs/run-log.md`, pass 15) |
| 2026-11-03 (pass 14) | Filter details not fixed by plan v2: the no-paper and with-paper filters use 1 sample per reader, in sittings of at most 25 questions in a seeded random order (same prompts as the takes, `TAKE_NONE` and `TAKE_PAPER`). A counterfactual twin counts as "right" if it earns the conclusion point (rubric point 1) and at least 0.5 with the full original. A pair is one unit in every filter: if either twin fails, both are dropped. Filters run in the plan's order (no paper, with paper, retrieval, pairs), and each is also counted alone on all candidates | Claude, in development on the reserve paper before any core-paper result | One sample keeps filtering cheap; the takes have 3. The conclusion is what differs between twins, so it is the point that must be right for both. Counting each filter alone shows their overlap for the funnel figure |
| 2026-11-03 (pass 13) | The retrieval filter uses two readers, Gemma-4-31B (plan v2) and GPT-5.6-sol, each given the same 3 keyword-retrieved passages, 1 sample, GLM-5.3 grading at minimal reasoning. A question is kept only if both earn at most 1 rubric point, and a counterfactual pair only if both twins pass. Tables are retrieved whole, with their captions | Claude, in development on the reserve paper before any core-paper result | The check asks whether the passages contain the answer, so it needs a reader that does not miss what is there. On the dev paper Gemma alone passed a pure table lookup (q18) by misreading a row. With the smart reader added, every question Gemma rejected was also rejected by GPT-5.6-sol (0 of 111 the other way), so Gemma adds nothing on this paper but is kept as the plan specifies. Cost: about $0.60 per paper for the smart reader. `docs/run-log.md` pass 13 |
| 2026-11-03 (pass 12) | Compression is one call on the whole paper by Claude Opus 5.5 (default reasoning), with prompt `COMPRESS_PAPER` p5: a cut list, shorthand, and per-section character budgets at 40% that a script measures. It is redone up to 3 times if it loses a number, changes the section structure or exceeds 0.5 of the original, and the shortest valid try is used. Replaces section-by-section compression by Kimi. Obfuscation is unchanged (Kimi K3) | Claude, following Jason's 2026-11-03 note (at most 0.5, ideally 0.35 to 0.5; Opus acceptable) | Section-wise compression stalled at 0.54 to 0.70 with every model tried, because a single section cannot cut what other sections repeat. Whole-paper on the dev paper with the p5 prompt: Opus 0.41, 0.39 and 0.44 over 3 tries, against Kimi K3 (medium) 0.53, 0.56 and 0.54. gpt-oss-20b broke the section structure; Qwen3.8-27B reached 0.60 at about 12 min per call. Opus writes quizzes only from the original, so it never sees a version. Risk: the same model chooses what the compressed version keeps and what the quiz asks, which could favour keeping quiz-relevant content. The number check and the information check guard against loss, not against that alignment. Trial: commit 09d3dc3 and `data/runs/v2dev-2607.05615v2/calls.jsonl` |
| 2026-11-03 | Plan v2 versions: Kimi K3 rewrites each section separately and in parallel, using one paper-wide glossary of opaque abbreviations, each spelled out once, in the first section after the abstract that uses it. Obfuscation keeps Kimi's default reasoning; compression uses reasoning effort "low". A section that loses a number is redone, and a compressed section above 0.55 of its source is redone, up to 3 tries | Claude | Kimi is not the writer, a reader or the grader. One call per section avoids output-length limits and runs in parallel. One glossary keeps the abbreviations consistent across sections. At default reasoning, 7 of 11 compression calls returned nothing (`docs/run-log.md`, pass 11) |
| 2026-11-03 | Paper versions for whole-paper tests: no paper (floor), original, obfuscated, compressed, and compressed plus obfuscated. Every version keeps all facts | Jason | A 2x2 of compression and obfuscation separates their effects. The original is assumed legible because it was written for people |
| 2026-11-03 | Grade with GLM-5.3 at reasoning effort "minimal". OpenRouter refuses to turn its reasoning off entirely | Jason, after a check | On 616 re-graded pilot answers: same score 88.6%, rubric-point agreement 95.3% (kappa 0.90), mean change -0.009, with no version shifted by more than 0.03. Cost per answer fell from $0.0137 to $0.00066. See `data/runs/grader-check/summary.md` |
| 2026-11-03 | Focus on whole-paper tests: no paper, original, and whole-paper rewrites from clarified to hardest. Drop section removal and per-section scoring; per-section legibility is a follow-up | Jason | The goal is whole-paper understanding of methods, key results and conclusions. With the whole paper rewritten, every question depends on the changed text, and the no-paper condition is the floor |
| 2026-11-03 08:36 UTC | **Deviation from the preregistered obfuscation pick, before any affected result existed:** if the seeded section has no quiz item tagged to it, the pick moves to the next eligible section in seeded order that has one. Seeded picks that already had tagged items are unchanged (2608.25490v1 s10, 2607.20379v2 s6). Applied to 2609.00755v1 (seeded s5 "Related Works", 0 items, so now s7 "Appendix A Implementation Details", 3 items) and to the three papers not yet obfuscated | Claude | Eligibility counted tagged claims, not tagged items, so on 2609.00755v1 no item could respond to the obfuscation and the clear-versus-obfuscated test was empty by construction. The vacuous version's answers were never graded; they are kept in `data/runs/pilot-2609.00755v1/superseded-obf-s5/` |
| 2026-11-03 | Scaled-down pilot settings, fixed before any core-paper quiz exists: smart taker GPT-5.6-sol; rewriter Kimi K3 (the preregistration's "otherwise" branch, so the smart taker is not also the rewriter and GLM-5.3 is not grading its own answers); ladder readers Ministral-8B (weakest with-paper reader with a clear gain in dev2, 0.71) and Gemma-4-31B; grader GLM-5.3 (about $0.012 per grade, about $8 per paper); write-time screen as in dev2 with GPT-5.6-sol held out. Obfuscation prompt o1 in `scripts/quiz/prompts.py`; eligible sections are those with a tagged claim, excluding s0 and any section titled Introduction. No pipeline step starts once account usage passes $138 | Claude | Fills the open choices in `docs/preregistration.md`. Estimated cost about $14 per paper, about $85 for 6, which fits under the $140 stop from $42.55 |
| 2026-11-03 | Writer prompt p2 forbids questions answerable from their own wording, and a write-time screen sends items that Kimi K3 answers without the paper (mean rubric score above 0.5 over 2 samples) back to the writer, for up to 2 rewrite rounds. Kimi K3 screens because it is not the writer and not a held-out smart taker (GPT-5.6-sol and GLM-5.3), so the four-condition filter stays an independent check | Claude | Under p1, 24 of 27 items were answerable without the paper (`docs/run-log.md`) |
| 2026-11-03 | Develop and tune the pipeline on reserve paper 2607.05615v2, not on a core paper | Claude | Keeps prompt tuning off the 12 papers the pilot will judge |
| 2026-11-03 | Quizzes are taken as one sitting per sample (all items, shuffled order), and graded per answer against a 2 to 3 point rubric by a model from another family, blind to reader and condition. GLM-5.3 is the main grader; Qwen3.8-27B graded a subset as a check | Claude | One sitting per sample costs about 27 times less than one call per item. A per-item check (`paper_single`) was run for two readers to see whether the format matters |
| 2026-11-03 | Fixed provider for `openai/gpt-oss-20b` changed from DeepInfra to CoreWeave | Claude | DeepInfra rate-limited it (`docs/run-log.md`) |
| 2026-11-03 | Selection rule v2 requires a stratum term literally in the abstract | Claude | The arXiv API's phrase search let in off-topic papers. See `docs/run-log.md` |
| 2026-11-02 | The quiz is open book, and items are checked with four conditions: smart and dumb, without and with the paper | Jason | The reader should need understanding, not just the text. The conditions show which items measure the paper |
| 2026-11-02 | Run OneStopQA and eLife digest samples before the main pilot, and AI-written papers after, all exploratory | Jason | To inform next steps |
| 2026-11-02 | Use a range of models through OpenRouter, with no GPU for now. A final standardized reader still needs pinned weights | Jason | Jagged intelligence may affect which model passes which quiz. The Runpod GPU was out of stock and hosted endpoints need a key |
| 2026-11-02 | Knowledge gain becomes a supporting measure, not the pass criterion. Compare gain with per-claim and per-section measures on partly obfuscated papers, using `docs/measure-comparison-plan.md`. The plan's thresholds await Jason's sign-off | Jason's inbox note; plan drafted by Claude | Gain does not show the reader followed the whole paper, and can be low for clear papers on known material |
| 2026-11-02 | Select papers by the fixed hash-ordered rule in `docs/paper-selection.md`: arXiv, 2026-07 to 2026-09, three AI safety strata, CC BY, CC BY-SA or CC0 only | Claude | No hand-picking. Dates aim to post-date reader training data. Those licences allow sharing rewritten versions |
| 2026-11-02 | No account API key on the pod, since Jason cannot create one and does not want to ask the admin. Pods are created from Jason's signed-in session | Jason | Removes most shared-account risk; the agent cannot manage GPUs |
| 2026-11-02 | Use the shared Runpod account | Jason | Jason's compute budget is on this account, and other people's pods and secrets are on it too |
| 2026-11-02 | Claude runs on a Runpod CPU pod in a shell loop of `claude -p` passes (`scripts/agent-loop.sh`). GPUs are rented per batch | Jason, on Claude's recommendation | The pod stays up without Jason's laptop. Cloud routines run at most hourly, and their maximum run length is unconfirmed |
| 2026-11-02 | Use open-weight models for experiments; a frontier API for the quiz writer and attacker is optional | Claude's recommendation, API choice still open | Pinned weights keep the reader fixed. A weak attacker understates gaming |
| 2026-11-02 | Run the first experiments without human baselines | Jason | Get initial signal on other properties first |
| 2026-11-02 | The goal is fixed in `docs/goals.md`; experiment details may change | Jason | Details are expected to change once implemented |

## Claude Code facts behind the setup

From the docs at code.claude.com, read through a subagent on 2026-11-02. Not tested.

- `claude -p "..." --output-format json` returns `session_id`. `--max-budget-usd` caps spend.
- A `-p` run waits for its background commands, subagents and Monitor watches.
- Without `--bare`, `claude -p` uses the subscription login. `--bare` needs `ANTHROPIC_API_KEY`.
- `--dangerously-skip-permissions` equals `bypassPermissions`. It is refused as root and meant for an isolated VM or container only.
- The docs have no recipe for running `claude -p` from a shell loop, cron or systemd.

## 2026-11-03 09:10 UTC: exploratory per-section diagnostic (pass 8)

Added after seeing 2 of 6 pilot papers, so it is exploratory. For each paper and reader, `analyze_pilot.py` now also reports the score on kept items tagged to the obfuscated section, for clear, obfuscated and removed. Reason: the preregistered whole-quiz measures mix affected and unaffected items, so they cannot show whether a null comes from readers reading through the obfuscation or from the obfuscation hitting few items. The removed version is the positive control for each paper. No preregistered measure or bar changed.

## 2026-11-03, pass 17: compression repair step and unsigned number check

Two of the first three core papers failed compression. One failure was a defect in the number check (signs and `_` italics; `docs/run-log.md` pass 17); the check now compares unsigned numbers. The other lost a few real numbers in every whole-paper try. Rather than drop the paper or relax "no number lost", a repair call now reinserts the missing numbers into the best try (fewest missing, same sections), at most twice. Why: whole-paper redos lose different numbers each time, so more redos would not converge; a targeted edit keeps the length near the compressed try. Decided before any takes on the affected papers.

## 2026-11-04, pass 17: 9 core papers, not 12 (budget)

The first 3 core papers cost $73 of account usage ($175.28 → $248.27, including the compression fix and repair), about $24 per paper. $191 remains before the $439 stop line. 9 more papers would cost about $216 and 7 about $170 (only $20 margin, and unbalanced strata). Following plan v2 ("cut papers, keeping strata balanced, never questions per paper"), run 6 more: the next 2 per stratum in the drawn order of `data/selection/selection.json` (A: 2607.17117v2, 2609.34091v1; B: 2607.28685v1, 2609.30094v2; C: 2608.19216v1, 2608.30724v1). Cut the last of each stratum: 2608.13754v1, 2607.10112v1, 2609.02893v1. The rule is fixed by draw order, not by any result. Expected total about $392, leaving about $46 for reruns of defects.

**Superseded about 1 minute later:** Jason's note "Budget raised to $500" (stop line $639) arrived in the same pass. The 6-paper batch was stopped after a minute (its first calls are cached and reused) and relaunched with all 9 remaining core papers and `--stop-usage 639` (`logs/core2.log`). All 12 core papers will run; expected total usage about $464.

## 2026-11-04, pass 18: citation and thousands rules in the number check; writer keeps batches before a content-filter refusal

- **Number check:** a citation is a bracket with a capitalised name and a year 19xx/20xx (round or square), or `Name (year)` / `Name [year]`; LaTeX `{,}` is a thousands comma. Why: the old rule blanked any parenthesis with 4 digits (hiding real numbers such as 1060.00) and missed square-bracket citations, which failed 3 papers' compression on formatting alone (`docs/run-log.md` pass 18).
- **Writer:** when the provider's content filter refuses a batch, keep the earlier batches and stop, rather than failing the paper. Why: one core paper is about LLM safety; replacing it would change the preregistered paper set, and a smaller quiz is still informative if its size is reported. Its result will be flagged as below the 40-question target.

## 2026-11-04, pass 21: hidden-gap follow-up implementation

Implements Jason's follow-up (`docs/plan-v2.md`, last section) in `scripts/quiz/gap_v2.py`, prompts p9 (`GAP_INSIGHT`, `GAP_REMOVE`, `GAP_IMPLICIT`). Predictions were committed first (`docs/preregistration-v2.md`, commit fc3d893).

- **Two steps:** one Opus 5.5 call records the insight, dependent results, key definitions and reasoning steps (JSON), and both versions are written from the paper plus that note, so they target the same insight. The insight goes in `versions/gap-meta.json`, not `meta.json`, so the plan v2 version record stays as it was.
- **Checks, up to 3 tries per version:** same section headings, 0.8x to 1.25x the original's length, and for `imp` no lost number. A version failing all 3 is excluded (as preregistered). `rem`'s lost numbers are recorded only; losing results is its purpose.
- **Takes go to a separate `takes-gap.jsonl`** with conditions `rem` and `imp` only, on the same sittings (the split depends on reader and sample, not condition). Rewriting `takes-v2.jsonl` would replay every plan v2 condition and could make new calls for earlier failures, changing reported results. `takes_v2.take` and `question_scores` gained optional `conds`/`files` arguments; plan v2 output is unchanged (checked: re-running `summary` leaves `takes-summary.json` identical).
- **Judge:** `judge_v2.rate_one` takes the prompt variant as an argument, so both prompts run in one process; originals' ratings are reused from the plan v2 judge files.

## 2026-11-04: exploratory human comparison on OneStopQA (Jason's request)

Jason asked whether the blog's claim that human readers show our pattern (readability ratings fall, comprehension barely moves) holds on data with real human answers, and to plot it even if it refutes the claim. This is **exploratory**; the plan below was written before any model result.

- **Data:** OneStopQA ([berzak/onestop-qa](https://github.com/berzak/onestop-qa) commit 7a05875, CC BY-SA 4.0). `human_experiments/prolific/qa.tsv`: 324 multiple-choice questions, each shown with the Advanced or the Elementary version of its Guardian paragraph, 3 Prolific participants per item (about 970 answers per level); `guess.tsv` is the same questions without the paragraph. The correct option is index 0.
- **Readers:** GPT-5.6 Sol (1 sample), Gemma 4 31B, Ministral 8B, Llama 3.2 3B (3 samples), same providers as plan v2, temperature 0.7, options shuffled per item and sample. Conditions: Advanced paragraph, Elementary paragraph, no paragraph. Reply is one letter; anything else counts as wrong.
- **Ratings:** each reader rates each paragraph version 1 to 10 for an adult reading carefully (prompt adapted from `JUDGE_HUMAN_V2`), same samples as above.
- **Measures:** accuracy per reader and level, with 95% bootstrap intervals clustered by question for the Advanced minus Elementary difference; humans the same way. Mean rating per level.
- **Caveat fixed in advance:** Advanced vs Elementary is a mild, real-world simplification, much milder than our obfuscation, and there are no human readability ratings, so this tests only the comprehension half of the claim directly.
- Spend cap $30. Code `scripts/onestop/run_onestop.py`, output `data/onestop/`.

## 2026-11-04: public release

Jason approved publishing a cleaned copy as https://github.com/AverroesLabs/legibility-quiz-pilot (public), with the blog post as README. Built from this repo's tracked files minus agent and ops files (CLAUDE.md, INBOX.md, STATE.md, STOP, NOTIFY_JASON.md, loop, watchdog and pod scripts, .github), internal docs (docs/background, style guide, pod setup, watchdog log, LW-post, pitch-doc), with the private doc links in goals.md and the shared-account rows in decisions.md removed. Added MIT (code), CC BY-SA 4.0 (text and data) and per-paper attribution. The blog links to the repo.

## 2026-11-12: new branch v4/tamper-templates from V1

Jason judged the Codex-run v3 (branch `v3/openai-overnight-20261111`) a failure: it spent $2,076 of OpenAI usage, $2,039 of it on gpt-6.1-sol for error construction, validation and agent work, and under $5 on weak readers, and it produced no completed comparison under its final protocol. This branch restarts from tag `V1` and ports only data checked by hand:

- `data/papers/jmlr-2014-dropout/` (JMLR 2014 Dropout paper; PDF and text match the hashes in its `meta.json`; pdftotext layout with page headers, tables readable).
- `data/selection/v3-expanded-13.json`: the 12 core papers plus Dropout. The 12 core texts are byte-identical to V1 and match the manifest hashes.

No v3 code, protocols, prompts or results are carried over. Experiments now use OpenAI models only, because that is the API access Jason has; the key goes in `.env` as `OPENAI_API_KEY`.

## 2026-11-12, pod pass 1: ledger prices for the cheap readers

`llm.py` charged models missing from `PRICES` at the Sol rate ($10/$20 per M tokens). That overstated a gpt-5-nano call about 100 times, and the reader grid of experiment 3 would have spent the $500 ledger on paper. Added gpt-5-nano at $0.10/$0.80 and gpt-4.1-mini at $0.80/$3.20, twice OpenAI's published list prices, because v3's gateway charged gpt-5-mini at twice list. These still err high. gpt-5.6-luna stays at the fallback rate because I have no reliable price for it, so its ledger cost is an overestimate. Jason sees the real figure on the OpenAI usage page; a real luna price would make the ledger more accurate.

## 2026-11-12, pod pass 2: contradiction inventory before planting errors

Experiment 0 showed that V1's `obf` and `compobf` copies, and the original of 2609.02893v1, contain real internal contradictions (rewrite defects, and one real appendix-table mismatch). The design assumes the copies differ only in legibility and contain no errors except planted ones. So before experiment 2, gpt-6.1-sol lists the contradictions already present (plan v4 step 0b). I chose to avoid the listed passages rather than repair V1's texts: repairs would change fixed texts that earlier results (V1) were measured on, and every repair would itself need checking. If a copy turns out to have so many defects that little text is free for planting, repair becomes the fallback and will be recorded here. Sol is used because the inventory sets what counts as a planted-error catch, so its accuracy matters more than its cost (about $0.3 per copy).

## 2026-11-12: experiment 2 construction choices

- One planted error per copy. Each error is a list of exact text edits, so a planted copy is rebuilt from `data/v4/exp2-errors/errors.jsonl` with `planted_text()` and is not stored.
- Grid types as written into the design prompt: "carried" errors are two consistent edits (a wrong intermediate statement and a later claim built on it) against unchanged data; "omitted" errors drop a caveat or exception from a summary while the body keeps it.
- The design call sees the clarified copy and `compobf`, so that every error's claim and evidence exist in the most compressed copy.
- An error survives only if it is planted in all four copies, the strong reader (gpt-6.1-sol, medium, 1 sample) catches it in all four, and the pair check finds it planted equally. The pair check compares each copy with its own text before planting, so differences in detail that the copies already had (the clarified copy's evidence placed next to claims) are allowed; detail added or removed by the planting is not.

## 2026-11-12: after experiment 3, reading the grid

Read with the key in `calibration-table.md`. The false-accusation gate passes (at most 9% per cell). No cell meets the proposed bars: the stronger rows are "low clarified, lower illegible", between the floor and the window, and the illegible penalty (about 0.15 pooled, mostly `compobf`) is much smaller than predicted. The key says to move toward stronger readers or larger budgets when the clarified rate is low. Next moves, in order:

1. **Stronger reader settings** on the same 19 errors: gpt-5.6-luna medium and gpt-5-mini high (2 samples, 4 copies; about $25 to $35 at the ledger's luna fallback rate). Drop gpt-5-nano and gpt-4.1-mini, which are on the floor. This tests whether a setting exists with clarified at least 0.7 that still loses catches on the illegible copies.
2. **Pointed check** on the missed clarified-copy errors for the best reader (told the section and claim): it tells finding from judging, and so whether lower clarified rates are a search problem (fixable by budget) or bad errors.
3. **More errors** before trusting any gap: 19 errors give intervals about ±0.2. New errors on the remaining development papers, with the same construction checks, once a promising setting exists.
Exploratory results do not change these choices; the strong-reader gate question stays with Jason.

## 2026-11-12: repair V1's illegible copies (Jason's note, ~18:00 UTC)

Jason's INBOX note replaces the pod pass 2 choice ("avoid, don't repair") for `obf` and `compobf`. Defects stay in V1's texts, which are left untouched. Repaired copies are saved as `data/v4/repaired/<paper>.<copy>.md`, with each patch's diff and reason. A gpt-6.1-sol consistency check against the original is rerun until it lists nothing. Errors in the original stay and are still avoided. Then only the work that touches repaired copies is rerun: re-planting, checks, clean reviews and the experiment 3 readers on `obf`/`compobf`. The two grids are reported side by side. This comes before the stronger-reader step above, because defects in one copy only could bias that copy's catch rate.

## 2026-11-12, pod pass 6: how the repair check is run

- One call per round both lists defects and writes their patches (`find`/`replace`, an exact unique substring of the copy), which halves the cost. A patch whose `find` is not unique is not applied and is logged as unapplied. The loop stops when a check lists nothing, after 4 rounds, or when no listed patch applies.
- The checker is told that consistent code names (the glossary), harder wording, omitted detail in `compobf`, and inconsistencies the original already has are not defects. That way the repair restores fidelity to the original without "fixing" errors that every copy shares.
- Clarified copies get one round, report only. Jason's note says they already passed an equivalent check, so I do not change them. The one difference found (2608.13754v1, "all levels" against "the levels of each ... each with a citation") is outside every planted error.
- Next, re-planting into the repaired copies: re-run `exp2_errors`' plant, strong-reader and pair checks with `source()` pointing at the repaired `obf`/`compobf`. The clarified and original planted copies stay as they are, and the 19 surviving errors (plus 3 exploratory) are the set. Then run experiment 3's readers and clean reviews on the repaired planted copies only.

## 2026-11-12, pod pass 7: experiment 3r reuses V1's planted edits where they still apply

Re-planting every error from scratch would change the planted wording too, mixing re-planting noise into the V1-versus-repaired comparison. So V1's edits are reused wherever every old text still occurs once in the repaired copy and every evidence quote is still found (43 of 44 pairs); only the rest are re-planted with experiment 2's plant(). The strong-reader and pair checks are rerun on all of them, as Jason's note asks. Result: the repairs moved catch rates by less than 0.1 (no measurable distortion), so the next step returns to the stronger-reader experiment (experiment 4), on the repaired copies, since those are now the better texts.

## 2026-11-12, pod pass 8: after experiment 4

Stronger settings raised catch rates on the illegible copies as much as or more than on the clarified copy, so the clarified-minus-illegible gap shrank (luna 0.21 → 0.10, gpt-5-mini 0.08 → 0.03). Only gpt-5-mini high reaches clarified 0.70, with illegible 0.68. Per the preregistered reading, a stronger reader is not the way into the window. Next, in order:

1. **Pointed check** (item 2 above) for mini-high and luna-low: for each clarified-copy miss, tell the reader the section and claim and ask if it is consistent. Cheap (about $2). It separates finding from judging.
2. **Harder illegible copies.** The current copies cost a strong reader little. Candidates: stronger compression, definitions removed or moved far from use, evidence split across sections. Each must pass the same consistency check against the original. This changes the design in `calibration-table.md`, so I will write the proposal here and in the report before building it, and flag it to Jason as not blocking.
3. More errors on new papers only once some setting shows a gap whose interval excludes 0 at clarified ≥ 0.7.

## 2026-11-12, pod pass 9: after experiment 5, proposal for search-hard illegible copies

Experiment 5 measured that the readers judge planted errors correctly when pointed at them, in every copy (0.91 to 1.00), and rarely accuse the unplanted passage (1 of 34). So unpointed misses are failures to find, and the illegible copies cost readers search, not judgment. The current illegible copies change wording (codes, compression) but leave each claim near its evidence. Proposal, changing the illegible side of `calibration-table.md`:

- **New illegible copy `scatter`**, made from the original by gpt-6.1-sol: the same facts, with the content an error would be checked against moved away from the claims it supports. Numbers are reported once, in tables or an appendix, rather than restated next to claims. Definitions and caveats are stated once, far from their use. Cross-references ("Table 5", "Section 7.4") are removed from the claims. Ordinary wording, no codes, so any effect comes from search and not vocabulary.
- **Checks, as for the repaired copies:** every number of the original present (exp1's number check), and the gpt-6.1-sol consistency check against the original rounds until clean. Length within 0.9 to 1.1 times the original, so that length does not drive the effect.
- **Errors:** re-plant the 17 errors into `scatter` with experiment 2's plant prompt, the strong reader and the pair check, as in 3r. Then run luna-low and mini-high (2 samples) with clean reviews, plus the pointed check on `scatter` to confirm that judging is still intact.
- **Reading:** the approach is promising if a setting with clarified at least 0.7 drops on `scatter` by a gap whose interval excludes 0. If even `scatter` costs mini-high little, the conclusion is that current readers search well enough that illegibility of this kind is not measurable with one planted error per paper.
- Estimated cost about $25 (rewrite and checks about $6, re-planting and checks about $5, readers about $12, pointed about $2). Not blocking: flagged to Jason in `STATE.md`. Unless he objects, the next pass preregisters and builds it on one paper first.
- Also: add the 2609.02893v1 sentence "Interpretations of the greedy-selected PCs describe ... survives the move to other domains" to the avoid list for new errors (report weird result 14).

## 2026-11-12, pod pass 10: after experiment 6 (`scatter`)

Measured: `scatter` (evidence moved away from claims, plain wording, consistency-checked) costs neither reader any catches. On the 13 errors that survived planting, the catch rate is 0.62 for both readers, against 0.54 (luna-low) and 0.69 (mini-high) on clarified, with pointed rates of 0.92. By the preregistered reading, layout illegibility of this kind is not measurable with these readers and one planted error per paper. For luna-low, jargon and compression cost more than layout (0.38 vs 0.62).

Decision: the last cheap test within the design is **several planted errors per copy**, scoring the share found. With one error per paper, each review is a single near-coin-flip outcome. With several (for example 3 errors from different types, planted together in one copy, from the existing surviving errors where their passages do not overlap), each review gives a graded count, and readers that stop searching after one finding would show it. Estimated about $20. The next pass preregisters it (which errors are combined, the measure, the reading) and runs it on the existing clarified, repaired `obf`/`compobf` and `scatter` copies. If it shows no gap with an interval excluding 0 for a setting at clarified ≥ 0.7, the pilot is written up as a negative result on this design and flagged ready to Jason.

Not chosen without Jason: a limited-view reader (section by section with a short memory). It would make search the bottleneck, but it changes what the audit measures, so it is a next step in the report for him to decide.

## 2026-11-12, pod pass 11: after experiment 7 (several errors per copy)

Measured: with 3 to 5 errors planted together, neither reader reaches 0.7 on clarified (mini-high 0.65, luna-low 0.34), and no clarified-minus-illegible or clarified-minus-`scatter` gap has an interval excluding 0. luna-low is reversed (clarified lowest), linked to reporting fewer errors on clarified.

Decision: by the preregistered reading (plan v4, experiment 7) and `STATE.md` next step 2, the pilot is written up as a negative result on this design. The next pass finalises the report (bottom line rewritten as one negative answer, the open promise to classify other accusations either done cheaply or dropped), adds "report ready" to Waiting on Jason, commits `NOTIFY_JASON.md` and creates `STOP`. No further experiments are started without Jason: a limited-view reader is the main idea for a continuation, and it changes what the audit measures.

## 2026-11-12, pod pass 12: report finalised as a negative result

The bottom line of `docs/report-v4.md` is now one plain negative answer, with a summary figure (`docs/figures/v4-summary.png`, drawn by `scripts/v4/summary_figure.py` from each experiment's own `measures()`, no API calls). Results are reordered by experiment. Two earlier promises were dropped rather than done, because neither can change the bottom line, which rests on planted-error catch rates: classifying accusations outside the planted error and the inventory, and a second Sol sample on the copies where the strong reader missed an error only in `compobf`. Both are stated in the report. No new experiments: the continuation (a limited-view reader) changes what the audit measures and waits for Jason. The loop is stopped with a `STOP` file.

## 2026-11-12, pod pass 13: experiment 8 design (limited-view reader), on Jason's instruction

Jason approved the limited-view reader (`INBOX.md`, ~22:30 UTC). Choices made in the preregistration (plan v4, experiment 8), before any data:

- **Sections are level-1 and level-2 headings.** The clarified and original copies keep 26 to 45 headings at levels 3 to 6, while the jargon copies keep only level 2. Splitting at every heading would give the clarified copy about 45 chunks and `compobf` 11, so the chunk count would track heading depth rather than length. Splitting at level 2 gives 9 to 14 chunks per copy for every copy, and the length difference shows up as chunk size.
- **Variants:** A (section chunks ≤ 800 words, notes ≤ 300), B (notes ≤ 150), C (paragraph chunks ≤ 250 words, notes ≤ 300). B and C bracket A in the two directions Jason named: less memory, and smaller views (more notes rewrites).
- **Readers by cost:** luna-low only on A, because the ledger charges luna at the Sol fallback rate (about $72 for A alone). mini-high runs all three variants, so variants are compared on the same reader. mini-med on A last. This order is fixed in advance and cut from the end only for cost.
- **Primary set excludes `omitted`** (12 errors), as Jason asked; the 5 `omitted` errors are reported separately.

## 2026-11-13 (pass 19): ledger repriced to OpenAI list prices

Jason replaced `PRICES` in `llm.py` with OpenAI's published list prices on 2026-11-13 (source and date in the code). **Every ledger total before this date used inflated estimates** (v3's gateway rates, and the Sol fallback rate for gpt-5.6-luna), and so did every per-experiment cost quoted in the report and STATE before it. As Jason asked, no row was rewritten. One `kind: "reprice"` row per model was appended to `data/v4/spend.jsonl` at 03:1x UTC, with no job running. Each row's `usd` is the cost of every `settle` row's `usage` at the new prices (`llm._cost`, cached-input discount included) minus the logged `actual`:

| model | settled calls | logged | at list prices | reprice row |
|---|---|---|---|---|
| gpt-5-mini | 13,330 | $157.20 | $77.68 | −$79.52 |
| gpt-5.6-luna | 2,552 | $111.86 | $3.61 | −$108.25 |
| gpt-6.1-sol | 864 | $34.28 | $9.59 | −$24.68 |
| gpt-4.1-mini | 316 | $1.73 | $0.85 | −$0.89 |
| gpt-5-nano | 316 | $0.45 | $0.17 | −$0.28 |
| **total** | 17,378 | $305.52 | $91.90 | −$213.62 |

Every settle row has usage. One reservation never settled (`8bf40542a4fd3dba`, exp8 C, 01:18 UTC 13 Nov, $0.07), most likely a request lost during the pass 18 stash incident. It stays charged at its logged value. Ledger total: $305.59 before, **$91.96 after**. Rule for later: a job that started before a price change keeps the old `PRICES` until it restarts, and its rows get the same kind of correction afterwards.

## 2026-11-13 (pass 20): Batch API helper in `llm.py`

Jason's 03:30 note asked for the Batch API on large runs that can wait, through the ledger and cap, as one helper and not a framework. Design: `Calls.submit_batch` and `Calls.collect_batch`, about 100 lines in `llm.py`. Batch results go into the same `calls.jsonl` under the same keys as synchronous calls, so an experiment script can submit its large stage as a batch and then run its normal synchronous loop. That loop finds everything in cache, and anything the batch dropped is redone synchronously at full price. One reservation covers a whole batch, settled once from the summed usage at half rate. `collect_batch` must run from one process at a time, because the `done` record in `batches.jsonl` is what prevents a second settlement. Following the 03:05 note, batches are used only off the critical path. Tiny test batches took 1.5 to 7.3 minutes, but the API promises only 24 hours.

## 2026-11-13 (pass 21): limited-view reader resumes planted copies from saved clean notes

Jason's 03:30 note, item 3. A planted copy is the same text as its clean version up to the chunk the edit changes, and the reader is shown the same prompts ("part k of n", the same notes) on that prefix, so its state before that chunk has the same distribution as in a clean read. `scripts/v4/limited_reader.py` therefore reads each clean version once per sample and starts every planted copy at its first changed chunk with the saved notes. This is a change of procedure, not of reader: prompts, chunking, notes cap and parsing are experiment 8's. Consequences for experiment 9's preregistration: errors in the same version and sample share the clean prefix (resample by error in the bootstrap, as before); a review includes the clean read's findings before the planted chunk; copies whose edit changes the chunk count are read from the start. The check against experiment 8 is in `run-log.md` (pass 21): same grade in 67 of 68 copies, 63% of steps called.

## 2026-11-13 (pass 22): experiment 9 plan and anchor rule

The plan is in `plan-v4.md` ("Experiment 9 plan") and the anchor rule in `paper-selection.md` ("Reviewed anchors"). Both were committed before any experiment 9 output and before any anchor candidate was fetched. These are the choices Jason's notes left open:

- **`clar-<prompt>-obf` is not repaired against the original.** A real audit gets only the possibly illegible paper. Patching the clarifier's output from `orig` would hand it information the audit never has, and would inflate the `obf-max` row's denominator. Its differences from `orig` are measured and reported, and planted errors avoid those passages. This is the "avoid" approach of pass 2. Jason replaced it with repair for V1's copies (18:00 note), because those defects came from our own construction. Here the defects are part of what is measured. Experiment 3r found that repairs moved catch rates by less than 0.1, so avoiding them should cost little. `clar-<prompt>-orig` and `obf-max` are repaired, since both are constructions we control.
- **Prompt pass rule.** `clar-orig` must come clean within 4 repair rounds. `clar-obf` may have at most 1 changed or added item per 1,000 words of `orig`. Lost content does not fail a prompt, because recovering less from `obf-max` is the effect being measured, and an error whose evidence was lost fails the pair check anyway. The threshold is a judgment call fixed before data, chosen to drop only clarifiers that invent or alter content.
- **At most 4 repair rounds, one regeneration.** `repair.py` needed up to 3 rounds per copy in pass 6. A cap keeps a bad generation from spending without end.
- **Reading grades are computed in-script** (Flesch-Kincaid with a vowel-group syllable count, and Coleman-Liau, which needs no syllables). The pod's system Python refuses `pip install`, and these grades only need to compare versions with each other.
- **Anchors come from ICLR 2026 and TMLR.** Both have public OpenReview reviews. TMLR has no page limit and so can fill the 10,000 to 15,000 word band, which ICLR papers (10 pages) mostly cannot. Using the arXiv HTML reuses V1's extraction and checks. PDF extraction is not available on the pod (no `pdftotext`, `pypdf` or `fitz`).
- **Order:** the held-out papers' clean versions come after the calibration error stage has started, as a Batch job off the critical path. Nothing in calibration depends on them, and their checks are read only for pass or fail.

## 2026-11-13 (pass 23): anchor rule v2, because OpenReview's API blocks anonymous queries

Rule v1 (pass 22) drew from OpenReview's API. Pass 22 had only checked that the `groups` endpoint answers. On running, every `notes` query (API v1 and v2, ICLR 2026 and TMLR) returns HTTP 403 `ChallengeRequiredError`, a bot check. Passing it would mean getting around an access control, so I did not try. Rule v2 (`docs/paper-selection.md`) takes the same two venues from their public static lists of accepted papers (jmlr.org/tmlr/papers, iclr.cc/virtual/2026/papers.html), matches the area on titles only, and orders by a hash of the normalized title, because the ICLR list has no forum ids. All other screens, bands and places are unchanged. Written and committed before any candidate's text or arXiv record was fetched.

## 2026-11-13 (pass 24): `obf-max` details the plan left open

Fixed in `scripts/v4/obf_max.py` (prompt `obf-max-v1`) before any `obf-max` output was seen.

- **One whole-paper call, the model picks its own codes.** V1's `obf` rewrote section by section with a glossary from a separate call. A single call lets evidence and definitions move across sections, which per-section rewriting cannot do. The rewriter returns its glossary after a `===GLOSSARY===` line, and the glossary is passed to the check so intended codes are not reported as defects.
- **Length target 60 to 80 percent of `orig`, never at the cost of a fact.** "Compression" with every fact kept can only remove redundancy and wording. The target is in the prompt, not enforced, and the achieved ratio is in the manifest.
- **Missing facts count as defects.** `repair.py`'s check treats omitted detail as allowed; `obf-max` must keep every fact (Jason's 01:40 note), so the check uses `exp6_scatter.py`'s variant: omissions are defects, and numbers missing by the mechanical check are listed for the checker to confirm or dismiss as format changes.
- **Reading grades** strip markdown tables, display and inline math, headings and code before counting. Coleman-Liau counts letters, so opaque codes (short "words") lower it while Flesch-Kincaid rises with longer sentences: on V1's 2609.30094v2, `obf` grades FK 32.1 and CL 13.0 against `orig` FK 15.8 and CL 16.4. Report both and read them only as comparisons between versions.

## 2026-11-13, pass 25: clarification prompt set (experiment 9, step 4)

From `docs/clarification-approaches.md`: five prompts, `base` (unchanged), `plain` (target Flesch-Kincaid grade 12), `cohesion` (repair inference calls, 2× length cap), `explainer` (summary, glossary, section gists, claim-to-evidence maps) and `twostage` (extract the argument, then write). All keep `base`'s fidelity block verbatim, including "keep the inconsistency as it is", because a prompt that spells out reasoning could otherwise smooth over planted errors later. Chosen before any output; the texts go into the script before the first generation.

## Pass 26: clarification script choices (before any clarification was generated)

- The five prompt texts are in `scripts/v4/clarify.py`, committed before the first generation. `base` is experiment 1's `CLARIFY_USER` byte for byte (asserted in the script); the others replace only its Goals block (`cohesion` also sets the length cap to 2×; `twostage` appends the extraction to the user message). Generation uses gpt-6.1-sol at **high** effort, as the plan says, whereas experiment 1 used medium; so `clar-base-orig` is not a regeneration of experiment 1's copy.
- The length rule says "times the original"; for `clar-*-obf` the clarifier's "original" is `obf-max`, so its cap is relative to `obf-max`. Kept, since the audit would only have that text.
- **Clean** means the fidelity check (experiment 1's prompt, unchanged) lists no issues, minor ones included, matching experiment 1's copies (all three had zero issues).
- The fidelity check gives quotes, not patches, so `clar-*-orig` repair adds one patch call per round (gpt-6.1-sol medium, `PATCH_USER`) that turns the listed issues into find/replace patches; then the check is rerun. At most 4 fidelity checks per copy.
- For the pass rule on `clar-*-obf`, "fixed" issues (an error of the original silently corrected) count with "changed" and "added", because removing an inconsistency is a change to content.

## 2026-11-13 (pass 27): error-stage preregistration choices

Committed in `plan-v4.md` ("Experiment 9 error stage preregistration") before any error design or weak-reader data. Choices the notes left open:

- **Variant A, not C, for the limited-view readers.** Gaps were a near-tie (0.17 vs 0.19, intervals almost identical), C falsely accused in 19% of clean passes against 11% for A, and A costs less. Experiment 8's own rule sends near-ties to the cheaper option.
- **No staging.** At list prices the whole ladder on the 4 calibration papers is estimated at about $170 without gpt-5.5 (about $550 with it), well within the $1,895 left, so the 03:30 note's staging (which applies only when the budget does not fit) is not needed. With one carried prompt there is nothing to choose between anyway. Jason's own estimate ($850) used the old, inflated prices.
- **49 errors per paper (7 categories × 7: 2 easy, 3 medium, 2 hard)**, the nearest balanced count to "about 50".
- **Pointed question extended to "consistent with, and supported by"**, so the `unsupported` category can be caught when pointed; the same question is used for every category and the control.
- **Proposed bars** follow the earlier calibration key (clarified high, illegible low, low false accusations), restated as audit scores: `orig` row ≥ 0.85, `obf-max` ≤ 0.60, separation ≥ 0.25. They are proposals; Jason approves final bars.
- **Category exclusion in the frozen audit** only by a fixed rule (pooled `orig` row under 0.60), not by inspecting per-category separations, to limit overfitting on 4 papers.

## 2026-11-13 (pod pass 33): experiment 9 selection frozen: gpt55-A, pointed score

Applied the preregistered selection rule (`plan-v4.md`, experiment 9, "Proposed bars and selection rule") to the 5 readers' primary-set results. No reader × score meets the bars; gpt55-A is "close" on both scores (clarified-copy separation 0.22 [0.04, 0.40], pointed 0.24 [0.09, 0.41]); all others are "no". Largest separation: pointed 0.24; the clarified-copy score is within 0.05 with the same reader, so the near-tie goes to the pointed score. Frozen: **gpt-5.5 low, variant A limited view, pointed audit score, categories calculation, carried, causal, reversed, stretched** (none has a pooled `orig` row under 0.60). Next by the plan: anchor addendum, then the 9 held-out papers once. Not started, because Jason's 2026-11-13 03:05 note allows gpt-5.5 only on the 4 calibration papers; asked in STATE and by `NOTIFY_JASON.md`.

## 2026-11-13 (pod pass 34): anchor addendum choices (before any anchor data)

The addendum is in `plan-v4.md` ("Experiment 9 anchor addendum"). It was written while blocked on Jason's gpt-5.5 answer, so that the length rule is fixed before any anchor result exists.

- **Construction unchanged (49 errors, all categories)** although only 5 categories enter the frozen audit: anchor errors are then built exactly like the calibration papers' errors, and the extra Sol cost (a few dollars per anchor) is small next to comparability.
- **Frozen reader on `orig` and `obf-max` and primary survivors only:** the frozen score is the pointed score on those categories, so other reads would not enter any decision. This about halves the gpt-5.5 cost per anchor.
- **Dropout out of the length test:** its result was seen before the rule was written. It is shown in the figure, marked.
- **Rule (a) uses the "close" `orig` bar (0.75)** rather than "meets" (0.85), because the frozen audit itself only reached "close", and with about 8 to 11 primary errors per anchor one miss moves a row by 0.1.
- **Rule (b) is a band difference, not a slope test:** with 2 anchors per band a slope across 6 points adds nothing a band difference does not, and the band difference is easy to read. The slope is reported.
- **Ladder only for a length failure, and only the notes budget:** Jason's note names the memory budget. Doubling twice (600, 1,200) spans "a bit more memory" to "about a section remembered". Each step must keep the original 4 papers at "close", because more memory may also help the reader of the illegible copy and erase the separation. Chunk size is not changed: experiment 8's variant C showed chunking changes false accusations.
- **If no step passes, the held-out run still goes ahead with variant A:** the length failure is then a reported limit of the audit, and the held-out result is still the preregistered test.

## 2026-11-13 (pod pass 35): anchor `orig` inventory

The error stage's avoid list (`plan-v4.md`, "Avoid") includes the inventory of contradictions in `orig`; for Dropout, which had none, a new inventory was run first with the same `inventory.py` prompt. The anchors are treated the same way (one `err9.py inventory <paper>` call each, about $0.5), so that anchor errors are built exactly like the calibration papers' errors, as the addendum requires.

## 2026-11-13 (pod pass 35): held-out run preregistered; `explainer` only on held-out papers

Written before any held-out data, after Jason approved gpt-5.5 (16:15 UTC note). The plan said the held-out papers "get every clean version". Only `obf-max` and the `explainer` prompt are now made for them: the frozen pointed audit reads only `orig` and `obf-max`, the construction gate needs only the `clar-explainer` copies, and the other prompts' pass or fail on held-out papers would not enter any decision (the selection is already frozen). This saves about $25 and keeps the held-out run to what the frozen audit uses. The bars are the calibration bars, unchanged, so the held-out verdict is read on the same scale as the calibration verdict.

## 2026-11-13 (pod pass 43): held-out run paused; experiment 11 is the core (Jason, INBOX 2026-11-14 notes)

- **Held-out run paused** before any held-out data. The frozen gpt55-A selection was made for the section-by-section reader, now a side experiment. What the held-out run tests will be re-chosen after experiments 11 and 12, by the rule in `plan-v4.md` (experiment 11 preregistration, "Feeds the held-out choice"). `heldout.sh` is not run.
- **Ladder A1200 stopped and not restarted**, no verdict (`run-log.md`, pass 43). Experiment 10 and the `unsupported` second prompt are not started.
- **Experiment 11 choices:** 2 unaided samples per error and version for every reader (Jason's "2 samples"; the existing whole-mini-high rows had 1, so its sample 0 is reused and a sample 1 added under a new name, leaving experiment 9's rows untouched); 1 pointed check and control per error and version, as in experiment 9; primary categories only, as in the frozen runs; whole-paper prompt, grader and 32,000-token allowance unchanged. "Positive difference on the anchors" in Jason's step 5 is read as a pooled point difference > 0, the literal reading; its interval is reported beside it.
