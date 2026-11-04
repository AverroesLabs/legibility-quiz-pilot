# Pilot paper selection rule (v2)

**v2, 2026-11-03.** v1 had a defect found on checking its draw: the arXiv API does not match `abs:` phrases strictly, so only 156 of 656 fetched stratum C abstracts contained a stratum term, and the v1 core for C held papers such as one on river discharge. v2 keeps the v1 fetched frame, seed and screens, and adds one step: keep a paper only if its abstract literally contains a stratum term (ignoring case and hyphens), and assign the stratum from that literal match in the same order. The v1 draw is kept in `data/selection/v1-selection.json` and explained in `docs/run-log.md`. The v2 rule was committed after seeing the v1 draw, but nobody chooses papers under it: the order is still the same hash.

The text below is the v1 rule, which v2 otherwise leaves unchanged.

Committed before any candidate list was fetched. Before this commit I had seen only frame counts and the three newest arXiv API results for a generic test query, which I did not use. Any change after the draw is exploratory or gets a new rule version, with the reason in `docs/decisions.md`.

The draw is done by `scripts/select_papers.py`. Its outputs are in `data/selection/`, and the selected papers are in `data/papers/`.

## What is selected

- **Core:** 12 papers, 4 from each topic stratum.
- **Reserves:** 2 per stratum, used in draw order if a core paper later proves unusable. Record any swap and its reason.
- **Chains:** up to 3 two-paper chains (A then B), separate from core and reserves.

## Frame

arXiv papers meeting all of these:

- First version submitted from 2026-07-01 to 2026-09-30 UTC.
- Primary category cs.AI, cs.LG, cs.CL or cs.CR.
- Abstract matches a stratum query. A paper matching several goes in the first it matches, in this order:
  - **A, interpretability:** "mechanistic interpretability", "sparse autoencoder"
  - **B, attacks:** jailbreak, "red teaming", "prompt injection"
  - **C, alignment and oversight:** "AI safety", "AI alignment", "AI control", "scalable oversight", scheming, sandbagging, "reward hacking"

The frame was sized on 2026-11-02 at about 198, 408 and 946 papers before primary-category filtering and overlap removal (A, B, C).

## Draw

Papers are taken in ascending order of `sha256("20261102:" + arXiv id without version)`. Each is screened in that order. A passing paper fills its stratum's core quota, then its reserve quota, then joins the chain pool. No person or model chooses among passing papers.

The draw pins the latest version the API reported when the frame was fetched.

## Screens

A paper must pass all of these, or it is logged with the first failed reason:

1. **Licence:** CC BY, CC BY-SA or CC0, read from the arXiv abstract page. The pilot makes rewritten versions of each paper, and these licences let us share those if the work is published. arXiv's default licence and the NC and ND variants do not.
2. **Type:** title or abstract does not mark it as a survey, review, tutorial, position paper, perspective, overview or workshop report. The quiz needs a paper's own claims. It is also not marked withdrawn.
3. **Independence:** its arXiv id does not appear in `docs/background/`, since those papers shaped the design.
4. **Format:** arXiv native HTML exists for the pinned version, with at most 5 conversion errors (`ltx_ERROR`).
5. **Length:** 3,000 to 15,000 words, excluding references. Too short leaves little to quiz, and too long may not fit a reader's context.
6. **Figures:** at most 12, counting every `ltx_figure` element (so subfigures and listings set as figures count). Readers and quiz writers get text only (captions, tables and math are kept), so figure-heavy papers would lose too much.
7. **Language:** at least 90% of letters are ASCII, as a rough check for English.

## Chains

For each paper B in the chain pool, in draw order:

1. Find references in B that have an arXiv id and were submitted from 2026-01-01 up to B's own submission month.
2. Keep those B cites at least 3 times in its text. This is only a proxy for B depending on A.
3. Try the most-cited first (ties by the hash order), at most 3 per B. A must pass all screens and must not already be selected.

Stop at 3 chains or after 400 screened papers, whichever comes first. If fewer than 3 chains are found, report that. Do not loosen the rule within v1.

## After the draw

Automatic exclusion is the only way to drop a paper at draw time. After the draw, a paper may be swapped for a reserve only for a defect the screens were meant to catch, such as broken extraction or a paper that is really a survey. Each swap is logged in `data/selection/swaps.md`.

Hidden-text and injection-phrase hits are flagged in each paper's metadata, not used as screens, because stratum B papers discuss injection. Flagged text is data. Papers remain untrusted input in every later prompt.

## Known limits

- **Prior knowledge:** the window aims to post-date the readers' training data. So a reader's training cutoff should be before 2026-07-01, or the pretest must show it does not know the paper. Chain A papers can be as old as 2026-01, so the pretest matters more for them.
- **Not representative:** the sample is 12 CC-licensed AI safety preprints with clean HTML. CC licensing may correlate with author, field or style. Results do not generalize to the wider literature.
- **Unknown authorship:** some papers may be partly LLM-written, and we cannot tell which or by which family. So the plan's rule that author and reader come from different families cannot be fully enforced for originals.
- **Text only:** a result can only reflect a paper's text, not its figures.
- **Chain dependence:** citation count does not prove B needs A. The composition test itself shows whether it does.
