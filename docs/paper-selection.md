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

# Reviewed anchors (v4, experiment 9), rule v1

**Committed 2026-11-13, pod pass 22, before any candidate list or text was fetched.** So far I have only checked that two OpenReview group endpoints (`ICLR.cc/2026/Conference` and `TMLR`) return HTTP 200, without reading their bodies. Jason's 03:00 UTC note (`INBOX.md`) asks for 4 to 6 accepted papers from venues with public reviews, in AI safety or adjacent ML, spanning about 4,000 to 15,000 words of main text, chosen by a fixed rule with no one's judgment of clarity. Any change after the draw gets a new rule version, with the reason in `docs/decisions.md`. The draw will be done by `scripts/v4/select_anchors.py`, with output in `data/selection/anchors/` and papers in `data/papers/<arxiv id>v<version>/`, using V1's `paper_text.parse` and the cache helpers in `scripts/select_papers.py`.

## Frame

- **Venues, both with public reviews on OpenReview:** papers accepted at ICLR 2026 (OpenReview API v2, `content.venueid = ICLR.cc/2026/Conference`), and TMLR papers accepted from 2025-01-01 to 2026-10-31 (`content.venueid = TMLR`, by the acceptance date OpenReview reports). TMLR has no page limit, so it supplies the long band, which ICLR's 10-page limit cannot. If the API does not return ICLR 2026 accepted papers, ICLR 2025 (`ICLR.cc/2025/Conference`) is used instead, and this is logged. Recent venues also lower the chance that readers saw the paper in training.
- **Area:** the OpenReview title or abstract contains at least one of these terms, ignoring case and hyphens: "interpretability", "sparse autoencoder", "jailbreak", "red teaming", "prompt injection", "AI safety", "alignment", "oversight", "scheming", "sandbagging", "reward hacking", "backdoor", "adversarial", "robustness", "out-of-distribution", "unlearning", "privacy", "hallucination". This is a mechanical match only, so off-topic matches (for example image "alignment") stay in. This is the "adjacent ML" allowance.

## Draw

Papers are taken in ascending order of `sha256("20261113-anchors:" + OpenReview forum id)`. Each is screened in that order. A paper that passes fills a place in its length band. **Bands, by main-text words:** short 4,000 to 7,000, medium 7,000 to 10,000, long 10,000 to 15,000 (a paper of exactly 7,000 or 10,000 words goes in the lower band). **Two places per band, 6 anchors.** The draw stops when every band is full, or after 600 screened papers. A band that does not fill is reported and not loosened. If fewer than 4 anchors pass in total, a new rule version is written with the reason. No person or model chooses among passing papers.

## Screens, in order (first failure is logged)

1. **arXiv version:** an arXiv paper whose title matches the OpenReview title exactly (lower case, punctuation and whitespace removed) and that shares at least one author surname, found through the arXiv API title search (or the OpenReview note's arXiv link, if it has one). The version used is the earliest one submitted on or after the OpenReview decision date (the camera-ready, most likely). If no version is that recent, the latest version is used and this is recorded.
2. **Licence:** CC BY, CC BY-SA or CC0 on the arXiv abstract page, as for V1.
3. **Type:** not marked as a survey, review, tutorial, position paper, perspective, overview, workshop report or withdrawn (V1's patterns).
4. **Independence:** not one of the 13 papers in `data/selection/v3-expanded-13.json`, and not cited in `docs/background/`.
5. **Format:** arXiv native HTML exists for that version, with at most 5 `ltx_ERROR` elements.
6. **Main text:** the parsed body up to the first level-1 or level-2 heading that starts with "Appendix", "Appendices" or "Supplementary", or with a single capital letter used as a section number (such as "A ", "A.1"). Its words decide the band. A paper outside 4,000 to 15,000 words, or whose band is full, fails with the reason "length" or "band full".
7. **Figures:** at most 12 `ltx_figure` elements in the main text.
8. **Language:** at least 90% of the main text's letters are ASCII.

## What is recorded per anchor

OpenReview forum id and URL, venue and decision, acceptance date, rank hash, arXiv id and version, licence and its source URL, HTML sha256, main-text sha256, word and figure counts, conversion errors, the appendix cut point, the hidden-text and injection-phrase flags (as in V1, data only), and the date retrieved. `orig` for experiment 9 is the main text, with references and appendices removed. Every screened paper's failure reason goes in `data/selection/anchors/screen-log.jsonl`.

## Known limits

- **Reviewed is not the same as clearly written.** Acceptance shows that reviewers judged the work fit to publish, not that it is easy to read. The report claims only that the audit separates reviewed, published papers from deliberately garbled versions of them.
- **The arXiv text may differ from the reviewed PDF.** The camera-ready rule makes this unlikely, but it is not checked against the PDF.
- **Not representative:** the anchors are CC-licensed papers with clean arXiv HTML, from two venues.
- **Main-text cut:** appendices often hold evidence that the main text points to. Errors whose evidence would be in an appendix cannot be planted. This is the same for every version of an anchor.

# Reviewed anchors, rule v2 (replaces v1 before any draw)

**Committed 2026-11-13, pod pass 23, before any candidate's text or arXiv record was fetched.** Rule v1 could not be run: OpenReview's `notes` endpoints (API v1 and v2) now answer every anonymous query with HTTP 403 `ChallengeRequiredError`, a bot check. Only the `groups` endpoint answers. I did not try to get past the check. Rule v2 changes only the frame source and the order key; everything else in v1 stands. Reason in `docs/decisions.md` (pass 23). Before writing this I looked at the structure of the two listing pages below (counts, markup, the first entry of each), not at any paper's text.

- **Frame:** (1) TMLR's accepted-paper list, `https://jmlr.org/tmlr/papers/`, entries whose acceptance month is January 2025 to October 2026. Each entry gives title, authors, month and OpenReview forum id. (2) ICLR 2026's accepted-paper list, `https://iclr.cc/virtual/2026/papers.html` (the `/virtual/2026/poster/<id>` links), which gives title only. Authors and the forum id come from the paper's poster page, fetched only when the paper is screened. A title in both lists counts once, as TMLR.
- **Area:** the listed **title** contains one of v1's terms (same list, ignoring case and hyphens). The abstract is not used for the frame, because neither list carries it, and fetching one per listed paper (about 10,000) is not sensible.
- **Order:** ascending `sha256("20261113-anchors:" + normalized title)`, where the normalized title is lower case with everything but letters and digits removed. The forum id cannot be the key, since ICLR's list does not carry it.
- **Decision date** for screen 1: for TMLR, the first day of the listed acceptance month; for ICLR 2026, 2026-01-01 (decisions were sent in January 2026; the day is not checked). The arXiv version used is the earliest submitted on or after that date, else the latest (recorded).
- **Screen 1 (arXiv version):** arXiv API title search (`ti:` with the title's words); a hit counts if its normalized title equals the listed one and it shares an author surname (last word of a name, lower case). If several hits match, the earliest arXiv id is used.
- **Screen 3 (type):** V1's patterns, applied to the title and the arXiv abstract.
- **Screen 6 (main text):** the parsed body (V1's `paper_text.parse`) is cut at the first line `#`/`##` heading whose text starts with "Appendix", "Appendices" or "Supplementary", or with a single capital letter followed by a space or a period and digit ("A Proofs", "A.1"), after optional leading section numbers are stripped. References are already outside the body.
- **Screen 7 (figures):** `ltx_figure` elements counted in the HTML before the first element with class `ltx_appendix` (or in the whole HTML if there is none). This approximates "figures in the main text" without mapping HTML to the text cut.
- Bands, places (2 per band, 6 total), the 600-paper screen cap, licence, independence, format and language screens, recorded fields and known limits are as in v1. Output `data/selection/anchors/` (`frame.jsonl`, `screen-log.jsonl`, `selection.json`); papers in `data/papers/<arxiv id>v<version>/` with `text.md` the full body and `main.md` the main text (`orig`).
- **Added limit:** matching on titles only gives a smaller and more literal area frame than v1's title-or-abstract match.
