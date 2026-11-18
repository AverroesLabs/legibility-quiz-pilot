"""Plan v4 experiment 9, error stage (preregistration: plan-v4.md, "Experiment 9 error stage preregistration").

Construction stages, all gpt-6.1-sol:
  inventory - Dropout only: contradiction inventory of `orig` with inventory.py's prompt (no earlier accusations exist,
              so the accusation list is empty). The other papers use data/v4/inventory/inventory.jsonl (orig rows).
  design    - per paper, one call (high): 49 errors, 7 categories x (2 easy, 3 medium, 2 hard), as exact edits to
              `orig`, with `clar-explainer-orig` shown as an aid and the avoid list given.
  plant     - per paper, version (clar-explainer-orig, obf-max, clar-explainer-obf) and category: the same errors as
              exact edits to that version, in its style (experiment 2's plant prompt, reworded for an `orig` source).
              `orig` takes the design edits directly.
  gate      - per error: strong whole-paper review (experiment 0's prompt, Sol medium) of the planted
              clar-explainer-orig and clar-explainer-obf, graded by experiment 2's grade prompt (Sol low; one added
              sentence for `unsupported`); must be caught in both. Pair check (Sol medium) on all 4 planted versions.
              The same review on planted `orig` gives measured difficulty (not a gate).
Output data/v4/err9/: calls.jsonl (raw, cached), inventory-dropout.jsonl, errors.jsonl (one row per designed error).
Planted copies are not stored; planted_text(row, version) rebuilds them.

Reader stage (surviving errors only), per reader in READERS:
  clean   - each version read clean, 2 samples; each clean review graded against every surviving error of the paper
            (false accusations).
  planted - each error x version, 1 sample. Limited-view readers resume from clean sample 0 at the first changed chunk
            (limited_reader.read_planted); the whole-paper reader reads the whole planted copy (exp3.review).
  pointed - each error x version, 1 sample, planted and control (passage before planting, unplanted text). Limited
            view: the notes before the first changed chunk (clean sample 0) and every changed chunk; whole paper: the
            whole copy. For `omitted` the quoted passage is the summary plus the version's evidence quotes.
  Grading: exp3's grader (gpt-5-mini low) with GRADE_UNSUPPORTED added for that category; `regrade` has Sol low
  regrade 20 random planted reviews per reader (seed 0) over all papers.
Output data/v4/err9/reader-<reader>.jsonl (rows replaced per paper), calls in data/v4/err9/reader-calls.jsonl.
Usage:
  python3 scripts/v4/err9.py inventory [<paper>...]   # orig inventory; default Dropout; anchors too
  python3 scripts/v4/err9.py construct <paper>...
  python3 scripts/v4/err9.py read <reader> <paper>...
  python3 scripts/v4/err9.py regrade <reader>
  python3 scripts/v4/err9.py --scores [<reader>...]   # preregistered measures, bars, docs/figures/v4-exp9-readers.png
  python3 scripts/v4/err9.py --distance <reader>   # exploratory: catch against edit-to-evidence chunks
  python3 scripts/v4/err9.py --heldout <reader> [<paper>...]   # held-out measures, verdict, v4-exp9-heldout.png
  python3 scripts/v4/err9.py --anchors    # anchor addendum measures, length rule, docs/figures/v4-exp9-anchors.png
  python3 scripts/v4/err9.py --table      # survival table, and docs/figures/v4-exp9-survival.png
"""
import json, os, random, re, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from llm import Calls, ROOT, spent
import exp0_clean_reviews as exp0
import exp2_errors as exp2
import inventory as inv

OUT = ROOT / "data" / "v4" / "err9"
VDIR = ROOT / "data" / "v4" / "versions"
PAPERS = ["2609.30094v2", "2608.13754v1", "2609.02893v1", "jmlr-2014-dropout"]
VERSIONS = ["orig", "obf-max", "clar-explainer-orig", "clar-explainer-obf"]
GATE = ["clar-explainer-orig", "clar-explainer-obf"]
MODEL = "gpt-6.1-sol"
CATS = {
    "calculation": "Wrong calculation or number: a number, total, difference or percentage stated wrongly, so that it "
                   "disagrees with the numbers it is computed from or reported elsewhere in the paper.",
    "reversed": "Reversed comparison: a claim that A is higher, better, larger or earlier than B where the paper's "
                "data show the opposite.",
    "stretched": "Claim stretched beyond what was tested: a claim generalised to settings, models, data or "
                 "conditions the paper's own experiments did not cover (the paper states what was covered).",
    "causal": "Association presented as cause: a finding the paper reports as a correlation or observation is "
              "restated as a causal effect, where the paper's own design or statement shows it is only an association.",
    "carried": "Error carried into later results: two consistent edits, a wrong intermediate statement and a later "
               "claim built on it. They agree with each other but contradict the paper's data (a table or reported "
               "result), which stays unchanged.",
    "omitted": "Inconvenient result left out of a summary: a summary (abstract, contributions, discussion or "
               "conclusion) states a finding as general without an exception or negative result the paper reports "
               "elsewhere. The body still reports the exception, unchanged.",
    "unsupported": "Claim with no support: a new, specific claim (a result, number or finding) that the paper gives no "
                   "evidence for anywhere. The reader must notice that nothing in the paper supports it; nothing "
                   "contradicts it directly. Its 'evidence' is the passage(s) that report what was actually done, "
                   "which show the claim was not tested.",
}
LEVELS = {
    "easy": "the contradiction or missing support is visible in the same paragraph, or in the table it describes",
    "medium": "the evidence is in another section, or catching it needs one inference such as computing a percentage",
    "hard": "the evidence is far away (over 1,500 words in this text) or catching it needs two or more chained steps",
}
PER_LEVEL = {"easy": 2, "medium": 3, "hard": 2}

DESIGN_USER = """Below is a research paper (the original text), and a clarified rewrite of the same paper, given only
to help you understand it.

<original>
{orig}
</original>

<clarified_rewrite>
{clar}
</clarified_rewrite>

Design 49 internal errors to plant in the original text: 7 in each of these 7 categories,
{cats}

Within each category: 2 easy, 3 medium and 2 hard, where
{levels}

Requirements:
- Each error changes what the paper claims so that the paper's own content shows the change is wrong (or, for
  "unsupported", that the claim has no support in the paper). A careful reader checking claims against evidence should
  be able to catch it. No errors that need outside knowledge, no mere typos.
- The evidence must stay unchanged and must be a central fact of the paper (a reported number, table value, stated
  design or scope), so that it survives in any faithful rewrite of the paper.
- The edit must read naturally: same tone, no hedges or markers, no change in length beyond a few words.
- Errors must not overlap each other: different passages to edit, and no edit inside another error's evidence.
- Do not edit, and do not use as evidence, these passages (they already contain apparent contradictions or rewrite
  problems):
<avoid>
{avoid}
</avoid>

For each error give the edits to the original text. "old" must be an exact, unique substring of the original (copy it
character for character, at least 8 words, within one paragraph or table row); "new" is its replacement. "evidence"
quotes the content that shows the error (exact quotes from the original, under 40 words each). "steps" is the number
of reasoning steps a reader needs to connect the claim to the evidence.

Reply with JSON only:
{{"errors": [{{"id": "<category>-<level>-<n>", "category": "<category>", "level": "easy" | "medium" | "hard",
  "false_claim": "<what the planted text now wrongly says>",
  "refutation": "<what the paper's own content shows, and where>",
  "edits": [{{"old": "<exact text>", "new": "<replacement>"}}],
  "evidence": ["<exact quote>"], "steps": <integer>}}]}}"""

PLANT_USER = """Here is one version of a research paper. It is written in its own style (it may be a clarified
rewrite, or use coded jargon defined elsewhere in the text):

<copy>
{text}
</copy>

The errors below were planted in the original text of the same paper. Plant each one in this version: find the
passage that corresponds to the original's "old" text and change it so that this version now makes the same false
claim, written in this version's own style and vocabulary (keep its codes, abbreviations and sentence shape). Change
nothing else, and do not make the error more or less obvious than in the original. Keep the level of detail of
the passage you replace: do not add numbers, table or section references, or explanations that it did not have, even
if the original's edit has them. Then quote this version's counterpart of the evidence, unchanged.

<errors>
{errors}
</errors>

"old" must be an exact, unique substring of this version (copy it character for character, within one paragraph or
table row). If the claim or the evidence has no counterpart in this version, set "status" to "missing" and explain.

Reply with JSON only:
{{"errors": [{{"id": "<id>", "status": "ok" | "missing", "note": "<empty, or why missing>",
  "edits": [{{"old": "<exact text from this version>", "new": "<replacement>"}}],
  "evidence": ["<exact quote from this version, under 40 words>"]}}]}}"""

GRADE_UNSUPPORTED = ("\nThis planted error is a claim with no support in the paper: also count it as caught if one of "
                     "the reported errors flags the planted claim as unsupported or unverifiable from the paper.")

PAIR_USER = """The same error was planted in {n} versions of a research paper. The versions differ in writing style
(the original; a maximally obfuscated rewrite with coded jargon; a clarified rewrite of the original; a clarified
rewrite of the obfuscated version), and should otherwise say the same things.

Intended error: {false_claim}
Why it is wrong: {refutation}{unsup}

For each version: the passage before planting, the planted passage (with its surrounding paragraph) and that
version's quoted evidence:

{blocks}

Check whether the error is planted equally: does every version make the same false claim, with the same strength and
scope, and does every version contain the evidence? Also flag a version whose planting added detail that its own
passage did not have before (numbers, references, explanations), or removed detail it had, since that makes the error
easier or harder to see in that version. Compare each version only with its own passage before planting: the versions
are meant to differ in how much detail they give, and such differences are not problems. Ignore differences of style.
Reply with JSON only:
{{"equal": true | false, "problems": [{{"copy": "<version>", "problem": "<one sentence>"}}]}}"""
PAIR_UNSUPPORTED = ("\nThis error is a claim with no support: instead of contradicting evidence, check that no version "
                    "contains support for the planted claim.")


def text(p, v):
    """A paper's clean version. `obf-max` is `obf-max-r2` where the first sample failed its check (plan rule)."""
    path = VDIR / p / f"{v}.md"
    if v == "obf-max" and not path.exists():
        path = VDIR / p / "obf-max-r2.md"
    return path.read_text()


def norm(s):
    return re.sub(r"\s+", " ", s).strip().lower()


def avoid_list(p):
    """Quotes the designer must not edit or use as evidence (plan: inventory of orig, last-round fidelity issues of
    clar-explainer-obf, and for 2608.13754v1 the sentence with 32,491, which obf-max lost)."""
    out = []
    rows = [json.loads(l) for l in inventory_path(p).read_text().splitlines()]
    for r in rows:
        if r["paper"] == p and r["copy"] == "orig":
            out += [q for x in r.get("contradictions") or [] for q in x.get("quotes", [])]
    for l in (VDIR / "clarify-log.jsonl").read_text().splitlines():
        r = json.loads(l)
        if r["paper"] == p and r["version"] == "clar-explainer-obf":
            out += [q for i in r["rounds"][-1]["issues"] for q in (i.get("original_quote"), i.get("rewrite_quote")) if q]
    if p == "2608.13754v1":
        out.append("null drawn from the full 32,491-edge population")
    return out


def overlaps(a, b):
    a, b = norm(a), norm(b)
    return bool(a) and bool(b) and (a in b or b in a)


def word_distance(src, edits, evidence):
    """Smallest distance in words between any edited span and any evidence quote in src (0 if they overlap)."""
    def span(q):
        m = exp2.loose(q).search(src)
        return (len(src[:m.start()].split()), len(src[:m.end()].split())) if m else None
    es = [s for s in (span(e["old"]) for e in edits) if s]
    vs = [s for s in (span(q) for q in evidence) if s]
    if not es or not vs:
        return None
    return min(max(0, v[0] - e[1], e[0] - v[1]) for e in es for v in vs)


def planted_text(row, v):
    t, prob = exp2.apply(text(row["paper"], v), row["copies"][v]["edits"])
    assert prob is None, prob
    return t


def inventory_path(p):
    """V4's inventory for the 3 development papers; one file per paper run here (Dropout, anchors)."""
    if p in PAPERS[:3]:
        return ROOT / "data/v4/inventory/inventory.jsonl"
    return OUT / ("inventory-dropout.jsonl" if p == "jmlr-2014-dropout" else f"inventory-{p}.jsonl")


def run_inventory(calls, p):
    src = text(p, "orig")
    rec = calls.call(inv.MODEL, inv.SYSTEM, inv.USER.format(paper=src, accusations="[]"), tag=f"err9:inventory:{p}",
                     effort=inv.EFFORT, max_tokens=30000)
    d = inv.parse(rec["content"])
    row = {"paper": p, "copy": "orig", "model": inv.MODEL, "effort": inv.EFFORT, "model_returned": rec["model_returned"],
           "parsed": d is not None, "cost": rec["cost"]}
    if d:
        for x in d["contradictions"]:
            x["quotes_found"] = [exp2.found(src, q) for q in x.get("quotes", [])]
        row["contradictions"] = d["contradictions"]
    inventory_path(p).write_text(json.dumps(row) + "\n")
    print(f"{p} orig: {len(row.get('contradictions', []))} contradictions; ${rec['cost']:.3f}")


def design(calls, p, avoid):
    user = DESIGN_USER.format(orig=text(p, "orig"), clar=text(p, "clar-explainer-orig"),
                              cats="\n".join(f"- {c}: {d}" for c, d in CATS.items()),
                              levels="\n".join(f"- {k}: {d}" for k, d in LEVELS.items()),
                              avoid="\n".join(f"- {q}" for q in avoid) or "(none)")
    rec = calls.call(MODEL, exp2.SYSTEM, user, tag=f"err9:design:{p}", effort="high", max_tokens=100000)
    d = parse_reply(rec["content"], "errors")
    if d is None:
        raise RuntimeError(f"design {p}: unparsed")
    return d["errors"]


def plant(calls, p, v, errs):
    spec = [{k: e[k] for k in ("id", "category", "false_claim", "refutation", "edits", "evidence")} for e in errs]
    rec = calls.call(MODEL, exp2.SYSTEM, PLANT_USER.format(text=text(p, v), errors=json.dumps(spec, indent=1)),
                     tag=f"err9:plant:{p}:{v}:{errs[0]['category']}", effort="medium", max_tokens=40000)
    d = parse_reply(rec["content"], "errors")
    if d is None:
        raise RuntimeError(f"plant {p} {v}: unparsed")
    return {e["id"]: e for e in d["errors"] if isinstance(e, dict) and "id" in e}


def strong(calls, row, v):
    try:
        t = planted_text(row, v)
        rec = calls.call(MODEL, exp0.REVIEW_SYSTEM, exp0.REVIEW_USER.format(paper=t),
                         tag=f"err9:strong:{row['paper']}:{v}:{row['id']}", effort="medium", max_tokens=30000)
        review = exp0.parse(rec["content"])
        if review is None:
            return {"caught": None, "note": "review unparsed", "cost": rec["cost"]}
        user = exp2.GRADE_USER.format(false_claim=row["false_claim"], refutation=row["refutation"],
                                      planted=" ... ".join(e["new"] for e in row["copies"][v]["edits"]),
                                      review=json.dumps(review, indent=1))
        if row["category"] == "unsupported":
            user += GRADE_UNSUPPORTED
        g = calls.call(MODEL, exp2.SYSTEM, user, tag=f"err9:grade:{row['paper']}:{v}:{row['id']}", effort="low",
                       max_tokens=8000)
        gd = parse_reply(g["content"], "caught")
        return {"n_reported": len(review), "caught": gd["caught"] if gd else None, "grade": gd,
                "cost": rec["cost"] + g["cost"]}
    except RuntimeError as e:
        return {"caught": None, "note": f"failed: {e}"}


def pair(calls, row):
    blocks = []
    for v in VERSIONS:
        cc = row["copies"][v]
        t = planted_text(row, v)
        blocks.append(f"<version name=\"{v}\">\nBefore planting:\n" + "\n---\n".join(e["old"] for e in cc["edits"]) +
                      "\nPlanted passage(s):\n" + "\n---\n".join(exp2.context(t, e["new"], 400) for e in cc["edits"]) +
                      "\nEvidence quoted from this version:\n" + "\n".join(f"- {q}" for q in cc["evidence"]) +
                      "\n</version>")
    try:
        rec = calls.call(MODEL, exp2.SYSTEM, PAIR_USER.format(
            n=len(VERSIONS), false_claim=row["false_claim"], refutation=row["refutation"], blocks="\n\n".join(blocks),
            unsup=PAIR_UNSUPPORTED if row["category"] == "unsupported" else ""),
            tag=f"err9:pair:{row['paper']}:{row['id']}", effort="medium", max_tokens=16000)
    except RuntimeError as e:
        return {"equal": None, "problems": [{"copy": "", "problem": f"failed: {e}"}]}
    d = parse_reply(rec["content"], "equal")
    return d if d else {"equal": None, "problems": [{"copy": "", "problem": "unparsed"}]}


def construct(calls, p):
    avoid = avoid_list(p)
    errs = design(calls, p, avoid)
    by_cat = {}
    for e in errs:
        by_cat.setdefault(e.get("category"), []).append(e)
    jobs = [(v, es) for v in VERSIONS[1:] for es in by_cat.values()]
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(lambda j: plant(calls, p, j[0], j[1]), jobs))
    planted = {v: {} for v in VERSIONS[1:]}
    for (v, _), r in zip(jobs, res):
        planted[v].update(r)
    orig = text(p, "orig")
    rows = []
    for e in errs:
        row = {"paper": p, **{k: e.get(k) for k in ("id", "category", "level", "false_claim", "refutation", "steps")},
               "distance_words": word_distance(orig, e["edits"], e["evidence"]), "copies": {}, "problems": []}
        for v in VERSIONS:
            cc = {"edits": e["edits"], "evidence": e["evidence"]} if v == "orig" else planted[v].get(e["id"])
            if cc is None or cc.get("status") == "missing":
                row["problems"].append(f"{v}: missing ({(cc or {}).get('note', 'no entry')})")
                continue
            cc = {"edits": cc.get("edits") or [], "evidence": cc.get("evidence") or []}
            src = text(p, v)
            if not cc["edits"]:
                row["problems"].append(f"{v}: no edits")
            _, prob = exp2.apply(src, cc["edits"])
            if prob:
                row["problems"].append(f"{v}: {prob}")
            cc["evidence_found"] = [exp2.found(src, q) for q in cc["evidence"]]
            hit = [q for q in avoid for x in [ed["old"] for ed in cc["edits"]] + cc["evidence"] if overlaps(q, x)]
            if hit:
                row["problems"].append(f"{v}: edit or evidence overlaps the avoid list")
            row["copies"][v] = cc
        rows.append(row)
    ok = [r for r in rows if not r["problems"]]
    jobs = [(r, v) for r in ok for v in VERSIONS if v != "obf-max"]
    with ThreadPoolExecutor(12) as ex:
        res = list(ex.map(lambda j: strong(calls, j[0], j[1]), jobs))
        prs = list(ex.map(lambda r: pair(calls, r), ok))
    for (r, v), s in zip(jobs, res):
        r["copies"][v]["strong"] = s
    for r, pr in zip(ok, prs):
        r["pair"] = pr
    for r in rows:
        r["survives"] = (not r["problems"] and all(r["copies"][v]["strong"]["caught"] is True for v in GATE)
                         and r["pair"]["equal"] is True)
    return rows


# ---------- reader stage ----------

# name: (model, effort, variant or None for the whole-paper reader); fixed order of the preregistration
READERS = {"mini-high-A": ("gpt-5-mini", "high", "A"), "whole-mini-high": ("gpt-5-mini", "high", None),
           "luna-low-A": ("gpt-5.6-luna", "low", "A"), "mini54-A": ("gpt-5.4-mini", "medium", "A"),
           "gpt55-A": ("gpt-5.5", "low", "A"), "gpt55-A-anchors": ("gpt-5.5", "low", "A"),
           # anchor addendum ladder: all calibration papers (the 4 original and the anchors), one file per step
           "gpt55-A600": ("gpt-5.5", "low", "A600"), "gpt55-A1200": ("gpt-5.5", "low", "A1200"),
           # held-out run: only the one the addendum's rule selects is ever run (decisions.md, before the first call)
           "gpt55-A-heldout": ("gpt-5.5", "low", "A"), "gpt55-A600-heldout": ("gpt-5.5", "low", "A600"),
           "gpt55-A1200-heldout": ("gpt-5.5", "low", "A1200"),
           # Jason, INBOX 2026-11-16 ~19:30 UTC: the frozen anchor reader on clar-explainer-orig only (calibration
           # standard, exploratory); own file so gpt55-A-anchors' orig/obf-max rows stay untouched
           "gpt55-A-anchors-clar": ("gpt-5.5", "low", "A"),
           # experiment 11 (plan-v4.md): whole paper in context, orig, obf-max and clar-explainer-orig, primary survivors,
           # 2 planted samples
           "w11-mini-high": ("gpt-5-mini", "high", None), "w11-luna-low": ("gpt-5.6-luna", "low", None),
           "w11-mini54-med": ("gpt-5.4-mini", "medium", None), "w11-gpt55-low": ("gpt-5.5", "low", None),
           # Jason, INBOX 2026-11-17 ~02:00 and ~15:30 UTC: exploratory fill-in, the calibration readers' frozen settings
           # on the 6 anchors, primary survivors, orig, obf-max and clar-explainer-orig, as gpt55-A-anchors (+ -clar)
           "mini-high-A-anchors": ("gpt-5-mini", "high", "A"), "mini54-A-anchors": ("gpt-5.4-mini", "medium", "A"),
           "luna-low-A-anchors": ("gpt-5.6-luna", "low", "A")}
FILL = {"mini-high-A-anchors", "mini54-A-anchors", "luna-low-A-anchors"}
EXP11 = {"w11-mini-high", "w11-luna-low", "w11-mini54-med", "w11-gpt55-low"}
# Frozen audit runs (anchor addendum and held-out run, plan-v4.md): orig and obf-max only, primary-category survivors
# only; own file each, so the calibration readers' rows and scores are untouched.
FROZEN = {"gpt55-A-anchors", "gpt55-A-anchors-clar", "gpt55-A600", "gpt55-A1200", "gpt55-A-heldout", "gpt55-A600-heldout",
          "gpt55-A1200-heldout"}
# Reader whose rows on the 4 original papers sit beside each anchor reader (the figure and the ladder's condition).
CAL_READER = {"gpt55-A-anchors": "gpt55-A", "gpt55-A600": "gpt55-A600", "gpt55-A1200": "gpt55-A1200"}
CLEAN_SAMPLES = 2
N_REGRADE = 20
QUESTION = "Check whether this passage is consistent with, and supported by, the rest of the paper as you know it."

POINTED_LIMITED = """You are reading a research paper in order, one part at a time, and you never see earlier parts again.

Your notes from the earlier parts:
<notes>
{notes}
</notes>

Current part{s}:
<part>
{chunks}
</part>

{question}

<passage>
{passage}
</passage>

Reply with JSON only, in this form:
{{"consistent": true | false, "evidence": "<if not consistent or not supported: the content (from your notes or
the current part) that contradicts the passage or shows it lacks support, and where it is; otherwise empty>",
"explanation": "<one or two sentences>"}}"""

POINTED_WHOLE = """Paper:

<paper>
{paper}
</paper>

{question}

<passage>
{passage}
</passage>

Reply with JSON only, in this form:
{{"consistent": true | false, "evidence": "<if not consistent or not supported: the content of the paper that
contradicts the passage or shows it lacks support, and where it is; otherwise empty>",
"explanation": "<one or two sentences>"}}"""


def survivors(p=None):
    rows = [json.loads(l) for l in (OUT / "errors.jsonl").read_text().splitlines()]
    return [r for r in rows if r["survives"] and (p is None or r["paper"] == p)]


def grade(calls, err, v, errors, who=None, tag=""):
    """exp3.grade (experiment 3's grader, unchanged), plus GRADE_UNSUPPORTED for that category."""
    import exp3_grid as exp3
    who = who or exp3.GRADER
    if not errors:
        return {"caught": False if errors == [] else None, "grade": None}
    user = exp2.GRADE_USER.format(false_claim=err["false_claim"], refutation=err["refutation"],
                                  planted=exp3.planted(err, v), review=json.dumps(errors, indent=1))
    if err["category"] == "unsupported":
        user += GRADE_UNSUPPORTED
    try:
        g = calls.call(who[0], exp2.SYSTEM, user, tag=tag, effort=who[1], max_tokens=8000)
    except RuntimeError as e:
        return {"caught": None, "grade": None, "note": f"failed: {e}"}
    d = exp2.parse(g["content"], "caught")
    return {"caught": d["caught"] if d else None, "grade": d, "grade_cost": g["cost"]}


def pointed_passage(err, v, side):
    cc = err["copies"][v]
    psg = " ... ".join(e[side] for e in cc["edits"])
    if err["category"] == "omitted":
        psg += " ... " + " ... ".join(cc["evidence"])
    return psg


def parse_reply(content, key):
    """exp2.parse, falling back to experiment 8's parser, which doubles invalid backslash escapes (luna writes LaTeX
    such as \\( or \\pi into JSON strings; run-log passes 15 and 32)."""
    import exp8_limited as exp8
    d = exp2.parse(content, key)
    if d is None:
        m = re.search(r"\{.*\}", content or "", re.S)
        d = exp8.parse_json(m.group(0)) if m else None
        d = d if isinstance(d, dict) and key in d else None
    return d


def pointed(calls, rd, err, v, clean0, control):
    import exp8_limited as exp8
    import limited_reader as lr
    model, effort, variant = READERS[rd]
    p = err["paper"]
    t = text(p, v) if control else planted_text(err, v)
    psg = pointed_passage(err, v, "old" if control else "new")
    cond = "control" if control else "planted"
    out = {"kind": "pointed", "cond": cond, "reader": rd, "paper": p, "error": err["id"], "category": err["category"],
           "level": err["level"], "version": v, "passage": psg}
    if variant:
        ch_c = exp8.chunks(text(p, v), variant)
        ch_p = exp8.chunks(planted_text(err, v), variant)
        if len(ch_c) == len(ch_p):
            ks = [i for i, (a, b) in enumerate(zip(ch_c, ch_p)) if a != b]
            shown = [(ch_c if control else ch_p)[k] for k in ks]
            k0 = ks[0]
        else:   # planting moved a chunk boundary: show the run of chunks between the common prefix and suffix
            k0 = next(i for i, (a, b) in enumerate(zip(ch_c, ch_p)) if a != b)
            j = next(i for i, (a, b) in enumerate(zip(ch_c[::-1], ch_p[::-1])) if a != b)
            ch = ch_c if control else ch_p
            shown, ks = ch[k0:len(ch) - j], list(range(k0, len(ch) - j))
            out["chunk_count_changed"] = True
        notes = clean0["steps"][k0]["notes_before"]
        out["chunks_shown"] = [k + 1 for k in ks]
        user = POINTED_LIMITED.format(notes=notes, s="s" if len(shown) > 1 else "",
                                      chunks="\n\n[...]\n\n".join(shown), question=QUESTION, passage=psg)
    else:
        user = POINTED_WHOLE.format(paper=t, question=QUESTION, passage=psg)
    try:
        rec = calls.call(model, exp0.REVIEW_SYSTEM, user, tag=f"err9:pointed:{cond}:{rd}:{p}:{v}:{err['id']}",
                         effort=effort, max_tokens=32000)
    except RuntimeError as e:
        return {**out, "failed": str(e), "caught": None}
    d = parse_reply(rec["content"], "consistent")
    out.update(cost=rec["cost"], parsed=d is not None, reply=d)
    if d is None:
        return {**out, "caught": None}
    out["says_inconsistent"] = d["consistent"] is False
    if not out["says_inconsistent"]:
        return {**out, "caught": False}
    item = [{"claim": psg, "location": "", "evidence": d.get("evidence", ""), "explanation": d.get("explanation", "")}]
    out.update(grade(calls, err, v, item, tag=f"err9:pgrade:{cond}:{rd}:{p}:{v}:{err['id']}"))
    return out


# Whole-paper reads at high effort need the 32,000-token allowance set in experiment 4 (plan-v4.md); exp3.review's
# 16,000 default cut off 14 of 44 reads in reader 2's first slice (run-log pass 31).
WHOLE_MAX_TOKENS = 32000


def whole_review(calls, rd, text_, tag, s):
    """exp3.review; for experiment 11 readers an unparsed reply is re-read with parse_reply's fallbacks (gpt-5.5 writes
    LaTeX such as \\( into JSON strings, which exp0.parse rejects; run-log pass 43). The reply comes from the cache."""
    import exp3_grid as exp3
    model, effort, _ = READERS[rd]
    rev = exp3.review(calls, (rd, model, effort), text_, tag, s, max_tokens=WHOLE_MAX_TOKENS)
    if rd in EXP11 and rev.get("parsed") is False:
        rec = calls.call(model, exp0.REVIEW_SYSTEM, exp0.REVIEW_USER.format(paper=text_), tag=tag, sample=s,
                         effort=effort, max_tokens=WHOLE_MAX_TOKENS)
        d = parse_reply(rec["content"], "errors")
        if d is not None and isinstance(d["errors"], list):
            rev.update(parsed=True, errors=d["errors"], n_errors=len(d["errors"]), parse_fallback=True)
        else:
            items = salvage_items(rec["content"])
            if items:
                rev.update(parsed=True, errors=items, n_errors=len(items), parse_fallback="salvage")
    return rev


ITEM_RE = re.compile(r'\{\s*"claim"\s*:.*?"explanation"\s*:\s*"(?:[^"\\]|\\.)*"\s*\}', re.S)


def salvage_items(content):
    """Review items parsed one by one, for replies whose JSON is broken outside the items (luna appends stray text
    after the list; gpt-5-mini once wrote "" + "..."; run-log pass 43). Items that do not parse are dropped."""
    import exp8_limited as exp8
    out = []
    for m in ITEM_RE.finditer(content or ""):
        d = exp8.parse_json(m.group(0))
        if isinstance(d, dict) and "claim" in d:
            out.append(d)
    return out


def read_paper(calls, rd, p):
    import exp3_grid as exp3
    import limited_reader as lr
    model, effort, variant = READERS[rd]
    errs = survivors(p)
    vers = VERSIONS
    if rd in FROZEN or rd in EXP11 or rd in FILL:
        errs, vers = [r for r in errs if SETS["primary"](r["category"])], ["orig", "obf-max"]
    if rd in EXP11 or rd in FILL:     # Jason, INBOX 2026-11-14 ~22:45 UTC: the clarified original as a calibration standard
        vers = vers + ["clar-explainer-orig"]
    if rd == "gpt55-A-anchors-clar":
        vers = ["clar-explainer-orig"]

    def clean(j):
        v, s = j
        tag = f"err9:clean:{rd}:{p}:{v}"
        rev = (lr.read_clean(calls, rd_key(rd), variant, text(p, v), tag, s) if variant else
               whole_review(calls, rd, text(p, v), tag, s))
        return {"kind": "clean", "reader": rd, "paper": p, "version": v, "sample": s, **rev}

    def planted(j):
        r, v, s = j
        tag = f"err9:read:{rd}:{p}:{v}:{r['id']}"
        rev = (lr.read_planted(calls, rd_key(rd), variant, cleans[(v, 0)], planted_text(r, v), tag, 0) if variant else
               whole_review(calls, rd, planted_text(r, v), tag, s))
        row = {"kind": "planted", "reader": rd, "paper": p, "error": r["id"], "category": r["category"],
               "level": r["level"], "version": v, "sample": s, **rev}
        row.update(grade(calls, r, v, rev.get("errors"), tag=f"err9:grade:{rd}:{p}:{v}:{r['id']}:{s}"))
        return row

    with ThreadPoolExecutor(int(os.environ.get("V4_CONCURRENCY", "12"))) as ex:
        cl = list(ex.map(clean, [(v, s) for v in vers for s in range(CLEAN_SAMPLES)]))
        cleans = {(x["version"], x["sample"]): x for x in cl}
        pl = list(ex.map(planted, [(r, v, s) for r in errs for v in vers for s in range(2 if rd in EXP11 else 1)]))
        cg = list(ex.map(lambda j: {
            "kind": "cgrade", "reader": rd, "paper": p, "version": j[0]["version"], "sample": j[0]["sample"],
            "error": j[1]["id"], "category": j[1]["category"],
            **grade(calls, j[1], j[0]["version"], j[0].get("errors"),
                    tag=f"err9:cgrade:{rd}:{p}:{j[0]['version']}:{j[0]['sample']}:{j[1]['id']}")},
            [(c, r) for c in cl for r in errs]))
        pt = list(ex.map(lambda j: pointed(calls, rd, j[0], j[1], cleans[(j[1], 0)], j[2]),
                         [(r, v, c) for r in errs for v in vers for c in (False, True)]))
    return cl + pl + cg + pt


def rd_key(rd):
    """limited_reader takes exp8's reader names; register this reader's model and effort there."""
    import exp8_limited as exp8
    model, effort, _ = READERS[rd]
    exp8.READERS.setdefault(rd, (model, effort))
    return rd


def calls_path(rd):
    """Experiment 11 readers have one call cache each (reader-calls.jsonl is over 50 MB, and the four run as separate
    processes, which must not append to one file). The anchor fill-in readers (FILL) too."""
    return OUT / (f"reader-calls-{rd}.jsonl" if rd in EXP11 or rd in FILL else "reader-calls.jsonl")


def reader_path(rd):
    return OUT / f"reader-{rd}.jsonl"


def load_reader(rd):
    path = reader_path(rd)
    return [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []


def read_report(rows, rd):
    def rate(xs):
        n = [x for x in xs if x.get("caught") is not None]
        return f"{sum(bool(x['caught']) for x in n)}/{len(n)}"
    for p in sorted({x["paper"] for x in rows}):
        rs = [x for x in rows if x["paper"] == p]
        print(f"{rd} {p}")
        for v in [v for v in VERSIONS if any(x["version"] == v for x in rs)]:
            print(f"  {v:20s} unaided {rate([x for x in rs if x['kind'] == 'planted' and x['version'] == v])}"
                  f"  pointed {rate([x for x in rs if x['kind'] == 'pointed' and x['cond'] == 'planted' and x['version'] == v])}"
                  f"  control {rate([x for x in rs if x['kind'] == 'pointed' and x['cond'] == 'control' and x['version'] == v])}"
                  f"  fa {rate([x for x in rs if x['kind'] == 'cgrade' and x['version'] == v])}")
        reads = [x for x in rs if x["kind"] in ("clean", "planted")]
        print(f"  reads {len(reads)}, unparsed or failed steps {sum(x.get('steps_failed', 0) + (x.get('parsed') is False) + bool(x.get('failed')) for x in reads)}"
              f", grades None {sum(x.get('caught') is None for x in rs if x['kind'] != 'clean')}"
              f", cost of reads ${sum(x.get('cost', 0) for x in reads):.2f}")


def regrade(calls, rd):
    rows = load_reader(rd)
    byid = {(r["paper"], r["id"]): r for r in survivors()}
    pool = [x for x in rows if x["kind"] == "planted" and x.get("errors")]
    pick = random.Random(0).sample(pool, min(N_REGRADE, len(pool)))
    for x in pick:
        g = grade(calls, byid[(x["paper"], x["error"])], x["version"], x["errors"], ("gpt-6.1-sol", "low"),
                  tag=f"err9:check:{rd}:{x['paper']}:{x['version']}:{x['error']}")
        x["sol_caught"], x["sol_grade"] = g["caught"], g["grade"]
    reader_path(rd).write_text("".join(json.dumps(r) + "\n" for r in rows))
    ck = [x for x in pick if None not in (x.get("caught"), x.get("sol_caught"))]
    print(f"{rd} grader check: agree {sum(x['caught'] == x['sol_caught'] for x in ck)}/{len(ck)}")


# ---------- reader analysis (no API; preregistration "Measures" and "Proposed bars") ----------

SETS = {"primary": lambda c: c not in ("omitted", "unsupported"), "omitted": lambda c: c == "omitted",
        "all": lambda c: True}
ROWS = {"orig": ("orig", "clar-explainer-orig"), "obf-max": ("obf-max", "clar-explainer-obf")}


def _ratio(a, b):
    return a / b if b else float("nan")


def point(E):
    """E: list of per-error dicts {version: (unaided, pointed)}. Returns the measures for one (re)sample."""
    n = len(E)
    m = {}
    for v in VERSIONS:
        m[f"catch:{v}"] = sum(e[v][0] for e in E) / n
        m[f"pointed:{v}"] = sum(e[v][1] for e in E) / n
        pc = [e for e in E if e[v][1]]
        m[f"pscore:{v}"] = _ratio(sum(e[v][0] for e in pc), len(pc))
    for r, (v, c) in ROWS.items():
        m[f"clar:{r}"] = _ratio(m[f"catch:{v}"], m[f"catch:{c}"])
        m[f"point:{r}"] = m[f"pscore:{v}"]
    m["clar:sep"] = m["clar:orig"] - m["clar:obf-max"]
    m["point:sep"] = m["point:orig"] - m["point:obf-max"]
    m["sep:clar-point"] = m["clar:sep"] - m["point:sep"]
    return m


def scores(rd, set_="primary", B=2000):
    import math
    rows = load_reader(rd)
    errs = [r for r in survivors() if SETS[set_](r["category"]) and any(x["paper"] == r["paper"] for x in rows)]
    un = {(x["paper"], x["error"], x["version"]): x["caught"] is True for x in rows if x["kind"] == "planted"}
    pt = {(x["paper"], x["error"], x["version"]): x["caught"] is True for x in rows
          if x["kind"] == "pointed" and x["cond"] == "planted"}
    E = [{"paper": r["paper"], **{v: (un[(r["paper"], r["id"], v)], pt[(r["paper"], r["id"], v)]) for v in VERSIONS}}
         for r in errs]
    m = point(E)
    rng = random.Random(0)
    by_p = {}
    for e in E:
        by_p.setdefault(e["paper"], []).append(e)
    boot = [point([rng.choice(es) for es in by_p.values() for _ in es]) for _ in range(B)]
    ci = {}
    for k in m:
        xs = sorted(b[k] for b in boot if not math.isnan(b[k]))
        ci[k] = (xs[int(0.025 * len(xs))], xs[int(0.975 * len(xs)) - 1]) if xs else (float("nan"),) * 2
    # False accusations as in experiment 8 (the barred measure): clean reviews graded against each surviving error of
    # the paper in the set; the share graded caught. fa_any: share of clean passes matched to any error (stricter).
    fa, fa_any = {}, {}
    ids = {(r["paper"], r["id"]) for r in errs}
    for v in VERSIONS:
        g = [x for x in rows if x["kind"] == "cgrade" and x["version"] == v and (x["paper"], x["error"]) in ids
             and x["caught"] is not None]
        fa[v] = sum(x["caught"] for x in g) / len(g) if g else float("nan")
        passes = {}
        for x in g:
            passes[(x["paper"], x["sample"])] = passes.get((x["paper"], x["sample"]), False) or x["caught"]
        fa_any[v] = sum(passes.values()) / len(passes) if passes else float("nan")
    ctrl = [x for x in rows if x["kind"] == "pointed" and x["cond"] == "control"]
    m["control"] = sum(x["caught"] is True for x in ctrl) / len(ctrl)
    m["control_inconsistent"] = sum(bool(x.get("says_inconsistent")) for x in ctrl) / len(ctrl)
    return {"reader": rd, "set": set_, "n": len(E), "papers": sorted(by_p), "m": m, "ci": ci, "fa": fa, "fa_any": fa_any}


def verdict(S, kind):
    m, ci, fa = S["m"], S["ci"], S["fa"]
    sep, lo = m[f"{kind}:sep"], ci[f"{kind}:sep"][0]
    pooled_fa = sum(fa.values()) / len(fa)
    base = kind == "point" or (m["catch:clar-explainer-orig"] >= 0.60 and m["catch:clar-explainer-obf"] >= 0.50)
    if (base and m[f"{kind}:orig"] >= 0.85 and m[f"{kind}:obf-max"] <= 0.60 and sep >= 0.25 and lo > 0
            and max(fa.values()) <= 0.10):
        return "meets"
    if m[f"{kind}:orig"] >= 0.75 and sep >= 0.15 and lo > 0 and pooled_fa <= 0.15:
        return "close"
    return "no"


def print_scores(rd):
    for set_ in SETS:
        S = scores(rd, set_)
        m, ci = S["m"], S["ci"]
        f = lambda k: f"{m[k]:.2f} [{ci[k][0]:.2f}, {ci[k][1]:.2f}]"
        print(f"\n{rd}, {set_} set: {S['n']} errors on {len(S['papers'])} papers")
        for v in VERSIONS:
            print(f"  {v:20s} unaided {f('catch:' + v)}  pointed {f('pointed:' + v)}  fa {S['fa'][v]:.2f} (any, per pass {S['fa_any'][v]:.2f})")
        for kind, name in (("clar", "clarified-copy"), ("point", "pointed")):
            print(f"  {name:15s} orig row {f(kind + ':orig')}  obf-max row {f(kind + ':obf-max')}  "
                  f"separation {f(kind + ':sep')}" + (f"  verdict {verdict(S, kind)}" if set_ == "primary" else ""))
        print(f"  separation, clarified minus pointed {f('sep:clar-point')}; control accusations "
              f"{m['control']:.2f} (says inconsistent {m['control_inconsistent']:.2f})")


def chunk_distance(row, v):
    """Exploratory (weird result 26): chunks (reader variant A) between the edit and its nearest evidence quote in
    clean version v; None if a quote cannot be placed in a single chunk."""
    import exp8_limited as exp8
    cs = exp8.chunks(text(row["paper"], v), "A")
    at = lambda q: next((i for i, c in enumerate(cs) if exp2.found(c, q)), None)
    es = [i for i in (at(e["old"]) for e in row["copies"][v]["edits"]) if i is not None]
    vs = [i for i in (at(q) for q in row["copies"][v]["evidence"]) if i is not None]
    return min(abs(a - b) for a in es for b in vs) if es and vs else None


def print_distance(rd):
    """Exploratory, no API: unaided catch by version against edit-to-evidence distance in chunks."""
    rows = load_reader(rd)
    un = {(x["paper"], x["error"], x["version"]): x["caught"] is True for x in rows if x["kind"] == "planted"}
    errs = [r for r in survivors() if SETS["primary"](r["category"]) and (r["paper"], r["id"], "orig") in un]
    D = {(r["paper"], r["id"], v): chunk_distance(r, v) for r in errs for v in VERSIONS}
    print(f"{rd}: {len(errs)} primary errors on {len({r['paper'] for r in errs})} papers")
    print(f"  {'version':20s} chunks  mean dist  same chunk: catch (n)  1+ apart: catch (n)  unplaced")
    for v in VERSIONS:
        ks = [(r["paper"], r["id"], v) for r in errs]
        d = [D[k] for k in ks if D[k] is not None]
        same = [un[k] for k in ks if D[k] == 0]
        far = [un[k] for k in ks if D[k] is not None and D[k] > 0]
        nch = sum(len(__import__("exp8_limited").chunks(text(p, v), "A")) for p in {r["paper"] for r in errs})
        f = lambda xs: f"{sum(xs) / len(xs):.2f} ({len(xs)})" if xs else "  -  (0)"
        print(f"  {v:20s} {nch:6d}  {sum(d) / len(d):9.2f}  {f(same):>21s}  {f(far):>19s}  {len(ks) - len(d)}")
    pooled = [(D[k], un[k]) for k in D if D[k] is not None]
    for lo, hi in ((0, 0), (1, 1), (2, 3), (4, 99)):
        xs = [c for d, c in pooled if lo <= d <= hi]
        print(f"  pooled, distance {lo}-{hi}: catch {sum(xs) / len(xs) if xs else float('nan'):.2f} (n {len(xs)})")
    pairs = [(D[(r["paper"], r["id"], "orig")], D[(r["paper"], r["id"], "obf-max")],
              un[(r["paper"], r["id"], "orig")], un[(r["paper"], r["id"], "obf-max")]) for r in errs]
    pairs = [x for x in pairs if None not in x[:2]]
    for name, keep in (("closer in obf-max", lambda a, b: b < a), ("same distance", lambda a, b: a == b),
                       ("farther in obf-max", lambda a, b: b > a)):
        xs = [x for x in pairs if keep(x[0], x[1])]
        print(f"  {name:18s} n {len(xs):2d}: catch orig {sum(x[2] for x in xs)}, obf-max {sum(x[3] for x in xs)}"
              f"  (orig only {sum(x[2] and not x[3] for x in xs)}, obf-max only {sum(x[3] and not x[2] for x in xs)})")


def readers_figure(rds):
    """docs/figures/v4-exp9-readers.png: per reader (rows of panels), primary set pooled: left, unaided and pointed
    catch rates per version; right, the two audit scores' orig and obf-max rows with 95% intervals."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from averroes_figstyle import HATCHES, axes_style, bar, legend, save, swatch, plt
    fig, axs = plt.subplots(len(rds), 2, figsize=(12, 3.6 * len(rds)), squeeze=False,
                            gridspec_kw={"width_ratios": [3, 2], "hspace": 0.45})
    names = {"orig": "orig", "obf-max": "obf-max", "clar-explainer-orig": "clar-orig", "clar-explainer-obf": "clar-obf"}
    for (a0, a1), rd in zip(axs, rds):
        S = scores(rd)
        m, ci = S["m"], S["ci"]
        axes_style(a0, 0, 1, "catch rate")
        for i, v in enumerate(VERSIONS):
            bar(a0, i - 0.15, 0.28, m[f"catch:{v}"], ci=ci[f"catch:{v}"])
            bar(a0, i + 0.15, 0.28, m[f"pointed:{v}"], ci=ci[f"pointed:{v}"], hatch=HATCHES[1])
        a0.set_xticks(range(4), [names[v] for v in VERSIONS])
        a0.set_xlim(-0.5, 3.5)
        a0.set_title(f"{rd}: catch rates ({S['n']} primary errors, {len(S['papers'])} papers)", fontsize=10)
        top = max(1.2, max(ci[f"{k}:{r}"][1] for k in ("clar", "point") for r in ROWS) + 0.05)
        axes_style(a1, 0, top, "audit score", bold_top=False)
        a1.axhline(1, color="#4d4d4d", lw=0.8, ls="--")
        for i, kind in enumerate(("clar", "point")):
            for j, r in enumerate(ROWS):
                bar(a1, i + (j - 0.5) * 0.3, 0.28, m[f"{kind}:{r}"], hi=top, ci=ci[f"{kind}:{r}"],
                    hatch=HATCHES[2 * j])
        a1.set_xticks([0, 1], [f"clarified-copy\nsep {m['clar:sep']:.2f}", f"pointed\nsep {m['point:sep']:.2f}"])
        a1.set_xlim(-0.5, 1.5)
        a1.set_title("audit score: orig row vs obf-max row", fontsize=10)
    H = 3.6 * len(rds)  # keep the legend a fixed distance above the top row, whatever the number of readers
    fig.subplots_adjust(top=1 - 0.35 / H)
    legend(fig, [swatch(), swatch(hatch=HATCHES[1]), swatch(hatch=HATCHES[2])],
           ["unaided (left) / orig row (right)", "pointed", "obf-max row"], y=1 + 0.25 / H)
    save(fig, ROOT / "docs" / "figures" / "v4-exp9-readers.png")


# ---------- anchor addendum analysis (no API; plan-v4.md "Experiment 9 anchor addendum", Measures and length rule) ----

BANDS = ("short", "medium", "long")


def anchor_point(E):
    """Pointed rows and separation for orig and obf-max: unaided catch among errors the pointed check caught."""
    m = {}
    for v in ("orig", "obf-max"):
        pc = [e for e in E if e[v][1]]
        m[v] = _ratio(sum(e[v][0] for e in pc), len(pc))
        m[f"n:{v}"] = len(pc)
    m["sep"] = m["orig"] - m["obf-max"]
    return m


def paper_E(rows, p):
    """Per primary survivor of paper p: {version: (unaided caught, pointed caught)} for orig and obf-max."""
    un = {(x["error"], x["version"]): x["caught"] is True for x in rows if x["kind"] == "planted" and x["paper"] == p}
    pt = {(x["error"], x["version"]): x["caught"] is True for x in rows
          if x["kind"] == "pointed" and x["cond"] == "planted" and x["paper"] == p}
    return [{v: (un[(r["id"], v)], pt[(r["id"], v)]) for v in ("orig", "obf-max")}
            for r in survivors(p) if SETS["primary"](r["category"])]


def anchor_scores(rd="gpt55-A-anchors", B=2000):
    """Per anchor (bootstrap over errors) and pooled (stratified by anchor), the band difference and the length rule.
    For a ladder reader (rows on the 4 original papers too), also the ladder's second condition: the 4 original papers
    pooled at "close" or better on the pointed score (plan-v4.md, anchor addendum)."""
    import math
    import exp8_limited as exp8
    sel = json.loads((ROOT / "data/selection/anchors/selection.json").read_text())["anchors"]
    band = {a["paper"]: a["band"] for a in sel}
    words = {a["paper"]: a["main_words"] for a in sel}
    rows = load_reader(rd)
    done = [a["paper"] for a in sel if any(x["paper"] == a["paper"] for x in rows)]
    E = {p: paper_E(rows, p) for p in done}
    rng = random.Random(0)

    def ci(groups, fn, key):
        xs = sorted(x for x in (fn([rng.choice(es) for es in groups for _ in es])[key] for _ in range(B))
                    if not math.isnan(x))
        return (xs[int(0.025 * len(xs))], xs[int(0.975 * len(xs)) - 1]) if xs else (float("nan"),) * 2

    def bandiff(Es):
        by = {b: [e for p in Es for e in Es[p] if band[p] == b] for b in BANDS}
        return {"d": anchor_point(by["long"])["orig"] - anchor_point(by["short"])["orig"]}

    out = {"papers": {}, "band": band, "words": words}
    for p in done:
        m = anchor_point(E[p])
        cs = {v: len(exp8.chunks(text(p, v), "A")) for v in ("orig", "obf-max")}
        fa = {v: [x["caught"] for x in rows if x["kind"] == "cgrade" and x["paper"] == p and x["version"] == v
                  and x["caught"] is not None] for v in ("orig", "obf-max")}
        ctrl = [x["caught"] is True for x in rows if x["kind"] == "pointed" and x["cond"] == "control" and x["paper"] == p]
        out["papers"][p] = {"n": len(E[p]), "m": m, "ci": {k: ci([E[p]], anchor_point, k) for k in ("orig", "obf-max", "sep")},
                            "chunks": cs, "fa": {v: _ratio(sum(fa[v]), len(fa[v])) for v in fa},
                            "control": _ratio(sum(ctrl), len(ctrl))}
    allE = [e for p in done for e in E[p]]
    out["pooled"] = {"n": len(allE), "m": anchor_point(allE),
                     "ci": {k: ci([E[p] for p in done], anchor_point, k) for k in ("orig", "obf-max", "sep")}}
    fa_all = [x["caught"] for x in rows if x["kind"] == "cgrade" and x["paper"] in done and x["caught"] is not None]
    out["pooled"]["fa"] = _ratio(sum(fa_all), len(fa_all))
    cal = [p for p in PAPERS if any(x["paper"] == p for x in rows)]
    if cal:   # ladder reader: the 4 original papers pooled, "close" bar on the pointed score
        Ec = [paper_E(rows, p) for p in cal]
        m = anchor_point([e for es in Ec for e in es])
        lo = ci(Ec, anchor_point, "sep")[0]
        fa_c = [x["caught"] for x in rows if x["kind"] == "cgrade" and x["paper"] in cal and x["caught"] is not None]
        fa = _ratio(sum(fa_c), len(fa_c))
        out["calibration"] = {"papers": cal, "m": m, "sep_lo": lo, "fa": fa,
                              "close": m["orig"] >= 0.75 and m["sep"] >= 0.15 and lo > 0 and fa <= 0.15}
    have = {b: [p for p in done if band[p] == b] for b in BANDS}
    if have["long"] and have["short"]:
        d = bandiff(E)["d"]
        ds = []
        for _ in range(B):
            Eb = {p: [rng.choice(E[p]) for _ in E[p]] for p in done}
            x = bandiff(Eb)["d"]
            if not math.isnan(x):
                ds.append(x)
        ds.sort()
        out["length"] = {"d": d, "ci": (ds[int(0.025 * len(ds))], ds[int(0.975 * len(ds)) - 1]),
                         "anchors": {b: len(have[b]) for b in BANDS}}
    rule_a = {p: out["papers"][p]["m"]["orig"] >= 0.75 for p in done}
    L = out.get("length")
    rule_b = L is not None and (L["d"] >= -0.15 or L["ci"][0] <= 0 <= L["ci"][1])
    out["rule"] = {"a": rule_a, "b": rule_b, "passes": all(rule_a.values()) and rule_b,
                   "complete": len(done) == len(sel),
                   "length_failure": (not rule_b) or all(band[p] == "long" for p, ok in rule_a.items() if not ok)}
    import numpy as np
    xs = [math.log(words[p]) for p in done]
    for k in ("orig", "obf-max", "sep"):
        ys = [out["papers"][p]["m"][k] for p in done]
        ok = [(x, y) for x, y in zip(xs, ys) if not math.isnan(y)]
        out[f"slope:{k}"] = float(np.polyfit(*zip(*ok), 1)[0]) if len(ok) >= 2 else float("nan")
    return out


def print_anchors_clar(rd="gpt55-A-anchors", B=2000):
    """Exploratory (INBOX 2026-11-16 ~19:30 UTC): the 6 anchors' clarified original (`gpt55-A-anchors-clar`) beside the
    frozen reader's orig and obf-max rows, pooled over anchors, 95% bootstrap over errors within anchors. With rd a
    fill-in reader (FILL, INBOX 2026-11-17), the same table from that reader's one file. "diff" is obf-max - orig."""
    import math
    rows = load_reader(rd) + (load_reader("gpt55-A-anchors-clar") if rd == "gpt55-A-anchors" else [])
    print(rd)
    anchors = [a["paper"] for a in json.loads((ROOT / "data/selection/anchors/selection.json").read_text())["anchors"]]
    rows = [x for x in rows if x["paper"] in anchors]
    vs = [v for v in ("orig", "obf-max", "clar-explainer-orig") if any(x["version"] == v for x in rows)]
    done = sorted({x["paper"] for x in rows if x["version"] == vs[-1]})
    get = lambda kind, p, **k: {(x["error"], x["version"]): x["caught"] is True for x in rows if x["kind"] == kind
                                and x["paper"] == p and all(x.get(a) == b for a, b in k.items())}
    E = []
    for p in done:
        un, pt = get("planted", p), get("pointed", p, cond="planted")
        E.append([{v: (un[(r["id"], v)], pt[(r["id"], v)]) for v in vs}
                  for r in survivors(p) if SETS["primary"](r["category"])])

    def m(es):
        out = {}
        for v in vs:
            out[f"unaided:{v}"] = sum(e[v][0] for e in es) / len(es)
            out[f"pointed:{v}"] = sum(e[v][1] for e in es) / len(es)
            pc = [e for e in es if e[v][1]]
            out[f"row:{v}"] = _ratio(sum(e[v][0] for e in pc), len(pc))
        for k in ("unaided", "pointed"):
            out[f"{k}:diff"] = out[f"{k}:obf-max"] - out[f"{k}:orig"]
        return out
    rng = random.Random(0)
    bs = [m([rng.choice(es) for es in E for _ in es]) for _ in range(B)]
    M = m([e for es in E for e in es])
    print(f"anchors read in {vs[-1]}: {len(done)}; primary errors {sum(map(len, E))}")
    for k in ("unaided", "pointed", "row"):
        cells = []
        for v in vs:
            xs = sorted(b[f"{k}:{v}"] for b in bs if not math.isnan(b[f"{k}:{v}"]))
            cells.append(f"{v} {M[f'{k}:{v}']:.2f} [{xs[int(0.025 * len(xs))]:.2f}, {xs[int(0.975 * len(xs)) - 1]:.2f}]")
        if k != "row":
            xs = sorted(b[f"{k}:diff"] for b in bs)
            cells.append(f"obf-max - orig {M[f'{k}:diff']:+.2f} [{xs[int(0.025 * B)]:+.2f}, {xs[int(0.975 * B) - 1]:+.2f}]")
        print(f"  {k:8s} " + "; ".join(cells))
    for v in vs:
        fa = [x["caught"] for x in rows if x["kind"] == "cgrade" and x["version"] == v and x["paper"] in done
              and x["caught"] is not None]
        ctrl = [x["caught"] is True for x in rows if x["kind"] == "pointed" and x["cond"] == "control"
                and x["version"] == v and x["paper"] in done]
        print(f"  {v}: false accusations {sum(fa)}/{len(fa)}, pointed controls {sum(ctrl)}/{len(ctrl)}")


def print_anchors(rd="gpt55-A-anchors"):
    A = anchor_scores(rd)
    f = lambda m, c, k: f"{m[k]:.2f} [{c[k][0]:.2f}, {c[k][1]:.2f}]"
    for p, d in A["papers"].items():
        m = d["m"]
        print(f"{p} ({A['band'][p]}, {A['words'][p]} words, chunks orig {d['chunks']['orig']} obf-max {d['chunks']['obf-max']}): "
              f"{d['n']} primary errors; pointed orig row {f(m, d['ci'], 'orig')} (n={m['n:orig']}), obf-max row "
              f"{f(m, d['ci'], 'obf-max')} (n={m['n:obf-max']}), separation {f(m, d['ci'], 'sep')}; fa orig "
              f"{d['fa']['orig']:.2f} obf-max {d['fa']['obf-max']:.2f}; control {d['control']:.2f}")
    P = A["pooled"]
    print(f"pooled ({P['n']} errors): orig row {f(P['m'], P['ci'], 'orig')}, obf-max row {f(P['m'], P['ci'], 'obf-max')}, "
          f"separation {f(P['m'], P['ci'], 'sep')}; fa {P['fa']:.2f}")
    if "length" in A:
        L = A["length"]
        print(f"long minus short orig row {L['d']:.2f} [{L['ci'][0]:.2f}, {L['ci'][1]:.2f}] (anchors per band {L['anchors']})")
    print("slopes per log word: " + ", ".join(f"{k} {A['slope:' + k]:.2f}" for k in ("orig", "obf-max", "sep")))
    print(f"rule (a) per anchor {A['rule']['a']}; (b) {A['rule']['b']}; passes {A['rule']['passes']}; "
          f"complete {A['rule']['complete']}; failure is a length failure {A['rule']['length_failure']}")
    if "calibration" in A:
        C = A["calibration"]
        print(f"4 original papers ({len(C['papers'])} read): orig row {C['m']['orig']:.2f}, obf-max row "
              f"{C['m']['obf-max']:.2f}, separation {C['m']['sep']:.2f} (lower {C['sep_lo']:.2f}), fa {C['fa']:.2f}; "
              f"close or better {C['close']}; ladder step passes {A['rule']['passes'] and C['close']}")
    anchors_figure(A, rd)


def anchors_figure(A, rd="gpt55-A-anchors"):
    """docs/figures/v4-exp9-anchors.png (v4-exp9-anchors-<rd>.png for a ladder reader): per-paper pointed orig and
    obf-max rows against main-text words."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from averroes_figstyle import INK, SOFT, axes_style, save, plt
    import math
    import exp8_limited as exp8
    fig, ax = plt.subplots(figsize=(8, 4.2))
    name = READERS[rd][2]
    axes_style(ax, 0, 1.05, f"pointed audit score (gpt55-{name})", bold_top=False)
    ax.axhline(0.75, color=SOFT, lw=0.8, ls="--")
    rows = load_reader(CAL_READER[rd])
    cal = [p for p in PAPERS if any(x["paper"] == p for x in rows)]
    calw = {p: len(text(p, "orig").split()) for p in cal}
    calE = {p: anchor_point(paper_E(rows, p)) for p in cal}
    for j, (v, mk) in enumerate((("orig", "o"), ("obf-max", "s"))):
        off = 1.012 if j else 1 / 1.012
        for p, d in A["papers"].items():
            y, (lo, hi) = d["m"][v], d["ci"][v]
            if math.isnan(y):
                continue
            ax.errorbar(A["words"][p] * off, y, yerr=[[y - lo], [hi - y]], fmt=mk, color=INK,
                        mfc=INK if j == 0 else "white", ms=7, capsize=3, lw=1)
        for p, m in calE.items():
            if not math.isnan(m[v]):
                ax.plot(calw[p] * off, m[v], mk, color=SOFT, mfc="white", ms=6, alpha=0.8)
    ax.set_xscale("log")
    ax.set_xticks([4000, 6000, 8000, 12000, 16000], ["4k", "6k", "8k", "12k", "16k"])
    ax.minorticks_off()
    ax.set_xlabel("main-text words (orig)")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", color=INK, ls="", ms=7), Line2D([], [], marker="s", color=INK, mfc="white", ls="", ms=7),
         Line2D([], [], marker="o", color=SOFT, mfc="white", ls="", ms=6),
         Line2D([], [], marker="s", color=SOFT, mfc="white", ls="", ms=6)]
    ax.legend(h, ["anchor, orig row", "anchor, obf-max row", "calibration paper, orig row (seen before the rule)",
                  "calibration paper, obf-max row"],
              loc="lower left", fontsize=9)
    ax.set_title(f"Reviewed anchors, notes budget {exp8.VARIANTS[name][2]} words: pointed rows against length "
                 "(dashed: rule (a), 0.75)", fontsize=10)
    save(fig, ROOT / "docs" / "figures" / ("v4-exp9-anchors.png" if rd == "gpt55-A-anchors" else f"v4-exp9-anchors-{rd}.png"))


LADDER = (("gpt55-A-anchors", "gpt55-A", 300), ("gpt55-A600", "gpt55-A600", 600), ("gpt55-A1200", "gpt55-A1200", 1200))


def ladder_figure(B=2000):
    """docs/figures/v4-exp9-ladder.png: pooled pointed rows and separation per notes budget, on the 6 anchors and on the
    4 original papers (bootstrap over errors stratified by paper, seed 0). Budgets whose reads are missing are skipped."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from averroes_figstyle import INK, SOFT, axes_style, save, plt
    import math
    sel = [a["paper"] for a in json.loads((ROOT / "data/selection/anchors/selection.json").read_text())["anchors"]]
    fig, axs = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, (title, which) in zip(axs, (("6 reviewed anchors", "anchors"), ("4 original calibration papers", "cal"))):
        axes_style(ax, -0.2, 1.05, "pointed audit score" if which == "anchors" else None, bold_top=False)
        ax.axhline(0, color=SOFT, lw=0.6)
        xs = []
        for i, (ard, crd, budget) in enumerate(LADDER):
            if not reader_path(ard).exists() or not reader_path(crd).exists():
                continue
            rows = load_reader(ard if which == "anchors" else crd)
            ps = [p for p in (sel if which == "anchors" else PAPERS) if any(x["paper"] == p for x in rows)]
            E = [paper_E(rows, p) for p in ps]
            if not any(E):
                continue
            m = anchor_point([e for es in E for e in es])
            rng = random.Random(0)
            boot = [anchor_point([rng.choice(es) for es in E for _ in es]) for _ in range(B)]
            for j, (k, mk, face) in enumerate((("orig", "o", INK), ("obf-max", "s", "white"), ("sep", "D", SOFT))):
                b = sorted(x[k] for x in boot if not math.isnan(x[k]))
                lo, hi = b[int(0.025 * len(b))], b[int(0.975 * len(b)) - 1]
                ax.errorbar(i + (j - 1) * 0.18, m[k], yerr=[[m[k] - lo], [hi - m[k]]], fmt=mk, color=INK, mfc=face,
                            ms=7, capsize=3, lw=1)
                if k == "sep":
                    ax.annotate(f"{m[k]:.2f}", (i + 0.18, m[k]), xytext=(6, 0), textcoords="offset points",
                                va="center", fontsize=8, color=SOFT)
            xs.append((i, f"{budget} words\n({len(ps)} paper{'s' * (len(ps) > 1)})"))
        ax.set_xticks([x for x, _ in xs], [l for _, l in xs])
        ax.set_xlim(-0.5, len(LADDER) - 0.5)
        ax.set_xlabel("notes budget")
        ax.set_title(title, fontsize=10)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", color=INK, ls="", ms=7), Line2D([], [], marker="s", color=INK, mfc="white", ls="", ms=7),
         Line2D([], [], marker="D", color=INK, mfc=SOFT, ls="", ms=7)]
    fig.legend(h, ["orig row", "obf-max row", "separation (orig minus obf-max)"], loc="upper center", ncol=3,
               fontsize=9, frameon=False)
    fig.subplots_adjust(top=0.82)
    save(fig, ROOT / "docs" / "figures" / "v4-exp9-ladder.png")


# ---------- held-out run analysis (no API; plan-v4.md "Experiment 9 held-out run: preregistration", Measures, bars) ----

HELDOUT = ["2609.00755v1", "2607.17117v2", "2609.34091v1", "2608.25490v1", "2607.28685v1", "2607.10112v1",
           "2607.20379v2", "2608.19216v1", "2608.30724v1"]
# The same notes budget's reads on the anchors and on the 4 original papers, drawn beside the held-out papers.
HELDOUT_REF = {"gpt55-A-heldout": ("gpt55-A-anchors", "gpt55-A"), "gpt55-A600-heldout": ("gpt55-A600", "gpt55-A600"),
               "gpt55-A1200-heldout": ("gpt55-A1200", "gpt55-A1200")}


def heldout_scores(rd, papers=HELDOUT, B=2000):
    """Pooled pointed rows (bootstrap over errors stratified by paper, B resamples, seed 0), false accusations per
    clean pass per version and pooled, control accusations, per-paper rows, and the verdict on the proposed bars."""
    import math
    import exp8_limited as exp8
    rows = load_reader(rd)
    done = [p for p in papers if any(x["paper"] == p for x in rows)]
    E = {p: paper_E(rows, p) for p in done}
    ids = {(r["paper"], r["id"]) for r in survivors() if r["paper"] in done and SETS["primary"](r["category"])}
    cg = [x for x in rows if x["kind"] == "cgrade" and (x["paper"], x["error"]) in ids and x["caught"] is not None]
    ctrl = [x["caught"] is True for x in rows if x["kind"] == "pointed" and x["cond"] == "control" and x["paper"] in done]
    out = {"reader": rd, "papers": {}, "complete": done == list(papers)}
    for p in done:
        out["papers"][p] = {"n": len(E[p]), "m": anchor_point(E[p]), "words": len(text(p, "orig").split()),
                            "chunks": {v: len(exp8.chunks(text(p, v), READERS[rd][2])) for v in ("orig", "obf-max")},
                            "fa": {v: _ratio(sum(x["caught"] for x in cg if x["paper"] == p and x["version"] == v),
                                             sum(x["paper"] == p and x["version"] == v for x in cg))
                                   for v in ("orig", "obf-max")}}
    m = anchor_point([e for p in done for e in E[p]])
    rng = random.Random(0)
    boot = [anchor_point([rng.choice(E[p]) for p in done for _ in E[p]]) for _ in range(B)]
    ci = {}
    for k in ("orig", "obf-max", "sep"):
        xs = sorted(b[k] for b in boot if not math.isnan(b[k]))
        ci[k] = (xs[int(0.025 * len(xs))], xs[int(0.975 * len(xs)) - 1]) if xs else (float("nan"),) * 2
    fa = {v: _ratio(sum(x["caught"] for x in cg if x["version"] == v), sum(x["version"] == v for x in cg))
          for v in ("orig", "obf-max")}
    passes = {}
    for x in cg:
        k = (x["paper"], x["version"], x["sample"])
        passes[k] = passes.get(k, False) or x["caught"]
    out["pooled"] = {"n": sum(len(E[p]) for p in done), "m": m, "ci": ci, "fa": fa, "fa_pooled": _ratio(sum(x["caught"] for x in cg), len(cg)),
                     "fa_any": _ratio(sum(passes.values()), len(passes)), "control": _ratio(sum(ctrl), len(ctrl))}
    sep, lo = m["sep"], ci["sep"][0]
    if m["orig"] >= 0.85 and m["obf-max"] <= 0.60 and sep >= 0.25 and lo > 0 and max(fa.values()) <= 0.10:
        out["verdict"] = "meets"
    elif m["orig"] >= 0.75 and sep >= 0.15 and lo > 0 and out["pooled"]["fa_pooled"] <= 0.15:
        out["verdict"] = "close"
    else:
        out["verdict"] = "no"
    return out


def print_heldout(rd, papers=HELDOUT):
    H = heldout_scores(rd, papers)
    f = lambda m, c, k: f"{m[k]:.2f} [{c[k][0]:.2f}, {c[k][1]:.2f}]"
    for p, d in H["papers"].items():
        m = d["m"]
        print(f"{p} ({d['words']} words, chunks orig {d['chunks']['orig']} obf-max {d['chunks']['obf-max']}): {d['n']} "
              f"primary errors; pointed orig row {m['orig']:.2f} (n={m['n:orig']}), obf-max row {m['obf-max']:.2f} "
              f"(n={m['n:obf-max']}), separation {m['sep']:.2f}; fa orig {d['fa']['orig']:.2f} obf-max {d['fa']['obf-max']:.2f}")
    P = H["pooled"]
    print(f"pooled ({P['n']} errors, {len(H['papers'])} of {len(papers)} papers): orig row {f(P['m'], P['ci'], 'orig')} "
          f"(n={P['m']['n:orig']}), obf-max row {f(P['m'], P['ci'], 'obf-max')} (n={P['m']['n:obf-max']}), separation "
          f"{f(P['m'], P['ci'], 'sep')}; fa orig {P['fa']['orig']:.2f} obf-max {P['fa']['obf-max']:.2f} pooled "
          f"{P['fa_pooled']:.2f} (any error per pass {P['fa_any']:.2f}); control {P['control']:.2f}")
    print(f"verdict on the proposed bars: {H['verdict']}" + ("" if H["complete"] else " (INCOMPLETE: not all papers read)"))
    heldout_figure(H, rd)


def heldout_figure(H, rd):
    """docs/figures/v4-exp9-heldout.png: per-paper pointed orig and obf-max rows against main-text words, held-out
    papers in ink, anchors and calibration papers (same notes budget) in grey."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from averroes_figstyle import INK, SOFT, axes_style, save, plt
    from matplotlib.lines import Line2D
    import math
    import exp8_limited as exp8
    fig, ax = plt.subplots(figsize=(8, 4.2))
    name = READERS[rd][2]
    axes_style(ax, 0, 1.05, f"pointed audit score (gpt55-{name})", bold_top=False)
    ref = {}
    for r in dict.fromkeys(HELDOUT_REF.get(rd, ())):
        if reader_path(r).exists():
            rows = load_reader(r)
            for p in dict.fromkeys(x["paper"] for x in rows):
                ref[p] = (len(text(p, "orig").split()), anchor_point(paper_E(rows, p)))
    for j, (v, mk) in enumerate((("orig", "o"), ("obf-max", "s"))):
        off = 1.012 if j else 1 / 1.012
        for p, d in H["papers"].items():
            if not math.isnan(d["m"][v]):
                ax.plot(d["words"] * off, d["m"][v], mk, color=INK, mfc=INK if j == 0 else "white", ms=7)
        for p, (w, m) in ref.items():
            if not math.isnan(m[v]):
                ax.plot(w * off, m[v], mk, color=SOFT, mfc="white", ms=6, alpha=0.8)
    ax.set_xscale("log")
    ax.set_xticks([4000, 6000, 8000, 12000, 16000], ["4k", "6k", "8k", "12k", "16k"])
    ax.minorticks_off()
    ax.set_xlabel("main-text words (orig)")
    h = [Line2D([], [], marker="o", color=INK, ls="", ms=7), Line2D([], [], marker="s", color=INK, mfc="white", ls="", ms=7),
         Line2D([], [], marker="o", color=SOFT, mfc="white", ls="", ms=6),
         Line2D([], [], marker="s", color=SOFT, mfc="white", ls="", ms=6)]
    ax.legend(h, ["held-out paper, orig row", "held-out paper, obf-max row", "anchor or calibration paper, orig row",
                  "anchor or calibration paper, obf-max row"], loc="lower left", fontsize=9)
    P = H["pooled"]
    ax.set_title(f"Held-out papers, notes budget {exp8.VARIANTS[name][2]} words: pooled orig {P['m']['orig']:.2f}, "
                 f"obf-max {P['m']['obf-max']:.2f}, separation {P['m']['sep']:.2f} (verdict {H['verdict']})", fontsize=10)
    save(fig, ROOT / "docs" / "figures" / ("v4-exp9-heldout.png" if H["complete"] else "v4-exp9-heldout-partial.png"))


def table():
    rows = [json.loads(l) for l in (OUT / "errors.jsonl").read_text().splitlines()]
    for p in PAPERS:
        rs = [r for r in rows if r["paper"] == p]
        if not rs:
            continue
        print(f"\n{p}: {sum(r['survives'] for r in rs)} of {len(rs)} survive")
        fail = {"plant/avoid": sum(bool(r["problems"]) for r in rs)}
        ok = [r for r in rs if not r["problems"]]
        for v in GATE:
            fail[f"strong missed {v}"] = sum(r["copies"][v]["strong"]["caught"] is not True for r in ok)
        fail["pair not equal"] = sum(r["pair"]["equal"] is not True for r in ok)
        fail["strong caught orig (measured)"] = sum(r["copies"]["orig"]["strong"]["caught"] is True for r in ok)
        print("  " + "; ".join(f"{k}: {n}" for k, n in fail.items()))
        print("  survivors by category x level (easy/medium/hard):")
        for c in CATS:
            print(f"    {c:12s} " + " ".join(str(sum(r["survives"] for r in rs if r["category"] == c and r["level"] == l))
                                         for l in LEVELS))


def figure():
    """docs/figures/v4-exp9-survival.png: errors left after each construction stage, by category and designed level,
    pooled over the papers in errors.jsonl."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from averroes_figstyle import HATCHES, axes_style, bar, legend, save, swatch, plt
    rows = [json.loads(l) for l in (OUT / "errors.jsonl").read_text().splitlines()]
    n_p = len({r["paper"] for r in rows})
    stages = [("planted in all 4 versions", lambda r: not r["problems"]),
              ("+ strong reader catches in both clarified", lambda r: not r["problems"] and all(
                  r["copies"][v]["strong"]["caught"] is True for v in GATE)),
              ("+ pair check (survives)", lambda r: r["survives"])]
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [7, 3]})
    for ax, key, groups, top in ((axs[0], "category", list(CATS), 7 * n_p),
                                 (axs[1], "level", list(LEVELS), max(PER_LEVEL.values()) * 7 * n_p)):
        axes_style(ax, 0, top)
        for i, g in enumerate(groups):
            rs = [r for r in rows if r[key] == g]
            for j, (_, f) in enumerate(stages):
                bar(ax, i + (j - 1) * 0.27, 0.25, sum(map(f, rs)), hi=top, hatch=HATCHES[j], label="{:.0f}")
            ax.text(i, top, f"of {len(rs)}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(range(len(groups)), groups)
        ax.set_xlim(-0.5, len(groups) - 0.5)
    axs[0].set_ylabel("errors")
    axs[0].set_title(f"by category ({n_p} papers)", fontsize=10, pad=14)
    axs[1].set_title("by designed level", fontsize=10, pad=14)
    legend(fig, [swatch(hatch=HATCHES[j]) for j in range(3)], [s for s, _ in stages], y=1.1)
    save(fig, ROOT / "docs" / "figures" / "v4-exp9-survival.png")


def main():
    calls = Calls(OUT / "calls.jsonl")
    if sys.argv[1:2] == ["inventory"]:
        for p in sys.argv[2:] or ["jmlr-2014-dropout"]:
            run_inventory(calls, p)
    elif sys.argv[1:2] == ["construct"]:
        path = OUT / "errors.jsonl"
        old = [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
        for p in sys.argv[2:]:
            rows = construct(calls, p)
            old = [r for r in old if r["paper"] != p] + rows
            path.write_text("".join(json.dumps(r) + "\n" for r in old))
            print(f"{p}: {sum(r['survives'] for r in rows)} of {len(rows)} survive; ledger ${spent():.2f}")
    elif sys.argv[1:2] == ["read"]:
        rd = sys.argv[2]
        rcalls = Calls(calls_path(rd))
        for p in sys.argv[3:]:
            new = read_paper(rcalls, rd, p)
            rows = [x for x in load_reader(rd) if x["paper"] != p] + new
            reader_path(rd).write_text("".join(json.dumps(r) + "\n" for r in rows))
            read_report(new, rd)
            print(f"ledger ${spent():.2f}")
    elif sys.argv[1:2] == ["regrade"]:
        regrade(Calls(calls_path(sys.argv[2])), sys.argv[2])
    elif "--scores" in sys.argv:
        rds = sys.argv[sys.argv.index("--scores") + 1:] or [r for r in READERS if reader_path(r).exists() and r not in FROZEN | FILL]
        for rd in rds:
            print_scores(rd)
        readers_figure(rds)
    elif sys.argv[1:2] == ["--distance"]:
        print_distance(sys.argv[2])
    elif sys.argv[1:2] == ["--heldout"]:
        print_heldout(sys.argv[2], sys.argv[3:] or HELDOUT)
    elif "--ladder" in sys.argv:
        ladder_figure()
    elif "--anchors-clar" in sys.argv:
        print_anchors_clar(*sys.argv[sys.argv.index("--anchors-clar") + 1:][:1])
    elif "--anchors" in sys.argv:
        print_anchors(*sys.argv[sys.argv.index("--anchors") + 1:][:1])
    elif "--table" in sys.argv:
        table()
        figure()
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
