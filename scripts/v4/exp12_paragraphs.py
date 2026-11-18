"""Experiment 12 (plan-v4.md, "Experiment 12 preregistration"): every paragraph in turn, whole paper in context.

For each copy (the clean `orig` / `obf-max` version, and one planted copy per primary survivor), the reader is asked
about each paragraph in turn whether it is consistent with the rest of the paper, without being told which paragraph
holds the planted error. The paper comes first and the question last, so the paper prefix bills as cached input.

  python3 scripts/v4/exp12_paragraphs.py --count                       # paragraphs per version, no API
  python3 scripts/v4/exp12_paragraphs.py read <reader> <paper>... [--clean-only] [--versions orig]
  python3 scripts/v4/exp12_paragraphs.py batch [--slice]               # Batch API (INBOX 2026-11-17 ~20:00 UTC): one
                                                                       # round: collect ended batches, build their rows,
                                                                       # submit the next units within the $400 note cap
  python3 scripts/v4/exp12_paragraphs.py --scores [<reader>...]       # measures, docs/figures/v4-exp12-paragraphs.png
  python3 scripts/v4/exp12_paragraphs.py --new-sample <reader> [n] [seed]  # exploratory: random "new" false alarms
                                                                       # to label by hand (pass 48), no API
  python3 scripts/v4/exp12_paragraphs.py --elsewhere <reader>         # exploratory: grade non-planted flags in planted
                                                                       # copies with err9.grade (pass 48), then print
  python3 scripts/v4/exp12_paragraphs.py --stable <reader>            # exploratory: false alarms from paragraphs flagged
                                                                       # in >=80% of a version's copies (pass 51), no API
  python3 scripts/v4/exp12_paragraphs.py --headsplit <reader> [--count]  # exploratory (weird result 35, pass 52):
                                                                       # re-ask planted blocks split at headings
  python3 scripts/v4/exp12_paragraphs.py --regroup <reader>           # check: planted paragraphs with no edited word
                                                                       # (regrouping by the 40-word merge, pass 49), no API

Output: data/v4/exp12/<reader>.jsonl (one row per paragraph question), calls in data/v4/exp12/calls-<reader>.jsonl.
Caught: the planted paragraph (a paragraph of the planted copy that is not in the clean copy) is flagged and
experiment 9's grader (err9.grade) says the flag names the planted error. False alarm: a flag on any other paragraph.
"""
import collections, json, os, random, re, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from llm import Calls, ROOT, SPEND, _key, _params, _utc, price, spent
import err9
import exp0_clean_reviews as exp0
import exp8_limited as exp8

OUT = ROOT / "data" / "v4" / "exp12"
FIG = ROOT / "docs" / "figures" / "v4-exp12-paragraphs.png"
READERS = {"p12-mini-high": ("gpt-5-mini", "high"), "p12-gpt55-low": ("gpt-5.5", "low"),
           "p12-mini54-med": ("gpt-5.4-mini", "medium"), "p12-luna-low": ("gpt-5.6-luna", "low")}  # last two: exploratory
# Max output tokens. The preregistration's 32,000 stays for the direct runs; batch runs use the observed maximum with
# margin (INBOX 2026-11-17 ~20:00 UTC), because submit_batch reserves max_tokens x requests. gpt-5.5 low: exp 12 max
# 1,893 of 12,960; luna low: exp 11 whole-paper max 1,890; gpt-5.4-mini medium: exp 11 whole-paper max 21,191 (one
# outlier; median 3,276) and variant A max 8,371, so 16,000 for a one-paragraph question. Truncated = unparsed.
MAXTOK = {"p12-gpt55-low": 4000, "p12-luna-low": 4000, "p12-mini54-med": 16000}
ANCHORS = ["2601.02896v4", "2603.01214v2", "2509.22621v3", "2509.11379v3", "2510.17247v2", "2506.22982v1"]
VERS = ("orig", "obf-max", "clar-explainer-orig")   # third: Jason, INBOX 2026-11-14 ~22:45 UTC
MIN_WORDS = 40

USER = """Paper:

<paper>
{paper}
</paper>

Paragraph {i} of {n} of the paper above:

<paragraph>
{para}
</paragraph>

Is this paragraph consistent with the rest of the paper? If not, say what conflicts.

Reply with JSON only, in this form:
{{"consistent": true | false, "conflict": "<if not consistent: the claim in this paragraph that conflicts, quoted or
closely paraphrased; otherwise empty>", "evidence": "<the content elsewhere in the paper that it conflicts with, and
where it is; otherwise empty>", "explanation": "<one or two sentences>"}}"""


MARK = re.compile(r"^(?:[-*•·▪◦|]|\(?[0-9ivx]+[.):]|\(?[a-z][.)])$")      # a bare list marker or table bar
ITEM = re.compile(r"^(?:[-*•·▪◦]|\d+[.)]|\([0-9ivxa-z]+\))\s")


def paragraphs(t):
    """Natural paragraphs (blank-line blocks) of the main text, references and everything after dropped (exp8.sections).
    Lists and tables are kept whole: a block is joined to the one before it if that one ends with ':' or is a bare
    list marker, or if the block is itself a bare marker or starts with a list marker (pass 43 fix: bare '•' blocks
    had cut lists in half). Then a block under 40 words (headings, short lines) is merged into the next, and a short
    last block into the previous."""
    blocks = []
    for b in (b.strip() for s in exp8.sections(t) for b in re.split(r"\n\s*\n", s) if b.strip()):
        if blocks and (blocks[-1].endswith(":") or MARK.match(blocks[-1].splitlines()[-1].strip())
                       or MARK.match(b) or ITEM.match(b)):
            blocks[-1] += "\n\n" + b
        else:
            blocks.append(b)
    out, cur = [], []
    for b in blocks:
        cur.append(b)
        if len(" ".join(cur).split()) >= MIN_WORDS:
            out.append("\n\n".join(cur))
            cur = []
    if cur:
        if out:
            out[-1] += "\n\n" + "\n\n".join(cur)
        else:
            out.append("\n\n".join(cur))
    return out


def primary(p):
    return [r for r in err9.survivors(p) if err9.SETS["primary"](r["category"])]


def copies(p, v, clean_only=False):
    """(copy name, text, planted paragraph indices, error row or None)."""
    clean = err9.text(p, v)
    yield "clean", clean, [], None
    if clean_only:
        return
    base = set(paragraphs(clean))
    for r in primary(p):
        t = err9.planted_text(r, v)
        hit = [i for i, x in enumerate(paragraphs(t)) if x not in base]
        assert hit, f"{p} {v} {r['id']}: planted copy has no changed paragraph"
        yield r["id"], t, hit, r


def cached(calls, rd, user):
    """The logged reply at this reader's max_tokens or the preregistered 32,000 (gpt-5.5's direct calls), else None."""
    model, effort = READERS[rd]
    for mt in (MAXTOK.get(rd, 32000), 32000):
        rec = calls.cache.get(_key(model, exp0.REVIEW_SYSTEM, user, _params(model, mt, effort), 0))
        if rec:
            return rec


def ask(calls, rd, p, v, name, t, hit, err, i, paras, offline=False):
    """offline: build the row from logged replies only (batch runs); a question with no reply counts as failed."""
    model, effort = READERS[rd]
    row = {"reader": rd, "paper": p, "version": v, "copy": name, "para": i, "n_paras": len(paras),
           "words": len(paras[i].split()), "planted": i in hit}
    if err:
        row.update(error=err["id"], category=err["category"], level=err["level"])
    user = USER.format(paper=t, i=i + 1, n=len(paras), para=paras[i])
    try:
        rec = cached(calls, rd, user)
        if rec is None and offline:
            raise RuntimeError("no batch reply (truncated or failed; see the calls file)")
        rec = rec or calls.call(model, exp0.REVIEW_SYSTEM, user, tag=f"exp12:{rd}:{p}:{v}:{name}:{i}", effort=effort,
                                max_tokens=MAXTOK.get(rd, 32000))
    except RuntimeError as e:
        return {**row, "failed": str(e), "flag": None}
    d = err9.parse_reply(rec["content"], "consistent")
    parsed = d is not None
    m = None if parsed else re.search(r'"consistent"\s*:\s*(true|false)', rec["content"] or "")
    if m:       # malformed JSON with a readable verdict (pass 43 check): keep the raw reply as the explanation
        d, parsed = {"consistent": m.group(1) == "true", "conflict": "", "evidence": "",
                     "explanation": rec["content"][:4000]}, "fallback"
    row.update(cost=rec["cost"], parsed=parsed, reply=d,
               cached=rec.get("usage", {}).get("input_tokens_details", {}).get("cached_tokens"),
               input_tokens=rec.get("usage", {}).get("input_tokens"))
    row["flag"] = None if d is None else d["consistent"] is False
    if row["planted"] and row["flag"]:
        item = [{"claim": d.get("conflict") or paras[i], "location": f"paragraph {i + 1}",
                 "evidence": d.get("evidence", ""), "explanation": d.get("explanation", "")}]
        row.update(err9.grade(calls, err, v, item, tag=f"exp12:grade:{rd}:{p}:{v}:{name}:{i}"))
    return row


def read(rd, papers, clean_only=False, versions=VERS, offline=False):
    OUT.mkdir(parents=True, exist_ok=True)
    calls = Calls(OUT / f"calls-{rd}.jsonl")
    path = OUT / f"{rd}.jsonl"
    for p in papers:
        jobs = [(p, v, name, t, hit, err, i, paras) for v in versions
                for name, t, hit, err in copies(p, v, clean_only) for paras in [paragraphs(t)] for i in range(len(paras))]
        with ThreadPoolExecutor(int(os.environ.get("V4_CONCURRENCY", "16"))) as ex:
            new = list(ex.map(lambda j: ask(calls, rd, *j, offline=offline), jobs))
        keys = {(x["paper"], x["version"], x["copy"]) for x in new}
        old = [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
        rows = [x for x in old if (x["paper"], x["version"], x["copy"]) not in keys] + new
        path.write_text("".join(json.dumps(x) + "\n" for x in rows))
        bad = sum(x["flag"] is None for x in new)
        print(f"{rd} {p}: {len(new)} questions, unparsed or failed {bad}, flags {sum(bool(x['flag']) for x in new)}, "
              f"cost ${sum(x.get('cost', 0) for x in new):.2f}; ledger ${spent():.2f}", flush=True)


# ---------- Batch API runs (INBOX 2026-11-17 ~20:00 UTC) ----------

NOTE_START, NOTE_CAP = "2026-11-17T20:00:00Z", 400.0
# Jason's order: run 3 (luna), run 1 (gpt-5.5 clarified, slice paper reused), run 2 (gpt-5.4-mini); papers in the
# preregistered order (calibration, then anchors), one batch per paper and version.
UNITS = [(rd, p, v) for rd, vs, skip in (("p12-luna-low", VERS, ()), ("p12-gpt55-low", ("clar-explainer-orig",),
                                                                      ("2609.30094v2",)), ("p12-mini54-med", VERS, ()))
         for p in err9.PAPERS + ANCHORS if p not in skip for v in vs]


def note_spend():
    """Settled plus still-reserved spend of this note: exp 12 ledger rows (reader and grader) since the note."""
    tot = 0.0
    for f in sorted(SPEND.parent.glob(SPEND.stem + "*.jsonl")):
        for line in f.open():
            if '"exp12' in line:
                r = json.loads(line)
                if r.get("utc", "") >= NOTE_START and r.get("tag", "").startswith("exp12"):
                    tot += r["usd"]
    return tot


def worst(rd, items):
    """submit_batch's reservation for these items (batch rate)."""
    pin, _, pout = price(READERS[rd][0])
    return 0.5 * (sum(len(it["system"]) + len(it["user"]) for it in items) / 2.5 * pin
                  + len(items) * MAXTOK[rd] * pout) / 1e6


def unit_parts(rd, p, v):
    """The unit's questions in order, cut into consecutive parts of at most 100 MB of JSON (the Batch API takes
    200 MB per file) and $60 of reservation, so no single submission reserves a large share of the note cap."""
    items = [{"system": exp0.REVIEW_SYSTEM, "user": USER.format(paper=t, i=i + 1, n=len(ps), para=ps[i])}
             for _, t, _, _ in copies(p, v) for ps in [paragraphs(t)] for i in range(len(ps))]
    parts, cur, size = [], [], 0
    for it in items:
        n = len(json.dumps(it)) + 300
        if cur and (size + n > 100e6 or worst(rd, cur + [it]) > 60):
            parts.append(cur)
            cur, size = [], 0
        cur.append(it)
        size += n
    return parts + [cur]


def batch_round(slice_only=False):
    """One round, no waiting: collect every ended batch, build rows for collected units, then submit units in order
    while this note's settled plus reserved spend stays under NOTE_CAP. --slice: only each reader's first paper.
    Run from one process at a time (collect_batch's rule)."""
    OUT.mkdir(parents=True, exist_ok=True)
    calls = {rd: Calls(OUT / f"calls-{rd}.jsonl") for rd in {u[0] for u in UNITS}}
    for rd, c in calls.items():
        for b in [b for b in c.pending_batches() if b["tag"].startswith(f"exp12b:{rd}:")]:
            print(f"collect {b['tag']} {b['batch']}: {c.collect_batch(b['batch'])}", flush=True)
    recs = [json.loads(l) for l in (OUT / "batches.jsonl").read_text().splitlines()] \
        if (OUT / "batches.jsonl").exists() else []
    sub = {r["batch"]: r["tag"] for r in recs if "tag" in r}
    ended = {sub[r["batch"]] for r in recs if r.get("done")}
    pending = set(sub.values()) - ended
    first = {}
    for rd, p, _ in UNITS:
        first.setdefault(rd, p)
    blocked, left = None, []
    for rd, p, v in UNITS:
        parts = unit_parts(rd, p, v)
        tags = [f"exp12b:{rd}:{p}:{v}:{k}" for k in range(len(parts))]
        if all(t in ended for t in tags):
            if not any(r["paper"] == p and r["version"] == v for r in load(rd)):
                read(rd, [p], versions=(v,), offline=True)
            continue
        model, effort = READERS[rd]
        for tag, items in zip(tags, parts):
            if tag in ended:
                continue
            if tag in pending or (slice_only and p != first[rd]) or blocked:
                left.append(tag)
                continue
            w = worst(rd, items)
            if note_spend() + w > NOTE_CAP:
                blocked = f"{tag}: note spend ${note_spend():.2f} + reservation ${w:.2f} > ${NOTE_CAP:.0f}"
                left.append(tag)
                continue
            bid = calls[rd].submit_batch(model, items, tag=tag, max_tokens=MAXTOK[rd], effort=effort)
            print(f"submit {tag}: {len(items)} requests, reserved ${w:.2f}, batch {bid}", flush=True)
            left.append(tag)
            if bid is None:     # every question already has a reply: mark the part ended
                with (OUT / "batches.jsonl").open("a") as f:
                    f.write(json.dumps({"batch": f"cached:{tag}", "tag": tag, "n": 0, "utc": NOTE_START}) + "\n"
                            + json.dumps({"batch": f"cached:{tag}", "done": True, "status": "cached"}) + "\n")
    npend = sum(1 for rd, c in calls.items() for b in c.pending_batches() if b["tag"].startswith(f"exp12b:{rd}:"))
    print(f"{_utc()} note spend ${note_spend():.2f} (settled + reserved) of ${NOTE_CAP:.0f}; ledger ${spent():.2f}; "
          f"parts not done {len(left)}, in flight {npend}" + (f"; BLOCKED {blocked}" if blocked else ""), flush=True)
    return left, blocked, npend


# ---------- analysis (no API) ----------

def load(rd):
    path = OUT / f"{rd}.jsonl"
    return [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []


def caught_by_error(rows):
    """{(paper, error): {version: 1.0 / 0.0}}: caught if any planted paragraph of that copy is flagged and graded caught."""
    acc = collections.defaultdict(dict)
    grouped = collections.defaultdict(list)
    for r in rows:
        if r["copy"] != "clean" and r["planted"]:
            grouped[(r["paper"], r["error"], r["version"])].append(r)
    for (p, e, v), rs in grouped.items():
        if any(r["flag"] is None or (r["flag"] and r.get("caught") is None) for r in rs):
            continue
        acc[(p, e)][v] = 1.0 if any(r["flag"] and r.get("caught") for r in rs) else 0.0
    vs = vers(rows)   # gpt-5.5 low read orig and obf-max only (INBOX 2026-11-16 ~19:30 UTC)
    return {k: d for k, d in acc.items() if all(v in d for v in vs)}


def vers(rows):
    """The versions a reader actually read, in VERS order."""
    seen = {r["version"] for r in rows}
    return tuple(v for v in VERS if v in seen)


def boot(pe, B=2000):
    by_paper = collections.defaultdict(list)
    for (p, _), d in pe.items():
        by_paper[p].append(d)
    mean = lambda s, v: sum(d[v] for d in s) / len(s)
    vals = list(pe.values())
    vs = tuple(v for v in VERS if v in vals[0])
    out = {"n": len(vals), "papers": len(by_paper), **{v: mean(vals, v) for v in vs}}
    out["diff"] = out["orig"] - out["obf-max"]
    rnd, bs = random.Random(0), {k: [] for k in vs + ("diff",)}
    for _ in range(B):
        s = [x for lst in by_paper.values() for x in (rnd.choice(lst) for _ in lst)]
        for v in vs:
            bs[v].append(mean(s, v))
        bs["diff"].append(bs["orig"][-1] - bs["obf-max"][-1])
    for k, b in bs.items():
        b.sort()
        out[k + ":ci"] = (b[int(0.025 * B)], b[int(0.975 * B) - 1])
    return out


def scores(rd):
    rows = load(rd)
    for g, papers in (("4 calibration papers", err9.PAPERS), ("6 reviewed papers", ANCHORS)):
        rs = [r for r in rows if r["paper"] in papers]
        if not rs:
            continue
        pe = caught_by_error(rs)
        print(f"{rd} [{g}]")
        if pe:
            s = boot(pe)
            f = lambda k: (f"{s[k]:+.2f} [{s[k + ':ci'][0]:+.2f}, {s[k + ':ci'][1]:+.2f}]" if k == "diff" else
                           f"{s[k]:.2f} [{s[k + ':ci'][0]:.2f}, {s[k + ':ci'][1]:.2f}]")
            cl = f('clar-explainer-orig') if 'clar-explainer-orig' in s else "not read"
            print(f"  caught  garbled {f('obf-max')}  original {f('orig')}  clarified {cl}"
                  f"  orig-obf {f('diff')}  ({s['n']} errors)")
        new = mark_inherited(rs)
        for v in vers(rs):
            fa = [r for r in rs if r["version"] == v and not r["planted"] and r["flag"] is not None]
            cl = [r for r in fa if r["copy"] == "clean"]
            pl = [r for r in fa if r["copy"] != "clean"]
            wrong = [r for r in rs if r["version"] == v and r["planted"] and r["flag"] and r.get("caught") is False]
            c1, c2 = fa_boot(rs, v), fa_boot(rs, v, True)
            print(f"  {v:8s} false alarms per paragraph {rate(fa):.3f} [{c1[0]:.3f}, {c1[1]:.3f}] ({sum(r['flag'] for r in fa)}/{len(fa)}),"
                  f" per 1,000 words {rate(fa, 1):.2f} [{c2[0]:.2f}, {c2[1]:.2f}]; clean copies {sum(r['flag'] for r in cl)}/{len(cl)}"
                  f" ({rate(cl, 1):.2f} per 1,000 words); planted paragraph flagged for another reason {len(wrong)}")
            nw = sum(1 for r in pl if r["flag"] and id(r) in new)
            print(f"  {'':8s} exploratory, planted copies: new {nw}/{len(pl)} = {nw / max(len(pl), 1):.3f} per paragraph,"
                  f" {1000 * nw / max(sum(r['words'] for r in pl), 1):.2f} per 1,000 words;"
                  f" inherited {sum(r['flag'] for r in pl) - nw}")
        for w, u in ((False, "per paragraph"), (True, "per 1,000 words")):
            lo, hi = fa_diff_boot(rs, w)
            print(f"  exploratory, garbled - original false alarms {u}: {fa_rate(rs, 'obf-max', w) - fa_rate(rs, 'orig', w):+.3f}"
                  f" [{lo:+.3f}, {hi:+.3f}]")
        up = [p for p in papers if any(r["paper"] == p for r in rs)]
        for w, u in ((False, "per paragraph"), (True, "per 1,000 words")):
            k = sum(fa_rate([r for r in rs if r["paper"] == p], "obf-max", w) > fa_rate([r for r in rs if r["paper"] == p], "orig", w)
                    for p in up)
            print(f"  papers with garbled above original {u}: {k} of {len(up)}")
        for p in [p for p in papers if any(r["paper"] == p for r in rs)]:
            pr = [r for r in rs if r["paper"] == p and not r["planted"] and r["flag"] is not None]
            line = "  ".join(f"{v} {sum(r['flag'] for r in pr if r['version'] == v)}/{sum(r['version'] == v for r in pr)}"
                             for v in vers(rs))
            print(f"    {p:20s} false alarms {line}")
        print(f"  questions {len(rs)}, unparsed or failed {sum(r['flag'] is None for r in rs)}, "
              f"cost ${sum(r.get('cost', 0) for r in rs):.2f}")


BARS = [("obf-max", "xxxx", "Garbled version"), ("orig", "////", "Original paper"),
        ("clar-explainer-orig", "", "Clarified original")]     # as scripts/blog_part2_final.py
GROUPS = (("4 calibration papers", err9.PAPERS), ("6 reviewed papers", ANCHORS))


def rate(rs, per_words=False):
    """Flags per paragraph, or per 1,000 words of the paragraphs asked about (Jason, INBOX 2026-11-14 ~01:00 UTC)."""
    d = sum(r["words"] for r in rs) / 1000 if per_words else len(rs)
    return sum(r["flag"] for r in rs) / d if d else float("nan")


def fa_rate(rs, v, per_words=False):
    return rate([r for r in rs if r["version"] == v and not r["planted"] and r["flag"] is not None], per_words)


def fa_clusters(rs, v):
    """Paragraph clusters for fa_boot: per paper, a list of [flags, questions, words] per paragraph text in version v."""
    rs = [r for r in rs if r["version"] == v and not r["planted"] and r["flag"] is not None]
    texts, errs = {}, {}
    cl = collections.defaultdict(lambda: [0, 0, 0])
    for r in rs:
        p, c = r["paper"], r["copy"]
        if (p, c) not in texts:
            if c == "clean":
                texts[p, c] = paragraphs(err9.text(p, v))
            else:
                errs.setdefault(p, {e["id"]: e for e in primary(p)})
                texts[p, c] = paragraphs(err9.planted_text(errs[p][c], v))
        k = cl[p, texts[p, c][r["para"]]]
        k[0] += r["flag"]; k[1] += 1; k[2] += r["words"]
    by_paper = collections.defaultdict(list)
    for (p, _), x in cl.items():
        by_paper[p].append(x)
    return by_paper


def fa_draw(by_paper, rnd, per_words):
    s = [x for lst in by_paper.values() for x in (rnd.choice(lst) for _ in lst)]
    d = sum(x[2] for x in s) / 1000 if per_words else sum(x[1] for x in s)
    return sum(x[0] for x in s) / d


def fa_boot(rs, v, per_words=False, B=2000):
    """Exploratory interval (defined pass 58, before the last 3 papers' rows existed; not preregistered): 95% bootstrap
    of fa_rate. Clusters are paragraph texts within (paper, version), so one paragraph's rows across every copy move
    together; clusters are resampled within each paper (as boot() resamples errors within papers), seed 0."""
    by_paper, rnd = fa_clusters(rs, v), random.Random(0)
    if not by_paper:
        return float("nan"), float("nan")
    bs = sorted(fa_draw(by_paper, rnd, per_words) for _ in range(B))
    return bs[int(0.025 * B)], bs[int(0.975 * B) - 1]


def fa_diff_boot(rs, per_words=False, B=2000, a="obf-max", b="orig"):
    """Exploratory (plan-v4.md, defined pass 60): 95% bootstrap of fa_rate(a) - fa_rate(b); each replicate resamples
    each version's clusters within papers as fa_boot does, versions independently, seed 0."""
    ca, cb, rnd = fa_clusters(rs, a), fa_clusters(rs, b), random.Random(0)
    if not ca or not cb:
        return float("nan"), float("nan")
    bs = sorted(fa_draw(ca, rnd, per_words) - fa_draw(cb, rnd, per_words) for _ in range(B))
    return bs[int(0.025 * B)], bs[int(0.975 * B) - 1]


def mark_inherited(rs):
    """Exploratory split (plan-v4.md, experiment 12, pass 47): ids of planted-copy false-alarm rows whose paragraph
    text is NOT flagged in the same version's clean copy ("new"); the other flagged ones are "inherited"."""
    by = collections.defaultdict(dict)
    for r in rs:
        by[(r["paper"], r["version"])].setdefault(r["copy"], {})[r["para"]] = r
    new = set()
    for (p, v), cps in by.items():
        if "clean" not in cps:
            continue
        clean = paragraphs(err9.text(p, v))
        flagged = {clean[i] for i, r in cps["clean"].items() if r["flag"]}
        errs = {e["id"]: e for e in primary(p)}
        for name, prs in cps.items():
            if name == "clean" or not any(r["flag"] and not r["planted"] for r in prs.values()):
                continue
            paras = paragraphs(err9.planted_text(errs[name], v))
            new |= {id(r) for i, r in prs.items() if r["flag"] and not r["planted"] and paras[i] not in flagged}
    return new


def new_sample(rd, n=30, seed=12):
    """Exploratory (pass 48): a seeded random sample of "new" planted-copy false alarms, printed beside the planted
    false claim, for hand labelling. Labels: data/v4/exp12/new-fa-labels-<reader>.jsonl (k = sample index)."""
    rows = load(rd)
    new = mark_inherited(rows)
    errs = {(r["paper"], r["id"]): r for p in {x["paper"] for x in rows} for r in primary(p)}
    nw = [r for r in rows if r["copy"] != "clean" and not r["planted"] and r["flag"] and id(r) in new]
    random.seed(seed)
    for k, r in enumerate(random.sample(nw, min(n, len(nw)))):
        rp = r["reply"]
        print(f"[{k}] {r['paper']} {r['version']} {r['copy']} para {r['para']}\n"
              f"  PLANTED: {errs[(r['paper'], r['copy'])]['false_claim']}\n"
              f"  FLAG: {(rp.get('conflict') or '')[:300]} || {(rp.get('explanation') or '')[:300]}\n")
    print(f"{len(nw)} new false alarms in {len({r['paper'] for r in rows})} papers")


def elsewhere(rd):
    """Exploratory (plan-v4.md, experiment 12, pass 48): grade every flag on a non-planted paragraph of a planted
    copy with err9.grade; a credited flag "found elsewhere" names the planted error from another paragraph.
    Resumes from the call cache; rewrites data/v4/exp12/elsewhere-<reader>.jsonl for the papers present."""
    rows = load(rd)
    errs = {(r["paper"], r["id"]): r for p in {x["paper"] for x in rows} for r in primary(p)}
    todo = [r for r in rows if r["copy"] != "clean" and not r["planted"] and r["flag"]]
    calls = Calls(OUT / f"calls-elsewhere-{rd}.jsonl")

    def g(r):
        d = r["reply"]
        item = [{"claim": d.get("conflict") or "", "location": f"paragraph {r['para'] + 1}",
                 "evidence": d.get("evidence", ""), "explanation": d.get("explanation", "")}]
        out = err9.grade(calls, errs[(r["paper"], r["copy"])], r["version"], item,
                    tag=f"exp12:elsewhere:{rd}:{r['paper']}:{r['version']}:{r['copy']}:{r['para']}")
        return {k: r[k] for k in ("paper", "version", "copy", "para", "words")} | out

    with ThreadPoolExecutor(16) as ex:
        out = list(ex.map(g, todo))
    (OUT / f"elsewhere-{rd}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in out))
    found = {(x["paper"], x["version"], x["copy"], x["para"]) for x in out if x["caught"]}
    print(f"{rd}: graded {len(out)} flags, failed {sum(x['caught'] is None for x in out)}, "
          f"cost ${sum(x.get('grade_cost', 0) for x in out):.2f}")
    new = mark_inherited(rows)
    for v in VERS:
        fa = [r for r in rows if r["version"] == v and not r["planted"] and r["flag"] is not None]
        el = [r for r in fa if (r["paper"], r["version"], r["copy"], r["para"]) in found]
        rest = [r for r in fa if (r["paper"], r["version"], r["copy"], r["para"]) not in found]
        nw = [r for r in el if id(r) in new]
        print(f"  {v:20s} found elsewhere {len(el)} ({len(nw)} new, {len(el) - len(nw)} inherited); false alarms without "
              f"them {rate(rest):.3f} per paragraph ({sum(r['flag'] for r in rest)}/{len(rest)}), "
              f"{rate(rest, 1):.2f} per 1,000 words (all: {rate(fa):.3f}, {rate(fa, 1):.2f})")
    lab = OUT / f"new-fa-labels-{rd}.jsonl"
    if lab.exists():        # hand labels; each row names its flag (paper, version, copy, para), pass 49 fix
        agree = [(l["label"] == "names-planted-error", (l["paper"], l["version"], l["copy"], l["para"]) in found)
                 for l in map(json.loads, lab.read_text().splitlines())]
        print(f"  hand labels ({len(agree)}): grader agrees {sum(a == b for a, b in agree)}/{len(agree)}; hand yes {sum(a for a, _ in agree)},"
              f" grader yes {sum(b for _, b in agree)}, both {sum(a and b for a, b in agree)}")


def edit_words(p, v, row):
    """[(planted paragraph index, word offset in it)] of every word the planted copy's word-level diff changes."""
    import difflib
    pl = paragraphs(err9.planted_text(row, v))
    pos = [(i, k) for i, para in enumerate(pl) for k in range(len(para.split()))]
    sm = difflib.SequenceMatcher(None, " ".join(paragraphs(err9.text(p, v))).split(), " ".join(pl).split(),
                                 autojunk=False)
    return pl, sorted({pos[min(j, len(pos) - 1)] for t, _, _, j1, j2 in sm.get_opcodes() if t != "equal"
                       for j in range(j1, max(j2, j1 + 1))})


def parts(block):
    """A block split before every line that starts with '#' (a markdown heading)."""
    out = []
    for line in block.split("\n"):
        if line.startswith("#") or not out:
            out.append([])
        out[-1].append(line)
    return ["\n".join(x).strip() for x in out if "\n".join(x).strip()]


def headsplit(rd, count_only=False):
    """Exploratory (plan-v4.md, experiment 12, pass 52; weird result 35). Affected: a planted copy whose edited
    words in a planted block all fall after the block's first part (parts() splits before each heading line).
    For each, ask (a) the same block again as sample 1 (resampling control) and (b) each part of the block on its
    own, numbered as the block. Caught as preregistered (flag graded by err9.grade), on any part for (b)."""
    rows = load(rd)
    errs = {(r["paper"], r["id"]): r for p in {x["paper"] for x in rows} for r in primary(p)}
    pre = collections.defaultdict(list)
    for r in rows:
        if r["copy"] != "clean" and r["planted"]:
            pre[(r["paper"], r["version"], r["copy"])].append(r)
    jobs = []
    for (p, v, c), rr in sorted(pre.items()):
        pl, ed = edit_words(p, v, errs[(p, c)])
        for i in sorted({r["para"] for r in rr}):
            ps = parts(pl[i])
            bounds = [sum(len(x.split()) for x in ps[:k]) for k in range(len(ps) + 1)]
            mine = [k for k in (o for j, o in ed if j == i)]
            if len(ps) < 2 or not mine or min(mine) < bounds[1]:
                continue
            edit_part = {next(q for q in range(len(ps)) if bounds[q] <= o < bounds[q + 1]) for o in mine}
            jobs.append((p, v, c, i, len(pl), pl, ps, edit_part, any(r.get("caught") for r in rr)))
    n_parts = sum(len(j[6]) for j in jobs)
    print(f"{rd}: {len(jobs)} affected planted blocks, {n_parts} parts; asks {len(jobs) + n_parts}")
    for v in VERS:
        print(f"  {v:20s} {sum(j[1] == v for j in jobs)} blocks")
    if count_only:
        return
    model, effort = READERS[rd]
    calls = Calls(OUT / f"calls-headsplit-{rd}.jsonl")

    def one(t, pl, i, para, err, v, tag, sample):
        rec = calls.call(model, exp0.REVIEW_SYSTEM, USER.format(paper=t, i=i + 1, n=len(pl), para=para),
                         tag=tag, sample=sample, effort=effort, max_tokens=32000)
        d = err9.parse_reply(rec["content"], "consistent")
        out = {"flag": None if d is None else d["consistent"] is False, "cost": rec["cost"], "reply": d}
        if out["flag"]:
            item = [{"claim": d.get("conflict") or para, "location": f"paragraph {i + 1}",
                     "evidence": d.get("evidence", ""), "explanation": d.get("explanation", "")}]
            out.update(err9.grade(calls, err, v, item, tag=tag + ":grade"))
        return out

    def run(j):
        p, v, c, i, n, pl, ps, edit_part, pre_caught = j
        err, t = errs[(p, c)], err9.planted_text(errs[(p, c)], v)
        tag = f"exp12:headsplit:{rd}:{p}:{v}:{c}:{i}"
        rs = one(t, pl, i, pl[i], err, v, tag + ":resample", 1)
        sp = [one(t, pl, i, x, err, v, f"{tag}:part{k}", 0) | {"part": k, "edit_part": k in edit_part,
              "words": len(x.split())} for k, x in enumerate(ps)]
        return {"paper": p, "version": v, "copy": c, "para": i, "n_parts": len(ps), "pre_caught": pre_caught,
                "resample": rs, "split": sp, "resample_caught": bool(rs.get("caught")),
                "split_caught": any(x.get("caught") for x in sp)}

    with ThreadPoolExecutor(16) as ex:
        out = list(ex.map(run, jobs))
    (OUT / f"headsplit-{rd}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in out))
    cost = sum(x["resample"]["cost"] + sum(y["cost"] for y in x["split"]) for x in out)
    print(f"cost ${cost:.2f} (+ grades); unparsed {sum(x['resample']['flag'] is None for x in out) + sum(y['flag'] is None for x in out for y in x['split'])}")
    print(f"  {'version':20s} blocks  preregistered  resample  heading-split  flags on parts without the edit")
    for v in VERS:
        o = [x for x in out if x["version"] == v]
        other = [y for x in o for y in x["split"] if not y["edit_part"]]
        print(f"  {v:20s} {len(o):5d}  {sum(x['pre_caught'] for x in o):13d}  {sum(x['resample_caught'] for x in o):8d}"
              f"  {sum(x['split_caught'] for x in o):13d}  {sum(bool(y['flag']) for y in other)}/{len(other)}")


def regroup(rd):
    """Planted copies with more than one planted paragraph. A paragraph counts as edited if a word of the planted
    copy's word-level diff against the clean copy falls in it; the others are planted only because the edit moved a
    block across the 40-word merge. Prints whether any caught result depends on them, and their flags."""
    import difflib
    by = collections.defaultdict(list)
    for x in load(rd):
        by[(x["paper"], x["version"], x["copy"])].append(x)
    n = differ = flags = paras = 0
    for (p, v, c), rr in sorted(by.items()):
        hit = sorted(x["para"] for x in rr if x["planted"])
        if len(hit) < 2:
            continue
        n += 1
        row = next(r for r in primary(p) if r["id"] == c)
        pl = paragraphs(err9.planted_text(row, v))
        owner = [i for i, para in enumerate(pl) for _ in para.split()]
        sm = difflib.SequenceMatcher(None, " ".join(paragraphs(err9.text(p, v))).split(), " ".join(pl).split(),
                                     autojunk=False)
        edited = {owner[min(j, len(owner) - 1)] for t, _, _, j1, j2 in sm.get_opcodes() if t != "equal"
                  for j in range(j1, max(j2, j1 + 1))}
        extra = [x for x in rr if x["planted"] and x["para"] not in edited]
        c_pre = any(x.get("caught") for x in rr if x["planted"])
        c_strict = any(x.get("caught") for x in rr if x["planted"] and x["para"] in edited)
        differ += c_pre != c_strict
        flags += sum(x["flag"] for x in extra)
        paras += len(extra)
        print(f"  {p:14s} {v:20s} {c:22s} planted {hit} edited {sorted(set(hit) & edited)}"
              f"  caught {c_pre}/{c_strict}  flags on regrouped {sum(x['flag'] for x in extra)}")
    print(f"{rd}: {n} copies with >1 planted paragraph; {paras} regrouped-only paragraphs, {flags} flags on them; "
          f"caught differs in {differ}")


def stable(rd):
    """Exploratory check, no API (pass 51): how much of each version's false-alarm rate comes from paragraphs the
    reader flags in at least 80% of that version's copies (same paragraph text, clean and planted, at least 3 asks).
    Such repeat flags point at the paper's own content rather than at chance."""
    texts, by = {}, collections.defaultdict(lambda: collections.defaultdict(list))
    for r in load(rd):
        if r["planted"] or r["flag"] is None:
            continue
        p, v, c = r["paper"], r["version"], r["copy"]
        if (p, v, c) not in texts:
            t = err9.text(p, v) if c == "clean" else err9.planted_text({e["id"]: e for e in primary(p)}[c], v)
            texts[p, v, c] = paragraphs(t)
        by[p, v][texts[p, v, c][r["para"]]].append(r["flag"])
    print(f"{'paper':20s} {'version':20s} asks flags  fa/ask  share-stable  stable-paras  fa/ask-without-stable")
    for (p, v), d in sorted(by.items()):
        n, fl = sum(map(len, d.values())), sum(map(sum, d.values()))
        st = [f for f in d.values() if len(f) >= 3 and sum(f) / len(f) >= 0.8]
        sn, sf = sum(map(len, st)), sum(map(sum, st))
        print(f"{p:20s} {v:20s} {n:4d} {fl:5d}  {fl / n:6.2f}  {sf / max(fl, 1):12.2f}  {len(st):3d} of {len(d):3d}"
              f"  {(fl - sf) / max(n - sn, 1):6.2f}")


def done(rd, ps):
    """' (n of m done)' while a run is partial, so a partial figure is not read as final."""
    n = len({r["paper"] for r in load(rd) if r["paper"] in ps})
    return "" if n == len(ps) else f"\n({n} of {len(ps)} done)"


def figure(rds):
    """Share of planted errors caught, per reader and paper group; false alarms per paragraph and per 1,000 words."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from averroes_figstyle import axes_style, bar, legend, plt, save, swatch
    cols = [(rd, g, ps) for rd in rds for g, ps in GROUPS if any(r["paper"] in ps for r in load(rd))]
    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(max(10, 2.3 * len(cols)), 14))   # stacked: up to 8 columns
    axes_style(a1, 0, 1, ylabel="Share of planted errors caught")
    axes_style(a2, 0, 0.6, ylabel="False alarms per paragraph")
    axes_style(a3, 0, 8, ylabel="False alarms per 1,000 words")
    for i, (rd, g, ps) in enumerate(cols):
        rs = [r for r in load(rd) if r["paper"] in ps]
        pe = caught_by_error(rs)
        b = boot(pe) if pe else None
        for j, (v, h, _) in enumerate(BARS):
            if v not in vers(rs):
                continue
            if b:
                bar(a1, i + (j - 1) * 0.27, 0.25, b[v], ci=b[v + ":ci"], label="{:.2f}", hatch=h)
            bar(a2, i + (j - 1) * 0.27, 0.25, fa_rate(rs, v), ci=fa_boot(rs, v), hi=0.6, label="{:.2f}", hatch=h)
            bar(a3, i + (j - 1) * 0.27, 0.25, fa_rate(rs, v, True), ci=fa_boot(rs, v, True), hi=8, label="{:.2f}", hatch=h)
    for ax in (a1, a2, a3):
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([f"{READERS[rd][0]} {READERS[rd][1]}\n{g.replace(' papers', chr(10) + 'papers')}" + done(rd, ps) for rd, g, ps in cols], fontsize=10)
        ax.set_xlim(-0.55, max(len(cols), 1) - 0.45)
    legend(fig, [swatch(hatch=h) for _, h, _ in BARS], [n for _, _, n in BARS], y=1.02)
    fig.subplots_adjust(bottom=0.06, top=0.93, hspace=0.45)
    save(fig, FIG)


def main():
    a = sys.argv[1:]
    if a[:1] == ["--count"]:
        for p in err9.PAPERS + ANCHORS:
            n = len(primary(p))
            per = {v: len(paragraphs(err9.text(p, v))) for v in VERS}
            print(f"{p:20s} errors {n:2d}  paragraphs {per}  questions {sum(c * (n + 1) for c in per.values())}")
    elif a[:1] == ["read"]:
        vs = a[a.index("--versions") + 1].split(",") if "--versions" in a else VERS
        papers = [x for x in a[2:] if not x.startswith("--") and x not in (",".join(vs),)]
        read(a[1], papers, "--clean-only" in a, vs)
    elif a[:1] == ["batch"]:
        left, blocked, npend = batch_round("--slice" in a)
        if not left:
            print("ALL DONE")
        elif blocked and not npend:
            print("STOPPED: cap reached with nothing in flight")
    elif a[:1] == ["--scores"]:
        rds = a[1:] or [r for r in READERS if (OUT / f"{r}.jsonl").exists()]
        for rd in rds:
            scores(rd)
        figure(rds)
    elif a[:1] == ["--elsewhere"]:
        elsewhere(a[1])
    elif a[:1] == ["--headsplit"]:
        headsplit(a[1], "--count" in a)
    elif a[:1] == ["--regroup"]:
        regroup(a[1])
    elif a[:1] == ["--stable"]:
        stable(a[1])
    elif a[:1] == ["--new-sample"]:
        new_sample(a[1], *map(int, a[2:4]))
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
