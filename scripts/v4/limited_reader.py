"""Limited-view reader with saved notes (plan v4 next step 1c, for experiment 9).

The reader is experiment 8's, unchanged: the same prompts, chunking, notes cap, JSON parsing and one retry, imported
from exp8_limited, so a step's cache key is identical to experiment 8's. What changes is how planted copies are read.
`read_clean` reads a clean version once per sample and saves the notes before every chunk (and the final notes).
A planted copy is identical to its clean version up to the first chunk the planted edit changes, chunk k. Reading it
from the start would repeat steps 1..k-1 of a read of the same text, so `read_planted` instead starts at chunk k with
the clean read's notes from before chunk k and reads on to the end. Its review is the clean read's findings from steps
1..k-1 (marked "from_clean") plus its own from step k on. The prefix is the same text with the same prompts, so the
resumed read has the same distribution as a full read. Errors in the same version and sample share everything before
their chunk, so a bootstrap must resample by error (as before) and treat the clean read as shared.
If the planted copy has a different number of chunks (the edit moved a chunk boundary), "part k of n" would differ in
every prompt, so the copy is read from the start instead ("resumed_at": 1) and counted.

Usage (validation against experiment 8, variant A, gpt-5-mini high, 1 sample; output in data/v4/limited-reader-check/):
  python scripts/v4/limited_reader.py --check [--slice]
"""
import hashlib, json, os, re, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from llm import Calls, ROOT, spent
import exp8_limited as exp8          # importing it points exp2.source at the repaired and scatter copies
import exp2_errors as exp2
import exp3_grid as exp3
import exp3r_repaired as exp3r

FIRST_NOTES = "(none yet: this is the first part)"


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def step(calls, reader, variant, k, n, notes, chunk, tag, s):
    """One step of experiment 8's reader (same prompts, retry and parsing as exp8.read). Returns (step record, notes
    after the step)."""
    model, effort = exp8.READERS[reader]
    cap = exp8.VARIANTS[variant][2]
    user = exp8.STEP_USER.format(k=k, n=n, notes=notes, chunk=chunk, cap=cap)
    d, rec, note = None, None, None
    for attempt in range(2):
        try:
            rec = calls.call(model, exp8.STEP_SYSTEM.format(notes=cap), user, tag=f"{tag}:step{k}",
                             sample=s + 100 * attempt, effort=effort, max_tokens=exp8.MAX_TOKENS)
        except RuntimeError as e:
            note = f"failed: {e}"
            continue
        m = re.search(r"\{.*\}", rec["content"], re.S)
        d = exp8.parse_json(m.group(0)) if m else None
        if not isinstance(d, dict) or not isinstance(d.get("errors"), list):
            d = None
        if d is not None:
            break
        note = "unparsed"
    st = {"k": k, "chunk_sha": sha(chunk), "chunk_words": exp8.words(chunk), "ok": d is not None,
          "note": None if d else note, "cost": rec["cost"] if rec else 0, "notes_before": notes, "errors": []}
    if d is None:
        return st, notes
    st["errors"] = [{**e, "step": k} for e in d["errors"] if isinstance(e, dict)]
    new = str(d.get("notes") or "")
    st["notes_words"] = exp8.words(new)
    st["notes_cut"] = exp8.words(new) > exp8.SLACK * cap
    if st["notes_cut"]:
        new = " ".join(new.split()[:int(exp8.SLACK * cap)])
    return st, (new if new.strip() else notes)


def _run(calls, reader, variant, ch, k0, notes, tag, s):
    steps = []
    for k in range(k0, len(ch) + 1):
        st, notes = step(calls, reader, variant, k, len(ch), notes, ch[k - 1], tag, s)
        steps.append(st)
    return steps, notes


def summary(steps, text, final_notes, **extra):
    errors = [e for st in steps for e in st["errors"]]
    return {"errors": errors, "n_errors": len(errors), "parsed": all(x["ok"] for x in steps),
            "steps_failed": sum(not x["ok"] for x in steps), "chunks": len(steps),
            "notes_cut": sum(bool(x.get("notes_cut")) for x in steps), "words": exp8.words(text),
            "cost": sum(x["cost"] for x in steps), "final_notes": final_notes, "steps": steps, **extra}


def read_clean(calls, reader, variant, text, tag, s):
    """Read a clean version in full. The record keeps every step's notes_before and the final notes."""
    ch = exp8.chunks(text, variant)
    steps, notes = _run(calls, reader, variant, ch, 1, FIRST_NOTES, tag, s)
    return summary(steps, text, notes)


def read_planted(calls, reader, variant, clean, text, tag, s):
    """Read a planted copy, resuming from `clean` (read_clean's record for the same version, reader, variant and
    sample) at the first chunk that differs. Steps actually called are those from resumed_at on."""
    ch = exp8.chunks(text, variant)
    old = clean["steps"]
    fallback = len(ch) != len(old) or not all(x["ok"] for x in old)
    if fallback:
        k0 = 1
    else:
        k0 = next((i + 1 for i, c in enumerate(ch) if sha(c) != old[i]["chunk_sha"]), None)
        if k0 is None:
            raise ValueError(f"{tag}: planted copy has the same chunks as the clean version")
    notes = old[k0 - 1]["notes_before"] if k0 > 1 else FIRST_NOTES
    steps, notes = _run(calls, reader, variant, ch, k0, notes, tag, s)
    prefix = [{**x, "errors": [{**e, "from_clean": True} for e in x["errors"]], "cost": 0} for x in old[:k0 - 1]]
    return summary(prefix + steps, text, notes, resumed_at=k0, fallback=fallback, steps_called=len(steps))


# ---------- validation against experiment 8 (variant A, mini-high, sample 0) ----------

def check():
    sl = "--slice" in sys.argv
    out = ROOT / "data" / "v4" / "limited-reader-check"
    out.mkdir(parents=True, exist_ok=True)
    calls = Calls(out / "calls.jsonl")
    # Seed the cache in memory from experiment 8's log, so the clean reads replay experiment 8's clean reads for
    # free (sample 0). Replayed calls are not copied into this run's log; they are in data/v4/exp8-limited/calls.jsonl.
    for line in (exp8.OUT / "calls.jsonl").open():
        r = json.loads(line)
        if "content" in r:
            calls.cache[r["key"]] = r      # last wins, as in Calls (exp8 has 40 keys logged twice by concurrent calls)
    v, rd, s = "A", "mini-high", 0
    errs = exp8.load_errors()
    copies = ["clarified", "orig", "obf", "compobf"]
    if sl:
        errs = errs[:1]
    pc = sorted({(r["paper"], c) for r in errs for c in copies})
    with ThreadPoolExecutor(int(os.environ.get("V4_CONCURRENCY", "12"))) as ex:
        cleans = dict(zip(pc, ex.map(lambda x: read_clean(calls, rd, v, exp2.source(*x), f"exp8{v}:clean:{rd}:{x[0]}:{x[1]}", s), pc)))

        def run(j):
            r, c = j
            rev = read_planted(calls, rd, v, cleans[(r["paper"], c)], exp2.planted_text(r["paper"], c, r),
                               f"lrcheck{v}:read:{rd}:{r['paper']}:{c}:{r['id']}", s)
            row = {"kind": "planted", "variant": v, "reader": rd, "paper": r["paper"], "error": r["id"], "type": r["type"],
                   "copy": c, "sample": s, **rev}
            row.update(exp3.grade(calls, r, c, rev["errors"], tag=f"lrcheck{v}:grade:{rd}:{r['paper']}:{c}:{r['id']}"))
            return row
        revs = list(ex.map(run, [(r, c) for r in errs for c in copies]))
    # Replay check: the clean reads must match experiment 8's clean reviews exactly (same findings).
    e8 = {(x["paper"], x["copy"]): x for x in exp3r.load(exp8.OUT / "reviews-A.jsonl")
          if x["kind"] == "clean" and x["reader"] == rd and x["sample"] == s}
    same = sum(json.dumps([e for e in cleans[k]["errors"]], sort_keys=True) ==
               json.dumps(e8[k]["errors"], sort_keys=True) for k in pc if k in e8)
    print(f"replay: {same}/{len(pc)} clean reads identical to experiment 8's")
    for k in pc:
        cleans[k] = {"paper": k[0], "copy": k[1], "reader": rd, "variant": v, "sample": s, "kind": "clean", **cleans[k]}
    (out / "reviews.jsonl").write_text("".join(json.dumps(x) + "\n" for x in list(cleans.values()) + revs))
    full = sum(x["chunks"] for x in revs)
    called = sum(x["steps_called"] for x in revs)
    print(f"planted reads {len(revs)}: steps called {called} of {full} ({called / full:.0%}); edit in chunk 1 "
          f"{sum(x['resumed_at'] == 1 and not x['fallback'] for x in revs)}; read from start because the chunk count "
          f"changed {sum(x['fallback'] for x in revs)}; steps failed {sum(x['steps_failed'] for x in revs)}")
    # Experiment 8's sample-0 planted reads shared the clean prefix through the call cache whenever the prefix steps
    # matched, so most resumed reads replay them; compare per row with sample 0, and with both samples.
    e8p = {(x["paper"], x["error"], x["copy"], x["sample"]): x for x in exp3r.load(exp8.OUT / "reviews-A.jsonl")
           if x["kind"] == "planted" and x["reader"] == rd}
    for c in copies:
        a = [x for x in revs if x["copy"] == c]
        b0 = [e8p[(x["paper"], x["error"], c, 0)] for x in a]
        b1 = [e8p[(x["paper"], x["error"], c, 1)] for x in a]
        print(f"{c:9s} resumed {sum(bool(x['caught']) for x in a)}/{len(a)}   exp8 sample 0 "
              f"{sum(bool(x['caught']) for x in b0)}/{len(b0)} (same grade {sum(bool(x['caught']) == bool(y['caught']) for x, y in zip(a, b0))}"
              f"/{len(a)})   exp8 sample 1 {sum(bool(x['caught']) for x in b1)}/{len(b1)}")
    keys = {r["key"] for r in map(json.loads, (out / "calls.jsonl").open()) if "content" in r}
    print(f"new calls in this run's log {len(keys)} (cost in the ledger); ledger total ${spent():.2f}")

if __name__ == "__main__":
    if "--check" in sys.argv:
        check()
    else:
        print(__doc__)
