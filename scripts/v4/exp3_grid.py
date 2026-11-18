"""Plan v4 experiment 3: the coarse grid. Weak readers review planted copies; a grader decides each catch.

Stages:
  review  - each reader setting of experiment 0 reviews every surviving error's planted copy (4 copies, 2 samples),
            with experiment 0's review prompt (exp0.REVIEW_SYSTEM / REVIEW_USER, unchanged). Also clean reviews of
            the clarified copies (5 readers x 2 samples x 3 papers), which experiment 0 did not have.
  grade   - gpt-5-mini (low effort) grades each planted-copy review against its planted error with experiment 2's
            grading prompt (exp2.GRADE_USER, unchanged). For the false-accusation baseline it grades every clean
            review (experiment 0's orig/obf/compobf reviews and the new clarified ones) against each surviving error
            of that paper, using that copy's planted text: a "catch" there is an accusation of an error that is absent.
  check   - gpt-6.1-sol (low effort) regrades 50 random planted-copy reviews with the same prompt (seed 0), to check
            the gpt-5-mini grader.
Errors: the 19 that survived experiment 2 (preregistered set), plus, as an exploratory set, the 3 errors that failed
only because the strong reader missed them in an illegible or original copy (clarified caught, pair check passed).
Output in data/v4/exp3-grid/: calls.jsonl (raw, cached), reviews.jsonl, grades.jsonl. Rerunning resumes from the cache.
Usage:
  python scripts/v4/exp3_grid.py --slice    # one error x 4 copies, all readers, 1 sample, graded
  python scripts/v4/exp3_grid.py            # everything
  python scripts/v4/exp3_grid.py --figure   # docs/figures/v4-exp3-grid.png
"""
import json, random, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from llm import Calls, ROOT, spent
import exp0_clean_reviews as exp0
import exp2_errors as exp2

OUT = ROOT / "data" / "v4" / "exp3-grid"
COPIES = exp2.COPIES
READERS = exp0.READERS
SAMPLES = 2
GRADER = ("gpt-5-mini", "low")
CHECKER = ("gpt-6.1-sol", "low")
N_CHECK = 50


def load_errors():
    rows = [json.loads(l) for l in (ROOT / "data" / "v4" / "exp2-errors" / "errors.jsonl").read_text().splitlines()]
    for r in rows:
        r["set"] = ("prereg" if r["survives"] else
                    "exploratory" if (not r["problems"] and r["pair"]["equal"] is True
                                      and r["copies"]["clarified"]["strong"]["caught"]) else None)
    return [r for r in rows if r["set"]]


def planted(err, c):
    return " ... ".join(e["new"] for e in err["copies"][c]["edits"])


def review(calls, reader, text, tag, s, max_tokens=16000):
    label, model, effort = reader
    try:
        rec = calls.call(model, exp0.REVIEW_SYSTEM, exp0.REVIEW_USER.format(paper=text), tag=tag, sample=s,
                         effort=effort, max_tokens=max_tokens)
    except RuntimeError as e:
        return {"failed": str(e)}
    errs = exp0.parse(rec["content"])
    return {"model_returned": rec["model_returned"], "parsed": errs is not None, "errors": errs,
            "n_errors": len(errs) if errs is not None else None, "cost": rec["cost"]}


def grade(calls, err, c, errors, who=GRADER, tag="exp3:grade"):
    if not errors:      # failed, unparsed or empty review: nothing to grade
        return {"caught": False if errors == [] else None, "grade": None}
    try:
        g = calls.call(who[0], exp2.SYSTEM, exp2.GRADE_USER.format(
            false_claim=err["false_claim"], refutation=err["refutation"], planted=planted(err, c),
            review=json.dumps(errors, indent=1)), tag=tag, effort=who[1], max_tokens=8000)
    except RuntimeError as e:
        return {"caught": None, "grade": None, "note": f"failed: {e}"}
    d = exp2.parse(g["content"], "caught")
    return {"caught": d["caught"] if d else None, "grade": d}


def main():
    sl = "--slice" in sys.argv
    OUT.mkdir(parents=True, exist_ok=True)
    calls = Calls(OUT / "calls.jsonl")
    errs = load_errors()
    if sl:
        errs = [r for r in errs if r["set"] == "prereg"][:1]
    samples = 1 if sl else SAMPLES

    # Reviews of planted copies.
    jobs = [(r, c, rd, s) for r in errs for c in COPIES for rd in READERS for s in range(samples)]

    def run(j):
        r, c, rd, s = j
        rev = review(calls, rd, exp2.planted_text(r["paper"], c, r), f"exp3:review:{rd[0]}:{r['paper']}:{c}:{r['id']}", s)
        out = {"kind": "planted", "set": r["set"], "reader": rd[0], "paper": r["paper"], "error": r["id"],
               "type": r["type"], "copy": c, "sample": s, **rev}
        out.update(grade(calls, r, c, rev.get("errors"), tag=f"exp3:grade:{rd[0]}:{r['paper']}:{c}:{r['id']}"))
        return out

    # Clean reviews of the clarified copies (new), and of orig/obf/compobf (experiment 0), graded against each error.
    papers = sorted({r["paper"] for r in errs})
    clean = []
    if not sl:
        for p in papers:
            text = exp2.source(p, "clarified")
            for rd in READERS:
                for s in range(SAMPLES):
                    clean.append((p, "clarified", rd, s, text))
    exp0_rows = [json.loads(l) for l in (exp0.OUT / "reviews.jsonl").read_text().splitlines()]

    def run_clean(j):
        p, c, rd, s, text = j
        rev = review(calls, rd, text, f"exp3:clean:{rd[0]}:{p}:{c}", s)
        return {"kind": "clean", "reader": rd[0], "paper": p, "copy": c, "sample": s, **rev}

    with ThreadPoolExecutor(12) as ex:
        rows = list(ex.map(run, jobs))
        new_clean = list(ex.map(run_clean, clean))
        cleans = new_clean + ([] if sl else [
            {"kind": "clean", "reader": r["reader"], "paper": r["paper"], "copy": r["copy"], "sample": r["sample"],
             "errors": r.get("errors"), "source": "exp0"} for r in exp0_rows if r["paper"] in papers])
        gjobs = [(cr, r) for cr in cleans for r in errs if r["paper"] == cr["paper"]]
        cgrades = list(ex.map(lambda j: {
            "kind": "clean", "set": j[1]["set"], "reader": j[0]["reader"], "paper": j[0]["paper"],
            "error": j[1]["id"], "type": j[1]["type"], "copy": j[0]["copy"], "sample": j[0]["sample"],
            **grade(calls, j[1], j[0]["copy"], j[0].get("errors"),
                    tag=f"exp3:cgrade:{j[0]['reader']}:{j[0]['paper']}:{j[0]['copy']}:{j[1]['id']}")}, gjobs))

        # Grader check: Sol regrades a random sample of planted-copy reviews that have something to grade.
        pool = [x for x in rows if x.get("errors") and x["set"] == "prereg"]
        pick = random.Random(0).sample(pool, min(N_CHECK if not sl else 5, len(pool)))
        byid = {(r["paper"], r["id"]): r for r in errs}
        checks = list(ex.map(lambda x: grade(calls, byid[(x["paper"], x["error"])], x["copy"], x["errors"], CHECKER,
                                             tag=f"exp3:check:{x['reader']}:{x['paper']}:{x['copy']}:{x['error']}"),
                             pick))
    for x, chk in zip(pick, checks):
        x["sol_caught"] = chk["caught"]
        x["sol_grade"] = chk["grade"]

    sfx = "-slice" if sl else ""
    (OUT / f"reviews{sfx}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows + new_clean))
    (OUT / f"clean-grades{sfx}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in cgrades))
    summary(rows, cgrades)
    print(f"ledger total ${spent():.2f}")


def summary(rows, cgrades):
    for rd in READERS:
        line = []
        for c in COPIES:
            v = [x["caught"] for x in rows if x["reader"] == rd[0] and x["copy"] == c and x["set"] == "prereg"]
            fa = [x["caught"] for x in cgrades if x["reader"] == rd[0] and x["copy"] == c and x["set"] == "prereg"]
            line.append(f"{c}: {sum(map(bool, v))}/{len(v)} (fa {sum(map(bool, fa))}/{len(fa)})")
        print(f"{rd[0]:9s} " + "  ".join(line))
    chk = [x for x in rows if "sol_caught" in x and x["caught"] is not None and x["sol_caught"] is not None]
    print(f"grader check: agree {sum(x['caught'] == x['sol_caught'] for x in chk)}/{len(chk)}")
    nf = sum(x.get("caught") is None and x.get("errors") != [] for x in rows)
    print(f"planted reviews ungraded (failed or unparsed): {nf}/{len(rows)}")


def cells(set_="prereg"):
    """Per (reader, type, measure): (rate, lo, hi, n_errors). Bootstrap 95% over errors, 2000 resamples, seed 0.
    Measures: clarified, illegible (obf + compobf pooled), orig, accuse (clean reviews, all four copies)."""
    rows = [json.loads(l) for l in (OUT / "reviews.jsonl").read_text().splitlines()]
    cg = [json.loads(l) for l in (OUT / "clean-grades.jsonl").read_text().splitlines()]
    copies = {"clarified": ["clarified"], "illegible": ["obf", "compobf"], "orig": ["orig"]}
    out, rng = {}, random.Random(0)
    for rd in READERS:
        for t in exp2.TYPES:
            for m in ["clarified", "illegible", "orig", "accuse"]:
                src = cg if m == "accuse" else [x for x in rows if x["kind"] == "planted" and x["copy"] in copies[m]]
                by = {}
                for x in src:
                    if x["reader"] == rd[0] and x["type"] == t and x["set"] == set_ and x["caught"] is not None:
                        by.setdefault((x["paper"], x["error"]), []).append(bool(x["caught"]))
                ks = list(by)
                if not ks:
                    out[rd[0], t, m] = (float("nan"),) * 3 + (0,)
                    continue
                rate = lambda kk: sum(sum(by[k]) for k in kk) / sum(len(by[k]) for k in kk)
                boots = sorted(rate([rng.choice(ks) for _ in ks]) for _ in range(2000))
                out[rd[0], t, m] = (rate(ks), boots[50], boots[1949], len(ks))
    return out


def figure():
    """Measured grid in the layout of the predicted one (docs/figures/v4-exp3-grid.png), and a markdown table."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import calibration_table as ct
    c = cells()
    types = list(exp2.TYPES)
    ct.WINDOW_LABEL = "Meets proposed bars"
    ct.NOTE = "Measured: 19 planted errors on 3 papers, 2 samples per copy; bars are shares, outlines use the proposed bars."
    ct.figure(lambda r, k: tuple(c[READERS[r][0], types[k], m][0] for m in ("clarified", "illegible", "accuse")),
              str(ROOT / "docs" / "figures" / "v4-exp3-grid.png"))
    print("| reader | type | errors | clarified | illegible (obf+compobf) | original | false accusation |")
    print("|---|---|---|---|---|---|---|")
    f = lambda v: f"{v[0]:.2f} [{v[1]:.2f}, {v[2]:.2f}]"
    for rd in READERS:
        for t in types:
            print(f"| {rd[0]} | {t} | {c[rd[0], t, 'clarified'][3]} | " +
                  " | ".join(f(c[rd[0], t, m]) for m in ("clarified", "illegible", "orig", "accuse")) + " |")


if __name__ == "__main__":
    figure() if "--figure" in sys.argv else main()
