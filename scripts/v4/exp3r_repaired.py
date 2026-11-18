"""Plan v4 experiment 3r (Jason's repair note): experiment 3 rerun on the repaired obf/compobf copies.

Preregistration: docs/plan-v4.md, "Experiment 3r preregistration". Stages:
  replant - for each of experiment 3's 22 errors and each repaired illegible copy: reuse V1's edits if every old
            text occurs once in the repaired copy and every evidence quote is found there; otherwise re-plant with
            experiment 2's plant() on the repaired copy.
  checks  - experiment 2's strong reader on the repaired obf/compobf, and its pair check over all four copies
            (clarified and orig planted copies, and their strong-reader results, reused).
  readers - experiment 3's readers (2 samples) on the repaired planted copies, graded by gpt-5-mini; clean reviews of
            the repaired copies (3 samples), graded against each error; Sol regrades 30 random planted reviews.
Output in data/v4/exp3r-repaired/: calls.jsonl (raw, cached), errors.jsonl, reviews.jsonl, clean-grades.jsonl.
Usage:
  python scripts/v4/exp3r_repaired.py --slice    # first paper only
  python scripts/v4/exp3r_repaired.py            # all
  python scripts/v4/exp3r_repaired.py --figure   # docs/figures/v4-exp3r-repaired.png and comparison table
"""
import copy as cp, json, random, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from llm import Calls, ROOT, spent
import exp0_clean_reviews as exp0
import exp2_errors as exp2
import exp3_grid as exp3

OUT = ROOT / "data" / "v4" / "exp3r-repaired"
REP = ROOT / "data" / "v4" / "repaired"
ILL = ["obf", "compobf"]
READERS = exp3.READERS
CLEAN_SAMPLES = 3
N_CHECK = 30

_v1_source = exp2.source


def source(p, c):
    return (REP / f"{p}.{c}.md").read_text() if c in ILL else _v1_source(p, c)


exp2.source = source     # every exp2/exp3 function below now reads the repaired obf/compobf


def replant(calls, p, errs):
    """New error rows for paper p, with obf/compobf edits valid on the repaired copies."""
    avoid = exp2.avoid_list(p)
    rows = [cp.deepcopy(e) for e in errs]
    for r in rows:
        r.pop("pair", None)
        r["replanted"] = []
    for c in ILL:
        src = source(p, c)
        redo = [r["id"] for r in rows if exp2.apply(src, r["copies"][c]["edits"])[1] is not None
                or not all(exp2.found(src, q) for q in r["copies"][c]["evidence"])]
        new = {}
        if redo:
            spec = [{**{k: r[k] for k in ("id", "type", "false_claim", "refutation")},
                     "edits": r["copies"]["clarified"]["edits"]} for r in rows if r["id"] in redo]
            new = exp2.plant(calls, p, c, spec)
        for r in rows:
            r["problems"] = [x for x in r["problems"] if not x.startswith(c + ":")]
            if r["id"] in redo:
                r["replanted"].append(c)
                cc = new.get(r["id"])
                if cc is None or cc.get("status") == "missing":
                    r["problems"].append(f"{c}: missing ({(cc or {}).get('note', 'no entry')})")
                    continue
                r["copies"][c] = {"edits": cc["edits"], "evidence": cc["evidence"]}
            cc = r["copies"][c] = {"edits": r["copies"][c]["edits"], "evidence": r["copies"][c]["evidence"]}
            prob = exp2.apply(src, cc["edits"])[1]
            if prob:
                r["problems"].append(f"{c}: {prob}")
            cc["evidence_found"] = [exp2.found(src, q) for q in cc["evidence"]]
            cc["touches_avoid"] = [q for q in avoid if any(q in ed["old"] or ed["old"] in q for ed in cc["edits"])]
            if cc["touches_avoid"]:
                r["problems"].append(f"{c}: edits a passage on the avoid list")
    return rows


def checks(calls, rows):
    ok = [r for r in rows if not r["problems"]]
    jobs = [(r, c) for r in ok for c in ILL]
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(lambda j: exp2.strong(calls, j[0]["paper"], j[1], j[0]), jobs))
        prs = list(ex.map(lambda r: exp2.pair(calls, r["paper"], r), ok))
    for (r, c), s in zip(jobs, res):
        r["copies"][c]["strong"] = s
    for r, pr in zip(ok, prs):
        r["pair"] = pr
    for r in rows:
        passed = (not r["problems"] and r["pair"]["equal"] is True
                  and all(r["copies"][c]["strong"]["caught"] for c in exp2.COPIES))
        r["set_v1"] = r["set"]
        r["set"] = "prereg" if (passed and r["set_v1"] == "prereg") else "exploratory" if not r["problems"] else None
    return rows


def main():
    sl = "--slice" in sys.argv
    OUT.mkdir(parents=True, exist_ok=True)
    calls = Calls(OUT / "calls.jsonl")
    v1 = exp3.load_errors()
    papers = sorted({r["paper"] for r in v1})[:1 if sl else None]
    with ThreadPoolExecutor(3) as ex:
        rows = [r for rs in ex.map(lambda p: replant(calls, p, [e for e in v1 if e["paper"] == p]), papers) for r in rs]
    rows = checks(calls, rows)
    sfx = "-slice" if sl else ""
    (OUT / f"errors{sfx}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    for r in rows:
        st = " ".join(f"{c}:{'Y' if r['copies'][c].get('strong', {}).get('caught') else 'n'}" for c in ILL)
        print(f"{r['paper']} {r['id']:18s} v1={r['set_v1']} now={r['set']} replanted={r['replanted']} {st} "
              f"pair={r.get('pair', {}).get('equal')} {'; '.join(r['problems'])}")
    errs = [r for r in rows if r["set"]]

    jobs = [(r, c, rd, s) for r in errs for c in ILL for rd in READERS for s in range(exp3.SAMPLES)]

    def run(j):
        r, c, rd, s = j
        rev = exp3.review(calls, rd, exp2.planted_text(r["paper"], c, r), f"exp3r:review:{rd[0]}:{r['paper']}:{c}:{r['id']}", s)
        out = {"kind": "planted", "set": r["set"], "reader": rd[0], "paper": r["paper"], "error": r["id"],
               "type": r["type"], "copy": c, "sample": s, **rev}
        out.update(exp3.grade(calls, r, c, rev.get("errors"), tag=f"exp3r:grade:{rd[0]}:{r['paper']}:{c}:{r['id']}"))
        return out

    clean = [(p, c, rd, s) for p in papers for c in ILL for rd in READERS for s in range(CLEAN_SAMPLES)]

    def run_clean(j):
        p, c, rd, s = j
        rev = exp3.review(calls, rd, source(p, c), f"exp3r:clean:{rd[0]}:{p}:{c}", s)
        return {"kind": "clean", "reader": rd[0], "paper": p, "copy": c, "sample": s, **rev}

    with ThreadPoolExecutor(12) as ex:
        revs = list(ex.map(run, jobs))
        cleans = list(ex.map(run_clean, clean))
        gjobs = [(cr, r) for cr in cleans for r in errs if r["paper"] == cr["paper"]]
        cgrades = list(ex.map(lambda j: {
            "kind": "clean", "set": j[1]["set"], "reader": j[0]["reader"], "paper": j[0]["paper"],
            "error": j[1]["id"], "type": j[1]["type"], "copy": j[0]["copy"], "sample": j[0]["sample"],
            **exp3.grade(calls, j[1], j[0]["copy"], j[0].get("errors"),
                         tag=f"exp3r:cgrade:{j[0]['reader']}:{j[0]['paper']}:{j[0]['copy']}:{j[1]['id']}")}, gjobs))
        pool = [x for x in revs if x.get("errors") and x["set"] == "prereg"]
        pick = random.Random(0).sample(pool, min(N_CHECK, len(pool)))
        byid = {(r["paper"], r["id"]): r for r in errs}
        chk = list(ex.map(lambda x: exp3.grade(calls, byid[(x["paper"], x["error"])], x["copy"], x["errors"],
                                               exp3.CHECKER, tag=f"exp3r:check:{x['reader']}:{x['paper']}:{x['copy']}:{x['error']}"),
                          pick))
    for x, g in zip(pick, chk):
        x["sol_caught"], x["sol_grade"] = g["caught"], g["grade"]
    (OUT / f"reviews{sfx}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in revs + cleans))
    (OUT / f"clean-grades{sfx}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in cgrades))
    for rd in READERS:
        print(f"{rd[0]:9s} " + "  ".join(
            f"{c}: {sum(bool(x['caught']) for x in revs if x['reader'] == rd[0] and x['copy'] == c and x['set'] == 'prereg')}"
            f"/{sum(x['reader'] == rd[0] and x['copy'] == c and x['set'] == 'prereg' for x in revs)}" for c in ILL))
    ck = [x for x in revs if "sol_caught" in x and x["caught"] is not None and x["sol_caught"] is not None]
    print(f"grader check: agree {sum(x['caught'] == x['sol_caught'] for x in ck)}/{len(ck)}")
    print(f"planted reviews ungraded: {sum(x.get('caught') is None and x.get('errors') != [] for x in revs)}/{len(revs)}")
    print(f"ledger total ${spent():.2f}")


def load(path):
    return [json.loads(l) for l in path.read_text().splitlines()]


def compare():
    """Per reader, pooled over types, on errors in both preregistered sets: catch rates per copy (V1 and repaired),
    repaired minus V1 differences with paired bootstrap 95% intervals over errors (2000 resamples, seed 0), and
    false-accusation rates. Returns {reader: {measure: (value, lo, hi)}} and the common error list."""
    rep = {(r["paper"], r["id"]) for r in load(OUT / "errors.jsonl") if r["set"] == "prereg"}
    v1 = {(r["paper"], r["id"]) for r in exp3.load_errors() if r["set"] == "prereg"}
    common = sorted(rep & v1)
    old = [x for x in load(exp3.OUT / "reviews.jsonl") if x["kind"] == "planted"]
    new = [x for x in load(OUT / "reviews.jsonl") if x["kind"] == "planted"]
    fa_old, fa_new = load(exp3.OUT / "clean-grades.jsonl"), load(OUT / "clean-grades.jsonl")
    measures = {"clarified": (old, ["clarified"]), "orig": (old, ["orig"]),
                "obf V1": (old, ["obf"]), "obf repaired": (new, ["obf"]),
                "compobf V1": (old, ["compobf"]), "compobf repaired": (new, ["compobf"]),
                "illegible V1": (old, ILL), "illegible repaired": (new, ILL),
                "fa V1": (fa_old, ILL), "fa repaired": (fa_new, ILL)}
    out, rng = {}, random.Random(0)
    for rd in READERS:
        by = {}
        for m, (src, cs) in measures.items():
            for x in src:
                k = (x["paper"], x["error"])
                if x["reader"] == rd[0] and x["copy"] in cs and k in common and x["caught"] is not None:
                    by.setdefault(m, {}).setdefault(k, []).append(bool(x["caught"]))
        rate = lambda m, ks: sum(sum(by[m].get(k, [])) for k in ks) / max(1, sum(len(by[m].get(k, [])) for k in ks))
        boots = [[rng.choice(common) for _ in common] for _ in range(2000)]
        res = {}
        for m in measures:
            v = sorted(rate(m, b) for b in boots)
            res[m] = (rate(m, common), v[50], v[1949])
        for c in ["obf", "compobf", "illegible", "fa"]:
            d = sorted(rate(f"{c} repaired", b) - rate(f"{c} V1", b) for b in boots)
            res[f"{c} diff"] = (rate(f"{c} repaired", common) - rate(f"{c} V1", common), d[50], d[1949])
        out[rd[0]] = res
    return out, common


def figure():
    sys.path.insert(0, str(ROOT / "scripts"))
    from averroes_figstyle import HATCHES, axes_style, bar, legend, save, swatch, plt
    res, common = compare()
    ms = ["clarified", "obf V1", "obf repaired", "compobf V1", "compobf repaired"]
    hs = [HATCHES[0], HATCHES[1], HATCHES[2], HATCHES[3], HATCHES[4]]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    axes_style(ax, 0, 1, ylabel="catch rate")
    w = 0.16
    for i, rd in enumerate(READERS):
        for j, m in enumerate(ms):
            v = res[rd[0]][m]
            bar(ax, i + (j - 2) * w, w * 0.92, v[0], ci=(v[1], v[2]), hatch=hs[j], label=None)  # values in the table
    ax.set_xticks(range(len(READERS)), [r[0] for r in READERS])
    ax.set_xlim(-0.5, len(READERS) - 0.5)
    legend(fig, [swatch(hatch=h) for h in hs], ["clarified", "jargon (obf), V1", "jargon (obf), repaired",
                                                 "compressed (compobf), V1", "compressed (compobf), repaired"],
           y=1.1, ncol=3)
    save(fig, ROOT / "docs" / "figures" / "v4-exp3r-repaired.png")
    f = lambda v: f"{v[0]:.2f} [{v[1]:.2f}, {v[2]:.2f}]"
    print(f"{len(common)} errors in both preregistered sets")
    print("| reader | clarified | obf V1 | obf repaired | compobf V1 | compobf repaired | illegible: repaired - V1 "
          "| clarified - illegible, V1 | clarified - illegible, repaired | false acc. V1 | false acc. repaired |")
    print("|---" * 11 + "|")
    for rd in READERS:
        r = res[rd[0]]
        print(f"| {rd[0]} | " + " | ".join(f(r[m]) for m in ms + ["illegible diff"]) +
              f" | {r['clarified'][0] - r['illegible V1'][0]:.2f} | {r['clarified'][0] - r['illegible repaired'][0]:.2f}"
              f" | {r['fa V1'][0]:.2f} | {r['fa repaired'][0]:.2f} |")


if __name__ == "__main__":
    figure() if "--figure" in sys.argv else main()
