"""Plan v4 experiment 7: several planted errors per copy.

Preregistration: docs/plan-v4.md, "Experiment 7 preregistration". Per paper, the errors of the experiment 6 set
(`set6`) are planted together into one copy of each kind (clarified, repaired obf, repaired compobf, scatter), in id
order; an error is skipped if its edits no longer apply exactly once, or if any error's evidence that is found in
its single-planted copy is no longer found in the joint copy, in any of the four copies. Readers luna-low and mini-high review each multi-planted copy
(SAMPLES samples); each review is graded against each planted error with experiment 3's grader.
Output in data/v4/exp7-multi/: calls.jsonl (raw, cached), plan.jsonl (which errors are combined per paper),
reviews.jsonl (one row per review, with a grade per planted error).
Usage:
  python scripts/v4/exp7_multi.py --slice    # 2609.30094v2 only
  python scripts/v4/exp7_multi.py            # all three papers
  python scripts/v4/exp7_multi.py --figure   # docs/figures/v4-exp7-multi.png and the table
"""
import json, random, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from llm import Calls, ROOT, spent
import exp2_errors as exp2
import exp3_grid as exp3
import exp3r_repaired as exp3r
import exp4_stronger as exp4
import exp6_scatter as exp6        # importing it points exp2.source at scatter and the repaired obf/compobf

OUT = ROOT / "data" / "v4" / "exp7-multi"
COPIES = ["clarified", "obf", "compobf", "scatter"]
GROUPS = {"clarified": ["clarified"], "illegible": ["obf", "compobf"], "scatter": ["scatter"]}
READERS = exp6.READERS
SAMPLES, N_CHECK = 4, 20
SLICE = "2609.30094v2"


def set6():
    return [r for r in exp3r.load(exp6.OUT / "errors.jsonl") if r["set6"]]


def combine(p, errs):
    """Greedy joint planting in id order. Returns (kept rows, skipped [(id, reason)])."""
    kept, skipped = [], []
    for r in sorted(errs, key=lambda r: r["id"]):
        trial, why = kept + [r], None
        for c in COPIES:
            text = exp2.source(p, c)
            for e in trial:
                text, prob = exp2.apply(text, e["copies"][c]["edits"])
                if prob:
                    why = f"{c}: {prob}"
                    break
            if why:
                break
            lost = [(e["id"], q[:60]) for e in trial for q in e["copies"][c]["evidence"]
                    if exp2.found(exp2.planted_text(p, c, e), q) and not exp2.found(text, q)]
            if lost:
                why = f"{c}: evidence lost {lost}"
                break
        (skipped.append((r["id"], why)) if why else kept.append(r))
    return kept, skipped


def multi_text(p, c, errs):
    text = exp2.source(p, c)
    for e in errs:
        text, prob = exp2.apply(text, e["copies"][c]["edits"])
        assert prob is None, prob
    return text


def main():
    sl = "--slice" in sys.argv
    OUT.mkdir(parents=True, exist_ok=True)
    calls = Calls(OUT / "calls.jsonl")
    errs = set6()
    papers = [SLICE] if sl else sorted({r["paper"] for r in errs})
    sfx = "-slice" if sl else ""
    plan = {}
    for p in papers:
        kept, skipped = combine(p, [r for r in errs if r["paper"] == p])
        plan[p] = kept
        print(f"{p}: planted together {[r['id'] for r in kept]}; skipped {skipped}")
        with open(OUT / f"plan{sfx}.jsonl", "a" if p != papers[0] else "w") as f:
            f.write(json.dumps({"paper": p, "errors": [r["id"] for r in kept], "skipped": skipped}) + "\n")

    def run(j):
        p, c, rd, s = j
        rev = exp4.review(calls, rd, multi_text(p, c, plan[p]), f"exp7:review:{rd[0]}:{p}:{c}", s)
        grades = [{"error": r["id"], "type": r["type"],
                   **exp3.grade(calls, r, c, rev.get("errors"), tag=f"exp7:grade:{rd[0]}:{p}:{c}:{r['id']}")}
                  for r in plan[p]]
        return {"reader": rd[0], "paper": p, "copy": c, "sample": s, **rev, "grades": grades}

    jobs = [(p, c, rd, s) for p in papers for c in COPIES for rd in READERS for s in range(SAMPLES)]
    with ThreadPoolExecutor(12) as ex:
        revs = list(ex.map(run, jobs))
        pool = [(x, g) for x in revs if x.get("errors") for g in x["grades"]]
        pick = random.Random(0).sample(pool, min(4 if sl else N_CHECK, len(pool)))
        byid = {(r["paper"], r["id"]): r for r in errs}
        chk = list(ex.map(lambda xg: exp3.grade(calls, byid[(xg[0]["paper"], xg[1]["error"])], xg[0]["copy"],
                                                xg[0]["errors"], exp3.CHECKER,
                                                tag=f"exp7:check:{xg[0]['reader']}:{xg[0]['paper']}:{xg[1]['error']}"),
                          pick))
    for (x, g), k in zip(pick, chk):
        g["sol_caught"], g["sol_grade"] = k["caught"], k["grade"]
    (OUT / f"reviews{sfx}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in revs))
    for rd in READERS:
        for c in COPIES:
            gs = [g for x in revs if x["reader"] == rd[0] and x["copy"] == c for g in x["grades"]]
            ne = [x["n_errors"] for x in revs if x["reader"] == rd[0] and x["copy"] == c and x.get("n_errors") is not None]
            print(f"{rd[0]:9s} {c:9s} found {sum(bool(g['caught']) for g in gs)}/{len(gs)}  "
                  f"mean reported {sum(ne) / max(1, len(ne)):.1f}")
    ck = [g for x in revs for g in x["grades"] if "sol_caught" in g and None not in (g["caught"], g["sol_caught"])]
    print(f"grader check: agree {sum(g['caught'] == g['sol_caught'] for g in ck)}/{len(ck)}")
    print(f"reviews failed or unparsed: {sum(not x.get('parsed') for x in revs)}/{len(revs)}")
    print(f"ledger total ${spent():.2f}")


def measures():
    """Per reader: share of planted errors found per copy group, gaps clarified minus illegible and minus scatter
    (paired bootstrap over the errors planted, 2000 resamples, seed 0), mean errors reported per review; and, for
    comparison, the single-error catch rates of the same errors from experiment 6's measures."""
    revs = exp3r.load(OUT / "reviews.jsonl")
    keys = sorted({(x["paper"], g["error"]) for x in revs for g in x["grades"]})
    rng = random.Random(0)
    boots = [[rng.choice(keys) for _ in keys] for _ in range(2000)]
    out = {}
    for rd in [r[0] for r in READERS]:
        by, nrep = {}, {}
        for x in revs:
            if x["reader"] != rd:
                continue
            m = next(m for m, cs in GROUPS.items() if x["copy"] in cs)
            if x.get("n_errors") is not None:
                nrep.setdefault(m, []).append(x["n_errors"])
            for g in x["grades"]:
                if g["caught"] is not None:
                    by.setdefault(m, {}).setdefault((x["paper"], g["error"]), []).append(bool(g["caught"]))
        rate = lambda m, ks: sum(sum(by[m].get(k, [])) for k in ks) / max(1, sum(len(by[m].get(k, [])) for k in ks))
        res = {}
        for m in GROUPS:
            v = sorted(rate(m, b) for b in boots)
            res[m] = (rate(m, keys), v[50], v[1949])
            res[f"reported {m}"] = sum(nrep.get(m, [])) / max(1, len(nrep.get(m, [])))
        for a in ["illegible", "scatter"]:
            g = sorted(rate("clarified", b) - rate(a, b) for b in boots)
            res[f"gap {a}"] = (res["clarified"][0] - res[a][0], g[50], g[1949])
        out[rd] = res
    return out, len(keys)


def figure():
    sys.path.insert(0, str(ROOT / "scripts"))
    from averroes_figstyle import HATCHES, axes_style, bar, legend, save, swatch, plt
    res, n = measures()
    single, _ = exp6.measures()
    ms = list(GROUPS)
    hs = HATCHES[:2]
    fig, axs = plt.subplots(1, len(READERS), figsize=(8.5, 3.4), sharey=True)
    for ax, rd in zip(axs, READERS):
        axes_style(ax, 0, 1, ylabel="share of planted errors found" if ax is axs[0] else None)
        for j, m in enumerate(ms):
            for k, (src, h) in enumerate(zip([single[rd[0]], res[rd[0]]], hs)):
                v = src[m]
                bar(ax, j + (k - 0.5) * 0.36, 0.33, v[0], ci=(v[1], v[2]), hatch=h, label="{:.2f}")
        ax.axhline(0.7, lw=0.6, ls=":", color="0.4")
        ax.set_xticks(range(len(ms)), ["clarified", "illegible\n(repaired obf + compobf)", "scatter"])
        ax.set_xlim(-0.5, len(ms) - 0.5)
        ax.set_title(rd[0], fontsize=9)
    legend(fig, [swatch(hatch=h) for h in hs],
           ["one error per copy (experiments 3r, 4, 6)", f"all errors together (experiment 7, {n} errors)"],
           y=1.08, ncol=2)
    save(fig, ROOT / "docs" / "figures" / "v4-exp7-multi.png")
    f = lambda v: f"{v[0]:.2f} [{v[1]:.2f}, {v[2]:.2f}]"
    print(f"{n} errors planted together")
    print("| reader | clarified | illegible (repaired) | scatter | clarified - illegible | clarified - scatter "
          "| errors reported per review (clar / ill / scatter) |")
    print("|---" * 7 + "|")
    for rd in READERS:
        r = res[rd[0]]
        print(f"| {rd[0]} | " + " | ".join(f(r[m]) for m in ["clarified", "illegible", "scatter", "gap illegible",
                                                             "gap scatter"])
              + " | " + " / ".join(f"{r['reported ' + m]:.1f}" for m in GROUPS) + " |")


if __name__ == "__main__":
    figure() if "--figure" in sys.argv else main()
