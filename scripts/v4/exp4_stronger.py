"""Plan v4 experiment 4: stronger reader settings on the repaired copies.

Preregistration: docs/plan-v4.md, "Experiment 4 preregistration". Two new settings (gpt-5.6-luna medium, gpt-5-mini
high) review the 17 errors in both preregistered sets (data/v4/exp3r-repaired/errors.jsonl) in all four copies
(clarified, orig, repaired obf, repaired compobf), 2 samples; gpt-5-mini low grades each review. Clean reviews: 2
samples per setting on each of the 4 unplanted copies of the 3 papers, graded against each error of that paper.
Sol (low) regrades 30 random planted reviews (seed 0). Prompts and grading are experiment 3's, unchanged.
Experiment 3/3r rows for luna-low and mini-med on the same copies are reused as reference rows in the analysis.
Output in data/v4/exp4-stronger/: calls.jsonl (raw, cached), reviews.jsonl, clean-grades.jsonl.
Usage:
  python scripts/v4/exp4_stronger.py --slice    # one error x 4 copies x both settings, 1 sample
  python scripts/v4/exp4_stronger.py            # all
  python scripts/v4/exp4_stronger.py --figure   # docs/figures/v4-exp4-stronger.png and the table
"""
import json, random, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from llm import Calls, ROOT, spent
import exp2_errors as exp2
import exp3_grid as exp3
import exp3r_repaired as exp3r     # importing it points exp2.source at the repaired obf/compobf

OUT = ROOT / "data" / "v4" / "exp4-stronger"
COPIES = exp2.COPIES
READERS = [("luna-med", "gpt-5.6-luna", "medium"), ("mini-high", "gpt-5-mini", "high")]
REFERENCE = ["luna-low", "mini-med"]          # from experiments 3 (clarified, orig) and 3r (repaired obf/compobf)
SAMPLES = 2
MAX_TOKENS = 32000
N_CHECK = 30


def load_errors():
    return [r for r in exp3r.load(exp3r.OUT / "errors.jsonl") if r["set"] == "prereg" and r["set_v1"] == "prereg"]


def review(calls, reader, text, tag, s):
    label, model, effort = reader
    try:
        rec = calls.call(model, exp3.exp0.REVIEW_SYSTEM, exp3.exp0.REVIEW_USER.format(paper=text), tag=tag, sample=s,
                         effort=effort, max_tokens=MAX_TOKENS)
    except RuntimeError as e:
        return {"failed": str(e)}
    errs = exp3.exp0.parse(rec["content"])
    return {"model_returned": rec["model_returned"], "parsed": errs is not None, "errors": errs,
            "n_errors": len(errs) if errs is not None else None, "cost": rec["cost"]}


def main():
    sl = "--slice" in sys.argv
    OUT.mkdir(parents=True, exist_ok=True)
    calls = Calls(OUT / "calls.jsonl")
    errs = load_errors()[:1 if sl else None]
    samples = 1 if sl else SAMPLES
    jobs = [(r, c, rd, s) for r in errs for c in COPIES for rd in READERS for s in range(samples)]

    def run(j):
        r, c, rd, s = j
        rev = review(calls, rd, exp2.planted_text(r["paper"], c, r), f"exp4:review:{rd[0]}:{r['paper']}:{c}:{r['id']}", s)
        out = {"kind": "planted", "set": "prereg", "reader": rd[0], "paper": r["paper"], "error": r["id"],
               "type": r["type"], "copy": c, "sample": s, **rev}
        out.update(exp3.grade(calls, r, c, rev.get("errors"), tag=f"exp4:grade:{rd[0]}:{r['paper']}:{c}:{r['id']}"))
        return out

    papers = sorted({r["paper"] for r in errs})
    clean = [] if sl else [(p, c, rd, s) for p in papers for c in COPIES for rd in READERS for s in range(SAMPLES)]

    def run_clean(j):
        p, c, rd, s = j
        rev = review(calls, rd, exp2.source(p, c), f"exp4:clean:{rd[0]}:{p}:{c}", s)
        return {"kind": "clean", "reader": rd[0], "paper": p, "copy": c, "sample": s, **rev}

    with ThreadPoolExecutor(12) as ex:
        revs = list(ex.map(run, jobs))
        cleans = list(ex.map(run_clean, clean))
        gjobs = [(cr, r) for cr in cleans for r in errs if r["paper"] == cr["paper"]]
        cgrades = list(ex.map(lambda j: {
            "kind": "clean", "set": "prereg", "reader": j[0]["reader"], "paper": j[0]["paper"],
            "error": j[1]["id"], "type": j[1]["type"], "copy": j[0]["copy"], "sample": j[0]["sample"],
            **exp3.grade(calls, j[1], j[0]["copy"], j[0].get("errors"),
                         tag=f"exp4:cgrade:{j[0]['reader']}:{j[0]['paper']}:{j[0]['copy']}:{j[1]['id']}")}, gjobs))
        pool = [x for x in revs if x.get("errors")]
        pick = random.Random(0).sample(pool, min(N_CHECK if not sl else 3, len(pool)))
        byid = {(r["paper"], r["id"]): r for r in errs}
        chk = list(ex.map(lambda x: exp3.grade(calls, byid[(x["paper"], x["error"])], x["copy"], x["errors"],
                                               exp3.CHECKER, tag=f"exp4:check:{x['reader']}:{x['paper']}:{x['copy']}:{x['error']}"),
                          pick))
    for x, g in zip(pick, chk):
        x["sol_caught"], x["sol_grade"] = g["caught"], g["grade"]
    sfx = "-slice" if sl else ""
    (OUT / f"reviews{sfx}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in revs + cleans))
    (OUT / f"clean-grades{sfx}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in cgrades))
    for rd in READERS:
        print(f"{rd[0]:9s} " + "  ".join(
            f"{c}: {sum(bool(x['caught']) for x in revs if x['reader'] == rd[0] and x['copy'] == c)}"
            f"/{sum(x['reader'] == rd[0] and x['copy'] == c for x in revs)}" for c in COPIES) +
              f"  fa {sum(bool(x['caught']) for x in cgrades if x['reader'] == rd[0])}"
              f"/{sum(x['reader'] == rd[0] for x in cgrades)}")
    ck = [x for x in revs if "sol_caught" in x and x["caught"] is not None and x["sol_caught"] is not None]
    print(f"grader check: agree {sum(x['caught'] == x['sol_caught'] for x in ck)}/{len(ck)}")
    print(f"planted reviews ungraded: {sum(x.get('caught') is None and x.get('errors') != [] for x in revs)}/{len(revs)}")
    print(f"reviews failed or unparsed: {sum(not x.get('parsed') for x in revs + cleans)}/{len(revs + cleans)}")
    print(f"ledger total ${spent():.2f}")


def rows_all():
    """Planted and clean-grade rows for the new settings and the reference settings, on the 17 errors."""
    keys = {(r["paper"], r["id"]) for r in load_errors()}
    new = exp3r.load(OUT / "reviews.jsonl") + exp3r.load(OUT / "clean-grades.jsonl")
    old = [x for x in exp3r.load(exp3.OUT / "reviews.jsonl") + exp3r.load(exp3.OUT / "clean-grades.jsonl")
           if x["copy"] in ("clarified", "orig")]
    rep = exp3r.load(exp3r.OUT / "reviews.jsonl") + exp3r.load(exp3r.OUT / "clean-grades.jsonl")
    ref = [x for x in old + rep if x["reader"] in REFERENCE]
    # Clean-grade rows carry an "error" key; clean reviews do not and are not used here.
    return [x for x in new + ref if "error" in x and (x["paper"], x["error"]) in keys]


def measures():
    """Per setting, pooled over types: catch rate on clarified, illegible (repaired obf+compobf), orig; the
    clarified-minus-illegible gap (paired); false accusations (clean reviews, all 4 copies). Bootstrap 95% over
    errors, 2000 resamples, seed 0."""
    rows = rows_all()
    common = sorted({(r["paper"], r["id"]) for r in load_errors()})
    groups = {"clarified": ["clarified"], "illegible": ["obf", "compobf"], "orig": ["orig"]}
    out, rng = {}, random.Random(0)
    boots = [[rng.choice(common) for _ in common] for _ in range(2000)]
    for label in [r[0] for r in READERS] + REFERENCE:
        by = {}
        for x in rows:
            if x["reader"] != label or x.get("caught") is None:
                continue
            ms = ["accuse"] if x["kind"] == "clean" else [m for m, cs in groups.items() if x["copy"] in cs]
            for m in ms:
                by.setdefault(m, {}).setdefault((x["paper"], x["error"]), []).append(bool(x["caught"]))
        rate = lambda m, ks: sum(sum(by[m].get(k, [])) for k in ks) / max(1, sum(len(by[m].get(k, [])) for k in ks))
        res = {}
        for m in ["clarified", "illegible", "orig", "accuse"]:
            v = sorted(rate(m, b) for b in boots)
            res[m] = (rate(m, common), v[50], v[1949])
        g = sorted(rate("clarified", b) - rate("illegible", b) for b in boots)
        res["gap"] = (res["clarified"][0] - res["illegible"][0], g[50], g[1949])
        out[label] = res
    return out, len(common)


def figure():
    sys.path.insert(0, str(ROOT / "scripts"))
    from averroes_figstyle import HATCHES, axes_style, bar, legend, save, swatch, plt
    res, n = measures()
    labels = REFERENCE + [r[0] for r in READERS]
    ms = ["clarified", "orig", "illegible", "accuse"]
    names = ["clarified", "original", "illegible (repaired obf + compobf)", "false accusation (clean copies)"]
    hs = [HATCHES[0], HATCHES[1], HATCHES[2], HATCHES[3]]
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    axes_style(ax, 0, 1, ylabel="share")
    w = 0.2
    for i, lb in enumerate(labels):
        for j, m in enumerate(ms):
            v = res[lb][m]
            bar(ax, i + (j - 1.5) * w, w * 0.92, v[0], ci=(v[1], v[2]), hatch=hs[j], label="{:.2f}")
    for y in (0.7, 0.35, 0.10):
        ax.axhline(y, lw=0.6, ls=":", color="0.4")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_xlim(-0.5, len(labels) - 0.5)
    legend(fig, [swatch(hatch=h) for h in hs], names, y=1.1, ncol=2)
    save(fig, ROOT / "docs" / "figures" / "v4-exp4-stronger.png")
    f = lambda v: f"{v[0]:.2f} [{v[1]:.2f}, {v[2]:.2f}]"
    print(f"{n} errors; dotted lines: proposed bars 0.70 (clarified), 0.35 (illegible), 0.10 (false accusation)")
    print("| setting | clarified | original | illegible | clarified - illegible | false accusation |")
    print("|---|---|---|---|---|---|")
    for lb in labels:
        print(f"| {lb} | " + " | ".join(f(res[lb][m]) for m in ["clarified", "orig", "illegible", "gap", "accuse"]) + " |")


if __name__ == "__main__":
    figure() if "--figure" in sys.argv else main()
