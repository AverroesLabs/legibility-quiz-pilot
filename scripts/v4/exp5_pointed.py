"""Plan v4 experiment 5: pointed check. Is a miss a search failure or a judging failure?

Preregistration: docs/plan-v4.md, "Experiment 5 preregistration". luna-low and mini-high are given a whole copy and
one quoted passage (the planted text) and asked whether the passage is consistent with the rest of the paper. 17
errors (experiment 4's), copies clarified, repaired obf, repaired compobf, 1 sample. Control: the same question on the
unplanted clarified copy, quoting the passage before planting. A "consistent: false" reply is graded with experiment
3's grader as a one-item review. Sol (low) regrades all control accusations and 20 random planted replies (seed 0).
Output in data/v4/exp5-pointed/: calls.jsonl (raw, cached), replies.jsonl.
Usage:
  python scripts/v4/exp5_pointed.py --slice    # one error, both settings
  python scripts/v4/exp5_pointed.py            # all
  python scripts/v4/exp5_pointed.py --figure   # docs/figures/v4-exp5-pointed.png and the tables
"""
import json, random, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from llm import Calls, ROOT, spent
import exp2_errors as exp2
import exp3_grid as exp3
import exp3r_repaired as exp3r     # importing it points exp2.source at the repaired obf/compobf
import exp4_stronger as exp4

OUT = ROOT / "data" / "v4" / "exp5-pointed"
COPIES = ["clarified", "obf", "compobf"]
READERS = [("luna-low", "gpt-5.6-luna", "low"), ("mini-high", "gpt-5-mini", "high")]
MAX_TOKENS = 32000
N_CHECK = 20

POINTED_USER = """Paper:

<paper>
{paper}
</paper>

Check whether this passage is consistent with the rest of the paper:

<passage>
{passage}
</passage>

Reply with JSON only, in this form:
{{"consistent": true | false, "evidence": "<if not consistent: the content of the paper that contradicts the passage,
and where it is; otherwise empty>", "explanation": "<one or two sentences>"}}"""


def passage(err, c, side):
    return " ... ".join(e[side] for e in err["copies"][c]["edits"])


def ask(calls, reader, err, c, control):
    label, model, effort = reader
    p = err["paper"]
    text = exp2.source(p, c) if control else exp2.planted_text(p, c, err)
    psg = passage(err, c, "old" if control else "new")
    cond = "control" if control else "planted"
    out = {"cond": cond, "reader": label, "paper": p, "error": err["id"], "type": err["type"], "copy": c,
           "passage": psg}
    try:
        rec = calls.call(model, exp3.exp0.REVIEW_SYSTEM, POINTED_USER.format(paper=text, passage=psg),
                         tag=f"exp5:{cond}:{label}:{p}:{c}:{err['id']}", effort=effort, max_tokens=MAX_TOKENS)
    except RuntimeError as e:
        return {**out, "failed": str(e), "caught": None}
    d = exp2.parse(rec["content"], "consistent")
    out.update(model_returned=rec["model_returned"], cost=rec["cost"], parsed=d is not None, reply=d)
    if d is None:
        return {**out, "caught": None}
    out["says_inconsistent"] = d["consistent"] is False
    if not out["says_inconsistent"]:
        return {**out, "caught": False}
    item = [{"claim": psg, "location": "", "evidence": d.get("evidence", ""), "explanation": d.get("explanation", "")}]
    out.update(exp3.grade(calls, err, c, item, tag=f"exp5:grade:{cond}:{label}:{p}:{c}:{err['id']}"))
    return out


def main():
    sl = "--slice" in sys.argv
    OUT.mkdir(parents=True, exist_ok=True)
    calls = Calls(OUT / "calls.jsonl")
    errs = exp4.load_errors()[:1 if sl else None]
    jobs = ([(rd, r, c, False) for r in errs for c in COPIES for rd in READERS] +
            [(rd, r, "clarified", True) for r in errs for rd in READERS])
    with ThreadPoolExecutor(12) as ex:
        rows = list(ex.map(lambda j: ask(calls, *j), jobs))
        byid = {(r["paper"], r["id"]): r for r in errs}
        pool = [x for x in rows if x["cond"] == "planted" and x.get("says_inconsistent")]
        pick = ([x for x in rows if x["cond"] == "control" and x.get("says_inconsistent")] +
                random.Random(0).sample(pool, min(3 if sl else N_CHECK, len(pool))))

        def check(x):
            item = [{"claim": x["passage"], "location": "", "evidence": x["reply"].get("evidence", ""),
                     "explanation": x["reply"].get("explanation", "")}]
            return exp3.grade(calls, byid[(x["paper"], x["error"])], x["copy"], item, exp3.CHECKER,
                              tag=f"exp5:check:{x['cond']}:{x['reader']}:{x['paper']}:{x['copy']}:{x['error']}")
        chk = list(ex.map(check, pick))
    for x, g in zip(pick, chk):
        x["sol_caught"], x["sol_grade"] = g["caught"], g["grade"]
    sfx = "-slice" if sl else ""
    (OUT / f"replies{sfx}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    for rd in READERS:
        for cond, cs in [("planted", COPIES), ("control", ["clarified"])]:
            sub = [x for x in rows if x["reader"] == rd[0] and x["cond"] == cond]
            print(f"{rd[0]:9s} {cond:8s} " + "  ".join(
                f"{c}: caught {sum(bool(x['caught']) for x in sub if x['copy'] == c)}"
                f" inconsistent {sum(bool(x.get('says_inconsistent')) for x in sub if x['copy'] == c)}"
                f"/{sum(x['copy'] == c for x in sub)}" for c in cs))
    ck = [x for x in pick if x.get("caught") is not None and x.get("sol_caught") is not None]
    print(f"grader check: agree {sum(x['caught'] == x['sol_caught'] for x in ck)}/{len(ck)}")
    print(f"failed or unparsed: {sum(not x.get('parsed') for x in rows)}/{len(rows)}")
    print(f"ledger total ${spent():.2f}")


def unpointed():
    """(reader, paper, error, copy) -> True if caught in a strict majority of unpointed samples."""
    by = {}
    for x in exp4.rows_all():
        if x["kind"] == "planted" and x["caught"] is not None and x["reader"] in [r[0] for r in READERS]:
            by.setdefault((x["reader"], x["paper"], x["error"], x["copy"]), []).append(bool(x["caught"]))
    return {k: sum(v) * 2 > len(v) for k, v in by.items()}


def measures():
    rows = exp3r.load(OUT / "replies.jsonl")
    up = unpointed()
    keys = sorted({(r["paper"], r["id"]) for r in exp4.load_errors()})
    rng = random.Random(0)
    boots = [[rng.choice(keys) for _ in keys] for _ in range(2000)]
    groups = {"clarified": ["clarified"], "illegible": ["obf", "compobf"]}
    res = {}
    for rd in [r[0] for r in READERS]:
        sub = [x for x in rows if x["reader"] == rd and x["caught"] is not None]
        pl = [x for x in sub if x["cond"] == "planted"]

        def rate(cs, ks, xs=pl):
            ks = set(ks)
            v = [bool(x["caught"]) for x in xs if x["copy"] in cs and (x["paper"], x["error"]) in ks]
            return sum(v) / len(v) if v else float("nan")

        def rate_b(cs, b):   # bootstrap: each resampled error counts as often as drawn
            v = [bool(x["caught"]) for k in b for x in pl if (x["paper"], x["error"]) == k and x["copy"] in cs]
            return sum(v) / len(v) if v else float("nan")
        r = {}
        for m, cs in groups.items():
            v = sorted(rate_b(cs, b) for b in boots)
            r[m] = (rate(cs, keys), v[50], v[1949])
        g = sorted(rate_b(groups["clarified"], b) - rate_b(groups["illegible"], b) for b in boots)
        r["gap"] = (r["clarified"][0] - r["illegible"][0], g[50], g[1949])
        for m, cs in groups.items():
            for lab, want in [("missed", False), ("found", True)]:
                v = [bool(x["caught"]) for x in pl if x["copy"] in cs
                     and up.get((rd, x["paper"], x["error"], x["copy"])) is want]
                r[f"{m}:{lab}"] = (sum(v), len(v))
        ctl = [x for x in sub if x["cond"] == "control"]
        r["control:inconsistent"] = (sum(bool(x.get("says_inconsistent")) for x in ctl), len(ctl))
        r["control:graded"] = (sum(bool(x["caught"]) for x in ctl), len(ctl))
        res[rd] = r
    return res, len(keys)


def figure():
    sys.path.insert(0, str(ROOT / "scripts"))
    from averroes_figstyle import HATCHES, axes_style, bar, legend, save, swatch, plt
    res, n = measures()
    labels = [r[0] for r in READERS]
    ref, _ = exp4.measures()
    fig, ax = plt.subplots(figsize=(7, 3.6))
    axes_style(ax, 0, 1, ylabel="catch rate")
    names = ["clarified, unpointed review", "illegible, unpointed review", "clarified, pointed", "illegible, pointed"]
    hs = HATCHES[:4]
    w = 0.2
    for i, lb in enumerate(labels):
        vals = [ref[lb]["clarified"], ref[lb]["illegible"], res[lb]["clarified"], res[lb]["illegible"]]
        for j, v in enumerate(vals):
            bar(ax, i + (j - 1.5) * w, w * 0.92, v[0], ci=(v[1], v[2]), hatch=hs[j], label="{:.2f}")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_xlim(-0.5, len(labels) - 0.5)
    legend(fig, [swatch(hatch=h) for h in hs], names, y=1.1, ncol=2)
    save(fig, ROOT / "docs" / "figures" / "v4-exp5-pointed.png")
    f = lambda v: f"{v[0]:.2f} [{v[1]:.2f}, {v[2]:.2f}]"
    fr = lambda v: f"{v[0]}/{v[1]}"
    print(f"{n} errors, 1 pointed sample per error and copy; unpointed rates from experiments 3/3r/4")
    print("| setting | pointed clarified | pointed illegible | pointed gap | unpointed gap |")
    print("|---|---|---|---|---|")
    for lb in labels:
        print(f"| {lb} | {f(res[lb]['clarified'])} | {f(res[lb]['illegible'])} | {f(res[lb]['gap'])} | {f(ref[lb]['gap'])} |")
    print("\n| setting | clarified: pointed catch on missed / found | illegible: missed / found | control: says inconsistent / graded as the planted reason |")
    print("|---|---|---|---|")
    for lb in labels:
        r = res[lb]
        print(f"| {lb} | {fr(r['clarified:missed'])} / {fr(r['clarified:found'])} | "
              f"{fr(r['illegible:missed'])} / {fr(r['illegible:found'])} | "
              f"{fr(r['control:inconsistent'])} / {fr(r['control:graded'])} |")


if __name__ == "__main__":
    figure() if "--figure" in sys.argv else main()
