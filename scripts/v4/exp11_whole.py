"""Experiment 11 analysis (plan-v4.md, "Experiment 11 preregistration" and its amendment): whole-paper error
checking, per reader, on the 4 calibration papers and the 6 reviewed anchors, as three rates (garbled `obf-max`,
original `orig`, clarified original `clar-explainer-orig`, a calibration standard only). The difference is `orig` minus
`obf-max`. No API calls.

Run: uv run --with matplotlib python scripts/v4/exp11_whole.py
Prints the preregistered measures and writes docs/figures/v4-exp11-whole.png.
Rates average each error's samples; intervals resample errors within papers (2000 resamples, seed 0).
False accusations: share of clean reads with at least one item graded a catch of some primary survivor.
"""
import collections
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from averroes_figstyle import *  # noqa: E402,F403

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(ROOT, "data", "v4", "err9")
FIG = os.path.join(ROOT, "docs", "figures", "v4-exp11-whole.png")
CAL = ["2609.30094v2", "2608.13754v1", "2609.02893v1", "jmlr-2014-dropout"]
READERS = [("w11-mini-high", "GPT-5 mini high"), ("w11-luna-low", "GPT-5.6 Luna low"),
           ("w11-mini54-med", "GPT-5.4 mini medium"), ("w11-gpt55-low", "GPT-5.5 low")]
GROUPS = [("calibration", "4 calibration papers"), ("anchors", "6 reviewed papers")]
VERS = ("orig", "obf-max", "clar-explainer-orig")
BARS = [("obf-max", "xxxx", "Garbled version"), ("orig", "////", "Original paper"),
        ("clar-explainer-orig", "", "Clarified original")]     # as scripts/blog_part2_final.py


def load(rd):
    path = os.path.join(D, f"reader-{rd}.jsonl")
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


def in_group(p, g):
    return (p in CAL) == (g == "calibration")


def per_error(rows, kind, cond="planted"):
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        if r["kind"] != kind or r.get("caught") is None or (kind == "pointed" and r["cond"] != cond):
            continue
        acc[(r["paper"], r["error"])][r["version"]].append(1.0 if r["caught"] else 0.0)
    return {k: {v: sum(x) / len(x) for v, x in d.items()} for k, d in acc.items() if all(d[v] for v in VERS)}


def summary(rows, kind="planted", cond="planted", B=2000):
    pe = per_error(rows, kind, cond)
    if not pe:
        return None
    by_paper = collections.defaultdict(list)
    for (p, _), d in pe.items():
        by_paper[p].append(d)
    mean = lambda s, v: sum(d[v] for d in s) / len(s)
    out = {"n": len(pe), "papers": len(by_paper)}
    vals = list(pe.values())
    for v in VERS:
        out[v] = mean(vals, v)
    out["diff"] = out["orig"] - out["obf-max"]
    rnd, boots = random.Random(0), {k: [] for k in VERS + ("diff",)}
    for _ in range(B):
        s = [x for lst in by_paper.values() for x in (rnd.choice(lst) for _ in lst)]
        for v in VERS:
            boots[v].append(mean(s, v))
        boots["diff"].append(boots["orig"][-1] - boots["obf-max"][-1])
    for k, b in boots.items():
        b.sort()
        out[k + ":ci"] = (b[int(0.025 * B)], b[int(0.975 * B) - 1])
    return out


def false_acc(rows, v):
    hit = collections.defaultdict(bool)
    for r in rows:
        if r["kind"] == "cgrade" and r["version"] == v:
            hit[(r["paper"], r["sample"])] |= bool(r.get("caught"))
    return sum(hit.values()), len(hit)


def fmt(s, k):
    lo, hi = s[k + ":ci"]
    return f"{s[k]:+.2f} [{lo:+.2f}, {hi:+.2f}]" if k == "diff" else f"{s[k]:.2f} [{lo:.2f}, {hi:.2f}]"


def report():
    res = {}
    for rd, name in READERS:
        rows = load(rd)
        for g, gname in GROUPS:
            rs = [r for r in rows if in_group(r["paper"], g)]
            if not rs:
                continue
            u, pt, ct = summary(rs), summary(rs, "pointed"), summary(rs, "pointed", "control")
            res[(rd, g)] = (u, pt)
            reads = [r for r in rs if r["kind"] in ("clean", "planted")]
            print(f"{name} [{gname}]: {u['n']} errors on {u['papers']} papers")
            for lab, x in (("unaided", u), ("pointed", pt)):
                print(f"  {lab}  garbled {fmt(x, 'obf-max')}  original {fmt(x, 'orig')}  clarified "
                      f"{fmt(x, 'clar-explainer-orig')}  orig-obf {fmt(x, 'diff')}")
            print("  control accusations  " + "  ".join(f"{v} {ct[v]:.2f}" for v in VERS))
            print("  false accusations  " + "  ".join(f"{v} {a}/{n}" for v in VERS for a, n in [false_acc(rs, v)]))
            print(f"  reads {len(reads)}, unparsed or failed {sum((r.get('parsed') is False) + bool(r.get('failed')) for r in reads)}"
                  f", grades None {sum(r.get('caught') is None for r in rs if r['kind'] != 'clean')}"
                  f", cost of reads and pointed ${sum(r.get('cost', 0) for r in rs if r['kind'] != 'cgrade'):.2f}")
    return res


def figure(res):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, (g, gname) in zip(axes, GROUPS):
        axes_style(ax, 0, 1, ylabel="Share of planted errors found unaided" if g == "calibration" else None)
        names = [(rd, n) for rd, n in READERS if (rd, g) in res]
        for i, (rd, _) in enumerate(names):
            u = res[(rd, g)][0]
            for j, (v, h, _) in enumerate(BARS):
                bar(ax, i + (j - 1) * 0.27, 0.25, u[v], ci=u[v + ":ci"], label="{:.2f}", hatch=h)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.replace(" ", "\n", 1) for _, n in names], fontsize=10)
        ax.set_xlim(-0.55, max(len(names), 1) - 0.45)
        ax.set_title(gname, fontsize=12)
    legend(fig, [swatch(hatch=h) for _, h, _ in BARS], [n for _, _, n in BARS], y=1.02)
    fig.subplots_adjust(bottom=0.18, top=0.82, wspace=0.08)
    save(fig, FIG)


if __name__ == "__main__":
    figure(report())
