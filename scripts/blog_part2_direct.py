"""Figures for the Part 2 blog post, direct-test version: errors planted in the paper itself (orig) and in its
garbled version (obf-max), no clarified copies. Computed from the experiment 9 reader rows.

Run: uv run --with matplotlib python scripts/blog_part2_direct.py
Writes docs/blog/part2-direct-fig-{1,2,3}.png and prints the numbers used.
Primary error set: experiment 9 survivors excluding `omitted` and `unsupported`. Rates average each error's
samples; intervals resample errors within papers (2000 resamples, seed 0).
"""
import collections
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
from averroes_figstyle import *  # noqa: E402,F403
from matplotlib.lines import Line2D  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data", "v4", "err9")
OUT = os.path.join(ROOT, "docs", "blog")
ORIG = dict(face=PAPER, hatch="////")
GARB = dict(face=PAPER, hatch="xxxx")

SURV = {(e["paper"], e["id"]) for e in map(json.loads, open(os.path.join(D, "errors.jsonl")))
        if e.get("survives") and e["category"] not in ("omitted", "unsupported")}


CAL = ("2609.30094v2", "2608.13754v1", "2609.02893v1", "jmlr-2014-dropout")   # the 4 calibration papers
ANCHORS = ("2601.02896v4", "2603.01214v2", "2509.22621v3", "2509.11379v3", "2510.17247v2", "2506.22982v1")  # 6 reviewed
ALL = CAL + ANCHORS                                                               # all ten papers, 82 primary errors


def per_error(fname, kind, papers=None):
    """{(paper, error): {version: mean caught}} for orig and obf-max. fname may be a list of sibling files."""
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in (json.loads(line) for f in ([fname] if isinstance(fname, str) else fname) for line in open(os.path.join(D, f))):
        if r["kind"] != kind or r.get("caught") is None or (kind == "pointed" and r.get("cond") != "planted"):
            continue
        if papers is not None and r["paper"] not in papers:
            continue
        if (r["paper"], r["error"]) in SURV and r["version"] in ("orig", "obf-max"):
            acc[(r["paper"], r["error"])][r["version"]].append(1.0 if r["caught"] else 0.0)
    return {k: {v: sum(x) / len(x) for v, x in d.items()} for k, d in acc.items() if d["orig"] and d["obf-max"]}


def summary(fname, kind="planted", papers=None):
    pe = per_error(fname, kind, papers)
    by_paper = collections.defaultdict(list)
    for (p, _), d in pe.items():
        by_paper[p].append(d)
    rnd = random.Random(0)
    boots = {"orig": [], "obf-max": [], "diff": []}
    for _ in range(2000):
        s = [x for lst in by_paper.values() for x in (rnd.choice(lst) for _ in lst)]
        o = sum(d["orig"] for d in s) / len(s)
        g = sum(d["obf-max"] for d in s) / len(s)
        boots["orig"].append(o), boots["obf-max"].append(g), boots["diff"].append(o - g)
    out = {"n": len(pe), "papers": len(by_paper)}
    vals = list(pe.values())
    out["orig"] = sum(d["orig"] for d in vals) / len(vals)
    out["obf-max"] = sum(d["obf-max"] for d in vals) / len(vals)
    out["diff"] = out["orig"] - out["obf-max"]
    for k, b in boots.items():
        b.sort()
        out[k + ":ci"] = (b[50], b[1949])
    return out


READERS = [("GPT-5 mini high\nwhole paper", "reader-whole-mini-high.jsonl"),
           ("GPT-5 mini high\none section at a time", "reader-mini-high-A.jsonl"),
           ("GPT-5.6 Luna low\none section at a time", "reader-luna-low-A.jsonl"),
           ("GPT-5.4 mini\none section at a time", "reader-mini54-A.jsonl"),
           ("GPT-5.5 low\none section at a time", "reader-gpt55-A.jsonl")]


def fig1(kind="planted", out="part2-direct-fig-1.png", ylabel="Share of planted errors found"):
    """kind "planted": the reader reviews the paper unaided. kind "pointed": it is shown the planted passage."""
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    axes_style(ax, 0, 1, ylabel=ylabel)
    for g, (name, f) in enumerate(READERS):
        s = summary(f, kind)
        print(f"fig1[{kind}] {name.replace(chr(10), ' ')}: n={s['n']} orig {s['orig']:.2f} {s['orig:ci']} "
              f"garbled {s['obf-max']:.2f} {s['obf-max:ci']} diff {s['diff']:+.2f} {s['diff:ci']}")
        for i, (v, st) in enumerate((("orig", ORIG), ("obf-max", GARB))):
            bar(ax, g + (i - 0.5) * 0.34, 0.31, s[v], ci=s[v + ":ci"], label="{:.2f}", **st)
    ax.set_xticks(range(len(READERS)))
    ax.set_xticklabels([n for n, _ in READERS], fontsize=11)
    ax.set_xlim(-0.55, len(READERS) - 0.45)
    legend(fig, [swatch(**ORIG), swatch(**GARB)], ["Error planted in the original paper",
                                                    "Same error in the garbled version"], y=1.02)
    fig.subplots_adjust(bottom=0.17, top=0.88)
    save(fig, os.path.join(OUT, out))


def fig2():
    panels = [("GPT-5 mini high, whole paper", "reader-whole-mini-high.jsonl"),
              ("GPT-5.5 low, one section at a time", "reader-gpt55-A.jsonl")]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.4), sharey=True)
    for ax, (title, f) in zip(axes, panels):
        axes_style(ax, 0, 1, ylabel="Share of planted errors caught" if ax is axes[0] else None)
        for x, kind, face in ((0, "planted", PAPER), (1, "pointed", QUIET)):
            s = summary(f, kind)
            print(f"fig2 {title} {kind}: orig {s['orig']:.2f} garbled {s['obf-max']:.2f}")
            for i, (v, st) in enumerate((("orig", ORIG), ("obf-max", GARB))):
                bar(ax, x + (i - 0.5) * 0.32, 0.29, s[v], ci=s[v + ":ci"], label="{:.2f}",
                    face=face, hatch=st["hatch"])
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Reviewing\nunaided", "Shown the\nplanted passage"], fontsize=11)
        ax.set_xlim(-0.6, 1.6)
        ax.set_title(title, fontsize=12)
    legend(fig, [swatch(**ORIG), swatch(**GARB)], ["Original paper", "Garbled version"], y=1.02)
    fig.subplots_adjust(bottom=0.17, top=0.8, wspace=0.12)
    save(fig, os.path.join(OUT, "part2-direct-fig-2.png"))


def fig3():
    rows = [(n.replace("one section at a time", "section by section"), f, "4 calibration papers") for n, f in READERS] + [
        ("GPT-5.5 low\nsection by section", "reader-gpt55-A-anchors.jsonl", "6 reviewed papers"),
        ("GPT-5.5 low\n600-word notes", "reader-gpt55-A600.jsonl", "all 10 papers")]
    fig, ax = plt.subplots(figsize=(13, 4.8))
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.set_ylim(-0.3, 0.45)
    ax.axhline(0, color=INK, lw=1.2)
    ax.grid(axis="y", color=QUIET, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_ylabel("Share found in original\nminus share in garbled")
    for i, (name, f, papers) in enumerate(rows):
        s = summary(f)
        print(f"fig3 {name.replace(chr(10), ' ')} [{papers}]: n={s['n']} diff {s['diff']:+.2f} {s['diff:ci']}")
        lo, hi = s["diff:ci"]
        new = papers != "4 calibration papers"
        ax.errorbar(i, s["diff"], yerr=[[s["diff"] - lo], [hi - s["diff"]]], fmt="s" if new else "o", ms=7,
                    mfc=INK if new else PAPER, mec=INK, ecolor=INK, elinewidth=1.4, capsize=3, zorder=4)
    ax.axvline(len(READERS) - 0.5, color=QUIET, lw=1, ls=":")
    ax.text((len(READERS) - 1) / 2, 0.41, "4 calibration papers", ha="center", fontsize=11, color=SOFT)
    ax.text(len(READERS) + 0.5, 0.41, "after freezing GPT-5.5", ha="center", fontsize=11, color=SOFT)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([f"{n}\n({p})" if p != "4 calibration papers" else n for n, _, p in rows], fontsize=10)
    ax.set_xlim(-0.5, len(rows) - 0.5)
    ax.tick_params(axis="x", length=0)
    fig.subplots_adjust(bottom=0.24)
    save(fig, os.path.join(OUT, "part2-direct-fig-3.png"))


LEVELS = [("obf-max", "Garbled"), ("orig", "Original"), ("clar-explainer-orig", "Clarified\noriginal")]


def levels(fname, kind, papers=None):
    """Mean catch rate and 95% interval per legibility level, over errors present in all three levels."""
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    rows = (json.loads(line) for f in ([fname] if isinstance(fname, str) else fname) for line in open(os.path.join(D, f)))
    for r in rows:
        if r["kind"] != kind or r.get("caught") is None or (kind == "pointed" and r.get("cond") != "planted"):
            continue
        if papers is not None and r["paper"] not in papers:
            continue
        if (r["paper"], r["error"]) in SURV and r["version"] in dict(LEVELS):
            acc[(r["paper"], r["error"])][r["version"]].append(1.0 if r["caught"] else 0.0)
    items = {k: {v: sum(x) / len(x) for v, x in d.items()} for k, d in acc.items() if all(d[v] for v, _ in LEVELS)}
    by_paper = collections.defaultdict(list)
    for (p, _), d in items.items():
        by_paper[p].append(d)
    rnd, boots = random.Random(0), {v: [] for v, _ in LEVELS}
    for _ in range(2000):
        smp = [x for lst in by_paper.values() for x in (rnd.choice(lst) for _ in lst)]
        for v, _ in LEVELS:
            boots[v].append(sum(d[v] for d in smp) / len(smp))
    out = {}
    for v, _ in LEVELS:
        b = sorted(boots[v])
        out[v] = (sum(d[v] for d in items.values()) / len(items), b[50], b[1949])
    return out


def fig_calibration():
    """Catch rate across three known legibility levels, one line per reader; left unaided, right pointed."""
    styles = [("o", PAPER, "-"), ("s", PAPER, "--"), ("^", PAPER, ":"), ("D", PAPER, "-."), ("o", INK, "-")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, kind, title in zip(axes, ("planted", "pointed"), ("Reviewing unaided", "Shown the planted passage")):
        axes_style(ax, 0, 1, ylabel="Share of planted errors caught" if ax is axes[0] else None)
        for j, ((name, f), (m, face, ls)) in enumerate(zip(READERS, styles)):
            lv = levels(f, kind)
            xs = [i + (j - 2) * 0.05 for i in range(len(LEVELS))]
            ys = [lv[v][0] for v, _ in LEVELS]
            print(f"calib[{kind}] {name.replace(chr(10), ' ')}: " + " ".join(f"{v}={lv[v][0]:.2f}" for v, _ in LEVELS))
            ax.plot(xs, ys, ls=ls, color=INK, lw=1.2, zorder=3)
            for x, (v, _) in zip(xs, LEVELS):
                y, lo, hi = lv[v]
                ax.errorbar(x, y, yerr=[[y - lo], [hi - y]], fmt=m, ms=6, mfc=face, mec=INK, ecolor=SOFT,
                            elinewidth=0.9, capsize=2, zorder=4)
        ax.set_xticks(range(len(LEVELS)))
        ax.set_xticklabels([lab for _, lab in LEVELS], fontsize=11)
        ax.set_xlim(-0.4, len(LEVELS) - 0.6)
        ax.set_title(title, fontsize=12)
    handles = [Line2D([], [], marker=m, ls=ls, color=INK, mfc=face, mec=INK, ms=6) for m, face, ls in styles]
    legend(fig, handles, [n.replace("\n", ", ") for n, _ in READERS], y=1.04, ncol=3)
    fig.subplots_adjust(bottom=0.15, top=0.78, wspace=0.08)
    save(fig, os.path.join(OUT, "part2-direct-fig-calibration.png"))


if __name__ == "__main__":
    fig_calibration()
    fig1()
    fig1("pointed", "part2-direct-fig-pointed.png", "Share caught when shown the passage")
    fig2()
    fig3()
