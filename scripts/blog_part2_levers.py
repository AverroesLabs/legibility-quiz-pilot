"""Levers figure for the Part 2 blog post: what each change we tried did on the calibration map, pooled over all ten
papers (82 primary errors). Each panel is the map (catch rate on originals vs garbled), zoomed, with the
diagonal and target corner; arrows run from the base setting to the changed one. No API calls.

Run: uv run --with matplotlib python scripts/blog_part2_levers.py
Writes docs/blog/part-2/fig-5-levers.png and prints the points.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from blog_part2_direct import ALL, D, OUT, SURV, summary  # noqa: E402
from averroes_figstyle import *  # noqa: E402,F403
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

TARGET_X, TARGET_Y = 0.70, 0.35
LO = 0.0
W = {"GPT-5.4 mini": "mini54-med", "GPT-5.6 Luna": "luna-low", "GPT-5 mini": "mini-high", "GPT-5.5": "gpt55-low"}
A = {"GPT-5.4 mini": ["reader-mini54-A.jsonl", "reader-mini54-A-anchors.jsonl"],
     "GPT-5.6 Luna": ["reader-luna-low-A.jsonl", "reader-luna-low-A-anchors.jsonl"],
     "GPT-5 mini": ["reader-mini-high-A.jsonl", "reader-mini-high-A-anchors.jsonl"],
     "GPT-5.5": ["reader-gpt55-A.jsonl", "reader-gpt55-A-anchors.jsonl"]}
ERR = {(e["paper"], e["id"]): e for e in map(json.loads, open(os.path.join(D, "errors.jsonl")))}


def pt(fname, kind="planted", keys=None):
    s = summary(fname, kind, ALL) if keys is None else subset(fname, kind, keys)
    return s["orig"], s["obf-max"]


def subset(fname, kind, keys):
    from blog_part2_direct import per_error
    pe = {k: v for k, v in per_error(fname, kind, ALL).items() if k in keys}
    return {"orig": sum(d["orig"] for d in pe.values()) / len(pe),
            "obf-max": sum(d["obf-max"] for d in pe.values()) / len(pe)}


def base(ax, title):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlim(LO, 1), ax.set_ylim(LO, 1)
    ax.set_aspect("equal")
    ax.grid(color=QUIET, lw=0.6)
    ax.set_axisbelow(True)
    ax.plot([0, 1], [0, 1], color=SOFT, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.add_patch(Rectangle((TARGET_X, LO), 1 - TARGET_X, TARGET_Y - LO, facecolor=(0.36, 0.72, 0.36, 0.18), edgecolor=(0.25, 0.6, 0.25, 0.7), lw=1,
                           zorder=0))
    ax.set_title(title, fontsize=14)
    ax.tick_params(labelsize=11)
    ax.set_xticks([0, 0.5, 1]), ax.set_yticks([0, 0.5, 1])


def arrow(ax, a, b, label=None, off=(0.01, -0.03), fs=12):
    ax.plot(*a, marker="o", ms=4.5, mfc=PAPER, mec=INK, zorder=4)
    ax.plot(*b, marker="o", ms=5.5, mfc=INK, mec=INK, zorder=4)
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=10, color=INK, lw=1.1,
                                 shrinkA=3, shrinkB=4, zorder=3))
    if label:
        ax.text(b[0] + off[0], b[1] + off[1], label, fontsize=fs, zorder=5)


def dots(ax, pts):
    for label, (x, y), (dx, dy) in pts:
        ax.plot(x, y, marker="o", ms=5.5, mfc=INK, mec=INK, zorder=4)
        ax.text(x + dx, y + dy, label, fontsize=12, zorder=5)
    xs, ys = zip(*[p for _, p, _ in pts])
    ax.plot(xs, ys, color=INK, lw=0.8, zorder=2)


def main():
    fig, axes = plt.subplots(2, 3, figsize=(11, 7.6))
    ax = axes[0, 0]
    base(ax, "Which reader")
    off = {"GPT-5.4 mini": (0.03, -0.07), "GPT-5.6 Luna": (-0.5, 0.03), "GPT-5 mini": (-0.36, 0.05),
           "GPT-5.5": (-0.12, -0.09)}
    dots(ax, [(n, pt(f"reader-w11-{rd}.jsonl"), off[n]) for n, rd in W.items()])
    print("reader", {n: pt(f"reader-w11-{rd}.jsonl") for n, rd in W.items()})

    ax = axes[0, 1]
    base(ax, "Whole paper to sections")
    for n, rd in W.items():
        a, b = pt(f"reader-w11-{rd}.jsonl"), pt(A[n])
        print("view", n, a, b)
        arrow(ax, a, b, n, off={"GPT-5.5": (-0.05, -0.09), "GPT-5.4 mini": (-0.37, 0.05),
                                "GPT-5.6 Luna": (0.04, -0.06), "GPT-5 mini": (-0.36, 0.04)}[n])

    ax = axes[0, 2]
    base(ax, "GPT-5.5: smaller notes")
    pts = [("1,200 words", pt("reader-gpt55-A1200.jsonl"), (-0.3, 0.05)),
           ("600", pt("reader-gpt55-A600.jsonl"), (-0.13, -0.01)),
           ("300", pt(A["GPT-5.5"]), (-0.15, -0.02))]
    print("notes", [(l, p) for l, p, _ in pts])
    arrow(ax, pts[0][1], pts[1][1]), arrow(ax, pts[1][1], pts[2][1])
    for l, p, o in pts:
        ax.text(p[0] + o[0], p[1] + o[1], l, fontsize=12)

    ax = axes[1, 0]
    base(ax, "Told where the error is")
    for n, rd in W.items():
        a, b = pt(f"reader-w11-{rd}.jsonl"), pt(f"reader-w11-{rd}.jsonl", "pointed")
        print("pointed", n, a, b)
        arrow(ax, a, b)
    ax.text(0.04, 0.78, "every reader moves\nto the top corner", fontsize=12, color=SOFT)

    lev = {lv: {k for k in SURV if k[0] in ALL and ERR[k]["level"] == lv} for lv in ("easy", "medium", "hard")}
    ax = axes[1, 1]
    base(ax, "GPT-5.5: error difficulty")
    dots(ax, [(lv, pt("reader-w11-gpt55-low.jsonl", keys=k), {"easy": (-0.08, 0.04), "medium": (-0.3, -0.04),
                                                                  "hard": (-0.2, -0.03)}[lv]) for lv, k in lev.items()])
    print("level gpt55", {lv: pt("reader-w11-gpt55-low.jsonl", keys=k) for lv, k in lev.items()})

    # 3 errors on 2509.11379v3 have no measured distance (word_distance found no evidence span); left out here
    dist = {k: ERR[k]["distance_words"] for k in SURV if k[0] in ALL and ERR[k]["distance_words"] is not None}
    ds = sorted(dist.values())
    t1, t2 = ds[len(ds) // 3], ds[2 * len(ds) // 3]
    bins = {"near": (-1, t1), "middle": (t1, t2), "far": (t2, 1e9)}
    ax = axes[1, 2]
    base(ax, "GPT-5.5: evidence distance")
    keys = {b: {k for k, d in dist.items() if lo < d <= hi} for b, (lo, hi) in bins.items()}
    dots(ax, [(b, pt("reader-w11-gpt55-low.jsonl", keys=k), {"near": (-0.12, -0.08), "middle": (-0.28, -0.03), "far": (0.02, 0.02)}[b]) for b, k in keys.items()])
    print("distance gpt55", t1, t2, {b: pt("reader-w11-gpt55-low.jsonl", keys=k) for b, k in keys.items()})

    for a in axes[1]:
        a.set_xlabel("Caught in originals", fontsize=13)
    for a in axes[:, 0]:
        a.set_ylabel("Caught in garbled", fontsize=13)
    fig.subplots_adjust(wspace=0.3, hspace=0.3, top=0.96, bottom=0.07)
    save(fig, os.path.join(OUT, "part-2/fig-5-levers.png"))


if __name__ == "__main__":
    main()
