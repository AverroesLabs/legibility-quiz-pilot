"""Calibration map for the Part 2 blog post: each reader setting as a point, catch rate on text people can follow
(x, the original papers) against catch rate on text people cannot (y, the garbled versions). The diagonal means no
discrimination; the shaded corner is the target. All points are measured and pooled over all ten papers (82 errors):
experiment 11 (whole paper), experiment 12 (every paragraph) and experiment 9 (one section at a time, 300-word notes).
Region labels are schematic; measured moves per lever are in blog_part2_levers.py.

Run: uv run --with matplotlib python scripts/blog_part2_calibration_map.py
Writes docs/blog/part-2/fig-4-calibration-map.png.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from blog_part2_direct import ALL, OUT, summary
from blog_part2_final import exp12_bars  # noqa: E402
from averroes_figstyle import *  # noqa: E402,F403
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

MOCK = "#f4c7c3"
TARGET_X, TARGET_Y = 0.70, 0.35          # proposed: catch at least 0.70 on legible text, at most 0.35 on illegible

# (label, x, y, marker, measured?, label offset); all pooled over the ten papers
SEC = {"gpt55-low": ["reader-gpt55-A.jsonl", "reader-gpt55-A-anchors.jsonl"],
       "mini-high": ["reader-mini-high-A.jsonl", "reader-mini-high-A-anchors.jsonl"],
       "luna-low": ["reader-luna-low-A.jsonl", "reader-luna-low-A-anchors.jsonl"],
       "mini54-med": ["reader-mini54-A.jsonl", "reader-mini54-A-anchors.jsonl"]}
SHAPE = {"GPT-5.5": "o", "GPT-5 mini": "s", "GPT-5.6 Luna": "^", "GPT-5.4 mini": "D"}
METHOD = {"o": ("Whole paper", INK), "^": ("Asked every paragraph", "#4E6E8E"), "s": ("One section at a time", "#C8A26B")}
NAME = {"gpt55-low": "GPT-5.5", "mini-high": "GPT-5 mini", "luna-low": "GPT-5.6 Luna", "mini54-med": "GPT-5.4 mini"}
OFF = {"o": {"gpt55-low": (0.02, -0.012), "mini-high": (-0.19, 0.012), "luna-low": (0.02, -0.008), "mini54-med": (-0.18, -0.008)},
       "s": {"gpt55-low": (-0.035, -0.045), "mini-high": (-0.18, -0.012), "luna-low": (0.025, -0.02), "mini54-med": (0.02, -0.012)},
       "^": {"gpt55-low": (-0.14, -0.018), "mini-high": (-0.07, 0.022), "luna-low": (-0.19, -0.012), "mini54-med": (0.02, -0.006)}}


def points():
    xy = lambda s: (s["orig"], s["obf-max"])   # noqa: E731
    pts = []
    for rd in NAME:
        pts.append((NAME[rd], *xy(summary(f"reader-w11-{rd}.jsonl", "planted", ALL)), "o", True, OFF["o"][rd]))
        pts.append((NAME[rd], *xy(summary(SEC[rd], "planted", ALL)), "s", True, OFF["s"][rd]))
        c, _ = exp12_bars(f"p12-{rd}")
        pts.append((NAME[rd], c[1][0], c[2][0], "^", True, OFF["^"][rd]))   # VERSIONS order: clarified, original, garbled
    for p in pts:
        print(p[3], p[0], round(p[1], 2), round(p[2], 2))
    return pts


def main():
    fig, ax = plt.subplots(figsize=(8.2, 7.4))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlim(0, 1), ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("Share of planted errors caught in text people can follow\n(original papers)")
    ax.set_ylabel("Share caught in text people cannot follow\n(garbled versions)")
    ax.grid(color=QUIET, lw=0.6)
    ax.set_axisbelow(True)
    ax.plot([0, 1], [0, 1], color=SOFT, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.add_patch(Rectangle((TARGET_X, 0), 1 - TARGET_X, TARGET_Y, facecolor=(0.36, 0.72, 0.36, 0.18), edgecolor=(0.25, 0.6, 0.25, 0.7), lw=1.2, zorder=0))
    ax.text(0.85, 0.17, "Target", ha="center", va="center", fontsize=11)
    ax.text(0.30, 0.92, "Garbled text is easier:\ncheck how it was built", ha="center", fontsize=10.5, color=SOFT)
    ax.text(0.62, 0.97, "Reader sees through bad writing", ha="center", fontsize=10.5, color=SOFT)
    ax.text(0.13, 0.03, "Reader too weak,\nor errors too hard", ha="center", fontsize=10.5, color=SOFT)
    ax.text(0.2, 0.225, "No discrimination", rotation=45, ha="center", va="center", fontsize=10.5, color=SOFT)
    # how the model read = fill (black, slate, ochre: distinct in grayscale); which model = marker shape
    for label, x, y, method, _, _ in points():
        ax.plot(x, y, marker=SHAPE[label], ms=10, mfc=METHOD[method][1], mec=INK, mew=1.1, ls="", zorder=4)
    legend(fig, [swatch(face=c) for c, _ in [(v[1], 0) for v in METHOD.values()]],
           [v[0] for v in METHOD.values()], y=1.0, ncol=3)
    models = [Line2D([], [], marker=SHAPE[n], ls="", mfc="#d4d4d4", mec=INK, mew=1.1, ms=10) for n in NAME.values()]
    legend(fig, models, list(NAME.values()), y=0.955, ncol=4)
    fig.subplots_adjust(top=0.86, bottom=0.13)
    save(fig, os.path.join(OUT, "part-2/fig-4-calibration-map.png"))


if __name__ == "__main__":
    main()
