"""Setup figure for the Part 2 post, drawn like Part 1's design figure (scripts/blog_figures.py, fig_design): the ten
papers, the three versions with the hatches they have in the result figures, the planted errors, the three ways of
reading, and what is measured. No data is read.

Run: uv run --with matplotlib python scripts/blog_part2_setup.py
Writes docs/blog/part-2/fig-1-setup.png.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from averroes_figstyle import INK, plt, save  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "blog", "part-2")
HATCH = {"clar": "", "orig": "////", "garb": "xxxx"}   # as VERSIONS in blog_part2_final.py
FS, DY = 13, 0.6
BULLET = r"$\bullet$ "


def main():
    fig, ax = plt.subplots(figsize=(9, 10.1))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 20)
    ax.set_ylim(3.7, 25.5)
    ax.axis("off")

    def box(x, y, w, h, title, lines, swatches=None, fs=FS):
        ax.add_patch(Rectangle((x, y), w, h, facecolor="white", edgecolor=INK, lw=0.9))
        ax.text(x + 0.3, y + h - 0.35, title, ha="left", va="top", fontsize=fs + 1.5, fontname="cmb10")
        ty = y + h - 1.35
        for k, line in enumerate(lines):
            tx = x + 0.3
            sw = swatches[k] if swatches and k < len(swatches) else ""
            if sw:
                ax.add_patch(Rectangle((tx, ty - 0.2), 0.8, 0.42, facecolor="white", hatch=HATCH[sw], edgecolor=INK,
                                       lw=0.8))
            if sw != "":
                tx += 1.1          # text after a swatch, and its continuation lines (None), line up
            ax.text(tx, ty, line, ha="left", va="center", fontsize=fs)
            ty -= DY

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.0, mutation_scale=14))

    box(4.0, 23.4, 12.0, 2.0, "10 machine learning papers", ["Most on AI safety"])
    box(0.1, 14.6, 9.6, 8.1, "Three versions, same facts",
        ["Clarified: summary first,", "a glossary, each claim", "listed with its evidence",
         "Original", "Garbled: compressed to 2/3,", "terms swapped for codes,", "evidence moved away from",
         "claims, steps left unstated", "GPT-6.1 Sol writes both and", "checks that no fact is lost"],
        swatches=["clar", None, None, "orig", "garb", None, None, None])
    box(10.3, 14.6, 9.6, 8.1, "Planted errors",
        ["GPT-6.1 Sol plants one per copy,", "the same edit in every version:",
         BULLET + "wrong number or calculation", BULLET + "reversed comparison",
         BULLET + "claim stretched past the tests", BULLET + "association stated as cause",
         BULLET + "error carried into later results", "Kept only if Sol finds it in the",
         "clarified versions: 82 errors"])
    box(0.1, 7.7, 19.8, 6.1, "Four readers, three ways of reading",
        ["GPT-5.5, GPT-5 mini, GPT-5.6 Luna, GPT-5.4 mini"])
    for x, title, lines in [(0.5, "Whole paper", ["Review the paper and", "list any internal", "inconsistencies"]),
                            (6.9, "Every paragraph", ["Whole paper in view,", "asked about each", "paragraph in turn"]),
                            (13.3, "One section at a time", ["One pass, notes of at", "most 300 words, no", "looking back"])]:
        box(x, 8.1, 6.2, 3.3, title, lines, fs=12.5)
    box(4.0, 3.9, 12.0, 3.0, "Measure",
        [BULLET + "share of planted errors caught", BULLET + "false alarms: correct text flagged",
         "GPT-5 mini grades each review"])
    arrow(8.0, 23.4, 4.9, 22.6)
    arrow(12.0, 23.4, 15.1, 22.6)
    arrow(4.9, 14.6, 8.0, 13.8)
    arrow(15.1, 14.6, 12.0, 13.8)
    arrow(10.0, 7.7, 10.0, 6.9)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    save(fig, os.path.join(OUT, "fig-1-setup.png"))


if __name__ == "__main__":
    main()
