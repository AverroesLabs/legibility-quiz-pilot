"""Figures for the blog post. Usage: python scripts/blog_figures.py (needs matplotlib, numpy).

Reads data/analysis/v2-ci.json (scripts/quiz/ci_v2.py), v2-pooled.json, onestop.json and the published means of
Ryba et al. (2021) below. Writes docs/blog/fig-*.png.

Style: the Averroes Labs style guide where it does not cost readability. Black on white, Computer Modern (cmr10, the face Latin
Modern digitises), hatching instead of colour, no colour at all. Every data figure shows bars with 95% intervals (per-paper points can be switched on with SHOW_POINTS). The
axis limits are the limits of each scale, drawn bold. Figure messages live in the captions in the post, not in the
figures.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/blog"
INK, SOFT, QUIET = "#000000", "#4d4d4d", "#d6d6d6"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False, "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "hatch.linewidth": 0.6, "font.size": 12, "legend.fontsize": 11.5, "axes.linewidth": 0.8})

READERS = [("openai/gpt-5.6-sol", "GPT-5.6 Sol\n(strong reference)"), ("google/gemma-4-31b-it", "Gemma 4 31B"),
           ("mistralai/ministral-8b-2512", "Ministral 8B"), ("meta-llama/llama-3.2-3b-instruct", "Llama 3.2 3B")]
SHOW_POINTS = False  # Jason, 2026-11-04: per-paper points distracted more than they helped; bars and intervals only
SHOWN = READERS[:3]  # Llama sits near its floor; the rating figures leave it out, as the post does
# Bar styles: (label, face, hatch, edge). Hatching tells conditions apart.
BAR = {"none": ("No paper", "white", "....", SOFT), "orig": ("Original", "white", "", INK),
       "comp": ("Compressed", "white", "////", INK), "obf": ("Obfuscated", "white", "\\\\\\\\", INK),
       "compobf": ("Compressed + obfuscated", "white", "xxxx", INK),
       "imp": ("Fluent, key insight implicit", "white", "--", INK),
       "imp2": ("Fluent, key insight implicit", "white", "--", INK),  # Follow-up 3 replaces imp in figure 4
       "rem": ("Fluent, key results removed", "white", "------", INK),
       "Ele": ("Simplified (Elementary)", "white", "", INK), "Adv": ("Original, complex (Advanced)", "white", "xxxx", INK),
       "nopass": ("No passage", "white", "....", SOFT)}
# One look per condition throughout the post: no paper or passage = dots, original = plain, compressed = /,
# obfuscated = \, both = crosshatch. OneStopQA: the simplified text is plain (the easy
# baseline, like our original papers) and the complex original is crosshatched (the hardest text, like ours). The hidden-gap versions share horizontal lines that
# grow denser as more is taken from the paper: insight implicit (sparse), key results removed (dense).


def axes_style(ax, lo, hi, ylabel=None, bold_top=True):
    """Hairline axes; the scale's limits as bold lines (no top line for a scale without an upper limit); quiet
    horizontal grid."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_visible(False)
    ax.set_ylim(lo, hi)
    ax.axhline(lo, color=INK, lw=1.8, zorder=5, clip_on=False)
    if bold_top:
        ax.axhline(hi, color=INK, lw=1.8, zorder=5, clip_on=False)
    ax.grid(axis="y", color=QUIET, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0, labelsize=12)
    ax.tick_params(axis="y", direction="out", length=3, width=0.8, labelsize=11)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12)


def bar(ax, x, w, key, mean, ci, points, lo, hi, fmt="{:.2f}", seed=0):
    """One bar from the scale's floor, its 95% interval, its sample points and its value."""
    _, fc, hatch, ec = BAR[key]
    ax.add_patch(Rectangle((x - w / 2, lo), w, mean - lo, facecolor=fc, hatch=hatch, edgecolor=ec, lw=0.9, zorder=2))
    if points and SHOW_POINTS:
        jit = np.random.default_rng(seed).uniform(-w * 0.28, w * 0.28, len(points))
        ax.scatter(x + jit, points, s=6, facecolor="white", edgecolor=SOFT, linewidths=0.5, zorder=3)
    ax.errorbar(x, mean, yerr=[[mean - ci[0]], [ci[1] - mean]], fmt="none", ecolor=INK, elinewidth=1.4, capsize=3,
                capthick=1.4, zorder=4)
    # the value, always at the foot of its bar on a white label, so it never collides with points or the top line
    fs = 10 if w >= 0.24 else 8.5  # narrow bars (four per group) get smaller labels so they stay inside the bar
    ax.text(x, lo + 0.012 * (hi - lo), fmt.format(mean), ha="center", va="bottom", fontsize=fs, zorder=6,
            bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none"))


def legend(fig, keys, y=0.99, ncol=None):
    h = [Patch(facecolor=BAR[k][1], hatch=BAR[k][2], edgecolor=BAR[k][3], lw=0.9) for k in keys]
    labels = [BAR[k][0] for k in keys]
    if SHOW_POINTS:
        h.append(Line2D([], [], marker="o", ls="none", mfc="white", mec=SOFT, mew=0.5, ms=3.5))
        labels.append("One paper")
    fig.legend(h, labels, loc="upper center", bbox_to_anchor=(0.5, y),
               ncol=ncol or len(h), frameon=False, handlelength=1.6, columnspacing=1.2)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=200, facecolor="white")
    plt.close(fig)


def ci_data():
    return json.loads((ROOT / "data/analysis/v2-ci.json").read_text())


def fig_design():
    """The experiment as a top-to-bottom flow: papers split into the quiz (written and filtered on the original) and
    the five versions; both feed the readers; the grader scores the answers. Each version is shown with the bar
    pattern it has in the result figures."""
    fig, ax = plt.subplots(figsize=(9, 8.8))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 19.9)
    ax.axis("off")
    fs, dy = 13, 0.6
    bullet = r"$\bullet$ "

    def box(x, y, w, h, title, lines, swatches=None, lw=0.9):
        ax.add_patch(Rectangle((x, y), w, h, facecolor="white", edgecolor=INK, lw=lw))
        ax.text(x + 0.3, y + h - 0.35, title, ha="left", va="top", fontsize=fs + 1.5, fontname="cmb10")
        ty = y + h - 1.35
        for k, line in enumerate(lines):
            tx = x + 0.3
            sw = swatches[k] if swatches and k < len(swatches) else ""
            if sw:
                _, fc, hatch, ec = BAR[sw]
                ax.add_patch(Rectangle((tx, ty - 0.2), 0.8, 0.42, facecolor=fc, hatch=hatch, edgecolor=ec, lw=0.8))
            if sw != "":
                tx += 1.1  # text after a swatch, and its continuation lines (None), line up
            ax.text(tx, ty, line, ha="left", va="center", fontsize=fs)
            ty -= dy

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.0,
                                                                     mutation_scale=14))

    box(4.0, 17.9, 12.0, 1.95, "12 papers", ["Recent arXiv AI safety preprints, Jul to Sep 2026"])
    box(0.1, 9.6, 9.6, 7.6, "The quiz, written on the original",
        ["Write (Claude Opus 5.5):", bullet + "about 190 questions per paper", bullet + "answers implied, never stated",
         "", "Filter: keep a question only if", bullet + "the strong reader passes it",
         bullet + "readers fail it without the paper", bullet + "3 retrieved paragraphs fail",
         "2,154 written, 716 kept"])
    box(10.3, 9.6, 9.6, 7.6, "Five versions, same facts",
        ["No paper (the floor)", "Original", "Compressed to about 0.4", "of the length (Claude Opus 5.5)",
         "Obfuscated: scrambled order,", "nested sentences, opaque", "abbreviations (Kimi K3)",
         "Compressed + obfuscated", "Check: every number survives,", "the strong reader keeps 90%"],
        swatches=["none", "orig", "comp", None, "obf", None, None, "compobf"])
    box(4.0, 4.1, 12.0, 4.6, "Four readers take every question",
        [bullet + "GPT-5.6 Sol (strong reference)", bullet + "Gemma 4 31B", bullet + "Ministral 8B",
         bullet + "Llama 3.2 3B", "3 samples, every version"])
    box(4.0, 0.1, 12.0, 3.1, "Grade", ["GLM-5.3 checks 2 or 3 yes/no", "rubric points per answer,",
                                       "blind to reader and version"])
    arrow(8.0, 17.9, 4.9, 17.2)
    arrow(12.0, 17.9, 15.1, 17.2)
    arrow(4.9, 9.6, 8.0, 8.7)
    arrow(15.1, 9.6, 12.0, 8.7)
    arrow(10.0, 4.1, 10.0, 3.2)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    save(fig, "fig-1-design")


def fig_main():
    """Quiz score per reader and condition: pooled mean over 716 questions with its 95% interval (bootstrap over
    questions within papers), and each paper's mean as a point."""
    cq = ci_data()["quiz"]
    conds = ["none", "orig", "comp", "obf", "compobf"]
    fig, ax = plt.subplots(figsize=(9, 5.6))
    w = 0.16
    for g, (r, _) in enumerate(READERS):
        for k, c in enumerate(conds):
            q = cq[r][c]
            bar(ax, g + (k - 2) * w * 1.08, w, c, q["pooled"], q["pooled_ci95"], list(q["paper_mean_by_paper"].values()),
                0, 1, seed=g * 10 + k)
    axes_style(ax, 0, 1, "Quiz score (share of rubric points)")
    ax.set_xticks(range(len(READERS)))
    ax.set_xticklabels([n for _, n in READERS])
    ax.set_xlim(-0.5, len(READERS) - 0.5)
    legend(fig, conds, ncol=3)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.84, bottom=0.11)
    save(fig, "fig-2-main-result")


def two_panels(name, conds, quiz_key, quiz_ci_key, keys_note=None):
    """Left: model rating of how readable each version is for a human ML researcher (1 to 10), mean over papers with
    a 95% interval over papers and each paper's rating as a point. Right: quiz score with its 95% interval and each
    paper's score as a point."""
    c = ci_data()
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 5.2))
    n = len(conds)
    w = 0.8 / n
    for ax, kind in zip(axes, ("rating", "quiz")):
        for g, (r, _) in enumerate(SHOWN):
            for k, cond in enumerate(conds):
                x = g + (k - (n - 1) / 2) * w * 1.06
                if kind == "rating":
                    d = c["rating_human_prompt"][r][cond]
                    bar(ax, x, w, cond, d["mean"], d["ci95"], list(d["by_paper"].values()), 1, 10, "{:.1f}",
                        seed=g * 10 + k)
                else:
                    d = c["quiz"][r][cond]
                    bar(ax, x, w, cond, d[quiz_key], d[quiz_ci_key], list(d["paper_mean_by_paper"].values()), 0, 1,
                        seed=g * 10 + k)
        if kind == "rating":
            axes_style(ax, 1, 10, "Legibility rating (1 to 10)")
            ax.set_yticks(range(1, 11))
        else:
            axes_style(ax, 0, 1, "Quiz score (share of rubric points)")
        ax.set_xticks(range(len(SHOWN)))
        ax.set_xticklabels([n.split("\n")[0] for _, n in SHOWN])
        ax.set_xlim(-0.5, len(SHOWN) - 0.5)
    legend(fig, conds, ncol=len(conds))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.83, bottom=0.08, wspace=0.25)
    save(fig, name)


def fig_notice():
    two_panels("fig-3-rating-vs-quiz", ["orig", "comp", "obf", "compobf"], "pooled", "pooled_ci95")


def fig_gap():
    """Hidden-gap follow-ups, each quiz written for the text it audits. Left: legibility rating (1 to 10) of the
    original, the 'insight implicit' version and the 'key results removed' version (mean over papers, 95% interval
    over papers). Middle: quiz score: the original quiz on the original and on 'implicit' (which keeps every fact),
    and a quiz written and filtered on 'removed' (scripts/quiz/gap_requiz.py), taken with that version; mean over
    papers, 95% interval over questions within papers. 'Implicit' covers the 7 papers that passed its length check.
    Right: questions kept per paper, original quiz against the quiz written on 'removed' (11 comparable papers;
    2607.10112v1 left out because the writer's content filter cut its original quiz short), lines join each paper's
    two counts."""
    c = ci_data()
    conds = ["orig", "imp2", "rem"]  # imp2: Follow-up 3's stronger implicit version (all 12 papers), replacing imp
    fig, axes = plt.subplots(1, 3, figsize=(11, 5.2), gridspec_kw={"width_ratios": [3, 3, 1.5]})
    w = 0.8 / 3
    for ax, kind in zip(axes[:2], ("rating", "quiz")):
        for g, (r, _) in enumerate(SHOWN):
            for k, cond in enumerate(conds):
                x = g + (k - 1) * w * 1.06
                if kind == "rating":
                    d = c["rating_human_prompt"][r][cond]
                    bar(ax, x, w, cond, d["mean"], d["ci95"], list(d["by_paper"].values()), 1, 10, "{:.1f}",
                        seed=g * 10 + k)
                else:
                    d = c["quiz_requiz"][r]["orig"] if cond == "rem" else c["quiz"][r][cond]
                    bar(ax, x, w, cond, d["paper_mean"], d["paper_mean_ci95"], list(d["paper_mean_by_paper"].values()),
                        0, 1, seed=g * 10 + k)
        if kind == "rating":
            axes_style(ax, 1, 10, "Legibility rating (1 to 10)")
            ax.set_yticks(range(1, 11))
        else:
            axes_style(ax, 0, 1, "Quiz score, quiz written on that text")
        ax.set_xticks(range(len(SHOWN)))
        ax.set_xticklabels([n.split("\n")[0] for _, n in SHOWN])
        ax.set_xlim(-0.5, len(SHOWN) - 0.5)
    ax = axes[2]
    gap = json.loads((ROOT / "data/analysis/v2-gap-requiz.json").read_text())["kept"]
    pairs = [v for p, v in gap.items() if p != "2607.10112v1"]
    a, b = np.array([v[0] for v in pairs], float), np.array([v[1] for v in pairs], float)
    rng = np.random.default_rng(0)
    for k, (vals, key) in enumerate(((a, "orig"), (b, "rem"))):
        boots = vals[rng.integers(0, len(vals), (4000, len(vals)))].mean(1)
        bar(ax, k, 0.6, key, vals.mean(), list(np.percentile(boots, [2.5, 97.5])), None, 0, 100, "{:.0f}")
    if SHOW_POINTS:
        for x0, x1 in zip(a, b):
            ax.plot([0.12, 0.88], [x0, x1], color=SOFT, lw=0.6, marker="o", ms=3, mfc="white", mec=SOFT, mew=0.5,
                    zorder=3)
    axes_style(ax, 0, 100, "Questions kept per paper", bold_top=False)  # a count has no upper limit
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Original", "Results\nremoved"])
    ax.set_xlim(-0.55, 1.55)
    legend(fig, conds, ncol=3)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.86, bottom=0.12, wspace=0.3)
    save(fig, "fig-4-hidden-gap")


def fig_onestop():
    """OneStopQA (data/analysis/onestop.json): accuracy with a 95% interval over questions, and each of the 20
    articles' accuracy as a point."""
    d = json.loads((ROOT / "data/analysis/onestop.json").read_text())
    groups = [("humans", "Humans\n(Prolific)")] + [(r, n.split("\n")[0]) for r, n in READERS]
    levels = [("none", "nopass"), ("Ele", "Ele"), ("Adv", "Adv")]  # as figure 2: floor, plain baseline, harder text
    fig, ax = plt.subplots(figsize=(9, 5.2))
    w = 0.26
    for g, (key, _) in enumerate(groups):
        src = d["humans"] if key == "humans" else d["readers"][key]
        for k, (lv, style) in enumerate(levels):
            s = src[lv]
            bar(ax, g + (k - 1) * w * 1.06, w, style, s["acc"], s["ci95"], list(src["by_article"][lv].values()),
                0, 1, seed=g * 10 + k)
    axes_style(ax, 0, 1, "Accuracy (chance 0.25)")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([n for _, n in groups])
    ax.set_xlim(-0.5, len(groups) - 0.5)
    h = [Patch(facecolor=BAR[s][1], hatch=BAR[s][2], edgecolor=BAR[s][3], lw=0.9) for _, s in levels]
    fig.legend(h, [BAR[s][0] for _, s in levels], loc="upper center", bbox_to_anchor=(0.5, 0.99),
               ncol=3, frameon=False, handlelength=1.6, columnspacing=1.2)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.84, bottom=0.12)
    save(fig, "fig-6-onestop-humans")


# Ryba, Doubleday, Dry, Semmler and Connell (2021), Frontiers in Psychology 12:714321 (indexed in PubMed under the
# first author's earlier name, Freeling), means (SD) by version as published (raw data not public; searched OSF,
# Figshare, Zenodo, Dryad, GitHub and the Adelaide repository on 2026-11-04). Readability: 1-5 rating reported in %; understanding: % correct on 2
# multiple-choice questions per abstract. 347 readability and 628 understanding responses over 4 versions.
RYBA = {"levels": ["Easiest", "Easy", "Hard", "Hardest"],
            "readability": [(66.29, 29.94), (58.51, 28.53), (52.31, 26.27), (44.44, 29.44)],
            "understanding": [(57.87, 49.52), (39.47, 49.04), (40.79, 49.31), (47.95, 50.13)],
            "n_read": 347, "n_und": 628}


def fig_ryba():
    """Humans in Ryba et al. (2021) next to our readers: readability rating and comprehension, both 0 to 100, across
    four versions from most to least readable. Model ratings (1 to 10, human-reader prompt) rescaled as
    (r - 1) / 9 * 100 with a 95% interval over papers; quiz score x 100 with a 95% interval over questions. Human
    intervals: +-1.96 SD / sqrt(n / 4), assuming equal n per version. Lines, not bars: the message is two slopes, and
    the human study published only means."""
    c = ci_data()
    vers = ["orig", "comp", "obf", "compobf"]
    resc = lambda v: (v - 1) / 9 * 100  # noqa: E731
    fig, axes = plt.subplots(1, 5, figsize=(10, 4.6), sharey=True)
    x = np.arange(4)
    for ax, (key, title) in zip(axes, [("humans", "Humans (Ryba 2021)")] +
                                [(r, n.split("\n")[0]) for r, n in READERS]):
        if key == "humans":
            series = {s: ([v[0] for v in RYBA[s]], [1.96 * v[1] / np.sqrt(n / 4) for v in RYBA[s]])
                      for s, n in (("readability", RYBA["n_read"]), ("understanding", RYBA["n_und"]))}
            rate, rerr = series["readability"]
            comp, qerr = series["understanding"]
            labels = ["Easiest", "Easy", "Hard", "Hardest"]
        else:
            cr, cq = c["rating_human_prompt"][key], c["quiz"][key]
            rate = [resc(cr[v]["mean"]) for v in vers]
            comp = [100 * cq[v]["pooled"] for v in vers]
            rerr = [[r - resc(cr[v]["ci95"][0]) for r, v in zip(rate, vers)],
                    [resc(cr[v]["ci95"][1]) - r for r, v in zip(rate, vers)]]
            qerr = [[q - 100 * cq[v]["pooled_ci95"][0] for q, v in zip(comp, vers)],
                    [100 * cq[v]["pooled_ci95"][1] - q for q, v in zip(comp, vers)]]
            labels = ["Original", "Compressed", "Obfuscated", "Both"]  # caption: Both = compressed + obfuscated
        ax.errorbar(x, rate, yerr=rerr, color=INK, lw=1.4, marker="o", ms=5, mfc=INK, capsize=2.5, zorder=3)
        ax.errorbar(x + 0.06, comp, yerr=qerr, color=INK, lw=1.4, ls="--", marker="s", ms=5, mfc="white", capsize=2.5,
                    zorder=3)
        yr, yc = rate[-1], comp[-1]
        if abs(yr - yc) < 9:  # keep the two end labels apart
            mid, sgn = (yr + yc) / 2, 1 if yr >= yc else -1
            yr, yc = mid + 5 * sgn, mid - 5 * sgn
        ax.text(3.3, yr, f"{rate[-1] - rate[0]:+.0f}", fontsize=11, va="center")
        ax.text(3.3, yc, f"{comp[-1] - comp[0]:+.0f}", fontsize=11, va="center")
        axes_style(ax, 0, 100)
        ax.tick_params(axis="x", labelsize=10.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=40, ha="right", rotation_mode="anchor")
        ax.set_xlim(-0.3, 4.2)
        ax.set_title(title, fontsize=11.5, loc="left")
    axes[0].set_ylabel("Score (0 to 100)", fontsize=12)
    fig.supxlabel("Version, from most to least readable", fontsize=12, y=0.01)
    fig.legend([Line2D([], [], color=INK, lw=1.4, marker="o", ms=5),
                Line2D([], [], color=INK, lw=1.4, ls="--", marker="s", ms=5, mfc="white"),
                Line2D([], [], ls="none")],
               ["Readability rating", "Comprehension score", "Numbers: change from the first version to the last"],
               loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=3, frameon=False, handlelength=1.8)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.8, bottom=0.27, wspace=0.12)
    save(fig, "fig-5-ryba-vs-models")


if __name__ == "__main__":
    for f in (fig_design, fig_main, fig_notice, fig_gap, fig_onestop, fig_ryba):
        f()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
