"""Bar-chart figures for the Part 2 blog post. Each reader gets three bars: clarified original, original and garbled.
Readers run strongest to weakest, left to right. Every figure is pooled over all ten papers (82 primary errors).
Fig 1: experiment 11, whole paper. Fig 2: experiment 12, every paragraph, computed with scripts/v4/exp12_paragraphs.py
(caught: boot over errors within papers; false alarms per 1,000 words: fa_boot over paragraphs within papers).
Fig 3: experiment 9 section by section (300-word notes), calibration-paper files plus their -anchors siblings.
fig_oracle (experiment 11 pointed) is no longer in the post.
Full numbers and provenance: docs/part2-agent-report.md.

Run: uv run --with matplotlib python scripts/blog_part2_final.py
Writes docs/blog/part-2/fig-1-whole-paper.png, fig-2-every-paragraph.png and fig-3-section-by-section.png.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from blog_part2_direct import ALL, ANCHORS, CAL, OUT, levels, summary  # noqa: E402
from averroes_figstyle import *  # noqa: E402,F403

MOCK = "#f4c7c3"            # light red: mock placeholder, not measured
# order clarified, original, garbled, as in the Part 1 post
VERSIONS = [("clar-explainer-orig", "", "Clarified original"), ("orig", "////", "Original paper"),
            ("obf-max", "xxxx", "Garbled version")]


def measured(fname, kind="planted", papers=None):
    """[(value, ci, is_mock)] in VERSIONS order, all measured."""
    lv = levels(fname, kind, papers)
    return [(lv[v][0], (lv[v][1], lv[v][2]), False) for v, _, _ in VERSIONS]


def measured_two(fname, clar_mock, kind="planted"):
    """Garbled and original measured; clarified original a mock placeholder. VERSIONS order."""
    s = summary(fname, kind)
    return [(clar_mock, None, True), (s["orig"], s["orig:ci"], False), (s["obf-max"], s["obf-max:ci"], False)]


# experiment 11 readers (whole paper), measured on the 4 calibration papers
# order: strongest to weakest reader, left to right (as in Part 1), by catch rate on originals with the whole paper
W11 = [("GPT-5.5\nlow effort", "gpt55-low"), ("GPT-5 mini\nhigh effort", "mini-high"),
       ("GPT-5.6 Luna\nlow effort", "luna-low"), ("GPT-5.4 mini\nmedium effort", "mini54-med")]


def mock(g, o, c):
    """Arguments garbled, original, clarified; returned in VERSIONS order."""
    return [(c, None, True), (o, None, True), (g, None, True)]


def draw(ax, groups, ylabel=None, hi=1):
    axes_style(ax, 0, hi, ylabel=ylabel)
    w = 0.26
    for gi, (_, bars) in enumerate(groups):
        for i, ((v, ci, is_mock), (_, hatch, _)) in enumerate(zip(bars, VERSIONS)):
            if v is None:          # not run (e.g. GPT-5.5 every paragraph skipped the clarified original for cost)
                continue
            bar(ax, gi + (i - 1) * w, w * 0.92, v, ci=ci, face=MOCK if is_mock else PAPER, hatch=hatch,
                label="{:.2f}")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([n for n, _ in groups], fontsize=10.5)
    ax.set_xlim(-0.6, len(groups) - 0.4)


def legend_all(fig, y=1.03, mock=True):
    handles = [swatch(hatch=h) for _, h, _ in VERSIONS] + ([swatch(face=MOCK)] if mock else [])
    legend(fig, handles, [lab for _, _, lab in VERSIONS] + (["Mock placeholder"] if mock else []), y=y)


def fig1():
    """Result 1: whole paper in context, review unaided. All 10 development and reviewed papers pooled (82 errors);
    nothing was tuned on the first four, so they are not kept apart."""
    papers = tuple(CAL) + tuple(ANCHORS)
    fig, ax = plt.subplots(figsize=(10, 4.6))
    groups = [(n, measured(f"reader-w11-{rd}.jsonl", "planted", papers)) for n, rd in W11]
    draw(ax, groups, "Share of planted errors found")
    legend_all(fig, mock=False)
    fig.subplots_adjust(bottom=0.17, top=0.84)
    save(fig, os.path.join(OUT, "part-2/fig-1-whole-paper.png"))


def ms(v, lo, hi):
    return (v, (lo, hi), False)


P12 = [("GPT-5.5\nlow effort", "p12-gpt55-low"), ("GPT-5 mini\nhigh effort", "p12-mini-high"),
       ("GPT-5.6 Luna\nlow effort", "p12-luna-low"), ("GPT-5.4 mini\nmedium effort", "p12-mini54-med")]


def exp12_bars(rd):
    """([caught bars], [false-alarm bars]) in VERSIONS order for one experiment 12 reader, all ten papers."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "v4"))
    import exp12_paragraphs as X
    rows = X.load(rd)
    s = X.boot(X.caught_by_error(rows))
    caught = [ms(s[v], *s[v + ":ci"]) for v, _, _ in VERSIONS]
    fa = [ms(X.fa_rate(rows, v, True), *X.fa_boot(rows, v, True)) for v, _, _ in VERSIONS]
    return caught, fa


def fig2():
    """Result 2: whole paper in context, asked about every paragraph in turn (experiment 12), four readers, all ten
    papers. Left: planted errors caught. Right: false alarms, non-planted paragraphs flagged per 1,000 words
    (exploratory interval: bootstrap over paragraphs within papers)."""
    bars = {n: exp12_bars(rd) for n, rd in P12}
    for n, (c, f) in bars.items():
        print("fig2", n.replace("\n", " "), [round(b[0], 2) for b in c], [round(b[0], 2) for b in f])
    fig, axes = plt.subplots(2, 1, figsize=(10, 8.4), gridspec_kw={"hspace": 0.45})
    draw(axes[0], [(n, bars[n][0]) for n, _ in P12], "Share caught")
    axes[0].set_title("Planted errors caught", fontsize=12)
    draw(axes[1], [(n, bars[n][1]) for n, _ in P12], "Per 1,000 words", hi=5)
    axes[1].set_title("False alarms", fontsize=12)
    legend_all(fig, y=1.0, mock=False)
    fig.subplots_adjust(bottom=0.08, top=0.9)
    save(fig, os.path.join(OUT, "part-2/fig-2-every-paragraph.png"))


def fig_oracle():
    """Special case of Result 2, shown later and smaller: the reader is told which passage holds the error
    (experiment 11 pointed check, 4 calibration papers). An upper bound, not usable on a new paper."""
    groups = [(n, measured(f"reader-w11-{rd}.jsonl", "pointed", CAL)) for n, rd in W11]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    draw(ax, groups, "Share caught")
    handles = [swatch(hatch=h) for _, h, _ in VERSIONS]
    legend(fig, handles, [lab for _, _, lab in VERSIONS], y=1.03)
    fig.subplots_adjust(bottom=0.2, top=0.82)
    save(fig, os.path.join(OUT, "part2-final-fig-oracle.png"))


def fig3():
    """Result 3: reading one section at a time with notes of at most 300 words, four readers, all ten papers."""
    files = [("GPT-5.5\nlow effort", ["reader-gpt55-A.jsonl", "reader-gpt55-A-anchors.jsonl",
                                       "reader-gpt55-A-anchors-clar.jsonl"]),
             ("GPT-5 mini\nhigh effort", ["reader-mini-high-A.jsonl", "reader-mini-high-A-anchors.jsonl"]),
             ("GPT-5.6 Luna\nlow effort", ["reader-luna-low-A.jsonl", "reader-luna-low-A-anchors.jsonl"]),
             ("GPT-5.4 mini\nmedium effort", ["reader-mini54-A.jsonl", "reader-mini54-A-anchors.jsonl"])]
    groups = [(n, measured(f, papers=ALL)) for n, f in files]
    for n, b in groups:
        print("fig3", n.replace("\n", " "), [round(x[0], 2) for x in b])
    fig, ax = plt.subplots(figsize=(10, 4.6))
    draw(ax, groups, "Share of planted errors found")
    legend_all(fig, mock=False)
    fig.subplots_adjust(bottom=0.17, top=0.84)
    save(fig, os.path.join(OUT, "part-2/fig-3-section-by-section.png"))


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    # fig_oracle()   # not in the post
    print("wrote docs/blog/part-2/fig-1..3")
