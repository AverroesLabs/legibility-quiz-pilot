"""Figures for docs/pilot-report.md, regenerated from the raw run data.

Usage: python scripts/quiz/make_figures.py   (needs matplotlib and numpy)
Writes light and dark PNGs to docs/figures/. Uncertainty is a 95% bootstrap interval over items, as in the report.
"""
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RUNS, OUT = ROOT / "data/runs", ROOT / "docs/figures"
GRADER = "z-ai/glm-5.3"
SEED, B = 20261102, 10000

THEMES = {
    "light": dict(surface="#fcfcfb", text="#0b0b0b", muted="#52514e", grid="#e6e5e0", mid="#f0efec",
                  series=["#2a78d6", "#eb6834", "#1baf7a", "#eda100"], red="#e34948"),
    "dark": dict(surface="#1a1a19", text="#f0efec", muted="#c3c2b7", grid="#33332f", mid="#383835",
                 series=["#3987e5", "#d95926", "#199e70", "#c98500"], red="#e66767"),
}

SHORT = {"openai/gpt-5.6-sol": "GPT-5.6 Sol (smart)", "z-ai/glm-5.3": "GLM-5.3 (smart)",
         "mistralai/ministral-8b-2512": "Ministral 8B", "google/gemma-4-31b-it": "Gemma 4 31B",
         "meta-llama/llama-3.2-3b-instruct": "Llama 3.2 3B", "ibm-granite/granite-4.2-8b": "Granite 4.2 8B",
         "qwen/qwen3.5-9b": "Qwen3.5 9B", "mistralai/ministral-14b-2512": "Ministral 14B",
         "openai/gpt-oss-20b": "gpt-oss-20b", "qwen/qwen3.8-27b": "Qwen3.8 27B"}
PILOT_READERS = ["openai/gpt-5.6-sol", "google/gemma-4-31b-it", "mistralai/ministral-8b-2512"]
# Valid tests first: papers where removing the section measurably hurt its items.
PAPER_ORDER = ["2608.25490v1", "2609.00755v1", "2607.20379v2", "2608.19216v1", "2607.28685v1", "2607.17117v2"]
N_VALID = 2


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def item_means(grades, reader, cond, items=None, grader=GRADER):
    per = defaultdict(list)
    for g in grades:
        if g["grader"] == grader and g["reader"] == reader and g["cond"] == cond:
            if items is None or g["item"] in items:
                per[g["item"]].append(g["score"])
    return {i: statistics.mean(v) for i, v in per.items()}


def mean_ci(values):
    xs = np.array(list(values), dtype=float)
    if len(xs) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(SEED)
    boots = xs[rng.integers(0, len(xs), size=(B, len(xs)))].mean(axis=1)
    return xs.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def style(fig, axes, t):
    fig.patch.set_facecolor(t["surface"])
    for ax in np.atleast_1d(axes).ravel():
        ax.set_facecolor(t["surface"])
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(t["grid"])
        ax.tick_params(colors=t["muted"], labelsize=9, length=0)
        ax.xaxis.label.set_color(t["muted"])
        ax.yaxis.label.set_color(t["muted"])
        ax.title.set_color(t["text"])


def header(fig, t, title, subtitle):
    fig.text(0.01, 0.985, title, ha="left", va="top", fontsize=13, weight="bold", color=t["text"])
    fig.text(0.01, 0.935, subtitle, ha="left", va="top", fontsize=9.5, color=t["muted"])


def save(fig, name, theme):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}-{theme}.png", dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)


def legend(ax, t, y=0.875, **kw):
    """Legend in its own row under the subtitle, so it never covers data or the header."""
    fig = ax.get_figure()
    handles, labels = [], []
    for a in fig.axes:
        for h, l in zip(*a.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    kw.pop("loc", None)
    kw.setdefault("ncol", len(labels))
    return fig.legend(handles, labels, frameon=False, fontsize=9, labelcolor=t["text"], loc="upper left",
                      bbox_to_anchor=(0.005, y), handlelength=1.5, columnspacing=1.6, **kw)


# ---------- Pilot ----------

def pilot():
    rows = {r["paper"]: r for r in json.loads((RUNS / "pilot-analysis.json").read_text())}
    return [rows[p] for p in PAPER_ORDER]


def section_name(title):
    words = title.split()
    if words and words[0].rstrip(".").isdigit():
        words = words[1:]
    name = " ".join(words)
    return name if len(name) <= 30 else name[:28] + "..."


def fig_obfuscation(t, theme):
    papers = pilot()
    fig, axes = plt.subplots(1, 3, figsize=(12, 6.2), sharey=True)
    style(fig, axes, t)
    header(fig, t, "Removing a section hurts the questions about it; making it hard to read does not",
           "Drop in score on kept items about the changed section (clear minus changed). Dots: mean; lines: 95% interval "
           "over items.\nTop 2 papers are valid tests. Shaded: the content is repeated elsewhere, so removal does not hurt "
           "and no measure could detect a change.")
    labels = [f"{p['paper']}\n{section_name(p['obf_title'])}\n{len(p['items_tagged_obf'])} item"
              f"{'s' if len(p['items_tagged_obf']) != 1 else ''}" for p in papers]
    for ax, reader in zip(axes, PILOT_READERS):
        ax.axvline(0, color=t["muted"], lw=1)
        ax.axhspan(N_VALID - 0.5, len(papers) - 0.5, color=t["mid"], zorder=0, lw=0)
        for y, p in enumerate(papers):
            tag = p["tagged"][reader]
            for key, dy, color, name in (("clear_minus_rem", -0.16, t["series"][0], "Section removed"),
                                         ("clear_minus_obf", 0.16, t["series"][1], "Section made hard to read")):
                est, lo, hi = tag[key]
                ax.plot([lo, hi], [y + dy, y + dy], color=color, lw=2, solid_capstyle="round")
                ax.plot(est, y + dy, "o", ms=8, color=color, mec=t["surface"], mew=2,
                        label=name if y == 0 else None)
        ax.set_title(SHORT[reader], fontsize=10.5, loc="left")
        ax.set_xlim(-0.5, 1.05)
        ax.set_xticks([-0.25, 0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xlabel("Score drop (0 = no effect)")
        ax.grid(axis="x", color=t["grid"], lw=0.8)
        ax.set_axisbelow(True)
    axes[0].set_yticks(range(len(papers)))
    axes[0].set_yticklabels(labels, fontsize=8.5, color=t["text"])
    axes[0].invert_yaxis()
    legend(axes[2], t, y=0.855)
    fig.subplots_adjust(left=0.23, right=0.99, top=0.76, bottom=0.11, wspace=0.08)
    save(fig, "pilot-obfuscation-vs-removal", theme)


def fig_measures(t, theme):
    papers = pilot()
    ms = ["M1", "M2", "M3", "M4", "M5"]
    names = ["M1\ngain", "M2\nmean", "M3\ncoverage", "M4\nweakest\nsection", "M5\nintegration"]
    cmap = LinearSegmentedColormap.from_list("div", [t["series"][0], t["mid"], t["red"]])
    fig, axes = plt.subplots(1, 3, figsize=(12, 5.2), sharey=True)
    style(fig, axes, t)
    header(fig, t, "No measure detects the hard-to-read section on more than 1 of 6 papers (Bar 1 needed 4)",
           "Each cell: clear minus hard-to-read score for one paper and measure. Red: the change lowered the score. "
           "Outlined: counted as a detection.")
    for ax, reader in zip(axes, PILOT_READERS):
        grid = np.full((len(papers), len(ms)), np.nan)
        hits = []
        for y, p in enumerate(papers):
            for x, m in enumerate(ms):
                d = p["readers"][reader]["detect"].get(m) or {}
                if d.get("diff") is not None:
                    grid[y, x] = d["diff"]
                    if d.get("detects"):
                        hits.append((x, y))
        ax.imshow(grid, cmap=cmap, vmin=-0.5, vmax=0.5, aspect="auto")
        for y in range(len(papers)):
            for x in range(len(ms)):
                v = grid[y, x]
                ax.text(x, y, "n/a" if np.isnan(v) else f"{v:+.2f}", ha="center", va="center", fontsize=8.5,
                        color=t["muted"] if np.isnan(v) else t["text"])
        for x, y in hits:
            ax.add_patch(plt.Rectangle((x - 0.47, y - 0.47), 0.94, 0.94, fill=False, ec=t["text"], lw=2))
        ax.axhline(N_VALID - 0.5, color=t["surface"], lw=4)
        ax.set_xticks(range(len(ms)))
        ax.set_xticklabels(names, fontsize=8, color=t["text"])
        ax.set_title(f"{SHORT[reader]}: {len(hits)} detection{'s' if len(hits) != 1 else ''}", fontsize=10.5,
                     loc="left")
        ax.spines[:].set_visible(False)
    axes[0].set_yticks(range(len(papers)))
    axes[0].set_yticklabels([p["paper"] + (" (valid)" if i < N_VALID else "") for i, p in enumerate(papers)],
                            fontsize=8.5, color=t["text"])
    fig.subplots_adjust(left=0.12, right=0.99, top=0.82, bottom=0.14, wspace=0.06)
    save(fig, "pilot-measures", theme)


# ---------- Development runs ----------

def dev2_kept(grades):
    smart = ["openai/gpt-5.6-sol", "z-ai/glm-5.3"]
    dumb = sorted({g["reader"] for g in grades} - set(smart))
    quiz = json.loads((RUNS / "dev2-2607.05615v2/quiz-claude-opus-5.5-1-screened.json").read_text())
    pooled = defaultdict(list)
    for g in grades:
        if g["grader"] == GRADER:
            pooled[(g["item"], "smart" if g["reader"] in smart else "dumb" if g["reader"] in dumb else "x",
                    g["cond"])].append(g["score"])
    m = lambda i, who, c: statistics.mean(pooled[(i, who, c)]) if pooled[(i, who, c)] else float("nan")
    return [it["id"] for it in quiz["items"] if m(it["id"], "smart", "none") <= 0.5
            and m(it["id"], "smart", "paper") >= 0.5 and m(it["id"], "dumb", "none") <= 0.5]


def fig_leakage(t, theme):
    runs = {"p1": jsonl(RUNS / "dev-2607.05615v2/grades.jsonl"), "p2": jsonl(RUNS / "dev2-2607.05615v2/grades.jsonl")}
    kept = {"p1": (3, 27), "p2": (len(dev2_kept(runs["p2"])), 24)}
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1, 1.6]})
    style(fig, (a, b), t)
    header(fig, t, "A rewritten prompt plus a screening step stopped most questions leaking",
           "Development runs on one reserve paper. Left: questions that pass the four-condition filter. "
           "Right: smart takers on all questions, without and with the paper.")
    for y, (run, label) in enumerate((("p1", "Prompt p1"), ("p2", "Prompt p2 + screen"))):
        k, n = kept[run]
        a.barh(y, n, color=t["grid"], height=0.55)
        a.barh(y, k, color=t["series"][0], height=0.55)
        a.text(k + 0.4, y, f"{k} of {n} kept", va="center", fontsize=9.5, color=t["text"])
    a.set_yticks([0, 1])
    a.set_yticklabels(["Prompt p1", "Prompt p2\n+ screen"], color=t["text"], fontsize=9.5)
    a.invert_yaxis()
    a.set_xlim(0, 30)
    a.set_xlabel("Questions")
    a.set_title("Questions kept", fontsize=10.5, loc="left")
    rows = [(r, run) for run in ("p1", "p2") for r in ("openai/gpt-5.6-sol", "z-ai/glm-5.3")]
    for y, (reader, run) in enumerate(rows):
        pts = {c: mean_ci(item_means(runs[run], reader, c).values()) for c in ("none", "paper")}
        b.plot([pts["none"][0], pts["paper"][0]], [y, y], color=t["grid"], lw=2, zorder=1)
        for c, color, name in (("none", t["series"][1], "Without the paper"), ("paper", t["series"][0], "With the paper")):
            est, lo, hi = pts[c]
            b.plot([lo, hi], [y, y], color=color, lw=2, zorder=2)
            b.plot(est, y, "o", ms=8, color=color, mec=t["surface"], mew=2, zorder=3, label=name if y == 0 else None)
    b.set_yticks(range(len(rows)))
    b.set_yticklabels([f"{SHORT[r].split(' (')[0]}, {'p1' if run == 'p1' else 'p2 + screen'}" for r, run in rows],
                      color=t["text"], fontsize=9.5)
    b.invert_yaxis()
    b.set_xlim(0, 1.02)
    b.set_xlabel("Mean rubric score, all questions")
    b.set_title("Smart takers could answer p1 questions without reading", fontsize=10.5, loc="left")
    b.grid(axis="x", color=t["grid"], lw=0.8)
    b.set_axisbelow(True)
    legend(b, t)
    fig.subplots_adjust(left=0.1, right=0.99, top=0.76, bottom=0.14, wspace=0.35)
    save(fig, "dev-leakage", theme)


def ladder_panel(ax, t, grades, items, title, first):
    readers = sorted({g["reader"] for g in grades if g["cond"] in ("none", "paper")})
    stats = {}
    for r in readers:
        pts = {c: mean_ci(item_means(grades, r, c, items).values()) for c in ("none", "paper")}
        if not np.isnan(pts["paper"][0]):
            stats[r] = pts
    order = sorted(stats, key=lambda r: stats[r]["paper"][0])
    for y, r in enumerate(order):
        pts = stats[r]
        ax.plot([pts["none"][0], pts["paper"][0]], [y, y], color=t["grid"], lw=2, zorder=1)
        for c, color, name in (("none", t["series"][1], "Without the paper"), ("paper", t["series"][0], "With the paper")):
            est, lo, hi = pts[c]
            if np.isnan(est):
                continue
            ax.plot([lo, hi], [y, y], color=color, lw=2, zorder=2)
            ax.plot(est, y, "o", ms=8, color=color, mec=t["surface"], mew=2, zorder=3,
                    label=name if (first and y == len(order) - 1) else None)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([SHORT.get(r, r) for r in order], color=t["text"], fontsize=9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("Mean rubric score")
    ax.set_title(title, fontsize=10.5, loc="left")
    ax.grid(axis="x", color=t["grid"], lw=0.8)
    ax.set_axisbelow(True)


def fig_ladder(t, theme):
    g1, g2 = jsonl(RUNS / "dev-2607.05615v2/grades.jsonl"), jsonl(RUNS / "dev2-2607.05615v2/grades.jsonl")
    kept = set(dev2_kept(g2))
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 5.2))
    style(fig, (a, b), t)
    header(fig, t, "Readers gain from the paper by very different amounts, and size does not set the order",
           "Development runs on one reserve paper, sorted by score with the paper. Lines: 95% interval over items.")
    ladder_panel(a, t, g1, None, "Run 1: all 27 questions (most answerable without the paper)", True)
    ladder_panel(b, t, g2, kept, f"Run 2: the {len(kept)} kept questions", False)
    legend(a, t)
    fig.subplots_adjust(left=0.12, right=0.99, top=0.78, bottom=0.12, wspace=0.45)
    save(fig, "dev-reader-ladder", theme)


def fig_lookup(t, theme):
    g2 = jsonl(RUNS / "dev2-2607.05615v2/grades.jsonl")
    kept = set(dev2_kept(g2))
    conds = [("none", "No paper"), ("paper", "Full paper"), ("scrambled", "Paragraphs\nshuffled"),
             ("retrieved", "3 retrieved\nparagraphs")]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    style(fig, ax, t)
    header(fig, t, "Most kept questions test finding a passage, not following the argument",
           f"Run 2, the {len(kept)} kept questions. Shuffling costs the smart reader nothing; "
           "3 keyword-retrieved paragraphs are often enough.")
    for k, (reader, dx) in enumerate((("openai/gpt-5.6-sol", -0.1), ("google/gemma-4-31b-it", 0.1))):
        color = t["series"][k]
        for x, (c, _) in enumerate(conds):
            est, lo, hi = mean_ci(item_means(g2, reader, c, kept).values())
            if np.isnan(est):
                continue
            ax.plot([x + dx, x + dx], [lo, hi], color=color, lw=2)
            ax.plot(x + dx, est, "o", ms=8, color=color, mec=t["surface"], mew=2,
                    label=SHORT[reader] if x == 0 else None)
            ax.text(x + dx + 0.06, est, f"{est:.2f}", va="center", fontsize=8.5, color=t["text"])
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([n for _, n in conds], color=t["text"], fontsize=9.5)
    ax.set_ylim(-0.02, 1.05)
    ax.set_ylabel("Mean rubric score")
    ax.grid(axis="y", color=t["grid"], lw=0.8)
    ax.set_axisbelow(True)
    legend(ax, t)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.76, bottom=0.15)
    save(fig, "dev-lookup-baselines", theme)


def fig_graders(t, theme):
    g1 = jsonl(RUNS / "dev-2607.05615v2/grades.jsonl")
    by = defaultdict(dict)
    for g in g1:
        by[(g["item"], g["reader"], g["cond"], g["sample"])][g["grader"]] = round(g["score"], 2)
    pairs = [(v[GRADER], v["qwen/qwen3.8-27b"]) for v in by.values() if GRADER in v and "qwen/qwen3.8-27b" in v]
    levels = sorted({x for p in pairs for x in p})
    idx = {v: i for i, v in enumerate(levels)}
    grid = np.zeros((len(levels), len(levels)))
    for a, b in pairs:
        grid[idx[b], idx[a]] += 1
    same = sum(a == b for a, b in pairs)
    cmap = LinearSegmentedColormap.from_list("seq", [t["surface"], t["series"][0]])
    fig, ax = plt.subplots(figsize=(7, 6))
    style(fig, ax, t)
    header(fig, t, f"Two grader models gave the same score on {same / len(pairs):.0%} of {len(pairs)} answers",
           "Run 1 answers graded by both. Cells count answers; the diagonal is agreement.")
    ax.imshow(np.log1p(grid), cmap=cmap, origin="lower")
    for y in range(len(levels)):
        for x in range(len(levels)):
            if grid[y, x]:
                ax.text(x, y, int(grid[y, x]), ha="center", va="center", fontsize=9,
                        color=t["surface"] if grid[y, x] > 40 else t["text"])
    ticks = [f"{v:.2f}" for v in levels]
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels(ticks, color=t["text"], fontsize=8.5)
    ax.set_yticks(range(len(levels)))
    ax.set_yticklabels(ticks, color=t["text"], fontsize=8.5)
    ax.set_xlabel("GLM-5.3 score (primary grader)")
    ax.set_ylabel("Qwen3.8 27B score (check)")
    ax.spines[:].set_visible(False)
    fig.subplots_adjust(left=0.14, right=0.97, top=0.84, bottom=0.1)
    save(fig, "dev-grader-agreement", theme)


# ---------- Spend ----------

def fig_spend(t, theme):
    stage = {"take": "Readers answering", "grade": "Grading", "write": "Writing and screening quizzes",
             "revise": "Writing and screening quizzes", "screen": "Writing and screening quizzes",
             "obfuscate": "Making versions"}
    groups = [("Development run 1", ["dev-2607.05615v2"]), ("Development run 2", ["dev2-2607.05615v2"]),
              ("Pilot, 6 papers", sorted(p.name for p in RUNS.glob("pilot-*") if p.is_dir()))]
    stages = ["Readers answering", "Grading", "Writing and screening quizzes", "Making versions", "Other"]
    data = np.zeros((len(groups), len(stages)))
    for gi, (_, dirs) in enumerate(groups):
        for d in dirs:
            path = RUNS / d / "calls.jsonl"
            if not path.exists():
                continue
            for c in jsonl(path):
                s = stage.get(str(c.get("tag", "")).split(":")[0], "Other")
                data[gi, stages.index(s)] += float(c.get("cost") or 0)
    keep = [i for i, s in enumerate(stages) if data[:, i].sum() > 0.005]
    stages, data = [stages[i] for i in keep], data[:, keep]
    colors = t["series"] + [t["muted"]]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    style(fig, ax, t)
    header(fig, t, f"Logged model spend: ${data.sum():.0f}, mostly grading",
           "Summed from each run's call log. OpenRouter's total was $138.96, including calls outside these runs.")
    for gi in range(len(groups)):
        left = 0.0
        for si in range(len(stages)):
            v = data[gi, si]
            if v <= 0:
                continue
            ax.barh(gi, v, left=left, color=colors[si], height=0.55, ec=t["surface"], lw=2,
                    label=stages[si] if gi == 0 or (gi > 0 and data[:gi, si].sum() == 0) else None)
            left += v
        ax.text(left + 0.8, gi, f"${left:.0f}", va="center", fontsize=9.5, color=t["text"])
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels([g for g, _ in groups], color=t["text"], fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel("USD")
    ax.set_xlim(0, data.sum(axis=1).max() * 1.12)
    ax.grid(axis="x", color=t["grid"], lw=0.8)
    ax.set_axisbelow(True)
    legend(ax, t)
    fig.subplots_adjust(left=0.17, right=0.99, top=0.72, bottom=0.17)
    save(fig, "spend", theme)


# ---------- Design diagrams ----------

from matplotlib.patches import FancyBboxPatch  # noqa: E402


def box(ax, t, x, y, w, h, title, body, color=None, fill=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12", lw=2,
                                ec=color or t["muted"], fc=fill or t["surface"]))
    ax.text(x + 0.15, y + h - 0.18, title, ha="left", va="top", fontsize=10.5, weight="bold", color=t["text"])
    ax.text(x + 0.15, y + h - 0.55, body, ha="left", va="top", fontsize=8.6, color=t["muted"], linespacing=1.35)


def arrow(ax, t, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=t["muted"], lw=1.6, shrinkA=0, shrinkB=0))


def fig_design_pipeline(t, theme):
    fig, ax = plt.subplots(figsize=(13, 5.4))
    style(fig, ax, t)
    ax.set_xlim(0, 26)
    ax.set_ylim(0, 9.6)
    ax.axis("off")
    header(fig, t, "How the pilot works",
           "Each paper gets one quiz. Readers answer it with no paper and with three versions of the paper, "
           "then a grader scores every answer.")
    blue, orange, aqua = t["series"][0], t["series"][1], t["series"][2]
    box(ax, t, 0.2, 3.0, 3.6, 3.6, "6 papers", "Recent AI safety\npreprints, drawn\nat random.\n2 each: interpret-\nability, attacks,\nalignment")
    box(ax, t, 4.8, 5.4, 4.4, 3.4, "1. Write quiz", "Claude Opus 5.5 lists\n5-8 key claims, then\n24-29 free-response\nquestions, each with\na 2-3 point rubric", blue)
    box(ax, t, 10.2, 5.4, 4.4, 3.4, "2. Screen for leaks", "Kimi K3 answers\nwithout the paper.\nIf it scores > 0.5,\nthe question is\nrewritten (2 rounds)", blue)
    box(ax, t, 4.8, 0.6, 9.8, 3.6, "3. Make 3 versions of the paper", "Clear: the original text\nHard to read: one section rewritten by Kimi K3 to be hard to\n    follow, with every fact kept\nRemoved: that same section deleted", orange)
    box(ax, t, 15.6, 3.0, 4.2, 3.6, "4. Readers answer", "GPT-5.6 Sol (smart),\nGemma 4 31B,\nMinistral 8B.\nNo paper, then each\nversion; open book;\n2 samples", aqua)
    box(ax, t, 20.8, 5.4, 5.0, 3.4, "5. Grade", "GLM-5.3 checks each\nrubric point, blind to\nreader and version.\nScore = share of\npoints met", t["muted"])
    box(ax, t, 20.8, 0.6, 5.0, 3.6, "6. Filter, then compare", "Keep only questions that\nneed the paper (next\nfigure). Compare scores\nacross versions, per\nsection of the paper", t["muted"])
    arrow(ax, t, 3.8, 6.0, 4.8, 7.1)
    arrow(ax, t, 3.8, 3.6, 4.8, 2.4)
    arrow(ax, t, 9.2, 7.1, 10.2, 7.1)
    arrow(ax, t, 14.6, 7.1, 15.6, 5.6)
    arrow(ax, t, 14.6, 2.4, 15.6, 4.0)
    arrow(ax, t, 19.8, 5.6, 20.8, 7.1)
    arrow(ax, t, 23.3, 5.4, 23.3, 4.2)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.84, bottom=0.02)
    save(fig, "design-pipeline", theme)


def fig_design_versions(t, theme):
    fig, ax = plt.subplots(figsize=(12, 5.2))
    style(fig, ax, t)
    ax.set_xlim(0, 24)
    ax.set_ylim(-1.6, 9.2)
    ax.axis("off")
    header(fig, t, "Three versions of each paper separate legibility from missing information",
           "One section is changed. If the quiz measures legibility, the hard-to-read version should score lower "
           "than clear, though not as low as removed.")
    names = ["Abstract", "Introduction", "Method", "Results", "Changed section", "Discussion", "Conclusion"]
    cols = [("Clear", None), ("Hard to read", "hard"), ("Section removed", "removed")]
    blue, orange = t["series"][0], t["series"][1]
    for k, (label, kind) in enumerate(cols):
        x0 = 1 + k * 8
        ax.text(x0, 8.5, label, fontsize=11.5, weight="bold", color=t["text"])
        for i, n in enumerate(names):
            y = 7.4 - i * 1.0
            changed = n == "Changed section"
            if changed and kind == "removed":
                ax.add_patch(FancyBboxPatch((x0, y), 6, 0.75, boxstyle="round,pad=0,rounding_size=0.08", lw=1.5,
                                            ls=(0, (4, 3)), ec=t["muted"], fc=t["surface"]))
                ax.text(x0 + 3, y + 0.37, "[section omitted]", ha="center", va="center", fontsize=8.5, color=t["muted"])
                continue
            fc = orange if (changed and kind == "hard") else t["grid"]
            ax.add_patch(FancyBboxPatch((x0, y), 6, 0.75, boxstyle="round,pad=0,rounding_size=0.08", lw=0, fc=fc))
            hard = changed and kind == "hard"
            ax.text(x0 + 0.2, y + 0.37, "Rewritten: same facts, hard to follow" if hard else n, va="center",
                    fontsize=9, color="#ffffff" if hard else t["text"], weight="bold" if changed else None)
    expect = ["Questions on the changed\nsection: answerable",
              "Expected if legibility matters:\nlower than clear",
              "Positive control: information is\ngone, so scores should drop"]
    seen = ["Observed (valid paper, GPT-5.6): 0.94", "Observed: 0.96, no drop", "Observed: 0.36, a large drop"]
    for k in range(3):
        x0 = 1 + k * 8
        ax.text(x0, 0.35, expect[k], fontsize=9, color=t["text"], va="top")
        ax.text(x0, -1.05, seen[k], fontsize=9, color=blue if k == 2 else t["muted"], va="top",
                weight="bold" if k else None)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.84, bottom=0.02)
    save(fig, "design-versions", theme)


def fig_design_filter(t, theme):
    fig, ax = plt.subplots(figsize=(11, 5.2))
    style(fig, ax, t)
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 10)
    ax.axis("off")
    header(fig, t, "Four conditions decide which questions are valid",
           "Run on the clear version only. Thresholds are on a 0 to 1 rubric score, averaged over samples.")
    blue = t["series"][0]
    ax.text(8.2, 8.7, "Without the paper", ha="center", fontsize=11, weight="bold", color=t["text"])
    ax.text(16.8, 8.7, "With the paper", ha="center", fontsize=11, weight="bold", color=t["text"])
    ax.text(3.6, 6.0, "Smart reader\n(GPT-5.6 Sol)", ha="right", va="center", fontsize=11, weight="bold", color=t["text"])
    ax.text(3.6, 2.2, "Weaker readers\n(Gemma 4 31B,\nMinistral 8B)", ha="right", va="center", fontsize=11,
            weight="bold", color=t["text"])
    box(ax, t, 4.0, 4.4, 8.4, 3.6, "Should be low", "Drop the question if above 0.5.\nOtherwise background knowledge or\nthe question's own wording answers it,\nso it does not measure the paper.")
    box(ax, t, 12.6, 4.4, 8.4, 3.6, "Should be high", "Drop the question if below 0.5.\nOtherwise the question or its answer\nkey is wrong, since even a strong\nreader with the paper fails it.")
    box(ax, t, 4.0, 0.4, 8.4, 3.6, "Should be low", "Drop the question if their average\nis above 0.5, for the same reason as\nabove: it must not be answerable by\nthe audited reader without reading.")
    box(ax, t, 12.6, 0.4, 8.4, 3.6, "The audit result", "Never used to drop questions:\ndropping the ones weak readers fail\nwould inflate their scores.\nThe comparison happens here.", blue)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.84, bottom=0.02)
    save(fig, "design-filter", theme)


# ---------- Plan v2 ----------

V2_READERS = ["openai/gpt-5.6-sol", "google/gemma-4-31b-it", "mistralai/ministral-8b-2512",
              "meta-llama/llama-3.2-3b-instruct"]
V2_CONDS = [("none", "No paper"), ("orig", "Original"), ("obf", "Obfuscated"), ("comp", "Compressed"),
            ("compobf", "Compressed\n+ obfuscated")]


def v2_summaries():
    """Every plan v2 run with takes, dev paper first. Source: data/runs/*/takes-summary.json (takes_v2.py)."""
    runs = sorted(RUNS.glob("v2*/takes-summary.json"), key=lambda p: (not p.parent.name.startswith("v2dev"), p))
    return [(p.parent.name, json.loads(p.read_text())) for p in runs]


def v2_pooled():
    """Pooled summary over core papers (not the dev paper): per reader and condition, the mean over all kept
    questions of every core paper (each question's mean over samples), a 95% bootstrap interval over questions, and
    the change from the original, paired by question. Same layout as a takes-summary.json."""
    per = defaultdict(dict)
    runs = v2_question_means("v2-")
    n_q, n_p = sum(len(typ) for typ, _ in runs.values()), len(runs)
    for name, (_, means) in runs.items():
        for (rd, c, i), m in means.items():
            per[(rd, c)][name + ":" + i] = m
    if not n_p:
        return None
    readers = {}
    for (rd, c), qs in per.items():
        m, lo, hi = mean_ci(qs.values())
        res = {"mean": m, "ci95": [lo, hi]}
        orig = per.get((rd, "orig"), {})
        if c != "orig" and orig:
            common = [q for q in qs if q in orig]
            d, dlo, dhi = mean_ci(qs[q] - orig[q] for q in common)
            res.update(minus_orig=d, minus_orig_ci95=[dlo, dhi])
        readers.setdefault(rd, {})[c] = res
    return {"paper": f"All {n_p} core papers pooled", "kept": n_q, "readers": readers}


def fig_v2_conditions(t, theme):
    runs = v2_summaries()
    if not runs:
        return
    pooled = v2_pooled()
    if pooled:
        runs = [("pooled", pooled)] + runs
    n = len(runs)
    cols = min(n, 3)
    rows = -(-n // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(max(4.6 * cols + 0.6, 10), 3.9 * rows + 1.3), squeeze=False)
    style(fig, axes, t)
    H = 3.9 * rows + 1.3  # header placed in inches, so it does not drift into the panels on tall figures
    fig.text(0.01, 1 - 0.2 / H, "Scores by reader across the five conditions, one panel per paper", ha="left",
             va="top", fontsize=13, weight="bold", color=t["text"])
    fig.text(0.01, 1 - 0.55 / H, "Mean rubric score on kept questions, 3 samples, fixed 0 to 1 axis. Bars: 95% "
             "interval over questions. First panel pools the core papers; numbers there are the change from the "
             "original to obfuscated.", ha="left", va="top", fontsize=9.5, color=t["muted"])
    for k, (name, s) in enumerate(runs):
        ax = axes[k // cols][k % cols]
        for j, r in enumerate(V2_READERS):
            res = s["readers"].get(r, {})
            xs = [i + (j - 1.5) * 0.08 for i, (c, _) in enumerate(V2_CONDS) if c in res]
            ys = [res[c]["mean"] for c, _ in V2_CONDS if c in res]
            lo = [res[c]["mean"] - res[c]["ci95"][0] for c, _ in V2_CONDS if c in res]
            hi = [res[c]["ci95"][1] - res[c]["mean"] for c, _ in V2_CONDS if c in res]
            col = t["series"][j]
            ax.errorbar(xs, ys, yerr=[lo, hi], color=col, lw=2, elinewidth=1.2, capsize=0, marker="o", ms=7,
                        mec=t["surface"], mew=1.5, label=SHORT.get(r, r) if k == 0 else None)
        ax.set_xticks(range(len(V2_CONDS)))
        ax.set_xticklabels([l for _, l in V2_CONDS], fontsize=8.5, color=t["text"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Mean rubric score")
        tag = " (development)" if name.startswith("v2dev") else ""
        ax.set_title(f"{s['paper']}{tag}: {s['kept']} questions", fontsize=10.5, loc="left",
                     weight="bold" if name == "pooled" else "normal")
        if name == "pooled":  # drops from the original, as annotations beside the obfuscated points
            for j, r in enumerate(V2_READERS):
                o = s["readers"].get(r, {}).get("obf")
                if o and "minus_orig" in o:
                    ax.annotate(f"{o['minus_orig']:+.2f}", (2 + (j - 1.5) * 0.08, o["mean"]), xytext=(14, 0),
                                textcoords="offset points", va="center", fontsize=8, color=t["series"][j],
                                bbox=dict(fc=t["surface"], ec="none", pad=0.5), zorder=5)
        ax.grid(axis="y", color=t["grid"], lw=0.8)
        ax.set_axisbelow(True)
    for k in range(n, rows * cols):
        axes[k // cols][k % cols].set_visible(False)
    legend(axes[0][0], t, y=1 - 0.85 / H)
    fig.subplots_adjust(left=0.08, right=0.98, top=1 - 1.6 / H,
                        bottom=0.13 / rows, wspace=0.3, hspace=0.45)
    save(fig, "v2-conditions", theme)


def v2_kept_files():
    """Every plan v2 run with a kept-question file, dev paper first. Source: kept-v2.json (filter_v2.py combine)."""
    runs = sorted(RUNS.glob("v2*/kept-v2.json"), key=lambda p: (not p.parent.name.startswith("v2dev"), p))
    return [(p.parent.name, json.loads(p.read_text())) for p in runs]


def fig_v2_funnel(t, theme):
    runs = v2_kept_files()
    if not runs:
        return
    n = len(runs)
    cols = min(n, 3)
    rows = -(-n // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(max(4.4 * cols + 0.8, 9), 2.9 * rows + 1.3), squeeze=False)
    style(fig, axes, t)
    header(fig, t, "Question funnel: questions left after each filter, one panel per paper",
           "Filters applied in order on the original paper. Red: removed at that step. Dashed line: 40-question target.")
    for k, (name, kf) in enumerate(runs):
        ax = axes[k // cols][k % cols]
        stages = kf["funnel"]
        labels = [st["stage"].capitalize().replace("Counterfactual pair", "Pair check") for st in stages]
        left = [st["left"] for st in stages]
        removed = [st.get("removed", 0) for st in stages]
        y = np.arange(len(stages))[::-1]
        ax.barh(y, left, color=t["series"][0], height=0.62)
        ax.barh(y, removed, left=left, color=t["red"], height=0.62, alpha=0.75)
        for yi, l, r in zip(y, left, removed):
            ax.text(l + r + 2, yi, f"{l}" + (f"  (-{r})" if r else ""), va="center", fontsize=8.5, color=t["text"])
        ax.axvline(40, color=t["muted"], ls="--", lw=1)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8.5, color=t["text"])
        ax.set_xlim(0, max(left[0] * 1.3, 60))
        ax.set_xlabel("Questions")
        tag = " (development)" if name.startswith("v2dev") else ""
        ax.set_title(f"{kf['paper']}{tag}", fontsize=10.5, loc="left")
    for k in range(n, rows * cols):
        axes[k // cols][k % cols].set_visible(False)
    fig.subplots_adjust(left=0.13, right=0.97, top=1 - 1.25 / (2.9 * rows + 1.3), bottom=0.6 / (2.9 * rows + 1.3),
                        wspace=0.55, hspace=0.6)
    save(fig, "v2-funnel", theme)


def fig_v2_types(t, theme):
    runs = v2_kept_files()
    if not runs:
        return
    written, kept = defaultdict(int), defaultdict(int)
    for _, kf in runs:
        for ty, c in kf["by_type"].items():
            written[ty] += c["written"]
            kept[ty] += c["kept"]
    types = sorted(written, key=lambda ty: -kept[ty] / written[ty])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    style(fig, axes, t)
    header(fig, t, f"Kept questions by type, pooled over {len(runs)} paper(s)",
           "Left: candidates written and kept. Right: share of each type that survived all four filters.")
    x = np.arange(len(types))
    axes[0].bar(x - 0.2, [written[ty] for ty in types], 0.4, color=t["grid"], label="Written")
    axes[0].bar(x + 0.2, [kept[ty] for ty in types], 0.4, color=t["series"][0], label="Kept")
    rate = [kept[ty] / written[ty] for ty in types]
    axes[1].bar(x, rate, 0.55, color=t["series"][0])
    for xi, r, ty in zip(x, rate, types):
        axes[1].text(xi, r + 0.015, f"{r:.0%}", ha="center", fontsize=8.5, color=t["text"])
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([ty.replace("counterfactual", "counter-\nfactual") for ty in types], fontsize=9,
                           color=t["text"])
        ax.grid(axis="y", color=t["grid"], lw=0.8)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Questions")
    axes[1].set_ylabel("Survival rate")
    axes[1].set_ylim(0, 1)
    legend(axes[0], t)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.78, bottom=0.1, wspace=0.25)
    save(fig, "v2-kept-by-type", theme)


def v2_question_means(prefix):
    """{run: (type by question, {(reader, condition, question): mean over samples})} for runs whose directory
    name starts with prefix, from takes_v2.question_scores, so the counterfactual pair rule matches the summary."""
    import sys
    sys.path.insert(0, str(ROOT / "scripts/quiz"))
    import takes_v2
    out = {}
    for p in sorted(RUNS.glob(f"{prefix}*/takes-v2.jsonl")):
        _, items, _, per_q, _ = takes_v2.question_scores(p.parent)
        typ = {it["id"]: it["type"] for it in items}
        out[p.parent.name] = (typ, {k: statistics.mean(v) for k, v in per_q.items() if k[2] in typ})
    return out


def v2_item_scores():
    """{(reader, condition, type): [per-question mean over samples]} on kept questions, core papers pooled
    (the development paper only until a core paper has takes)."""
    out = defaultdict(list)
    for typ, means in (v2_question_means("v2-") or v2_question_means("v2dev")).values():
        for (rd, c, i), m in means.items():
            out[(rd, c, typ[i])].append(m)
    return out


def fig_v2_by_type(t, theme):
    sc = v2_item_scores()
    if not sc:
        return
    types = sorted({k[2] for k in sc})
    n_papers = len(list(RUNS.glob("v2-*/takes-v2.jsonl"))) or 1
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.6), squeeze=False)
    style(fig, axes, t)
    header(fig, t, f"Score by question type, per reader and condition ({n_papers} core paper(s) pooled; exploratory)",
           "Mean rubric score on kept questions, 3 samples. Bars: 95% interval over questions; few questions per type.")
    conds = [("none", "No paper"), ("orig", "Original"), ("obf", "Obfuscated"), ("comp", "Compressed"),
             ("compobf", "Compressed + obfuscated")]
    colors = [t["muted"]] + t["series"]
    for j, rd in enumerate(V2_READERS):
        ax = axes[j // 2][j % 2]
        for ci, (c, lab) in enumerate(conds):
            for ti, ty in enumerate(types):
                m, lo, hi = mean_ci(sc.get((rd, c, ty), []))
                if np.isnan(m):
                    continue
                xpos = ti + (ci - 2) * 0.14
                ax.errorbar([xpos], [m], yerr=[[m - lo], [hi - m]], color=colors[ci], marker="o", ms=5, lw=0,
                            elinewidth=1, capsize=0, label=lab if (j == 0 and ti == 0) else None)
        nq = [len(sc.get((rd, "orig", ty), [])) for ty in types]
        ax.set_xticks(range(len(types)))
        ax.set_xticklabels([f"{ty.replace('counterfactual', 'counter-factual')}\n(n={q})" for ty, q in zip(types, nq)], fontsize=8.5, color=t["text"])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Mean rubric score")
        ax.set_title(SHORT.get(rd, rd), fontsize=10.5, loc="left")
        ax.grid(axis="y", color=t["grid"], lw=0.8)
        ax.set_axisbelow(True)
    legend(axes[0][0], t, y=0.895)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.83, bottom=0.08, wspace=0.18, hspace=0.42)
    save(fig, "v2-score-by-type", theme)


def fig_v2_versions(t, theme):
    """Plan figures 5 to 7: position between floor and original, information check, version lengths."""
    runs = v2_summaries()
    if not runs:
        return
    vers = [("obf", "Obfuscated"), ("comp", "Compressed"), ("compobf", "Compressed\n+ obfuscated")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), gridspec_kw=dict(width_ratios=[1, 1, 1.5]))
    style(fig, axes, t)
    header(fig, t, f"Versions: length, information kept, and where each reader lands ({len(runs)} paper(s), one dot each)",
           "Left: characters relative to the original. Middle: smart reader's score relative to the original (pass at 0.9). "
           "Right: 0 = no-paper score, 1 = original score.")
    rng = np.random.default_rng(SEED)
    for k, (name, s) in enumerate(runs):
        meta = json.loads((RUNS / name / "versions/meta.json").read_text())
        mk = "D" if name.startswith("v2dev") else "o"
        for i, (v, _) in enumerate(vers):
            dx = rng.uniform(-0.12, 0.12)
            axes[0].scatter(i + dx, meta["chars"][v] / meta["chars"]["orig"], color=t["series"][0], marker=mk, s=36,
                            zorder=3)
            ic = s["information_check"].get(v)
            if ic:
                axes[1].scatter(i + dx, ic["ratio"], color=t["series"][0] if ic["pass"] else t["red"], marker=mk,
                                s=36, zorder=3)
        for j, r in enumerate(V2_READERS[1:]):
            res = s["readers"].get(r, {})
            for i, (v, _) in enumerate(vers):
                if v in res and "position" in res[v]:
                    axes[2].scatter(i + (j - 1) * 0.22, res[v]["position"], color=t["series"][j + 1], marker=mk, s=36,
                                    zorder=3, label=SHORT.get(r, r) if (k == 0 and i == 0) else None)
    axes[0].axhspan(0.35, 0.5, color=t["mid"], zorder=0)
    axes[0].set_ylim(0, 1.15)
    axes[0].set_ylabel("Length / original")
    axes[1].axhline(0.9, color=t["muted"], ls="--", lw=1)
    axes[1].set_ylim(0.6, 1.15)
    axes[1].set_ylabel("Smart reader score / original")
    axes[2].axhline(0, color=t["muted"], lw=0.8)
    axes[2].axhline(1, color=t["muted"], lw=0.8, ls="--")
    axes[2].set_ylabel("Position (0 no paper, 1 original)")
    for ax in axes:
        ax.set_xticks(range(len(vers)))
        ax.set_xticklabels([l for _, l in vers], fontsize=8.5, color=t["text"])
        ax.set_xlim(-0.5, len(vers) - 0.5)
        ax.grid(axis="y", color=t["grid"], lw=0.8)
        ax.set_axisbelow(True)
    axes[0].set_title("Length (shaded: compression target)", fontsize=10.5, loc="left")
    axes[1].set_title("Information check", fontsize=10.5, loc="left")
    axes[2].set_title("Weaker readers' position", fontsize=10.5, loc="left")
    legend(axes[2], t, y=0.85)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.72, bottom=0.12, wspace=0.28)
    save(fig, "v2-versions", theme)


def fig_v2_vs_smart(t, theme):
    """Each weaker reader's change from the original minus the smart reader's change, same questions
    (scripts/quiz/analyze_v2.py). Below 0: the version hurts the weaker reader more than the smart one."""
    path = ROOT / "data/analysis/v2-pooled.json"
    if not path.exists():
        return
    a = json.loads(path.read_text())
    vs = a["all_versions"]["vs_smart"]
    vers = [("obf", "Obfuscated"), ("comp", "Compressed"), ("compobf", "Compressed + obfuscated")]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    style(fig, ax, t)
    header(fig, t, "How much more each version hurts a weaker reader than the smart reader",
           f"(weaker reader's change from original) minus (smart reader's), same questions, {len(a['papers'])} core "
           "papers pooled.\nBar: 95% interval, questions resampled within papers. Small dots: single papers.")
    for j, r in enumerate(V2_READERS):
        if r not in vs:
            continue
        for i, (v, _) in enumerate(vers):
            d = vs[r].get(v)
            if not d:
                continue
            y = i + (j - 2) * 0.22
            ax.scatter(list(d["per_paper"].values()), [y] * len(d["per_paper"]), s=12, color=t["series"][j],
                       alpha=0.45, lw=0, zorder=2)
            ax.errorbar([d["mean"]], [y], xerr=[[d["mean"] - d["ci95"][0]], [d["ci95"][1] - d["mean"]]],
                        color=t["series"][j], elinewidth=2, capsize=0, marker="o", ms=8, mec=t["surface"], mew=1.5,
                        zorder=3, label=SHORT.get(r, r) if i == 0 else None)
            ax.annotate(f"{d['mean']:+.3f}", (d["ci95"][1], y), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=8, color=t["muted"])
    ax.axvline(0, color=t["muted"], lw=1)
    ax.set_yticks(range(len(vers)))
    ax.set_yticklabels([l for _, l in vers], fontsize=9.5, color=t["text"])
    ax.invert_yaxis()
    ax.set_xlabel("Extra drop for the weaker reader (rubric score; negative = hurts the weaker reader more)")
    ax.grid(axis="x", color=t["grid"], lw=0.8)
    ax.set_axisbelow(True)
    legend(ax, t, y=0.85)
    fig.subplots_adjust(left=0.2, right=0.97, top=0.75, bottom=0.14)
    save(fig, "v2-vs-smart", theme)


def fig_v2_bars(t, theme):
    """Preregistered bar 1 per paper (docs/preregistration-v2.md): each weaker reader's drop (orig - version) with
    its 95% interval, against the 0.10 line. Filled marker: bars 1 and 2 both met. Source: v2-pooled.json "bars"."""
    path = ROOT / "data/analysis/v2-pooled.json"
    if not path.exists():
        return
    b = json.loads(path.read_text()).get("bars")
    if not b:
        return
    papers = sorted(b["per_paper"], key=lambda p: (not b["gate"][p]["passes"], -b["gate"][p]["kept"]))
    weaker = [r for r in V2_READERS[1:] if r in b["per_paper"][papers[0]]]
    vers = [("obf", "Obfuscated"), ("compobf", "Compressed + obfuscated")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 1.9 + 0.42 * len(papers)), sharey=True, sharex=True)
    style(fig, axes, t)
    header(fig, t, "Preregistered per-paper bar: does a weaker reader drop at least 0.10?",
           "Drop from original (positive = lower score on the version), 95% interval over questions. Dashed line: "
           "the 0.10 bar.\nFilled: bars 1 and 2 met (drop >= 0.10, interval above 0, smart reader drops less, "
           "version passes the information check; paper passes the gate).")
    for a, (v, label) in zip(axes, vers):
        for i, p in enumerate(papers):
            for j, r in enumerate(weaker):
                x = b["per_paper"][p][r][v]
                y, col = i + (j - 1) * 0.22, t["series"][V2_READERS.index(r)]
                met = x["bar1"] and x["bar2"] and b["gate"][p]["passes"]
                a.errorbar([x["drop"]], [y], xerr=[[x["drop"] - x["ci95"][0]], [x["ci95"][1] - x["drop"]]],
                           color=col, elinewidth=1.6, capsize=0, marker="o", ms=6.5, mew=1.5,
                           mfc=col if met else t["surface"], mec=col, zorder=3,
                           label=SHORT.get(r, r) if i == 0 and v == "obf" else None)
        a.axvline(0, color=t["muted"], lw=1)
        a.axvline(0.10, color=t["red"], lw=1.2, ls="--")
        a.set_title(label, fontsize=10.5, loc="left", color=t["text"])
        a.set_xlabel("Drop from original (rubric score)")
        a.grid(axis="x", color=t["grid"], lw=0.8)
        a.set_axisbelow(True)
    axes[0].set_yticks(range(len(papers)))
    axes[0].set_yticklabels([f"{p} ({b['gate'][p]['kept']} q)" + ("" if b["gate"][p]["passes"] else ", below gate")
                             for p in papers], fontsize=8.5, color=t["text"])
    axes[0].invert_yaxis()
    h = 1.9 + 0.42 * len(papers)
    legend(axes[0], t, y=1 - 0.9 / h)
    fig.subplots_adjust(left=0.2, right=0.98, top=1 - 1.6 / h, bottom=0.55 / h, wspace=0.08)
    save(fig, "v2-bars", theme)


def fig_v2_runs_grid(t, theme):
    """Figure 13 (Jason, INBOX 2026-11-04): every run as a point. Rows: core papers (most kept questions first);
    columns: readers. A point is one sample's mean score over the paper's kept questions in one condition (3 per
    condition per cell), from takes_v2.question_scores (counterfactual pair rule included), lightly jittered."""
    import sys
    sys.path.insert(0, str(ROOT / "scripts/quiz"))
    import takes_v2
    runs = []
    for p in sorted(RUNS.glob("v2-*/takes-summary.json")):
        paper, items, _, per_q, _ = takes_v2.question_scores(p.parent)
        ids = [it["id"] for it in items]
        pts = {}
        for r in V2_READERS:
            for c, _ in V2_CONDS:
                per = [per_q.get((r, c, i), []) for i in ids]
                n = min((len(v) for v in per), default=0)  # samples present for every question
                pts[(r, c)] = [sum(v[k] for v in per) / len(per) for k in range(n)]
        runs.append((paper, len(ids), pts))
    if not runs:
        return
    runs.sort(key=lambda x: -x[1])
    colors = [t["muted"]] + t["series"]  # no paper in grey, then the four versions
    markers = ["o", "s", "D", "^", "v"]
    rows, cols = len(runs), len(V2_READERS)
    H = 0.62 * rows + 1.9
    fig, axes = plt.subplots(rows, cols, figsize=(12, H), sharex=True, squeeze=False)
    style(fig, axes, t)
    fig.text(0.01, 1 - 0.2 / H, "Every run, paper by reader: one point per sample and condition", ha="left",
             va="top", fontsize=13, weight="bold", color=t["text"])
    fig.text(0.01, 1 - 0.55 / H, "A point is one sample's mean rubric score over the paper's kept questions "
             "(3 samples per condition). Fixed 0 to 1 axis; vertical position is jitter only.", ha="left",
             va="top", fontsize=9.5, color=t["muted"])
    rng = np.random.default_rng(SEED)
    for i, (paper, kept, pts) in enumerate(runs):
        for j, r in enumerate(V2_READERS):
            ax = axes[i][j]
            for k, (c, label) in enumerate(V2_CONDS):
                xs = pts[(r, c)]
                ax.scatter(xs, rng.uniform(-0.35, 0.35, len(xs)), s=22, marker=markers[k], color=colors[k],
                           edgecolors=t["surface"], linewidths=0.5, alpha=0.9, zorder=3,
                           label=label if i == 0 and j == 0 else None)
            ax.set_xlim(0, 1)
            ax.set_xticks([0.2, 0.4, 0.6, 0.8])  # no labels at 0 and 1, which collide between columns
            ax.set_ylim(-0.6, 0.6)
            ax.set_yticks([])
            ax.grid(axis="x", color=t["grid"], lw=0.7)
            ax.set_axisbelow(True)
            ax.spines["left"].set_visible(False)
            if i == 0:
                ax.set_title(SHORT.get(r, r), fontsize=10, color=t["text"])
            if j == 0:
                ax.set_ylabel(f"{paper}\n{kept} q", rotation=0, ha="right", va="center", fontsize=8.5,
                              color=t["text"])
            if i < rows - 1:
                ax.tick_params(labelbottom=False)
    for ax in axes[-1]:
        ax.set_xlabel("Mean rubric score")
    legend(axes[0][0], t, y=1 - 0.85 / H)
    fig.subplots_adjust(left=0.11, right=0.985, top=1 - 1.55 / H, bottom=0.6 / H, wspace=0.08, hspace=0.12)
    save(fig, "v2-runs-grid", theme)


JUDGE_VERS = [("orig", "Original"), ("obf", "Obfuscated"), ("comp", "Compressed"), ("compobf", "Compressed\n+ obfuscated")]


def fig_v2_judge(t, theme, src="v2-judge", name="v2-judge",
                 title="Quiz vs. asking a model: quiz score (top) and the reader's own legibility rating (bottom)"):
    """Figure 14 (Jason, required): quiz score and direct legibility rating side by side, per reader and version.
    Small dots: one paper (quiz: paper mean over kept questions; rating: mean of 3 samples). Large marks: mean over
    the 12 papers with a 95% bootstrap interval over papers. Source: data/analysis/v2-judge.json (judge_v2.py)."""
    path = ROOT / f"data/analysis/{src}.json"
    if not path.exists():
        return
    res = json.loads(path.read_text())
    papers = res["papers"]
    fig, axes = plt.subplots(2, len(V2_READERS), figsize=(12, 6.6), sharey="row")
    style(fig, axes, t)
    header(fig, t, title,
           f"Small dots: one paper each ({len(papers)} core papers). Bars: mean over papers with 95% interval "
           "(bootstrap over papers). Exploratory baseline.")
    rng = np.random.default_rng(SEED)
    for j, r in enumerate(V2_READERS):
        for row, (key, lo, hi, ylabel) in enumerate([("quiz", 0, 1, "Quiz score (0 to 1)"),
                                                      ("rating", 1, 10, "Rating (1 to 10)")]):
            ax = axes[row][j]
            for k, (c, label) in enumerate(JUDGE_VERS):
                vals = [res["per_paper"][p][r][key].get(c) for p in papers]
                vals = [v for v in vals if v is not None]
                m, l, h = mean_ci(vals)
                ax.bar(k, m, width=0.62, color=t["series"][k], alpha=0.35, zorder=2,
                       label=label.replace("\n", " ") if row == 0 and j == 0 else None)
                ax.errorbar(k, m, yerr=[[m - l], [h - m]], color=t["text"], lw=1.2, capsize=3, zorder=4)
                ax.scatter(k + rng.uniform(-0.18, 0.18, len(vals)), vals, s=11, color=t["series"][k],
                           edgecolors=t["surface"], linewidths=0.4, zorder=3)
            ax.set_ylim(lo - (0 if lo == 0 else 0.3), hi + (0.02 if hi == 1 else 0.3))
            ax.set_xticks([])
            ax.grid(axis="y", color=t["grid"], lw=0.7)
            ax.set_axisbelow(True)
            if row == 0:
                ax.set_title(SHORT.get(r, r), fontsize=10, color=t["text"])
            if j == 0:
                ax.set_ylabel(ylabel)
    legend(axes[0][0], t, y=0.885)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.80, bottom=0.04, wspace=0.08, hspace=0.12)
    save(fig, name, theme)


def fig_v2_judge_drops(t, theme):
    """Figure 14b: per paper, the quiz's drop (original minus version) against the rating's drop, per reader.
    Agreement between methods would put points along a rising line. Source: data/analysis/v2-judge.json."""
    path = ROOT / "data/analysis/v2-judge.json"
    if not path.exists():
        return
    res = json.loads(path.read_text())
    fig, axes = plt.subplots(1, len(V2_READERS), figsize=(12, 4.2), sharey=True, sharex=True)
    style(fig, axes, t)
    header(fig, t, "Per paper: does the rating fall where the quiz score falls?",
           "One point per paper and hard version. x: quiz score drop from the original; y: rating drop "
           "(1 to 10 scale). Above 0 means the version is judged or scored worse.")
    for j, r in enumerate(V2_READERS):
        ax = axes[j]
        ax.axhline(0, color=t["grid"], lw=1, zorder=1)
        ax.axvline(0, color=t["grid"], lw=1, zorder=1)
        for k, (c, label) in enumerate(JUDGE_VERS[1:], 1):
            xs, ys = [], []
            for p in res["papers"]:
                pp = res["per_paper"][p][r]
                if c in pp["rating"] and "orig" in pp["rating"]:
                    xs.append(pp["quiz"]["orig"] - pp["quiz"][c])
                    ys.append(pp["rating"]["orig"] - pp["rating"][c])
            ax.scatter(xs, ys, s=22, color=t["series"][k], edgecolors=t["surface"], linewidths=0.5, zorder=3,
                       label=label.replace("\n", " ") if j == 0 else None)
        ax.set_title(SHORT.get(r, r), fontsize=10, color=t["text"])
        ax.set_xlabel("Quiz score drop")
        ax.grid(color=t["grid"], lw=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Rating drop")
    legend(axes[0], t, y=0.84)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.70, bottom=0.14, wspace=0.08)
    save(fig, "v2-judge-drops", theme)


def fig_v2_judge_human(t, theme):
    """Figure 14c: the same comparison with the rating prompt asking about a named human reader (Jason, 2026-11-04)."""
    fig_v2_judge(t, theme, src="v2-judge-human", name="v2-judge-human",
                 title="Quiz vs. asking a model about a human reader: quiz score (top) and rating (bottom)")


GAP_VERS = JUDGE_VERS + [("rem", "Fluent,\ninsight removed"), ("imp", "Fluent,\ninsight implicit")]
GAP_EXTRA = {"light": ["#8a5cd6", "#7a7a74"], "dark": ["#a07ae6", "#9a9a92"]}


def fig_v2_gap(t, theme):
    """Hidden-gap follow-up: quiz score (top) and the human-reader rating (bottom) for all six versions, per reader.
    Small dots: one paper; bars: mean over papers with a 95% bootstrap interval over papers. Sources:
    data/analysis/v2-judge-human.json (earlier versions) and v2-gap.json (gap_v2.py: rem, imp)."""
    gp, jp = ROOT / "data/analysis/v2-gap.json", ROOT / "data/analysis/v2-judge-human.json"
    if not gp.exists() or not jp.exists():
        return
    gap, jh = json.loads(gp.read_text()), json.loads(jp.read_text())
    papers = list(gap["papers"])
    colors = t["series"] + GAP_EXTRA[theme]

    def value(p, r, c, key):
        if c in ("rem", "imp"):
            if key == "quiz":
                return gap["papers"][p]["readers"][r].get(c)
            return gap["papers"][p].get("judge", {}).get("human", {}).get(f"{r}|{c}")
        return jh["per_paper"][p][r][key].get(c)

    fig, axes = plt.subplots(2, len(V2_READERS), figsize=(13, 6.8), sharey="row")
    style(fig, axes, t)
    header(fig, t, "Fluent versions with a hidden gap, next to the earlier rewrites: quiz score (top) and rating (bottom)",
           f"Rating: the reader's 1 to 10 legibility rating for a PhD reader outside the subfield. Small dots: one paper "
           f"each ({len(papers)} papers). Bars: mean over papers, 95% interval over papers.")
    rng = np.random.default_rng(SEED)
    for j, r in enumerate(V2_READERS):
        for row, (key, lo, hi, ylabel) in enumerate([("quiz", 0, 1, "Quiz score (0 to 1)"),
                                                      ("rating", 1, 10, "Rating (1 to 10)")]):
            ax = axes[row][j]
            for k, (c, label) in enumerate(GAP_VERS):
                vals = [v for p in papers if (v := value(p, r, c, key)) is not None]
                if not vals:
                    continue
                m, l, h = mean_ci(vals)
                ax.bar(k, m, width=0.62, color=colors[k], alpha=0.35, zorder=2,
                       label=label.replace("\n", " ") if row == 0 and j == 0 else None)
                ax.errorbar(k, m, yerr=[[m - l], [h - m]], color=t["text"], lw=1.2, capsize=3, zorder=4)
                ax.scatter(k + rng.uniform(-0.18, 0.18, len(vals)), vals, s=9, color=colors[k],
                           edgecolors=t["surface"], linewidths=0.4, zorder=3)
            ax.set_ylim(lo - (0 if lo == 0 else 0.3), hi + (0.02 if hi == 1 else 0.3))
            ax.set_xticks([])
            ax.grid(axis="y", color=t["grid"], lw=0.7)
            ax.set_axisbelow(True)
            if row == 0:
                ax.set_title(SHORT.get(r, r), fontsize=10, color=t["text"])
            if j == 0:
                ax.set_ylabel(ylabel)
    legend(axes[0][0], t, y=0.885, ncol=6)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.80, bottom=0.04, wspace=0.08, hspace=0.12)
    save(fig, "v2-gap", theme)


def main():
    for theme, t in THEMES.items():
        plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": t["text"]})
        for f in (fig_design_pipeline, fig_design_versions, fig_design_filter, fig_obfuscation, fig_measures, fig_leakage, fig_ladder, fig_lookup, fig_graders, fig_spend, fig_v2_conditions,
                  fig_v2_funnel, fig_v2_types, fig_v2_by_type, fig_v2_versions, fig_v2_vs_smart, fig_v2_bars, fig_v2_runs_grid,
                  fig_v2_judge, fig_v2_judge_drops, fig_v2_judge_human, fig_v2_gap):
            f(t, theme)
    print("wrote", len(list(OUT.glob("*.png"))), "files to", OUT)


if __name__ == "__main__":
    main()
