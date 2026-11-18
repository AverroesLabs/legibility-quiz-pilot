"""Calibration grid for the error-catching audit: predicted pattern, drawn before any v4 data.

Run: uv run --with matplotlib python scripts/calibration_table.py
Writes docs/figures/calibration-grid.png and the generated key in docs/calibration-table.md.

The grid's values are PREDICTIONS from the toy model in `predict`, not data. When real results arrive,
replace `predict` with measured rates and keep the layout.
"""
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from averroes_figstyle import *  # noqa: E402,F403

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "docs", "figures", "calibration-grid.png")
DOC = os.path.join(ROOT, "docs", "calibration-table.md")

# Bar meanings, kept the same in every figure of this study.
CLARIFIED = dict(face=PAPER, hatch="")
ILLEGIBLE = dict(face=PAPER, hatch="xxxx")
ACCUSE = dict(face="#ecc6c5", hatch="....")   # light failure red: an accusation with no error present

# Rows, weakest to strongest. The order is our guess and is itself something calibration measures.
READERS = ["gpt-5-nano, low effort", "gpt-4.1-mini", "gpt-5-mini, low effort",
           "gpt-5-mini, medium effort", "gpt-5.6-luna, low effort"]
# Columns, from evidence next to the error to evidence far from it.
ERRORS = ["Contradicted in the\nsame paragraph", "Contradicted by a result\nin another section",
          "Error carried into\nlater results", "Inconvenient result\nleft out of the summary"]

# Toy model behind the prediction. Catch probability is logistic in (reader strength - error difficulty
# - illegibility penalty). Weaker readers make more false accusations. All numbers are assumptions.
STRENGTH = [-1.0, 0.0, 0.9, 1.7, 2.8]
DIFFICULTY = [-0.6, 0.4, 1.2, 1.9]
PENALTY = 1.6          # how much harder the illegible copy makes an error, in the same units
SLOPE = 1.8
ACCUSATION = [0.30, 0.18, 0.10, 0.07, 0.05]

# What counts as the window, matching the selection rule in the doc.
MIN_CLARIFIED, MAX_ILLEGIBLE, MAX_ACCUSE = 0.70, 0.35, 0.10

# Text that changes when the grid shows measured rates instead of predictions.
WINDOW_LABEL = "Predicted window"
NOTE = "All bars are predictions from the toy model described in the doc, not measured data."


def logistic(x):
    return 1 / (1 + math.exp(-SLOPE * x))


def predict(r, c):
    clarified = logistic(STRENGTH[r] - DIFFICULTY[c])
    illegible = logistic(STRENGTH[r] - DIFFICULTY[c] - PENALTY)
    return clarified, illegible, ACCUSATION[r]


def in_window(vals):
    return vals[0] >= MIN_CLARIFIED and vals[1] <= MAX_ILLEGIBLE and vals[2] <= MAX_ACCUSE


def figure(values=None, path=None):
    values = values or predict
    path = path or FIG
    nr, nc = len(READERS), len(ERRORS)
    cw, ch, lw, top = 2.3, 1.55, 1.75, 1.25         # inches: cell width, cell height, label column, header
    W, H = lw + nc * cw, top + nr * ch + 0.5
    fig = plt.figure(figsize=(W, H))

    handles = [swatch(**CLARIFIED), swatch(**ILLEGIBLE), swatch(**ACCUSE), swatch(edge=INK)]
    handles[-1].set_linewidth(2.4)
    labels = ["Errors caught: clarified copy", "Errors caught: illegible copies",
              "False accusations: clean copies", WINDOW_LABEL]
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False,
               handlelength=1.6, columnspacing=1.4, fontsize=11)

    fig.text((lw + nc * cw / 2) / W, (H - 0.52) / H, "Error type, from evidence nearby to evidence far away",
             ha="center", va="center", fontsize=12)
    for c, name in enumerate(ERRORS):
        fig.text((lw + (c + 0.5) * cw) / W, (H - top + 0.08) / H, name, ha="center", va="bottom",
                 fontsize=11, linespacing=1.2)
    fig.text(0.12 / W, (H - top + 0.08) / H, "Reader setting,\nweakest at top", ha="left", va="bottom",
             fontsize=12, linespacing=1.2)

    for r, reader in enumerate(READERS):
        y0 = H - top - (r + 1) * ch
        fig.text(0.12 / W, (y0 + ch / 2) / H, reader.replace(", ", ",\n"), ha="left", va="center",
                 fontsize=11, linespacing=1.2)
        for c in range(nc):
            x0 = lw + c * cw
            vals = values(r, c)
            win = in_window(vals)
            fig.patches.append(Rectangle((x0 / W, y0 / H), cw / W, ch / H, transform=fig.transFigure,
                                         fill=False, edgecolor=INK, lw=2.4 if win else 0.5, zorder=3))
            ax = fig.add_axes([(x0 + 0.42) / W, (y0 + 0.2) / H, (cw - 0.6) / W, (ch - 0.4) / H])
            axes_style(ax, 0, 1, bold_top=False)
            ax.set_xlim(-0.7, 2.7)
            ax.set_xticks([])
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["0", "1"] if c == 0 else ["", ""], fontsize=9)
            for i, (v, st) in enumerate(zip(vals, (CLARIFIED, ILLEGIBLE, ACCUSE))):
                bar(ax, i, 0.66, v, face=st["face"], hatch=st["hatch"], label=None)

    fig.text((lw + nc * cw / 2) / W, 0.12 / H,
             NOTE,
             ha="center", va="bottom", fontsize=11, color=SOFT)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save(fig, path)


# The key: what each pattern in a grid cell means and which way to move. Gates come first.
GATES = [
    ("Clean copies get accused", "Any cell", "No catch rate can be read. A reader that flags everything also catches every planted error.",
     "Let the prompt allow the reader to report no errors. Require each finding to name the claim and the evidence against it. Read a sample of accusations by hand. If a clean paper really has the flaw, set that error aside and record it."),
    ("Error fails a construction check", "Any cell", "The strong reader cannot find the error in some copy, or the pair check finds it planted unequally.",
     "Repair or remove the error before any weak reader sees it. These checks never use weak-reader results, so removing errors here cannot bias the comparison."),
]
KEY = [
    ("Low", "Low", "Floor. The reader cannot check these errors even in clear text.",
     "Move down the grid to a stronger reader or larger budget, or left to errors with nearer evidence. Use the pointed check to tell a reader that cannot find the error from one that cannot reason it through."),
    ("High", "High", "Ceiling. The reader sees through the illegibility, as in V1.",
     "Move up to a weaker reader or smaller budget, or right to errors with more distant evidence. Run the local-window check, since an error caught from its paragraph alone does not test following the paper, and the edit-artifact check, since a planted passage can stand out against obfuscated text."),
    ("Low", "High", "The clarified copy makes errors harder to find. This is a clarification problem, never a reader problem.",
     "Run the fidelity and pair checks, and the pointed check on the clarified copy. Fix the clarification prompt, for example by keeping evidence near claims or capping length, and rebuild."),
    ("High", "Low", "Desired.",
     "Confirm the cell before trusting it. A clear original such as Dropout should score close to its clarified copy. False accusations should stay low. The held-out kind of illegibility should also score low."),
]


def markdown():
    rows = ["| Clarified copy | Illegible copies | What it means | What to do |", "|---|---|---|---|"]
    for name, where, means, do in GATES:
        rows.append(f"| **Gate:** {name} | {where} | {means} | {do} |")
    for cl, il, means, do in KEY:
        rows.append(f"| {cl} | {il} | {means} | {do} |")
    block = "\n".join(rows)
    with open(DOC) as f:
        doc = f.read()
    doc = re.sub(r"<!-- key:start -->.*<!-- key:end -->",
                 lambda _: f"<!-- key:start -->\n{block}\n<!-- key:end -->", doc, flags=re.S)
    with open(DOC, "w") as f:
        f.write(doc)


if __name__ == "__main__":
    figure()
    markdown()
    print("wrote", FIG, "and", DOC)
