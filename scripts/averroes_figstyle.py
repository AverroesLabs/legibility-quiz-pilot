"""Averroes Labs figure style for matplotlib. Matches the legibility-quiz post figures.

Use: copy this file into a repo's scripts/ folder (pods and public repos cannot see this directory), then
    from averroes_figstyle import *
See FIGURE-STYLE.md next to this file for the rules behind it.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

# Ink. Black on white is the default and is always acceptable.
INK, SOFT, QUIET, PAPER = "#000000", "#4d4d4d", "#d6d6d6", "#ffffff"

# Muted colors, for when color helps the reader. Each has a dark and a light shade. The dark shades differ in
# lightness from each other (about L* 45 vs 69), so they stay apart when printed in grey. Still pair every color
# with a second cue (hatch, line style or marker) so the figure reads in black and white.
SAFETY, SAFETY_LIGHT = "#4E6E8E", "#B7C6D6"            # slate blue
CAPABILITY, CAPABILITY_LIGHT = "#C8A26B", "#E8D5B5"    # ochre
ACCENT = "#7D5A73"                                     # plum: one highlighted series, used rarely
# Red is reserved for failure (a sabotaged, retracted or failed result). Never use it for anything else.
FAILURE = "#b8403f"

# Hatches tell categories apart without color. Same meaning in every figure of a post.
HATCHES = ["", "////", "\\\\\\\\", "xxxx", "....", "--", "------"]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["cmr10", "Latin Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True, "axes.unicode_minus": False,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": INK,
    "font.size": 12, "axes.labelsize": 12, "xtick.labelsize": 12, "ytick.labelsize": 11, "legend.fontsize": 11.5,
    "axes.linewidth": 0.8, "hatch.linewidth": 0.6, "lines.linewidth": 1.4, "patch.linewidth": 0.9,
    "axes.prop_cycle": plt.cycler(color=[INK]), "legend.frameon": False,
    "figure.facecolor": PAPER, "savefig.facecolor": PAPER, "axes.facecolor": PAPER})


def axes_style(ax, lo, hi, ylabel=None, bold_top=True, grid="y"):
    """Hairline left axis, no top/right spines, the scale's limits drawn as bold lines, a quiet grid.
    Pass bold_top=False when the scale has no upper limit (counts, years)."""
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.set_ylim(lo, hi)
    ax.axhline(lo, color=INK, lw=1.8, zorder=5, clip_on=False)
    if bold_top:
        ax.axhline(hi, color=INK, lw=1.8, zorder=5, clip_on=False)
    if grid:
        ax.grid(axis=grid, color=QUIET, lw=0.6)
        ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", direction="out", length=3, width=0.8)
    if ylabel:
        ax.set_ylabel(ylabel)


def bar(ax, x, w, value, lo=0, hi=1, ci=None, face=PAPER, hatch="", edge=INK, label="{:.2f}", horizontal=False):
    """One outlined bar from the scale's floor, an optional 95% interval, and its value on a white tag at the foot."""
    if horizontal:
        ax.add_patch(Rectangle((lo, x - w / 2), value - lo, w, facecolor=face, hatch=hatch, edgecolor=edge, zorder=2))
    else:
        ax.add_patch(Rectangle((x - w / 2, lo), w, value - lo, facecolor=face, hatch=hatch, edgecolor=edge, zorder=2))
    if ci is not None:
        err = [[value - ci[0]], [ci[1] - value]]
        kw = dict(fmt="none", ecolor=INK, elinewidth=1.4, capsize=3, capthick=1.4, zorder=4)
        ax.errorbar(value if horizontal else x, x if horizontal else value,
                    **({"xerr": err} if horizontal else {"yerr": err}), **kw)
    if label:
        pos = (lo + 0.012 * (hi - lo), x) if horizontal else (x, lo + 0.012 * (hi - lo))
        ax.text(*pos, label.format(value), ha="left" if horizontal else "center",
                va="center" if horizontal else "bottom", fontsize=10 if w >= 0.24 else 8.5, zorder=6,
                bbox=dict(boxstyle="square,pad=0.08", fc=PAPER, ec="none"))


def legend(fig, handles, labels, y=0.99, ncol=None):
    """Legend centered above the plot, no frame. Prefer direct labels on lines where they fit."""
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol or len(handles),
               frameon=False, handlelength=1.6, columnspacing=1.2)


def swatch(face=PAPER, hatch="", edge=INK):
    return Patch(facecolor=face, hatch=hatch, edgecolor=edge, lw=0.9)


def save(fig, path, dpi=200):
    fig.savefig(path, dpi=dpi, facecolor=PAPER, bbox_inches="tight")
    plt.close(fig)
