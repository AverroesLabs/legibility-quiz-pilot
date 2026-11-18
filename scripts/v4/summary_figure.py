"""Summary figure for report v4: clarified catch rate and clarified-minus-illegible gap for every reader setting
and design tested after the repair (experiments 4 to 8). No API calls; every number comes from the experiment
scripts' own measures(), so it matches their tables. Writes docs/figures/v4-summary.png and prints the rows."""
import sys
from pathlib import Path

import exp4_stronger as exp4
import exp5_pointed as exp5
import exp6_scatter as exp6
import exp7_multi as exp7
import exp8_limited as exp8

ROOT = Path(__file__).resolve().parents[2]


def rows():
    """(design, reader, clarified rate, gap) with (point, lo, hi) tuples, top to bottom."""
    r4, _ = exp4.measures()
    r5, _ = exp5.measures()
    r6, _ = exp6.measures()
    r7, _ = exp7.measures()
    out = []
    for rd in ["luna-low", "mini-med", "luna-med", "mini-high"]:
        out.append(("1 error, wording (exp 3r/4, 17 errors)", rd, r4[rd]["clarified"], r4[rd]["gap"]))
    for rd in ["luna-low", "mini-high"]:
        out.append(("1 error, layout: scatter (exp 6, 13 errors)", rd, r6[rd]["clarified"], r6[rd]["gap scatter"]))
    for rd in ["luna-low", "mini-high"]:
        out.append(("3-5 errors at once, wording (exp 7, 13 errors)", rd, r7[rd]["clarified"], r7[rd]["gap illegible"]))
    for rd in ["luna-low", "mini-high"]:
        out.append(("3-5 errors at once, scatter (exp 7)", rd, r7[rd]["clarified"], r7[rd]["gap scatter"]))
    for v, label in [("A", "sections up to 800 words, notes 300"), ("B", "sections, notes 150"),
                     ("C", "paragraphs up to 250 words, notes 300")]:
        if not (exp8.OUT / f"reviews-{v}.jsonl").exists():
            continue
        r8 = exp8.measures(v)
        for rd in [x for x in ["luna-low", "mini-med", "mini-high"] if x in r8]:
            out.append((f"limited view {v}: {label} (exp 8, 12 errors)", rd, r8[rd]["clarified"],
                        r8[rd]["gap illegible"]))
    for rd in ["luna-low", "mini-high"]:
        out.append(("pointed at the passage, wording (exp 5, 17 errors)", rd, r5[rd]["clarified"], r5[rd]["gap"]))
    return out


def main():
    sys.path.insert(0, str(ROOT / "scripts"))
    from averroes_figstyle import INK, QUIET, SOFT, save, plt
    rs = rows()
    n = len(rs)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 0.42 * n + 1.2), sharey=True,
                                 gridspec_kw={"width_ratios": [1, 1], "wspace": 0.08})
    ys = list(range(n))[::-1]
    for ax, col, lo, hi, ref, title in [(a1, 2, 0, 1.14, 0.7, "catch rate, clarified copy\n(dotted: proposed bar 0.70)"),
                                        (a2, 3, -0.45, 0.6, 0.0, "clarified minus illegible\n(dotted: +0.35, implied by bars 0.70 and 0.35)")]:
        for y, r in zip(ys, rs):
            v = r[col]
            ax.plot([v[1], v[2]], [y, y], color=INK, lw=1.4)
            ax.plot(v[0], y, "o", color=INK, ms=5, mfc="white" if r[1] == "luna-low" else INK)
            ax.text(hi, y, f"{v[0]:+.2f}" if col == 3 else f"{v[0]:.2f}", ha="right", va="center", fontsize=9,
                    color=SOFT)
        ax.axvline(ref, ls=":" if col == 2 else "-", lw=0.8, color="0.3")
        if col == 3:
            ax.axvline(0.35, ls=":", lw=0.8, color="0.3")
        ax.set_xlim(lo, hi)
        ax.set_title(title, fontsize=11)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="x", color=QUIET, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
    a1.set_yticks(ys, [r[1] for r in rs])
    a1.set_ylim(-0.7, n - 0.3)
    prev = None
    for y, r in zip(ys, rs):
        if r[0] != prev:
            a1.text(-0.02, y + 0.55, r[0], transform=a1.get_yaxis_transform(), ha="right", va="center",
                    fontsize=9.5, style="italic")
            if prev is not None:
                for ax in (a1, a2):
                    ax.axhline(y + 0.5, color=QUIET, lw=0.6)
            prev = r[0]
    save(fig, ROOT / "docs" / "figures" / "v4-summary.png")
    f = lambda v: f"{v[0]:.2f} [{v[1]:.2f}, {v[2]:.2f}]"
    print("| design | reader | clarified | clarified - illegible |")
    print("|---|---|---|---|")
    for r in rs:
        print(f"| {r[0]} | {r[1]} | {f(r[2])} | {f(r[3])} |")


if __name__ == "__main__":
    main()
