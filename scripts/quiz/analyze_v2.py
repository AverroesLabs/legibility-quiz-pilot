"""Pooled analysis over the core papers of plan v2 (docs/preregistration-v2.md, "Pooled model").

Preregistered (secondary to the per-paper results):
  per reader:  score ~ version + (1 | question) + (1 | paper), fit by statsmodels MixedLM (paper groups, question
               variance component), with 95% intervals from a bootstrap over questions within papers.
  pooled:      version x reader; reported as each weaker reader's change from the original minus the smart reader's
               change on the same questions (also the exploratory comparison named in the report's next steps).

A question's score in a condition is its mean over samples, from takes_v2.question_scores (counterfactual pair rule
included), so it matches takes-summary.json. Only questions with a score in every condition for that reader enter.
Two variants: all versions, and versions that pass a paper's information check only (as for the preregistered bars).

Usage: python3.12 scripts/quiz/analyze_v2.py   ->  data/analysis/v2-pooled.json, and data/analysis/v2-tables.md
(the report's tables: pooled with intervals that resample questions within papers, then each paper's takes-summary)
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/quiz"))
import takes_v2  # noqa: E402

RUNS = ROOT / "data/runs"
OUT = ROOT / "data/analysis/v2-pooled.json"
TABLES = ROOT / "data/analysis/v2-tables.md"
SHORT = {takes_v2.filter_v2.SMART: "GPT-5.6-sol (smart)", "google/gemma-4-31b-it": "Gemma-4-31B",
         "mistralai/ministral-8b-2512": "Ministral-8B", "meta-llama/llama-3.2-3b-instruct": "Llama-3.2-3B"}
HEAD = ("| Reader | No paper | Original | Obfuscated (Δ) | Compressed (Δ) | Compressed + obfuscated (Δ) |\n"
        "|---|---|---|---|---|---|")
CONDS = takes_v2.CONDITIONS
VERSIONS = ("obf", "comp", "compobf")
SMART = takes_v2.filter_v2.SMART
N_BOOT, SEED = 10000, 20261103


def load():
    """Long table: paper, question, reader, condition, score (mean over samples); and info-check pass per version."""
    rows, info = [], {}
    for p in sorted(RUNS.glob("v2-*/takes-summary.json")):  # written when a paper's takes finish
        d = p.parent
        summ = json.loads((d / "takes-summary.json").read_text())
        paper, items, _, per_q, _ = takes_v2.question_scores(d)
        ids = {it["id"] for it in items}
        info[paper] = {v: c["pass"] for v, c in summ.get("information_check", {}).items()}
        for (rd, c, i), v in per_q.items():
            if i in ids:
                rows.append((paper, f"{paper}:{i}", rd, c, sum(v) / len(v)))
    df = pd.DataFrame(rows, columns=["paper", "q", "reader", "cond", "score"])
    return df, info


def wide(df, reader):
    w = df[df.reader == reader].pivot_table(index=["paper", "q"], columns="cond", values="score")
    return w.dropna(subset=list(CONDS)).reset_index()


def boot_within(paper_codes, x, rng):
    """Bootstrap means of x (rows = questions; columns = statistics), resampling questions within each paper."""
    groups = [np.flatnonzero(paper_codes == g) for g in np.unique(paper_codes)]
    out = np.empty((N_BOOT, x.shape[1]))
    for b in range(N_BOOT):
        idx = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups])
        out[b] = x[idx].mean(0)
    return np.percentile(out, [2.5, 97.5], axis=0).T


def mixed(w):
    """Fixed effects of each condition against orig from score ~ cond + (1|paper) + (1|question)."""
    long = w.melt(id_vars=["paper", "q"], value_vars=list(CONDS), var_name="cond", value_name="score")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = smf.mixedlm("score ~ C(cond, Treatment('orig'))", long, groups="paper", re_formula="1",
                          vc_formula={"q": "0 + C(q)"}).fit(reml=True)
    fe = {k.split("[T.")[1].rstrip("]"): round(float(v), 4) for k, v in fit.fe_params.items() if "[T." in k}
    return fe, {"paper_var": round(float(fit.cov_re.iloc[0, 0]), 5),
                "question_var": round(float(fit.vcomp[0]), 5), "resid_var": round(float(fit.scale), 5),
                "converged": bool(fit.converged)}


def analyze(df, info, only_passing):
    rng = np.random.default_rng(SEED)
    readers = [r for r in takes_v2.READERS if r in set(df.reader)]
    res = {"readers": {}, "vs_smart": {}}
    ws = {r: wide(df, r) for r in readers}
    keep = lambda w, v: w[[info.get(p, {}).get(v, False) for p in w.paper]] if only_passing else w
    for r in readers:
        w = ws[r]
        fe, var = mixed(w)
        out = {"questions": len(w), "papers": int(w.paper.nunique()), "mixed_model": {"vs_orig": fe, **var},
               "means": {c: round(float(w[c].mean()), 4) for c in CONDS}, "minus_orig": {}}
        for v in ("none",) + VERSIONS:
            wv = keep(w, v) if v != "none" else w
            if not len(wv):
                continue
            x = (wv[v] - wv["orig"]).to_numpy()[:, None]
            lo, hi = boot_within(wv.paper.to_numpy(), x, rng)[0]
            out["minus_orig"][v] = {"mean": round(float(x.mean()), 4), "ci95": [round(lo, 4), round(hi, 4)],
                                    "questions": len(wv), "papers": int(wv.paper.nunique())}
        if only_passing:
            out["mixed_model"] = "not fit (variant applies the information check per version)"
        res["readers"][r] = out
    # version x reader: (weaker change) - (smart change), paired by question
    sm = ws.get(SMART)
    for r in readers:
        if r == SMART or sm is None:
            continue
        j = ws[r].merge(sm, on=["paper", "q"], suffixes=("", "_s"))
        res["vs_smart"][r] = {}
        for v in VERSIONS:
            jv = keep(j, v)
            if not len(jv):
                continue
            x = ((jv[v] - jv["orig"]) - (jv[v + "_s"] - jv["orig_s"])).to_numpy()[:, None]
            lo, hi = boot_within(jv.paper.to_numpy(), x, rng)[0]
            per_paper = jv.assign(x=x[:, 0]).groupby("paper").x.mean()
            res["vs_smart"][r][v] = {"mean": round(float(x.mean()), 4), "ci95": [round(lo, 4), round(hi, 4)],
                                     "questions": len(jv), "papers_below_zero": int((per_paper < 0).sum()),
                                     "papers": int(len(per_paper)),
                                     "per_paper": {k: round(float(v), 4) for k, v in per_paper.items()}}
    return res


WEAKER = [r for r in takes_v2.READERS if r != SMART]
GATE_KEPT, GATE_SMART_ORIG, BAR_DROP = 30, 0.6, 0.10


def bars(df, info):
    """The preregistered bars (docs/preregistration-v2.md, "Paper gate" and "Proposed bars"), per paper from each
    takes-summary.json. Drop = orig - version (positive = lower score on the version).
    Bar 5's "questions needed" is not specified further in the preregistration; it is computed here (pass 19) as
    n = ((1.96 + 0.84) * sd / delta)^2, 80% power for a two-sided 95% interval to exclude 0, where sd is the
    per-question spread of the paired difference (mean over samples) pooled over gated papers."""
    out = {"gate": {}, "per_paper": {}, "bar3": {}, "bar4": {}, "bar5": {}}
    summ = {json.loads(p.read_text())["paper"]: json.loads(p.read_text()) for p in RUNS.glob("v2-*/takes-summary.json")}
    for paper, s in sorted(summ.items()):
        sm = s["readers"][SMART]
        gated = s["kept"] >= GATE_KEPT and sm["orig"]["mean"] >= GATE_SMART_ORIG
        out["gate"][paper] = {"kept": s["kept"], "smart_orig": sm["orig"]["mean"], "passes": gated}
        pp = {}
        for r in WEAKER:
            res = s["readers"][r]
            pp[r] = {}
            for v in VERSIONS:
                drop, ci = -res[v]["minus_orig"], [-x for x in reversed(res[v]["minus_orig_ci95"])]
                sdrop = -sm[v]["minus_orig"]
                b1 = drop >= BAR_DROP and ci[0] > 0
                b2 = sdrop < drop and info[paper].get(v, False)
                pp[r][v] = {"drop": round(drop, 4), "ci95": [round(x, 4) for x in ci], "smart_drop": round(sdrop, 4),
                            "bar1": b1, "bar2": b2, "conclusive": ci[0] > 0 or ci[1] < 0}
        out["per_paper"][paper] = pp
    gated = [p for p, g in out["gate"].items() if g["passes"]]
    for v in ("obf", "compobf"):
        valid = [p for p in gated if info[p].get(v, False)]
        out["bar3"][v] = {"gated_papers_with_valid_version": len(valid), "by_reader": {}}
        for r in WEAKER:
            hits = [p for p in valid if out["per_paper"][p][r][v]["bar1"] and out["per_paper"][p][r][v]["bar2"]]
            out["bar3"][v]["by_reader"][r] = {"papers_meeting_bars_1_and_2": hits, "count": len(hits),
                                              "holds": bool(valid) and len(hits) >= len(valid) / 2}
        out["bar3"][v]["holds"] = any(x["holds"] for x in out["bar3"][v]["by_reader"].values())
    for r in WEAKER:
        rows = [out["per_paper"][p][r] for p in gated]
        out["bar4"][r] = {"papers": len(rows), "mean_comp_drop": round(float(np.mean([x["comp"]["drop"] for x in rows])), 4) if rows else None,
                          "mean_obf_drop": round(float(np.mean([x["obf"]["drop"] for x in rows])), 4) if rows else None,
                          "papers_comp_drop_at_least_obf": sum(x["comp"]["drop"] >= x["obf"]["drop"] for x in rows)}
    g = df[df.paper.isin(gated)]
    for r in takes_v2.READERS:
        w = wide(g, r)
        rec = {}
        for v in VERSIONS:
            sd = float((w[v] - w["orig"]).groupby(w.paper).std(ddof=1).pow(2).mean() ** 0.5) if len(w) else float("nan")
            widths = [summ[p]["readers"][r][v]["minus_orig_ci95"][1] - summ[p]["readers"][r][v]["minus_orig_ci95"][0] for p in gated]
            concl = [(lambda c: c[0] > 0 or c[1] < 0)(summ[p]["readers"][r][v]["minus_orig_ci95"]) for p in gated]
            rec[v] = {"per_question_sd": round(sd, 4), "median_interval_width": round(float(np.median(widths)), 4) if widths else None,
                      "conclusive_fraction": round(sum(concl) / len(concl), 3) if concl else None,
                      "questions_for_0.10": int(np.ceil(((1.96 + 0.84) * sd / 0.10) ** 2)),
                      "questions_for_0.15": int(np.ceil(((1.96 + 0.84) * sd / 0.15) ** 2))}
        out["bar5"][r] = rec
    out["gated_papers"] = gated
    return out


def cell(m, ci, d=None, dci=None):
    """Absolute score with interval; for versions, the score and the change from original with its interval."""
    if d is None:
        return f"{m:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]"
    txt = f"{d:+.2f} [{dci[0]:+.2f}, {dci[1]:+.2f}]".replace("-", "−")
    return f"{m:.2f} ({'**' + txt + '**' if dci[0] > 0 or dci[1] < 0 else txt})"


def tables(df, info):
    """Markdown tables for the report: pooled over core papers, then one per paper, largest first."""
    rng = np.random.default_rng(SEED)
    lines = [f"### All core papers pooled ({len(info)} papers, {df.q.nunique()} questions, 3 samples)", "", HEAD]
    for r in takes_v2.READERS:
        w = wide(df, r)
        x = w[list(CONDS)].to_numpy()
        diffs = x - w[["orig"]].to_numpy()
        ci = boot_within(w.paper.to_numpy(), np.hstack([x, diffs]), rng)
        k = len(CONDS)
        row = [cell(x[:, j].mean(), ci[j]) if c in ("none", "orig") else
               cell(x[:, j].mean(), None, diffs[:, j].mean(), ci[k + j]) for j, c in enumerate(CONDS)]
        lines.append(f"| {SHORT[r]} | " + " | ".join(row) + " |")
    lines += ["", "Score [95% interval]; Δ is the change from the original, paired by question; bold where its "
              "interval excludes 0. Intervals resample questions within each paper, so papers keep their weight. "
              "All versions included, whether or not they pass a paper's information check. "
              "From [`analyze_v2.py`](../scripts/quiz/analyze_v2.py)."]
    papers = sorted(RUNS.glob("v2-*/takes-summary.json"), key=lambda p: -json.loads(p.read_text())["kept"])
    for p in papers:
        s = json.loads(p.read_text())
        lines += ["", f"### {s['paper']} ({s['kept']} questions)", "", HEAD]
        for r in takes_v2.READERS:
            res = s["readers"][r]
            row = [cell(res[c]["mean"], res[c]["ci95"]) if c in ("none", "orig") else
                   cell(res[c]["mean"], None, res[c]["minus_orig"], res[c]["minus_orig_ci95"]) for c in CONDS]
            lines.append(f"| {SHORT[r]} | " + " | ".join(row) + " |")
        ic = s.get("information_check", {})
        meta = json.loads((p.parent / "versions/meta.json").read_text()) if (p.parent / "versions/meta.json").exists() else {}
        orig_len = (p.parent / "versions/orig.md").stat().st_size if (p.parent / "versions/orig.md").exists() else None
        lens = ", ".join(f"{v} {(p.parent / f'versions/{v}.md').stat().st_size / orig_len:.2f}x"
                         for v in ("comp", "compobf") if orig_len and (p.parent / f"versions/{v}.md").exists())
        lines += ["", "Information check (smart reader's score / original): " +
                  ", ".join(f"{v} {c['ratio']:.2f}{'' if c['pass'] else ' (fails, below 0.9)'}" for v, c in ic.items()) +
                  f". Lengths (bytes): {lens}. Failed grades {s['failed']}, unanswered {s['unanswered']}. "
                  f"Data: [`takes-summary.json`](../{p.relative_to(ROOT)})."]
    TABLES.write_text("\n".join(lines) + "\n")


def main():
    df, info = load()
    if df.empty:
        sys.exit("no core-paper takes yet")
    out = {"papers": sorted(info), "information_check": info,
           "all_versions": analyze(df, info, False), "passing_versions": analyze(df, info, True),
           "bars": bars(df, info)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    a = out["all_versions"]
    print(f"{len(info)} core papers; information check failures:",
          {p: [v for v, ok in c.items() if not ok] for p, c in info.items() if not all(c.values())} or "none")
    for r, o in a["readers"].items():
        mm = o["mixed_model"]
        print(f"{r:34} n={o['questions']:4}  " + "  ".join(
            f"{v} {d['mean']:+.3f} [{d['ci95'][0]:+.3f},{d['ci95'][1]:+.3f}] (mm {mm['vs_orig'].get(v, float('nan')):+.3f})"
            for v, d in o["minus_orig"].items()))
    print("weaker reader's change minus the smart reader's, same questions:")
    for r, o in a["vs_smart"].items():
        print(f"{r:34}  " + "  ".join(f"{v} {d['mean']:+.3f} [{d['ci95'][0]:+.3f},{d['ci95'][1]:+.3f}] "
                                      f"({d['papers_below_zero']}/{d['papers']} papers <0)" for v, d in o.items()))
    b = out["bars"]
    print(f"gate: {len(b['gated_papers'])} of {len(info)} papers pass", {p: g for p, g in b["gate"].items() if not g["passes"]})
    for v, x in b["bar3"].items():
        print(f"bar 3 ({v}): holds={x['holds']}, gated papers with valid version {x['gated_papers_with_valid_version']}:",
              {r.split('/')[1]: y["count"] for r, y in x["by_reader"].items()})
    for r, x in b["bar5"].items():
        print(f"bar 5 {r:34}", {v: (y["median_interval_width"], y["conclusive_fraction"], y["questions_for_0.10"], y["questions_for_0.15"]) for v, y in x.items()})
    tables(df, info)
    print("wrote", OUT.relative_to(ROOT), "and", TABLES.relative_to(ROOT))


if __name__ == "__main__":
    main()
