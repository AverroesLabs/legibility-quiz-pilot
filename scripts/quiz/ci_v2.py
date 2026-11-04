"""95% intervals for the blog figures (data/analysis/v2-ci.json).

Quiz scores: per reader and condition, each question's score is the mean over its samples (pair rule applied, as in
takes_v2.question_scores). Two summaries, each with a bootstrap that resamples questions within each paper:
  pooled      mean over all kept questions (the report's pooled mean)
  paper_mean  mean of the per-paper means (each paper weighted equally; used where papers differ by condition)
Ratings (human-reader judge prompt, 1 to 10): mean over papers of each paper's mean rating, bootstrap over papers.

    python scripts/quiz/ci_v2.py
"""
import json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import takes_v2  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CONDS = ("none", "orig", "comp", "obf", "compobf", "rem", "imp", "imp2")
N, SEED = 4000, 20261104


def ci(a):
    return [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)]


def quiz_intervals(prefix="v2-"):
    runs = sorted(d for d in (ROOT / "data/runs").glob(prefix + "*") if (d / "takes-summary.json").exists())
    per = {}  # (reader, cond) -> {paper: array of question scores}
    for d in runs:
        files = ("takes-v2.jsonl",) + tuple(f for f in ("takes-gap.jsonl", "takes-imp2.jsonl") if (d / f).exists())
        paper, items, rows, per_q, _ = takes_v2.question_scores(d, files)
        paper = paper.removesuffix("-rem")
        for (r, c, i), v in per_q.items():
            per.setdefault((r, c), {}).setdefault(paper, []).append(sum(v) / len(v))
    rng = np.random.default_rng(SEED)
    out = {}
    for (r, c), papers in per.items():
        arrs = [np.array(v) for v in papers.values()]
        sums, counts, pmeans = np.zeros(N), sum(len(a) for a in arrs), np.zeros((N, len(arrs)))
        for k, a in enumerate(arrs):
            b = a[rng.integers(0, len(a), (N, len(a)))]
            sums += b.sum(1)
            pmeans[:, k] = b.mean(1)
        out.setdefault(r, {})[c] = {
            "pooled": round(float(np.concatenate(arrs).mean()), 4), "pooled_ci95": ci(sums / counts),
            "paper_mean": round(float(np.mean([a.mean() for a in arrs])), 4), "paper_mean_ci95": ci(pmeans.mean(1)),
            "papers": len(arrs), "questions": int(counts),
            "paper_mean_by_paper": {p: round(float(np.mean(v)), 4) for p, v in papers.items()}}
    return out


def rating_intervals():
    rng = np.random.default_rng(SEED)
    j = json.loads((ROOT / "data/analysis/v2-judge-human.json").read_text())
    g = json.loads((ROOT / "data/analysis/v2-gap.json").read_text())["judge"]["human"]
    out = {}
    for r in j["readers"]:
        pm = {c: {p: j["per_paper"][p][r]["rating"][c] for p in j["papers"] if c in j["per_paper"][p][r]["rating"]}
              for c in ("orig", "comp", "obf", "compobf")}
        for c in ("rem", "imp"):
            if c in g.get(r, {}):
                pm[c] = g[r][c]["paper_means"]
        imp2 = {}  # Follow-up 3: human-prompt ratings of the stronger implicit version
        for p in j["papers"]:
            f = ROOT / f"data/runs/v2-{p}/judge-human-imp2.jsonl"
            if f.exists():
                v = [x["rating"] for x in map(json.loads, f.read_text().splitlines()) if x.get("reader") == r
                     and "rating" in x]
                if v:
                    imp2[p] = sum(v) / len(v)
        if imp2:
            pm["imp2"] = imp2
        for c, by in pm.items():
            a = np.array(list(by.values()), dtype=float)
            out.setdefault(r, {})[c] = {"mean": round(float(a.mean()), 4), "papers": len(a), "by_paper": by,
                                        "ci95": ci(a[rng.integers(0, len(a), (N, len(a)))].mean(1))}
    return out


if __name__ == "__main__":
    # quiz_requiz: the quiz written on each paper's "key results removed" version and taken on that version
    # (scripts/quiz/gap_requiz.py); its "orig" condition is that version, "none" is no paper.
    res = {"quiz": quiz_intervals(), "quiz_requiz": quiz_intervals("v2rem-"), "rating_human_prompt": rating_intervals(),
           "method": __doc__.strip().splitlines()[2:8]}
    (ROOT / "data/analysis/v2-ci.json").write_text(json.dumps(res, indent=1))
    pooled = json.loads((ROOT / "data/analysis/v2-pooled.json").read_text())["all_versions"]["readers"]
    for r, v in res["quiz"].items():
        print(r.split("/")[1], {c: (v[c]["pooled"], pooled.get(r, {}).get("means", {}).get(c), v[c]["pooled_ci95"])
                                for c in ("orig", "obf") if c in v})
    print({r.split("/")[1]: {c: (x["mean"], x["ci95"]) for c, x in v.items()}
           for r, v in res["rating_human_prompt"].items()})
