"""Does grading with minimal hidden reasoning agree with the pilot's full-reasoning grades?

Re-grades a stratified sample of pilot answers with GLM-5.3 at reasoning effort "minimal" and compares them with
the original GLM-5.3 grades (default reasoning). Writes data/runs/grader-check/{calls,results}.jsonl and summary.md.
Usage: python scripts/quiz/grader_check.py [--per-cell 8]
"""
import argparse
import json
import random
import statistics
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import llm  # noqa: E402
import quiz  # noqa: E402

RUNS = Path(__file__).resolve().parents[2] / "data/runs"
OUT = RUNS / "grader-check"
GRADER = "z-ai/glm-5.3"


def sample_jobs(per_cell, seed=20261103):
    jobs = []
    for d in sorted(RUNS.glob("pilot-*")):
        if not d.is_dir():
            continue
        quiz_items = {it["id"]: it for it in json.loads((d / "quiz-claude-opus-5.5-1-screened.json").read_text())["items"]}
        answers = {}
        for line in (d / "answers.jsonl").read_text().splitlines():
            a = json.loads(line)
            answers[(a["item"], a["reader"], a["cond"], a["sample"])] = a.get("answer")
        cells = defaultdict(list)
        for line in (d / "grades.jsonl").read_text().splitlines():
            g = json.loads(line)
            k = (g["item"], g["reader"], g["cond"], g["sample"])
            if g["grader"] == GRADER and answers.get(k) and g["item"] in quiz_items:
                cells[(g["reader"], g["cond"])].append((k, g["points"], g["score"]))
        rng = random.Random(f"{seed}:{d.name}")
        for (reader, cond), rows in sorted(cells.items()):
            for k, pts, score in rng.sample(rows, min(per_cell, len(rows))):
                jobs.append({"paper": d.name.removeprefix("pilot-"), "item": k[0], "reader": reader, "cond": cond,
                             "sample": k[3], "answer": answers[k], "item_def": quiz_items[k[0]],
                             "orig_points": pts, "orig_score": score})
    return jobs


def kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cell", type=int, default=8)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    led = llm.Ledger(OUT / "calls.jsonl", spend_cap=5.0)
    jobs = sample_jobs(a.per_cell)
    print(f"{len(jobs)} answers to re-grade", flush=True)

    def run(j):
        try:
            return j, quiz.grade_one(led, j["item_def"], GRADER, j["answer"], reasoning_effort="minimal")
        except Exception as e:  # report, do not hide
            return j, {"error": str(e)[:200]}

    with ThreadPoolExecutor(8) as ex:
        results = list(ex.map(run, jobs))
    rows = []
    with (OUT / "results.jsonl").open("w") as f:
        for j, r in results:
            row = {k: j[k] for k in ("paper", "item", "reader", "cond", "sample", "orig_points", "orig_score")}
            row.update({"new_points": r.get("points"), "new_score": r.get("score"), "error": r.get("error")})
            rows.append(row)
            f.write(json.dumps(row) + "\n")

    ok = [r for r in rows if r["new_points"] is not None and len(r["new_points"]) == len(r["orig_points"])]
    pa = [p for r in ok for p in r["orig_points"]]
    pb = [p for r in ok for p in r["new_points"]]
    same_score = sum(r["orig_score"] == r["new_score"] for r in ok)
    diffs = [r["new_score"] - r["orig_score"] for r in ok]
    calls = [json.loads(l) for l in (OUT / "calls.jsonl").read_text().splitlines()]
    cost = sum(c.get("cost") or 0 for c in calls)
    L = ["# Grader check: minimal reasoning vs full reasoning (GLM-5.3)", "",
         f"Re-graded {len(ok)} of {len(rows)} sampled pilot answers ({len(rows) - len(ok)} failed or mismatched). "
         f"Spend ${cost:.2f}, about ${cost / max(len(calls), 1):.5f} per answer.", "",
         f"- Rubric points: {len(pa)}; agreement {sum(x == y for x, y in zip(pa, pb)) / len(pa):.1%}; "
         f"Cohen's kappa {kappa(pa, pb):.2f}",
         f"- Same answer score: {same_score / len(ok):.1%}",
         f"- Mean score change (minimal minus full): {statistics.mean(diffs):+.3f}", "",
         "## By condition (the check that matters: grader change must not differ by version)", "",
         "| Condition | Answers | Full | Minimal | Change | Same score |", "|---|---|---|---|---|---|"]
    by = defaultdict(list)
    for r in ok:
        by[r["cond"]].append(r)
    for c in sorted(by):
        rs = by[c]
        L.append(f"| {c} | {len(rs)} | {statistics.mean(r['orig_score'] for r in rs):.3f} | "
                 f"{statistics.mean(r['new_score'] for r in rs):.3f} | "
                 f"{statistics.mean(r['new_score'] - r['orig_score'] for r in rs):+.3f} | "
                 f"{sum(r['orig_score'] == r['new_score'] for r in rs) / len(rs):.0%} |")
    L += ["", "## By reader", "", "| Reader | Answers | Change | Same score |", "|---|---|---|---|"]
    byr = defaultdict(list)
    for r in ok:
        byr[r["reader"]].append(r)
    for rd in sorted(byr):
        rs = byr[rd]
        L.append(f"| {rd} | {len(rs)} | {statistics.mean(r['new_score'] - r['orig_score'] for r in rs):+.3f} | "
                 f"{sum(r['orig_score'] == r['new_score'] for r in rs) / len(rs):.0%} |")
    (OUT / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
