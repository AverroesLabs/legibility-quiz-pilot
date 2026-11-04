#!/usr/bin/env python3
"""Follow-up 2 (docs/preregistration-v2.md): a quiz written on the hidden-gap version, as in real use.

For each core paper, the existing "insight removed" version (data/runs/v2-<paper>/versions/rem.md) becomes a paper
of its own, data/papers/<paper>-rem/text.md. The plan v2 pipeline then runs on it unchanged: write_v2 writes the
quiz, filter_v2 filters it, and takes_v2 has all four readers take the kept questions with that text ("orig") and
with no paper ("none"). Run directories: data/runs/v2rem-<paper>/.

    python scripts/quiz/gap_requiz.py run [PAPER ...] [--stop-usage 839]
    python scripts/quiz/gap_requiz.py analyze        # writes data/analysis/v2-gap-requiz.json
"""
import argparse, json, shutil, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import quiz, run_v2, takes_v2  # noqa: E402

ROOT = quiz.ROOT
CORE = sorted(p.name[3:] for p in (ROOT / "data/runs").glob("v2-*") if (p / "versions/rem.md").exists())


def prepare(paper):
    pid = f"{paper}-rem"
    src = ROOT / "data/runs" / f"v2-{paper}" / "versions/rem.md"
    pd = ROOT / "data/papers" / pid
    pd.mkdir(parents=True, exist_ok=True)
    text = src.read_text()
    (pd / "text.md").write_text(text)
    (pd / "SOURCE.md").write_text(f"Copy of data/runs/v2-{paper}/versions/rem.md (the 'insight removed' version), "
                                  "used as a paper of its own by scripts/quiz/gap_requiz.py.\n")
    d = ROOT / "data/runs" / f"v2rem-{paper}"
    (d / "versions").mkdir(parents=True, exist_ok=True)
    (d / "versions/orig.md").write_text(text)
    return pid, d


def one(paper, cap, stop):
    pid, d = prepare(paper)
    job = run_v2.Paper(pid, d, cap, stop)
    try:
        job.quiz_and_filters()
        job.stage("takes", takes_v2.take, job.led, d, ("none", "orig"))
        status = "done"
    except run_v2.Stopped as e:
        status = f"stopped: {e}"
    except Exception as e:  # noqa: BLE001  recorded per paper so others continue
        status = f"failed: {type(e).__name__}: {e}"
    (d / "requiz-status.json").write_text(json.dumps({"paper": paper, "status": status}))
    print(f"{paper}: {status}", flush=True)
    return status


def summ(d):
    s = json.loads((d / "takes-summary.json").read_text())
    return {"kept": s["kept"], "readers": {r: {c: s["readers"][r][c]["mean"] for c in ("orig", "none")
                                               if c in s["readers"][r]} for r in s["readers"]}}


def analyze():
    out = {"papers": {}}
    for p in CORE:
        a, b = ROOT / f"data/runs/v2-{p}", ROOT / f"data/runs/v2rem-{p}"
        if not (b / "takes-summary.json").exists():
            continue
        out["papers"][p] = {"original_quiz": summ(a), "rem_quiz": summ(b)}
    readers = takes_v2.READERS
    pooled = {}
    for r in readers:
        for which in ("original_quiz", "rem_quiz"):
            for c in ("orig", "none"):
                v = [x[which]["readers"][r][c] for x in out["papers"].values() if c in x[which]["readers"].get(r, {})]
                pooled.setdefault(r, {}).setdefault(which, {})[c] = round(sum(v) / len(v), 4) if v else None
        diffs = [x["rem_quiz"]["readers"][r]["orig"] - x["original_quiz"]["readers"][r]["orig"]
                 for x in out["papers"].values()]
        pooled[r]["rem_minus_original_with_paper"] = {
            "mean": round(sum(diffs) / len(diffs), 4), "papers_lower": sum(d < 0 for d in diffs), "papers": len(diffs)}
    out["pooled_paper_means"] = pooled
    out["kept"] = {p: (x["original_quiz"]["kept"], x["rem_quiz"]["kept"]) for p, x in out["papers"].items()}
    (ROOT / "data/analysis/v2-gap-requiz.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({"kept": out["kept"], "pooled": pooled}, indent=1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["run", "analyze"])
    p.add_argument("papers", nargs="*")
    p.add_argument("--cap", type=float, default=15.0)
    p.add_argument("--stop-usage", type=float, default=839.0)
    a = p.parse_args()
    if a.cmd == "analyze":
        return analyze()
    papers = a.papers or CORE
    with ThreadPoolExecutor(len(papers)) as ex:
        print(list(ex.map(lambda x: one(x, a.cap, a.stop_usage), papers)), flush=True)


if __name__ == "__main__":
    main()
