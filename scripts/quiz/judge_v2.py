#!/usr/bin/env python3.12
"""Exploratory baseline (Jason, 2026-11-04; docs/plan-v2.md "Baseline: just ask a model"): instead of a quiz, ask
each reader how legible each version is, and compare with the quiz. Plan fixed in docs/decisions.md (pass 20)
before any rating was collected.

Usage: python3.12 scripts/quiz/judge_v2.py [rate|analyze] [--cap 40] [--stop-usage 639]

rate: the 4 quiz readers x 4 versions (orig, obf, comp, compobf) x SAMPLES samples for every core paper, one
version per call (prompts.JUDGE_V2, rating 1 to 10 with anchors at 1, 5 and 10, plus a reason of at most 50
words). Temperature 0.7, seed from the sample, as in the takes. A reply whose rating cannot be parsed is asked
again once as a new sample number (SAMPLES + k). Calls go to data/runs/v2-<paper>/calls.jsonl (tag judge_<cond>);
ratings to data/runs/v2-<paper>/judge-v2.jsonl.

analyze: readability formulas (textstat; LaTeX and markdown markup stripped) for each version, then per reader and
paper: mean rating per version, whether the original is rated above each hard version (paper level, ties count
half) and the share of the 9 sample pairs (original sample i, version sample j) where it is, the same two figures
for the quiz (paper-level means, and the 9 pairs of per-sample run scores), and Spearman correlations over papers
between judge and quiz (original rating vs original quiz score; rating drop vs quiz drop). Writes
data/analysis/v2-judge.json and v2-judge-tables.md.
"""
import argparse, json, re, sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import llm, prompts, quiz, takes_v2

READERS = takes_v2.READERS
VERSIONS = ("orig", "obf", "comp", "compobf")
HARD = VERSIONS[1:]
SAMPLES = 3
CORE = sorted(p.name[3:] for p in (quiz.ROOT / "data/runs").glob("v2-*") if (p / "takes-summary.json").exists())
OUT = quiz.ROOT / "data/analysis"
# --prompt self (the original baseline, "a reader at your level") or human (a named human reader; Jason, 2026-11-04).
VARIANTS = {"self": {"prompt": "JUDGE_V2", "file": "judge-v2.jsonl", "tag": "judge_", "out": "v2-judge"},
            "human": {"prompt": "JUDGE_HUMAN_V2", "file": "judge-human-v2.jsonl", "tag": "judge_human_",
                      "out": "v2-judge-human"}}
V = VARIANTS["self"]


def parse_rating(content):
    try:
        obj = quiz.extract_json(content)
        r = int(round(float(obj["rating"])))
        reason = str(obj.get("reason", ""))
    except (ValueError, KeyError, TypeError):
        m = re.search(r'"?rating"?\s*[:=]\s*(\d+(?:\.\d+)?)', content)
        if not m:
            return None
        r, reason = int(round(float(m.group(1)))), ""
    return (r, reason) if 1 <= r <= 10 else None


def rate_one(led, paper, text, reader, cond, sample, v=None):
    """v: a VARIANTS entry (gap_v2.py rates both prompts at once); default the one chosen by --prompt."""
    v = v or V
    msg = [{"role": "user", "content": getattr(prompts, v["prompt"]).format(paper=text)}]
    mt = 16000 if reader == takes_v2.filter_v2.SMART else 2000
    row = {"paper": paper, "reader": reader, "condition": cond, "sample": sample}
    for k in range(2):  # an unparseable reply is asked again once, as a new sample number
        s = sample if k == 0 else SAMPLES + sample
        try:
            rec = led.call(reader, msg, tag=f"{v['tag']}{cond}", sample=s, max_tokens=mt, temperature=0.7, seed=s,
                           timeout=600)
        except RuntimeError as e:
            row["error"] = str(e)[:300]
            return row
        got = parse_rating(rec["content"])
        if got:
            row.update(rating=got[0], reason=got[1], call=rec["key"], call_sample=s)
            row.pop("error", None)
            return row
        row["error"] = f"unparseable: {rec['content'][:200]}"
    return row


def rate(cap, stop_usage):
    usage = llm.account_usage()
    print(f"account usage ${usage:.2f}, stop line ${stop_usage}")
    if usage >= stop_usage:
        raise SystemExit("account usage at or above the stop line")

    def paper_job(paper):
        d = quiz.ROOT / f"data/runs/v2-{paper}"
        led = llm.Ledger(d / "calls.jsonl", cap / len(CORE))
        texts = {c: (d / "versions" / f"{c}.md").read_text() for c in VERSIONS}
        jobs = [(r, c, s) for r in READERS for c in VERSIONS for s in range(SAMPLES)]
        with ThreadPoolExecutor(16) as ex:
            rows = list(ex.map(lambda j: rate_one(led, paper, texts[j[1]], *j), jobs))
        (d / V["file"]).write_text("".join(json.dumps(x) + "\n" for x in rows))
        bad = sum("error" in x for x in rows)
        print(f"{paper}: {len(rows)} ratings, {bad} failed, new spend ${led.new_spend:.2f}", flush=True)
        return led.new_spend

    with ThreadPoolExecutor(len(CORE)) as ex:
        spent = sum(ex.map(paper_job, CORE))
    print(f"total new spend ${spent:.2f}; account usage now ${llm.account_usage():.2f}")


def plain(text):
    """Strip LaTeX math, commands and markdown markup so the formulas count words and sentences of prose."""
    t = re.sub(r"\$\$.*?\$\$|\\\[.*?\\\]|\\\(.*?\\\)|\$[^$\n]*\$", " X ", text, flags=re.S)
    t = re.sub(r"\\[a-zA-Z]+\*?(\{[^{}]*\})?", " ", t)
    t = re.sub(r"^\s*\|.*\|\s*$", " ", t, flags=re.M)  # tables
    t = re.sub(r"[#*_`>{}\\|]", " ", t)
    return t


def readability(paper):
    import textstat
    d = quiz.ROOT / f"data/runs/v2-{paper}/versions"
    out = {}
    for c in VERSIONS:
        t = plain((d / f"{c}.md").read_text())
        out[c] = {"fk_grade": textstat.flesch_kincaid_grade(t), "flesch_ease": textstat.flesch_reading_ease(t),
                  "fog": textstat.gunning_fog(t), "words": len(t.split())}
    return out


def above(a, b):
    """1 if a > b, 0.5 if tied, 0 if below."""
    return 1.0 if a > b else 0.5 if a == b else 0.0


def spearman(x, y):
    """[rho, p]; None when either input is constant (e.g. a reader rating every original 10)."""
    from scipy.stats import spearmanr
    if len(set(x)) < 2 or len(set(y)) < 2:
        return None
    r = spearmanr(x, y)
    return [round(float(r.statistic), 3), round(float(r.pvalue), 3)]


def quiz_runs(paper):
    """Per (reader, condition): paper-level quiz mean (takes-summary) and per-sample run scores (each sample's mean
    over kept questions, pair rule applied, as in figure 13)."""
    d = quiz.ROOT / f"data/runs/v2-{paper}"
    summ = json.loads((d / "takes-summary.json").read_text())
    _, items, _, per_q, _ = takes_v2.question_scores(d)
    runs = {}
    for r in READERS:
        for c in VERSIONS:
            vals = [per_q.get((r, c, it["id"])) for it in items]
            vals = [v for v in vals if v]
            runs[(r, c)] = [sum(v[s] for v in vals if len(v) > s) / sum(len(v) > s for v in vals)
                            for s in range(SAMPLES)]
    return {(r, c): summ["readers"][r][c]["mean"] for r in READERS for c in VERSIONS}, runs


def analyze():
    judge = defaultdict(list)
    failed = 0
    for paper in CORE:
        for line in (quiz.ROOT / f"data/runs/v2-{paper}/{V['file']}").read_text().splitlines():
            x = json.loads(line)
            if "rating" in x:
                judge[(paper, x["reader"], x["condition"])].append(x["rating"])
            else:
                failed += 1
    read = {p: readability(p) for p in CORE}
    res = {"papers": CORE, "failed_ratings": failed, "readability": read, "per_paper": {}, "readers": {}}
    for r in READERS:
        rr = {"rating_mean": {}, "judge_orig_above": {}, "judge_pairs_orig_above": {}, "quiz_orig_above": {},
              "quiz_pairs_orig_above": {}}
        qmeans, jmeans, qdrop, jdrop = {}, {}, defaultdict(list), defaultdict(list)
        for paper in CORE:
            qm, runs = quiz_runs(paper)
            jm = {c: sum(v) / len(v) for c in VERSIONS if (v := judge[(paper, r, c)])}
            res["per_paper"].setdefault(paper, {})[r] = {
                "rating": {c: round(jm[c], 3) for c in jm}, "ratings": {c: judge[(paper, r, c)] for c in VERSIONS},
                "quiz": {c: qm[(r, c)] for c in VERSIONS}}
            jmeans[paper], qmeans[paper] = jm.get("orig"), qm[(r, "orig")]
            for c in HARD:
                if "orig" in jm and c in jm:
                    rr["judge_orig_above"].setdefault(c, []).append(above(jm["orig"], jm[c]))
                    rr["judge_pairs_orig_above"].setdefault(c, []).extend(
                        above(a, b) for a in judge[(paper, r, "orig")] for b in judge[(paper, r, c)])
                    jdrop[c].append(jm["orig"] - jm[c])
                    qdrop[c].append(qm[(r, "orig")] - qm[(r, c)])
                rr["quiz_orig_above"].setdefault(c, []).append(above(qm[(r, "orig")], qm[(r, c)]))
                rr["quiz_pairs_orig_above"].setdefault(c, []).extend(
                    above(a, b) for a in runs[(r, "orig")] for b in runs[(r, c)])
        for k in ("judge_orig_above", "judge_pairs_orig_above", "quiz_orig_above", "quiz_pairs_orig_above"):
            rr[k] = {c: round(sum(v) / len(v), 3) for c, v in rr[k].items()}
        for c in VERSIONS:
            vals = [v for p in CORE for v in judge[(p, r, c)]]
            rr["rating_mean"][c] = round(sum(vals) / len(vals), 3) if vals else None
        ps = [p for p in CORE if jmeans[p] is not None]
        rr["spearman_orig_rating_vs_orig_quiz"] = spearman([jmeans[p] for p in ps], [qmeans[p] for p in ps])
        rr["spearman_rating_drop_vs_quiz_drop"] = {c: spearman(jdrop[c], qdrop[c]) for c in HARD if len(jdrop[c]) > 2}
        res["readers"][r] = rr
    res["readability_orig_easier"] = {
        f: {c: round(sum(above(read[p]["orig"][f], read[p][c][f]) if f == "flesch_ease"
                         else above(read[p][c][f], read[p]["orig"][f]) for p in CORE) / len(CORE), 3) for c in HARD}
        for f in ("fk_grade", "flesch_ease", "fog")}
    OUT.mkdir(exist_ok=True)
    (OUT / f"{V['out']}.json").write_text(json.dumps(res, indent=1))
    (OUT / f"{V['out']}-tables.md").write_text(tables(res))
    print(tables(res))


def sp(v):
    return "n/a (constant)" if v is None else f"{v[0]:+.2f} ({v[1]:.2f})"


def tables(res):
    short = {r: r.split("/")[1] for r in READERS}
    L = [f"Ratings 1 to 10, {len(res['papers'])} core papers x {SAMPLES} samples; {res['failed_ratings']} failed.", "",
         "| Reader | Rating: orig | obf | comp | compobf | Judge: orig above obf / comp / compobf (papers) | "
         "Quiz: same | Judge, sample pairs | Quiz, run pairs | Spearman orig rating vs orig quiz (p) |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r, rr in res["readers"].items():
        f = lambda d: " / ".join(f"{d[c]:.2f}" for c in HARD)
        s = rr["spearman_orig_rating_vs_orig_quiz"]
        L.append(f"| {short[r]} | " + " | ".join(f"{rr['rating_mean'][c]:.2f}" for c in VERSIONS) +
                 f" | {f(rr['judge_orig_above'])} | {f(rr['quiz_orig_above'])} | {f(rr['judge_pairs_orig_above'])}"
                 f" | {f(rr['quiz_pairs_orig_above'])} | {sp(s)} |")
    L += ["", "Spearman over papers, judge rating drop vs quiz score drop (orig minus version), rho (p):", "",
          "| Reader | obf | comp | compobf |", "| --- | --- | --- | --- |"]
    for r, rr in res["readers"].items():
        L.append(f"| {short[r]} | " + " | ".join(sp(v) for v in rr["spearman_rating_drop_vs_quiz_drop"].values()) + " |")
    L += ["", "Readability formulas: share of papers where the original reads easier than the version:", "",
          "| Formula | obf | comp | compobf |", "| --- | --- | --- | --- |"]
    for fml, v in res["readability_orig_easier"].items():
        L.append(f"| {fml} | " + " | ".join(f"{v[c]:.2f}" for c in HARD) + " |")
    return "\n".join(L) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["rate", "analyze"])
    p.add_argument("--cap", type=float, default=40.0, help="new spend cap in dollars, split evenly over papers")
    p.add_argument("--stop-usage", type=float, default=639.0)
    p.add_argument("--prompt", choices=sorted(VARIANTS), default="self")
    a = p.parse_args()
    V.clear()
    V.update(VARIANTS[a.prompt])
    rate(a.cap, a.stop_usage) if a.cmd == "rate" else analyze()


if __name__ == "__main__":
    main()
