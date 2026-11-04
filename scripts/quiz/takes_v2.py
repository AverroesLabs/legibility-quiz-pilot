#!/usr/bin/env python3.12
"""Plan v2 takes: every reader answers the kept questions in each of the five conditions, graded as answers arrive.

Usage: python3.12 scripts/quiz/takes_v2.py --run v2dev-2607.05615v2 [take|summary] [--cap 10]

take: 4 readers (filter_v2.SMART + filter_v2.WEAKER) x 5 conditions (none, orig, obf, comp, compobf) x SAMPLES
samples. The kept questions (kept-v2.json) go in sittings of at most 25 in a seeded random order; each sitting
with the condition's full version text (prompts.TAKE_PAPER), or no text (prompts.TAKE_NONE). The sitting split
depends on reader and sample but not on the condition, so versions are compared on identical sittings.
Temperature 0.7, seed from the sample number. Each sitting's answers go to the grader as soon as it returns.
Grading (plan v2): GLM-5.3 at minimal reasoning, prompts.GRADE_V2, which quotes the words that earn each point;
"I don't know" scores 0 without a call; answers are cut to ANSWER_WORDS words before grading (graders favour
length). A point whose quote is not found in the answer is flagged (quote_ok false), not changed.
Writes takes-v2.jsonl (one row per question, reader, condition and sample). Resumable via calls.jsonl.

summary: per reader and condition, mean score over kept questions with a 95% bootstrap interval over questions
(each question's score is its mean over samples), a counterfactual pair scored 0 in a sample unless both twins
earn their conclusion point, the position between no paper (0) and original (1), the drop from the original with
an interval, the rate of "right conclusion, wrong reason", and the information check (the smart reader's score on
each version at least INFO_MIN of its score on the original). Writes takes-summary.json.
"""
import argparse, json, random, re, sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import filter_v2, llm, prompts, quiz

READERS = (filter_v2.SMART,) + filter_v2.WEAKER
CONDITIONS = ("none", "orig", "obf", "comp", "compobf")
SAMPLES = 3
ANSWER_WORDS = 100  # above the 99th percentile of every reader's answers in the dev filters (51 to 92 words)
INFO_MIN = 0.9
GRADER = filter_v2.GRADER


def cap_words(answer):
    words = answer.split()
    if len(words) <= ANSWER_WORDS:
        return answer, False
    return " ".join(words[:ANSWER_WORDS]), True


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s.%-]", " ", s.lower())).strip()


def parse_grade(content):
    """Quotes of LaTeX answers carry backslashes such as \\( that are invalid JSON escapes (4 of 2,940 dev grades,
    docs/run-log.md pass 16): double any backslash that does not start a valid escape, then parse."""
    try:
        return quiz.extract_json(content)
    except ValueError:
        return quiz.extract_json(re.sub(r'\\\\|\\(?![\\"/bfnrtu])',
                                        lambda m: m.group(0) if len(m.group(0)) == 2 else "\\\\", content))


def grade_v2(led, item, answer):
    n = len(item["rubric"])
    if not answer:
        return {"points": [False] * n, "quotes": [""] * n, "note": "no answer", "score": 0.0, "missing": True}
    if filter_v2.IDK.match(answer.strip()):
        return {"points": [False] * n, "quotes": [""] * n, "note": "I don't know", "score": 0.0}
    rubric = "\n".join(f"{i}. {p}" for i, p in enumerate(item["rubric"], 1))
    prompt = prompts.GRADE_V2.format(question=item["question"], answer=item["answer"], rubric=rubric, student=answer)
    msg = [{"role": "user", "content": prompt}]
    kw = dict(tag="grade_v2", reasoning_effort=quiz.GRADE_REASONING_EFFORT, timeout=180)
    res = None
    for sample in (0, 1):  # an unparseable reply is asked again once, as a new sample
        try:
            rec = led.call(GRADER, msg, max_tokens=8000, sample=sample, **kw)
        except llm.LengthError:
            rec = led.call(GRADER, msg, max_tokens=24000, sample=sample, **kw)
        try:
            res = parse_grade(rec["content"])
            break
        except ValueError:
            if sample:
                raise
    pts = [bool(p) for p in res["points"]]
    quotes = [str(x) for x in (res.get("quotes") or [])]
    if len(pts) != n:
        raise RuntimeError(f"grader returned {len(pts)} points for {n}")
    quotes = (quotes + [""] * n)[:n]
    ans = norm(answer)
    quote_ok = [(not p) or bool(norm(qt)) and norm(qt) in ans for p, qt in zip(pts, quotes)]
    return {"points": pts, "quotes": quotes, "quote_ok": quote_ok, "note": res.get("note"),
            "score": sum(pts) / n, "grade_call": rec["key"]}


def version_texts(d, conds=CONDITIONS):
    v = d / "versions"
    return {c: None if c == "none" else (v / f"{c}.md").read_text() for c in conds}


def kept_items(d):
    q = json.loads((d / "quiz-v2.json").read_text())
    kept = set(json.loads((d / "kept-v2.json").read_text())["kept"])
    return q["paper"], [it for it in q["items"] if it["id"] in kept]


def take(led, d, conds=CONDITIONS, out="takes-v2.jsonl"):
    """conds and out let gap_v2.py add conditions in a separate file, on the same sittings."""
    paper, items = kept_items(d)
    texts = version_texts(d, conds)
    jobs = [(r, c, s, g) for r in READERS for s in range(SAMPLES)
            for g in filter_v2.sittings(items, f"{paper}|take|{r}|s{s}") for c in conds]
    rows, gfuts = [], []
    with ThreadPoolExecutor(48) as gex:

        def grade_row(reader, cond, sample, sit, it, pos, ans, rec):
            capped, cut = cap_words(ans) if ans else (ans, False)
            row = {"item": it["id"], "type": it["type"], "pair": it["pair"], "reader": reader, "condition": cond,
                   "sample": sample, "sitting": sit, "position": pos, "answer": ans, "truncated": cut,
                   "words": len(ans.split()) if ans else 0, "finish": rec.get("finish"), "call": rec.get("key")}
            try:
                row.update(grade_v2(led, it, capped))
            except (RuntimeError, ValueError, KeyError) as e:
                row["error"] = str(e)[:300]
            return row

        def sit(j):
            reader, cond, sample, group = j
            sid = ",".join(it["id"] for it in group)
            try:
                got = filter_v2.sitting_one(led, paper, group, reader, texts[cond], f"take_{cond}", sample)
            except RuntimeError as e:  # the whole sitting failed: record each question as an error row
                got = [(it, i, None, {"error": str(e)[:300]}) for i, it in enumerate(group, 1)]
            for it, pos, ans, rec in got:
                if "error" in rec:
                    rows.append({"item": it["id"], "reader": reader, "condition": cond, "sample": sample,
                                 "error": rec["error"]})
                else:
                    gfuts.append(gex.submit(grade_row, reader, cond, sample, sid, it, pos, ans, rec))

        with ThreadPoolExecutor(32) as tex:
            list(tex.map(sit, jobs))
        rows += [f.result() for f in gfuts]
    rows.sort(key=lambda x: (READERS.index(x["reader"]), conds.index(x["condition"]), x["sample"],
                             int(x["item"][1:])))
    (d / out).write_text("".join(json.dumps(x) + "\n" for x in rows))
    bad = [x for x in rows if "error" in x]
    print(f"{paper} takes: {len(rows)} rows, {len(bad)} failed, "
          f"{sum(1 for x in rows if x.get('missing'))} unanswered, {sum(1 for x in rows if x.get('truncated'))} cut")
    return summary(d) if out == "takes-v2.jsonl" else rows


def boot(per_q, n=10000, seed=20261103):
    """95% interval of the mean over questions. per_q: list of per-question values (or tuples for a difference)."""
    rnd = random.Random(seed)
    k = len(per_q)
    means = sorted(sum(per_q[rnd.randrange(k)] for _ in range(k)) / k for _ in range(n))
    return [round(means[int(0.025 * n)], 4), round(means[int(0.975 * n) - 1], 4)]


def question_scores(d, files=("takes-v2.jsonl",)):
    """Scores per (reader, condition, kept question): one per sample, with the counterfactual pair rule applied.
    Returns (paper, items, rows, per_q, rcwr). Also used by make_figures.py, so figures match the summary.
    files: gap_v2.py adds "takes-gap.jsonl" (other conditions; the pair rule works within a condition)."""
    paper, items = kept_items(d)
    rows = [json.loads(l) for f in files for l in (d / f).read_text().splitlines()]
    ok = [r for r in rows if "error" not in r]
    twins = defaultdict(list)
    for it in items:
        if it["pair"]:
            twins[it["pair"]].append(it["id"])
    cell = {(r["reader"], r["condition"], r["sample"], r["item"]): r for r in ok}

    def score(r):  # counterfactual pair: no credit in a sample unless both twins earn the conclusion point
        if r["pair"]:
            for t in twins[r["pair"]]:
                o = cell.get((r["reader"], r["condition"], r["sample"], t))
                if o is None or not o["points"][0]:
                    return 0.0
        return r["score"]

    per_q = defaultdict(list)
    rcwr = defaultdict(lambda: [0, 0])  # right conclusion, wrong reason: point 1 earned, point 2 not
    for r in ok:
        per_q[(r["reader"], r["condition"], r["item"])].append(score(r))
        if r["points"][0]:
            rcwr[(r["reader"], r["condition"])][0] += 1
            rcwr[(r["reader"], r["condition"])][1] += not r["points"][1]
    return paper, items, rows, per_q, rcwr


def summary(d):
    paper, items, rows, per_q, rcwr = question_scores(d)
    ok = [r for r in rows if "error" not in r]
    ids = [it["id"] for it in items]
    out = {"paper": paper, "kept": len(ids), "samples": SAMPLES, "rows": len(rows),
           "failed": len(rows) - len(ok), "unanswered": sum(1 for r in ok if r.get("missing")),
           "truncated": sum(1 for r in ok if r.get("truncated")),
           "quote_not_found": sum(1 for r in ok for p, q in zip(r["points"], r.get("quote_ok", [])) if p and not q),
           "points_awarded": sum(sum(r["points"]) for r in ok), "readers": {}}
    for reader in READERS:
        res = {}
        qm = {c: {i: sum(v) / len(v) for i in ids if (v := per_q.get((reader, c, i)))} for c in CONDITIONS}
        for c in CONDITIONS:
            vals = list(qm[c].values())
            if not vals:
                continue
            m = sum(vals) / len(vals)
            res[c] = {"mean": round(m, 4), "ci95": boot(vals), "questions": len(vals),
                      "right_conclusion_wrong_reason": round(rcwr[(reader, c)][1] / max(rcwr[(reader, c)][0], 1), 4)}
            common = [i for i in qm[c] if i in qm["orig"]]
            if c != "orig" and common:
                diffs = [qm[c][i] - qm["orig"][i] for i in common]
                res[c]["minus_orig"] = round(sum(diffs) / len(diffs), 4)
                res[c]["minus_orig_ci95"] = boot(diffs)
        if "none" in res and "orig" in res and res["orig"]["mean"] > res["none"]["mean"]:
            lo, hi = res["none"]["mean"], res["orig"]["mean"]
            for c in res:
                res[c]["position"] = round((res[c]["mean"] - lo) / (hi - lo), 4)
        out["readers"][reader] = res
    smart = out["readers"].get(filter_v2.SMART, {})
    if "orig" in smart:
        out["information_check"] = {c: {"ratio": round(smart[c]["mean"] / smart["orig"]["mean"], 4),
                                        "pass": smart[c]["mean"] >= INFO_MIN * smart["orig"]["mean"]}
                                    for c in ("obf", "comp", "compobf") if c in smart}
    (d / "takes-summary.json").write_text(json.dumps(out, indent=1))
    print(f"{paper}: {out['kept']} kept, {out['failed']} failed, {out['unanswered']} unanswered, "
          f"{out['truncated']} cut, {out['quote_not_found']} of {out['points_awarded']} awarded points with quote not found")
    for reader, res in out["readers"].items():
        print(f"  {reader:34}", "  ".join(f"{c} {v['mean']:.2f} [{v['ci95'][0]:.2f},{v['ci95'][1]:.2f}]"
                                          for c, v in res.items()))
    print("  information check:", out.get("information_check"))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--cap", type=float, default=10.0)
    p.add_argument("cmd", choices=["take", "summary"], nargs="?", default="take")
    a = p.parse_args()
    d = quiz.ROOT / "data/runs" / a.run
    if a.cmd == "summary":
        summary(d)
    else:
        take(llm.Ledger(d / "calls.jsonl", a.cap), d)


if __name__ == "__main__":
    main()
