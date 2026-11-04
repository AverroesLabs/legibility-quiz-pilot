#!/usr/bin/env python3.12
"""Plan v2 question filters, run on the original version only.

Usage: python3.12 scripts/quiz/filter_v2.py --run v2dev-2607.05615v2 {retrieval,nopaper,withpaper,combine}
       [--reader MODEL] [--cap 3]

retrieval: a reader (google/gemma-4-31b-it per plan v2, and openai/gpt-5.6-sol; see docs/decisions.md, pass 13)
answers each candidate given only the 3 paragraphs of the original that the
keyword search (quiz.retrieve) returns for the question, one question per call, 1 sample. GLM-5.3 grades at
minimal reasoning. A question passes if the answer earns at most 1 rubric point (plan v2: "at most 1 of 3
points"; for 2-point rubrics this is also 1 point) for every retrieval reader. Writes
data/runs/<run>/filter-retrieval-<reader>.jsonl, one line per candidate with the passages' hash, the answer, the points and pass/fail. Resumable via calls.jsonl.

nopaper: every reader (smart and the 3 weaker) answers the questions with no paper (prompts.TAKE_NONE), in
sittings of at most 25 questions in a seeded random order. The smart reader takes NOPAPER_SMART_SAMPLES samples
(each with its own sitting order and seed) and its score is their mean; the weaker readers take 1 (docs/decisions.md,
pass 15). Writes filter-nopaper-<reader>.jsonl, one row per question and sample.
withpaper: the smart reader answers with the full original (prompts.TAKE_PAPER), same sittings. Writes
filter-withpaper-<reader>.jsonl.
combine: applies the filters in plan v2's order (no paper: smart's mean over its samples <= 0.5 and weaker readers' mean <= 0.5; with
paper: smart >= 0.5; retrieval: <= 1 point for every retrieval reader; counterfactual pairs: both twins earn the
conclusion point and >= 0.5 with the paper). A pair is one unit: if either twin fails any filter, both go.
Writes kept-v2.json (kept ids and the funnel, in order) and prints survival by type.
"""
import argparse, hashlib, json, random, re, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import llm, prompts, quiz

RETRIEVERS = ("google/gemma-4-31b-it", "openai/gpt-5.6-sol")
SMART = "openai/gpt-5.6-sol"
WEAKER = ("google/gemma-4-31b-it", "mistralai/ministral-8b-2512", "meta-llama/llama-3.2-3b-instruct")
SITTING_MAX = 25
# One sample of the smart reader's no-paper decision agreed with a second on only 80% of questions (pass 14).
NOPAPER_SMART_SAMPLES = 2
GRADER = "z-ai/glm-5.3"
IDK = re.compile(r"^\W*i (?:do not|don'?t) know\W*$", re.I)


def grade(led, item, answer):
    """Plan v2: "I don't know" scores 0 without a grader call."""
    if answer and IDK.match(answer.strip()):
        return {"points": [False] * len(item["rubric"]), "note": "I don't know", "score": 0.0}
    return quiz.grade_one(led, item, GRADER, answer)


def passages(text):
    """Paragraphs, except that a table is one passage. The paper text puts a blank line between table rows, so
    plain paragraphs split a table into header-less rows and the keyword search could not return a whole table
    (defect found on the dev paper, docs/run-log.md pass 13). A table block runs from a paragraph starting with
    '|' through short fragments (row labels) and further rows, up to and including its 'Table N:' caption."""
    out, block = [], None
    for p in quiz.paragraphs(text):
        if p.startswith("|"):
            block = (block or []) + [p]
        elif block is not None and re.match(r"Table \d+[:.]", p):
            out.append("\n".join(block + [p]))
            block = None
        elif block is not None and len(p) < 80:
            block.append(p)
        else:
            if block is not None:
                out.append("\n".join(block))
                block = None
            out.append(p)
    if block is not None:
        out.append("\n".join(block))
    return out


def retrieval_one(led, paper, text, it, reader):
    found = quiz.retrieve(text, it["question"], units=passages(text))
    prompt = prompts.TAKE_RETRIEVED.format(paper=found, questions=f"Q1. {it['question']}")
    msg = [{"role": "system", "content": prompts.GUARD}, {"role": "user", "content": prompt}]
    rec = led.call(reader, msg, tag=f"filter_retrieval:{paper}", temperature=0.7, seed=quiz.seed_int(0),
                   max_tokens=4000 if reader.startswith("google") else 16000, timeout=300)
    answer = quiz.parse_answers(rec["content"], 1).get(1) or rec["content"].strip()
    g = grade(led, it, answer)
    return {"item": it["id"], "type": it["type"], "pair": it["pair"],
            "passages_sha256": hashlib.sha256(found.encode()).hexdigest(), "answer": answer,
            "points": g["points"], "note": g.get("note"), "earned": sum(g["points"]),
            "pass": sum(g["points"]) <= 1, "call": rec["key"]}


def sittings(items, key):
    """Seeded random order, split into the fewest near-equal sittings of at most SITTING_MAX questions."""
    order = items[:]
    random.Random(quiz.seed_int(key)).shuffle(order)
    n = -(-len(order) // SITTING_MAX)
    return [order[i::n] for i in range(n)]


def sitting_one(led, paper, group, reader, text, check, sample=0):
    qs = "\n".join(f"Q{i}. {it['question']}" for i, it in enumerate(group, 1))
    prompt = (prompts.TAKE_NONE.format(questions=qs) if text is None
              else prompts.TAKE_PAPER.format(paper=text, questions=qs))
    msg = [{"role": "system", "content": prompts.GUARD}, {"role": "user", "content": prompt}]
    kw = dict(tag=(check if check.startswith("take") else f"filter_{check}") + f":{paper}", temperature=0.7, seed=quiz.seed_int(sample), timeout=900)
    mt = 16000 if reader == SMART else 8000
    try:
        rec = led.call(reader, msg, max_tokens=mt, **kw)
        got = quiz.parse_answers(rec["content"], len(group))
    except llm.LengthError:
        rec, got = {"finish": "length"}, {}
    if rec["finish"] == "length" and len(got) < len(group):
        rec = led.call(reader, msg, max_tokens=quiz.RETRY_MAX_TOKENS if reader == SMART else 16000, **kw)
        got = quiz.parse_answers(rec["content"], len(group))
    return [(it, i, got.get(i), rec) for i, it in enumerate(group, 1)]


def take_check(led, q, readers, text, check, d, samples=None):
    """Run the sittings of every reader and sample in parallel, grade each answer, write one file per reader.
    samples maps reader -> number of samples (default 1). Sample 0 keeps the original sitting key and seed, so
    earlier 1-sample runs come from the cache."""
    samples = samples or {}
    jobs = [(r, s, g) for r in readers for s in range(samples.get(r, 1))
            for g in sittings(q["items"], f"{q['paper']}|{check}|{r}" + (f"|s{s}" if s else ""))]
    with ThreadPoolExecutor(16) as ex:
        answered = list(ex.map(lambda j: (j[0], j[1], sitting_one(led, q["paper"], j[2], j[0], text, check, j[1])),
                               jobs))

    def grade_row(x):
        reader, sample, (it, pos, ans, rec) = x
        row = {"item": it["id"], "type": it["type"], "pair": it["pair"], "reader": reader, "sample": sample,
               "position": pos,
               "answer": ans, "finish": rec["finish"], "call": rec["key"]}
        try:
            g = grade(led, it, ans)
            row.update(points=g["points"], note=g.get("note"), score=g["score"])
        except (RuntimeError, ValueError, KeyError) as e:
            row["error"] = str(e)[:300]
        return row

    with ThreadPoolExecutor(24) as ex:
        rows = list(ex.map(grade_row, [(r, s, a) for r, s, sit in answered for a in sit]))
    for r in readers:
        rs = sorted((x for x in rows if x["reader"] == r), key=lambda x: (int(x["item"][1:]), x["sample"]))
        (d / f"filter-{check}-{r.split('/')[1]}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rs))
        ok = [x for x in rs if "error" not in x]
        print(f"{q['paper']} {check} {r}: {len(ok)} graded, {len(rs) - len(ok)} failed, "
              f"{sum(x['answer'] is None for x in rs)} unanswered, mean {sum(x['score'] for x in ok) / max(len(ok), 1):.3f}")


class Ungraded(RuntimeError):
    """A check left items ungraded (failed calls). Pass 18: this was SystemExit, which is not an Exception, so in
    run_v2.py it ended the paper's thread silently, with no status, on 2 core papers."""


def load(d, check, reader):
    path = d / f"filter-{check}-{reader.split('/')[1]}.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    bad = [r["item"] for r in rows if "error" in r]
    if bad:
        raise Ungraded(f"{path.name}: ungraded items {bad}; re-run that check first (it resumes from the cache)")
    out = {}
    for r in rows:  # several samples of one question: the score is their mean, other fields from sample 0
        if r["item"] in out:
            o = out[r["item"]]
            o["scores"].append(r["score"])
            o["score"] = sum(o["scores"]) / len(o["scores"])
        else:
            out[r["item"]] = dict(r, scores=[r["score"]]) if "score" in r else r
    return out


def combine(q, d):
    none = {r: load(d, "nopaper", r) for r in (SMART,) + WEAKER}
    withp = load(d, "withpaper", SMART)
    ret = {r: load(d, "retrieval", r) for r in RETRIEVERS}
    twins = {}
    for it in q["items"]:
        if it["pair"]:
            twins.setdefault(it["pair"], []).append(it["id"])

    def pair_right(it):
        if not it["pair"]:
            return True
        return all(withp[i]["points"][0] and withp[i]["score"] >= 0.5 for i in twins[it["pair"]])

    checks = [
        ("no paper", lambda it: none[SMART][it["id"]]["score"] <= 0.5
         and sum(none[r][it["id"]]["score"] for r in WEAKER) / len(WEAKER) <= 0.5),
        ("with paper", lambda it: withp[it["id"]]["score"] >= 0.5),
        ("retrieval", lambda it: all(ret[r][it["id"]]["earned"] <= 1 for r in RETRIEVERS)),
        ("counterfactual pair", pair_right),
    ]
    alive = list(q["items"])
    funnel = [{"stage": "written", "left": len(alive)}]
    for name, ok in checks:
        fail = {it["id"] for it in alive if not ok(it)}
        fail |= {i for it in alive if it["id"] in fail and it["pair"] for i in twins[it["pair"]]}
        before = len(alive)
        alive = [it for it in alive if it["id"] not in fail]
        funnel.append({"stage": name, "removed": before - len(alive), "left": len(alive)})
    # Each check on its own, on all candidates, so the overlap between checks is visible.
    alone = {name: sum(not ok(it) for it in q["items"]) for name, ok in checks}
    types = sorted({it["type"] for it in q["items"]})
    by_type = {t: {"written": sum(it["type"] == t for it in q["items"]),
                   "kept": sum(it["type"] == t for it in alive)} for t in types}
    out = {"paper": q["paper"], "kept": [it["id"] for it in alive], "funnel": funnel,
           "fail_alone": alone, "by_type": by_type}
    (d / "kept-v2.json").write_text(json.dumps(out, indent=1))
    for f in funnel:
        print(f"{f['stage']:>20}: {f.get('removed', '')!s:>4} removed, {f['left']} left")
    print("failing each check alone:", alone)
    for t, v in by_type.items():
        print(f"  {t}: {v['kept']}/{v['written']}")


def retrieval(led, q, d, reader):
    text = quiz.paper_text(q["paper"])

    def safe(it):
        try:
            return retrieval_one(led, q["paper"], text, it, reader)
        except (RuntimeError, ValueError, KeyError) as e:
            return {"item": it["id"], "type": it["type"], "error": str(e)[:300]}

    with ThreadPoolExecutor(12) as ex:
        rows = list(ex.map(safe, q["items"]))
    (d / f"filter-retrieval-{reader.split('/')[1]}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    ok = [r for r in rows if "error" not in r]
    print(f"{q['paper']} retrieval {reader}: {len(ok)} graded, {len(rows) - len(ok)} failed; "
          f"pass {sum(r['pass'] for r in ok)}")


def run_check(led, q, d, check, reader=RETRIEVERS[0]):
    if check == "combine":
        return combine(q, d)
    if check == "nopaper":
        return take_check(led, q, (SMART,) + WEAKER, None, "nopaper", d, {SMART: NOPAPER_SMART_SAMPLES})
    if check == "withpaper":
        return take_check(led, q, (SMART,), quiz.paper_text(q["paper"]), "withpaper", d)
    return retrieval(led, q, d, reader)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--cap", type=float, default=3.0)
    p.add_argument("--reader", default=RETRIEVERS[0], choices=RETRIEVERS)
    p.add_argument("check", choices=["retrieval", "nopaper", "withpaper", "combine"])
    a = p.parse_args()
    d = quiz.ROOT / "data/runs" / a.run
    q = json.loads((d / "quiz-v2.json").read_text())
    run_check(llm.Ledger(d / "calls.jsonl", a.cap), q, d, a.check, a.reader)


if __name__ == "__main__":
    main()
