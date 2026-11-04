"""Exploratory human comparison on OneStopQA (docs/decisions.md, 2026-11-04).

Our readers answer the OneStopQA multiple-choice questions that Prolific participants answered, on the Advanced and
Elementary versions of each paragraph and with no paragraph, and rate each paragraph version's readability.

    python scripts/onestop/run_onestop.py run       # model calls (resumable; cached in data/onestop/calls.jsonl)
    python scripts/onestop/run_onestop.py analyze   # writes data/analysis/onestop.json
"""
import csv, json, random, re, sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "quiz"))
import llm  # noqa: E402

HUMAN = ROOT / "data" / "onestop" / "human"  # copied from berzak/onestop-qa@7a05875 human_experiments/prolific
OUT = ROOT / "data" / "onestop"
SMART = "openai/gpt-5.6-sol"
READERS = [SMART, "google/gemma-4-31b-it", "mistralai/ministral-8b-2512", "meta-llama/llama-3.2-3b-instruct"]
SAMPLES = {SMART: 1}
DEFAULT_SAMPLES = 3
LEVELS = ["Ele", "Adv"]
LETTERS = "ABCD"

QA = """Read the passage, then answer the question about it.

<passage>
{paragraph}
</passage>

Question: {question}
{options}

Reply with only the letter of the correct answer (A, B, C or D)."""

GUESS = """Answer this multiple-choice question about a news article you have not seen. Give your best guess.

Question: {question}
{options}

Reply with only the letter of the answer (A, B, C or D)."""

RATE = """Below is a paragraph from a news article. Judge how readable it is for this human reader: an adult native English speaker with no special background, reading carefully with as much time as they need.

<passage>
{paragraph}
</passage>

How easy would it be for that reader to understand this paragraph, from 1 to 10?
- 1: They could not follow it; they could not explain what it says.
- 5: They could follow parts of it, but some of it would be unclear to them.
- 10: They could follow all of it easily, and explain what it says.

Judge the paragraph as it is written. Reply with only a JSON object: {{"reason": "<at most 30 words>", "rating": <integer from 1 to 10>}}"""


def n_samples(r):
    return SAMPLES.get(r, DEFAULT_SAMPLES)


def load_items():
    """OneStopQA items from the Prolific QA file: question id -> {question, options (correct first), paragraph by level}."""
    qs = {}
    for row in csv.DictReader(open(HUMAN / "qa.tsv"), delimiter="\t"):
        iid = row["item_id"]
        if not iid.startswith("os"):
            continue
        qid, level = iid.rsplit("_", 1)
        q = qs.setdefault(qid, {"question": row["question"], "options": row["options"].split("|"), "paragraph": {}})
        q["paragraph"][level] = row["paragraph"].strip().strip('"')
    return qs


def human_answers(name):
    out = []
    for row in csv.DictReader(open(HUMAN / name), delimiter="\t"):
        if row["item_id"].startswith("os"):
            qid, level = row["item_id"].rsplit("_", 1)
            out.append((qid, level, row["answer_response"] == "0"))
    return out


def shuffled(q, qid, sample):
    order = list(range(4))
    random.Random(f"{qid}|{sample}").shuffle(order)
    text = "\n".join(f"{LETTERS[i]}. {q['options'][j]}" for i, j in enumerate(order))
    return text, LETTERS[order.index(0)]


def parse_letter(s):
    """The letter if the reply or any line of it is just a letter (any markup), or the reply states
    'answer is X', 'Answer: X', 'choose: X' or 'guess is X'; else None (scored wrong)."""
    t = s.strip().replace("*", "")
    lines = [ln.strip() for ln in t.split("\n") if ln.strip()] or [""]
    letter = r"\W*\(?([ABCD])\)?[\W]*"
    m = re.fullmatch(letter, t) or next((mm for ln in lines if (mm := re.fullmatch(letter, ln))), None) or \
        re.match(r"\W*([ABCD])[\.\):]\s", t + " ") or \
        re.search(r"(?:answer(?: is)?|choose|choice(?: is)?|guess(?: is)?)\s*[:\-]?\s*\(?([ABCD])\b", t, re.I)
    return m.group(1).upper() if m else None


def parse_rating(s):
    m = re.search(r'"rating"\s*:\s*(\d+)', s)
    return int(m.group(1)) if m and 1 <= int(m.group(1)) <= 10 else None


def jobs(qs):
    paras = {}
    for qid, q in qs.items():
        for lv, p in q["paragraph"].items():
            paras[(qid.rsplit("_", 1)[0], lv)] = p  # one paragraph per article_paragraph and level
    for r in READERS:
        for s in range(n_samples(r)):
            for qid, q in qs.items():
                opts, gold = shuffled(q, qid, s)
                for lv in LEVELS:
                    yield ("qa", r, s, qid, lv, gold,
                           QA.format(paragraph=q["paragraph"][lv], question=q["question"], options=opts))
                yield ("qa", r, s, qid, "none", gold, GUESS.format(question=q["question"], options=opts))
            for (pid, lv), p in paras.items():
                yield ("rate", r, s, pid, lv, None, RATE.format(paragraph=p))


def run():
    qs = load_items()
    led = llm.Ledger(OUT / "calls.jsonl", spend_cap=30)
    js = list(jobs(qs))
    random.Random(0).shuffle(js)  # interleave providers so each provider's concurrency limit is used at once
    print(f"{len(qs)} questions, {len(js)} jobs, {len(led.cache)} cached", flush=True)

    def one(j):
        kind, r, s, ident, lv, gold, prompt = j
        mt = 4000 if r == SMART else 300
        try:
            rec = led.call(r, [{"role": "user", "content": prompt}], tag=f"onestop-{kind}-{lv}", sample=s,
                           max_tokens=mt, temperature=0.7, seed=s)
        except Exception as e:  # noqa: BLE001  logged as an error row by the ledger
            return {"kind": kind, "reader": r, "sample": s, "id": ident, "level": lv, "error": str(e)[:200]}
        c = rec["content"]
        row = {"kind": kind, "reader": r, "sample": s, "id": ident, "level": lv, "raw": c[:300]}
        if kind == "qa":
            row.update(gold=gold, answer=parse_letter(c), correct=parse_letter(c) == gold)
        else:
            row["rating"] = parse_rating(c)
        return row

    rows = []
    with ThreadPoolExecutor(48) as ex:
        for i, row in enumerate(ex.map(one, js)):
            rows.append(row)
            if i % 500 == 0:
                print(i, f"new spend ${led.new_spend:.2f}", flush=True)
    with open(OUT / "results.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"done; new spend ${led.new_spend:.2f}; errors {sum('error' in r for r in rows)}")


def boot_diff(by_q, a, b, n=10000, seed=0):
    """Mean(b) - mean(a) with a 95% bootstrap interval clustered by question. by_q: qid -> {level: [0/1...]}."""
    qids = [q for q in by_q if by_q[q].get(a) and by_q[q].get(b)]
    da = np.array([np.mean(by_q[q][a]) for q in qids])
    db = np.array([np.mean(by_q[q][b]) for q in qids])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(qids), (n, len(qids)))
    boots = db[idx].mean(1) - da[idx].mean(1)
    return float(db.mean() - da.mean()), [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))], len(qids)


def by_article(by_q):
    """Accuracy per article (qid os<article>_<paragraph>_<question>) and level: the sample points in the figure."""
    acc = defaultdict(lambda: defaultdict(list))
    for qid, lv in by_q.items():
        for level, xs in lv.items():
            acc[level][qid.split("_")[0]].extend(xs)
    return {level: {a: float(np.mean(xs)) for a, xs in arts.items()} for level, arts in acc.items()}


def summarize(by_q):
    out = {"by_article": by_article(by_q)}
    for lv in LEVELS + ["none"]:
        vals = [x for q in by_q.values() for x in q.get(lv, [])]
        if vals:
            # interval: bootstrap over questions of each question's mean (answers within a question are clustered)
            qm = np.array([np.mean(q[lv]) for q in by_q.values() if q.get(lv)])
            w = np.array([len(q[lv]) for q in by_q.values() if q.get(lv)])
            idx = np.random.default_rng(1).integers(0, len(qm), (10000, len(qm)))
            boots = (qm[idx] * w[idx]).sum(1) / w[idx].sum(1)
            out[lv] = {"acc": float(np.mean(vals)), "n": len(vals),
                       "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}
    d, ci, nq = boot_diff(by_q, "Ele", "Adv")
    out["adv_minus_ele"] = {"diff": d, "ci95": ci, "questions": nq}
    return out


def analyze():
    res = {"readers": {}, "humans": {}, "ratings": {}}
    hq = defaultdict(lambda: defaultdict(list))
    for qid, lv, ok in human_answers("qa.tsv"):
        hq[qid][lv].append(ok)
    for qid, lv, ok in human_answers("guess.tsv"):
        hq[qid]["none"].append(ok)
    res["humans"] = summarize(hq)
    # Human response time per answer (reading the paragraph and answering), by level.
    rtq = defaultdict(lambda: defaultdict(list))
    for row in csv.DictReader(open(HUMAN / "qa.tsv"), delimiter="\t"):
        if row["item_id"].startswith("os"):
            qid, level = row["item_id"].rsplit("_", 1)
            rtq[qid][level].append(np.log(float(row["RT"]) / 1000))
    d, ci, nq = boot_diff(rtq, "Ele", "Adv")
    res["humans"]["rt"] = {lv: float(np.exp(np.median([x for q in rtq.values() for x in q.get(lv, [])])))
                           for lv in LEVELS}
    res["humans"]["rt"]["adv_over_ele"] = {"ratio": float(np.exp(d)), "ci95": [float(np.exp(c)) for c in ci],
                                           "questions": nq, "note": "geometric-mean ratio, Advanced / Elementary"}
    rows = [json.loads(l) for l in open(OUT / "results.jsonl")]
    res["errors"] = sum("error" in r for r in rows)
    res["unparsed"] = sum(r.get("kind") == "qa" and "error" not in r and r["answer"] is None for r in rows)
    for r in READERS:
        mq = defaultdict(lambda: defaultdict(list))
        rt = defaultdict(lambda: defaultdict(list))
        for x in rows:
            if x["reader"] != r or "error" in x:
                continue
            if x["kind"] == "qa":
                mq[x["id"]][x["level"]].append(x["correct"])
            elif x["rating"] is not None:
                rt[x["id"]][x["level"]].append(x["rating"])
        res["readers"][r] = summarize(mq)
        rr = {lv: float(np.mean([v for p in rt.values() for v in p.get(lv, [])])) for lv in LEVELS}
        pd = [np.mean(rt[p]["Adv"]) - np.mean(rt[p]["Ele"]) for p in rt if rt[p].get("Adv") and rt[p].get("Ele")]
        rr["adv_minus_ele"] = float(np.mean(pd))
        rr["paragraphs"] = len(pd)
        rr["share_adv_lower"] = float(np.mean([x < 0 for x in pd]))
        res["ratings"][r] = rr
        # per-question agreement with humans on the paragraph conditions
        pairs = [(np.mean(hq[q][lv]), np.mean(mq[q][lv])) for q in mq for lv in LEVELS if hq[q].get(lv) and mq[q].get(lv)]
        h, m = np.array(pairs).T
        res["readers"][r]["item_corr_with_humans"] = float(np.corrcoef(h, m)[0, 1])
    res["source"] = "berzak/onestop-qa@7a058755e1ff0d3f835a67109c669c7e294d4239 human_experiments/prolific"
    (ROOT / "data" / "analysis" / "onestop.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    {"run": run, "analyze": analyze}[sys.argv[1]]()
