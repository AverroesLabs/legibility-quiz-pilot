#!/usr/bin/env python3.12
"""Plan v2 quiz writer: a claim map and about 120 candidate questions per paper, from the original only.

Usage: python3.12 scripts/quiz/write_v2.py --run v2dev-2607.05615v2 --paper 2607.05615v2 [--batches 5] [--cap 6]

Batch 1 is WRITE_V2 (claim map and about 40 questions). Each later batch is WRITE_V2_MORE, given the claim map
and every question already written, so it covers new aspects. Batches are sequential for that reason. The
writer is Claude Opus 5.5 (plan v2). Each item is validated (type, rubric of 2 or 3 points, known claim ids,
counterfactual pairs of exactly two). Invalid items and near-duplicates (word-set Jaccard >= DUP_JACCARD
against an earlier item) are dropped and recorded with the reason. Writes data/runs/<run>/quiz-v2.json.
Resumable: repeated calls come from calls.jsonl.
"""
import argparse, hashlib, json, re, sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import llm, prompts, quiz

WRITER = "anthropic/claude-opus-5.5"
TYPES = ("failure", "why", "counterfactual", "transfer", "connect", "scope")
DUP_JACCARD = 0.6
BATCHES = 5  # 5 batches gave 160 candidates and 57 kept on the dev paper; 3 gave 111 and 30 (run-log, pass 14)
STOP = set("the a an of to in on for and or is are be by with that this which what would does do it its as "
           "from at if paper paper's why how than more less".split())


def words(q):
    return {w for w in re.findall(r"[a-z0-9]+", q.lower()) if w not in STOP}


def jaccard(a, b):
    return len(a & b) / max(1, len(a | b))


def rules():
    w = prompts.WRITE_V2
    return w[w.index("Every question must satisfy"):w.index("Return only JSON")].strip()


def call(led, prompt, tag):
    msg = [{"role": "system", "content": prompts.GUARD}, {"role": "user", "content": prompt}]
    rec = led.call(WRITER, msg, tag=tag, max_tokens=48000, timeout=1200)
    if rec["finish"] == "length":
        # Hidden reasoning grows with each batch (19k to 46k completion tokens over batches 1 to 4 on the dev
        # paper), and batch 5 hit 48k mid-JSON (docs/run-log.md pass 14). Opus 5.5 allows 128k output.
        rec = led.call(WRITER, msg, tag=tag, max_tokens=96000, timeout=2400)
    return rec


def filtered_before(led, tag):
    """True if the call log already holds a content-filter refusal for this tag (refusals are not cached)."""
    return led.path.exists() and any(r.get("tag") == tag and "content_filter" in (r.get("error") or "")
                                     for r in map(json.loads, led.path.read_text().splitlines()))


def validate(items, claim_ids, batch):
    """Return (valid items with batch-unique ids, rejected items with reasons)."""
    ok, bad = [], []
    for it in items:
        it = dict(it, batch=batch, orig_id=it.get("id"))
        reason = None
        if it.get("type") not in TYPES:
            reason = f"type {it.get('type')!r}"
        elif not (isinstance(it.get("rubric"), list) and 2 <= len(it["rubric"]) <= 3):
            reason = f"rubric size {len(it.get('rubric') or [])}"
        elif not all(isinstance(it.get(k), str) and it[k].strip() for k in ("question", "answer")):
            reason = "missing question or answer"
        elif not it.get("claims") or not set(it["claims"]) <= claim_ids:
            reason = f"claims {it.get('claims')}"
        (bad if reason else ok).append(dict(it, reject=reason) if reason else it)
    pairs = defaultdict(list)
    for it in ok:
        if it["type"] == "counterfactual":
            pairs[str(it.get("pair"))].append(it)
    for pid, members in pairs.items():
        if pid in ("None", "") or len(members) != 2:
            for it in members:
                ok.remove(it)
                bad.append(dict(it, reject=f"counterfactual pair {pid!r} has {len(members)} items"))
        else:
            for it in members:
                it["pair"] = f"b{batch}-{pid}"
    for it in ok:
        if it["type"] != "counterfactual":
            it["pair"] = None
    return ok, bad


def write(led, d, paper, batches):
    """Write the candidates for one paper into d/quiz-v2.json and return them."""
    text = quiz.paper_text(paper)
    rec = call(led, prompts.WRITE_V2.format(paper=text), f"write_v2:{paper}:b1")
    first = quiz.extract_json(rec["content"])
    claims = first["claims"]
    claim_ids = {c["id"] for c in claims}
    kept, rejected, costs, raw_counts = [], [], [rec["cost"]], [len(first["items"])]

    def add(items, batch):
        ok, bad = validate(items, claim_ids, batch)
        rejected.extend(bad)
        for it in ok:
            w = words(it["question"])
            # A pair's twins are near-identical by design, so they are not compared with each other.
            twin = next((k for k in kept if not (it["pair"] and k["pair"] == it["pair"])
                         and jaccard(w, words(k["question"])) >= DUP_JACCARD), None)
            if twin:
                rejected.append(dict(it, reject=f"near-duplicate of {twin['id']}"))
                continue
            it["id"] = f"q{len(kept) + 1}"
            kept.append(it)

    add(first["items"], 1)
    claim_text = json.dumps(claims, indent=1)
    filtered = None
    for b in range(2, batches + 1):
        existing = "\n".join(f"{k['id']} ({k['type']}): {k['question']}" for k in kept)
        prompt = prompts.WRITE_V2_MORE.format(rules=rules(), claims=claim_text, existing=existing, paper=text)
        tag = f"write_v2:{paper}:b{b}"
        try:
            if filtered_before(led, tag):
                raise RuntimeError("finish=content_filter (in the call log from an earlier run; not retried)")
            rec = call(led, prompt, tag)
        except RuntimeError as e:
            if "content_filter" not in str(e):
                raise
            # Pass 18: the provider's content filter refused batch 2 of a paper on LLM safety (2607.10112v1) on
            # all 4 tries. A later batch would send the same prompt (kept is unchanged), so stop here and keep
            # the batches already written; the shortfall shows in written_per_batch and filtered_batch.
            filtered = b
            print(f"{paper} batch {b}: refused by the content filter; keeping batches 1 to {b - 1}", flush=True)
            break
        items = quiz.extract_json(rec["content"])["items"]
        costs.append(rec["cost"])
        raw_counts.append(len(items))
        add(items, b)
        print(f"{paper} batch {b}: {len(items)} written, {len(kept)} kept so far", flush=True)

    # A pair that lost one twin to the duplicate check cannot be scored; drop the survivor too.
    pc = Counter(k["pair"] for k in kept if k["pair"])
    for k in [k for k in kept if k["pair"] and pc[k["pair"]] != 2]:
        kept.remove(k)
        rejected.append(dict(k, reject="twin dropped"))
    out = {"paper": paper, "writer": WRITER, "prompt_version": prompts.PROMPT_VERSION,
           "paper_sha256": hashlib.sha256(text.encode()).hexdigest(), "batches": batches,
           "written_per_batch": raw_counts, "cost_per_batch": costs, "claims": claims,
           "filtered_batch": filtered, "items": kept, "rejected": rejected,
           "types": dict(Counter(k["type"] for k in kept))}
    (d / "quiz-v2.json").write_text(json.dumps(out, indent=1))
    print(f"{paper}: {sum(raw_counts)} written, {len(kept)} valid candidates, {len(rejected)} rejected; "
          f"types {out['types']}; cost ${sum(costs):.2f}")
    for r in Counter(r["reject"].split(" of ")[0] for r in rejected).most_common():
        print("  rejected:", *r)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--paper", required=True)
    p.add_argument("--batches", type=int, default=BATCHES)
    p.add_argument("--cap", type=float, default=6.0)
    a = p.parse_args()
    d = quiz.ROOT / "data/runs" / a.run
    write(llm.Ledger(d / "calls.jsonl", a.cap), d, a.paper, a.batches)


if __name__ == "__main__":
    main()
