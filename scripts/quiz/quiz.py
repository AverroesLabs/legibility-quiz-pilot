#!/usr/bin/env python3.12
"""Quiz pipeline: write a quiz on a paper, have readers take it, grade, summarize.

Run directory: data/runs/<run>/ holding calls.jsonl (every API call), quiz-<writer>-<seed>.json,
answers.jsonl, grades.jsonl and summary.md. All steps are resumable: repeated calls come from calls.jsonl.
"""
import argparse, hashlib, json, random, re, statistics, sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import llm, prompts

ROOT = Path(__file__).resolve().parents[2]
RETRY_MAX_TOKENS = 48000


def seed_int(*parts):
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def paper_text(paper):
    return (ROOT / "data/papers" / paper / "text.md").read_text()


def sections(text):
    """Split into top-level sections: the part before the first '## ' (title, abstract) is s0."""
    out, cur = [], {"id": "s0", "title": "Title and abstract", "text": ""}
    for line in text.splitlines(keepends=True):
        if line.startswith("## "):
            out.append(cur)
            cur = {"id": f"s{len(out)}", "title": line[3:].strip(), "text": ""}
        cur["text"] += line
    out.append(cur)
    return out


def paragraphs(text):
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def scramble(text, seed):
    ps = paragraphs(text)
    random.Random(seed).shuffle(ps)
    return "\n\n".join(ps)


def retrieve(text, query, k=3, units=None):
    """Top k paragraphs (or other units, a list of strings) by idf-weighted overlap with the question words."""
    tok = lambda s: set(re.findall(r"[a-z0-9]{3,}", s.lower()))
    ps = units if units is not None else paragraphs(text)
    docs = [tok(p) for p in ps]
    df = defaultdict(int)
    for d in docs:
        for w in d:
            df[w] += 1
    q = tok(query)
    score = lambda d: sum(1 / df[w] for w in q & d)
    best = sorted(range(len(ps)), key=lambda i: -score(docs[i]))[:k]
    return "\n\n".join(ps[i] for i in sorted(best))


def extract_json(s):
    """Parse the outermost JSON object in a reply, tolerating code fences."""
    start, end = s.find("{"), s.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object")
    return json.loads(s[start:end + 1])


def parse_answers(content, n):
    """Map question number to answer text. Questions with no answer are absent."""
    marks = list(re.finditer(r"^[\s*#>_-]*ANSWER\s+Q(\d+)\s*[:.\-]*[*_]*[ \t]*", content, re.M | re.I))
    out = {}
    for m, nxt in zip(marks, marks[1:] + [None]):
        q = int(m.group(1))
        if 1 <= q <= n and q not in out:
            out[q] = content[m.end():nxt.start() if nxt else len(content)].strip().strip("*_ ").strip()
    return out


def validate_quiz(quiz, section_ids):
    problems = []
    claim_ids = {c["id"] for c in quiz["claims"]}
    if not 5 <= len(claim_ids) <= 8:
        problems.append(f"{len(claim_ids)} claims")
    per_claim = defaultdict(int)
    for it in quiz["items"]:
        if it.get("kind") == "claim":
            per_claim[it["claim"]] += 1
        if it.get("claim") not in claim_ids | {None}:
            problems.append(f"{it['id']} unknown claim")
        if not set(it["sections"]) <= section_ids:
            problems.append(f"{it['id']} unknown section")
        if not 2 <= len(it["rubric"]) <= 3:
            problems.append(f"{it['id']} rubric size {len(it['rubric'])}")
    problems += [f"{c} has {per_claim[c]} items" for c in claim_ids if per_claim[c] < 3]
    return problems


def cmd_write(a, led):
    text = paper_text(a.paper)
    secs = sections(text)
    listing = "\n".join(f"{s['id']}: {s['title']}" for s in secs)
    msg = [{"role": "system", "content": prompts.GUARD},
           {"role": "user", "content": prompts.WRITE.format(sections=listing, paper=text)}]
    rec = led.call(a.writer, msg, tag=f"write:{a.paper}", sample=a.seed, max_tokens=32000)
    quiz = extract_json(rec["content"])
    problems = validate_quiz(quiz, {s["id"] for s in secs})
    quiz.update(paper=a.paper, writer=a.writer, seed=a.seed, prompt_version=prompts.PROMPT_VERSION,
                paper_sha256=hashlib.sha256(text.encode()).hexdigest(), problems=problems)
    path = a.dir / f"quiz-{a.writer.split('/')[1]}-{a.seed}.json"
    path.write_text(json.dumps(quiz, indent=1))
    print(path, len(quiz["claims"]), "claims", len(quiz["items"]), "items; problems:", problems, f"cost ${rec['cost']:.3f}")


def writer_rules():
    w = prompts.WRITE
    return w[w.index("Each question must meet"):w.index("Return only JSON")].strip()


def screen_items(led, quiz, items, screener, grader, samples, rnd):
    """Mean no-paper score per item for a screener that is neither the writer nor a held-out smart taker."""
    sub = dict(quiz, items=items)
    byid = {it["id"]: it for it in items}
    res = defaultdict(list)
    with ThreadPoolExecutor(16) as ex:
        answers = [r for rs in ex.map(lambda s: take_one(led, sub, "", screener, "none", 100 * rnd + s), range(samples))
                   for r in rs]
        def safe_grade(r):
            try:
                return grade_one(led, byid[r["item"]], grader, r["answer"])["score"]
            except (RuntimeError, ValueError, KeyError) as e:
                print("grade FAILED", r["item"], str(e)[:150], flush=True)
                return None
        for r, g in zip(answers, ex.map(safe_grade, answers)):
            if g is not None:
                res[r["item"]].append({"sample": r["sample"], "answer": r["answer"], "score": g})
    return res


def cmd_screen(a, led):
    """Write-time leakage screen: items the screener answers without the paper are sent back to the writer."""
    src = Path(a.quiz)
    quiz = json.loads(src.read_text())
    if quiz["writer"] in (a.screener,) or a.screener in a.heldout.split(","):
        raise SystemExit("screener must differ from the writer and the held-out smart takers")
    text = paper_text(quiz["paper"])
    secs = sections(text)
    listing = "\n".join(f"{s['id']}: {s['title']}" for s in secs)
    items = {it["id"]: it for it in quiz["items"]}
    history = defaultdict(list)
    todo = list(items)
    for rnd in range(a.rounds + 1):
        res = screen_items(led, quiz, [items[i] for i in todo], a.screener, a.grader, a.samples, rnd)
        leaked = []
        for i in todo:
            m = statistics.mean([x["score"] for x in res[i]] or [0.0])
            history[i].append({"round": rnd, "question": items[i]["question"], "screen_mean": m, "answers": res[i]})
            if m > a.threshold:
                leaked.append(i)
        print(f"round {rnd}: {len(leaked)} of {len(todo)} items leaked", flush=True)
        if not leaked or rnd == a.rounds:
            break
        block = "\n\n".join(json.dumps(items[i]) + "\nNo-paper answer: " + (res[i][0]["answer"] or "")
                             for i in leaked)
        msg = [{"role": "system", "content": prompts.GUARD},
               {"role": "user", "content": prompts.REVISE.format(rules=writer_rules(), items=block,
                                                                sections=listing, paper=text)}]
        rec = led.call(quiz["writer"], msg, tag=f"revise:{quiz['paper']}:{rnd}", sample=quiz["seed"], max_tokens=32000)
        for it in extract_json(rec["content"])["items"]:
            if it.get("id") in leaked:
                items[it["id"]] = it
        todo = leaked
    for i, it in items.items():
        it["screen"] = {"screener": a.screener, "grader": a.grader, "threshold": a.threshold,
                        "leaked": history[i][-1]["screen_mean"] > a.threshold, "history": history[i]}
    quiz["items"] = [items[it["id"]] for it in quiz["items"]]
    quiz["problems"] = validate_quiz(quiz, {s["id"] for s in secs})
    quiz["screened_from"] = src.name
    out = src.with_name(src.stem + "-screened.json")
    out.write_text(json.dumps(quiz, indent=1))
    n = sum(it["screen"]["leaked"] for it in quiz["items"])
    print(out, f"{n} of {len(items)} items still leak; problems: {quiz['problems']}; run dir spend ${led.spent():.3f}")


def take_one(led, quiz, text, reader, cond, sample):
    """Run one sitting. Returns answer records."""
    items = quiz["items"]
    rng = random.Random(seed_int(reader, cond, sample, quiz["seed"]))
    temp = 0.7
    if cond in ("retrieved", "paper_single"):
        sittings = [[it] for it in items]
    else:
        order = items[:]
        rng.shuffle(order)
        sittings = [order]
    out = []
    for group in sittings:
        qs = "\n".join(f"Q{i}. {it['question']}" for i, it in enumerate(group, 1))
        if cond == "none":
            prompt = prompts.TAKE_NONE.format(questions=qs)
        elif cond in ("paper", "paper_single", "obf", "rem"):
            prompt = prompts.TAKE_PAPER.format(paper=text, questions=qs)
        elif cond == "scrambled":
            prompt = prompts.TAKE_SCRAMBLED.format(paper=scramble(text, seed_int(quiz["paper"], sample)), questions=qs)
        elif cond == "retrieved":
            prompt = prompts.TAKE_RETRIEVED.format(paper=retrieve(text, group[0]["question"]), questions=qs)
        else:
            raise SystemExit(f"unknown condition {cond}")
        msg = [{"role": "system", "content": prompts.GUARD}, {"role": "user", "content": prompt}]
        try:
            rec = led.call(reader, msg, tag=f"take:{quiz['paper']}:{cond}", sample=sample, temperature=temp,
                           seed=seed_int(sample))
            got = parse_answers(rec["content"], len(group))
        except llm.LengthError:
            rec, got = {"finish": "length"}, {}
        if rec["finish"] == "length" and len(got) < len(group):
            # Hidden reasoning used up the token limit. A cut-off sitting would score as wrong answers.
            rec = led.call(reader, msg, tag=f"take:{quiz['paper']}:{cond}", sample=sample, temperature=temp,
                           seed=seed_int(sample), max_tokens=RETRY_MAX_TOKENS)
            got = parse_answers(rec["content"], len(group))
        if cond == "retrieved" or cond == "paper_single":
            got = got or {1: rec["content"].strip()}
        for i, it in enumerate(group, 1):
            out.append({"quiz": quiz["seed"], "writer": quiz["writer"], "paper": quiz["paper"], "item": it["id"],
                        "reader": reader, "cond": cond, "sample": sample, "position": i,
                        "answer": got.get(i), "finish": rec["finish"], "call": rec["key"]})
    return out


def eligible_sections(quiz, secs):
    """Sections with at least one tagged claim, never the title/abstract (s0) or the introduction."""
    tagged = {sid for c in quiz["claims"] for sid in c["sections"]}
    return [s for s in secs if s["id"] in tagged and s["id"] != "s0" and "introduction" not in s["title"].lower()]


def cmd_obfuscate(a, led):
    """Write versions/obf.md (one section rewritten to be hard to follow) and versions/rem.md (same section omitted).
    The section is picked by seed_int(paper, "obf") among eligible sections; the quiz writer never sees this."""
    quiz = json.loads(Path(a.quiz).read_text())
    if a.rewriter in a.takers.split(",") or a.rewriter == quiz["writer"]:
        raise SystemExit("rewriter must not be the writer or a taker")
    text = paper_text(quiz["paper"])
    secs = sections(text)
    elig = eligible_sections(quiz, secs)
    # Seeded pick; if the quiz has no item tagged to it, take the next eligible section in order that has one
    # (deviation logged 2026-11-03 08:36 UTC in docs/decisions.md: a section no item tests makes the obfuscation vacuous).
    start = seed_int(quiz["paper"], "obf") % len(elig)
    order = elig[start:] + elig[:start]
    tested = [s for s in order if any(s["id"] in it["sections"] for it in quiz["items"])]
    if not tested:
        raise SystemExit("no eligible section has a quiz item tagged to it")
    pick = tested[0]
    heading, body = pick["text"].split("\n", 1)
    msg = [{"role": "system", "content": prompts.GUARD},
           {"role": "user", "content": prompts.OBFUSCATE.format(section=body.strip())}]
    rec = led.call(a.rewriter, msg, tag=f"obfuscate:{quiz['paper']}", max_tokens=32000)
    new = re.sub(r"^```[a-z]*\n|\n```\s*$", "", rec["content"].strip()).strip()
    if new.startswith("<section>"):
        new = new.removeprefix("<section>").removesuffix("</section>").strip()
    out = a.dir / "versions"
    out.mkdir(exist_ok=True)
    join = lambda repl: "".join(repl if s["id"] == pick["id"] else s["text"] for s in secs)
    (out / "obf.md").write_text(join(f"{heading}\n\n{new}\n\n"))
    (out / "rem.md").write_text(join(f"{heading}\n\n[This section has been omitted.]\n\n"))
    meta = {"paper": quiz["paper"], "quiz": Path(a.quiz).name, "section": pick["id"], "title": pick["title"],
            "eligible": [s["id"] for s in elig], "rewriter": a.rewriter, "prompt_version": prompts.PROMPT_VERSION,
            "seeded_pick": order[0]["id"], "call": rec["key"], "finish": rec["finish"], "chars_before": len(body.strip()), "chars_after": len(new),
            "items_tagged": [it["id"] for it in quiz["items"] if pick["id"] in it["sections"]]}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if k != "items_tagged"}), len(meta["items_tagged"]), "items tagged",
          f"cost ${rec['cost']:.3f}")


def cmd_probe(a, led):
    """Familiarity probe: given only the title, what does each model say about the paper? Judge by hand."""
    title = json.loads((ROOT / "data/papers" / a.paper / "meta.json").read_text())["title"]
    out = [f"# Familiarity probe: {a.paper}", "", f"Title given: {title}", ""]
    for m in a.models.split(","):
        rec = led.call(m, [{"role": "user", "content": prompts.PROBE.format(title=title)}], tag=f"probe:{a.paper}",
                       temperature=0.7, max_tokens=16000)
        out += [f"## {m}", "", rec["content"].strip(), ""]
    path = a.dir / f"probe-{a.paper}.md"
    path.write_text("\n".join(out) + "\n")
    print(path, f"run dir spend ${led.spent():.3f}")


def cmd_take(a, led):
    quiz = json.loads(Path(a.quiz).read_text())
    clear = paper_text(quiz["paper"])
    texts = {c: (a.dir / "versions" / f"{c}.md").read_text() for c in ("obf", "rem") if c in a.conds.split(",")}
    jobs = [(r, c, s) for r in a.readers.split(",") for c in a.conds.split(",") for s in range(a.samples)]
    path = a.dir / "answers.jsonl"
    seen = set()
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            seen.add((r["writer"], r["quiz"], r["reader"], r["cond"], r["sample"]))
    jobs = [j for j in jobs if (quiz["writer"], quiz["seed"], *j) not in seen]
    failed = 0
    with ThreadPoolExecutor(a.workers) as ex, path.open("a") as f:
        futs = {ex.submit(take_one, led, quiz, texts.get(j[1], clear), *j): j for j in jobs}
        for fu in as_completed(futs):
            try:
                for r in fu.result():
                    f.write(json.dumps(r) + "\n")
                f.flush()
            except RuntimeError as e:
                failed += 1
                print("FAILED", futs[fu], str(e)[:300], flush=True)
    print(f"{len(jobs)} sittings, {failed} failed. Spend in this run dir: ${led.spent():.3f}")


# Grading with minimal hidden reasoning (Jason, 2026-11-03). In the pilot, hidden reasoning was 98% of grader
# output tokens. Agreement with full reasoning is checked in data/runs/grader-check/ (scripts/quiz/grader_check.py).
GRADE_REASONING_EFFORT = "minimal"


def grade_one(led, item, grader, answer, reasoning_effort=GRADE_REASONING_EFFORT):
    if not answer:
        return {"points": [False] * len(item["rubric"]), "note": "no answer", "score": 0.0, "missing": True}
    rubric = "\n".join(f"{i}. {p}" for i, p in enumerate(item["rubric"], 1))
    prompt = prompts.GRADE.format(question=item["question"], answer=item["answer"], rubric=rubric, student=answer)
    msg = [{"role": "user", "content": prompt}]
    try:
        rec = led.call(grader, msg, tag="grade", max_tokens=8000, reasoning_effort=reasoning_effort)
    except llm.LengthError:
        rec = led.call(grader, msg, tag="grade", max_tokens=24000, reasoning_effort=reasoning_effort)
    try:
        res = extract_json(rec["content"])
    except ValueError:
        # Pass 18: GLM-5.3 put unescaped quotes inside its note ("... "cached evaluation" ..."), so the JSON
        # failed to parse on every rerun of a cached call. The points list itself is unambiguous: read it alone.
        m = re.search(r'"points"\s*:\s*\[((?:\s*(?:true|false)\s*,?)+)\]', rec["content"])
        if not m:
            raise
        res = {"points": [t == "true" for t in re.findall(r"true|false", m.group(1))], "note": "note unparsable"}
    pts = [bool(p) for p in res["points"]]
    if len(pts) != len(item["rubric"]):
        raise RuntimeError(f"grader returned {len(pts)} points for {len(item['rubric'])}")
    return {"points": pts, "note": res.get("note"), "score": sum(pts) / len(pts)}


def load_quizzes(d):
    out = {}
    for p in sorted(d.glob("quiz-*.json")):
        if p.with_name(p.stem + "-screened.json").exists():
            continue  # the screened version replaces it
        q = json.loads(p.read_text())
        out[(q["writer"], q["seed"])] = {it["id"]: it for it in q["items"]} | {"_quiz": q}
    return out


def cmd_grade(a, led):
    quizzes = load_quizzes(a.dir)
    answers = [json.loads(l) for l in (a.dir / "answers.jsonl").read_text().splitlines()]
    path = a.dir / "grades.jsonl"
    seen = set()
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            seen.add(r["id"])
    jobs = []
    for r in answers:
        for g in a.graders.split(","):
            rid = hashlib.sha256(json.dumps([r["writer"], r["quiz"], r["item"], r["reader"], r["cond"], r["sample"], g]).encode()).hexdigest()[:16]
            if rid not in seen:
                jobs.append((rid, r, g))
    failed = 0
    with ThreadPoolExecutor(a.workers) as ex, path.open("a") as f:
        def work(j):
            rid, r, g = j
            return rid, r, g, grade_one(led, quizzes[(r["writer"], r["quiz"])][r["item"]], g, r["answer"])
        futs = [ex.submit(work, j) for j in jobs]
        for fu in as_completed(futs):
            try:
                rid, r, g, res = fu.result()
            except (RuntimeError, ValueError, KeyError) as e:
                failed += 1
                print("FAILED", str(e)[:200], flush=True)
                continue
            f.write(json.dumps({"id": rid, "writer": r["writer"], "quiz": r["quiz"], "item": r["item"], "reader": r["reader"],
                                "cond": r["cond"], "sample": r["sample"], "grader": g, **res}) + "\n")
            f.flush()
    print(f"{len(jobs)} grades, {failed} failed. Spend in this run dir: ${led.spent():.3f}")


def cmd_summary(a, led):
    grades = [json.loads(l) for l in (a.dir / "grades.jsonl").read_text().splitlines()]
    quizzes = load_quizzes(a.dir)
    graders = sorted({g["grader"] for g in grades})
    conds = [c for c in ("none", "paper", "obf", "rem", "paper_single", "scrambled", "retrieved") if any(g["cond"] == c for g in grades)]
    primary = a.primary or graders[0]
    L = [f"# Summary, grader {primary}", ""]
    L += ["## Mean score by reader and condition (mean over items and samples; share missing in brackets)", "",
          "| Reader | " + " | ".join(conds) + " |", "|---|" + "---|" * len(conds)]
    cell = defaultdict(list)
    for g in grades:
        if g["grader"] == primary:
            cell[(g["reader"], g["cond"])].append(g)
    for r in sorted({k[0] for k in cell}):
        row = []
        for c in conds:
            v = cell.get((r, c))
            row.append(f"{statistics.mean(x['score'] for x in v):.2f} ({sum(1 for x in v if x.get('missing')) / len(v):.0%})" if v else "")
        L.append(f"| {r} | " + " | ".join(row) + " |")
    if len(graders) > 1:
        pairs = defaultdict(dict)
        for g in grades:
            pairs[(g["writer"], g["quiz"], g["item"], g["reader"], g["cond"], g["sample"])][g["grader"]] = g["score"]
        a0, b0 = graders[0], graders[1]
        both = [(p[a0], p[b0]) for p in pairs.values() if a0 in p and b0 in p]
        diff = statistics.mean(abs(x - y) for x, y in both)
        L += ["", f"Grader agreement ({a0} vs {b0}, {len(both)} answers): mean absolute difference {diff:.3f}, "
              f"means {statistics.mean(x for x, _ in both):.3f} vs {statistics.mean(y for _, y in both):.3f}, "
              f"exact {sum(x == y for x, y in both) / len(both):.2f}."]
    L += ["", "## Per item (mean score, all readers in the row group, primary grader)", ""]
    for (w, s), q in quizzes.items():
        L += [f"### Writer {w}, seed {s}", ""]
        rs = sorted({k[0] for k in cell})
        L += ["| Item | kind | form | " + " | ".join(f"{r.split('/')[1][:12]}:{c[:3]}" for r in rs for c in ("none", "paper")) + " |",
              "|---|---|---|" + "---|" * (2 * len(rs))]
        byitem = defaultdict(list)
        for g in grades:
            if g["grader"] == primary and g["writer"] == w and g["quiz"] == s:
                byitem[(g["item"], g["reader"], g["cond"])].append(g["score"])
        for it in q["_quiz"]["items"]:
            vals = []
            for r in rs:
                for c in ("none", "paper"):
                    v = byitem.get((it["id"], r, c))
                    vals.append(f"{statistics.mean(v):.2f}" if v else "")
            L.append(f"| {it['id']} | {it.get('kind','')[:3]} | {it.get('form','')[:10]} | " + " | ".join(vals) + " |")
    L += four_condition_filter(grades, quizzes, primary)
    (a.dir / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


SMART = ("openai/gpt-5.6-sol", "z-ai/glm-5.3", "moonshotai/kimi-k3", "anthropic/claude-opus-5.5")
# Development thresholds for the four-condition filter. Not preregistered bars.
SMART_NONE_MAX, SMART_PAPER_MIN, DUMB_NONE_MAX = 0.5, 0.5, 0.5


def bootstrap_ci(xs, n=10000, seed=20261102):
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choices(xs, k=len(xs))) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def four_condition_filter(grades, quizzes, primary):
    """Drop items the smart taker answers without the paper or fails with it, or that the ladder answers
    without the paper. Never uses the dumb readers' with-paper scores. The writer never counts as a smart taker."""
    L = ["", "## Four-condition filter (development thresholds, not preregistered)", ""]
    for (w, s), q in quizzes.items():
        sc = defaultdict(list)
        for g in grades:
            if g["grader"] == primary and g["writer"] == w and g["quiz"] == s:
                sc[(g["item"], g["reader"], g["cond"])].append(g["score"])
        smart = [r for r in SMART if r != w and any(k[1] == r for k in sc)]
        dumb = sorted({k[1] for k in sc} - set(SMART))
        m = lambda it, rs, c: statistics.mean([x for r in rs for x in sc.get((it, r, c), [])] or [float("nan")])
        kept, rows = [], []
        for it in q["_quiz"]["items"]:
            i = it["id"]
            sn, sp, dn = m(i, smart, "none"), m(i, smart, "paper"), m(i, dumb, "none")
            why = [n for n, bad in (("smart-none", sn > SMART_NONE_MAX), ("smart-paper", sp < SMART_PAPER_MIN),
                                    ("dumb-none", dn > DUMB_NONE_MAX)) if bad]
            if not why:
                kept.append(i)
            rows.append(f"| {i} | {sn:.2f} | {sp:.2f} | {dn:.2f} | {', '.join(why) or 'kept'} |")
        L += [f"Smart takers: {', '.join(smart)}. Kept {len(kept)} of {len(rows)} items.", "",
              "| Item | smart none | smart paper | ladder none | outcome |", "|---|---|---|---|---|"] + rows
        L += ["", "Score on kept items (mean over items of the per-item mean; 95% bootstrap CI over items):", "",
              "| Reader | none | paper | gain |", "|---|---|---|---|"]
        for r in sorted({k[1] for k in sc}):
            cells = []
            per = {c: [statistics.mean(sc[(i, r, c)]) for i in kept if sc.get((i, r, c))] for c in ("none", "paper")}
            for c in ("none", "paper"):
                xs = per[c]
                cells.append(f"{statistics.mean(xs):.2f} [{bootstrap_ci(xs)[0]:.2f}, {bootstrap_ci(xs)[1]:.2f}]" if xs else "")
            gains = [statistics.mean(sc[(i, r, "paper")]) - statistics.mean(sc[(i, r, "none")])
                     for i in kept if sc.get((i, r, "paper")) and sc.get((i, r, "none"))]
            cells.append(f"{statistics.mean(gains):.2f}" if gains else "")
            L.append(f"| {r} | " + " | ".join(cells) + " |")
    return L


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--cap", type=float, default=5.0, help="max new spend, USD, for this command")
    sub = p.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write"); w.add_argument("--paper", required=True); w.add_argument("--writer", required=True); w.add_argument("--seed", type=int, default=1)
    t = sub.add_parser("take"); t.add_argument("--quiz", required=True); t.add_argument("--readers", required=True); t.add_argument("--conds", required=True); t.add_argument("--samples", type=int, default=3); t.add_argument("--workers", type=int, default=8)
    g = sub.add_parser("grade"); g.add_argument("--graders", required=True); g.add_argument("--workers", type=int, default=16)
    sc = sub.add_parser("screen"); sc.add_argument("--quiz", required=True); sc.add_argument("--screener", required=True); sc.add_argument("--grader", required=True); sc.add_argument("--heldout", required=True, help="smart takers used later; the screener must not be one"); sc.add_argument("--samples", type=int, default=2); sc.add_argument("--threshold", type=float, default=0.5); sc.add_argument("--rounds", type=int, default=2)
    ob = sub.add_parser("obfuscate"); ob.add_argument("--quiz", required=True); ob.add_argument("--rewriter", required=True); ob.add_argument("--takers", required=True)
    pr = sub.add_parser("probe"); pr.add_argument("--paper", required=True); pr.add_argument("--models", required=True)
    s = sub.add_parser("summary"); s.add_argument("--primary")
    a = p.parse_args()
    a.dir = ROOT / "data/runs" / a.run
    led = llm.Ledger(a.dir / "calls.jsonl", a.cap)
    {"write": cmd_write, "screen": cmd_screen, "obfuscate": cmd_obfuscate, "probe": cmd_probe, "take": cmd_take, "grade": cmd_grade, "summary": cmd_summary}[a.cmd](a, led)


if __name__ == "__main__":
    main()
