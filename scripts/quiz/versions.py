#!/usr/bin/env python3.12
"""Plan v2 paper versions: obfuscated, compressed, and compressed + obfuscated, each of the whole paper.

Usage: python3.12 scripts/quiz/versions.py --run v2dev-2607.05615v2 --paper 2607.05615v2 [--cap 10]

Writes data/runs/<run>/versions/{orig,obf,comp,compobf}.md and versions/meta.json. The rewriter is Kimi K3 (not
the quiz writer, a reader or the grader). Compression is one call on the whole paper, so repetition between
sections can be cut (section-by-section compression stalled at 0.54 to 0.70 of the original's length; see
docs/decisions.md, 2026-11-03 pass 12). It is redone, up to 3 tries, if it loses a number, changes the section
structure or exceeds half the original's length; the shortest try that loses no number and keeps the structure
is used. Obfuscation is per section, in parallel, one call each. A paper-wide
glossary of opaque abbreviations is fixed first, so every obfuscated section uses the same ones, and each term
is defined only once, in the first section after the abstract that uses it. Every section is checked for lost
numbers; an obfuscated section that loses any is redone once with the missing numbers listed. The whole-version number
check is recorded in meta.json. Resumable: repeated calls come from calls.jsonl.
"""
import argparse, json, re, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import llm, prompts, quiz

REWRITER = "moonshotai/kimi-k3"
VERSIONS = ("obf", "comp", "compobf")
# Compressor (model, reasoning effort, target percent of the original's length). Chosen in the compressor trial
# (commit 09d3dc3, data/runs/v2dev-2607.05615v2/compress-trial/); see docs/decisions.md.
COMPRESSOR = ("anthropic/claude-opus-5.5", None, 40)
COMP_MAX_RATIO = 0.5  # plan v2: a third to a half of the original's length

# A citation is a bracket holding a capitalised name and a year 19xx/20xx, or "Name (year)". Pass 18: the earlier
# pattern took any parenthesis with 4 digits, so "(min 57.33, max 1060.00)" was blanked, and it missed square-bracket
# citations "[Zyma et al., 2019]" that a compressor rewrites as "(Zyma 2019)".
YEAR = r"\b(?:19|20)\d{2}[a-z]?\b"
NAME = r"[A-ZÀ-ÖØ-Þ][^\W\d_]+"  # capitalised, accented letters allowed (Méloux)
CITATION = re.compile(rf"\([^()]*{NAME}[^()]*{YEAR}[^()]*\)|\[[^\[\]]*{NAME}[^\[\]]*{YEAR}[^\[\]]*\]"
                      rf"|{NAME}(?: et al\.| and {NAME})?\s*[(\[]{YEAR}[)\]]")
LATEX_THOUSANDS = re.compile(r"(?<=\d)\{,\}(?=\d{3})")  # "2{,}000" -> "2,000" (pass 18: read as "2" and "000")
LATEX_CMD = re.compile(r"\\[A-Za-z]+")  # e.g. \geq in "\geq0.6", which would hide the number from NUMBER
LATEX_LEN = re.compile(r"\[-?[\d.]+(?:pt|em|ex|mm|cm)\]")  # layout spacing such as \\[2.0pt], not content
# Unsigned, and allowed after "_": a compressor turns the LaTeX table cell "- $0.02$" into "−0.02" and
# "$\mathit{13.00}$" into "_13.00_" (pass 17), which an earlier signed, \w-bounded pattern counted as lost.
NUMBER = re.compile(r"(?<![A-Za-z0-9.])\d+(?:[.,]\d+)*(?:\.\d+)?%?")


def numbers(text):
    """Numbers stated in a text, without sign, ignoring citation years and LaTeX spacing. Commas in thousands are removed,
    '%' is dropped. LaTeX commands are blanked first, so a number glued to one (\\geq0.6) is still found."""
    text = LATEX_LEN.sub(" ", LATEX_CMD.sub(" ", CITATION.sub(" ", LATEX_THOUSANDS.sub(",", text))))
    out = set()
    for m in NUMBER.finditer(text):
        n = m.group().rstrip("%")
        if re.fullmatch(r"\d{1,3}(,\d{3})+(\.\d+)?", n):
            n = n.replace(",", "")
        out.update(x for x in n.split(",") if x)
    return out


def missing_numbers(orig, new):
    return sorted(numbers(orig) - numbers(new), key=lambda x: (len(x), x))


def split_front(sec):
    """For s0, keep the title and author block verbatim and rewrite only the abstract onward."""
    m = re.search(r"^#+\s*Abstract\s*$", sec["text"], re.M | re.I)
    if sec["id"] != "s0":
        head, body = sec["text"].split("\n", 1)
        return head + "\n\n", body.strip()
    if not m:
        return "", sec["text"].strip()
    return sec["text"][:m.end()] + "\n\n", sec["text"][m.end():].strip()


def clean(content):
    new = re.sub(r"^```[a-z]*\n|\n```\s*$", "", content.strip()).strip()
    for tag in ("section", "text"):
        if new.startswith(f"<{tag}>"):
            new = new.removeprefix(f"<{tag}>").removesuffix(f"</{tag}>").strip()
    return new


def call(led, prompt, tag, sample=0, effort=None, model=REWRITER):
    """effort None keeps Kimi's default reasoning (obfuscation, glossary). Compression uses "medium": at the
    default, 7 of 11 section compression calls on the dev paper spent the whole 32k-token limit on hidden reasoning
    ($0.48 each) and returned no text. If a call hits the limit with no text, retry once at "low"; meta.json
    records which was used."""
    msg = [{"role": "system", "content": prompts.GUARD}, {"role": "user", "content": prompt}]
    try:
        return led.call(model, msg, tag=tag, sample=sample, max_tokens=32000, timeout=900, reasoning_effort=effort)
    except llm.LengthError:
        if effort == "low":
            raise
        return led.call(model, msg, tag=tag, sample=sample, max_tokens=32000, timeout=900, reasoning_effort="low")


def make_glossary(led, paper, text):
    rec = call(led, prompts.GLOSSARY.format(paper=text), f"glossary:{paper}")
    terms = quiz.extract_json(rec["content"])["terms"]
    seen, out = set(), []
    for t in terms:
        if t["term"] and t["abbrev"] and t["abbrev"] not in seen and t["term"].lower() in text.lower():
            seen.add(t["abbrev"])
            out.append({"term": t["term"], "abbrev": t["abbrev"]})
    return out, rec


def define_in(glossary, secs):
    """Each term is defined in the first section after s0 that mentions it (s0 if none does)."""
    where = {}
    for g in glossary:
        hits = [s["id"] for s in secs if s["id"] != "s0" and g["term"].lower() in s["text"].lower()]
        where[g["term"]] = hits[0] if hits else "s0"
    return where


def gloss_lines(glossary, where, sid):
    define = [g for g in glossary if where[g["term"]] == sid]
    use = [g for g in glossary if where[g["term"]] != sid]
    f = lambda gs: "\n".join(f"- {g['term']} -> {g['abbrev']}" for g in gs) or "(none)"
    return f(define), f(use)


def obfuscate_section(led, paper, sec_id, body, glossary, where):
    """Up to 2 tries: redo if numbers are lost. Returns the best try and a record: a try cut off at the token limit
    is used only if every try was, and among the rest the one that lost the fewest numbers (the first on a tie).
    Before pass 19 the last try was used, so a truncated redo replaced a near-complete first try (run-log pass 19)."""
    define, use = gloss_lines(glossary, where, sec_id)
    info, extra, texts = {"tries": []}, "", []
    for attempt in range(2):
        prompt = prompts.OBFUSCATE_V2.format(define=define, use=use, section=body) + extra
        rec = call(led, prompt, f"obf:{paper}:{sec_id}", sample=attempt)
        new = clean(rec["content"])
        miss = missing_numbers(body, new)
        texts.append(new)
        info["tries"].append({"model": REWRITER, "call": rec["key"], "finish": rec["finish"], "cost": rec["cost"],
                              "reasoning_effort": rec["params"].get("reasoning_effort"),
                              "chars": len(new), "ratio": round(len(new) / len(body), 3), "missing": miss})
        if not miss and rec["finish"] != "length":
            break
        extra = prompts.REDO_NUMBERS.format(missing=", ".join(miss))
    tries = info["tries"]
    info["used_try"] = min(range(len(tries)), key=lambda i: (tries[i]["finish"] == "length", len(tries[i]["missing"]), i))
    return texts[info["used_try"]], info


def budgets(secs, pct):
    """Per-section character budgets: pct of each section, at least the heading plus 150 characters."""
    return {s["id"]: max(len(s["text"].split("\n", 1)[0]) + 150, round(pct / 100 * len(s["text"]))) for s in secs}


def compress(led, paper, text, compressor=COMPRESSOR, tries=3):
    """Whole-paper compression, with per-section budgets checked by script. Returns the text used and a record."""
    model, effort, pct = compressor
    secs = quiz.sections(text)
    ids, heads = [s["id"] for s in secs], {s["id"]: s["text"].split("\n", 1)[0].strip() for s in secs}
    budget = budgets(secs, pct)
    prompt = prompts.COMPRESS_PAPER.format(pct=pct, target=round(pct / 100 * len(text)), paper=text,
                                           budgets="\n".join(f"- {heads[i]}: {budget[i]}" for i in ids))
    info, extra, best, texts = {"budgets": budget, "tries": []}, "", None, []
    for attempt in range(tries):
        rec = call(led, prompt + extra, f"comp:{paper}", sample=attempt, effort=effort, model=model)
        new = clean(rec["content"])
        texts.append(new)
        miss = missing_numbers(text, new)
        ratio = len(new) / len(text)
        nsecs = quiz.sections(new)
        same = [s["id"] for s in nsecs] == ids
        over = {s["id"]: len(s["text"]) for s in nsecs if same and len(s["text"]) > 1.15 * budget[s["id"]]}
        info["tries"].append({"model": model, "call": rec["key"], "finish": rec["finish"], "cost": rec["cost"],
                              "reasoning_effort": rec["params"].get("reasoning_effort"), "chars": len(new),
                              "ratio": round(ratio, 3), "missing": miss, "same_sections": same,
                              "over_budget": over})
        if not miss and same and (best is None or len(new) < len(best)):
            best, info["used_try"] = new, attempt
        if not miss and same and ratio <= COMP_MAX_RATIO:
            break
        extra = (prompts.REDO_NUMBERS.format(missing=", ".join(miss)) if miss else "") + \
                (prompts.REDO_SECTIONS if not same else "")
        if ratio > COMP_MAX_RATIO:
            extra += prompts.REDO_BUDGET.format(
                pct=round(100 * ratio), target=round(pct / 100 * len(text)),
                over="\n".join(f"- {heads[i]}: {n} / {budget[i]}" for i, n in over.items()) or "(not measured)")
    if best is None:
        best = repair(led, paper, text, ids, info, texts)
    if best is None:
        raise RuntimeError(f"compression of {paper}: no try or repair kept every number and the section structure")
    return best, info


def number_lines(text, miss, width=400):
    """Each missing number with the first original line that contains it, cut to width characters around it."""
    out = []
    for m in miss:
        line = next((ln for ln in text.splitlines() if m in numbers(ln)), "")
        i = line.find(m)
        if len(line) < 120:  # hard-wrapped prose: take the surrounding text instead
            j = text.find(line)
            line, i = text[max(0, j - 150):j + len(line) + 150].replace("\n", " "), 150 + i
        if len(line) > width:
            line = line[max(0, i - width // 2):i + width // 2]
        out.append(f"- {m}: {line.strip()}")
    return "\n".join(out)


def repair(led, paper, text, ids, info, texts, tries=2):
    """Pass 17 fix: on 2 of 3 core papers every whole-paper try lost 1 to 27 numbers, each try different ones,
    because each redo rewrites the paper from scratch. Start from the try with the same sections and fewest
    missing numbers, and ask the compressor to reinsert only the missing ones, at most twice."""
    model = info["tries"][0]["model"]
    cands = [i for i, t in enumerate(info["tries"]) if t["same_sections"]]
    if not cands:
        return None
    start = min(cands, key=lambda i: len(info["tries"][i]["missing"]))
    cur = texts[start]
    info["repairs"] = []
    for attempt in range(tries):
        miss = missing_numbers(text, cur)
        prompt = prompts.REPAIR_NUMBERS.format(missing=number_lines(text, miss), text=cur)
        rec = call(led, prompt, f"comprepair:{paper}", sample=attempt, effort=None, model=model)
        new = clean(rec["content"])
        nmiss = missing_numbers(text, new)
        same = [s["id"] for s in quiz.sections(new)] == ids
        info["repairs"].append({"model": model, "from_try": start, "call": rec["key"],
                                "finish": rec["finish"], "cost": rec["cost"], "chars": len(new),
                                "ratio": round(len(new) / len(text), 3), "missing_before": miss,
                                "missing": nmiss, "same_sections": same})
        if same and len(nmiss) < len(miss):
            cur = new
        if same and not nmiss:
            info["used_try"] = f"repair{attempt}"
            return new
    return None


def obfuscate(led, paper, secs, glossary, where, workers=16):
    """Obfuscate every section in parallel. paper is the call tag prefix."""
    jobs = []
    for s in secs:
        head, body = split_front(s)
        if not body:
            jobs.append((s["id"], head, None))
        else:
            jobs.append((s["id"], head, body))
    def one(j):
        sid, head, body = j
        if body is None:
            return head, {"tries": []}
        new, info = obfuscate_section(led, paper, sid, body, glossary, where)
        return f"{head}{new}\n\n", info
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(one, jobs))
    text = "".join(r[0] for r in res)
    info = {sid: r[1] for (sid, _, _), r in zip(jobs, res)}
    return text, info


def make(led, d, paper, compressor=None):
    """Write d/versions/{orig,obf,comp,compobf}.md and meta.json for one paper, and return meta."""
    out = d / "versions"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    text = quiz.paper_text(paper)
    secs = quiz.sections(text)
    (out / "orig.md").write_text(text)
    if compressor:
        model, _, effort = compressor.partition(":")
        t0c = time.time()
        new, info = compress(led, paper, text, (model, effort or None, COMPRESSOR[2]))
        info["seconds"] = round(time.time() - t0c)
        name = model.split("/")[1]
        (out / f"comp-trial-{name}.md").write_text(new)
        (out / f"comp-trial-{name}.json").write_text(json.dumps(info, indent=1))
        print(model, [(t["ratio"], len(t["missing"]), t["same_sections"]) for t in info["tries"]],
              f"used {info['used_try']}: {len(new) / len(text):.3f}x; {info['seconds']} s")
        return
    glossary, grec = make_glossary(led, paper, text)
    where = define_in(glossary, secs)
    meta = {"paper": paper, "rewriter": REWRITER, "prompt_version": prompts.PROMPT_VERSION,
            "glossary": [g | {"defined_in": where[g["term"]]} for g in glossary], "glossary_call": grec["key"],
            "chars": {"orig": len(text)}, "sections": {}, "missing": {}, "seconds": {}}
    with ThreadPoolExecutor(2) as ex:  # obf and comp at once; compobf needs comp
        fo = ex.submit(obfuscate, led, paper, secs, glossary, where)
        fc = ex.submit(compress, led, paper, text)
        vt = {"obf": fo.result(), "comp": fc.result()}
    comp_secs = quiz.sections(vt["comp"][0])
    vt["compobf"] = obfuscate(led, paper + ":comp", comp_secs, glossary, define_in(glossary, comp_secs))
    for v in VERSIONS:
        vtext, info = vt[v]
        (out / f"{v}.md").write_text(vtext)
        ref = vt["comp"][0] if v == "compobf" else text
        meta["chars"][v] = len(vtext)
        meta["sections"][v] = info
        meta["missing"][v] = {"vs_original": missing_numbers(text, vtext),
                              "vs_source": missing_numbers(ref, vtext)}
    meta["seconds"]["total"] = round(time.time() - t0)
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    for v in VERSIONS:
        print(paper, v, f"{meta['chars'][v] / len(text):.2f}x length;", len(meta["missing"][v]["vs_original"]),
              "numbers missing vs original:", meta["missing"][v]["vs_original"][:20])
    print(f"{paper}: glossary {len(glossary)} terms; {meta['seconds']['total']} s")
    return meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--paper", required=True)
    p.add_argument("--cap", type=float, default=10.0)
    p.add_argument("--compressor", help="trial only: model[:effort], compress only, write versions/comp-trial-<model>.md")
    a = p.parse_args()
    d = quiz.ROOT / "data/runs" / a.run
    make(llm.Ledger(d / "calls.jsonl", a.cap), d, a.paper, a.compressor)


if __name__ == "__main__":
    main()
