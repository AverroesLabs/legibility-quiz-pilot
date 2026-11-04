#!/usr/bin/env python3.12
"""Hidden-gap follow-up (Jason, 2026-11-04; docs/plan-v2.md "Follow-up: fluent versions with a hidden gap").
Predictions: docs/preregistration-v2.md, section "Follow-up: fluent versions with a hidden gap".

Usage: python3.12 scripts/quiz/gap_v2.py run [PAPER ...] [--stages versions,takes,judge] [--cap 12] [--stop-usage 639]
       python3.12 scripts/quiz/gap_v2.py analyze

run (all core papers if none given, in one process so per-provider limits hold), per paper in data/runs/v2-<paper>/:
  versions: Claude Opus 5.5 identifies the crucial insight (prompts.GAP_INSIGHT), then writes two fluent versions
    of the whole paper in parallel: versions/rem.md (insight and dependent results removed, prompts.GAP_REMOVE) and
    versions/imp.md (insight, definitions and connecting reasoning left implicit, prompts.GAP_IMPLICIT). Up to 3
    tries each; a try passes if it keeps the section structure, is LO..HI times the original's length and, for imp
    only, loses no number. Redos say what failed. The passing try is used, else the version is marked failed and
    left out (preregistered). Record: versions/gap-meta.json.
  takes: takes_v2.take on the kept quiz with conditions rem and imp only (same sittings as plan v2), written to
    takes-gap.jsonl. Original and no-paper rows stay in takes-v2.jsonl.
  judge: both judge prompts (judge_v2.VARIANTS), 4 readers x 3 samples, on rem and imp; judge-gap.jsonl and
    judge-human-gap.jsonl. Original ratings come from the existing judge files.
Resumable: repeated calls come from calls.jsonl.

analyze: per paper and pooled (bootstrap over questions within papers), the preregistered quantities. Writes
data/analysis/v2-gap.json and v2-gap-tables.md.
"""
import argparse, json, sys, time, traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import filter_v2, judge_v2, llm, prompts, quiz, takes_v2, versions

WRITER = "anthropic/claude-opus-5.5"
GAP = ("rem", "imp")
LO, HI = 0.8, 1.25
CORE = judge_v2.CORE
READERS = takes_v2.READERS
SMART, WEAKER = filter_v2.SMART, filter_v2.WEAKER
OUT = quiz.ROOT / "data/analysis"
N_BOOT = 10000


def call(led, prompt, tag, sample):
    msg = [{"role": "system", "content": prompts.GUARD}, {"role": "user", "content": prompt}]
    return led.call(WRITER, msg, tag=tag, sample=sample, max_tokens=64000, timeout=1800)


def heads(text):
    return [s["text"].split("\n", 1)[0].strip() for s in quiz.sections(text)]


def write_version(led, paper, kind, text, note):
    n = len(text)
    lo, hi = round(LO * n), round(HI * n)
    common = prompts.GAP_COMMON.format(chars=n, lo=lo, hi=hi)
    tmpl = prompts.GAP_REMOVE if kind == "rem" else prompts.GAP_IMPLICIT
    prompt = tmpl.format(common=common, note=note, paper=text)
    info, extra, texts = {"tries": []}, "", []
    for attempt in range(3):
        rec = call(led, prompt + extra, f"gap_{kind}:{paper}", attempt)
        new = versions.clean(rec["content"])
        texts.append(new)
        miss = versions.missing_numbers(text, new)
        same = heads(new) == heads(text)
        ok_len = lo <= len(new) <= hi
        ok = same and ok_len and (kind == "rem" or not miss) and rec["finish"] != "length"
        info["tries"].append({"model": WRITER, "call": rec["key"], "finish": rec["finish"], "cost": rec["cost"],
                              "chars": len(new), "ratio": round(len(new) / n, 3), "same_sections": same,
                              "missing": miss, "pass": ok})
        if ok:
            info["used_try"] = attempt
            return new, info
        extra = ""
        if kind == "imp" and miss:
            extra += prompts.REDO_NUMBERS.format(missing=", ".join(miss))
        if not same:
            extra += prompts.REDO_SECTIONS
        if not ok_len:
            extra += prompts.REDO_GAP_LENGTH.format(chars=len(new), lo=lo, hi=hi)
    info["failed"] = True
    return None, info


def make_versions(led, paper, d):
    meta_path = d / "versions" / "gap-meta.json"
    text = (d / "versions" / "orig.md").read_text()
    rec = call(led, prompts.GAP_INSIGHT.format(paper=text), f"gap_insight:{paper}", 0)
    note = quiz.extract_json(rec["content"])
    meta = {"paper": paper, "writer": WRITER, "prompt_version": prompts.PROMPT_VERSION, "insight": note,
            "insight_call": rec["key"], "chars": {"orig": len(text)}, "versions": {}}
    note_txt = json.dumps(note, indent=1)
    with ThreadPoolExecutor(2) as ex:
        res = dict(zip(GAP, ex.map(lambda k: write_version(led, paper, k, text, note_txt), GAP)))
    for k, (new, info) in res.items():
        meta["versions"][k] = info
        if new is not None:
            (d / "versions" / f"{k}.md").write_text(new)
            meta["chars"][k] = len(new)
            meta["versions"][k]["missing_vs_original"] = versions.missing_numbers(text, new)
    meta_path.write_text(json.dumps(meta, indent=1))
    print(paper, {k: (v.get("used_try", "FAILED"), [(t["ratio"], len(t["missing"]), t["same_sections"])
                                                  for t in v["tries"]]) for k, v in meta["versions"].items()},
          flush=True)
    return meta


def ok_versions(d):
    meta = json.loads((d / "versions" / "gap-meta.json").read_text())
    return tuple(k for k in GAP if not meta["versions"][k].get("failed"))


def judge(led, paper, d):
    conds = ok_versions(d)
    texts = {c: (d / "versions" / f"{c}.md").read_text() for c in conds}
    for name, file in (("self", "judge-gap.jsonl"), ("human", "judge-human-gap.jsonl")):
        v = judge_v2.VARIANTS[name]
        jobs = [(r, c, s) for r in READERS for c in conds for s in range(judge_v2.SAMPLES)]
        with ThreadPoolExecutor(16) as ex:
            rows = list(ex.map(lambda j: judge_v2.rate_one(led, paper, texts[j[1]], *j, v=v), jobs))
        (d / file).write_text("".join(json.dumps(x) + "\n" for x in rows))
        print(f"{paper} judge {name}: {len(rows)} ratings, {sum('error' in x for x in rows)} failed", flush=True)


def run_paper(paper, stages, cap, stop_usage):
    d = quiz.ROOT / f"data/runs/v2-{paper}"
    led = llm.Ledger(d / "calls.jsonl", cap)
    status = {"paper": paper, "stages": {}}
    try:
        for st in stages:
            usage = llm.account_usage()
            if usage >= stop_usage:
                raise RuntimeError(f"account usage ${usage:.2f} at or above stop line ${stop_usage}")
            t0 = time.time()
            if st == "versions":
                make_versions(led, paper, d)
            elif st == "takes":
                conds = ok_versions(d)
                if conds:
                    takes_v2.take(led, d, conds=conds, out="takes-gap.jsonl")
            elif st == "judge":
                judge(led, paper, d)
            status["stages"][st] = round(time.time() - t0)
        status["status"] = "done"
    except Exception as e:
        status.update(status="failed", error=f"{type(e).__name__}: {e}"[:500])
        traceback.print_exc()
    status["new_spend"] = round(led.new_spend, 3)
    (d / "gap-pipeline.json").write_text(json.dumps(status, indent=1))
    print(paper, status, flush=True)
    return status


# ---------- analysis ----------

def per_question(d):
    """{(reader, condition): {item: mean score over samples}} for none, orig and the gap versions."""
    files = ("takes-v2.jsonl",) + (("takes-gap.jsonl",) if (d / "takes-gap.jsonl").exists() else ())
    _, items, _, per_q, _ = takes_v2.question_scores(d, files)
    out = {}
    for (r, c, i), v in per_q.items():
        out.setdefault((r, c), {})[i] = sum(v) / len(v)
    return [it["id"] for it in items], out


def judge_means(d, file, conds):
    got = {}
    p = d / file
    if p.exists():
        for line in p.read_text().splitlines():
            x = json.loads(line)
            if "rating" in x and x["condition"] in conds:
                got.setdefault((x["reader"], x["condition"]), []).append(x["rating"])
    return got


def analyze():
    import numpy as np
    rng = np.random.default_rng(20261103)
    res = {"papers": {}, "pooled": {}, "judge": {}}
    stats = {}  # (reader, stat) -> list of (paper, per-question array)
    jrows = {}  # (prompt, reader, cond) -> list of paper-mean ratings; and all sample ratings
    for paper in CORE:
        d = quiz.ROOT / f"data/runs/v2-{paper}"
        if not (d / "versions" / "gap-meta.json").exists():
            continue
        meta = json.loads((d / "versions" / "gap-meta.json").read_text())
        conds = ok_versions(d)
        ids, q = per_question(d)
        gated = len(ids) >= 30 and np.mean([q[(SMART, "orig")][i] for i in ids]) >= 0.6
        pr = {"kept": len(ids), "gated": bool(gated), "versions_ok": list(conds),
              "insight": meta["insight"].get("insight"), "chars_ratio": {k: round(meta["chars"][k] / meta["chars"]["orig"], 3)
                                                                         for k in conds}, "readers": {}}
        for r in READERS:
            rr = {c: round(float(np.mean([q[(r, c)][i] for i in ids])), 4) for c in ("none", "orig") + conds
                  if (r, c) in q}
            for c in conds:
                if (r, c) not in q:
                    continue
                diff = np.array([q[(r, "orig")][i] - q[(r, c)][i] for i in ids])
                rr[f"drop_{c}"] = round(float(diff.mean()), 4)
                rr[f"drop_{c}_ci95"] = takes_v2.boot(list(diff))
                if gated:  # pooled statistics use gated papers only (preregistered paper gate)
                    stats.setdefault((r, f"drop_{c}"), []).append((paper, diff))
                if r != SMART and (SMART, c) in q:
                    sd = np.array([q[(SMART, "orig")][i] - q[(SMART, c)][i] for i in ids])
                    rr[f"drop_{c}_minus_smart"] = round(float((diff - sd).mean()), 4)
                    rr[f"drop_{c}_minus_smart_ci95"] = takes_v2.boot(list(diff - sd))
                    if gated:
                        stats.setdefault((r, f"drop_{c}_minus_smart"), []).append((paper, diff - sd))
            pr["readers"][r] = rr
        sm = pr["readers"][SMART]
        pr["information_check"] = {c: {"ratio": round(sm[c] / sm["orig"], 4), "pass": sm[c] >= 0.9 * sm["orig"]}
                                   for c in conds if c in sm}
        # judge: original ratings from the plan v2 files, gap ratings from the new ones
        for name, old, new in (("self", "judge-v2.jsonl", "judge-gap.jsonl"),
                               ("human", "judge-human-v2.jsonl", "judge-human-gap.jsonl")):
            jm = judge_means(d, old, ("orig",)) | judge_means(d, new, conds)
            pr.setdefault("judge", {})[name] = {f"{r}|{c}": round(sum(v) / len(v), 3) for (r, c), v in jm.items()}
            for (r, c), v in jm.items():
                jrows.setdefault((name, r, c), []).append((paper, sum(v) / len(v), v))
        res["papers"][paper] = pr
    for (r, s), lst in stats.items():
        codes = np.concatenate([[k] * len(a) for k, (_, a) in enumerate(lst)])
        x = np.concatenate([a for _, a in lst])
        groups = [np.flatnonzero(codes == g) for g in np.unique(codes)]
        bs = np.array([x[np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups])].mean()
                       for _ in range(N_BOOT)])
        per_paper_ci = [res["papers"][p]["readers"][r][f"{s}_ci95"] for p, _ in lst]
        res["pooled"].setdefault(r, {})[s] = {
            "mean": round(float(x.mean()), 4), "ci95": [round(float(v), 4) for v in np.percentile(bs, [2.5, 97.5])],
            "papers": len(lst), "questions": int(len(x)),
            "papers_ci_above_0": sum(ci[0] > 0 for ci in per_paper_ci),
            "papers_mean_above_0": sum(float(a.mean()) > 0 for _, a in lst)}
    for (name, r, c), lst in jrows.items():
        res["judge"].setdefault(name, {}).setdefault(r, {})[c] = {
            "mean": round(float(np.mean([v for _, _, vs in lst for v in vs])), 3), "papers": len(lst),
            "paper_means": {p: round(m, 3) for p, m, _ in lst}}
    for name, rd in res["judge"].items():
        for r, cd in rd.items():
            if "orig" not in cd:
                continue
            for c in GAP:
                if c in cd:
                    common = [p for p in cd[c]["paper_means"] if p in cd["orig"]["paper_means"]]
                    cd[c]["minus_orig"] = round(cd[c]["mean"] - cd["orig"]["mean"], 3)
                    cd[c]["papers_within_1"] = sum(abs(cd[c]["paper_means"][p] - cd["orig"]["paper_means"][p]) <= 1
                                                   for p in common)
    ic = [pr["information_check"].get("imp", {}).get("pass") for pr in res["papers"].values() if pr["gated"]]
    res["imp_information_check_pass"] = f"{sum(bool(x) for x in ic)} of {sum(x is not None for x in ic)} gated papers"
    OUT.mkdir(exist_ok=True)
    (OUT / "v2-gap.json").write_text(json.dumps(res, indent=1, default=lambda o: o.item()))  # numpy scalars
    (OUT / "v2-gap-tables.md").write_text(tables(res))
    print(tables(res))


def short(r):
    return r.split("/")[1]


def tables(res):
    L = ["## Pooled quiz drops on gated papers (orig minus version; bootstrap over questions within papers)", "",
         "| reader | statistic | mean [95% CI] | papers with CI above 0 | papers with mean above 0 |", "|---|---|---|---|---|"]
    for r in READERS:
        for s, v in res["pooled"].get(r, {}).items():
            L.append(f"| {short(r)} | {s} | {v['mean']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] | "
                     f"{v['papers_ci_above_0']} of {v['papers']} | {v['papers_mean_above_0']} of {v['papers']} |")
    L += ["", f"Information check on imp (smart reader at least 0.9 of original): {res['imp_information_check_pass']}.",
          "", "## Judge ratings (mean over papers and samples)", "",
          "| prompt | reader | orig | rem (minus orig; papers within 1) | imp (minus orig; papers within 1) |",
          "|---|---|---|---|---|"]
    for name, rd in res["judge"].items():
        for r in READERS:
            cd = rd.get(r, {})
            if "orig" not in cd:
                continue
            cell = lambda c: (f"{cd[c]['mean']:.2f} ({cd[c]['minus_orig']:+.2f}; {cd[c]['papers_within_1']} of "
                              f"{cd[c]['papers']})" if c in cd else "-")
            L.append(f"| {name} | {short(r)} | {cd['orig']['mean']:.2f} | {cell('rem')} | {cell('imp')} |")
    L += ["", "## Per paper: quiz score orig / rem / imp", "",
          "| paper | kept | gated | length rem, imp | " + " | ".join(short(r) for r in READERS) + " |",
          "|---|---|---|---|" + "---|" * len(READERS)]
    for p, pr in res["papers"].items():
        cells = []
        for r in READERS:
            rr = pr["readers"][r]
            cells.append(" / ".join(f"{rr[c]:.2f}" if c in rr else "-" for c in ("orig",) + GAP))
        L.append(f"| {p} | {pr['kept']} | {'yes' if pr['gated'] else 'no'} | "
                 f"{', '.join(str(pr['chars_ratio'].get(c, '-')) for c in GAP)} | " + " | ".join(cells) + " |")
    return "\n".join(L) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["run", "analyze"])
    p.add_argument("papers", nargs="*")
    p.add_argument("--stages", default="versions,takes,judge")
    p.add_argument("--cap", type=float, default=12.0, help="new spend cap per paper, dollars")
    p.add_argument("--stop-usage", type=float, default=639.0)
    a = p.parse_args()
    if a.cmd == "analyze":
        return analyze()
    papers = a.papers or CORE
    stages = a.stages.split(",")
    print(f"account usage ${llm.account_usage():.2f}; papers {papers}; stages {stages}", flush=True)
    with ThreadPoolExecutor(len(papers)) as ex:
        sts = list(ex.map(lambda pp: run_paper(pp, stages, a.cap, a.stop_usage), papers))
    print(f"done: {sum(s['status'] == 'done' for s in sts)} of {len(sts)}; new spend "
          f"${sum(s['new_spend'] for s in sts):.2f}; account usage ${llm.account_usage():.2f}", flush=True)


if __name__ == "__main__":
    main()
