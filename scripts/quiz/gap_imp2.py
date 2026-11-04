#!/usr/bin/env python3
"""Follow-up 3 (docs/preregistration-v2.md): a stronger "insight implicit" version, imp2, without a length floor.

    python scripts/quiz/gap_imp2.py run [PAPER ...] [--stop-usage 839]
    python scripts/quiz/gap_imp2.py analyze      # data/analysis/v2-imp2.json

run, per core paper in data/runs/v2-<paper>/ (all papers in one process so per-provider limits hold):
  version: Claude Opus 5.5 writes versions/imp2.md from the paper and the insight note already in
    versions/gap-meta.json (prompts.GAP_IMPLICIT_STRONG). Up to 3 tries; a try passes if it keeps the section
    headings, loses no number and is LO..HI times the original's length. Record: versions/imp2-meta.json.
  defined: GLM-5.3 (minimal reasoning) checks which key terms of the note the original and imp2 define in words.
  takes: the existing kept quiz on imp2, all readers, 3 samples, same sittings (takes-imp2.jsonl).
  judge: human-reader judge prompt on imp2, all readers, 3 samples (judge-human-imp2.jsonl).
Resumable: repeated calls come from calls.jsonl.
"""
import argparse, json, sys, traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import filter_v2, gap_v2, judge_v2, llm, prompts, quiz, takes_v2, versions  # noqa: E402

LO, HI = 0.4, 1.25
CORE = judge_v2.CORE
READERS = takes_v2.READERS
SMART = filter_v2.SMART
GRADER = filter_v2.GRADER


def write_imp2(led, paper, d):
    text = (d / "versions/orig.md").read_text()
    note = json.loads((d / "versions/gap-meta.json").read_text())["insight"]
    n = len(text)
    lo, hi = round(LO * n), round(HI * n)
    prompt = prompts.GAP_IMPLICIT_STRONG.format(common=prompts.GAP_STRONG_COMMON.format(chars=n, lo=lo, hi=hi),
                                                note=json.dumps(note, indent=1), paper=text)
    info, extra = {"writer": gap_v2.WRITER, "prompt_version": prompts.PROMPT_VERSION, "tries": []}, ""
    for attempt in range(3):
        rec = gap_v2.call(led, prompt + extra, f"gap_imp2:{paper}", attempt)
        new = versions.clean(rec["content"])
        miss = versions.missing_numbers(text, new)
        same = gap_v2.heads(new) == gap_v2.heads(text)
        ok_len = lo <= len(new) <= hi
        ok = same and ok_len and not miss and rec["finish"] != "length"
        info["tries"].append({"call": rec["key"], "finish": rec["finish"], "cost": rec["cost"], "chars": len(new),
                              "ratio": round(len(new) / n, 3), "same_sections": same, "missing": miss, "pass": ok})
        if ok:
            info["used_try"] = attempt
            (d / "versions/imp2.md").write_text(new)
            break
        extra = ""
        if miss:
            extra += prompts.REDO_NUMBERS.format(missing=", ".join(miss))
        if not same:
            extra += prompts.REDO_SECTIONS
        if not ok_len:
            extra += prompts.REDO_GAP_LENGTH.format(chars=len(new), lo=lo, hi=hi)
    else:
        info["failed"] = True
    (d / "versions/imp2-meta.json").write_text(json.dumps(info, indent=1))
    print(paper, "imp2", info.get("used_try", "FAILED"), [(t["ratio"], len(t["missing"]), t["same_sections"])
                                                          for t in info["tries"]], flush=True)
    return not info.get("failed")


def defined_check(led, paper, d):
    note = json.loads((d / "versions/gap-meta.json").read_text())["insight"]
    terms = note.get("key_definitions", [])
    out = {"terms": terms}
    for cond in ("orig", "imp2"):
        text = (d / f"versions/{cond}.md").read_text()
        msg = [{"role": "system", "content": prompts.GUARD},
               {"role": "user", "content": prompts.DEFINED_CHECK.format(terms=json.dumps(terms, indent=1), paper=text)}]
        rec = led.call(GRADER, msg, tag=f"defined_{cond}", max_tokens=8000, reasoning_effort=quiz.GRADE_REASONING_EFFORT)
        try:
            out[cond] = quiz.extract_json(rec["content"])["defined"]
        except Exception as e:  # noqa: BLE001  recorded, not fatal
            out[cond] = None
            out[f"{cond}_error"] = str(e)[:200]
    (d / "versions/imp2-defined.json").write_text(json.dumps(out, indent=1))


def judge(led, paper, d):
    text = (d / "versions/imp2.md").read_text()
    v = judge_v2.VARIANTS["human"]
    jobs = [(r, "imp2", s) for r in READERS for s in range(judge_v2.SAMPLES)]
    with ThreadPoolExecutor(12) as ex:
        rows = list(ex.map(lambda j: judge_v2.rate_one(led, paper, text, *j, v=v), jobs))
    (d / "judge-human-imp2.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))


def run_paper(paper, cap, stop):
    d = quiz.ROOT / f"data/runs/v2-{paper}"
    led = llm.Ledger(d / "calls.jsonl", cap)
    status = {"paper": paper}
    try:
        if llm.account_usage() >= stop:
            raise RuntimeError(f"account usage at or above stop line ${stop}")
        if write_imp2(led, paper, d):
            with ThreadPoolExecutor(3) as ex:
                futs = [ex.submit(defined_check, led, paper, d),
                        ex.submit(takes_v2.take, led, d, ("imp2",), "takes-imp2.jsonl"),
                        ex.submit(judge, led, paper, d)]
                for f in futs:
                    f.result()
            status["status"] = "done"
        else:
            status["status"] = "version failed (excluded)"
    except Exception as e:  # noqa: BLE001
        status.update(status="failed", error=f"{type(e).__name__}: {e}"[:400])
        traceback.print_exc()
    status["new_spend"] = round(led.new_spend, 3)
    (d / "imp2-pipeline.json").write_text(json.dumps(status, indent=1))
    print(paper, status, flush=True)
    return status


def analyze():
    rng = np.random.default_rng(20261104)
    res = {"papers": {}, "pooled": {}, "ratings": {}}
    stats, ratings = {}, {}
    for paper in CORE:
        d = quiz.ROOT / f"data/runs/v2-{paper}"
        if not (d / "takes-imp2.jsonl").exists():
            continue
        _, items, _, per_q, _ = takes_v2.question_scores(d, ("takes-v2.jsonl", "takes-imp2.jsonl"))
        ids = [it["id"] for it in items]
        q = {}
        for (r, c, i), v in per_q.items():
            q.setdefault((r, c), {})[i] = sum(v) / len(v)
        gated = len(ids) >= 30 and np.mean([q[(SMART, "orig")][i] for i in ids]) >= 0.6
        meta = json.loads((d / "versions/imp2-meta.json").read_text())
        defined = json.loads((d / "versions/imp2-defined.json").read_text())
        pr = {"kept": len(ids), "gated": bool(gated), "ratio": meta["tries"][meta["used_try"]]["ratio"],
              "defined": {c: defined.get(c) for c in ("orig", "imp2")}, "readers": {}}
        for r in READERS:
            o = np.array([q[(r, "orig")][i] for i in ids])
            m = np.array([q[(r, "imp2")][i] for i in ids])
            s_o = np.array([q[(SMART, "orig")][i] for i in ids])
            s_m = np.array([q[(SMART, "imp2")][i] for i in ids])
            pr["readers"][r] = {"orig": round(float(o.mean()), 4), "imp2": round(float(m.mean()), 4),
                                "drop": round(float((o - m).mean()), 4), "drop_ci95": takes_v2.boot(list(o - m))}
            if gated:
                stats.setdefault((r, "drop"), []).append(o - m)
                if r != SMART:
                    stats.setdefault((r, "drop_minus_smart"), []).append((o - m) - (s_o - s_m))
        sm = pr["readers"][SMART]
        pr["information_check"] = {"ratio": round(sm["imp2"] / sm["orig"], 4), "pass": sm["imp2"] >= 0.9 * sm["orig"]}
        for f, cond in (("judge-human-v2.jsonl", "orig"), ("judge-human-imp2.jsonl", "imp2")):
            for line in (d / f).read_text().splitlines():
                x = json.loads(line)
                if "rating" in x and x["condition"] == cond:
                    ratings.setdefault((x["reader"], cond), {}).setdefault(paper, []).append(x["rating"])
        res["papers"][paper] = pr
    for (r, s), lst in stats.items():
        x = np.concatenate(lst)
        bs = [np.concatenate([a[rng.integers(0, len(a), len(a))] for a in lst]).mean() for _ in range(10000)]
        res["pooled"].setdefault(r, {})[s] = {"mean": round(float(x.mean()), 4),
                                               "ci95": [round(float(v), 4) for v in np.percentile(bs, [2.5, 97.5])],
                                               "papers": len(lst), "papers_mean_above_0": int(sum(a.mean() > 0 for a in lst))}
    for (r, c), by in ratings.items():
        res["ratings"].setdefault(r, {})[c] = round(float(np.mean([np.mean(v) for v in by.values()])), 3)
    for r in res["ratings"]:
        if {"orig", "imp2"} <= set(res["ratings"][r]):
            common = [p for p in ratings[(r, "imp2")] if p in ratings[(r, "orig")]]
            res["ratings"][r]["orig_same_papers"] = round(float(np.mean([np.mean(ratings[(r, "orig")][p]) for p in common])), 3)
            res["ratings"][r]["imp2_minus_orig"] = round(res["ratings"][r]["imp2"] - res["ratings"][r]["orig_same_papers"], 3)
    gp = [p for p in res["papers"].values() if p["gated"]]
    res["information_check_pass"] = f"{sum(p['information_check']['pass'] for p in gp)} of {len(gp)} gated papers"
    fr = []
    for p in res["papers"].values():
        o, m = p["defined"]["orig"], p["defined"]["imp2"]
        if o and m and any(o):
            fr.append(sum(a and not b for a, b in zip(o, m)) / sum(o))
    res["share_of_defined_terms_undefined_in_imp2"] = {"mean": round(float(np.mean(fr)), 3) if fr else None,
                                                       "papers": len(fr)}
    (quiz.ROOT / "data/analysis/v2-imp2.json").write_text(json.dumps(res, indent=1, default=lambda o: o.item()))
    print(json.dumps({k: res[k] for k in ("pooled", "ratings", "information_check_pass",
                                          "share_of_defined_terms_undefined_in_imp2")}, indent=1))
    for p, pr in res["papers"].items():
        print(p, pr["ratio"], pr["gated"], {r.split("/")[1][:7]: (pr["readers"][r]["orig"], pr["readers"][r]["imp2"])
                                            for r in READERS})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["run", "analyze"])
    p.add_argument("papers", nargs="*")
    p.add_argument("--cap", type=float, default=10.0)
    p.add_argument("--stop-usage", type=float, default=839.0)
    a = p.parse_args()
    if a.cmd == "analyze":
        return analyze()
    papers = a.papers or CORE
    with ThreadPoolExecutor(len(papers)) as ex:
        sts = list(ex.map(lambda x: run_paper(x, a.cap, a.stop_usage), papers))
    print("done", sum(s["status"] == "done" for s in sts), "of", len(sts), "new spend",
          round(sum(s["new_spend"] for s in sts), 2), flush=True)


if __name__ == "__main__":
    main()
