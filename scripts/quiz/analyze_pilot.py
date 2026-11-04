#!/usr/bin/env python3
"""Analysis of the scaled-down pilot, as fixed in docs/preregistration.md.

Reads data/runs/pilot-<paper>/ (screened quiz, grades.jsonl, versions/meta.json) for every paper that has grades,
and writes data/runs/pilot-analysis.md and pilot-analysis.json. Standard library only.
"""
import json, random, statistics, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "data/runs"
PAPERS = ["2609.00755v1", "2608.25490v1", "2607.20379v2", "2607.17117v2", "2607.28685v1", "2608.19216v1"]
STRATUM = {"2609.00755v1": "A", "2607.17117v2": "A", "2608.25490v1": "B", "2607.28685v1": "B",
           "2607.20379v2": "C", "2608.19216v1": "C"}
SMART, LADDER = "openai/gpt-5.6-sol", ["mistralai/ministral-8b-2512", "google/gemma-4-31b-it"]
READERS = [SMART] + LADDER
GRADER = "z-ai/glm-5.3"
VERSIONS = ["paper", "obf", "rem"]
# Four-condition filter thresholds, the development values used since dev2 (quiz.py), applied on the clear version.
SMART_NONE_MAX, SMART_PAPER_MIN, DUMB_NONE_MAX = 0.5, 0.5, 0.5
COVER = 0.7
B, SEED = 10000, 20261102


def mean(xs):
    return statistics.mean(xs) if xs else float("nan")


def measures(items, post, pre, cover=COVER):
    """items: kept item dicts. post/pre: item id -> score. Returns M1..M5 and the weakest section id."""
    ids = [it["id"] for it in items if it["id"] in post]
    out = {"M1": mean([post[i] - pre[i] for i in ids if i in pre]), "M2": mean([post[i] for i in ids])}
    claims = defaultdict(list)
    for it in items:
        if it["id"] in post and it.get("kind") == "claim" and it.get("claim"):
            claims[it["claim"]].append(post[it["id"]])
    out["M3"] = mean([mean(v) >= cover for v in claims.values()]) if claims else float("nan")
    secs = defaultdict(list)
    for it in items:
        if it["id"] in post:
            for s in it["sections"]:
                secs[s].append(post[it["id"]])
    secs = {s: mean(v) for s, v in secs.items() if len(v) >= 3}
    out["M4"] = min(secs.values()) if secs else float("nan")
    out["weakest"] = min(secs, key=secs.get) if secs else None
    out["n_sections"] = len(secs)
    out["M5"] = mean([post[it["id"]] for it in items if it["id"] in post and it.get("kind") == "integration"])
    return out


def load(paper):
    d = RUNS / f"pilot-{paper}"
    qp, gp = d / "quiz-claude-opus-5.5-1-screened.json", d / "grades.jsonl"
    if not (qp.exists() and gp.exists() and (d / "pilot.done").exists()):
        return None
    quiz = json.loads(qp.read_text())
    sc = defaultdict(list)  # (item, reader, cond, sample) -> score
    for line in gp.read_text().splitlines():
        g = json.loads(line)
        if g["grader"] == GRADER:
            sc[(g["item"], g["reader"], g["cond"], g["sample"])].append(g["score"])
    meta = json.loads((d / "versions/meta.json").read_text())
    # Answers the grader never scored (e.g. GLM returning empty content at its token limit). They are left out of
    # every measure, so report how many there are and where.
    ungraded = defaultdict(int)
    graded = {(g[0], g[1], g[2], g[3]) for g in sc}
    for line in (d / "answers.jsonl").read_text().splitlines():
        r = json.loads(line)
        if (r["item"], r["reader"], r["cond"], r["sample"]) not in graded:
            ungraded[f"{r['reader'].split('/')[1]}:{r['cond']}"] += 1
    meta["ungraded"] = dict(ungraded)
    return quiz, sc, meta


def per_item(sc, reader, cond, samples=(0, 1)):
    out = defaultdict(list)
    for (i, r, c, s), v in sc.items():
        if r == reader and c == cond and s in samples:
            out[i] += v
    return {i: mean(v) for i, v in out.items()}


def analyse(paper):
    got = load(paper)
    if not got:
        return None
    quiz, sc, meta = got
    items = quiz["items"]
    sn, sp = per_item(sc, SMART, "none"), per_item(sc, SMART, "paper")
    dn = {i: mean([per_item(sc, r, "none")[i] for r in LADDER if i in per_item(sc, r, "none")]) for i in sn}
    kept = [it for it in items if sn.get(it["id"], 1) <= SMART_NONE_MAX and sp.get(it["id"], 0) >= SMART_PAPER_MIN
            and dn.get(it["id"], 1) <= DUMB_NONE_MAX]
    kept_claims = {it["claim"] for it in kept if it.get("kind") == "claim"}
    smart_clear = mean([sp[it["id"]] for it in kept if it["id"] in sp])
    gate = len(kept) >= 10 and len(kept_claims) >= 4 and smart_clear >= 0.75
    res = {"paper": paper, "stratum": STRATUM[paper], "n_items": len(items), "kept": [it["id"] for it in kept],
           "kept_claims": sorted(kept_claims), "smart_clear_kept": smart_clear, "gate": gate,
           "obf_section": meta["section"], "obf_title": meta["title"], "eligible": meta["eligible"],
           "items_tagged_obf": [i for i in meta["items_tagged"] if i in {it["id"] for it in kept}],
           "chars": [meta["chars_before"], meta["chars_after"]], "ungraded": meta["ungraded"], "readers": {}}
    for r in READERS:
        pre = per_item(sc, r, "none")
        rr = {}
        for v in VERSIONS:
            post = per_item(sc, r, v)
            m = measures(kept, post, pre)
            m["by_sample"] = [measures(kept, per_item(sc, r, v, (s,)), pre) for s in (0, 1)]
            m["M3_sweep"] = {str(t): measures(kept, post, pre, t)["M3"] for t in (0.5, 0.6, 0.7, 0.8, 0.9)}
            rr[v] = m
        det = {}
        for k in ("M1", "M2", "M3", "M4", "M5"):
            c, o, rem = rr["paper"][k], rr["obf"][k], rr["rem"][k]
            sd = statistics.stdev([x[k] for x in rr["paper"]["by_sample"]]) if all(
                x[k] == x[k] for x in rr["paper"]["by_sample"]) else float("nan")
            thr = max(0.10, 2 * sd) if sd == sd else 0.10
            det[k] = {"clear": c, "obf": o, "rem": rem, "diff": c - o, "diff_rem": c - rem, "sd": sd, "thr": thr,
                      "detects": (c - o) > thr, "ci": boot_diff(kept, sc, r, k)}
        rr["detect"] = det
        rr["localized"] = rr["obf"]["weakest"] == meta["section"]
        res["readers"][r] = rr
    res["tagged"] = tagged_diag(items, res["items_tagged_obf"], [it["id"] for it in kept], sc)
    res["info_check"] = res["readers"][SMART]["obf"]["M2"] >= 0.8 * res["readers"][SMART]["paper"]["M2"]
    return res


def tagged_diag(items, tagged, kept_ids, sc):
    """EXPLORATORY (added 2026-11-03 09:08 UTC after 2 papers, not preregistered): mean score on kept items tagged to
    the obfuscated section vs other kept items, per reader and condition, with 95% item bootstrap intervals for
    clear minus obfuscated and clear minus removed on the tagged items."""
    out = {}
    other = [i for i in kept_ids if i not in tagged]
    for r in READERS:
        s = {v: per_item(sc, r, v) for v in ["none"] + VERSIONS}
        row = {v: {"tagged": mean([s[v][i] for i in tagged if i in s[v]]),
                   "other": mean([s[v][i] for i in other if i in s[v]])} for v in s}
        for alt in ("obf", "rem"):
            ids = [i for i in tagged if i in s["paper"] and i in s[alt]]
            rng = random.Random(SEED)
            vals = sorted(mean([s["paper"][i] - s[alt][i] for i in rng.choices(ids, k=len(ids))]) for _ in range(B)) if ids else []
            row["clear_minus_" + alt] = [mean([s["paper"][i] - s[alt][i] for i in ids]),
                                         vals[int(0.025 * B)] if vals else float("nan"), vals[int(0.975 * B) - 1] if vals else float("nan")]
        out[r] = row
    return out


def boot_diff(kept, sc, reader, k):
    """95% item-level bootstrap interval for clear minus obfuscated on measure k."""
    pre, c, o = per_item(sc, reader, "none"), per_item(sc, reader, "paper"), per_item(sc, reader, "obf")
    rng = random.Random(SEED)
    vals = []
    n = B if k in ("M1", "M2", "M5") else B // 10  # M3, M4 are slower; 1,000 resamples
    for _ in range(n):
        smp = [dict(it, id=f"{it['id']}#{j}") for j, it in enumerate(rng.choices(kept, k=len(kept)))]
        rc = {s["id"]: c[s["id"].split("#")[0]] for s in smp if s["id"].split("#")[0] in c}
        ro = {s["id"]: o[s["id"].split("#")[0]] for s in smp if s["id"].split("#")[0] in o}
        rp = {s["id"]: pre[s["id"].split("#")[0]] for s in smp if s["id"].split("#")[0] in pre}
        d = measures(smp, rc, rp)[k] - measures(smp, ro, rp)[k]
        if d == d:
            vals.append(d)
    if not vals:
        return [float("nan")] * 2
    vals.sort()
    return [vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]]


def f(x):
    return "" if x is None or x != x else f"{x:.2f}"


def report(results):
    done = [r for r in results if r]
    gated = [r for r in done if r["gate"]]
    L = ["# Scaled-down pilot: analysis", "",
         "Generated by `scripts/quiz/analyze_pilot.py` from `data/runs/pilot-*/`, grader GLM-5.3. Bars are the proposed, "
         "unapproved ones in `docs/preregistration.md`. Kept items use the development four-condition thresholds on the "
         "clear version.", "",
         f"Papers finished: {len(done)} of 6. Passing the quiz gate: {len(gated)}.", "",
         "## Quiz gate and obfuscation", "",
         "| Paper | Stratum | Items | Kept | Claims kept | GPT clear (kept) | Gate | Obfuscated section | Kept items on it | Chars before, after | Info check |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in done:
        L.append(f"| {r['paper']} | {r['stratum']} | {r['n_items']} | {len(r['kept'])} | {len(r['kept_claims'])} | "
                 f"{f(r['smart_clear_kept'])} | {'pass' if r['gate'] else 'FAIL'} | {r['obf_section']} {r['obf_title']} | "
                 f"{len(r['items_tagged_obf'])} | {r['chars'][0]}, {r['chars'][1]} | {'pass' if r['info_check'] else 'FAIL'} |")
    L += ["", "Answers left ungraded (excluded from all measures), by reader and condition:", ""]
    for r in done:
        u = r["ungraded"]
        L.append(f"- {r['paper']}: {sum(u.values())}" + (" (" + ", ".join(f"{k} {v}" for k, v in sorted(u.items())) + ")" if u else ""))
    L += ["", "## Measures: clear / obfuscated / section removed, and clear minus obfuscated with 95% item bootstrap CI", "",
          "`*` marks detection (difference above max(0.10, 2 x between-sample SD on clear)).", ""]
    for r in done:
        L += [f"### {r['paper']} (stratum {r['stratum']}, gate {'pass' if r['gate'] else 'FAIL'})", "",
              "| Reader | " + " | ".join(["M1", "M2", "M3", "M4", "M5"]) + " | weakest section in obf (obf is " + r["obf_section"] + ") |",
              "|---|---|---|---|---|---|---|"]
        for rd, rr in r["readers"].items():
            cells = []
            for k in ("M1", "M2", "M3", "M4", "M5"):
                d = rr["detect"][k]
                cells.append(f"{f(d['clear'])} / {f(d['obf'])} / {f(d['rem'])}; {f(d['diff'])}{'*' if d['detects'] else ''} "
                             f"[{f(d['ci'][0])}, {f(d['ci'][1])}]")
            L.append(f"| {rd.split('/')[1]} | " + " | ".join(cells) +
                     f" | {rr['obf']['weakest']} of {rr['obf']['n_sections']} |")
        L.append("")
    L += ["## Bars (proposed, unapproved)", ""]
    ok = [r for r in gated if r["info_check"]]
    for rd in LADDER:
        L.append(f"**Bar 1, {rd}:** papers passing gate and information check: {len(ok)}. Detections by measure: " +
                 ", ".join(f"{k} {sum(r['readers'][rd]['detect'][k]['detects'] for r in ok)}" for k in ("M1", "M2", "M3", "M4", "M5")) +
                 ". Signal needs at least 4.")
        L.append("")
    L.append("**Bar 3 (per detecting paper and measure, ladder readers):** count where the obfuscated score is closer "
             "to clear than the section-removed score is.")
    for rd in LADDER:
        for k in ("M1", "M2", "M3", "M4", "M5"):
            det = [r for r in ok if r["readers"][rd]["detect"][k]["detects"]]
            if det:
                closer = sum(abs(r["readers"][rd]["detect"][k]["diff"]) < abs(r["readers"][rd]["detect"][k]["diff_rem"]) for r in det)
                L.append(f"- {rd.split('/')[1]} {k}: {closer} of {len(det)}")
    L += ["", "**Bar 4 (localization, descriptive):** papers where the weakest section in the obfuscated version is the "
          "obfuscated one, with chance (sum over papers of 1 / sections scored)."]
    for rd in READERS:
        hits = sum(r["readers"][rd]["localized"] for r in gated)
        chance = sum(1 / r["readers"][rd]["obf"]["n_sections"] for r in gated if r["readers"][rd]["obf"]["n_sections"])
        L.append(f"- {rd.split('/')[1]}: {hits} of {len(gated)} (chance {chance:.1f})")
    L += ["", "## Exploratory: items on the obfuscated section", "",
          "Not preregistered; added after the first 2 papers finished. Mean score on kept items tagged to the "
          "obfuscated section (and, after the slash, on the other kept items). The last two columns are clear minus "
          "obfuscated and clear minus section removed on the tagged items, with 95% item bootstrap intervals. If the "
          "obfuscation hid the section, the obfuscated drop should approach the removed drop.", "",
          "| Paper | Tagged items | Reader | none | clear | obf | removed | clear - obf | clear - removed |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in done:
        for rd, t in r["tagged"].items():
            co, cr = t["clear_minus_obf"], t["clear_minus_rem"]
            L.append(f"| {r['paper']} | {len(r['items_tagged_obf'])} | {rd.split('/')[1]} | " +
                     " | ".join(f"{f(t[v]['tagged'])} / {f(t[v]['other'])}" for v in ("none", "paper", "obf", "rem")) +
                     f" | {f(co[0])} [{f(co[1])}, {f(co[2])}] | {f(cr[0])} [{f(cr[1])}, {f(cr[2])}] |")
    return "\n".join(L) + "\n"


def main():
    results = [analyse(p) for p in PAPERS]
    (RUNS / "pilot-analysis.json").write_text(json.dumps([r for r in results if r], indent=1))
    md = report(results)
    (RUNS / "pilot-analysis.md").write_text(md)
    print(md)


if __name__ == "__main__":
    sys.exit(main())
