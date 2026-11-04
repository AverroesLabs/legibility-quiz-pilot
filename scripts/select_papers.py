#!/usr/bin/env python3
"""Draw the pilot papers by the rule in docs/paper-selection.md. Standard library only.

Run from the repo root:  python3 -I scripts/select_papers.py frame | draw
"frame" fetches the frame once. "draw" runs the screens and can resume from the cache.
Downloads are untrusted data. They stay in the cache and data directories, and are only parsed.
"""
import datetime as dt
import gzip
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_text

SEED = "20261102"
RULE_VERSION = "v2"
ROOT = Path(__file__).resolve().parent.parent
SEL = ROOT / "data" / "selection"
PAPERS = ROOT / "data" / "papers"
CACHE = Path(os.environ.get("LQ_CACHE", Path.home() / "lq-cache"))
UA = "legibility-quiz-pilot/0.1 (research; python-urllib)"
DELAY = 3.2
ATOM = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}

WINDOW = ("202607010000", "202609302359")
PRIMARY = {"cs.AI", "cs.LG", "cs.CL", "cs.CR"}
STRATA = {
    "A": ['"mechanistic interpretability"', '"sparse autoencoder"'],
    "B": ["jailbreak", '"red teaming"', '"prompt injection"'],
    "C": ['"AI safety"', '"AI alignment"', '"AI control"', '"scalable oversight"', "scheming", "sandbagging", '"reward hacking"'],
}
CORE_QUOTA, RESERVE_QUOTA, MAX_CHAINS, MAX_SCREENED = 4, 2, 3, 400
MIN_WORDS, MAX_WORDS, MAX_FIGURES, MAX_ERRORS, MIN_ASCII = 3000, 15000, 12, 5, 0.90
MIN_CITES, MAX_A_TRIES, CHAIN_A_START = 3, 3, "2601"
LICENCES = re.compile(r"creativecommons\.org/(licenses/by/|licenses/by-sa/|publicdomain/zero/)")
TYPE_RE = re.compile(r"\b(survey|review|tutorial|position paper|perspective|overview|workshop report|withdrawn)\b", re.I)
TYPE_ABS_RE = re.compile(r"\b(this (survey|review|tutorial|perspective|overview)|we (survey|review the literature)|systematic (review|literature)|position paper|this paper has been withdrawn|paper is withdrawn)\b", re.I)

_last = 0.0


def fetch(url):
    """Polite GET. Returns (status, bytes). A 404 is a result, not an error."""
    global _last
    for attempt in range(6):
        wait = _last + DELAY - time.time()
        if wait > 0:
            time.sleep(wait)
        _last = time.time()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=90) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return 404, b""
            print(f"HTTP {e.code} on {url}; retry {attempt}", flush=True)
        except Exception as e:  # network errors: retry with backoff
            print(f"{type(e).__name__} on {url}; retry {attempt}", flush=True)
        time.sleep(15 * (attempt + 1))
    raise RuntimeError(f"gave up on {url}")


def cached(name, url):
    path = CACHE / name
    if path.exists():
        data = path.read_bytes()
        return (404, b"") if data == b"__404__" else (200, data)
    status, data = fetch(url)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"__404__" if status == 404 else data)
    return status, data


def parse_entries(xml_bytes):
    root = ET.fromstring(xml_bytes)
    total = root.find("{http://a9.com/-/spec/opensearch/1.1/}totalResults")
    out = []
    for e in root.findall("a:entry", ATOM):
        raw = e.find("a:id", ATOM).text.strip().rsplit("/abs/", 1)[1]
        m = re.fullmatch(r"(.+)v(\d+)", raw)
        if not m:
            continue
        prim = e.find("x:primary_category", ATOM)
        out.append({
            "arxiv_id": m.group(1),
            "version": int(m.group(2)),
            "title": " ".join(e.find("a:title", ATOM).text.split()),
            "abstract": " ".join(e.find("a:summary", ATOM).text.split()),
            "published": e.find("a:published", ATOM).text.strip(),
            "primary_category": prim.get("term") if prim is not None else None,
            "authors": [a.find("a:name", ATOM).text.strip() for a in e.findall("a:author", ATOM)],
        })
    return int(total.text) if total is not None and total.text else 0, out


def rank(arxiv_id):
    return hashlib.sha256(f"{SEED}:{arxiv_id}".encode()).hexdigest()


def cmd_frame():
    SEL.mkdir(parents=True, exist_ok=True)
    cats = "(" + " OR ".join(f"cat:{c}" for c in sorted(PRIMARY)) + ")"
    date = f"submittedDate:[{WINDOW[0]} TO {WINDOW[1]}]"
    frame, counts = {}, {}
    for stratum, terms in STRATA.items():
        q = f"{date} AND {cats} AND (" + " OR ".join(f"abs:{t}" for t in terms) + ")"
        rows, start, total = [], 0, None
        while total is None or start < total:
            url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
                {"search_query": q, "start": start, "max_results": 200, "sortBy": "submittedDate", "sortOrder": "ascending"})
            _, data = fetch(url)
            total, entries = parse_entries(data)
            if not entries and start < total:
                print(f"empty page at {start}/{total}; retrying", flush=True)
                time.sleep(20)
                continue
            rows += entries
            start += len(entries)
            print(f"{stratum}: {start}/{total}", flush=True)
        counts[stratum] = {"api_total": total, "rows": len(rows)}
        for r in rows:
            if r["arxiv_id"] not in frame:
                r["stratum"] = stratum
                frame[r["arxiv_id"]] = r
    lo = dt.datetime.strptime(WINDOW[0], "%Y%m%d%H%M").replace(tzinfo=dt.timezone.utc)
    hi = dt.datetime.strptime(WINDOW[1], "%Y%m%d%H%M").replace(tzinfo=dt.timezone.utc) + dt.timedelta(minutes=1)
    kept = []
    for r in frame.values():
        when = dt.datetime.fromisoformat(r["published"].replace("Z", "+00:00"))
        if r["primary_category"] in PRIMARY and lo <= when < hi:
            r["rank"] = rank(r["arxiv_id"])
            kept.append(r)
    kept.sort(key=lambda r: r["rank"])
    with gzip.GzipFile(SEL / "frame.jsonl.gz", "wb", mtime=0) as f:
        for r in kept:
            f.write((json.dumps(r, sort_keys=True) + "\n").encode())
    per = {s: sum(r["stratum"] == s for r in kept) for s in STRATA}
    meta = {"fetched_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "rule_version": RULE_VERSION, "seed": SEED,
            "api_counts": counts, "unique_in_any_stratum": len(frame), "after_filters": len(kept), "per_stratum": per}
    (SEL / "frame-meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


def literal_stratum(abstract):
    """First stratum with a term literally in the abstract, ignoring case and hyphens. Rule v2."""
    text = abstract.lower().replace("-", " ")
    for stratum, terms in STRATA.items():
        if any(t.strip('"').lower() in text for t in terms):
            return stratum
    return None


def background_ids():
    ids = set()
    for p in (ROOT / "docs" / "background").glob("*.md"):
        ids |= set(re.findall(r"\b\d{4}\.\d{4,5}\b", p.read_text()))
    return ids


def licence_of(arxiv_id):
    _, data = cached(f"abs/{arxiv_id}.html", f"https://arxiv.org/abs/{arxiv_id}")
    m = re.search(rb'abs-license"><a href="([^"]+)"', data)
    return m.group(1).decode() if m else None


def screen(r, background):
    """Return (reason it failed or None, parsed paper or None, licence, html bytes)."""
    aid = r["arxiv_id"]
    lic = licence_of(aid)
    if not lic or not LICENCES.search(lic):
        return f"licence:{lic}", None, lic, None
    if TYPE_RE.search(r["title"]) or TYPE_ABS_RE.search(r["abstract"]):
        return "type", None, lic, None
    if aid in background:
        return "independence", None, lic, None
    status, html = cached(f"html/{aid}v{r['version']}.html", f"https://arxiv.org/html/{aid}v{r['version']}")
    if status != 200:
        return "format:no_html", None, lic, None
    parsed = paper_text.parse(html.decode("utf-8", "replace"))
    if parsed["conversion_errors"] > MAX_ERRORS:
        return f"format:errors={parsed['conversion_errors']}", None, lic, None
    if not MIN_WORDS <= parsed["words"] <= MAX_WORDS:
        return f"length:{parsed['words']}", None, lic, None
    if parsed["figures"] > MAX_FIGURES:
        return f"figures:{parsed['figures']}", None, lic, None
    if parsed["ascii_ratio"] < MIN_ASCII:
        return f"language:{parsed['ascii_ratio']}", None, lic, None
    return None, parsed, lic, html


def save_paper(r, parsed, lic, html, role):
    d = PAPERS / f"{r['arxiv_id']}v{r['version']}"
    d.mkdir(parents=True, exist_ok=True)
    with gzip.GzipFile(d / "raw.html.gz", "wb", mtime=0) as f:
        f.write(html)
    (d / "text.md").write_text(parsed["body"] + "\n")
    (d / "references.md").write_text(parsed["references"] + "\n")
    meta = {k: r[k] for k in ("arxiv_id", "version", "title", "authors", "abstract", "published", "primary_category", "stratum", "rank") if k in r}
    meta.update({
        "role": role,
        "licence": lic,
        "source_url": f"https://arxiv.org/abs/{r['arxiv_id']}v{r['version']}",
        "html_sha256": hashlib.sha256(html).hexdigest(),
        "text_sha256": hashlib.sha256((parsed["body"] + "\n").encode()).hexdigest(),
        "changes": "Converted from arXiv HTML to plain text; images removed.",
        "words": parsed["words"], "figures": parsed["figures"], "conversion_errors": parsed["conversion_errors"],
        "ascii_ratio": parsed["ascii_ratio"], "hidden_style_hits": parsed["hidden_style_hits"],
        "injection_phrase_hits": parsed["injection_phrase_hits"],
    })
    (d / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")


def lookup(arxiv_id):
    _, data = cached(f"meta/{arxiv_id}.xml", f"https://export.arxiv.org/api/query?id_list={arxiv_id}&max_results=1")
    _, entries = parse_entries(data)
    return entries[0] if entries else None


def cmd_draw():
    with gzip.open(SEL / "frame.jsonl.gz", "rb") as f:
        fetched = [json.loads(line) for line in f]
    frame = []
    for r in fetched:
        r["stratum"] = literal_stratum(r["abstract"])
        if r["stratum"]:
            frame.append(r)
    frame_sizes = {s: sum(r["stratum"] == s for r in frame) for s in STRATA}
    background = background_ids()
    core, reserve = {s: [] for s in STRATA}, {s: [] for s in STRATA}
    pool, chains, log, used = [], [], [], set()
    tried_a = {}

    def done():
        return all(len(core[s]) == CORE_QUOTA and len(reserve[s]) == RESERVE_QUOTA for s in STRATA) and len(chains) >= MAX_CHAINS

    def try_chain(b, parsed_b):
        month = b["arxiv_id"][:4]
        cands = [c for c in parsed_b["cited_arxiv"] if c["in_text_citations"] >= MIN_CITES
                 and CHAIN_A_START <= c["arxiv_id"][:4] <= month and c["arxiv_id"] not in used and c["arxiv_id"] != b["arxiv_id"]]
        cands.sort(key=lambda c: (-c["in_text_citations"], rank(c["arxiv_id"])))
        for c in cands[:MAX_A_TRIES]:
            aid = c["arxiv_id"]
            if aid not in tried_a:
                a = lookup(aid)
                if a is None:
                    tried_a[aid] = ("not_found", None, None, None, None)
                else:
                    a["stratum"] = "chain"
                    a["rank"] = rank(aid)
                    tried_a[aid] = (*screen(a, background), a)
            reason, parsed, lic, html, a = tried_a[aid]
            log.append({"arxiv_id": aid, "as": "chain_A", "for_B": b["arxiv_id"], "in_text_citations": c["in_text_citations"], "result": reason or "pass"})
            if reason is None and a["primary_category"] is not None:
                return a, parsed, lic, html, c["in_text_citations"]
        return None

    screened = 0
    for r in frame:
        if done() or screened >= MAX_SCREENED:
            break
        reason, parsed, lic, html = screen(r, background)
        screened += 1
        entry = {"arxiv_id": r["arxiv_id"], "stratum": r["stratum"], "order": screened, "result": reason or "pass"}
        if reason is None:
            s = r["stratum"]
            if len(core[s]) < CORE_QUOTA:
                core[s].append(r)
                role = "core"
            elif len(reserve[s]) < RESERVE_QUOTA:
                reserve[s].append(r)
                role = "reserve"
            else:
                role = "chain_pool"
            entry["role"] = role
            used.add(r["arxiv_id"])
            if role != "chain_pool":
                save_paper(r, parsed, lic, html, role)
            pool.append((r, parsed, lic, html, role))
        log.append(entry)
        # Chains are only taken from papers beyond core and reserves, in draw order.
        if reason is None and entry["role"] == "chain_pool" and len(chains) < MAX_CHAINS:
            found = try_chain(r, parsed)
            if found:
                a, pa, la, ha, n = found
                used.add(a["arxiv_id"])
                save_paper(a, pa, la, ha, "chain_A")
                save_paper(r, parsed, lic, html, "chain_B")
                chains.append({"A": f"{a['arxiv_id']}v{a['version']}", "B": f"{r['arxiv_id']}v{r['version']}", "in_text_citations_of_A_in_B": n})
        if screened % 25 == 0:
            print(f"screened {screened}; core {[len(core[s]) for s in STRATA]} reserve {[len(reserve[s]) for s in STRATA]} chains {len(chains)}", flush=True)

    with (SEL / "screening-log.jsonl").open("w") as f:
        for e in log:
            f.write(json.dumps(e, sort_keys=True) + "\n")
    ids = lambda rows: [f"{r['arxiv_id']}v{r['version']}" for r in rows]
    tally = {}
    for e in log:
        if e.get("as") != "chain_A":
            key = e["result"].split(":")[0]
            tally[key] = tally.get(key, 0) + 1
    summary = {"rule_version": RULE_VERSION, "seed": SEED, "drawn_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
               "fetched_frame": len(fetched), "literal_frame": frame_sizes, "screened": screened, "screen_outcomes": tally,
               "core": {s: ids(core[s]) for s in STRATA}, "reserve": {s: ids(reserve[s]) for s in STRATA}, "chains": chains,
               "stop_reason": "all quotas met" if done() else ("screen cap" if screened >= MAX_SCREENED else "frame exhausted")}
    (SEL / "selection.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    {"frame": cmd_frame, "draw": cmd_draw}[sys.argv[1]]()
