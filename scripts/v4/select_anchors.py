"""Reviewed-anchor draw for experiment 9, rule v2 in docs/paper-selection.md ("Reviewed anchors, rule v2").

python scripts/v4/select_anchors.py   # builds the frame, screens in hash order, writes data/selection/anchors/
Network reads go through select_papers.cached (polite, cached in ~/lq-cache). No OpenAI calls.
"""
import datetime as dt
import gzip
import hashlib
import html as htmlmod
import json
import re
import sys
import unicodedata
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paper_text  # noqa: E402
import select_papers as sp  # noqa: E402

sp.DELAY = 15.0  # arxiv.org robots.txt asks Crawl-delay: 15; the API answered 429 at 3.2 s (pass 23)

RULE = "anchors-v2"
SEED = "20261113-anchors:"
OUT = sp.ROOT / "data" / "selection" / "anchors"
TMLR_URL = "https://jmlr.org/tmlr/papers/"
ICLR_URL = "https://iclr.cc/virtual/2026/papers.html"
TERMS = ["interpretability", "sparse autoencoder", "jailbreak", "red teaming", "prompt injection", "AI safety", "alignment",
         "oversight", "scheming", "sandbagging", "reward hacking", "backdoor", "adversarial", "robustness",
         "out-of-distribution", "unlearning", "privacy", "hallucination"]
BANDS = {"short": (4000, 7000), "medium": (7000, 10000), "long": (10000, 15000)}  # lower bound exclusive except short
PLACES, MAX_SCREENED, MAX_ERRORS, MAX_FIGURES, MIN_ASCII = 2, 600, 5, 12, 0.90
MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                      "september", "october", "november", "december"], 1)}
APPENDIX = re.compile(r"^#{1,2} (?:[\d.]+\s+)?(?:Appendix|Appendices|Supplementary|[A-Z](?: |\.\d))")
FIGURE = re.compile(r'<figure[^>]*class="[^"]*\bltx_figure\b')


def norm(title):
    """Rank key (rule v2): lower case, only letters and digits kept (accented letters are letters)."""
    return "".join(c for c in title.lower() if c.isalnum())


def fold(title):
    """Title match key: accents and TeX accent commands folded to ASCII, then only a-z0-9."""
    t = re.sub(r"\\[`'^\"~=.uvHc]\{?", "", title)
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", t.lower())


def area(title):
    t = title.lower().replace("-", " ")
    return any(term.lower().replace("-", " ") in t for term in TERMS)


def surname(name):
    return name.split()[-1].lower() if name.split() else ""


def frame():
    entries = {}
    _, raw = sp.cached("anchors/tmlr-papers.html", TMLR_URL)
    for item in re.findall(r'<li class="item[^"]*">(.*?)</li>', raw.decode(), re.S):
        title = htmlmod.unescape(re.sub(r"<[^>]+>", "", re.search(r"<h4>(.*?)</h4>", item, re.S).group(1))).strip()
        authors = [a.strip() for a in htmlmod.unescape(re.search(r"<i>(.*?)</i>", item, re.S).group(1)).split(",")]
        month, year = re.search(r"</i>,\s*([A-Za-z]+) (\d{4})", item).groups()
        start = dt.date(int(year), MONTHS[month.lower()], 1)
        if not dt.date(2025, 1, 1) <= start <= dt.date(2026, 10, 1):
            continue
        forum = re.search(r"openreview\.net/forum\?id=([\w-]+)", item).group(1)
        entries[norm(title)] = {"venue": "TMLR", "title": " ".join(title.split()), "authors": authors,
                                "decision_date": start.isoformat(), "forum_id": forum}
    _, raw = sp.cached("anchors/iclr2026-papers.html", ICLR_URL)
    for pid, title in re.findall(r'<li><a href="/virtual/2026/poster/(\d+)">(.*?)</a></li>', raw.decode()):
        title = " ".join(htmlmod.unescape(title).split())
        entries.setdefault(norm(title), {"venue": "ICLR 2026", "title": title, "poster_id": pid,
                                         "decision_date": "2026-01-01"})
    rows = []
    for key, e in entries.items():
        if key and area(e["title"]):
            e["rank"] = hashlib.sha256((SEED + key).encode()).hexdigest()
            rows.append(e)
    return len(entries), sorted(rows, key=lambda e: e["rank"])


def iclr_details(e):
    _, raw = sp.cached(f"anchors/iclr-poster/{e['poster_id']}.html", f"https://iclr.cc/virtual/2026/poster/{e['poster_id']}")
    text = raw.decode("utf-8", "replace")
    m = re.search(r'<script type="application/ld\+json">(.*?)</script>', text, re.S)
    e["authors"] = [a["name"] for a in json.loads(m.group(1)).get("author", [])] if m else []
    f = re.search(r"openreview\.net/forum\?id=([\w-]+)", text)
    e["forum_id"] = f.group(1) if f else None


def arxiv_match(e):
    """Screen 1. Returns (arXiv entry with chosen version, note) or (None, reason)."""
    names = {surname(a) for a in e.get("authors", [])}
    queries = [f'ti:"{" ".join(re.sub(r"[^A-Za-z0-9 ]", " ", e["title"]).split())}"']
    if not e["title"].isascii():  # pass 23 fix: the phrase query drops accented letters; AND the all-ASCII words instead
        queries.append(" AND ".join(f"ti:{w}" for w in re.sub(r"[^\w ]", " ", e["title"]).split() if w.isascii() and len(w) > 2))
    for query in queries:
        q = urllib.parse.quote(query)
        key = hashlib.sha256(q.encode()).hexdigest()[:20]
        _, data = sp.cached(f"anchors/arxiv-search/{key}.xml",
                            f"https://export.arxiv.org/api/query?search_query={q}&max_results=20")
        _, hits = sp.parse_entries(data)
        hits = [h for h in hits if fold(h["title"]) == fold(e["title"]) and names & {surname(a) for a in h["authors"]}]
        if hits:
            break
    if not hits:
        return None, "arxiv:no_match"
    hit = sorted(hits, key=lambda h: h["arxiv_id"])[0]
    # Version dates come from the abstract page's submission history.
    _, absb = sp.cached(f"abs/{hit['arxiv_id']}.html", f"https://arxiv.org/abs/{hit['arxiv_id']}")
    versions = []
    for v, date in re.findall(rb"\[v(\d+)\]</strong>\s*\w{3}, (\d{1,2} \w{3} \d{4})", absb):
        versions.append((int(v), dt.datetime.strptime(date.decode(), "%d %b %Y").date()))
    if not versions:
        return None, "arxiv:no_version_dates"
    after = [v for v, d in versions if d >= dt.date.fromisoformat(e["decision_date"])]
    hit["version"] = min(after) if after else max(v for v, _ in versions)
    hit["version_note"] = "earliest on or after decision" if after else "none after decision; latest used"
    hit["version_dates"] = {str(v): d.isoformat() for v, d in versions}
    return hit, None


def main_text(body):
    lines = body.split("\n")
    for i, line in enumerate(lines):
        if i > 0 and APPENDIX.match(line):
            return "\n".join(lines[:i]).rstrip(), line
    return body, None


def band_of(words):
    if 4000 <= words <= 7000:
        return "short"
    if 7000 < words <= 10000:
        return "medium"
    if 10000 < words <= 15000:
        return "long"
    return None


def screen(e, background, filled):
    """Returns (failure reason or None, details dict)."""
    d = {}
    if e["venue"] == "ICLR 2026":
        iclr_details(e)
    hit, reason = arxiv_match(e)
    if reason:
        return reason, d
    aid, ver = hit["arxiv_id"], hit["version"]
    d["arxiv"] = hit
    lic = sp.licence_of(aid)
    d["licence"] = lic
    if not lic or not sp.LICENCES.search(lic):
        return f"licence:{lic}", d
    if sp.TYPE_RE.search(hit["title"]) or sp.TYPE_ABS_RE.search(hit["abstract"]):
        return "type", d
    if aid in background:
        return "independence", d
    status, html = sp.cached(f"html/{aid}v{ver}.html", f"https://arxiv.org/html/{aid}v{ver}")
    if status != 200:
        return "format:no_html", d
    parsed = paper_text.parse(html.decode("utf-8", "replace"))
    d["parsed"], d["html"] = parsed, html
    if parsed["conversion_errors"] > MAX_ERRORS:
        return f"format:errors={parsed['conversion_errors']}", d
    main, cut = main_text(parsed["body"])
    words = len(main.split())
    d.update(main=main, cut=cut, words=words)
    band = band_of(words)
    if band is None:
        return f"length:{words}", d
    if filled[band] >= PLACES:
        return f"band full:{band}:{words}", d
    s = html.decode("utf-8", "replace")
    app = s.find("ltx_appendix")
    d["figures"] = len(FIGURE.findall(s if app < 0 else s[:app]))
    if d["figures"] > MAX_FIGURES:
        return f"figures:{d['figures']}", d
    letters = [c for c in main if c.isalpha()]
    d["ascii_ratio"] = round(sum(c.isascii() for c in letters) / len(letters), 4)
    if d["ascii_ratio"] < MIN_ASCII:
        return f"language:{d['ascii_ratio']}", d
    d["band"] = band
    return None, d


def save(e, d):
    hit, parsed, html = d["arxiv"], d["parsed"], d["html"]
    pdir = sp.PAPERS / f"{hit['arxiv_id']}v{hit['version']}"
    pdir.mkdir(parents=True, exist_ok=True)
    with gzip.GzipFile(pdir / "raw.html.gz", "wb", mtime=0) as f:
        f.write(html)
    (pdir / "text.md").write_text(parsed["body"] + "\n")
    (pdir / "main.md").write_text(d["main"] + "\n")
    (pdir / "references.md").write_text(parsed["references"] + "\n")
    forum = e.get("forum_id")
    meta = {
        "role": "reviewed_anchor", "rule": RULE, "rank": e["rank"], "band": d["band"],
        "venue": e["venue"], "decision_date_used": e["decision_date"], "forum_id": forum,
        "openreview_url": f"https://openreview.net/forum?id={forum}" if forum else None,
        "listed_title": e["title"], "listed_authors": e.get("authors"),
        "arxiv_id": hit["arxiv_id"], "version": hit["version"], "version_note": hit["version_note"],
        "version_dates": hit["version_dates"], "title": hit["title"], "authors": hit["authors"],
        "abstract": hit["abstract"], "primary_category": hit["primary_category"],
        "licence": d["licence"], "licence_source": f"https://arxiv.org/abs/{hit['arxiv_id']}",
        "source_url": f"https://arxiv.org/html/{hit['arxiv_id']}v{hit['version']}",
        "html_sha256": hashlib.sha256(html).hexdigest(),
        "text_sha256": hashlib.sha256((parsed["body"] + "\n").encode()).hexdigest(),
        "main_sha256": hashlib.sha256((d["main"] + "\n").encode()).hexdigest(),
        "main_words": d["words"], "body_words": parsed["words"], "appendix_cut": d["cut"],
        "main_figures": d["figures"], "conversion_errors": parsed["conversion_errors"], "ascii_ratio": d["ascii_ratio"],
        "hidden_style_hits": parsed["hidden_style_hits"], "injection_phrase_hits": parsed["injection_phrase_hits"],
        "retrieved_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "changes": "Converted from arXiv HTML to plain text; images removed; main.md cut at the first appendix heading.",
    }
    (pdir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    return f"{hit['arxiv_id']}v{hit['version']}"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    listed, rows = frame()
    with (OUT / "frame.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    print(f"listed {listed}, area frame {len(rows)}", flush=True)
    core13 = json.loads((sp.SEL / "v3-expanded-13.json").read_text())["papers"]
    background = sp.background_ids() | {p["id"].rsplit("v", 1)[0] for p in core13}
    filled = {b: 0 for b in BANDS}
    chosen, log, screened = [], [], 0
    for e in rows:
        if all(v >= PLACES for v in filled.values()) or screened >= MAX_SCREENED:
            break
        screened += 1
        reason, d = screen(e, background, filled)
        entry = {"order": screened, "title": e["title"], "venue": e["venue"], "rank": e["rank"], "forum_id": e.get("forum_id"),
                 "arxiv": f"{d['arxiv']['arxiv_id']}v{d['arxiv']['version']}" if "arxiv" in d else None,
                 "main_words": d.get("words"), "result": reason or "pass"}
        if reason is None:
            filled[d["band"]] += 1
            entry["band"] = d["band"]
            chosen.append({"paper": save(e, d), "band": d["band"], "venue": e["venue"], "main_words": d["words"]})
        log.append(entry)
        print(f"{screened:4d} {entry['result'][:30]:30s} {e['venue']:9s} {e['title'][:60]}  filled={filled}", flush=True)
    with (OUT / "screen-log.jsonl").open("w") as f:
        for x in log:
            f.write(json.dumps(x, sort_keys=True) + "\n")
    tally = {}
    for x in log:
        k = x["result"].split(":")[0]
        tally[k] = tally.get(k, 0) + 1
    summary = {"rule": RULE, "seed": SEED, "drawn_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
               "listed": listed, "area_frame": len(rows), "screened": screened, "outcomes": tally, "anchors": chosen,
               "filled": filled, "stop_reason": "all bands full" if all(v >= PLACES for v in filled.values())
               else ("screen cap" if screened >= MAX_SCREENED else "frame exhausted")}
    (OUT / "selection.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
