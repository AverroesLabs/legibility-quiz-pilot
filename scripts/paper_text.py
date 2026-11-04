"""Convert arXiv native (LaTeXML) HTML into plain text for readers and quiz writers.

Standard library only. The input is untrusted: this parses it and never runs it.
Keeps headings, paragraphs, tables, listings, footnotes, captions and math (as LaTeX).
Drops images, navigation and scripts.
"""
import re
from html.parser import HTMLParser

VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}
SKIP_TAGS = {"script", "style", "nav", "header", "footer", "svg", "annotation", "annotation-xml"}
SKIP_CLASSES = {"ltx_listing_data", "ltx_TOC", "package-alerts", "ltx_page_footer", "ltx_page_logo"}
BLOCK_TAGS = {"p", "div", "section", "figure", "figcaption", "li", "ul", "ol", "table", "article", "blockquote", "h1", "h2", "h3", "h4", "h5", "h6"}
HIDDEN_STYLE = re.compile(r"display:\s*none|visibility:\s*hidden|font-size:\s*0(?:\D|$)|opacity:\s*0(?:\D|$)|color:\s*(?:#fff(?:fff)?\b|white\b|rgb\(255,\s*255,\s*255\))", re.I)
INJECTION = re.compile(r"ignore (?:all |any )?(?:previous|prior|above) instructions|give (?:a|only) positive review|as an ai (?:reviewer|model|assistant),? (?:you|please)|do not highlight any negatives", re.I)
ARXIV_ID = re.compile(r"(?:arxiv[:\s./a-z]*)(\d{4}\.\d{4,5})", re.I)


class _Parser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []  # (tag, skip, section)
        self.parts = []
        self.refs_parts = []
        self.in_article = False
        self.figures = 0
        self.errors = 0
        self.hidden_style_hits = 0
        self.bib = {}  # bib anchor -> text
        self.cites = {}  # bib anchor -> count of in-text citations
        self.current_bib = None
        self.heading = None

    def _skipping(self):
        return any(s[1] for s in self.stack)

    def _in_bib(self):
        return any(s[2] for s in self.stack)

    def _out(self):
        return self.refs_parts if self._in_bib() else self.parts

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        classes = set((a.get("class") or "").split())
        if tag == "article":
            self.in_article = True
        if not self.in_article:
            if tag not in VOID:
                self.stack.append((tag, False, False))
            return
        skip = tag in SKIP_TAGS or bool(classes & SKIP_CLASSES)
        is_bib = "ltx_bibliography" in classes
        if "ltx_ERROR" in classes:
            self.errors += 1
        if HIDDEN_STYLE.search(a.get("style") or ""):
            self.hidden_style_hits += 1
        if tag == "figure" and "ltx_figure" in classes:
            self.figures += 1
        if tag == "li" and "ltx_bibitem" in classes:
            self.current_bib = a.get("id")
            self.bib[self.current_bib] = ""
        if tag == "a" and not self._in_bib() and not self._skipping():
            href = a.get("href") or ""
            if "#bib.bib" in href:
                key = href.split("#", 1)[1]
                self.cites[key] = self.cites.get(key, 0) + 1
        if tag == "a" and self.current_bib:
            self.bib[self.current_bib] += " " + (a.get("href") or "") + " "
        if not self._skipping():
            out = self._out()
            if tag == "math":
                alt = (a.get("alttext") or "").strip()
                if alt:
                    out.append(f" ${alt}$ ")
                skip = True
            elif tag == "img":
                pass
            elif tag == "br":
                out.append("\n")
            elif tag in ("td", "th"):
                out.append(" | ")
            elif tag == "tr":
                out.append("\n")
            elif tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
                out.append("\n\n" + "#" * int(tag[1]) + " ")
            elif tag in BLOCK_TAGS or "ltx_listingline" in classes:
                out.append("\n\n" if "ltx_listingline" not in classes else "\n")
        if tag not in VOID:
            self.stack.append((tag, skip, is_bib))

    def handle_endtag(self, tag):
        if tag == "article":
            self.in_article = False
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == tag:
                del self.stack[i:]
                break
        if tag == "li":
            self.current_bib = None
        if self.in_article and not self._skipping() and tag in BLOCK_TAGS:
            self._out().append("\n\n")

    def handle_data(self, data):
        if not self.in_article or self._skipping():
            return
        data = data.replace("␣", " ")
        self._out().append(data)
        if self.current_bib:
            self.bib[self.current_bib] += data


def _clean(parts):
    t = "".join(parts)
    t = re.sub(r"[ \t\r\f\v]+", " ", t)
    t = re.sub(r" ?\n ?", "\n", t)
    t = re.sub(r"(?m)^(#{1,6})\s*\n+\s*(?=\S)", r"\1 ", t)  # heading text on its marker's line
    t = re.sub(r"(?<!\n)\n\| ", " | ", t)  # table cells of one row on one line
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def parse(html):
    """Return the body text, references, counts and citation data for one paper."""
    p = _Parser()
    p.feed(html)
    p.close()
    body = _clean(p.parts)
    refs = _clean(p.refs_parts)
    letters = [c for c in body if c.isalpha()]
    ascii_ratio = sum(c.isascii() for c in letters) / len(letters) if letters else 0.0
    cited = []
    for key, text in p.bib.items():
        m = ARXIV_ID.search(text)
        if m:
            cited.append({"arxiv_id": m.group(1), "in_text_citations": p.cites.get(key, 0)})
    return {
        "body": body,
        "references": refs,
        "words": len(body.split()),
        "figures": p.figures,
        "conversion_errors": p.errors,
        "ascii_ratio": round(ascii_ratio, 4),
        "hidden_style_hits": p.hidden_style_hits,
        "injection_phrase_hits": [m.group(0) for m in INJECTION.finditer(body)],
        "cited_arxiv": cited,
    }
