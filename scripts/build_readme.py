"""Build README.md from the blog posts in posts/, so the series can be read start to finish on the repository's front
page while each post also stands alone. Edit the posts (and docs/about-repo.md), never README.md, then run:

    python scripts/build_readme.py

README = series intro and contents, every post in order (headings shifted down one level, links rewritten to work from
the repository root), then the repository guide from docs/about-repo.md.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
POSTS = [("posts/part-1-quiz.md", "Part 1"), ("posts/part-2-planted-errors.md", "Part 2")]

INTRO = """# Legibility of research papers to humans: an automated test, in parts

Averroes Labs is trying to build an automated test of whether a research paper is legible to people, one that a model
can run on any paper. This repository holds the posts written so far, in order, and the code and data behind them.
Each post can also be read on its own:

{contents}

Below, the posts follow one another in full, then a guide to the repository.
"""


def title(text):
    return re.search(r"^# (.+)$", text, re.M).group(1).strip()


def anchor(heading):
    """GitHub's heading slug."""
    s = re.sub(r"[^\w\- ]", "", heading.lower()).strip()
    return s.replace(" ", "-")


def for_readme(text, part):
    text = re.sub(r"^(#{1,5}) ", lambda m: "#" + m.group(1) + " ", text, flags=re.M)   # demote headings one level
    text = re.sub(r"^## (.+)$", lambda m: f"## {part}: {m.group(1)}", text, count=1, flags=re.M)
    text = text.replace("](../", "](")                                              # paths relative to the root
    for path, p in POSTS:                                                           # cross-links become anchors
        name = Path(path).name
        heading = f"{p}: {title((ROOT / path).read_text())}"
        text = text.replace(f"]({name})", f"](#{anchor(heading)})")
    text = re.sub(r"\n---\n\n(Next|Previous): .*\n?$", "\n", text.rstrip() + "\n")  # drop per-post footer nav
    return text.rstrip() + "\n"


def main():
    contents = "\n".join(f"- [{p}: {title((ROOT / path).read_text())}]({path})" for path, p in POSTS)
    parts = [INTRO.format(contents=contents)]
    for path, p in POSTS:
        parts.append(for_readme((ROOT / path).read_text(), p))
    about = (ROOT / "docs/about-repo.md").read_text().replace("](../", "](")
    parts.append(about)
    (ROOT / "README.md").write_text("\n---\n\n".join(s.rstrip() + "\n" for s in parts))
    print("wrote README.md from", ", ".join(path for path, _ in POSTS))


if __name__ == "__main__":
    main()
