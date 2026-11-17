"""Build README.md: the project's motivation, a link and short description for each post in posts/, then the repository
guide from docs/about-repo.md. The posts themselves are read in posts/. Edit the posts, the descriptions below or
docs/about-repo.md, never README.md, then run:

    python scripts/build_readme.py
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (path, label, short description). Keep each description to two or three plain sentences.
POSTS = [
    ("posts/part-1-quiz.md", "Part 1",
     "A strong model writes an open-book quiz on a paper and weaker models take it. Rewriting papers to be hard to "
     "follow lowered quiz scores only slightly, while the same models rated the rewrites as far harder to read."),
    ("posts/part-2-planted-errors.md", "Part 2",
     "We plant an error in a paper and in a garbled version of it, and ask a model to find it. Most readers found as "
     "many errors in garbled text as in the original; only the strongest, GPT-5.5, found clearly fewer, and asked "
     "about every paragraph, readers flagged more correct text when it was garbled."),
]

INTRO = """# Can a model test whether a research paper is legible to people?

AI systems are starting to write a large share of AI safety research, and people can only oversee work they can
follow. One fix is to have a model translate a hard paper into a readable one, but then readers trust a translation
nobody has audited. We would rather require the paper itself to be legible, and check that automatically, at the scale
at which AI will produce papers.

Averroes Labs is trying to build that check. Each post below tries one version of it, using weak models as stand-in
readers and comparing each paper with a version rewritten to be hard to follow while keeping every fact. A useful test
should score the hard version lower, the way a person would struggle with it.

## Why test with a reader, not a rater?

The cheaper options are to ask a model how legible a paper is, to train a model to rate it, or to find a direction
inside a model that tracks readability. Part 1 tried the first. Models rated our garbled rewrites as far harder to
read, yet their ratings had little to do with how well the same models understood the text, and a fluent paper with
its key results quietly removed fooled most of them. A rating judges how a text looks, not whether a reader can use it.

Training a rater or fitting a probe needs labels of how legible research papers are to people, which do not exist at
scale. Existing readability labels grade school texts by difficulty, so a model trained on them would mostly learn
surface difficulty. A rater is also something AI writers can be optimized against, and polish is exactly what fooled
the ratings in Part 1. A reader that has to answer questions about the paper, or catch an error planted in it, can
only succeed by using what the paper says, which is harder to fake. We have not tested trained raters or probes
ourselves. Once human legibility labels for papers exist, they would calibrate any of these approaches, and comparing
them would be worth doing.

## Posts

{posts}
"""


def title(text):
    return re.search(r"^# (.+)$", text, re.M).group(1).strip()


def main():
    items = "\n".join(f"**[{label}: {title((ROOT / path).read_text())}]({path})**  \n{desc}\n"
                      for path, label, desc in POSTS)
    about = (ROOT / "docs/about-repo.md").read_text().replace("](../", "](")
    (ROOT / "README.md").write_text(INTRO.format(posts=items).rstrip() + "\n\n---\n\n" + about.rstrip() + "\n")
    print("wrote README.md:", len(POSTS), "posts")


if __name__ == "__main__":
    main()
