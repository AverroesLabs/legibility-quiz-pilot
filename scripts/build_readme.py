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
inside a model that tracks readability. Each of these predicts legibility from how a text looks. A reader test
measures it, by having a model try to use the paper. That difference matters most where this test is meant to work.

**Legibility is about use.** A paper is legible if a reader can follow its argument and check its claims. A reader test
does that task, and a rater guesses at the outcome. Part 1 asked models for ratings directly. They rated our garbled
rewrites as far harder to read, yet their ratings had little to do with how well the same models understood the text.

**The writers will be optimized against the test.** A rater is a fixed function, and a writer trained against it can
satisfy it without becoming legible. In Part 1 a fluent paper with its key results quietly removed fooled most of the
ratings. A test in which a reader must extract the paper's content can only be passed by making that content
extractable. This is why Kirchner et al.'s prover-verifier games made model outputs easier for people to check: the
writer was trained against a reader that had to verify its work, not against a judge of style.

**AI-written papers will be hard to follow in new ways.** Unfamiliar notation, compressed reasoning, or shorthand only
another model reads would not appear in any rater's training data. A reader fails on an unusable text whatever made it
unusable.

**Oversight needs mistakes to be catchable.** People read AI research to find what is wrong with it. Planting an error
and checking whether a reader finds it measures that directly, while a rater can call a paper clear that hides an
error. Comparing two versions of the same paper also separates how it is written from how hard its ideas are, which a
rater mixes together, and a failed question or missed error shows the writer where the paper loses its reader.

A reader test costs more to run, and our weak readers so far respond to bad writing much less than people likely do,
which the posts below report. If a reader test can be calibrated against people, its results would also supply the
labels that a cheap rater or probe lacks today. We have not tested trained raters or probes ourselves.

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
