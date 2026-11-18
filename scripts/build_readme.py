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
     "We plant an error in a paper and in a garbled version of it, and ask a model to find it. Only the strongest "
     "model, GPT-5.5, found clearly fewer errors in the garbled version, and asked about every paragraph, every model "
     "flagged more correct text when it was garbled."),
]

INTRO = """# Can a model test whether a research paper is legible to people?

AI systems are starting to write a large share of AI safety research, and people can only oversee work they can
follow. A model could translate each hard paper into a readable one, but then readers would trust a translation nobody
has checked. We would rather require the paper itself to be legible, and check that automatically, because AI will
produce papers faster than people can review them.

Averroes Labs is building that check, and each post below tries one version of it. A weaker model stands in for the
reader, and we compare how it does on a paper and on a version rewritten to be hard to follow while keeping every fact.
A useful test should score the hard version lower, as a person would.

## Why make the model use the paper

The cheaper options are to ask a model how legible a paper is, to train a model to rate legibility, or to find a
direction inside a model that tracks readability. Each gives a score from one look at the text. Our tests instead make
a model use the paper, either by answering questions about its content or by finding an error planted in it. A rater
would be fine if it first had to show what it got out of the paper. What we doubt is a score that gives no sign the
paper was used.

We prefer use for four reasons. First, legibility means a reader can follow a paper's argument and check its claims,
and a test that makes the model do this measures it directly. In Part 1, models rated our garbled rewrites as far
harder to read, yet their ratings had little to do with how well the same models understood the text.

Second, writers will learn to pass whatever test we use. A rating can be passed by text that only looks readable, and
in Part 1 a fluent paper with its key results quietly removed still got good ratings. A reader test is harder to fool,
because the questions or planted errors are chosen after the paper is written, so the paper has to contain content the
reader can actually use. It can still be gamed by writing that a model can use and a person cannot. Our readers already
barely noticed garbling that would stop a person, so a reader test has to be checked against people from time to time.

Third, AI-written papers will be hard to follow in new ways, such as unfamiliar notation or shorthand that only another
model reads. A rater trained on today's papers has never seen these, but a reader fails on an unusable text whatever
made it unusable.

Fourth, people read AI research to find what is wrong with it. Planting an error and checking whether a reader finds it
tests that directly, while a rater can call a paper clear even when it hides an error.

A reader test costs more to run, and so far our readers respond to bad writing much less than people likely do, as the
posts report. If it can be matched to people, a reader test could also supply the training labels that a cheap rater
or probe needs. We have not tested trained raters or probes ourselves.

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
