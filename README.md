# Can a model test whether a research paper is legible to people?

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

**[Part 1: Can a Quiz Test Whether a Paper Is Legible to Humans?](posts/part-1-quiz.md)**  
A strong model writes an open-book quiz on a paper and weaker models take it. Rewriting papers to be hard to follow lowered quiz scores only slightly, while the same models rated the rewrites as far harder to read.

**[Part 2: Can a Reader That Catches Planted Errors Tell Whether a Paper Is Legible?](posts/part-2-planted-errors.md)**  
We plant an error in a paper and in a garbled version of it, and ask a model to find it. Most readers found as many errors in garbled text as in the original; only the strongest, GPT-5.5, found clearly fewer, and asked about every paragraph, readers flagged more correct text when it was garbled.

---

## About this repository

This repository holds the code, prompts, model outputs and analysis behind the posts, so anyone can check a number or rerun a step. It is a cleaned copy of our working repository. The work was run by an AI research agent (Claude) with Jason Fantl as PI, which is why the logs are written in the first person and some files mention internal notes (`STATE.md`, `INBOX.md`, `CLAUDE.md`) that are not included here.

### Where things are

| Path | Contents |
|---|---|
| `docs/report-v2.md` | The full technical report, with every number, interval and follow-up |
| `docs/preregistration-v2.md` | Predictions and analysis plans, each committed before its results were collected |
| `docs/plan-v2.md`, `docs/experiment-plan.md`, `docs/quiz-design.md`, `docs/goals.md` | Goal and design |
| `docs/decisions.md`, `docs/run-log.md` | Every design decision with its reason, and every run, failure and rerun |
| `docs/pilot-report.md`, `docs/preregistration.md` | The earlier 6-paper pilot that led to plan v2 |
| `docs/blog/` | Figures in this post (`scripts/blog_figures.py`) |
| `docs/figures/` | Report figures, light and dark (`scripts/quiz/make_figures.py`) |
| `scripts/quiz/` | The pipeline: `prompts.py` (all prompts), `versions.py` (rewrites), `write_v2.py`, `filter_v2.py`, `takes_v2.py`, `judge_v2.py`, `gap_v2.py`, `gap_requiz.py`, `gap_imp2.py`, analysis in `analyze_v2.py` and `ci_v2.py` |
| `scripts/onestop/run_onestop.py` | The OneStopQA comparison |
| `scripts/select_papers.py`, `scripts/paper_text.py` | Paper selection rule and arXiv HTML to text conversion |
| `data/selection/` | The sampling frame and the selection log |
| `data/papers/` | Paper texts and metadata; see `data/papers/ATTRIBUTION.md` |
| `data/runs/v2-<paper>/` | Per paper: rewritten versions, the quiz, filter results, every answer and grade, ratings, and `calls.jsonl` with every model call and its cost |
| `data/runs/v2rem-<paper>/` | Follow-up 2, the quiz written on the key-results-removed version |
| `data/onestop/` | OneStopQA human data (copied from the original release) and our model answers |
| `data/analysis/` | The analysis outputs every reported number comes from |
| `reports/`, `research_notes/` | Literature reviews done for the design and the related-work section |

### Rerunning

You need Python 3.12 and an [OpenRouter](https://openrouter.ai) key. Put the key in `.env` (copy `.env.example`), then `pip install -r requirements.txt`.

Every model call is cached in a `calls.jsonl` file keyed by its full request, so rerunning a script with the data here replays the cached replies at no cost and rebuilds every analysis file. Delete a run directory's `calls.jsonl` to make fresh calls. Fresh calls will not match exactly, since the models are served through OpenRouter and their weights and providers are not pinned.

```
python scripts/quiz/run_v2.py <paper> [<paper> ...]   # versions, quiz, filters and takes for the main study
python scripts/quiz/judge_v2.py rate                 # legibility ratings
python scripts/quiz/gap_v2.py run                    # hidden-gap versions, takes and ratings
python scripts/quiz/gap_requiz.py run                # Follow-up 2
python scripts/quiz/gap_imp2.py run                  # Follow-up 3
python scripts/onestop/run_onestop.py run            # OneStopQA
python scripts/quiz/analyze_v2.py                    # pooled analysis -> data/analysis/
python scripts/quiz/ci_v2.py                         # intervals for the figures
python scripts/blog_figures.py                       # figures in this post
```

Each script's docstring gives its options and outputs. The full study cost about $650 in API calls, mostly for writing and filtering quizzes.

### Licences

Code is under the MIT licence (`LICENSE`). Our text, quizzes, model outputs and analysis are under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) (`LICENSE-DATA.md`). The papers belong to their authors and are shared under their own Creative Commons licences, listed in `data/papers/ATTRIBUTION.md`; the rewritten versions deliberately distort them and are not the authors' work. The OneStopQA data is © Berzak, Malmaud and Levy, CC BY-SA 4.0 (`data/onestop/human/`).
