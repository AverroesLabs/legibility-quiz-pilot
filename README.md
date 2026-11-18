# Can a model test whether a research paper is legible to people?

AI systems are starting to write a large share of AI safety research, and people can only oversee work they can
follow. One fix is to have a model translate a hard paper into a readable one, but then readers trust a translation
nobody has audited. We would rather require the paper itself to be legible, and check that automatically, at the scale
at which AI will produce papers.

Averroes Labs is trying to build that check. Each post below tries one version of it, using weak models as stand-in
readers and comparing each paper with a version rewritten to be hard to follow while keeping every fact. A useful test
should score the hard version lower, the way a person would struggle with it.

## Why test by using the paper, not by rating it?

The cheaper options are to ask a model how legible a paper is, to train a model to rate it, or to find a direction
inside a model that tracks readability. Each gives a score from one look at the text. Our tests instead make a model
use the paper, by answering questions that need its content or by finding an error planted in it, and score what the
model could extract. The line that matters is not reader versus rater. A rater that had to show what it got out of the
paper before scoring it would be a use test too. What we doubt is any score that never shows the paper was used.

**Legibility is about use.** A paper is legible if a reader can follow its argument and check its claims. A reader test
does that task, and a rater guesses at the outcome. Part 1 asked models for ratings directly. They rated our garbled
rewrites as far harder to read, yet their ratings had little to do with how well the same models understood the text.

**The writers will be optimized against the test.** Any fixed test can be gamed. What differs is the gap between what
the test rewards and what we want. A rater rewards looking readable, and a writer can satisfy it while hiding content.
In Part 1 a fluent paper with its key results quietly removed fooled most of the ratings. A reader test can only be
passed by making the content extractable, because the questions or planted errors are drawn after the paper is written.
That narrows the gap to the difference between text this model can use and text people can use, but it does not close
it. A writer optimized against a model reader can drift toward text only machines read well, and our own readers
barely noticed garbling that would stop a person. Kirchner et al. saw the same risk in prover-verifier games, where
sneaky provers learned to fool the verifier. This remaining gap is why the reader has to be calibrated against people,
and why fresh questions, several readers and occasional human checks matter.

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

**[Part 2: Can Planted Errors Test Whether a Paper Is Legible?](posts/part-2-planted-errors.md)**  
We plant an error in a paper and in a garbled version of it, and ask a model to find it. Only the strongest model, GPT-5.5, found clearly fewer errors in the garbled version, and asked about every paragraph, every model flagged more correct text when it was garbled.

---

## About this repository

This repository holds the code, prompts, model outputs and analysis behind the posts, so anyone can check a number or rerun a step. It is a cleaned copy of our working repository. The work was run by an AI research agent (Claude) with Jason Fantl as PI, which is why the logs are written in the first person and some files mention internal notes (`STATE.md`, `INBOX.md`, `CLAUDE.md`) that are not included here.

### Part 1: the quiz

| Path | Contents |
|---|---|
| `docs/report-v2.md` | The full technical report, with every number, interval and follow-up |
| `docs/preregistration-v2.md` | Predictions and analysis plans, each committed before its results were collected |
| `docs/plan-v2.md`, `docs/experiment-plan.md`, `docs/quiz-design.md`, `docs/goals.md` | Goal and design |
| `docs/decisions.md`, `docs/run-log.md` | Every design decision with its reason, and every run, failure and rerun |
| `docs/pilot-report.md`, `docs/preregistration.md` | The earlier 6-paper pilot that led to plan v2 |
| `docs/blog/` | Figures in the Part 1 post (`scripts/blog_figures.py`) |
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

### Part 2: planted errors

Part 2 was run on OpenAI models only, with its own code under `scripts/v4/` and data under `data/v4/` ("v4" was the working name).

| Path | Contents |
|---|---|
| `docs/part2-agent-report.md` | Every number in the Part 2 post with the file and function it comes from, written for AI agents. Start here to check a result |
| `docs/report-v4.md`, `docs/report-v4-details.md` | The full technical report: every experiment (0 to 12), including those not in the post, and every unexpected result |
| `docs/plan-v4.md` | Design and predictions, each section committed before its data |
| `docs/decisions.md`, `docs/run-log.md` | Decisions and runs, continued from Part 1 (entries from 2026-11-12 on) |
| `docs/blog/part-2/` | Figures in the Part 2 post |
| `docs/figures/v4-*.png` | Report figures |
| `scripts/v4/llm.py` | Every OpenAI call: cache, cost ledger and spending cap |
| `scripts/v4/obf_max.py`, `scripts/v4/clarify.py` | The garbled and clarified versions |
| `scripts/v4/err9.py` | Error design, planting and the strong-reader filter; experiment 9 (one section at a time) |
| `scripts/v4/exp11_whole.py`, `scripts/v4/exp12_paragraphs.py` | Experiment 11 (whole paper) and experiment 12 (every paragraph) |
| `scripts/blog_part2_*.py`, `scripts/averroes_figstyle.py` | Figures in the Part 2 post |
| `data/v4/versions/<paper>/` | Original, garbled (`obf-max`) and clarified (`clar-*`) text of each paper |
| `data/v4/err9/` | The planted errors (`errors.jsonl`) and every review by every reader in experiments 9 and 11 |
| `data/v4/exp12/` | Experiment 12: every paragraph question and answer |
| `data/v4/exp0-*` to `data/v4/exp8-*` | Earlier experiments, reported in `docs/report-v4.md` but not in the post |
| `data/papers/` | The six peer-reviewed papers and Dropout were added; see `data/papers/ATTRIBUTION.md` |

Raw model calls (`calls*.jsonl`, `reader-calls*.jsonl`) and the spend ledgers (`spend*.jsonl`) are stored gzipped. The Part 2 figures need only the uncompressed files, so they rebuild without a key and without unzipping:

```
uv run --with matplotlib --with requests python scripts/blog_part2_final.py            # Figures 1 to 3
uv run --with matplotlib --with requests python scripts/blog_part2_calibration_map.py  # Figure 4
uv run --with matplotlib --with requests python scripts/blog_part2_levers.py           # Figure 5
```

To rerun the experiments, install the requirements (`pip install -r requirements.txt`) and put an OpenAI key in `.env` (`OPENAI_API_KEY`). Every call goes through `scripts/v4/llm.py`, which caches each reply by its full request. Run `gunzip -k` on the `calls*.jsonl.gz` files first and the steps below replay the cached replies at no cost. Without the caches they make fresh calls, which will differ somewhat from ours. `llm.py` stops before spending more than `V4_SPEND_CAP` dollars (default 2,500), counted in `data/v4/spend.jsonl`, which a fresh copy starts empty. Part 2 cost about $1,500 at OpenAI's list prices.

```
FIRST="2609.30094v2 2608.13754v1 2609.02893v1 jmlr-2014-dropout"
REVIEWED="2601.02896v4 2603.01214v2 2509.22621v3 2509.11379v3 2510.17247v2 2506.22982v1"

# The garbled and clarified versions of each paper
python3 scripts/v4/obf_max.py $FIRST $REVIEWED
python3 scripts/v4/clarify.py $FIRST $REVIEWED --prompts explainer

# Inconsistencies already in the originals, then the planted errors and the strong-reader filter
python3 scripts/v4/err9.py inventory jmlr-2014-dropout $REVIEWED
python3 scripts/v4/err9.py construct $FIRST $REVIEWED

# Result 1, whole paper (experiment 11)
for r in w11-gpt55-low w11-mini-high w11-luna-low w11-mini54-med; do python3 scripts/v4/err9.py read $r $FIRST $REVIEWED; done

# Result 2, every paragraph (experiment 12)
for r in p12-gpt55-low p12-mini-high p12-luna-low p12-mini54-med; do python3 scripts/v4/exp12_paragraphs.py read $r $FIRST $REVIEWED; done

# Result 3, one section at a time with 300-word notes (experiment 9); the readers' names differ by paper set
for r in gpt55-A mini-high-A luna-low-A mini54-A; do python3 scripts/v4/err9.py read $r $FIRST; done
for r in gpt55-A-anchors gpt55-A-anchors-clar mini-high-A-anchors luna-low-A-anchors mini54-A-anchors; do python3 scripts/v4/err9.py read $r $REVIEWED; done

# Figure 5's note sizes
for r in gpt55-A600 gpt55-A1200; do python3 scripts/v4/err9.py read $r $FIRST $REVIEWED; done
```

Then rebuild the figures with the three commands above. `docs/part2-agent-report.md` gives the function behind every number, and each script's docstring lists its other options, such as the preregistered scores (`err9.py --scores`, `exp12_paragraphs.py --scores`, `exp11_whole.py`).

### Rerunning Part 1

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
python scripts/blog_figures.py                       # figures in the Part 1 post
```

Each script's docstring gives its options and outputs. The full study cost about $650 in API calls, mostly for writing and filtering quizzes.

### Licences

Code is under the MIT licence (`LICENSE`). Our text, quizzes, model outputs and analysis are under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) (`LICENSE-DATA.md`). The papers belong to their authors and are shared under their own Creative Commons licences (Dropout under JMLR's CC-BY policy), listed in `data/papers/ATTRIBUTION.md`; the rewritten versions deliberately distort them and are not the authors' work. The OneStopQA data is © Berzak, Malmaud and Levy, CC BY-SA 4.0 (`data/onestop/human/`).
