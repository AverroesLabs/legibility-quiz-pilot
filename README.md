# Can a model test whether a research paper is legible to people?

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

**[Part 1: Can a Quiz Test Whether a Paper Is Legible to Humans?](posts/part-1-quiz.md)**  
A strong model writes an open-book quiz on a paper and weaker models take it. Rewriting papers to be hard to follow lowered quiz scores only slightly, while the same models rated the rewrites as far harder to read.

**[Part 2: Can Planted Errors Test Whether a Paper Is Legible to Humans?](posts/part-2-planted-errors.md)**  
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

To rerun Part 1, you need Python 3.12 and an [OpenRouter](https://openrouter.ai) key. Copy `.env.example` to `.env`, add the key, then run `pip install -r requirements.txt`.

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

### Part 2: planted errors

Part 2 used OpenAI models only. Its code is in `scripts/v4/` and its data in `data/v4/`, named after the project's fourth version.

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
uv run --with matplotlib python scripts/blog_part2_setup.py                            # Figure 1, the setup
uv run --with matplotlib --with requests python scripts/blog_part2_final.py            # Figures 2 to 4
uv run --with matplotlib --with requests python scripts/blog_part2_calibration_map.py  # Figure 5
uv run --with matplotlib --with requests python scripts/blog_part2_levers.py           # Figure 6
```

To rerun the experiments, install the requirements (`pip install -r requirements.txt`) and put an OpenAI key in `.env` as `OPENAI_API_KEY`. Every call goes through `scripts/v4/llm.py`, which caches each reply. If you first run `gunzip -k` on the `calls*.jsonl.gz` files, the steps below replay our cached replies at no cost. Without the caches they make fresh calls, which will differ somewhat from ours and cost about $1,500 in all. `llm.py` refuses to spend more than `V4_SPEND_CAP` dollars, 2,500 by default.

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

# Figure 6's note sizes
for r in gpt55-A600 gpt55-A1200; do python3 scripts/v4/err9.py read $r $FIRST $REVIEWED; done
```

Then rebuild the figures with the three commands above. `docs/part2-agent-report.md` gives the function behind every number, and each script's docstring lists its other options, such as the preregistered scores (`err9.py --scores`, `exp12_paragraphs.py --scores`, `exp11_whole.py`).

### Licences

Code is under the MIT licence (`LICENSE`). Our text, quizzes, model outputs and analysis are under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) (`LICENSE-DATA.md`). The papers belong to their authors and are shared under their own Creative Commons licences (Dropout under JMLR's CC-BY policy), listed in `data/papers/ATTRIBUTION.md`. The rewritten versions deliberately distort them and are not the authors' work. The OneStopQA data is © Berzak, Malmaud and Levy, CC BY-SA 4.0 (`data/onestop/human/`).
