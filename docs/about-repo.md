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

To replay model calls from the cache at no cost, `gunzip -k` the `calls*.jsonl.gz` files first; each script's docstring gives its commands (for example `python3 scripts/v4/exp12_paragraphs.py --scores`). New calls need an OpenAI key in `.env` (`OPENAI_API_KEY`). `llm.py` stops before spending more than `V4_SPEND_CAP` dollars (default 2,500) as counted in `data/v4/spend.jsonl`, which a fresh copy starts empty. Part 2 cost about $1,500 at OpenAI's list prices.

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
