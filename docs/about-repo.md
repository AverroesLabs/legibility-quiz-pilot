## About this repository

This repository holds the code, prompts, model outputs and analysis behind the posts above, so anyone can check a number or rerun a step. It is a cleaned copy of our working repository. The work was run by an AI research agent (Claude) with Jason Fantl as PI, which is why the logs are written in the first person and some files mention internal notes (`STATE.md`, `INBOX.md`, `CLAUDE.md`) that are not included here.

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
