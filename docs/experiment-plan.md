# Experiment plan (first pilot)

**Superseded for the next run by `docs/plan-v2.md`.** Kept as the record of the pilot.

Everything here may change. `docs/goals.md` is the fixed part. Record changes and reasons in `docs/decisions.md`.

## Scope

No human baselines for now (Jason's decision, 2026-11-02). This pilot looks for initial signal on the other properties.

It cannot show that scores track human understanding. It can show that the score is stable and responds to deliberate changes in clarity. Treat it as a pilot that justifies a human study. Do not write up any result as evidence of human validity.

The "clarified" and "compressed" versions are model-written and labelled by us, so a good discrimination result may partly reflect that labelling. Say so in any write-up.

## Starting design

- An independent smart model writes an open-book quiz on the paper's key claims. See `docs/quiz-design.md`.
- A smart and a dumb model each take the quiz without the paper and with it (four conditions). The dumb reader's score with the paper is the audit result. Pass criterion to be chosen; see below.
- About 12 papers in four versions: original, clarified, compressed, and rewritten to game the quiz. Selected by the rule in `docs/paper-selection.md`.
- Partly obfuscated versions of the same papers test whether a measure notices a reader losing the thread. See `docs/measure-comparison-plan.md`.
- A few two-paper chains test composition.
- Author, quiz writer and reader should come from different model families (Laurito et al. found LLMs prefer LLM-written text).
- Questions test transfer rather than recall. Avoid rewarding brevity (Roytburg et al.).

## Exploratory datasets

Jason's decision, 2026-11-02: run these to inform the next research steps. They need not appear in the eventual paper. Treat their results as exploratory, keep them separate from the main pilot, and do not use them to tune the main pilot after its bars are committed.

| Stage | Content | What it tests | Check first |
| --- | --- | --- | --- |
| Before the main pilot | [OneStopQA](https://huggingface.co/datasets/onestop_qa/blob/main/README.md): articles at 3 human-adapted reading levels, with human-written questions (CC BY-SA 4.0) | Whether the dumb model's score follows human-made difficulty, and how generated quizzes compare with human-written ones | The texts may be easy for models, so scores may be high at every level. A flat result does not rule out the method, but may call for a weaker reader |
| Before the main pilot | [eLife articles with editor-written digests](https://arxiv.org/abs/2210.09932) | Whether the same result, written clearly by people, is easier for the dumb model | The dataset files' licence. Quiz only claims both versions cover, since digests omit content |
| After the main pilot | AI-written papers, such as from Agents4Science or aiXiv | Whether the method works on the writing the audit is for | Licences and availability, neither checked yet |

Start with a small sample of each. Commit the sample rule before drawing.

## Properties to measure

| Property | Test | Good signal |
| --- | --- | --- |
| Discrimination | Original, clarified, compressed and gamed versions of each paper | Scores rank clarified above original above compressed |
| Whole-paper understanding | Claim and section scores on clear and partly obfuscated versions (`docs/measure-comparison-plan.md`) | The measure fails partly obfuscated papers and shows where the reader lost the thread |
| Learning gain (supporting) | Reader takes the quiz before and after reading | Gain comes from the paper, not prior knowledge. Gain does not show the whole paper was followed, and can be low for clear papers on known material |
| Reliability | Repeat with new quizzes and new runs | The same paper gets a similar score each time |
| Reader levels | Readers of different strength | Weaker readers score lower, and the ranking of papers stays stable |
| Quiz quality | Different models write quizzes on the same paper | Their scores agree, and a shared wrong answer key is rare |
| Gaming | Rewrite papers to raise the score, without seeing the items | The score rises without the paper getting clearer, which measures the vulnerability |
| Composition | Two-paper chains | Reader fails the second paper alone and passes after reading the first |

## Models

Open-weight models serve the experiments. Pick specific models and GPU sizes when the run starts, since none have been checked yet.

| Role | Source | Note |
| --- | --- | --- |
| Reader | Open-weight, pinned | Pinned weights keep the reader fixed. Use several strengths. Training cutoff should be before 2026-07-01, the start of the paper window |
| Quiz writer | Open-weight baseline; frontier API if a key is provided | Must understand the paper better than the reader |
| Rewriter (clarify, compress) | Open-weight | Test material only |
| Gaming attacker | Open-weight; frontier API if a key is provided | With only a weak attacker, report the gaming result as a lower bound on vulnerability |
| Research agent | Claude Code | Runs on Anthropic's models |

Jason has not yet decided whether to provide a frontier API key. If open-weight and frontier quiz writers give similar scores, the API can be dropped.

## Bars

Not set. Propose pass/fail bars in `docs/preregistration.md` and commit them before collecting the results they judge. Jason signs off on them; see the rules in `CLAUDE.md`.
