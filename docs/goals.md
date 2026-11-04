# Legibility Quiz: Research Goals

2026-11-02 · Jason Fantl

## Summary

We want to find out whether a quiz taken by a standardized reader model can measure whether people can follow a research paper. We will publish the answer in a paper and, if the method holds up, release it as a tool.

A quiz normally tests a reader against fixed material. We reverse that: fix the reader and test the material. If a reader model of about human ability learns enough from a paper to pass a quiz on it, the paper counts as legible to that level of reader.

This doc records the goal and the reasoning behind it. Experiment details will change as we learn what works, and they belong elsewhere. This work is the legibility part of Averroes Labs.

## The problem

AI will soon do most AI safety research, and people cannot oversee or steer work they cannot follow. AI reasoning is drifting from readable English, optimizing for correctness makes outputs harder for humans to check, and new sub-fields grow jargon faster than people can learn it. The result is weaker field epistemics and a risk of gradual disempowerment.

Translating an illegible paper with an LLM is the obvious fix, and it is often good enough. It fails as the basis for trust, for four reasons:

- **Audits do not transfer.** A paper may say a method worked on three of four benchmarks while its translation says it worked on benchmarks. The audit passes and the reader gets a false claim.
- **Readers cannot check it.** Anyone who needs the translation cannot read the original to compare them.
- **It can be attacked.** A paper could be written so that the original passes every audit while its translation misleads.
- **Re-auditing does not scale.** Readers translate privately and each gets a different text, so audits cannot be enforced and their cost grows with every reader.

It is simpler to require the legible version up front, audit that, and publish it as the paper. That requires a way to measure legibility without a person reading every paper. As of our October 2026 search, we have not found a gate that requires a weak reader to learn a paper's key claims. The closest work is KnowledgeGain, greCAPTCHA, and Comprehension Audits.

## The research question

**Does a standardized reader model's quiz result track whether people can follow a paper, closely enough to use as a gate?**

Answering that means answering six narrower questions:

1. **Human validity.** Do the reader model's results agree with what human readers can follow? The method depends on this, because the reader is only a stand-in for people.
2. **Discrimination.** Does the result separate legible writing from illegible writing, including different versions of the same content? It should measure understanding, not reward brevity or a particular style.
3. **Quiz quality.** Do generated quizzes test what matters in a paper, with correct answers, rather than surface recall?
4. **Resistance to gaming.** Can a paper be written to pass while people still cannot follow it? Any check that gates funding or publication will be optimized against.
5. **Composition.** Can understanding build across papers, so that a hard result is reachable through a chain of readable steps? If so, the gate limits the size of each step, not how deep a field can go.
6. **Reader levels.** Can different reader models represent different audiences, such as academic background or language proficiency, so that a result says who a paper is legible to?

The work does not ask whether a paper is correct. A paper can be clear and wrong, or sabotaged and easy to follow, so the quiz is not a sabotage defense. Soundness and sabotage belong to separate audits.

## What we will produce

**The paper** reports whether the method works, measured against human readers and against deliberate attempts to game it. We publish either way. A clear negative result, showing where reader models and people diverge, would still tell the field not to rely on this kind of gate.

**The tool** takes a paper and reports how legible it is to a chosen level of reader, with no person needing to read the paper. Three things should hold for it:

- **Models only per paper.** Each check costs compute, not reviewer time, so capacity grows with AI output. Humans still calibrate the method and set the bars, once per method rather than once per paper.
- **Comparable over time.** The reader stays standardized, so a result today means the same thing later, and the field can see whether AI-written work is drifting away from human readers.
- **Usable by decision-makers.** Funders and venues can require it and choose the reader level they want, and authors have a reason to write for people. It follows the Averroes approach: the protocol is an open standard, while the quiz items stay private.

## What success looks like

We succeed if we can give a defensible answer to the research question, judged against bars set before we see results. If the answer is yes, there is one more test: the gate passes papers people can follow, rejects papers they cannot, resists gaming, and runs without anyone reading the paper. The bars themselves are not set yet.

## What stays fixed

The goal holds even if the current experiment design does not. If a method fails, we replace the method. We revisit this doc only if the goal itself changes.

| Fixed | Expected to change |
| --- | --- |
| Legibility means people can follow the paper itself, not a translation | How we generate, format, and score quizzes |
| A standardized reader stands in for people | Which models act as reader and quiz writer, and how we build them |
| Human readers are the ground truth | How many papers and people we test, and how we recruit them |
| The method must resist authors trying to game it | Which attacks we test and which defenses we use |
| We test the method before relying on it, against bars set in advance | The bars and the statistics |
| We publish the result whichever way it goes | Venue, timeline, and how the tool is packaged |

## Questions the goal leaves open

- **Quiz writer errors.** A quiz writer that misreads a paper should produce a wrong answer key, which the reader then fails. That fails safe for gating, but it penalizes honest papers, and a writer and reader who share a misreading could still pass.
- **Feedback versus secrecy.** Authors need to know why a paper failed, but quiz items stay private and the quiz should not become a training signal. How much a result can reveal is unresolved.
- **Reader drift.** A fixed reader gives comparable results over time, but calibration to people may age and fixes may break that comparability.
