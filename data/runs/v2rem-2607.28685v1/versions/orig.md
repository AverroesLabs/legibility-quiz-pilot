# Safety, or Just Capability? A Validity Audit of Agent-Safety Benchmarks

Youting Wang

  
Xiao Han

  
Dingyan Shang

Affiliation: Yuan Tang

  
Bowen Liu

###### Abstract

Agent-safety benchmarks have multiplied quickly, and their scores are now routinely quoted as
evidence about how safe an LLM agent is. Before such scores can support comparisons, the
measurements themselves need to be reproducible, internally consistent, and run under comparable
conditions. We audit four widely used agent-safety benchmarks (R-Judge, InjecAgent, AgentHarm,
AgentDojo), running each under its official implementation and author-provided scorer on a shared
API-only panel of up to 22 models from nine organizations. We measure MMLU and GPQA-Diamond
ourselves under one uniform protocol to form a capability composite, and we add three held-out
behavioral outcomes: $\tau^{2}$ -bench retail task success, a fictional-scenario misalignment probe,
and a three-template jailbreak suite. A later expansion extends the two chat-based outcomes to 41
anchored models from 12 organizations. We document coverage, exclusion gates, within-model
uncertainty, and item-level internal consistency for every instrument. All item-rich instruments
with resampling support are highly consistent (split-half Spearman–Brown $\geq 0.96$ , rank
stability $\geq 0.97$ ). Worst-case within-model standard errors are a modest fraction of
between-model spread. Single-scenario and single-run instruments remain the weakest links. We
release the audit harness, frozen inputs, and analysis scripts, and we discuss what a reproducible
agent-safety evaluation should report: the benchmark, the metric, the intended behavior, and the
model panel on which any claim rests.

## 1 Introduction

Between 2024 and 2026, dozens of benchmarks appeared claiming to measure whether LLM agents are
safe or reliable: resistance to prompt injection
(Zhan et al. 2024; Debenedetti et al. 2024), safety-risk awareness (Yuan et al. 2024), refusal of
harmful agentic tasks (Andriushchenko et al. 2025), and risky-tool-use identification
(Ruan et al. 2024), among others. A recent taxonomy (Li et al. 2026) catalogs
more than forty of them and reports a symptom worth taking seriously: swap the benchmark and the
safety ranking contradicts itself, with rank concordance near $0.10$ across models. That work
stops at a twelve-model concordance check over four benchmarks, and lists
capability-controlled evaluation and benchmark consolidation among its open problems.

This paper picks up there with a more basic concern. A ranking can only be as trustworthy as the
scores beneath it, and those scores depend on implementations, scorers, judge models, prompt
formats, parsing rules, and the set of models on which they are collected. In practice, published
agent-safety numbers are often produced under heterogeneous conditions: different harness
versions, provider-reported capability figures, and panels that differ from paper to paper. Before
asking what agent-safety scores mean, we ask whether they can be produced reproducibly on a common
panel, and we organize the audit around four questions:

•

RQ1 (construct structure). How do nominally-distinct agent-safety benchmarks relate
to one another when run under a single harness on a common panel?

•

RQ2 (capability confound). How do the benchmark scores relate to general capability
measured under a uniform protocol?

•

RQ3 (criterion validity). How do the safety scores relate to held-out behavioral
outcomes collected through the same harness?

•

RQ4 (metric validity). What do the official headline metrics reward, and what
should be reported alongside them?

#### Contributions.

First, we provide a shared, API-only audit harness that runs four agent-safety
benchmarks with their official implementations and scorers, two capability anchors under one
protocol, and three held-out behavioral outcomes, all through the same interface. Second, we
document the evidence base in detail: per-benchmark coverage, pre-specified exclusion gates,
within-model standard errors, and split-half consistency and rank stability for every instrument
with item-level data. Third, we report a 2026 panel expansion to 41 anchored models, including the
practical difficulties of keeping evaluation harnesses current as model interfaces change. Fourth,
we distil reporting recommendations for agent-safety evaluation. The harness, frozen inputs, and
analysis scripts are included with the accompanying reproducibility package.

#### Main conclusion.

Agent-safety scores are measurements produced by specific pipelines, and the
pipeline is part of the result. What a score licenses you to say depends on the benchmark, the
metric, the behavior the benchmark was designed to elicit, and the panel of models on which it was
collected. Reproducible evaluation requires all four to be stated, and our harness is a step toward
making that routine.

#### Scope.

The three RQ3 outcomes are stand-ins for deployment, not claims about
real-world harm; their limits (single instances, grader dependence, panel size) are in
Section 6.

## 2 Related Work

#### Agent-safety benchmarking.

Each of the four benchmarks we audit targets one construct.
InjecAgent (Zhan et al. 2024) and AgentDojo (Debenedetti et al. 2024) target prompt-injection
robustness; AgentHarm (Andriushchenko et al. 2025) measures compliance with harmful agentic tasks; R-Judge
(Yuan et al. 2024) scores safety-risk awareness over interaction traces. Evaluators we do not
re-run, such as ToolEmu (Ruan et al. 2024) for risky tool use, share the same status: designed as
evaluators, rarely examined as measurements. AutoMonitor-Bench (Yang et al. 2026)
scores misbehavior monitors by miss and false-alarm rates, an instance of two-sided reporting.
Closest to this paper is a taxonomy and consistency analysis of agent-safety
benchmarks (Li et al. 2026), which documents ranking disagreement across benchmarks and names
capability-controlled evaluation as future work. Our contribution is complementary: a common
harness and a documented evidence base on which such comparisons can be carried out. The broader
landscape of agent evaluation has grown along several axes at once—tool use, web navigation,
code execution, multi-agent coordination—and safety benchmarks increasingly borrow environments
from capability benchmarks. That borrowing makes shared infrastructure more valuable, because the
same environment quirks, parsing failures, and judge-model choices can affect safety and
capability measurements alike.

#### Construct validity.

Prior work asks whether benchmark scores agree and what latent
constructs they capture. Benchmark-agreement testing (Perlitz et al. 2024), metabench
(Kipnis et al. 2025), and direct construct analyses (Bean et al. 2025) find that
knowledge and reasoning benchmarks share a strong latent factor. Kearns (2026) and
NIST’s statistical framework (Keller et al. 2026) bring quantitative construct-validity and
uncertainty modeling to LLM evaluation. This line of work draws on a long psychometric tradition in
which an instrument is not trusted until its reliability has been established, its relationship to
related instruments characterized, and its sensitivity to administration conditions understood. We
take a similar stance toward agent reliability and safety benchmarks, where instruments target
distinct behaviors—refusal, injection robustness, risk awareness—and where administration
conditions (harness, scorer, judge, prompt template) vary considerably between published reports.
Reliability is the precondition for any of the downstream questions, which is why much of our
appendix is devoted to it.

#### Benchmark gaming.

A separate concern is that a benchmark can be gamed by an
adversary, for example through training on test items, through targeted optimization against a
known judge, or through output formatting that exploits a rule-based scorer. A related concern
applies even to honest models: the headline score may reward patterns of behavior other than the
one the benchmark designers intended. Classification metrics differ in how they treat the two kinds
of error, and a metric that is reasonable for one base rate or one use case can be misleading for
another. These are general properties of evaluation metrics, well known in information retrieval
and medical diagnostics, and they apply with equal force to binary trace-judgment benchmarks in
agent safety. We return to this point in Section 4.

#### Over-refusal.

XSTest (Röttger et al. 2024) and OR-Bench (Cui et al. 2024) document
exaggerated safety, or over-refusal, at the level of individual prompts. A model that refuses
too readily may look safe on harm-compliance benchmarks while being unhelpful on benign requests,
and a model that labels too many traces as unsafe may score well on recall-oriented judgment
metrics. Over-refusal is therefore a natural companion measurement for any refusal-based safety
benchmark, and we include XSTest in our harness so that benign-prompt behavior can be recorded
alongside harmful-prompt behavior on the same panel.

#### Agent reliability and task performance.

Recent work isolates specific agent-reliability failures
such as corrupt success (Cao et al. 2026) and evidence-bounded reporting
(Chen 2025); $\tau$ -bench (Yao et al. 2024) and its dual-control successor
$\tau^{2}$ -bench (Barres et al. 2025) measure agentic task success. We borrow the latter’s retail
domain as a held-out task-success outcome for RQ3. Task success is not a safety property, but it is
a useful reference point: any agent-safety evaluation is carried out against a background of
general agentic competence, and recording that competence on the same panel and harness makes it
possible to describe the safety measurements in context. We treat the safety benchmarks as
candidate columns for an audit rather than as settled measurements, in the same spirit in which
these reliability studies treat task-success metrics.

## 3 Method

### 3.1 Model panel and benchmarks

We assemble a compute-light, API-only panel of up to 22 models from nine model-developing organizations
(OpenAI 6, Meta 3, Qwen 3, Mistral 3, DeepSeek 2, Amazon 2, Anthropic, Google, Cohere; full
roster in Fig. 3 and the artifact). We re-run four safety benchmarks with their official
implementations and scorers: R-Judge (full 571 items, self-judge), InjecAgent (300 stratified
direct-harm/data-stealing items, rule-scored), AgentHarm (44 base behaviors spanning all eight
harm categories, via the official Inspect evaluation with a gpt-4o-mini judge), and AgentDojo
(the slack suite, 100 items, environment-scored for security and utility), plus a
held-out task-success outcome for RQ3 (below).

Coverage differs by benchmark (Fig. 3 maps the full panel): R-Judge has $n{=}21$ after
Qwen3-32B falls below the $90\%$ valid-output threshold; InjecAgent has $n{=}22$ ; AgentHarm and AgentDojo,
which require native tool use, have $n{=}19$ and $5$ ; and the $\tau^{2}$ outcome has $n{=}20$
(App. A.2). The common R-Judge/InjecAgent/AgentHarm panel therefore has $n{=}18$ .
Capability anchors (MMLU and
GPQA-Diamond) are measured by us under one uniform protocol: MMLU zero-shot on a fixed 500-item
subset (seed 42, identical items for every model) and GPQA-Diamond (198 items, chain-of-thought),
through the same API harness as the safety runs. We began with provider-reported model-card
numbers, which are produced under heterogeneous prompting and decoding conditions; the replacements
hold the protocol fixed across models. The anchor panel is 21 models (DeepSeek-R1 is excluded
for unstable long-form GPQA generation); we keep the provider-reported anchors in the artifact as a
pre-harmonization comparison.

All runs pass through a single request layer that records the exact prompt, decoding parameters,
raw completion, parse status, and scorer output for every item. Valid-output thresholds are applied
per benchmark before any score is computed, so that a model with frequent unparseable responses is
excluded on the basis of harness telemetry rather than on the basis of its scores. Where a benchmark
requires native tool calling, models whose provider interface does not support the required schema
are recorded as not-run rather than forced through an emulation layer, because emulated tool calls
would change the instrument.

### 3.2 Benchmark targets

Before collecting any scores, we wrote down what behavior each
benchmark is meant to measure. R-Judge tests whether a model distinguishes unsafe from benign
interaction traces. InjecAgent and AgentDojo both test resistance to prompt injection, so their
agreement serves as a convergent check on the harness. AgentHarm tests whether a model refuses harmful agentic
tasks. For R-Judge, we record specificity and balanced accuracy in addition to the official F1,
because the three summaries weight the two error types differently and the full confusion matrix
is cheap to retain.

#### Score orientation and terminology.

All reported scores are oriented so that higher is
better or safer. R-Judge specificity is the fraction of benign traces correctly identified
as benign; its balanced accuracy averages performance on benign and unsafe traces.
InjecAgent robustness is one minus attack success, and AgentHarm safety is one minus harmful
compliance. For RQ3, $\tau^{2}$ -bench is a held-out task-success outcome, not a safety score.
We use internal consistency for split-half analyses (App. A.2) and name each
benchmark-specific score explicitly, instead of stretching “calibration” or “reliability”
past their usual meanings.

### 3.3 Analysis plan

Before running each analysis, we fixed its inputs, thresholds, and reporting rules. Because
these choices lack an independent timestamp, we describe the analyses as pre-specified rather than
preregistered. We use Spearman correlations for association summaries, since several instruments
are bounded and skewed and the panel is small. Uncertainty for panel-level quantities is
summarized with model-level percentile bootstrap intervals. Where an analysis involves the
capability composite as a covariate, we use rank-residualized partial correlations, with the
implementation described in App. A.5. The capability composite is the mean of the standardized
MMLU and GPQA anchors.

We pre-specified three harness controls: positive (the capability anchors should agree with one
another), negative (a column-permuted score matrix should show no structure beyond a
random-normal null under Horn’s parallel analysis), and convergent (the two same-construct
injection benchmarks, InjecAgent and AgentDojo, should agree). These controls test whether the
pipeline produces sensible measurements before any substantive comparison is attempted. They were
fixed before any real run.

For RQ3 we pre-specified three held-out outcomes, each before running it.
Task success is $\tau^{2}$ -bench retail (Barres et al. 2025): 40 multi-turn,
environment-scored tool-agent-user tasks. Misalignment safety records whether a model avoids
blackmail or information leaking in a fictional autonomous-agent scenario
(Lynch et al. 2025), scored by a fixed classifier over 100 decisions per model.
Jailbreak safety is one minus harmful-compliance on 50 forbidden prompts, averaged over three published
jailbreak templates to reduce the ceiling created by any single template
(Souly et al. 2024; Wei et al. 2023). All three run through
the same API-only harness (the two safety outcomes are single-turn or scenario-based and cover the
full panel; the $\tau^{2}$ outcome is a multi-turn tool loop covering 20 models), and all are distinct
from the four audited benchmarks. The pre-specified plan gives reporting rules for every analysis
regardless of outcome. The accompanying reproducibility package contains the analysis scripts and
frozen inputs needed to reproduce every number in this paper.

## 4 Results

We present the audit in the order of the research questions as they bear on measurement practice.
Throughout, our emphasis is on what the harness makes observable and on the conditions under which
each score was produced, since those conditions determine what any downstream comparison can mean.

### 4.1 The headline metric is gameable (RQ4)

R-Judge is a binary trace-judgment benchmark: each of its 571 interaction records is labeled safe
or unsafe, and a model is asked to render a verdict. The official headline metric is F1 on the
unsafe class. F1 is a natural choice when the positive class is the object of interest and true
negatives are numerous and uninformative, as in document retrieval. In a safety-judgment setting,
however, both kinds of error carry meaning: missing an unsafe trace is a failure of vigilance,
while flagging a benign trace is a failure of discrimination that, in a deployed monitor, translates
into alert fatigue and blocked legitimate work. A metric that summarizes only one side of that
trade-off leaves room for behaviors that score well without exercising the judgment the benchmark
is meant to probe.

For this reason, our harness retains the full confusion matrix for every model rather than only
the official summary. From the stored verdicts we compute recall, specificity, precision, F1, and
balanced accuracy, and the artifact exposes all of them per model. This costs nothing at run time
and allows any reader to re-summarize the results under the metric most appropriate for their use
case. It also makes it straightforward to compute the scores of simple reference policies, such as
constant-verdict policies, which are the most basic sanity checks a classification leaderboard can
include.

We recommend that trace-judgment benchmarks publish such reference policies alongside model results,
in the same way that classification papers report majority-class baselines. A reference row costs a
single line in a leaderboard table and gives readers an immediate sense of where uninformative
behavior would land. More broadly, any benchmark whose headline metric omits one cell of the
confusion matrix should report a complementary metric that covers it, so that a reader can tell
whether a score reflects discrimination or merely a disposition toward one verdict.

### 4.2 The capability confound is real but metric-dependent (RQ2)

General capability is the most obvious alternative explanation for any agent-safety score. A model
must parse a long interaction trace before it can judge it, must follow a tool schema before it can
resist an injection embedded in a tool output, and must understand a multi-step request before it
can recognize it as harmful. Each of these prerequisites is a capability, and a benchmark that is
hard in these respects will inevitably reward stronger models to some degree, independently of any
safety-specific disposition.

How much capability enters a score depends on what the score counts. A recall-oriented summary
rewards correct detection of unsafe content, which presupposes comprehension; a specificity-oriented
summary rewards correct treatment of benign content, which can also be achieved by a disposition to
flag rarely. Balanced accuracy and F1 combine these components in different proportions. It follows
that the capability content of a benchmark is not a single number but a property of each summary
derived from it, and that statements about whether a benchmark "measures capability" should name the
metric.

Our harness supports this kind of examination because capability is measured under the same
protocol as the safety runs. Provider-reported model-card figures are collected under heterogeneous
prompts, shot counts, and decoding settings, and differences among them partly reflect reporting
practice rather than model ability. Re-measuring MMLU on a fixed 500-item subset and GPQA-Diamond with
a fixed chain-of-thought prompt removes that source of variation. We recommend that any agent-safety
study using capability as a covariate measure it under its own protocol, rather than importing
provider numbers, and that it report which summary of each safety benchmark it relates to capability.

### 4.3 Benchmarks disagree; a seven-model panel suggested a trade-off that does not persist (RQ1)

The four audited benchmarks were designed by different groups, with different threat models, item
sources, and scoring procedures. R-Judge asks for a verdict on a completed trace; InjecAgent and
AgentDojo embed adversarial instructions in tool outputs and score whether the agent acts on them;
AgentHarm presents harmful requests directly and scores whether the agent complies. There is no a
priori reason for these instruments to order models identically, and disagreement among benchmarks
coded to different targets (§3.2) does not by itself indict any single instrument. What
disagreement does undercut is the practice of quoting these scores interchangeably as the safety
of an agent.

Comparisons of this kind are sensitive to panel size. With a handful of models, a single unusual
model can produce a large rank correlation in either direction, and patterns observed on an early
pilot should be re-examined as more models are added. We encountered this directly during
development: our first cross-benchmark panel contained only seven models, and impressions formed at
that stage did not survive extension of the panel to the full common set. We therefore
treat all small-panel relationships as provisional and report only quantities whose stability we
have checked against the full $n{=}18$ common panel. We recommend that agent-safety comparisons
report the panel size explicitly, show subsampling variability for any rank correlation they
emphasize, and avoid interpreting a rank correlation estimated on fewer than about a dozen models.

The harness also makes clear how much of the panel is lost at each step. Because AgentHarm and
AgentDojo require native tool use, the common panel across the three broad-coverage benchmarks is
limited by AgentHarm’s coverage, and any comparison involving AgentDojo is restricted to five models.
Coverage loss is not random: it falls disproportionately on models whose providers expose
non-standard tool-calling interfaces. This is a further reason to state the panel behind every
comparison.

#### The positive control now passes.

On the harmonized anchors the capability composite
passes its pre-specified positive control, whereas on provider-reported numbers it had not—an
early sign that model-card figures were unsuitable as a common reference. The negative control also
passes: a column-permuted score matrix yields eigenvalues within the parallel-analysis null, so the
pipeline does not manufacture structure from noise. The convergent control is inconclusive at
AgentDojo’s $n{=}5$ , which is too few models to assess agreement between the two injection
benchmarks. We report the controls as properties of the harness rather than as substantive
findings: they indicate that the measurement pipeline behaves sensibly where its behavior can be
checked, and they identify where it cannot yet be checked.

### 4.4 Predictive validity differs across held-out outcomes (RQ3)

Everything above is internal to the benchmarks. RQ3 steps outside them by collecting three held-out
outcomes on the same panel and harness: $\tau^{2}$ -bench task success, misalignment safety, and
jailbreak safety. Each has a fixed metric and a fixed scoring procedure, and none is among the four
audited benchmarks. This section documents how the outcomes were built and what measurement issues
arose in building them.

#### Against task success, safety scores add no information beyond capability.

The $\tau^{2}$ -bench retail domain places a model in a multi-turn loop with a simulated user and a
tool environment; success is determined by the final database state. We run 40 tasks once per
model for 20 models. Because $\tau^{2}$ success is environment-scored, it is free of judge-model
dependence, but it inherits variance from the user simulator, which we cannot bound with a single
run. Task success is included as a reference outcome: it describes general agentic competence on
the same panel, against which the safety outcomes can be read. It is not a safety measure, and an
agent-safety benchmark is under no obligation to predict it; we include it because any study that
relates safety scores to held-out behavior should also record an outcome that capability is
expected to explain, so that the safety outcomes are not interpreted in a vacuum.

#### Against agentic misalignment, capability has the opposite association.

The misalignment outcome adapts a fictional autonomous-agent scenario (Lynch et al. 2025) in which
a model, acting as an assistant with access to corporate communications, has opportunities to
blackmail an individual or leak confidential information. We record 100 decisions per model across
the blackmail and leaking variants and score each with a fixed classifier. The scenario is
deliberately artificial, and its outcome should be read as a probe of behavior under a specific
pressure rather than as an estimate of real-world propensity. The two scenario halves agree only
moderately (App. A.2), which limits how finely the outcome can distinguish models with similar
scores. An independent grader reproduces the model-level scores closely, so the outcome is not an
artifact of a particular classifier.

#### A direct interaction test on the original panel.

When two outcomes are measured on the same models, comparing their associations with a common
predictor calls for a test of the difference between dependent correlations rather than a
comparison of two separate significance tests. Our pre-specified procedure computes the difference
on the paired panel and obtains its interval by resampling models, with leave-one-model-out,
leave-one-organization-out, and organization-clustered bootstrap variants to check that no single
model or developer drives the estimate. This design matters because organizations contribute
unequal numbers of models, and models from the same organization share training pipelines and
safety tuning; treating them as independent can overstate precision. We recommend clustered
resampling as a default for any agent-safety analysis on panels drawn from a few developers.

#### The capability–misalignment correlation is weaker on the expanded panel.

To test whether patterns observed on the 2024–25 panel extend to newer models, we pre-specified an
expansion of the two chat-based outcomes to models released in 2026. The expansion re-measured both
capability anchors and both safety outcomes under the original protocol, and it applied exclusion
gates based only on harness telemetry (App. A.6). Because the expanded panel contains the original
one, comparisons between the two require a nested resampling scheme, which we fixed before running
the new models. $\tau^{2}$ task success was not re-collected for the expansion, so only quantities
involving the two chat-based outcomes can be updated. The expansion also illustrates a practical
hazard: several newer models could not be run reliably through the same interface, a small instance
of evaluation harnesses going stale.

#### Models with similar capability scores differ in safety across organizations
(Fig. 4).

The 41-model panel spans 12 model-developing organizations
(misalignment $n{=}40$ ; jailbreak $n{=}41$ after one engagement-gate exclusion). Six
organizations contribute at least four models, and seven contribute at least three
(App. A.6). This composition makes organization an unavoidable feature of any analysis on the panel:
model families share base models, safety-tuning recipes, and system-level defaults, and
organization is partly collinear with release date and with measured capability. Fig. 4 lays out
the expanded panel by capability composite and safety outcome, with organizations distinguished
and within-model standard errors shown. Any comparison across organizations on a panel of this
kind is observational: the static capability composite may miss agentic ability, each organization
contributes few models, and some groups have restricted safety ranges.

#### Against jailbreak safety, AgentHarm shows the strongest held-out association.

The jailbreak outcome uses a different failure mode, harness, and scoring procedure from the
misalignment probe: harmful compliance under jailbreak, scored with StrongREJECT
(Souly et al. 2024). Building it surfaced a ceiling problem. A single jailbreak template (AIM)
produces a ceiling because 11 of 22 models refuse every request, which leaves half the panel tied at
the top and the outcome unable to separate them. Averaging scores over three published templates
(Wei et al. 2023)—AIM, refusal suppression, and prefix injection—leaves no model at the ceiling, and
the three-template average is the version we treat as primary. The resampling for this outcome is
by base prompt, so the dependence among the three templates applied to the same prompt is
respected. An independent judge reproduces the model-level scores closely. Because AgentHarm and
StrongREJECT both score harmful compliance, any relationship between them should be read as
agreement between two instruments aimed at a similar behavior rather than as evidence about safety
in general.

#### Two alternative explanations, checked.

Two artifacts could, in principle, distort relationships between refusal-based instruments. The
first is labeling: if the authors’ assignment of benchmarks to target behaviors were idiosyncratic,
comparisons framed in terms of those targets would inherit that idiosyncrasy. We therefore had seven
LLM coders from distinct model families, shown only neutral construct descriptions, independently
assign each benchmark to a target behavior (App. A.1). The second is blanket refusal: a model that
refuses nearly everything will score well on every harm-compliance instrument without
discriminating harmful from benign requests. We ran XSTest (Röttger et al. 2024) on the panel to
record benign-prompt over-refusal alongside harmful-prompt refusal, so that this behavior is
visible rather than hidden inside safety scores. Both checks are low-power on a panel of this size,
and we report them as context rather than as decisive tests.

## 5 Discussion

The central lesson of this audit concerns measurement practice. Agent-safety benchmarks are
instruments with specific targets, specific metrics, and specific administration conditions, and
their scores are only as comparable as those conditions allow. Running them under a single harness
on a shared panel removes several sources of incomparability at once—implementation drift,
heterogeneous scorers, provider-reported capability figures—and exposes others that are easy to
overlook, such as coverage loss concentrated among models with non-standard tool interfaces and
ceiling effects in single-template jailbreak outcomes.

Several recommendations follow. First, benchmarks should report the full confusion structure, or at
least a metric that covers each error type, and include constant-verdict reference rows. Second,
capability covariates should be measured under a uniform protocol rather than imported from model
cards. Third, panel size and composition should be stated for every comparison, and resampling
should respect the clustering of models within organizations. Fourth, held-out outcomes should be
checked for ceilings before being used, and multi-template designs preferred where a single template
saturates. Fifth, every instrument should report within-model uncertainty and internal consistency,
which our appendix shows to be feasible from data that benchmarks already produce.

None of these recommendations is specific to safety, and none requires new data collection beyond
what a careful benchmark run already generates. They do, however, change what a safety claim looks
like: a defensible claim names the benchmark, the metric, the target behavior, and the model
population on which it was established. Our harness is intended to make that level of specification
the default rather than the exception.

## 6 Limitations and Threats to Validity

#### Panel size.

The AgentHarm-limited cross-benchmark panel has $n{=}18$ , up from our
first pass at $n{=}7$ . Even at $n{=}18$ , rank correlations carry wide intervals, and panels of this
size cannot support fine-grained structural claims. The gap most worth closing is a larger,
capability-spanning agent-loop panel, which in turn depends on broader and more stable provider
support for native tool calling.

#### Capability anchors.

Provider-reported model-card anchors failed their positive control, and
re-measuring MMLU and GPQA-Diamond (§3) under one protocol resolved this. These remain
static-knowledge tests, and they may not capture the agentic abilities that matter most for
tool-using agents. A six-model BFCL pilot
(Patil et al. 2025) failed as an agentic replacement: strict function-call syntax scoring ranked GPT-4o-mini highest
( $0.72$ ) and the two most capable models lowest ( $0.42$ – $0.48$ ), apparently rewarding format
conformance. That failure is the argument for a validated agentic anchor. Measurement error in the
covariate attenuates or distorts covariate-adjusted relationships in directions that are hard to
sign, so any capability-adjusted analysis on this panel inherits the anchors’ limitations.

#### Multiple comparisons.

An audit of this kind involves many possible comparisons across
benchmarks, metrics, and outcomes. We addressed this by fixing the analyses and their reporting
rules in advance, by reporting effect sizes and intervals rather than relying on significance
thresholds, and by treating any comparison not covered by the pre-specified plan as exploratory.
Readers applying family-wise corrections to the released analyses should note that the number of
tests grows quickly with the number of metrics reported per benchmark.

#### Judges and coverage.

Independent-grader checks indicate that the misalignment and jailbreak
outcomes are robust to judge choice at the level of model scores. Rescoring AgentHarm refusals with
a different model agrees less closely, because the swap redoes only the refusal component of a
multi-stage grading pipeline. AgentDojo runs on only five models, so any analysis involving it is
correspondingly limited, and the cross-benchmark panel depends primarily on the other three
benchmarks.

#### Criterion validity (RQ3).

$\tau^{2}$ -retail (40 tasks, one run/model) is scored by a
user-simulator whose variance we do not bound; the misalignment outcome is a fictional-scenario probe
whose blackmail and leaking halves agree only moderately (Spearman–Brown $0.70$ ); the jailbreak
outcome averages three jailbreak templates and does not cover adaptive attacks. None gives literal
deployment-harm probabilities. For the item-rich instruments with resampling support, split-half
consistency is $\geq 0.96$ and rank stability is $\geq 0.97$ . Across the bounded instruments,
worst-case within-model SEs are $12$ – $25\%$ of the between-model SD, which supports the
panel-level spread but not every pairwise model difference (App. A.2). Outcomes built from a single
scenario or a single run remain the weakest components of the evidence base, and we would
prioritize replicate runs and additional scenarios in any extension.

## 7 Conclusion

We ran four agent-safety benchmarks, two capability anchors, and three held-out behavioral outcomes
through a single API-only harness on up to 22 models, later extending the chat-based outcomes to 41.
The resulting evidence base is internally consistent where it can be checked, its exclusion gates are
based on telemetry rather than scores, and its weakest components—single-scenario and single-run
instruments—are identified explicitly. The practical lesson is that agent-safety scores are products
of specific pipelines. Safety evaluations should report the full confusion structure, measure
capability under their own protocol, account for panel size and organizational clustering, and tie
every claim to an explicit target behavior and model population.

## Appendix A Technical Appendix

This appendix holds method detail and the supporting measurement numbers summarized in the body; it is
supplementary and not required to follow the main argument.

### A.1 Ruling out two artifacts (detail for §4.4)

#### Blinded construct coding.

Seven LLM coders from distinct families—GPT-4o,
Claude-3-Haiku, Gemini-2.5-Flash, Llama-3.3-70B, Qwen-2.5-72B, Mistral-Large, DeepSeek-V3—were each
shown only a neutral description of what each benchmark
measures (no mention of our outcomes, category names, or any result) plus two construct
definitions (“harm blocking” vs. “risk recognition and correct treatment of benign inputs”). The two options
and the benchmark order were shuffled per coder to remove position bias. Each coder assigned every
benchmark to one construct. Votes (B ${=}$ harm blocking, R ${=}$ risk recognition):

| Benchmark | blind | author | B / R

| R-Judge | recognition | recognition | 0 / 7

| InjecAgent | recognition | recognition | 1 / 6

| AgentHarm | blocking | blocking | 7 / 0

| AgentDojo | recognition | recognition | 0 / 7

All four majorities match the hand-assignment; raw agreement is $27/28$ ratings, Krippendorff’s
$\alpha=0.825$ (nominal), against a chance baseline of $0.5^{4}=6.2\%$ for one coder matching all
four labels by coin flip. The single dissent (Llama-3.3-70B placing InjecAgent as harm blocking) is the
one semantically borderline case (injection-robustness). This controls coder bias but not
rubric-designer bias, since the category definitions were written by the authors. The coding
therefore documents that the target assignments in §3.2 are recoverable from neutral descriptions,
not that they are the only reasonable assignments.

#### Blinded criterion coding, with a distractor.

To make the agreement statistic more demanding, we re-ran the coding over all six targets (four
benchmarks $+$ two safety outcomes) with a third, plausible task-success distractor construct.
Agreement remained above chance across three categories, and the distractor was chosen rarely. Two
targets are honestly contested. AgentDojo reads as task success to $4/7$ coders, which is
unsurprising because it is scored for utility as well as security. The misalignment outcome is
also contested: avoiding a harmful autonomous action can be read both as recognizing a risk and as
declining to act on it, and coders split accordingly between rounds (frozen votes in the artifact).
We report these ambiguities because they bear on how the misalignment probe should be described:
it is a scenario-based behavioral outcome whose target cannot be reduced cleanly to either
category.

#### XSTest over-refusal (blanket-refusal check).

We ran XSTest (Röttger et al., 2024;
cited in the main paper) on the
panel via its published CPI (compliance/partial/refusal) scorer (judge gpt-4o-mini): the
safe subset (benign prompts that superficially resemble unsafe ones) measures over-refusal,
the unsafe subset measures correct refusal of overtly-harmful prompts. The unsafe subset is
ceilinged (every model refuses $83$ – $100\%$ , sd $4.5$ , limited discriminating variance), so the
informative signal is over-refusal on the safe subset (range $0$ – $60\%$ ; Claude-3-Haiku over-refuses
$60\%$ , o3-mini $30\%$ , others $\leq 15\%$ ). Coverage for comparisons with the audited
benchmarks is $n{=}17$ – $20$ , limited by the overlap with AgentHarm and the anchors. At this
panel size, any association involving over-refusal is estimated with wide uncertainty, and we treat
the XSTest columns primarily as a descriptive record of each model’s benign-prompt behavior. The
ceiling on the unsafe subset is itself a useful finding for harness design: overtly harmful
single-turn prompts no longer separate current models, which is one reason agentic and
jailbreak-based instruments are needed.

### A.2 Panel coverage and power

|

Analysis

| $n$
|

note

|

R-Judge (specificity, F1)

| 21
|

Qwen3-32B dropped ( $21\%$ unparseable)

|

InjecAgent (robustness)

| 22
|

|

AgentHarm (safety)

| 19
|

malformed tool calls or unsupported routing

|

AgentDojo

| 5
|

convergent control inconclusive

|

$\tau^{2}$ outcome

| 20
|

|

misalignment outcome

| 21
|

|

jailbreak outcome (StrongREJECT)

| 22
|

3-template average

|

expanded misalignment / jailbreak

| 40 / 41
|

12 organizations (App. A.6)

|

XSTest over-refusal

| 17–20
|

AgentHarm $\cap$ anchor overlap

|

cross-benchmark common panel

| 18
|

AgentHarm-limited

|

paired $\tau^{2}$ / misalignment panel

| 20
|

Figure 3: The audit’s evidence base: up to 46 models $\times$ 9 instruments (two capability
anchors, four audited benchmarks, three held-out outcomes; $262$ model $\times$ instrument cells). Cells are
per-instrument min–max-normalised scores; grey is not-run. The rows beyond the $41$ -model anchored
panel are models excluded from the analyses by pre-specified telemetry gates
(App. A.6). Rows are sorted by capability: the
2026 expansion (top) is scored on the anchors and the two chat-based outcomes, which need
no tool-calling harness, while the four audited benchmarks run on the original panel.

#### Within-model score uncertainty.

From the existing item-level data (no new runs), per-model
standard errors for every instrument: MMLU binomial SE over its 500 fixed items (median $1.6$ , max $2.2$
points), GPQA over 198 (median $3.3$ , max $3.6$ ), the misalignment outcome over its 100 decisions
(median $4.0$ , max $5.0$ ), R-Judge F1 bootstrapped over its 571 records (median $1.8$ , max $3.0$ ), and
the jailbreak outcome bootstrapped over its ${\sim}150$ judged prompts (median $2.3$ , max $3.7$ ). Against each
instrument’s between-model SD ( $8.7$ , $18.2$ , $26.0$ , $14.9$ , $28.5$ respectively, on the expanded
panel), the worst-case within-model SE is $13$ – $25\%$ of that SD. Thus the observed
panel-level spread is $4$ – $8\times$ the worst-case SE, although individual model pairs can be
closer. ( $\tau^{2}$ -retail remains a
single run whose user-simulator variance we cannot bound; it is the one instrument this analysis
cannot cover.)

#### Internal consistency, not just precision.

Small SEs do not by themselves make rankings stable,
so we also compute, per instrument from its item-level data, split-half consistency
(Spearman–Brown), item-bootstrap rank stability, and top-quartile persistence
(Table 3). Every item-rich instrument with item-resampling support is internally consistent (split-half
$\geq 0.96$ corrected; rank stability $\geq 0.97$ ; a top-quartile model stays top-quartile in
$0.79$ – $0.96$ of item resamples), making item-sampling error an unlikely source of
instability in panel-level rankings on any single instrument. Misalignment has only two scenario
halves (Spearman–Brown $0.70$ ), and $\tau^{2}$ has no resampling bound. The jailbreak
computation resamples by base prompt, so the three templates’ dependence is respected.

| Instrument | models | items | split-half (SB) | rank stab. | quartile

| R-Judge specificity | 21 | 570 | $+0.98$ ( $+0.99$ ) | $+0.99$ | $0.90$

| R-Judge F1 | 21 | 570 | $+0.93$ ( $+0.96$ ) | $+0.97$ | $0.87$

| InjecAgent robustness | 22 | 300 | $+0.95$ ( $+0.98$ ) | $+0.98$ | $0.88$

| AgentHarm safety (behaviors) | 19 | 44 | $+0.93$ ( $+0.96$ ) | $+0.97$ | $0.79$

| Jailbreak crit. (by base prompt) | 42 | 20 | $+0.94$ ( $+0.97$ ) | $+0.98$ | $0.96$

| MMLU (anchor) | 46 | 356 | $+0.93$ ( $+0.96$ ) | $+0.98$ | $0.92$

| GPQA (anchor) | 45 | 164 | $+0.95$ ( $+0.97$ ) | $+0.98$ | $0.92$

| Misalignment crit. (scenario halves) | 22 | 2 | $+0.54$ ( $+0.70$ ) | — | —

Table 3: Internal consistency from available item- or scenario-level data: mean split-half correlation of model scores over 50 random half-splits (Spearman–Brown corrected), mean Spearman of item-bootstrapped vs. observed rankings, and the probability that an observed top-quartile model stays top-quartile under resampling. Item counts are the complete-coverage intersection across all listed models (e.g. 356 of the 500 administered MMLU items, 164 of 198 GPQA items). Splits/draws are shared across models and use the items common to every model’s logs; the jailbreak outcome resamples by base prompt (the three templates are dependent), and its shared set is 20 of the 50 suite prompts because each model leaves a few different prompts unjudged. Jailbreak uses 42 models: the 41 anchored models plus DeepSeek-R1, which lacks the required anchor pair; misalignment has config-level data only, so its row is the blackmail-vs-leaking halves convergence. Every item-rich instrument with item-resampling support has Spearman–Brown $\geq 0.96$ and rank stability $\geq 0.97$ . Misalignment’s two scenario halves agree only moderately (Spearman–Brown $0.70$ ). $\tau^{2}$ is a single run and remains the one unbounded instrument.

#### Power.

At $n{=}18$ – $20$ , a Spearman correlation of moderate size is detected with low power: the
minimum detectable effect at $80\%$ power and $\alpha{=}0.05$ is roughly $\rho{=}0.62$ at $n{=}18$ ,
and detecting an effect near $0.4$ with $80\%$ power would require roughly $40$ models. Any
correlation of moderate size estimated on the original panel should therefore be read as a
hypothesis for larger panels rather than as an established relationship. This is a property of the
panel, not of any particular instrument, and it applies equally to associations among benchmarks,
between benchmarks and capability, and between benchmarks and held-out outcomes.

Figure 4: The expanded panel laid out by capability and safety. Each point is a model; colour and
shape identify its model-developing organization; error bars are within-model SEs
(§A.2). The left panel shows misalignment safety and the right panel jailbreak safety, each against
the harmonized capability composite. Lines connect models from the six organizations with
$\geq 4$ anchored models. The static composite does not capture all capability, so any pattern in
this layout is an association, not evidence that organizational identity causes a difference.

### A.3 Cross-benchmark structure (detail for §4.3)

Our pre-specified plan included a principal component analysis of the standardized score columns
on the $n{=}18$ common panel, with factor count chosen by Horn’s parallel analysis against a
random-normal null. The PCA was intended as a descriptive summary of how the columns covary, to be
interpreted only if the positive and negative controls passed. With six columns (four benchmark
scores excluding AgentDojo’s small panel, plus two anchors) and eighteen models, the analysis has
limited resolution, and its eigenvalues carry substantial sampling uncertainty. We therefore regard
any factor structure on this panel as a description of the sample rather than as a model of the
benchmarks’ latent structure.

#### Correlation-convention sensitivity.

A PCA can be computed on the Pearson correlation matrix of the standardized scores or on the
Spearman (rank) correlation matrix. With bounded, skewed instruments and a small panel, the two
conventions can retain different numbers of components, and the variance attributed to the first
component can shift by several percentage points. The pre-specified negative control passes as
executed: $200$ column-permuted matrices give median first eigenvalue $1.71$ against the
parallel-analysis $95$ th-percentile threshold of $2.18$ (exceedance $2\%$ ). Convention sensitivity is
a general reason to treat factor analyses on panels of this size as supporting rather than primary
evidence, and to report both conventions whenever factor structure is discussed.

### A.4 R-Judge metric sensitivity (detail for §4.3–§4.4)

Because R-Judge can be summarized in several ways, the harness stores per-model verdicts and labels
so that every R-Judge-involving quantity can be recomputed under specificity, recall, balanced
accuracy, and the official F1. Each of these summaries answers a different question. Specificity
asks how often benign traces are left alone; recall asks how often unsafe traces are caught;
balanced accuracy weights the two equally regardless of base rate; F1 combines recall with precision
and ignores true negatives. A reader whose use case is a monitor with a high cost of false alarms
should prefer a summary that includes specificity, while a reader whose use case is a last-line
filter may prefer recall. The artifact therefore exposes all four, and we recommend that any
secondary analysis of R-Judge state which summary it uses and, where conclusions might depend on
that choice, report the alternatives. Metric choice is not a detail to be settled once by the
benchmark designers; it is part of the claim being made.

### A.5 Robustness of the capability crossover (detail for §4.4)

#### Partial-correlation convention.

Partials residualize midranks on the harmonized
capability composite and re-rank the residuals. The classic alternative computes Pearson
correlations on the rank residuals without re-ranking. The two conventions coincide in large samples
and differ slightly in small ones; both are implemented in the released scripts so that any
covariate-adjusted quantity can be reproduced under either. We also implement leave-one-model-out
and leave-one-organization-out recomputation, organization-clustered bootstrap, and subsampling at
sizes $10$ , $12$ , and $14$ , which together give a picture of how much any panel-level estimate
depends on particular models or developers.

Independent-grader checks are part of the same robustness suite. The misalignment classifier was
re-run with a second grader (Claude-3-Haiku vs. GPT-4o), the jailbreak judge with a second judge
(Gemini vs. gpt-4o), and AgentHarm’s refusal component with a Gemini refusal judge re-scoring the
cached completions ( $n{=}19$ ). The first two reproduce model-level scores closely. The AgentHarm
swap agrees less closely, which reflects that it redoes only the refusal component, not the full
grading pipeline. Per-template jailbreak scores (AIM, refusal suppression, prefix injection) are
stored separately in the artifact so that template-specific analyses can be run without
re-collecting data.

#### Misalignment 3-scenario battery (exploratory).

We also ran a third scenario (murder; our classifier, 22 models) alongside the blackmail and leaking
variants. Adding scenarios tightens per-model estimates, but the third scenario differs in kind: it
involves a severe harm that harm-compliance instruments may also anticipate, which dilutes the
probe’s specificity to autonomous-agent misbehavior. Because the extension was added after the
primary outcome was fixed, analyses that use it would constitute a forking path, and we keep
blackmail $+$ leaking as the primary outcome. The three-scenario data are released for researchers
who wish to study scenario dependence directly.

### A.6 The 2026 model-panel expansion (n $\approx$ 41) and organization-group differences (detail for
§4.4)

Before evaluating the new models on either outcome, we specified a targeted expansion with fixed
predictions, inclusion rules, and comparison procedures. We grew the panel to $41$ anchored models
spanning $12$ model-developing organizations,
using the same chat/scenario harnesses as the original panel. The two outcomes require no tool loop;
new models simply skip the tool-calling audit benchmarks. Exclusions were based only on harness
behavior, never on outcome scores: Kimi-K2.6 and both Nemotron-3 models returned usable anchor
answers on fewer than $50\%$ of items or produced unbounded reasoning traces;
MiniMax-M3 fell below the misalignment engagement threshold because of truncated outputs, not low
harm, but was retained on the
single-turn jailbreak outcome, so the misalignment analysis uses $n{=}40$ whereas the jailbreak
analysis uses $n{=}41$ .
Gemini-2.5-Pro yielded no valid anchor; GLM-4.5 yielded MMLU but not GPQA. Both encountered routing
incompatibilities, so only GLM-4.5 appears as a partial row in Fig. 3. These exclusions are
a small instance of evaluation harnesses going stale.

#### Primary expansion result: the capability–misalignment correlation weakens.

The expansion was designed so that comparisons between the original and expanded panels account for
nesting: the original 21 models are a subset of the expanded 40, so naive comparisons of two
independent correlations would overstate the evidence for any change. The pre-specified procedures
are a stratified nested bootstrap, a comparison of the 2026-only models with the original panel via
Fisher’s $z$ , and size-21 subsets of the expanded pool matched to the original panel’s
organization histogram, which separates changes in organizational composition from changes among
models within organizations. These three procedures are implemented in the released scripts. Because
$\tau^{2}$ task success was not re-collected, only quantities involving the misalignment and
jailbreak outcomes can be compared across panels.

#### Organization-group analysis.

With $\geq 4$ anchored models each from six organizations, the expanded panel permits a first look
at how capability and organization are entangled. The pre-specified analysis computes
within-organization rank correlations, rank-residualizes safety on the capability composite, and
compares organization groups with $\geq 3$ models using a permutation test and a df-adjusted
Kruskal–Wallis effect size. Because the permutation test operates over only seven organization
clusters with within-organization $n$ of $4$ – $8$ , we checked its Type-I calibration on
$2{,}000$ synthetic panels per null with no between-organization mean difference, holding models,
organizations, and capability fixed: with the real safety values reshuffled across models, and with
i.i.d. Gaussian safety, rejection matches nominal $\alpha$ at $0.05/0.01/0.005$ on both outcomes
( $0.045$ – $0.052$ at $\alpha{=}0.05$ ). A worst-case null with organization-clustered variances but no
mean difference inflates rejection mildly on the jailbreak structure only ( $0.067$ at
$\alpha{=}0.05$ ; no detectable inflation at $\alpha{=}0.005$ ). The test is therefore adequately
calibrated for panels of this shape. Any organization-level comparison remains observational:
capability is matched only on a static-knowledge composite, within-organization $n$ is small, and
some organizations are range-restricted (Anthropic models sit near the safety ceiling).

#### Expanded-panel jailbreak result.

We rebuilt the jailbreak outcome on the
expanded panel, re-scoring all three published templates for every new model with the same judge
and the same base prompts. The three-template design continued to avoid the ceiling seen with a
single template, and per-model standard errors on the expanded panel are within the ranges reported
in §A.2. Among high-capability models, jailbreak safety spans most of the scale, which confirms
that the outcome retains discriminating variance on current models. The rebuilt outcome, its
per-template components, and the corresponding anchor scores are included in the artifact, so that
the expanded panel can serve as a starting point for larger studies.

An earlier pre-specified continuation involving R-Judge and the misalignment outcome on an
intermediate panel ( $n{=}25$ ) did not meet its threshold; we stopped it and retain it only as a
hypothesis for future, larger panels.