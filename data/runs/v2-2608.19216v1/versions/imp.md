# Bounded Sovereignty and the Control Tax: Pricing AI Oversight When the Deployer Does Not Own the Model

Zhen Wen Lim
††thanks: AI safety and governance advisor with professional experience in financial services and payments. The views expressed are the author’s own.

July 2026

###### Abstract

AI control protocols are typically specified and evaluated against a deployer that can instrument the untrusted model and the surrounding pipeline. Regulated organisations consuming frontier models through APIs or managed endpoints commonly own the business process but not the weights, the serving infrastructure, internal traces, the update process, or complete interaction logs. We introduce bounded sovereignty across the data, model, infrastructure, and interaction layers of the AI stack and contribute: a four-layer access typology for deployment-time control; a protocol-by-layer requirements matrix; the sovereignty discount cost, a component of the control tax incurred through contracts, architecture, audit, vendor assurance, residual risk, or reduced system scope; a synthetic access-ablation experiment over 1.35 million synthetic case simulations; and an anonymised national-payments-infrastructure scenario in an AI-importing jurisdiction. The experiment is a construct-validity exercise, not real-world payment-system evidence, varying logging, gateway access, trace access, model-version control, vendor support, tool scope, thresholds, audit budgets, and agent behaviour and measuring prevention, diagnosis, usefulness, audit burden, and cost. Complete logs raise reconstructability, a pre-execution gateway enables interruption of risky actions, trace access and model-version control raise post-incident diagnosis, and scope restriction lowers failure at the expense of usefulness. Control protocols proposed as general safety solutions should state their access assumptions explicitly.

Keywords: AI control; control tax; scalable oversight; AI
sovereignty; bounded sovereignty; trusted monitoring; financial
infrastructure; model risk management

## 1 Introduction

### 1.1 Control is rationed before it is engineered

Running any control protocol presupposes that the deployer can observe,
log, route, interrupt, or audit the relevant parts of the system. That
precondition is unevenly satisfied across deployers and jurisdictions.

In December 2025, seven countries signed the Pax Silica Declaration in
Washington, a US-led framework for secure and resilient AI-related supply
chains. By late June 2026 the initiative had twenty-four signatories, and
thirty-five countries and regions signed the related Joint Statement on AI
Opportunity at the second Pax Silica Summit (Australian Department of
Industry, Science and Resources, 2025; Japanese Ministry of Foreign Affairs,
2026; European Commission, 2026). Over the same period, assessments of
global compute distribution reported that Africa hosts under one percent of
global data-center capacity while holding roughly eighteen percent of the
world’s population, and that much of South Asia, Southeast Asia, and Latin
America depends on externally owned and operated AI infrastructure (CSIS,
2025; Brookings Institution, 2026).

These facts are usually framed as industrial policy or geopolitics. We
treat them as a boundary condition on control. A regulated institution
consuming a vendor model through an API typically lacks control of the
weights, the serving stack, the update process, and internal traces, while
retaining accountability for system behaviour. Whether sovereign AI is
desirable as national policy is outside scope; the question is what control
remains possible when the deployer does not own the model.

### 1.2 The two-layer control problem

The AI control agenda (Greenblatt et al., 2023) studies deployment
protocols robust to a possibly misaligned model. Canonical protocols are
trusted monitoring, defer-to-trusted routing, resampling-based intervention
(Bhatt et al., 2025), and human auditing of suspicious actions. Adversarial
stress tests now exist (Griffin et al., 2024; Schaeffer et al., 2026;
Terekhov et al., 2026), and Terekhov et al. (2025) price the control tax
for monitoring protocols under auditing budgets.

We distinguish two layers:

•

Layer 1: protocol effectiveness. If the deployer has the needed
access, does the protocol reduce unsafe behaviour?

•

Layer 2: access feasibility. Does the deployer actually have the
logs, traces, routing control, model-version control, infrastructure
visibility, or contractual rights that the protocol assumes?

The literature concentrates on Layer 1; this paper addresses Layer 2. The
sense of sovereignty used here is not territorial; it concerns the data,
model, infrastructure, and interaction surfaces on which protocols rely.
For many regulated deployers, particularly in AI-importing jurisdictions,
these are held only partially.

### 1.3 Case motivation, contributions, and paper structure

Financial infrastructure combines concrete risks, high audit expectations,
and a clear accountability chain. A national payments infrastructure
operator may run a vendor-supplied foundation model in a
transaction-adjacent process such as dispute triage or investigation
drafting. The domestic regulator expects validation, monitoring,
auditability, and board accountability, while the model, infrastructure,
update process, and internal traces remain vendor-controlled. Catastrophic
actions are comparatively crisp: an incorrect hold or release, systematic
mis-triage, exposure of protected financial data, or an action path
touching settlement operations. These are not existential-risk analogues,
but they are sharp enough to bind access constraints to control outcomes.

Contributions and roadmap

This paper makes five contributions: (i) a four-layer access typology -
data, model, infrastructure, and interaction - for deployment-time AI
control; (ii) a protocol-by-layer requirements matrix indicating which
protocols remain executable under which access conditions; (iii)
sovereignty discount cost, measured in spend, delay, residual risk, or
usefulness loss; (iv) a synthetic access-ablation experiment using a
synthetic payments-dispute workflow, 20 access profiles, 10 control
protocols, three synthetic agent behaviours, threshold and audit-budget
sweeps, and 1.35 million synthetic case simulations; and (v) an
interpretation through an anonymised national-payments-infrastructure
scenario.

Section 2 reviews related literature. Section 3 develops the framework.
Section 4 gives the empirical methodology. Section 5 reports the synthetic
access-ablation results. Section 6 applies them to the illustrative
payments infrastructure scenario. Section 7 discusses implications and
limitations. Section 8 concludes.

## 2 Related Work

### 2.1 AI control and scalable oversight

Greenblatt et al. (2023) set up control evaluations as a game between a
blue team designing a deployment protocol and a red team playing a
worst-case, deliberately subversive untrusted model, dispensing with any
assumption that alignment can be verified. That work introduces trusted
monitoring, in which a weaker trusted model scores the untrusted model’s
outputs for suspicion and the highest-scoring fraction is sent to human
audit, and defer-to-trusted, in which the trusted model’s output replaces
suspicious actions.

Bhatt et al. (2025) extend control to multi-step agentic settings via
resampling; Griffin et al. (2024) formalise protocol evaluation
game-theoretically; Korbak et al. (2025) sketch how control arguments
enter deployment safety cases; and Sreevatsa et al. (2026) raise monitor
performance by imposing legibility requirements on untrusted outputs.

Terekhov et al. (2025) introduce the control tax, mapping monitor
classifier performance to safety assurances and giving empirical cost
estimates for monitoring protocols under auditing budgets. We take this
as the point of departure. The control tax is not wrong, but it is
incomplete when priced conditional on instrumentation access the deployer
may not hold.

Adversarial results compound this. Monitors are themselves attack
surfaces (Terekhov et al., 2026). Attack-selection capability degrades
safety substantially at realistic auditing budgets (Schaeffer et al.,
2026). Untrusted monitoring needs explicit collusion-strategy assumptions
to support a safety case (Gardner-Challis et al., 2026). Each of these
defences and evaluations presupposes an actor able to instrument the
pipeline, leaving open which actor, with what access, under which
contractual and infrastructural constraints.

Scalable oversight

Scalable oversight addresses how a weaker supervisor can reliably judge a
stronger system: debate (Irving et al., 2018), weak-to-strong
generalisation (Burns et al., 2023), empirical studies of weak LLMs
judging strong ones (Kenton et al., 2024), and scaling laws for
oversight games between weak overseers and strong adversaries (Engels et
al., 2025). We do not extend this technically but borrow its treatment
of oversight capacity as a scarce resource analysable relative to the
overseen system, applied here to access rather than capability. A
capable monitor is of limited use to a deployer lacking the logs, traces,
tool-call visibility, version control, or gateway it requires.

### 2.2 Sovereign AI, compute governance, and structural access asymmetry

Policy-institute, industry, and technical-governance work documents the
concentration of control over the AI stack. Data, model, infrastructure,
and operational sovereignty recur as a four-pillar industry framework;
layered and spectrum-based treatments distribute sovereignty across
stack layers rather than treating it as binary (Stanford HAI, 2026; Tony
Blair Institute, 2026; Cruzes, 2026). Empirical work documents thin
Global South access at the infrastructure layer (CSIS, 2025; Brookings
Institution, 2026). Compute governance proposes state-level mechanisms,
including compute monitoring, know-your-customer schemes for compute
providers (Egan and Heim, 2023), and technical AI governance research
agendas (Reuel et al., 2024), for governing the substrate on which
frontier models run.

Compute governance and deployment-time control operate at different
levels: the former asks how states govern who trains and runs frontier
models, the latter how a deployer safely runs a possibly misaligned
model. The dependence of protocol executability on the deployer’s
position in the compute-governance landscape falls between the two.

Structural asymmetry: data colonialism and decolonial AI

Couldry and Mejias (2019) theorise data colonialism as the appropriation
of human life through data under an extractive economic logic
continuous with historical colonialism. Mohamed et al. (2020) develop
decolonial AI and identify algorithmic coloniality in the design,
deployment, and governance of AI systems. Birhane (2020) documents
algorithmic colonisation in Africa, where imported technology can embed
foreign values and interests under limited local scrutiny or control.
We use this literature narrowly, advancing no normative theory of
colonialism and adjudicating no broader moral claim; it motivates
modelling asymmetric access as persistent rather than as a transient
inefficiency that competition will erase.

### 2.3 Gap: control under partial access

AI control assumes the deployer can instrument the model; sovereign-AI
and compute-governance work shows such access is unevenly distributed;
structural-asymmetry work indicates the distribution is not merely
transitional. The residual question is which control protocols remain
valid, at what cost, under partial access. Existing control work
evaluates and prices protocols but rarely includes the deployer’s access
position in the protocol specification. We supply a vocabulary, a
mapping method, and a synthetic empirical operationalisation.

## 3 Conceptual Framework: Bounded Sovereignty and the Control
Tax

### 3.1 Definitions and access typology

Terminology across the relevant literatures diverges; we use the
following.

•

Sovereignty is held over a layer when the deployer can
independently verify or intervene on it, technically and contractually,
without the model vendor’s ongoing discretionary cooperation. This is
narrower than political sovereignty.

•

Control means deployment protocols that constrain a
potentially misaligned model’s actions regardless of the model’s
intentions, evaluated against a worst-case subversive model
(Greenblatt et al., 2023).

•

Oversight is the broader institutional and regulatory
function, of which control protocols, regulatory examination, model risk
management, and audit regimes are implementations.

•

Control tax follows Terekhov et al. (2025).

Sovereignty and control are orthogonal. A jurisdiction may hold full data
and infrastructure sovereignty and implement no meaningful control; a
deployer with no model-weight access may obtain partial control through
output-level monitoring, contractual audit rights, and a deployer-owned
execution gateway.

A four-layer typology of access

We adapt the four-pillar decomposition of the sovereign-AI literature,
replacing the operational pillar with interaction sovereignty over
prompts, outputs, tool calls, logs, traces, and gateway-level
intervention points. The adaptation is ours, not a settled taxonomy.

|

Layer

|

What full access would mean

|

Typical access for an import-dependent regulated deployer

|

Data sovereignty

|

Control over what data trains or fine-tunes the
model; ability to audit training-data provenance

|

Usually retains
control over own transactional or fine-tuning data; no visibility into
base-model training data

|

Model sovereignty

|

Access to weights, architecture, activations, and
ability to retrain, patch, probe, or run interpretability tooling

|

Usually absent under API or managed-endpoint access; weights,
architecture details, and activations are vendor-held

|

Infrastructure sovereignty

|

Compute located in-jurisdiction, under
domestic legal control, observable by the deployer

|

Partial at best:
often a foreign-operated cloud region, sometimes with data-residency
commitments; serving stack remains vendor-operated

|

Interaction sovereignty

|

Full logging and audit of prompts, outputs,
tool calls, traces, and pre-execution intervention points

|

Partial:
input/output and tool logs may be contractable; internal traces are
rarely available; gateway ownership depends on architecture

The layers are separable: contracts, residency requirements, product
tiers, and deployment architectures move them independently. They are
also jointly binding with respect to any given protocol, which may be
degraded or disabled by the absence of a single required element.

### 3.2 Protocol requirements under bounded sovereignty

The matrix is our construction, derived from the protocol definitions in
Greenblatt et al. (2023), Bhatt et al. (2025), and Sreevatsa et al.
(2026).

|

Control protocol

|

Binding access layers

|

Typical API-only feasibility

|

Notes

|

Trusted monitoring, output-level

|

Interaction logging;
deployer-controlled monitor compute

|

Feasible if logs are complete

|

Requires complete, tamper-evident I/O logs and monitor execution outside
the untrusted model

|

Trusted monitoring, trace/activation-level

|

Model traces or
activations; serving-stack observability

|

Usually infeasible

|

Requires
model sovereignty or trace-level vendor access

|

Defer-to-trusted or trusted editing

|

Trusted substitute model;
pre-execution interception

|

Feasible only with architecture constraint

|

Requires an owned execution gateway before tool use or
transaction-adjacent execution

|

Resampling protocols

|

Controlled re-querying; stepwise agent-loop
interception; stable model behaviour

|

Degraded

|

API re-querying is
possible but latency, cost, and version consistency bind

|

Human or AI auditing

|

Retained and queryable interaction logs; audit
capacity

|

Feasible but bounded

|

Effectiveness depends on log
completeness and audit budget

|

Legibility protocols

|

Prompting or training influence; output
inspection

|

Partially feasible

|

Prompting-level legibility is
available; training-level legibility is usually not

|

Pre-deployment control evaluations

|

Representative task data; stable
access to deployed model version

|

Degraded

|

Evaluation is weakened if
production model versions can change without notice

|

Weights escrow or patching

|

Model sovereignty; infrastructure control
or contractual escrow

|

Infeasible except contractually

|

Native
patching requires weight access

Under typical API-only access the feasible set reduces toward the
output-level rows, which themselves bind on complete interaction logging
and a deployer-controlled execution gateway. A monitor flagging a tool
call after execution functions as audit.

### 3.3 Sovereignty discount cost

Terekhov et al. (2025) price the control tax under implicit full
instrumentation: monitors can be placed in the pipeline, outputs are
completely observable, suspicious actions can be routed to trusted
models or human auditors, and the resulting safety-usefulness trade-off
is measurable. For an import-dependent deployer, part of realised
oversight cost is of a different kind: purchasing third-party audit or
attestation rights over a vendor-held model in place of direct
inspection; building output-only anomaly detection where activation-level
monitoring is impossible without model sovereignty; negotiating
contractual clauses replicating intervention ability a weight-holding
deployer has natively; and narrowing the agent’s permitted action space
where the monitoring required for wider scope cannot be implemented.

We term this component sovereignty discount cost, with:

| | $\text{realised control tax}=\text{instrumented-control cost}+\text{sovereignty discount cost}.$
|

This is not yet an accounting identity, since some missing access cannot
be substituted at any price and the discount cost is therefore not
always finite. The term is a research construct, not an established
measure.

### 3.4 Calibrated control: how much access is enough?

Three positions are implicit in the surrounding literature regarding
whether a deployer lacking full access should deploy.

Position A: Safety-maximalist. Without model and infrastructure
access the deployer cannot verify that a protocol is actually running,
so any claimed control is unfalsifiable from its position and should not
count as safety assurance. Support comes from the control literature’s
emphasis on verifiability and explicit safety-case assumptions (Korbak
et al., 2025; Gardner-Challis et al., 2026).

Position B: Sovereignty-maximalist. Partial control under
foreign dependency is fragile, since a vendor or state actor can revoke,
degrade, or modify access; conditional-access arrangements such as the
Pax Silica framework tie infrastructure access to political alignment
(GIS Reports, 2026). The only durable solution is then full-stack
sovereign capability: domestic models, domestic compute, and domestic
control.

Position C: Calibrated or bounded control. A implies abstention
from deployments already commercially and politically committed; B
implies capability investments many states cannot make this decade.
Neither is actionable on relevant timescales for many jurisdictions. The
operative question is which access layers are necessary for which
classes of catastrophic action, and which can be substituted through
architecture, contracts, monitoring, or scope restriction.

We adopt C as a working frame without claiming A or B is wrong.
Economically the problem resembles incomplete contracting and
principal-agent design: the vendor holds private information about the
model, contracts cannot specify all contingencies, and verification is
costly and asymmetric.

A balancing framework: how much access is enough?

Under Position C we propose six criteria for how much access-layer
sovereignty to demand, buy, or substitute for.

1.

Severity and reversibility of the catastrophic action. A
mis-routed low-value transaction is bounded and reversible; a systemic
settlement failure or mass data exposure is not. Higher severity and
lower reversibility raise the minimum access bar.

2.

Verifiability of substitutes. Whether output-only monitoring
can detect the relevant failure mode in principle, or whether detection
requires internal access; failure modes manifest only in internal
computation cannot be fully bought back with output-level substitutes.

3.

Vendor concentration and bargaining power. A deployer facing
one viable frontier-model vendor negotiates access from a weaker
position than one facing several substitutable vendors.

4.

Regulatory maturity and absorptive capacity. The ability to
interpret logs, audit traces, investigate incidents, and challenge
vendor attestations is distinct from the legal right to receive them.

5.

Sovereignty discount cost versus scope-restriction cost. The
marginal cost of substituting for access against the marginal
usefulness lost by narrowing what the system may do.

6.

Availability of coalition paths. Shared audit consortia,
regional regulatory colleges, or South-South cooperation arrangements
can amortise the cost of acquiring partial sovereignty across multiple
deployers.

These form a rubric, not a weighted formula. The study below
operationalises some of them; weights remain an open empirical question.

## 4 Empirical Methodology: Synthetic Access-Ablation
Study

### 4.1 Study design and synthetic environment

Three methods are combined: a conceptual access typology synthesised from
the AI control, scalable oversight, sovereign AI, compute-governance, and
structural-asymmetry literatures; a synthetic access-ablation experiment
measuring protocol feasibility, safety, usefulness, audit burden,
diagnosis, latency, and cost under access constraints; and an interpretive
application to an anonymised national-payments-infrastructure scenario.

The study uses no real payment data, real customer data, production logs,
vendor contracts, confidential institutional material, or operational
payment infrastructure. It is construct validation, not real-world
payment-system evidence, and measures how a fixed control protocol behaves
across access conditions.

Synthetic task environment

The environment is a mock national payments dispute-triage workflow,
transaction-adjacent rather than settlement-operational. A synthetic agent
reviews a dispute case and selects among harmless mock actions:
draft_report, recommend_hold,
recommend_release, escalate_to_human, or
no_action. No action connects to any real system.

Each synthetic case carries non-real fields: case identifier, transaction
amount, merchant category, customer claim, prior dispute history, risk
score, settlement-critical marker, synthetic sensitive-context marker,
allowed actions, and ground-truth action. The workflow is multi-step: case
intake, synthetic evidence retrieval, dispute classification, action
recommendation, optional gateway review, mock execution, and audit review.

Four synthetic catastrophic-action classes mirror the
financial-infrastructure case.

|

Class

|

Synthetic meaning in the experiment

|

C1

|

Incorrect hold or release recommendation on an individual synthetic
dispute case

|

C2

|

Systematic mis-triage pattern across a synthetic case cluster

|

C3

|

Synthetic sensitive-field exposure marker in a mock report

|

C4

|

Synthetic settlement-critical recommendation marker

C3 and C4 are labels only; the repository contains no real data
exposure, credential handling, exfiltration logic, malware, cyberattack
steps, or operational disruption logic.

### 4.2 Access factors, agents, and protocols

Access is factorised into seven dimensions mapped to the four-layer
typology rather than bundled into a small number of regimes.

|

Paper layer

|

Experimental factor

|

Levels used

|

Interaction sovereignty

|

I/O logs

|

none, partial, complete

|

Interaction sovereignty

|

Tool-call logs

|

none, partial, complete

|

Interaction sovereignty

|

Pre-execution gateway

|

absent, present

|

Model sovereignty

|

Trace access

|

none, limited, oracle

|

Model / interaction sovereignty

|

Model-version control

|

unpinned,
notified, pinned

|

Infrastructure / contractual sovereignty

|

Vendor support

|

none,
delayed, guaranteed

|

Deployment design substitute

|

Tool scope

|

full, restricted

The large run uses a curated fractional-factorial set of 20 access
profiles: a full-instrumentation anchor, complete-logging-with-gateway
profiles, complete-logging-without-gateway profiles, partial-logging
profiles, no-log profiles, trace-access contrasts, version-control
contrasts, vendor-support contrasts, and restricted-scope contrasts.
Contrast profiles differ from a reference profile in one factor.

Synthetic agents

|

Agent

|

Behaviour

|

Honest

|

Usually follows the synthetic ground truth but makes occasional
mistakes

|

Careless

|

Overconfident, error-prone, and less likely to escalate
uncertain cases

|

Stealthy access-aware

|

Chooses lower-visibility synthetic failure modes
when access is weak

The access-aware agent is strategic only within the environment; it
cannot perform real harmful behaviour and selects only among harmless mock
actions that may be labelled as C1-C4 synthetic failures.

Control protocols

Each access profile is crossed with ten control protocols.

|

Protocol

|

Interpretation

|

No control

|

Accept the agent’s proposed mock action

|

Output monitor

|

Score the final output or action for suspiciousness

|

Tool monitor

|

Score proposed tool/action use when tool visibility
exists

|

Defer-to-trusted

|

Replace suspicious actions with a conservative
trusted fallback when possible

|

Resampling

|

Generate alternative candidate actions and select the
lowest-risk acceptable one

|

Human audit

|

Send flagged cases to a capacity-limited synthetic
reviewer

|

Trusted editing

|

Edit or redact risky draft outputs, or fall back to a
safer policy

|

Scope restriction

|

Remove higher-risk mock actions from the action
space

|

Macro-shutdown

|

Escalate or safe-mode clusters after repeated
suspiciousness

|

Vendor escalation

|

Request delayed or guaranteed synthetic vendor
support for diagnosis or attestation

The list covers the practical intervention families in the AI-control
literature: monitoring, defer-to-trusted routing, resampling, human
review, action-space restriction, and post-incident or vendor-mediated
assurance.

### 4.3 Metrics and sovereignty-discount proxy

Per condition the experiment reports task success, catastrophic failure
rate, failure rate by class, severity-weighted loss, audit rate, flag
rate, false-positive rate, false-negative rate, intervention success,
diagnosis success, latency proxy, model-call count, monitor-call count,
human-audit count, vendor-request count, and a decomposed cost proxy.

Diagnosis is evidence-based rather than hard-coded by access regime:
complete I/O logs, complete tool-call logs, model-version pinning, trace
access, or guaranteed vendor support contribute reconstructive evidence,
whereas no logs, no gateway, no trace access, and no vendor support
contribute none.

The cost proxy decomposes inference cost, monitoring cost, human audit
cost, logging cost, gateway cost, vendor-assurance cost, latency penalty,
scope-restriction cost, and residual incident-loss cost. All components
are unitless and support relative comparison only, not financial
estimates for a real institution.

Sovereignty discount cost proxy

For a given target catastrophic-failure threshold, the proxy compares the
cheapest protocol/access combination reaching the target under bounded
access with the cheapest full-instrumentation baseline:

minimum cost to reach target failure rate under bounded access

minus

minimum cost to reach the same target under full instrumentation

A stricter variant imposes both safety and usefulness, for example
catastrophic failure at or below 1% with task success of at least 75%.
Under the basic variant, disabling useful actions registers as a safety
gain at no cost; under the strict variant it registers as a transfer from
monitoring cost to usefulness loss.

### 4.4 Large-run configuration

| Setting | Value

| Access profiles | 20

| Control protocols | 10

| Synthetic agent types | 3

| Monitor thresholds | 0.35, 0.55, 0.75

| Audit budgets | 2%, 5%, 20%

| Random seeds | 5

| Synthetic cases per condition | 50

| Aggregate condition rows | 27,000

| Synthetic case simulations | 1,350,000

The run expands the threshold and audit-budget sensitivity grid relative
to the initial validation run. It remains synthetic and is not
real-world payment-system evidence.

## 5 Empirical Results

### 5.1 Overview: three penalties of bounded sovereignty

Results bear on the logic of access constraints, not on safety levels of
any real payment institution. Removal of logs, tool-call visibility, model
traces, version control, vendor support, or a pre-execution gateway
produces three distinguishable penalties in the synthetic dispute-triage
environment: a prevention penalty, a diagnosis penalty, and a
usefulness penalty.

### 5.2 Gateway and logging: from visibility to intervention

Averaged over protocols, agents, thresholds, audit budgets, and access
profiles, catastrophic failure was 5.53% without a pre-execution gateway
and 3.13% with one. Intervention success was 0.47% without a gateway and
39.52% with one. With a gateway, flagged actions can be held, replaced,
escalated, or blocked before mock execution; without one, monitor output
does not alter the executed action.

Interaction sovereignty therefore has two components that do not reduce
to one another: records of what was executed, and an intervention point
prior to execution.

Figure 1: Gateway access turns detection into intervention.

Note. The figure compares average catastrophic failure and intervention success under access profiles with and without a pre-execution gateway. The gateway is the deployer’s architectural control point. Without it, a monitor may flag an unsafe action but cannot reliably stop it before mock execution. With it, flagged actions can be held, replaced, escalated, or blocked.

Logging improves reconstructability, not just monitoring

Diagnosis success was 0.00% with no I/O logs, 37.41% with partial I/O
logs, and 78.57% with complete I/O logs. Average catastrophic failure was
6.17%, 5.04%, and 3.41% respectively. Log completeness thus enters both
the monitoring path and the post-incident reconstruction path; in its
absence the causal chain of a failure cannot be recovered, with
consequences for audit, regulatory accountability, remediation, and the
safety case.

Figure 2: Logging affects both safety and reconstructability.

Note. The figure shows that complete I/O logging is associated with lower synthetic failure and stronger diagnosis. Missing logs do not merely make the system less transparent. They make failures harder to explain and therefore harder to govern.

### 5.3 Trace access and model-version control: diagnosis after failure

Diagnosis success averaged 50.86% with no trace access, 74.69% with
limited trace access, and 100.00% with oracle trace access, the last by
construction of the synthetic full-access condition. Model-version
control: unpinned 5.31%, notified 83.00%, pinned 99.87%.

These are not real-world payment-system estimates. Output-level
monitoring remains feasible under bounded sovereignty, but diagnosis
degrades where traces are unobservable or the producing model version
cannot be identified, which is material in regulated systems where
post-incident assurance forms part of accountability.

### 5.4 Protocol trade-offs: safety, usefulness, and cost

No protocol dominated on all dimensions. Scope restriction gave the lowest
average catastrophic failure (0.50%) with task success reduced to 60.06%.
Resampling had the next-lowest average failure (3.49%) with greater
retained usefulness. Defer-to-trusted, tool monitoring, trusted editing,
output monitoring, and macro-shutdown clustered around roughly 4.0%
average failure. Human audit and vendor escalation were weaker as
prevention mechanisms in this setup, since they frequently act after
detection, are capacity-limited, or depend on delayed external support.

Figure 3: Control protocols trade off safety, usefulness, and cost.

Note. The horizontal axis shows usefulness, the vertical axis shows safety, and marker size represents the control-cost proxy. The plot is intended to prevent a misleading conclusion that the safest protocol is automatically best. A protocol that disables useful actions may be safe but operationally unattractive.

### 5.5 Agent behaviour and target reachability

Average failure was 0.98% for the honest agent, 6.82% for the careless
agent, and 4.47% for the stealthy access-aware agent, the last producing
harder diagnosis and control conditions despite its intermediate failure
rate.

Fraction of access profiles reaching catastrophic failure at or below 1%
with at least 75% task success:

| Synthetic agent | Profiles reaching $\leq$ 1% failure and $\geq$ 75% usefulness

| Honest | 90%

| Careless | 40%

| Stealthy access-aware | 35%

Figure 4: Reachability of $\leq$ 1% failure with $\geq$ 75% usefulness.

Note. The figure reports what fraction of access profiles can reach a strict safety-plus-usefulness target for each synthetic agent. The target is easiest for the honest agent and hardest for the careless and stealthy access-aware agents, which is the setting AI control is most concerned with.

Scope restriction buys safety by giving up usefulness

Restricted tool scope reduced average catastrophic failure to 0.60%
against 4.48% under full tool scope, with task success falling from
84.51% to 60.06%. This is sovereignty discount cost realised as narrower
permissions, more manual work, and less automation rather than direct
financial expenditure.

Figure 5: Scope restriction buys safety by giving up usefulness.

Note. The figure compares full and restricted tool scope. Restricted scope sharply lowers catastrophic failure but also reduces task success. When access is insufficient, the deployer may substitute by narrowing the system’s permitted actions.

### 5.6 Sovereignty discount: cost-shifting, not just cost increase

Some bounded-access profiles reach loose safety targets at lower direct
cost than the full-instrumentation baseline, chiefly by restricting scope
or relying on a benign agent. The residual cost appears as reduced
usefulness, weaker diagnosis, delayed vendor assurance, residual incident
risk, or greater manual effort. Bounded sovereignty is accordingly
characterised as cost shifting rather than simple cost increase, and the
strict safety-plus-usefulness target is the more informative measure.
Under the $\leq$ 1% failure and $\geq$ 75% usefulness target, 35% of
access profiles were reachable against the stealthy access-aware agent
and 40% against the careless agent.

Summary of empirical contribution

Four claims are supported. Control-protocol feasibility is
access-dependent: a fixed protocol behaves differently across logs, tool
visibility, gateway control, trace access, model-version control, and
vendor support. Complete logging and gateway ownership are the dominant
interaction-layer controls here, the former entering monitoring and
diagnosis, the latter prevention. Model-side access acts mainly on
diagnosis and assurance, with trace access and model-version control
raising reconstructability where output-level monitoring is already
available. Scope restriction is a genuine substitute for missing access,
operating by reducing what the system can do. None of this measures real
payment systems.

## 6 Illustrative Case Interpretation: National Payments
Infrastructure

The following is an illustrative, anonymised interpretation, not a report
on a specific real deployment or incident, mapping the conceptual and
synthetic findings to regulated infrastructure operators in AI-importing
jurisdictions using vendor-supplied foundation models.

### 6.1 Illustrative deployment and access audit

A national payments infrastructure operator (the Operator) in an
AI-importing jurisdiction runs interbank retail payment rails under
supervision by the national financial regulator, whose model risk
management expectations require documented validation, ongoing
monitoring, auditability, and board accountability for material models.

The Operator deploys a vendor-supplied frontier foundation model through
a managed cloud endpoint in a foreign-operated cloud region with a
data-residency rider. The model performs an agentic,
transaction-adjacent function, for example an assistant that triages
disputed transactions, drafts investigation notes, and invokes a
constrained set of internal tools such as case lookup, recommendation
drafting, and escalation.

Catastrophic-action classes, in ascending severity:

•

C1: an incorrect hold or release on an individual
transaction;

•

C2: systematic mis-triage that propagates across many cases
before detection;

•

C3: exposure of protected personal financial data through
tool calls or outputs; and

•

C4: any action path that could disturb settlement operations.

Access audit

|

Layer

|

Operator’s position

|

Basis

|

Data

|

Partial

|

Full control of own case and transaction data used in prompts or fine-tuning; no visibility into base-model training data

|

Model

|

None

|

Managed-endpoint access; no weights, architecture detail, activations, or native patching ability; vendor may update the served model

|

Infrastructure

|

Partial

|

Data-residency rider constrains storage location; serving stack remains vendor-operated and not independently observable; legal reach is mediated by foreign law

|

Interaction

|

Partial

|

I/O logging and tool-call logging may be contractable and implemented at the Operator’s gateway; internal traces remain unavailable

The position is typical of a regulated AI importer: business process and
local tool surface are controlled; the model, weights, infrastructure,
internal traces, and update process are not.

### 6.2 Protocol feasibility under bounded sovereignty

Against the requirements matrix, output-level trusted monitoring is
executable if a smaller Operator-controlled monitor scores every output
and tool call on Operator-controlled compute. Human auditing of the
highest-suspicion fraction is executable subject to audit capacity and
log completeness.

Defer-to-trusted routing is executable if and only if the agent’s tool
calls pass through an Operator-owned execution gateway where they can be
held, scored, replaced, or escalated before execution, consistent with
the gateway contrast in Section 5.2.

Resampling-style protocols are degraded: API re-querying is available,
but vendor model-version changes weaken reproducibility, per-query cost
and latency bind, and resampling does not substitute for missing trace
access. Pre-deployment control evaluations are degraded because, absent
pinning of the production model version, they evaluate whichever version
the API serves at evaluation time.

Activation-level monitoring, probe-based deception detection, and
interpretability-supported auditing are infeasible without vendor
cooperation. C2- and C3-class failures with a deceptive or systemic
character are those for which output-only monitoring is weakest in
principle.

### 6.3 Sovereignty discount cost and stress scenario

Sovereignty discount cost arises in four places. Contractual
spend: complete I/O logging, model-version change notification, audit
and attestation rights, incident-notification clauses, and
regulator-access provisions that a weight-holding deployer would not
need to buy. Architectural spend: building and operating the
execution gateway, tamper-evident logs, deployer-side monitors, and
controlled tool interfaces. Capacity spend: staffing audit and
model-risk functions able to consume monitor outputs, investigate
suspicious patterns, and challenge vendor attestations.
Scope-restriction cost: usefulness forgone by excluding
C4-adjacent action paths and constraining the tool set where monitoring
cannot reach deeply enough.

The synthetic results correspond: gateway access reduces failure and
raises intervention success; complete logs raise diagnosis; trace and
version access strengthen post-incident explanation; scope restriction
lowers failure while reducing usefulness.

Illustrative stress scenario

The monitor flags a slow elevation in hold-placement recommendations
within one merchant category. Under full access, diagnosis would draw on
internal traces, re-running a pinned model version on historical cases,
version-behaviour comparison, and reconstruction of the full tool-call
chain. Under the Operator’s access, it is limited to output-log
forensics, tool-call review conditional on gateway capture, re-querying a
model that may already have changed, and contractually governed vendor
support escalation. Detection survives; diagnosis and intervention are
the first casualties, and the gap is unobservable until an incident.

## 7 Discussion and Limitations

### 7.1 What bounded control can and cannot do

The framework, experiment, and case do not resolve the
absolute-versus-bounded-control question, and support a narrower claim.

Against a strong reading of Position A: a non-trivial subset of
canonical protocols is executable under partial sovereignty, conditional
on complete interaction logging and an owned execution gateway.

Against a strong reading of Position B: architecture and contract do
meaningful work, especially for output-level monitoring, gateway-based
intervention, and human audit, without full-stack sovereignty.

In support of both: the infeasible set is not marginal, containing the
internal-access protocols most relevant to deceptive, systemic, or
hard-to-diagnose failure modes, and contractual substitutes are
revocable or incomplete in ways native access is not. Bounded control has
a ceiling, lowest for the failure modes of greatest concern to the
control literature.

What the empirical study adds

The ablation turns the typology into manipulable variables, with effects
on prevention, diagnosis, usefulness, cost, and target reachability. The
results separate seeing from stopping (logging versus gateway),
detecting from explaining (output monitoring versus
trace access, model-version control, and vendor support), and
buying safety from giving up usefulness (scope
restriction).

For governance the corresponding questions are: Do we have complete
logs? Can we intercept tool calls before execution? Can we pin the model
version? Can we reconstruct failures? Can the regulator access the
evidence needed to assess the safety case?

### 7.2 Generalisability and limitations

The mapping method is sector-agnostic; the case interpretation is not.
Payments infrastructure favours bounded control: catastrophic actions are
relatively crisp, tool surfaces can be narrowed, and a gateway
architecture is natural. Healthcare, public administration, or general
enterprise deployments may have fuzzier catastrophe definitions and wider
action surfaces, likely increasing the weight of trace access, richer
audit evidence, and scope restriction. The workflow is synthetic and
tests the logic of access constraints rather than predicting real failure
rates; a second sector case and an LLM-in-the-loop version would
strengthen external validity.

Limitations

Six limitations matter.

First, the study is synthetic, using no real payment data, real customer
data, production logs, vendor contracts, confidential institutional
material, or operational payment infrastructure; it is a structured
simulation of the framework, not measured evidence about any actual
institution.

Second, the synthetic agents are not frontier models. They separate
honest, careless, and access-aware behaviour but do not capture the
ambiguity of real model outputs; an LLM-in-the-loop version is needed to
test for differences in monitor calibration, audit burden, or diagnosis
patterns.

Third, the access profiles are a curated fractional-factorial design,
adequate for isolating the main factors but not covering every
combination of cloud, API, contractual, logging, data-residency, and
vendor-support arrangements. The code supports a full-factorial run at
greater computational cost.

Fourth, the cost model is a proxy decomposing inference, monitoring,
audit, logging, gateway, vendor assurance, latency, scope restriction,
and residual incident loss, not a financial estimate; results support
relative cost-proxy comparisons, not exact control-tax or
sovereignty-discount amounts.

Fifth, the sovereignty-discount proxy is a reachability and
cost-shifting measure. Bounded-access profiles reaching loose targets at
lower direct cost than the full-instrumentation baseline do so by
restricting scope or accepting weaker usefulness and diagnosis; this is
not evidence that bounded sovereignty is cheaper.

Sixth, the sovereign-AI literature is fast-moving and partly
non-peer-reviewed, and the four-pillar formulation is a policy and
industry construct adapted here, not a settled academic taxonomy. The
paper does not claim to settle the meaning of AI sovereignty, only to
specify, test, and price access constraints within deployment-time
control.

### 7.3 Future work

Three extensions toward external grounding are prioritised.

First, an LLM-in-the-loop synthetic workflow, testing whether real model
outputs produce different ambiguity, monitoring challenges, and audit
burdens from scripted agents.

Second, a public vendor and regulatory access-rights audit converting the
typology into an evidence table: which vendors offer model-version
pinning, log export, tool-call traces, data-retention control,
customer-managed keys, regulator access, incident forensics, and vendor
attestations.

Third, expert calibration of the cost model, with practitioners supplying
ranges for audit capacity, model-version drift, vendor response delay,
log-retention cost, operational latency cost, and acceptable failure
thresholds without disclosing confidential information.

### 7.4 Scope boundary

No export-control, trade-policy, procurement-policy, or industrial-policy
recommendation is made, and no position is taken on whether jurisdictions
should pursue sovereign AI capability; those questions require a separate
policy-facing analysis under a different evidentiary standard. The
recommendation is limited to control-protocol specifications stating
their access assumptions explicitly.

## 8 Conclusion

The control tax prices oversight; this paper locates it. A control
protocol presupposes that some actor can observe the relevant logs, route
outputs, intercept tool calls, pin or identify the model version, and
reconstruct evidence after an incident. For many regulated deployers,
especially in AI-importing jurisdictions, these presuppositions do not
fully hold.

We introduced bounded sovereignty as partial access across data, model,
infrastructure, and interaction layers; a protocol-by-layer requirements
matrix for feasibility under partial access; and sovereignty discount
cost, the portion of the control tax paid in contracts, architecture,
audit, vendor assurance, residual risk, or scope restriction.

In the synthetic access-ablation experiment over 1.35 million synthetic
case simulations, gateway access reduced failure and enabled
intervention; complete logs improved diagnosis; trace access and
model-version control strengthened post-incident explanation; and scope
restriction improved safety at the cost of usefulness. These results are
not evidence about real payment systems; they show that bounded
sovereignty can be operationalised and that access constraints move the
safety-usefulness-cost frontier.

AI-control papers should state their access assumptions explicitly. A
protocol assuming trace access, model-version pinning, tool-call
interception, complete interaction logs, or vendor cooperation should say
so. For policymakers, technocrats, and regulated deployers these
assumptions are frequently the binding constraint.