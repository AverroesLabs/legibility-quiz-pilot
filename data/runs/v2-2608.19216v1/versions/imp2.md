# Bounded Sovereignty and the Control Tax: Pricing AI Oversight When the Deployer Does Not Own the Model

Zhen Wen Lim
††thanks: AI safety and governance advisor with professional experience in financial services and payments. The views expressed are the author’s own.

July 2026

###### Abstract

Control protocols are typically specified for deployers able to instrument the model and its surrounding pipeline. Regulated organisations consuming frontier models through APIs or managed endpoints may control the business process but not weights, serving infrastructure, internal traces, update process, or full interaction logs. We study bounded sovereignty across the data, model, infrastructure, and interaction layers of the AI stack and its bearing on which control protocols can be executed in practice. Contributions: a four-layer access typology for deployment-time AI control; a protocol-by-layer matrix; sovereignty discount cost, the share of the control tax spent on contracts, architecture, audit, vendor assurance, residual risk, or reduced system scope in lieu of missing access; a synthetic access-ablation experiment over 1.35 million synthetic case simulations; and an anonymised national-payments-infrastructure scenario in an AI-importing jurisdiction. The experiment is a construct-validity exercise, not real-world payment-system evidence, varying logging, gateway access, trace access, model-version control, vendor support, tool scope, thresholds, audit budgets, and agent behaviour against prevention, diagnosis, usefulness, audit burden, and cost. Complete logs improve reconstruction, a pre-execution gateway enables blocking of risky actions before they occur, trace access and model-version control improve post-incident diagnosis, and scope restriction improves safety at the expense of usefulness. Control protocols proposed as general safety solutions should state their access assumptions explicitly.

Keywords: AI control; control tax; scalable oversight; AI
sovereignty; bounded sovereignty; trusted monitoring; financial
infrastructure; model risk management

## 1 Introduction

### 1.1 Control is rationed before it is engineered

Running a control protocol requires the ability to observe, log, route, interrupt, or audit the relevant parts of the system. Many deployers lack it.

In December 2025, seven countries signed the Pax Silica Declaration in
Washington, a US-led framework for secure and resilient AI-related supply
chains. By late June 2026 the initiative had twenty-four
signatories, and thirty-five countries and regions signed the related
Joint Statement on AI Opportunity at the second Pax Silica Summit
(Australian Department of Industry, Science and Resources, 2025; Japanese
Ministry of Foreign Affairs, 2026; European Commission, 2026). Assessments of global compute distribution in the same period report that Africa
hosts under one percent of global data-center capacity with roughly
eighteen percent of the world’s population, and that much of South Asia,
Southeast Asia, and Latin America depends on externally owned and operated
AI infrastructure (CSIS, 2025; Brookings Institution, 2026).

We treat these facts as a boundary condition for AI control rather than as industrial policy or geopolitics. A regulated institution using a vendor model through an API may lack control of weights, serving stack, update process, or internal traces while remaining accountable for system behaviour. Whether sovereign AI is desirable as national policy is out of scope; the question is what control remains possible when the deployer does not own the model.

### 1.2 The two-layer control problem

The AI control agenda (Greenblatt et al., 2023) includes trusted
monitoring, defer-to-trusted routing, resampling-based intervention (Bhatt
et al., 2025), and human auditing of suspicious actions, with adversarial stress-tests in Griffin et al. (2024), Schaeffer
et al. (2026), and Terekhov et al. (2026), and the control tax priced by Terekhov et al. (2025).

•

Layer 1: protocol effectiveness. Given the needed
access, does the protocol reduce unsafe behaviour?

•

Layer 2: access feasibility. Does the deployer hold the
logs, traces, routing control, model-version control, infrastructure
visibility, or contractual rights the protocol assumes?

Most AI-control work addresses Layer 1; this paper addresses Layer 2. Sovereignty here is not territorial; it concerns data, model, infrastructure, and interaction surfaces, over which many regulated deployers in AI-importing jurisdictions hold partial access.

### 1.3 Case motivation, contributions, and paper structure

A national payments infrastructure operator may use a
vendor-supplied foundation model in a transaction-adjacent process such as dispute triage or investigation drafting, under a domestic regulator expecting validation, monitoring, auditability, and board accountability, while model, infrastructure, update process, and internal traces remain vendor-controlled. Catastrophic actions are crisper than in general AI
benchmarks: an incorrect hold or release, systematic mis-triage, exposure of
protected financial data, or an action path touching settlement operations. These are not equivalent to existential risk.

Contributions and roadmap

Five contributions: (i) a four-layer access typology (data, model, infrastructure, interaction) for deployment-time AI control; (ii) a protocol-by-layer requirements matrix; (iii) sovereignty discount cost, i.e. spend, delay, residual risk, or usefulness loss under substitution for missing access; (iv) a synthetic access-ablation experiment on a synthetic payments-dispute workflow with 20 access profiles, 10 control protocols, three synthetic agent behaviours, threshold and audit-budget sweeps, and 1.35 million synthetic case simulations; (v) an anonymised national-payments-infrastructure scenario.

Section 2 reviews related work; Section 3 the framework; Section 4 the methodology; Section 5 results; Section 6 the payments scenario; Section 7 implications and limitations; Section 8 concludes.

## 2 Related Work

### 2.1 AI control and scalable oversight

Greenblatt et al. (2023) originate control evaluations with a
blue team designing a deployment protocol against a red team playing a
worst-case, deliberately subversive untrusted model, introducing trusted monitoring (a weaker trusted model scores outputs for suspicion, with human auditing
of the highest-scoring fraction) and defer-to-trusted (a trusted model’s output replaces suspicious actions). Extensions: resampling in multi-step agentic settings (Bhatt et al., 2025), game-theoretic protocol evaluation (Griffin et al., 2024), control-based safety cases (Korbak et al., 2025), and legibility requirements on untrusted outputs (Sreevatsa et al., 2026).

Terekhov et al. (2025) map monitor classifier
performance to safety assurances and estimate costs of monitoring protocols under auditing budgets. We take this as point of departure; it is not wrong, but incomplete when priced conditional on instrumentation access the deployer may lack.

Monitors are attack surfaces (Terekhov et al., 2026); attack selection degrades safety substantially at realistic auditing
budgets (Schaeffer et al., 2026); untrusted monitoring needs explicit
collusion-strategy assumptions for a safety case (Gardner-Challis
et al., 2026). Each presupposes some actor able to instrument the pipeline; which actor, with what access, under what contractual and infrastructural constraints, is left open.

Scalable oversight

Debate (Irving et al., 2018), weak-to-strong generalisation (Burns et al., 2023), weak LLMs judging strong ones (Kenton et al., 2024), and oversight scaling laws for weak overseers against strong adversaries (Engels et al., 2025) treat oversight capacity as a scarce resource assessable relative to the overseen system. We apply this resource logic to access rather than capability: a capable monitor is of limited use without the logs, traces, tool-call visibility, version control, or gateway it needs. No technical extension of scalable oversight is claimed.

### 2.2 Sovereign AI, compute governance, and structural access asymmetry

Policy-institute, industry, and technical-governance work decomposes AI sovereignty into data, model, infrastructure, and operational pillars, with layered and spectrum-based treatments distributing it across stack layers (Stanford HAI, 2026; Tony Blair Institute, 2026;
Cruzes, 2026). Global South infrastructure-layer access is thin (CSIS, 2025; Brookings Institution,
2026). Compute governance proposes compute monitoring, know-your-customer schemes for compute providers (Egan and Heim, 2023), and technical AI governance agendas (Reuel et al., 2024).

Compute governance asks how states govern who trains and runs frontier models; AI control asks how a deployer safely runs a possibly misaligned model. Neither asks how the deployer’s position in the compute-governance landscape bears on protocol executability. This paper sits at that seam.

Structural asymmetry: data colonialism and decolonial AI

Couldry and Mejias (2019) theorise data colonialism; Mohamed et al. (2020) identify algorithmic coloniality in AI design, deployment, and governance; Birhane (2020) documents algorithmic colonisation in Africa, where imported technology can embed foreign values and interests with limited local scrutiny. We use this literature only to treat asymmetric access as a persistent rather than transitional feature of the deployment landscape; no normative theory of colonialism is advanced.

### 2.3 Gap: control under partial access

AI control assumes instrumentation; sovereign-AI and compute-governance work shows uneven access; structural-asymmetry work suggests limited access persists. Open question: which control protocols remain valid, at
what cost, under partial access? Existing work evaluates and prices protocols but rarely includes the deployer’s access position in the protocol specification. We contribute a vocabulary, a mapping method, and a synthetic empirical operationalisation.

## 3 Conceptual Framework: Bounded Sovereignty and the Control
Tax

### 3.1 Definitions and access typology

•

Sovereignty: technical and contractual access sufficient
to independently verify or intervene on an AI system, held over a layer when it does not depend on the vendor’s ongoing discretionary cooperation; narrower than political sovereignty.

•

Control: deployment protocols constraining a
potentially misaligned model’s actions regardless of intent, evaluated against a worst-case subversive model
(Greenblatt et al., 2023).

•

Oversight: the broader institutional and regulatory
function, of which control protocols, regulatory examination,
model risk management, and audit regimes are implementations.

•

Control tax (Terekhov et al.,
2025).

Sovereignty and control are orthogonal. Full data and infrastructure sovereignty is compatible with no meaningful control; a deployer without weight access may achieve partial control via output-level monitoring, contractual audit rights, and a deployer-owned execution gateway.

A four-layer typology of access

Adapted from the four-pillar sovereign-AI decomposition, with the operational pillar replaced by interaction sovereignty (prompts, outputs, tool calls, logs, traces, gateway-level intervention points). The adaptation is ours, not a settled taxonomy.

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

The layers are separable (contracts, residency requirements, product tiers, and architectures move them independently) and jointly binding (each protocol presupposes a bundle; a missing element can degrade or disable it).

### 3.2 Protocol requirements under bounded sovereignty

Derived from protocol definitions in Greenblatt et al. (2023), Bhatt et al. (2025), and Sreevatsa et al. (2026).

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

Under typical API-only access the feasible set collapses toward
output-level protocols, conditional on complete interaction logging and a deployer-controlled execution gateway.

### 3.3 Sovereignty discount cost

Terekhov et al. (2025) price the control tax under implicit full instrumentation: monitors placed in the pipeline, complete output observation, routing to trusted models or human auditors. For an import-dependent deployer, part of realised oversight cost instead goes to third-party audit or attestation rights over a vendor-held model; output-only anomaly detection in place of activation-level monitoring; contractual clauses simulating native intervention ability; and narrowing the agent’s permitted action space.

| | $\text{realised control tax}=\text{instrumented-control cost}+\text{sovereignty discount cost}.$
|

Some missing access cannot be substituted at any price, so the discount is not always finite and the expression is not an accounting identity. The term is a research construct, not an established measure.

### 3.4 Calibrated control: how much access is enough?

Three positions are implicit in the literature.

Position A: Safety-maximalist. Control a deployer cannot verify as running, for lack of model and infrastructure access, is unfalsifiable from its position and should not count as safety assurance (cf. Korbak et al., 2025; Gardner-Challis et al., 2026).

Position B: Sovereignty-maximalist. Partial control under
foreign dependency is fragile; vendors or state actors can revoke,
degrade, or modify access, and conditional-access arrangements such as Pax Silica tie infrastructure access to political alignment (GIS Reports, 2026). Only full-stack domestic models, compute, and control are durable.

Position C: Calibrated or bounded control. A implies abstention from already-committed deployments; B implies capability investments many states cannot make this decade. The operative question is which access layers are necessary for which classes of catastrophic action, and which can be substituted through architecture, contracts, monitoring, or scope restriction.

We adopt C as a working frame without claiming A or B wrong. The problem resembles incomplete contracting and principal-agent design: a vendor with private information about the model, contracts unable to specify all contingencies, costly and asymmetric verification.

A balancing framework: how much access is enough?

1.

Severity and reversibility of the catastrophic action; higher severity and
lower reversibility raise the minimum access bar.

2.

Verifiability of substitutes: whether output-only monitoring can detect the failure mode even in principle.

3.

Vendor concentration and bargaining power: one viable frontier-model vendor versus several substitutable ones.

4.

Regulatory maturity and absorptive capacity: the ability to interpret logs, audit traces, investigate incidents, and challenge
vendor attestations, as distinct from the legal right to receive them.

5.

Sovereignty discount cost versus scope-restriction cost: marginal cost of substitution against marginal usefulness lost to restricted scope.

6.

Availability of coalition paths: shared audit consortia,
regional regulatory colleges, or South-South cooperation amortising acquisition cost across deployers.

The criteria are a rubric, not a weighted formula; weights remain open.

## 4 Empirical Methodology: Synthetic Access-Ablation
Study

### 4.1 Study design and synthetic environment

Three methods: literature synthesis across AI control, scalable oversight, sovereign
AI, compute governance, and structural asymmetry; a synthetic access-ablation experiment measuring protocol feasibility, safety, usefulness, audit burden, diagnosis,
latency, and cost; and an interpretive application to an anonymised national-payments-infrastructure scenario.

No real payment data, customer data, production logs, vendor contracts,
confidential institutional material, or operational payment
infrastructure are used. The study is construct validation, not real-world payment-system evidence.

Synthetic task environment

A mock national payments dispute-triage workflow, transaction-adjacent rather than settlement-operational. A synthetic agent selects among harmless mock actions draft_report, recommend_hold,
recommend_release, escalate_to_human,
no_action, none connected to any real system. Cases carry non-real fields: case identifier,
transaction amount, merchant category, customer claim, prior dispute
history, risk score, settlement-critical marker, synthetic
sensitive-context marker, allowed actions, ground-truth action. Steps: case intake, synthetic evidence retrieval,
dispute classification, action recommendation, optional gateway review,
mock execution, audit review.

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

Access is factorised into seven dimensions.

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

A curated fractional-factorial set of 20 access
profiles: a full-instrumentation anchor,
complete-logging-with-gateway, complete-logging-without-gateway, partial-logging, and no-log profiles, plus trace-access, version-control, vendor-support, and restricted-scope contrasts.

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

The access-aware agent is strategic only within the environment and selects only among harmless mock actions labelled as C1-C4 synthetic
failures.

Control protocols

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

The list covers monitoring, defer-to-trusted routing, resampling,
human review, action-space restriction, and post-incident or
vendor-mediated assurance.

### 4.3 Metrics and sovereignty-discount proxy

Per condition: task success, catastrophic
failure rate, failure rate by class, severity-weighted loss, audit rate,
flag rate, false-positive rate, false-negative rate, intervention
success, diagnosis success, latency proxy, model-call count,
monitor-call count, human-audit count, vendor-request count, and
decomposed cost proxy.

Diagnosis success is scored on whether the evidence available under the profile suffices to reconstruct the failure, not hard-coded by regime; complete I/O logs, complete tool-call logs, model-version
pinning, trace access, or guaranteed vendor support supply more such evidence than no logs, no gateway, no trace access, and
no vendor support.

The cost proxy decomposes inference, monitoring,
human audit, logging, gateway, vendor-assurance,
latency penalty, scope-restriction, and residual incident-loss
cost; it is unitless, not a financial estimate for any institution.

Sovereignty discount cost proxy

For a target catastrophic-failure threshold:

minimum cost to reach target failure rate under bounded access

minus

minimum cost to reach the same target under full instrumentation

A strict variant additionally requires usefulness, e.g. catastrophic failure at or below 1% with task success of at least 75%; failure reduction obtained by disabling useful actions is booked as usefulness loss, not as free safety.

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

The threshold and audit-budget grid is expanded relative to the initial validation run. The study remains synthetic and is not real-world payment-system evidence.

## 5 Empirical Results

### 5.1 Overview: three penalties of bounded sovereignty

Results bear on the logic of access constraints, not on safety levels of any real payment institution. Three penalties appear: a prevention penalty (detection without pre-execution blocking absent a gateway), a diagnosis penalty (observed bad outcomes without reconstructive evidence), and a usefulness penalty (narrowed action space as substitute under weak access).

### 5.2 Gateway and logging: from visibility to intervention

Averaged over all protocols, agents, thresholds, audit budgets, and access
profiles, catastrophic failure was 5.53% without a
pre-execution gateway and 3.13% with one. Intervention
success was 0.47% without a gateway and 39.52% with one. Flagged actions can be held, replaced, escalated, or blocked before mock execution only with a gateway present.

Figure 1: Average catastrophic failure and intervention success, with and without a pre-execution gateway.

Note. Access profiles grouped by gateway presence.

Logging improves reconstructability, not just monitoring

Diagnosis success was 0.00% with no I/O logs,
37.41% with partial I/O logs, and 78.57% with complete I/O logs.
Average catastrophic failure was 6.17%, 5.04%, and 3.41% respectively.

Figure 2: Synthetic failure and diagnosis success by I/O log completeness.

Note. Complete I/O logging coincides with lower synthetic failure and higher diagnosis success.

### 5.3 Trace access and model-version control: diagnosis after failure

Diagnosis success averaged 50.86% with no
trace access, 74.69% with limited access, and
100.00% with oracle access, the last by construction of the synthetic full-access condition.
By model-version control: unpinned 5.31%, notified 83.00%,
pinned 99.87%. These are not real-world payment-system estimates. Output-level monitoring remains feasible across these contrasts.

### 5.4 Protocol trade-offs: safety, usefulness, and cost

No protocol dominated on all dimensions. Scope restriction had
the lowest average catastrophic failure (0.50%) with
task success of 60.06%. Resampling was next-lowest (3.49%) with higher usefulness. Defer-to-trusted, tool monitoring, trusted editing, output
monitoring, and macro-shutdown clustered around roughly 4.0%. Human audit and vendor escalation were weaker as prevention
mechanisms here, operating after detection, under capacity limits, or on delayed external support.

Figure 3: Protocols by usefulness (horizontal), safety (vertical), and control-cost proxy (marker size).

Note. The safest protocol is not thereby the best; protocols disabling useful actions are plotted at low usefulness.

### 5.5 Agent behaviour and target reachability

Average failure was 0.98% for the honest agent, 6.82% for the
careless agent, and 4.47% for the stealthy access-aware
agent, the last with harder diagnosis and control conditions.

For catastrophic failure at or below 1% with at least 75% task success, the fraction of access
profiles reaching the target was:

| Synthetic agent | Profiles reaching $\leq$ 1% failure and $\geq$ 75% usefulness

| Honest | 90%

| Careless | 40%

| Stealthy access-aware | 35%

Figure 4: Reachability of $\leq$ 1% failure with $\geq$ 75% usefulness.

Note. Fraction of access profiles reaching the strict target, by synthetic agent.

Scope restriction buys safety by giving up usefulness

Restricted tool scope gave average catastrophic failure of 0.60% against 4.48% under full scope; task success fell
from 84.51% to 60.06%. The discount is here paid in narrower permissions, more manual work, less automation, or lower usefulness rather than direct expenditure.

Figure 5: Catastrophic failure and task success, full versus restricted tool scope.

Note. Restricted scope lowers catastrophic failure and task success.

### 5.6 Sovereignty discount: cost-shifting, not just cost increase

Some bounded-access profiles reach loose safety targets at lower direct cost
than the full-instrumentation baseline, chiefly by restricting scope or
relying on a benign agent, with cost relocated to reduced usefulness, weaker diagnosis, delayed vendor assurance, residual incident
risk, or greater manual effort. Under the $\leq$ 1% failure and $\geq$ 75%
usefulness target, 35% of access profiles were reachable
against the stealthy access-aware agent and 40% against the
careless agent. None of this measures real payment systems.

## 6 Illustrative Case Interpretation: National Payments
Infrastructure

Illustrative and anonymised; not a report on any real deployment or incident.

### 6.1 Illustrative deployment and access audit

A national payments infrastructure operator (the Operator) in an AI-importing
jurisdiction runs interbank retail payment rails under a national financial regulator whose model risk management expectations require documented validation, ongoing
monitoring, auditability, and board accountability for material models. It deploys a vendor-supplied frontier foundation model through
a managed cloud endpoint in a foreign-operated cloud region with a
data-residency rider, in an agentic, transaction-adjacent function: triaging disputed transactions, drafting investigation notes, and invoking constrained internal tools (case lookup, recommendation
drafting, escalation).

Catastrophic-action classes, in ascending severity:

•

C1: an incorrect hold or release on an individual
transaction;

•

C2: systematic mis-triage propagating across many cases
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

This position is typical for a regulated AI importer.

### 6.2 Protocol feasibility under bounded sovereignty

Output-level trusted monitoring is executable with a smaller Operator-controlled monitor scoring every output and tool call on Operator-controlled compute. Human auditing of the highest-suspicion fraction is executable subject to audit capacity and log completeness. Defer-to-trusted is executable if and only if tool calls pass through an Operator-owned execution gateway where they can be held, scored, replaced, or escalated before execution.

Resampling is degraded: vendor model-version changes weaken reproducibility, per-query cost and latency bind, and it does not substitute for trace access. Pre-deployment control evaluations are degraded, testing whichever version the API serves at evaluation time absent pinning. Activation-level monitoring, probe-based deception
detection, and interpretability-supported auditing are infeasible without vendor cooperation; deceptive or systemic C2- and C3-class failures are where output-only monitoring is weakest in principle.

### 6.3 Sovereignty discount cost and stress scenario

Four locations: contractual spend (complete I/O logging,
model-version change notification, audit and attestation rights,
incident-notification clauses, regulator-access provisions unneeded by a weight-holding deployer); architectural spend (execution gateway, tamper-evident logs, deployer-side monitors, controlled tool interfaces); capacity spend (audit and model-risk staff to consume monitor outputs, investigate suspicious patterns, and challenge vendor attestations); scope-restriction cost (usefulness forgone by excluding C4-adjacent action paths and constraining the tool set). The synthetic results map onto these: gateway access lowers failure and raises intervention success; complete logs raise diagnosis; trace and version access strengthen post-incident explanation; scope restriction lowers failure and usefulness.

Illustrative stress scenario

The monitor flags a slow elevation in hold-placement
recommendations in one merchant category. Full access would permit inspecting internal traces, re-running
a pinned model version on historical cases, comparing version behaviour,
and reconstructing the full tool-call chain. Under the Operator’s access: output-log forensics, tool-call review if captured at the gateway, re-querying a possibly changed model, and contract-governed vendor support escalation. Detection may survive; diagnosis and intervention degrade first, invisibly until an incident.

## 7 Discussion and Limitations

### 7.1 What bounded control can and cannot do

The question of absolute versus bounded control is not settled. Against a strong reading of Position A, a non-trivial subset of canonical protocols is executable under partial sovereignty given complete interaction logging and an owned execution gateway. Against a strong reading of Position B, architecture and contract support output-level monitoring, gateway-based intervention, and human audit without full-stack sovereignty. In support of both, the infeasible set contains the internal-access protocols most relevant to deceptive, systemic, or hard-to-diagnose failure modes, and contractual substitutes are revocable or incomplete where native access is not. Bounded control is real but has a ceiling, lowest for the failure modes of greatest concern to the control literature.

What the empirical study adds

The access typology becomes a set of manipulable variables affecting prevention, diagnosis, usefulness,
cost, and target reachability. Three distinctions: seeing versus stopping (logging versus gateway); detecting versus explaining (output monitoring versus trace access, model-version control, and vendor support); buying safety versus giving up usefulness (scope restriction). For governance: Do
we have complete logs? Can we intercept tool calls before execution? Can
we pin the model version? Can we reconstruct failures? Can the regulator
access the evidence needed to assess the safety case?

### 7.2 Generalisability and limitations

The mapping method is sector-agnostic; the case is not. Payments offers crisp catastrophic actions, narrowable tool surfaces, and a natural gateway architecture. Healthcare, public
administration, or general enterprise deployments, with fuzzier catastrophe definitions and wider action surfaces, would likely raise the weight of trace access, richer audit evidence, and scope
restriction. The workflow is synthetic and does not predict failure rates in real systems; a second sector case and an LLM-in-the-loop environment would strengthen external validity.

Limitations

Six limitations.

First, the study is synthetic: no real payment
data, customer data, production logs, vendor contracts,
confidential institutional material, or operational payment
infrastructure.

Second, the synthetic agents are not frontier models and miss the ambiguity of real model outputs; an LLM-in-the-loop version is needed to test monitor calibration, audit burden, and diagnosis patterns.

Third, the 20 profiles are a curated fractional-factorial design, not covering every combination of cloud, API, contractual,
logging, data-residency, and vendor-support arrangement. A
full-factorial run is supported by the code but computationally larger.

Fourth, the cost model is a proxy over inference, monitoring,
audit, logging, gateway, vendor assurance, latency, scope restriction,
and residual incident loss; results support relative comparisons, not exact control-tax or sovereignty-discount amounts.

Fifth, the sovereignty-discount proxy is a reachability and
cost-shifting measure; lower direct cost for some bounded profiles at loose targets reflects scope restriction or weaker usefulness and diagnosis, not cheaper bounded sovereignty.

Sixth, the sovereign-AI literature is fast-moving and
partly non-peer-reviewed; the four-pillar formulation is a
policy and industry construct adapted here. No claim is made to settle the meaning of AI sovereignty.

### 7.3 Future work

Three extensions: an LLM-in-the-loop synthetic workflow; a public vendor and regulatory access-rights audit covering model-version pinning, log export, tool-call traces,
data-retention control, customer-managed keys, regulator access,
incident forensics, and vendor attestations; and expert calibration of audit capacity, model-version drift, vendor
response delay, log-retention cost, operational latency cost, and
acceptable failure thresholds, without disclosure of confidential
information.

### 7.4 Scope boundary

No export-control, trade-policy, procurement-policy, or
industrial-policy recommendation is made, and no position on sovereign AI capability is taken. The recommendation is limited to control-protocol specifications stating their access assumptions explicitly.

## 8 Conclusion

A control protocol presumes that someone can observe the right logs, route the right outputs,
intercept the right tool calls, pin or identify the right model version, and
reconstruct the right evidence after an incident; for many regulated
deployers in AI-importing jurisdictions these presumptions fail in part. We introduced bounded sovereignty across data, model, infrastructure, and interaction layers, a protocol-by-layer requirements matrix, and sovereignty discount cost, paid through contracts, architecture,
audit, vendor assurance, residual risk, or scope restriction. Over 1.35 million synthetic case
simulations, gateway access reduced failure and
enabled intervention, complete logs improved diagnosis, trace access and
model-version control strengthened post-incident explanation, and scope
restriction improved safety while reducing usefulness, shifting the safety-usefulness-cost frontier; these are not results about real payment systems. AI-control papers should state their
access assumptions explicitly, including trace access,
model-version pinning, tool-call interception, complete interaction logs, and
vendor cooperation.