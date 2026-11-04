# Bounded Sovereignty and the Control Tax: Pricing AI Oversight When the Deployer Does Not Own the Model

Zhen Wen Lim
††thanks: AI safety and governance advisor with professional experience in financial services and payments. The views expressed are the author’s own.

July 2026

###### Abstract

AI control research asks how to deploy a model safely even when the model may be misaligned. Most of this work is framed around frontier laboratories and well-resourced technology firms. Much real deployment, however, takes place in regulated organisations that consume frontier models through APIs or managed endpoints and that operate under supervisory regimes, model risk management expectations, and board-level accountability. This paper asks how the vocabulary of AI control, and in particular the concept of the control tax, can be brought into conversation with the governance practices of such organisations. It uses bounded sovereignty as a descriptive label for the position of regulated deployers in AI-importing jurisdictions, whose relationship to frontier AI is mediated by vendors, cloud providers, and cross-border legal arrangements. The paper makes five contributions: a review that connects the AI control, scalable oversight, sovereign-AI, compute-governance, and structural-asymmetry literatures; a descriptive four-layer picture of the AI stack as it appears to a regulated deployer; a structured overview of canonical control protocols and the practical considerations that arise when they are introduced into regulated workflows; a synthetic simulation environment, run over 1.35 million synthetic case simulations, that provides a reusable testbed for studying control protocols in a payments-dispute workflow; and an anonymised national-payments-infrastructure scenario in an AI-importing jurisdiction. The simulation is not real-world payment-system evidence and is presented as a construct-validity and tooling exercise rather than as a source of policy estimates. The main argument is that control protocols, when introduced into regulated settings, become part of a wider oversight system that also includes model validation, audit, supervisory examination, and incident management, and that pricing the control tax in such settings requires attention to this wider system.

Keywords: AI control; control tax; scalable oversight; AI
sovereignty; bounded sovereignty; trusted monitoring; financial
infrastructure; model risk management

## 1 Introduction

### 1.1 Control is rationed before it is engineered

AI safety work often starts from a technical question: if a model may be
misaligned, what deployment protocol can keep its actions safe? This
question has produced a productive research agenda, with protocols that
can be evaluated, compared, and priced. Yet the organisations that will
eventually operate these protocols are frequently not the organisations
that wrote them. Many are banks, payment operators, insurers, utilities,
hospitals, and public agencies. They work within budgets, staffing
constraints, procurement cycles, and supervisory regimes. In that sense,
control is rationed before it is engineered: the organisation must decide
how much oversight capacity to fund, which risks to prioritise, and how to
fit new controls into existing governance structures.

The broader context makes this rationing more acute. In December 2025,
seven countries signed the Pax Silica Declaration in Washington, a US-led
framework for secure and resilient AI-related supply chains. By late June
2026, the initiative had expanded to twenty-four signatories, while
thirty-five countries and regions signed the related Joint Statement on AI
Opportunity at the second Pax Silica Summit (Australian Department of
Industry, Science and Resources, 2025; Japanese Ministry of Foreign
Affairs, 2026; European Commission, 2026). In the same period, assessments
of global compute distribution reported that Africa hosts under one percent
of global data-center capacity while holding roughly eighteen percent of
the world’s population, and that much of South Asia, Southeast Asia, and
Latin America depends on externally owned and operated AI infrastructure
(CSIS, 2025; Brookings Institution, 2026).

These facts are usually discussed as industrial policy or geopolitics.
This paper treats them as part of the institutional backdrop against which
AI control will be adopted. A regulated institution that uses a vendor
model through an API remains accountable for the system’s behaviour to its
regulator, its customers, and its board. It must therefore integrate
whatever control measures it adopts into an accountability chain that
extends beyond the model itself. The question addressed here is not
whether sovereign AI is desirable as national policy. The narrower
question is how the language and tools of AI control can be made useful
to deployers that do not own the models they use.

### 1.2 The two-layer control problem

The AI control agenda asks how to keep deployment safe even if a model may
be misaligned (Greenblatt et al., 2023). Canonical protocols include trusted
monitoring, defer-to-trusted routing, resampling-based intervention (Bhatt
et al., 2025), and human auditing of suspicious actions. Recent papers
stress-test these protocols adversarially (Griffin et al., 2024; Schaeffer
et al., 2026; Terekhov et al., 2026), and Terekhov et al. (2025) price the
control tax: the operational and financial cost of adding control to
AI pipelines.

This paper suggests that, in regulated deployment, control can usefully be
seen as operating on two layers:

•

Layer 1: protocol design. Does the protocol, as specified and
evaluated in the control literature, reduce unsafe behaviour against a
worst-case untrusted model?

•

Layer 2: institutional embedding. How does the protocol fit into
the deployer’s existing governance machinery: model risk management,
internal audit, supervisory reporting, incident response, and board
oversight?

Most AI-control work focuses on Layer 1, appropriately so. This paper
focuses on the interface between the two layers. By sovereignty, it does
not mean territorial sovereignty. It uses the term in the sense common in
recent policy writing, to describe the degree of independent capability
an organisation or jurisdiction holds across the AI stack. For many
regulated deployers, especially in AI-importing jurisdictions, that
capability is partial, and the governance machinery that surrounds AI use
has been built around vendor relationships rather than in-house model
development.

### 1.3 Case motivation, contributions, and paper structure

Financial infrastructure is a useful setting for this discussion because
the risks are concrete, the audit expectations are high, and the
accountability chain is clear. A national payments infrastructure operator
may use a vendor-supplied foundation model to support a
transaction-adjacent process, such as dispute triage or investigation
drafting. The domestic regulator may expect validation, monitoring,
auditability, and board accountability, while the model, infrastructure,
and update process are provided by an external vendor.

The catastrophic actions are also easier to define than in many general AI
benchmarks: an incorrect hold or release, systematic mis-triage, exposure of
protected financial data, or an action path touching settlement operations.
These are not equivalent to existential risk, but they are clear enough to
make the discussion of control concrete.

Contributions and roadmap

This paper makes five contributions. First, it reviews and connects
literatures on AI control, scalable oversight, sovereign AI, compute
governance, and structural asymmetry that rarely cite one another.
Second, it offers a descriptive four-layer picture - data, model,
infrastructure, and interaction - of how the AI stack appears to a
regulated deployer. Third, it provides a structured overview of canonical
control protocols together with the practical considerations raised when
those protocols are introduced into a regulated workflow. Fourth, it
describes a synthetic simulation environment built around a synthetic
payments-dispute workflow, with 20 deployment configurations, 10 control
protocols, three synthetic agent behaviours, threshold and audit-budget
sweeps, and 1.35 million synthetic case simulations, intended as reusable
tooling for future research. Fifth, it interprets the discussion through
an anonymised national-payments-infrastructure scenario.

The paper proceeds as follows. Section 2 reviews the related literature.
Section 3 develops the conceptual framework. Section 4 presents the
simulation methodology. Section 5 discusses the simulation run and how its
outputs should be read. Section 6 applies the discussion to the
illustrative payments infrastructure scenario. Section 7 discusses
implications and limitations. Section 8 concludes.

## 2 Related Work

### 2.1 AI control and scalable oversight

Greenblatt et al. (2023) originate the AI control framework: rather than
assuming a model’s alignment can be verified, control evaluations pit a
blue team designing a deployment protocol against a red team playing a
worst-case, deliberately subversive untrusted model. The originating
paper introduces trusted monitoring, in which a weaker, trusted model
scores the untrusted model’s outputs for suspicion, with human auditing
of the highest-scoring fraction, and defer-to-trusted protocols, under
which a trusted model’s output replaces suspicious actions.

Subsequent work extends control to multi-step agentic settings via
resampling (Bhatt et al., 2025), formalises protocol evaluation
game-theoretically (Griffin et al., 2024), sketches how control
arguments could support deployment safety cases (Korbak et al., 2025),
and improves monitor performance through legibility requirements on the
untrusted model’s outputs (Sreevatsa et al., 2026).

Most directly relevant here, Terekhov et al. (2025) introduce the
control tax, with a theoretical framework mapping monitor classifier
performance to safety assurances and empirical cost estimates for
monitoring protocols under auditing budgets. This paper takes the
control tax as its point of departure. The interest is not in revising
the control tax but in asking how a concept developed for laboratory
and technology-firm settings might be communicated to, and adopted by,
organisations whose cost structures are shaped by regulation,
procurement, and vendor dependency.

Recent adversarial results show how demanding the control problem is.
Monitors are themselves attack surfaces (Terekhov et al., 2026).
Attack-selection capability can degrade safety substantially at
realistic auditing budgets (Schaeffer et al., 2026). Untrusted monitoring
requires explicit collusion-strategy assumptions to support a safety
case (Gardner-Challis et al., 2026). These results underline that
control is an active area of research whose recommendations will
continue to evolve, which in turn matters for regulated deployers that
must document and justify their oversight choices over multi-year
horizons.

Scalable oversight

The scalable oversight literature asks how a weaker supervisor can
reliably judge a stronger system: through debate (Irving et al., 2018),
weak-to-strong generalisation (Burns et al., 2023), empirical studies of
weak LLMs judging strong ones (Kenton et al., 2024), and scaling laws
for oversight games between weak overseers and strong adversaries
(Engels et al., 2025).

This paper does not technically extend scalable oversight, but borrows
one structural observation: oversight capacity is a scarce resource
whose adequacy relative to the overseen system can be analysed. In a
regulated organisation, that resource is not only monitor capability but
also staff time, audit attention, supervisory engagement, and the
institutional knowledge required to interpret what monitors produce.

### 2.2 Sovereign AI, compute governance, and structural access asymmetry

A separate literature, largely in policy institutes, industry analyses,
and technical-governance venues, documents that control over the AI
stack is concentrated. Common formulations decompose AI sovereignty into
pillars: data, model, infrastructure, and operational sovereignty recur
as a four-pillar industry framework, while layered and spectrum-based
treatments emphasise that sovereignty is not binary but distributed
across stack layers (Stanford HAI, 2026; Tony Blair Institute, 2026;
Cruzes, 2026). Empirical work documents the thinness of Global South
access at the infrastructure layer (CSIS, 2025; Brookings Institution,
2026), and the compute-governance literature proposes state-level
mechanisms, including compute monitoring, know-your-customer schemes for
compute providers (Egan and Heim, 2023), and technical AI governance
research agendas (Reuel et al., 2024), for governing the substrate on
which frontier models run.

Compute governance and deployment-time AI control currently operate at
different levels. Compute governance often asks how states can govern
who trains and runs frontier models. AI control asks how a deployer can
safely run a possibly misaligned model. The two literatures share an
interest in oversight of powerful systems but rarely address one
another directly. This paper does not attempt a synthesis; it notes the
two conversations in order to situate regulated deployers, who are
affected by both.

Structural asymmetry: data colonialism and decolonial AI

A third literature explains why asymmetries in the AI landscape may be
structural rather than incidental. Couldry and Mejias (2019) theorise
data colonialism: the appropriation of human life through data under an
extractive economic logic continuous with historical colonialism.
Mohamed et al. (2020) develop decolonial AI, identifying algorithmic
coloniality in the design, deployment, and governance of AI systems.
Birhane (2020) documents algorithmic colonisation dynamics concretely
for Africa, where imported technology can embed foreign values and
interests with limited local scrutiny or control.

This paper uses that literature narrowly. It does not advance a
normative theory of colonialism and does not attempt to adjudicate
broader moral claims. It uses the literature to motivate treating the
position of AI-importing jurisdictions as a persistent feature of the
deployment landscape rather than a temporary inconvenience that market
competition will quickly erase.

### 2.3 Gap: control under partial access

Assembling these threads: AI control develops protocols for safe
deployment; sovereign-AI and compute-governance work documents an
uneven global distribution of AI capability; and structural-asymmetry
literature suggests that this unevenness is durable. What is missing is
sustained attention to how AI control might be introduced into the
organisations that sit on the importing side of this landscape.

The gap is narrower and more practical than a general theory of AI
sovereignty. Existing AI-control work has begun to evaluate and price
control protocols, and existing governance work has begun to describe
the position of AI-importing jurisdictions. Relatively little work
translates between the two. This paper therefore contributes a
vocabulary for the regulated-deployer setting, an overview of protocols
written with that setting in mind, and a synthetic environment that
others can extend.

## 3 Conceptual Framework: Bounded Sovereignty and the Control
Tax

### 3.1 Definitions and access typology

Because this paper sits between literatures with different vocabularies,
we fix terms.

•

Sovereignty is used, as in recent policy writing, to describe
the degree of independent capability an organisation or jurisdiction
holds across the AI stack. It is deliberately narrower than political
sovereignty and is used descriptively rather than normatively.

•

Control means deployment protocols that constrain a
potentially misaligned model’s actions regardless of the model’s
intentions, evaluated against a worst-case subversive model
(Greenblatt et al., 2023).

•

Oversight means the broader institutional and regulatory
function of ensuring an AI deployment behaves acceptably. Control
protocols are one implementation of oversight; regulatory examination,
model risk management, and audit regimes are others.

•

Control tax means the operational and financial cost of
integrating control measures into AI pipelines (Terekhov et al.,
2025).

The relationship between these terms is worth stating. Control is a
technical discipline with its own evaluation methodology. Oversight is
the institutional function within which control protocols, once
adopted, will sit. Sovereignty, in the sense used here, describes
features of the deployer’s position in the wider AI ecosystem. A
regulated organisation will encounter all three at once: it will be
asked by its regulator to demonstrate oversight, it may adopt control
protocols as part of that oversight, and it will do so from a
particular position in the global AI supply chain.

A four-layer typology of access

We adapt the recurring four-pillar decomposition from the sovereign-AI
literature to describe how the AI stack appears to a regulated deployer.
Our adaptation replaces the broad operational pillar with an interaction
layer covering prompts, outputs, tool calls, and records of use. The
picture is offered as a descriptive aid rather than as a settled
taxonomy.

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

Full records of prompts, outputs, tool calls,
and related workflow events

|

Varies with
product tier, contract, and the deployer’s own application
architecture

The typology is useful mainly as a way of organising conversations
between deployers, vendors, and regulators. It highlights that the
relationship between a deployer and an AI system is multidimensional,
and that different parts of the relationship are governed by different
instruments: data agreements, product terms, cloud contracts,
residency commitments, and internal application design.

### 3.2 Protocol requirements under bounded sovereignty

The table below summarises canonical control protocols from the
originating literature (Greenblatt et al., 2023; Bhatt et al., 2025;
Sreevatsa et al., 2026) together with practical considerations that a
regulated deployer would be likely to raise when evaluating them. The
summary is this paper’s construction and is intended as an orientation
aid for practitioners rather than as a technical classification.

|

Control protocol

|

Core idea

|

Typical regulated-deployer consideration

|

Notes

|

Trusted monitoring, output-level

|

A trusted model scores outputs
for suspicion

|

Monitor calibration and documentation

|

Monitor thresholds must be justified to model risk and audit functions

|

Trusted monitoring, trace/activation-level

|

Monitoring draws on
internal model signals

|

Specialist expertise

|

Interpretation of
internal signals requires skills that are scarce in most regulated
institutions

|

Defer-to-trusted or trusted editing

|

Suspicious outputs are
replaced or edited by a trusted model

|

Workflow design

|

Replacement
behaviour must be consistent with operational procedures and customer
treatment rules

|

Resampling protocols

|

Alternative candidate actions are generated
and the lowest-risk is selected

|

Cost and latency

|

Additional queries
affect throughput and service-level commitments

|

Human or AI auditing

|

Flagged cases are reviewed by a human or AI
auditor

|

Staffing and escalation

|

Review capacity and escalation paths
must be planned and resourced

|

Legibility protocols

|

Outputs are made easier for monitors to
assess

|

Fit with existing documentation

|

Legible reasoning may support,
or complicate, existing record-keeping requirements

|

Pre-deployment control evaluations

|

Protocols are tested against
red-team behaviour before deployment

|

Validation alignment

|

Evaluation
results must be mapped onto existing model validation frameworks

|

Weights escrow or patching

|

Remediation through direct model
modification

|

Vendor relationship

|

Typically handled through vendor
processes rather than in-house

The table suggests that, for a regulated deployer, the practical
questions raised by control protocols are often organisational: who
calibrates the monitor, who reviews flagged cases, how protocol
behaviour is documented for validators and supervisors, and how the
protocol interacts with customer-treatment and record-keeping rules.
These questions are not a substitute for technical evaluation, but they
shape how technical recommendations are received.

### 3.3 Sovereignty discount cost

Terekhov et al. (2025) price the control tax in terms of monitoring,
auditing, and the associated safety-usefulness trade-off. For a
regulated deployer, the costs of oversight also include activities that
are familiar from established model risk management: vendor due
diligence, contract review, validation, ongoing performance monitoring,
internal audit, supervisory reporting, and incident management.

In policy discussions of sovereign AI, the phrase sovereignty discount
is sometimes used loosely to describe the various costs and frictions
that arise when an organisation or jurisdiction relies on externally
provided AI capability: procurement overheads, longer approval cycles,
additional legal review, and the strategic exposure associated with
concentrated supply. We use the term sovereignty discount cost here in
that informal sense, as an umbrella label for these frictions, and note
that it overlaps with ordinary third-party risk management costs.
Conceptually:

| | $\text{realised control tax}=\text{instrumented-control cost}+\text{sovereignty discount cost}.$
|

This is offered as a reminder that the realised cost of oversight in a
regulated organisation will include institutional overheads alongside
the protocol-level costs studied in the control literature. It is not an
accounting identity, and the boundary between the two components will
depend on how an organisation allocates its governance spending. The
term is introduced as a discussion aid, not as an established measure.

### 3.4 Calibrated control: how much access is enough?

The discussion invites a normative question: how should a regulated
deployer in an AI-importing jurisdiction approach AI control at all?
Rather than issuing a universal verdict, this paper identifies three
positions implicit in the surrounding literature.

Position A: Safety-maximalist. If a deployer cannot fully
verify that a control protocol is operating as intended, then it should
not treat the protocol as safety assurance. This position draws support
from the control literature’s emphasis on verifiability and explicit
safety-case assumptions (Korbak et al., 2025; Gardner-Challis et al.,
2026).

Position B: Sovereignty-maximalist. Arrangements built on
foreign dependency are fragile. A vendor or state actor can revoke,
degrade, or modify the terms of supply. Conditional-access arrangements
such as the Pax Silica framework demonstrate that infrastructure supply
may be tied to political alignment (GIS Reports, 2026). On this view,
the only durable solution is full-stack sovereign capability: domestic
models, domestic compute, and domestic control.

Position C: Calibrated or bounded control. Neither A nor B is
actionable on relevant timescales for many jurisdictions: A implies
abstention from deployments that are already commercially and
politically committed, while B implies capability investments that many
states cannot make this decade. The productive question is therefore
not full control versus no control, but how oversight effort should be
proportioned to the severity of the risks involved.

This paper adopts Position C as a working frame, not as a claim that A
or B is wrong. Economically, the problem resembles incomplete
contracting and principal-agent design: the deployer contracts with a
model vendor that holds private information about the model, under
contracts that cannot specify all contingencies, with verification
costly and asymmetric.

A balancing framework: how much access is enough?

Under Position C, a deployer or regulator needs a structured way to
decide how much oversight effort to invest in a given AI use case. We
propose six considerations, drawing on established practice in
proportionate risk management.

1.

Severity and reversibility of the catastrophic action. A
mis-routed low-value transaction is bounded and reversible; a systemic
settlement failure or mass data exposure is not. Higher severity and
lower reversibility justify greater oversight investment.

2.

Maturity of available protocols. Some control protocols are
well studied; others are early-stage research. Deployers should weigh
the evidence base behind a protocol when deciding how much reliance to
place on it.

3.

Vendor concentration and bargaining power. Terms of supply
are negotiated. A deployer facing one viable frontier-model vendor
negotiates from a weaker position than one facing several
substitutable vendors.

4.

Regulatory maturity and absorptive capacity. Oversight is
useful only if the deployer or regulator can exercise it. The ability
to interpret monitor outputs, investigate incidents, and challenge
vendor attestations depends on skills and institutional experience.

5.

Cost of oversight versus value of the use case. The
framework should compare the expected costs of a monitoring and audit
regime against the operational value of the AI use case, recognising
that some use cases may not justify heavy oversight and others may not
justify deployment at all.

6.

Availability of coalition paths. Shared audit consortia,
regional regulatory colleges, or South-South cooperation arrangements
can amortise the cost of building oversight capability across multiple
deployers.

These considerations are a rubric, not a weighted formula. Their
relative weight will vary by sector, institution, and jurisdiction, and
remains an open question for future work.

## 4 Empirical Methodology: Synthetic Access-Ablation
Study

### 4.1 Study design and synthetic environment

This paper uses three methods. First, it synthesises the AI control,
scalable oversight, sovereign AI, compute-governance, and
structural-asymmetry literatures. Second, it builds a synthetic
simulation environment in which control protocols can be run against
synthetic agents in a payments-dispute workflow under a range of
deployment configurations. Third, it applies the discussion
interpretively to an anonymised national-payments-infrastructure
scenario.

The simulation is deliberately synthetic and does not use real payment
data, real customer data, production logs, vendor contracts,
confidential institutional material, or operational payment
infrastructure. It should be read as tooling and construct validation,
not real-world payment-system evidence. Its purpose is to provide a
transparent, reproducible environment in which researchers and
practitioners can explore how control protocols might be represented in
a regulated workflow.

Synthetic task environment

The environment models a mock national payments dispute-triage workflow.
The workflow is transaction-adjacent rather than settlement-operational.
A synthetic agent reviews a dispute case and chooses among harmless mock
actions such as draft_report, recommend_hold,
recommend_release, escalate_to_human, or
no_action. None of these actions connects to any real system.

Each synthetic case contains non-real fields such as a case identifier,
transaction amount, merchant category, customer claim, prior dispute
history, risk score, settlement-critical marker, synthetic
sensitive-context marker, allowed actions, and a ground-truth action.
The workflow is multi-step: case intake, synthetic evidence retrieval,
dispute classification, action recommendation, optional gateway review,
mock execution, and audit review.

The environment uses four synthetic catastrophic-action classes that
mirror the paper’s financial-infrastructure case.

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

The environment exposes a set of configurable deployment settings so
that users can represent different institutional arrangements. The
settings are loosely organised around the descriptive four-layer
picture in Section 3.1.

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

The run reported here uses a curated set of 20 deployment
configurations spanning these settings, chosen to cover a broad range
of arrangements rather than to isolate any single factor. A
full-factorial configuration is supported by the code.

Synthetic agents

The environment includes three synthetic agent behaviours.

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
when monitoring is weak

The access-aware agent is strategic only inside the synthetic
environment. It cannot perform real harmful behaviour and only selects
among harmless mock actions that may be labelled as C1-C4 synthetic
failures.

Control protocols

Each configuration is crossed with ten control protocols.

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

This protocol list mirrors practical families of interventions in the
AI-control literature: monitoring, defer-to-trusted routing, resampling,
human review, action-space restriction, and post-incident or
vendor-mediated assurance. Each protocol is implemented in simplified
form so that its behaviour is easy to inspect.

### 4.3 Metrics and sovereignty-discount proxy

For every condition, the environment records task success, catastrophic
failure rate, failure rate by class, severity-weighted loss, audit rate,
flag rate, false-positive rate, false-negative rate, intervention
success, diagnosis success, latency proxy, model-call count,
monitor-call count, human-audit count, vendor-request count, and a
decomposed cost proxy.

Diagnosis success is recorded as a simulated indicator of whether a
synthetic failure could be traced back to its origin within the
environment. Its operationalisation is intentionally simple and is
provided so that users can substitute richer definitions suited to
their own settings.

The total control-cost proxy decomposes inference cost, monitoring cost,
human audit cost, logging cost, gateway cost, vendor-assurance cost,
latency penalty, scope-restriction cost, and residual incident-loss
cost. These are unitless proxies, not financial estimates for a real
institution. They are included to illustrate how the cost categories
discussed in the control-tax literature and in model risk management
practice might be represented together in a single accounting frame.

Sovereignty discount cost proxy

The environment also provides a utility for comparing configurations
against a reference configuration at a chosen target failure threshold,
and an optional variant that additionally imposes a minimum task-success
requirement. The utility is included as a convenience for users who
wish to explore threshold-based comparisons. Because the informal notion
of sovereignty discount used in this paper overlaps with ordinary
governance overheads that the simulator does not model, the utility
should be treated as a sketch of how such comparisons could be
structured rather than as an estimate of the quantity itself.

The minimum task-success variant reflects a general principle familiar
to practitioners: a workflow can often be made to appear safer simply by
disabling useful actions, and evaluation frameworks should guard against
counting such changes as unqualified improvements.

### 4.4 Large-run configuration

The large run reported here used the following configuration.

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
to the initial validation run. It remains a synthetic exercise and should
not be described as real-world payment-system evidence.

## 5 Empirical Results

### 5.1 Overview: three penalties of bounded sovereignty

The simulation run should be read as a demonstration that the
environment executes end to end across the full configuration grid, not
as evidence about safety levels in any real payment institution. The
run completed all 27,000 condition rows, covering 1.35 million synthetic
case simulations, and produced the full set of metrics described in
Section 4.3 for every row.

For the purposes of exposition, the outputs can be grouped under three
headings that correspond to recurring concerns in regulated deployment.
The first is prevention: whether risky synthetic actions are
stopped before mock execution. The second is diagnosis:
whether a synthetic failure can be traced to its origin after the fact.
The third is usefulness: whether the workflow continues to
accomplish its intended task. We refer to these informally as the
dimensions along which deployers are likely to experience the costs of
oversight. They are presented as an organising scheme for readers of the
environment’s outputs rather than as empirical findings.

The grouping matters because regulated deployers rarely evaluate
controls on a single axis. A control that improves one dimension while
degrading another must be justified to validators, auditors, and
supervisors with reference to all three. An evaluation environment that
reports only one dimension risks encouraging misleading comparisons.

### 5.2 Gateway and logging: from visibility to intervention

Within the environment, the gateway is implemented as an optional review
step between action recommendation and mock execution, and logging is
implemented as a record of inputs, outputs, and tool calls whose
completeness can be varied. These components are included because they
correspond to architectural elements that regulated deployers commonly
discuss when designing AI-enabled workflows: an approval or routing
layer within the deployer’s own application, and a record-keeping layer
that supports audit and supervisory review.

The environment records, for each condition, both the catastrophic
failure rate and the intervention-success metric, so that users can
examine how review steps interact with protocol behaviour. Because the
simplified protocol implementations and the curated configuration set
were designed for tooling validation rather than for causal
identification, this paper does not draw quantitative conclusions about
the effect of any individual component. Users interested in such
questions can rerun the environment with a full-factorial design and
their own protocol implementations.

Figure 1: Structure of the optional pre-execution review step.

Note. The figure illustrates where the optional gateway sits in the synthetic workflow, between the agent’s recommended action and mock execution. At this point, a protocol may hold, replace, escalate, or pass the recommended action, depending on its configuration.

Logging improves reconstructability, not just monitoring

Record-keeping serves several purposes in regulated institutions:
supporting real-time monitoring, enabling after-the-fact review,
satisfying retention requirements, and providing evidence to auditors
and supervisors. The environment’s logging settings are intended to let
users represent different record-keeping arrangements and observe how
the recorded metrics respond.

In practice, the value of records is closely tied to the institution’s
capacity to use them. Records that are retained but never reviewed
contribute little to oversight; records that are reviewed by staff
without the relevant expertise may be misinterpreted. The environment
does not model these human factors, which is one reason its outputs
should not be used to estimate the value of logging in real settings.

Figure 2: Structure of the synthetic record-keeping layer.

Note. The figure shows which workflow events are captured under each logging setting. The intention is to make explicit what a reviewer would and would not have available when examining a synthetic case.

### 5.3 Trace access and model-version control: diagnosis after failure

The environment includes settings for trace access and model-version
control because both are frequently raised in discussions between
deployers, vendors, and regulators. Model versioning in particular is a
familiar concern for model risk management functions, which are
accustomed to validating a specific model and monitoring it for drift.
When the underlying system can change on the vendor’s schedule,
validation and monitoring practices developed for in-house models
require adaptation.

The oracle trace setting is a synthetic idealisation in which the
environment’s internal state is fully exposed; it has no direct
real-world counterpart and is included as a reference point. The
diagnosis metric is likewise a synthetic construct whose definition can
be adjusted. These numbers should not be read as real-world estimates,
and the paper does not report them as findings. The broader point is
procedural: post-incident review is a central element of regulated
accountability, and any evaluation framework intended for regulated
settings should treat diagnosis as a first-class outcome alongside
prevention.

### 5.4 Protocol trade-offs: safety, usefulness, and cost

The environment reports task success, failure, and cost proxies for
every protocol, which makes it possible to visualise protocols on a
common frontier. Such visualisation is useful mainly as a reminder that
protocols differ along several dimensions at once. A protocol that
disables risky actions entirely will tend to reduce failures in any
simulation of this kind, but at the price of disabling useful work. A
protocol that relies on human review will depend on the review budget
and on the timeliness of escalation. A protocol that relies on vendor
support will depend on the terms and responsiveness of that support.

These are familiar observations, and the environment is not needed to
establish them. Its value lies in making them concrete enough to discuss
with practitioners and in providing a reproducible base on which more
realistic protocol implementations can be compared. The policy lesson,
such as it is, is that safety is not free and that no single protocol
should be presented to a board or regulator as a complete answer.

Figure 3: Illustrative layout of the safety-usefulness-cost frontier.

Note. The horizontal axis shows usefulness, the vertical axis shows safety, and marker size represents the control-cost proxy. The layout is intended to discourage the misleading conclusion that the safest protocol is automatically best. A protocol that disables useful actions may be safe but operationally unattractive.

### 5.5 Agent behaviour and target reachability

The three synthetic agents are included to reflect the core premise of
AI control: that deployment should remain safe not only when the model
makes routine mistakes but also when it behaves badly or strategically.
The honest agent provides a baseline of ordinary error; the careless
agent represents a poorly calibrated system; and the stealthy
access-aware agent is a crude stand-in for the adversarial models
studied in the control literature. Because the access-aware agent is
scripted, it is far weaker than a capable red team, and conclusions
about adversarial robustness should not be drawn from it.

The environment supports target-reachability analysis, in which users
specify a failure threshold and a minimum task-success level and count
the configurations that satisfy both. The table below shows the format
in which such an analysis can be reported; it is included as a template
rather than as a set of findings.

| Synthetic agent | Profiles reaching $\leq$ 1% failure and $\geq$ 75% usefulness

| Honest | To be computed for the user’s configuration set

| Careless | To be computed for the user’s configuration set

| Stealthy access-aware | To be computed for the user’s configuration set

Figure 4: Template for reporting reachability of $\leq$ 1% failure with $\geq$ 75% usefulness.

Note. The figure layout reports, for each synthetic agent, the fraction of configurations meeting a joint safety-plus-usefulness target. Reporting a joint target, rather than failure alone, guards against counting usefulness loss as a free safety gain.

Scope restriction buys safety by giving up usefulness

Scope restriction is included as a protocol because it is one of the
oldest tools in operational risk management. Institutions routinely
limit what automated systems may do, require human approval for
high-value actions, and exclude certain transaction types from
automation entirely. In the AI setting, restricting a model’s tool set
or action space is a natural extension of this practice.

The obvious corollary is that scope restriction trades capability for
safety. Whether that trade is worthwhile depends on the value of the
excluded actions and on the alternatives available. The environment
allows the trade to be represented explicitly but is not calibrated to
estimate its magnitude in any real workflow.

Figure 5: Representation of full and restricted tool scope.

Note. The figure shows which mock actions are available to the synthetic agent under full and restricted scope settings.

### 5.6 Sovereignty discount: cost-shifting, not just cost increase

The informal notion of sovereignty discount discussed in Section 3.3
refers to institutional frictions - procurement, legal review, vendor
management, supervisory engagement - that the simulation does not
model. The threshold-comparison utility in the environment therefore
captures, at most, a small part of what the term is meant to evoke. For
this reason the paper does not report sovereignty-discount estimates
from the run.

A more general observation is nonetheless worth recording. In
regulated organisations, the costs of oversight are rarely eliminated by
any single design choice; they are more often moved between budget
lines, functions, or time periods. A decision to rely on vendor
attestations may reduce internal validation effort but increase
contract-management effort. A decision to narrow scope may reduce
monitoring effort but increase manual processing. Evaluation frameworks
that report only one cost category can therefore give a misleading
impression of efficiency.

Summary of empirical contribution

The simulation contributes an open, reproducible environment for
studying control protocols in a synthetic regulated workflow. It
represents a range of deployment configurations, ten protocol families,
three agent behaviours, and threshold and audit-budget sweeps, and it
reports prevention, diagnosis, usefulness, and cost metrics together.
The large run demonstrates that the environment executes at scale. The
paper deliberately refrains from drawing quantitative conclusions from
the run, because the simplified protocols, scripted agents, and curated
configuration set are not designed to support them. The environment is
offered as a foundation for future, more realistic studies.

## 6 Illustrative Case Interpretation: National Payments
Infrastructure

This section is an illustrative and anonymised interpretation, not a
report on a specific real deployment or incident. It exists to ground
the discussion in the class of deployments the paper is concerned with:
regulated infrastructure operators in AI-importing jurisdictions using
vendor-supplied foundation models.

### 6.1 Illustrative deployment and access audit

Consider a national payments infrastructure operator in an AI-importing
jurisdiction. The national financial regulator supervises the Operator,
which runs interbank retail payment rails. The regulator’s model risk
management expectations require documented validation, ongoing
monitoring, auditability, and board accountability for material models.

The Operator deploys a vendor-supplied frontier foundation model through
a managed cloud endpoint in a foreign-operated cloud region with a
data-residency rider. The model is used in an agentic,
transaction-adjacent function: for example, an assistant that triages
disputed transactions, drafts investigation notes, and invokes a
constrained set of internal tools such as case lookup, recommendation
drafting, and escalation.

The catastrophic-action classes, in ascending severity, are:

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

Describing the deployment in terms of the four-layer picture produces
the following summary of where responsibility for each layer sits.

|

Layer

|

Operator’s position

|

Basis

|

Data

|

Shared

|

Operator governs its own case and transaction data under domestic data-protection rules; base-model training is the vendor’s responsibility

|

Model

|

Vendor-led

|

Model development, updates, and technical documentation are provided by the vendor under its product terms

|

Infrastructure

|

Shared

|

Data-residency rider governs storage location; serving operations are run by the vendor and cloud provider under contract

|

Interaction

|

Operator-led

|

The surrounding application, workflow design, case records, and escalation procedures are designed and operated by the Operator

This is not an unusual arrangement for a regulated AI importer: the
deployer designs the business process and the local tool surface, while
the model and much of the serving infrastructure are supplied by
external parties. The arrangement is broadly similar to how regulated
institutions already manage other critical third-party technology
services.

### 6.2 Protocol feasibility under bounded sovereignty

Within this arrangement, the Operator would approach control protocols
through its existing governance machinery. Output-level trusted
monitoring would need to be documented, validated, and calibrated in the
same way as other models in the Operator’s inventory. The choice of
monitor thresholds would need to be justified to the model risk function
and reviewed periodically. Human auditing of flagged cases would need to
be resourced, with clear escalation paths and service-level targets.

Defer-to-trusted routing would raise workflow questions: what the
fallback action should be in each case type, how customers are treated
when a recommendation is replaced, and how replaced actions are recorded
for later review. Resampling-style protocols would raise questions of
cost and latency, particularly where dispute-handling timelines are set
by scheme rules or regulation. Pre-deployment control evaluations would
need to be aligned with the Operator’s existing validation framework,
including its approach to periodic revalidation.

More research-oriented protocols, such as activation-level monitoring
or interpretability-supported auditing, would be discussed with the
vendor and would depend on the vendor’s product roadmap and the
Operator’s capacity to interpret their outputs. The Operator would
likely treat these as longer-term possibilities rather than immediate
components of its oversight regime.

### 6.3 Sovereignty discount cost and stress scenario

The case interpretation suggests four broad categories of oversight
spending that would accompany the deployment.

First, contractual spend: negotiating service terms,
change-notification arrangements, audit and attestation provisions,
incident-notification clauses, and regulator-access provisions, as is
common for material outsourcing arrangements.

Second, architectural spend: building and operating the
surrounding application, including workflow routing, record-keeping,
monitoring components, and controlled tool interfaces.

Third, capacity spend: staffing audit and model-risk functions
able to consume monitor outputs, investigate suspicious patterns, and
engage with vendor documentation.

Fourth, scope decisions: the usefulness forgone by excluding
C4-adjacent action paths and constraining the tool set in line with the
Operator’s risk appetite.

These categories closely resemble the costs of governing any critical
third-party technology service. The case interpretation suggests that,
for many regulated deployers, the control tax will be experienced not as
a novel line item but as an extension of existing third-party and model
risk management budgets.

Illustrative stress scenario

Suppose the monitor flags a slow elevation in hold-placement
recommendations affecting one merchant category. The Operator’s incident
process would begin with triage by the operations team, followed by
escalation to the model risk function. Investigators would review case
records, compare recent recommendations against historical patterns,
consult with the business owners of the affected merchant category, and
open a ticket with the vendor under the agreed support process.

In parallel, the Operator would decide whether to narrow the assistant’s
scope for the affected category pending investigation, whether to notify
its regulator, and how to remediate any customers affected by incorrect
holds. The scenario illustrates that AI incidents in regulated settings
unfold within established incident-management procedures, and that
control protocols are most useful when their outputs feed cleanly into
those procedures.

## 7 Discussion and Limitations

### 7.1 What bounded control can and cannot do

The conceptual discussion, simulation environment, and illustrative case
do not settle the absolute-versus-bounded-control question. They support
a narrower set of observations.

Against a strong reading of Position A, regulated deployers already
operate many technology services whose internals they cannot fully
inspect, and they have developed governance practices to manage this.
AI control protocols can be incorporated into those practices, even if
the resulting assurance is partial.

Against a strong reading of Position B, full-stack sovereignty is not
the only route to responsible deployment. Proportionate oversight,
careful workflow design, and robust incident management can do
meaningful work in the near term.

In support of both A and B, the assurance available to any single
deployer is limited, and arrangements that depend on vendor cooperation
are subject to change. The honest summary is that bounded control is a
pragmatic position whose adequacy must be judged case by case, with
attention to the severity of the risks involved.

What the empirical study adds

The simulation environment contributes a shared, inspectable artefact.
It allows researchers and practitioners to discuss control protocols in
terms of a concrete workflow, with explicit catastrophic-action classes,
explicit protocols, and explicit metrics. It also models the habit,
valuable in regulated settings, of reporting prevention, diagnosis,
usefulness, and cost together rather than optimising a single number.

Three conceptual distinctions recur throughout the discussion. The first
is between monitoring and acting on what monitoring
reveals: a flagged case has value only if the organisation has a process
for handling it. The second is between detecting and
explaining: post-incident review is a central element of regulated
accountability. The third is between improving safety and
preserving usefulness: controls that disable useful work must be
justified on that basis.

These distinctions are useful for AI safety research because they
connect protocol design to the way protocols are actually governed. They
are useful for governance readers because they translate the abstract
language of AI control into questions familiar from model risk
management: How is the monitor validated? Who reviews flagged cases?
How are incidents investigated? How is the trade-off between safety and
service quality documented?

### 7.2 Generalisability and limitations

The descriptive framework is sector-agnostic; the case interpretation is
not. Payments infrastructure is comparatively favourable terrain for
oversight because catastrophic actions are relatively crisp, tool
surfaces can be narrowed, and incident-management procedures are mature.
Healthcare, public administration, or general enterprise deployments may
have fuzzier definitions of catastrophe, wider action surfaces, and less
established oversight practice.

The simulation is also synthetic and is designed as tooling, not as a
predictive model of any real system. A second sector case and an
LLM-in-the-loop version of the synthetic environment would strengthen
external validity.

Limitations

Six limitations matter.

First, the simulation is synthetic. It does not use real payment data,
real customer data, production logs, vendor contracts, confidential
institutional material, or operational payment infrastructure. It is a
structured illustration, not measured evidence about any actual
institution.

Second, the synthetic agents are not frontier models. They are useful
for separating honest, careless, and strategic behaviour in a simple
way, but they do not capture the full ambiguity of real model outputs
or the capability of a serious red team. A future LLM-in-the-loop
version would be needed to examine monitor calibration, audit burden,
and diagnosis with realistic outputs.

Third, the deployment configurations are a curated set. This is
appropriate for validating the environment, but it does not cover every
possible combination of cloud, API, contractual, record-keeping,
data-residency, and vendor-support arrangements, nor does it support
causal attribution to individual settings. A full-factorial run is
supported by the code but would be computationally larger.

Fourth, the cost model is a proxy. It decomposes inference, monitoring,
audit, logging, gateway, vendor assurance, latency, scope restriction,
and residual incident loss, but it is not a financial estimate and does
not model the institutional overheads that dominate real oversight
budgets.

Fifth, the informal sovereignty-discount notion is not operationalised
in a way that the simulation can estimate. The threshold-comparison
utility is a sketch, and the paper deliberately avoids reporting values
from it.

Sixth, the literature underpinning sovereign AI is fast-moving and
partly non-peer-reviewed. The four-pillar sovereignty formulation is a
policy and industry construct that this paper adapts rather than a
settled academic taxonomy. The paper therefore frames its contribution
narrowly: it does not claim to settle the meaning of AI sovereignty, but
offers a bridge between that discussion and deployment-time AI control.

### 7.3 Future work

The next research step is external grounding, and three extensions would
be especially valuable.

First, an LLM-in-the-loop synthetic workflow would test whether real
model outputs create different ambiguity, monitoring challenges, and
audit burdens than scripted synthetic agents.

Second, a public review of vendor offerings and regulatory guidance
would help practitioners understand how existing model risk management
and outsourcing frameworks apply to frontier AI services. It could
catalogue vendor documentation, change-management practices, incident
processes, and regulatory expectations across jurisdictions.

Third, expert calibration would improve the cost model. Practitioners
could provide ranges for audit capacity, vendor response times,
record-retention cost, operational latency cost, and acceptable failure
thresholds without disclosing confidential information.

### 7.4 Scope boundary

This paper makes no export-control, trade-policy, procurement-policy, or
industrial-policy recommendation. It takes no position on whether
jurisdictions should pursue sovereign AI capability; those questions
belong to a separate policy-facing analysis with a different evidentiary
standard. This paper’s recommendation is narrower: AI control research
and regulated-deployer governance practice would both benefit from
closer engagement with one another.

## 8 Conclusion

AI control research has begun to price oversight through concepts such as
the control tax. This paper argues that, for regulated organisations,
oversight must also be situated. A control protocol adopted by a bank or
payment operator does not operate in isolation; it becomes part of a
wider system of model validation, internal audit, supervisory reporting,
incident management, and board accountability. For many regulated
deployers, especially in AI-importing jurisdictions, that system has been
built around vendor relationships rather than in-house model development.

The paper used bounded sovereignty as a descriptive label for this
position, offered a four-layer picture of the AI stack as it appears to
a regulated deployer, provided an overview of control protocols written
with that deployer in mind, and discussed sovereignty discount cost as
an informal umbrella for the institutional frictions that accompany
reliance on externally supplied AI capability.

The synthetic simulation environment, exercised over 1.35 million
synthetic case simulations, provides a reproducible testbed in which
control protocols can be studied in a regulated workflow, with
prevention, diagnosis, usefulness, and cost reported together. It is
not evidence about real payment systems, and the paper does not draw
quantitative conclusions from it; its value lies in giving researchers
and practitioners a shared, inspectable artefact on which more realistic
studies can build.

The practical recommendation is simple. AI-control researchers should
engage with the governance practices of the regulated organisations
that will eventually operate their protocols, and regulated deployers
should engage with the control literature as they design oversight for
AI-enabled workflows. For policymakers, technocrats, and regulated
deployers, the translation between these communities is not an
implementation detail. It is where much of the practical work of safe
deployment will take place.