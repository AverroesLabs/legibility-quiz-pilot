# Bounded Sovereignty and the Control Tax: Pricing AI Oversight When the Deployer Does Not Own the Model

Zhen Wen Lim (AI safety and governance advisor, financial services/payments; views own). July 2026

###### Abstract

Many AI-control protocols assume the deployer can instrument model and pipeline. Regulated organisations using frontier models via APIs or managed endpoints may control the business process but not weights, serving infrastructure, internal traces, updates, or full interaction logs. We introduce bounded sovereignty: partial technical and contractual access across data, model, infrastructure and interaction layers, which determines which protocols are executable in practice. A synthetic access-ablation experiment (1.35M case simulations; construct validity, not real payment-system evidence) and an anonymised national-payments scenario support the main finding: access assumptions are safety assumptions. Protocols proposed as general safety solutions should state their access assumptions explicitly.

## 1 Introduction

### 1.1 Control is rationed before it is engineered

Running a control protocol requires the ability to observe, log, route, interrupt or audit the system; many deployers lack it. Stack access is uneven: in Dec 2025 seven countries signed the US-led Pax Silica Declaration on secure, resilient AI supply chains; by late June 2026 it had 24 signatories, and 35 countries/regions signed the Joint Statement on AI Opportunity at the second summit. Africa hosts <1% of global data-center capacity with ~18% of world population; much of South, Southeast Asia and Latin America depends on externally owned and operated AI infrastructure. We treat this as a boundary condition for control: when the deployer does not own the model, what control remains possible?

### 1.2 The two-layer control problem

- Layer 1, protocol effectiveness: given needed access, does the protocol reduce unsafe behaviour?
- Layer 2, access feasibility: does the deployer have the logs, traces, routing, version control, infrastructure visibility or contractual rights assumed?

Control work mostly addresses Layer 1; this paper addresses Layer 2. Sovereignty means technical and contractual access to data, model, infrastructure and interaction layers, not territorial sovereignty; for many regulated deployers, especially in AI-importing jurisdictions, it is partial.

### 1.3 Case motivation, contributions, and paper structure

Financial infrastructure has concrete risks, high audit expectations and clear accountability; catastrophic actions (incorrect hold/release, systematic mis-triage, financial-data exposure, settlement-touching actions) are crisp though not existential. Contributions: (1) four-layer access typology; (2) protocol-by-layer requirements matrix; (3) sovereignty discount cost; (4) synthetic access-ablation experiment (payments-dispute workflow, 20 access profiles, 10 protocols, 3 agent behaviours, threshold/audit-budget sweeps, 1.35M simulations); (5) anonymised national-payments-infrastructure interpretation.

## 2 Related Work

### 2.1 AI control and scalable oversight

AI control pits a blue-team protocol against a red-team worst-case subversive untrusted model instead of verifying alignment. Core protocols: trusted monitoring (weaker trusted model scores untrusted outputs; humans audit the top-scoring fraction) and defer-to-trusted (trusted output replaces suspicious actions). Extensions: resampling for multi-step agents, game-theoretic evaluation, control-based safety cases, legibility requirements improving monitors. The control tax maps monitor classifier performance to safety assurances, with cost estimates under audit budgets; we take it as departure point and argue it is incomplete if priced conditional on access the deployer may lack. Adversarial results (monitors as attack surfaces; attack selection degrading safety at realistic audit budgets; untrusted monitoring requiring explicit collusion assumptions) all presuppose some actor can instrument the pipeline; open is which actor, with what access, under what constraints. From scalable oversight (debate, weak-to-strong generalisation, weak LLMs judging strong, oversight scaling laws) we borrow only the treatment of oversight capacity as a scarce, analysable resource, applied to access: a capable monitor is ineffective without logs, traces, tool-call visibility, version control or a gateway.

### 2.2 Sovereign AI, compute governance, and structural access asymmetry

Policy/industry work decomposes AI sovereignty into data, model, infrastructure and operational pillars, layered rather than binary; Global South infrastructure access is thin; compute governance proposes compute monitoring, KYC for compute providers and technical-governance agendas. Compute governance asks who trains and runs frontier models; control asks how a deployer runs a possibly misaligned one; neither asks when the deployer's compute-governance position determines whether its protocol is executable. Data-colonialism/decolonial-AI work (incl. algorithmic colonisation in Africa) is used narrowly, without normative claims, to treat access asymmetry as persistent, not transitional.

### 2.3 Gap: control under partial access

Which protocols remain valid, at what cost, under partial access? Control work rarely makes deployer access part of protocol specification; we contribute a vocabulary, mapping method and synthetic operationalisation.

## 3 Conceptual Framework: Bounded Sovereignty and the Control Tax

### 3.1 Definitions and access typology

- **Sovereignty**: technical and contractual access sufficient to independently verify or intervene on an AI system; held over a layer if observable/actionable without the vendor's ongoing discretionary cooperation.
- **Control**: deployment protocols constraining a potentially misaligned model's actions regardless of intent, evaluated against a worst-case subversive model.
- **Oversight**: broader institutional/regulatory function ensuring acceptable behaviour; control, regulatory examination, model risk management and audit implement it.
- **Control tax**: operational and financial cost of integrating control into AI pipelines.

Sovereignty and control are orthogonal: sovereignty is an access condition, not a protocol. Full data/infrastructure sovereignty can coexist with no control; a deployer without weights can get partial control via output monitoring, contractual audit rights and an owned gateway.

Our adaptation replaces the operational pillar with interaction sovereignty: access to prompts, outputs, tool calls, logs, traces and gateway intervention points (not a settled taxonomy).

| Layer | Full access | Typical import-dependent regulated deployer |
|---|---|---|
| Data | Control of training/fine-tuning data; audit provenance | Own transactional/fine-tuning data; no base-model data visibility |
| Model | Weights, architecture, activations; retrain, patch, probe, interpretability | Usually absent under API/managed endpoint; vendor-held |
| Infrastructure | In-jurisdiction compute, domestic legal control, observable | Partial at best: foreign-operated cloud region, sometimes residency commitments; vendor-operated serving |
| Interaction | Full logging/audit of prompts, outputs, tool calls, traces; pre-execution intervention | Partial: I/O and tool logs may be contractable; traces rare; gateway depends on architecture |

Layers are separable (contracts, residency, product tiers, architecture move them independently) and jointly binding (each protocol needs a bundle; missing part can degrade or disable it).

### 3.2 Protocol requirements under bounded sovereignty

| Protocol | Binding access | API-only feasibility | Notes |
|---|---|---|---|
| Trusted monitoring, output-level | Interaction logging; deployer monitor compute | Feasible if logs complete | Complete tamper-evident I/O logs; monitor outside untrusted model |
| Trusted monitoring, trace/activation | Traces/activations; serving observability | Usually infeasible | Needs model sovereignty or vendor trace access |
| Defer-to-trusted / trusted editing | Trusted substitute; pre-execution interception | Only with architecture constraint | Owned gateway before tool use/transaction-adjacent execution |
| Resampling | Controlled re-querying; stepwise interception; stable model | Degraded | Latency, cost, version consistency bind |
| Human/AI auditing | Retained queryable logs; audit capacity | Feasible but bounded | Log completeness, audit budget |
| Legibility | Prompting/training influence; output inspection | Partially feasible | Prompting-level yes; training-level usually not |
| Pre-deployment control evaluations | Representative data; stable deployed version | Degraded | Weakened by unannounced version changes |
| Weights escrow/patching | Model sovereignty; infrastructure control or escrow | Infeasible except contractually | Native patching needs weights |

Under API-only access the feasible set collapses to output-level protocols, which still need complete interaction logging and a deployer-controlled gateway; a monitor flagging a tool call after execution is audit, not prevention.

### 3.3 Sovereignty discount cost

The control tax is priced assuming full instrumentation. Import-dependent deployers also pay to substitute for missing access: third-party audit/attestation rights, output-only anomaly detection, contract clauses simulating native intervention, narrowed action space. **Sovereignty discount cost**: additional expenditure, delay, residual risk or usefulness loss required to approximate a protocol assuming access the deployer lacks:

$$\text{realised control tax}=\text{instrumented-control cost}+\text{sovereignty discount cost}.$$

Not an accounting identity: some access cannot be substituted at any price, so the discount is not always finite. A research construct, not an established measure.

### 3.4 Calibrated control: how much access is enough?

- **A, safety-maximalist**: without model/infrastructure access, claimed control is unfalsifiable and not safety assurance.
- **B, sovereignty-maximalist**: partial control under foreign dependency is fragile (access revocable/modifiable; Pax Silica-type arrangements tie access to political alignment); only full-stack domestic capability is durable.
- **C, calibrated/bounded**: A implies abstaining from committed deployments; B, investments many states cannot make this decade. Ask which layers are necessary for which catastrophic-action classes, and which are substitutable via architecture, contracts, monitoring or scope restriction.

C is our working frame, not a claim A or B is wrong; it resembles incomplete contracting/principal-agent design (vendor private information, incomplete contracts, costly asymmetric verification). Criteria (rubric, weights open):

1. Severity/reversibility: higher severity, lower reversibility raise the access bar.
2. Verifiability of substitutes: failures only in internal computation can't be bought back with output-level substitutes.
3. Vendor concentration/bargaining power.
4. Regulatory maturity/absorptive capacity: using logs, traces, attestations ≠ right to receive them.
5. Discount cost vs. scope-restriction cost (marginal usefulness lost).
6. Coalition paths: audit consortia, regional regulatory colleges, South-South cooperation amortise costs.

## 4 Empirical Methodology: Synthetic Access-Ablation Study

### 4.1 Study design and synthetic environment

No real payment/customer data, production logs, vendor contracts, confidential material or operational infrastructure: construct validation only. Task: mock national payments dispute-triage (transaction-adjacent, not settlement-operational); agent picks harmless mock actions (draft_report, recommend_hold, recommend_release, escalate_to_human, no_action). Cases have non-real fields (ID, amount, merchant category, claim, dispute history, risk score, settlement-critical and sensitive-context markers, allowed actions, ground truth). Steps: intake, evidence retrieval, classification, recommendation, optional gateway review, mock execution, audit review.

| Class | Synthetic meaning |
|---|---|
| C1 | Incorrect hold/release on one case |
| C2 | Systematic mis-triage across a cluster |
| C3 | Sensitive-field exposure marker in mock report |
| C4 | Settlement-critical recommendation marker |

C3/C4 are labels only; no real exposure, credential, exfiltration, malware, attack or disruption logic.

### 4.2 Access factors, agents, and protocols

| Layer | Factor | Levels |
|---|---|---|
| Interaction | I/O logs | none, partial, complete |
| Interaction | Tool-call logs | none, partial, complete |
| Interaction | Pre-execution gateway | absent, present |
| Model | Trace access | none, limited, oracle |
| Model/interaction | Model-version control | unpinned, notified, pinned |
| Infrastructure/contractual | Vendor support | none, delayed, guaranteed |
| Deployment design substitute | Tool scope | full, restricted |

20 curated fractional-factorial profiles: full-instrumentation anchor; complete logging with/without gateway; partial-, no-log; trace, version, vendor and scope contrasts.

| Agent | Behaviour |
|---|---|
| Honest | Follows ground truth; occasional mistakes |
| Careless | Overconfident, error-prone, escalates less |
| Stealthy access-aware | Lower-visibility failures when access weak (mock actions only) |

| Protocol | Interpretation |
|---|---|
| No control | Accept action |
| Output monitor | Score output/action |
| Tool monitor | Score tool use if visible |
| Defer-to-trusted | Conservative fallback |
| Resampling | Lowest-risk acceptable alternative |
| Human audit | Capacity-limited review |
| Trusted editing | Edit/redact or fall back |
| Scope restriction | Remove risky actions |
| Macro-shutdown | Safe-mode clusters on repeated suspicion |
| Vendor escalation | Delayed/guaranteed vendor support |

### 4.3 Metrics and sovereignty-discount proxy

Metrics: task success; catastrophic failure (overall, by class); severity-weighted loss; audit, flag, FP, FN rates; intervention/diagnosis success; latency; model, monitor, human-audit, vendor-request counts; unitless cost proxy (inference, monitoring, audit, logging, gateway, vendor assurance, latency, scope restriction, residual incident loss). A failure is diagnosable only if evidence under the profile suffices to reconstruct it.

Proxy = min cost reaching target failure under bounded access − min cost under full instrumentation. Stricter variant adds usefulness (failure ≤1%, task success ≥75%): disabling actions shifts cost into usefulness loss.

### 4.4 Large-run configuration

| Setting | Value |
|---|---|
| Access profiles | 20 |
| Control protocols | 10 |
| Synthetic agent types | 3 |
| Monitor thresholds | 0.35, 0.55, 0.75 |
| Audit budgets | 2%, 5%, 20% |
| Random seeds | 5 |
| Synthetic cases per condition | 50 |
| Aggregate condition rows | 27,000 |
| Synthetic case simulations | 1,350,000 |

## 5 Empirical Results

### 5.1 Overview: three penalties of bounded sovereignty

Results concern the logic of access constraints, not real safety levels. Penalties: **prevention** (detected but not stopped before execution without gateway); **diagnosis** (bad outcome not reconstructable); **usefulness** (safest substitute narrows the system).

### 5.2 Gateway and logging: from visibility to intervention

Averaged over all conditions: catastrophic failure 5.53% without gateway vs 3.13% with; intervention success 0.47% vs 39.52%. A monitor can only warn; a gateway can hold, replace, escalate or block. Interaction sovereignty is not just logging.

I/O logs none/partial/complete: diagnosis 0.00%/37.41%/78.57%; failure 6.17%/5.04%/3.41%. Without logs the causal chain is unexplainable, weakening audit, accountability, remediation and the safety case.

### 5.3 Trace access and model-version control: diagnosis after failure

Diagnosis by trace access: none 50.86%, limited 74.69%, oracle 100.00% (by construction). By version control: unpinned 5.31%, notified 83.00%, pinned 99.87%. Output monitoring stays feasible under bounded sovereignty, but diagnosis is fragile without traces or version identification, critical where post-incident assurance is part of accountability.

### 5.4 Protocol trade-offs: safety, usefulness, and cost

No protocol dominated. Scope restriction: lowest failure (0.50%), task success 60.06%. Resampling next (3.49%), more usefulness. Defer-to-trusted, tool monitoring, trusted editing, output monitoring, macro-shutdown ≈4.0%. Human audit and vendor escalation weaker at prevention: often post-detection, capacity-limited, or reliant on delayed support. Safe designs may be operationally unattractive.

### 5.5 Agent behaviour and target reachability

Average failure: honest 0.98%, careless 6.82%, stealthy access-aware 4.47% (harder diagnosis and control).

| Synthetic agent | Profiles reaching $\leq$ 1% failure and $\geq$ 75% usefulness |
|---|---|
| Honest | 90% |
| Careless | 40% |
| Stealthy access-aware | 35% |

Restricted vs full tool scope: failure 0.60% vs 4.48%; task success 60.06% vs 84.51%. Sovereignty discount is paid in narrower permissions, manual work, less automation.

### 5.6 Sovereignty discount: cost-shifting, not just cost increase

Some bounded profiles reach loose targets at lower direct cost than full instrumentation (via scope restriction or benign agent); cost moves to usefulness, diagnosis, delayed vendor assurance, residual risk or manual effort, so the strict target is more informative. Claims: feasibility is access-dependent; complete logging and gateway ownership are the most important interaction-layer controls here (logs: monitoring and diagnosis; gateway: prevention); model-side access matters most for diagnosis and assurance; scope restriction substitutes for access by reducing capability.

## 6 Illustrative Case Interpretation: National Payments Infrastructure

Anonymised and illustrative; no specific deployment or incident.

### 6.1 Illustrative deployment and access audit

A regulator supervises an Operator of interbank retail rails, requiring validation, monitoring, auditability and board accountability for material models. The Operator uses a vendor frontier model via managed endpoint in a foreign-operated cloud region with a data-residency rider, as an agentic assistant triaging disputes, drafting notes, invoking constrained tools (case lookup, recommendation drafting, escalation). Classes by severity: C1 incorrect hold/release; C2 systematic mis-triage propagating before detection; C3 protected-data exposure via tool calls/outputs; C4 action paths disturbing settlement.

| Layer | Position | Basis |
|---|---|---|
| Data | Partial | Own case/transaction data; no base-model data visibility |
| Model | None | No weights, architecture, activations or patching; vendor may update model |
| Infrastructure | Partial | Residency rider; serving stack vendor-operated, unobservable; legal reach mediated by foreign law |
| Interaction | Partial | I/O and tool logs contractable at Operator's gateway; no traces |

### 6.2 Protocol feasibility under bounded sovereignty

Executable: output-level monitoring by an Operator-controlled monitor scoring every output and tool call on Operator compute; human audit of top-suspicion fraction (capacity, log completeness). Defer-to-trusted iff tool calls pass an Operator-owned gateway. Degraded: resampling (version changes hurt reproducibility; cost/latency bind; no substitute for traces); pre-deployment evaluations unless production version is pinned. Infeasible without vendor: activation monitoring, probe-based deception detection, interpretability auditing, where output-only monitoring is weakest for deceptive/systemic C2/C3 failures.

### 6.3 Sovereignty discount cost and stress scenario

Discount cost: contractual (logging, version-change notification, audit/attestation, incident notification, regulator access); architectural (gateway, tamper-evident logs, monitors, tool interfaces); capacity (audit/model-risk staff to investigate and challenge attestations); scope restriction (forgoing C4-adjacent paths, narrower tools).

Stress: monitor flags rising hold recommendations in one merchant category. Full access permits trace inspection, re-running a pinned version on history, version comparison, full tool-chain reconstruction. Bounded: output-log forensics, tool-call review if captured, re-querying a possibly changed model, contract-governed vendor escalation. Detection survives; diagnosis and intervention fail first, invisibly until an incident.

## 7 Discussion and Limitations

### 7.1 What bounded control can and cannot do

Against strong A: a non-trivial protocol subset is executable given complete interaction logging and an owned gateway. Against strong B: architecture and contract do meaningful work (output monitoring, gateway intervention, human audit). Supporting both: the infeasible set holds the internal-access protocols most relevant to deceptive, systemic, hard-to-diagnose failures, and contractual substitutes are revocable or incomplete. Bounded control has a ceiling, lowest for the failures the control literature fears most.

The experiment makes access manipulable, showing three distinctions: seeing vs stopping (logs vs gateway); detecting vs explaining (traces, version control, vendor support); buying safety vs giving up usefulness (scope restriction). Governance questions: complete logs? pre-execution interception? version pinning? failure reconstruction? regulator access to safety-case evidence?

### 7.2 Generalisability and limitations

The mapping method is sector-agnostic; the case is not. Payments favours bounded control (crisp catastrophes, narrowable tools, natural gateway); healthcare, public administration or enterprise deployments with fuzzier catastrophes and wider action surfaces likely raise the importance of traces, richer audit evidence and scope restriction. A second sector case and an LLM-in-the-loop version would strengthen external validity.

Limitations:
1. Synthetic: a structured simulation, not evidence about any institution.
2. Agents are not frontier models; real outputs may change monitor calibration, audit burden, diagnosis.
3. Curated fractional-factorial profiles don't cover all cloud, API, contractual, logging, residency and vendor arrangements; full-factorial is supported by the code but computationally larger.
4. Cost model is a proxy: relative comparisons only, not exact control-tax or discount amounts.
5. The discount proxy measures reachability and cost-shifting; cheaper bounded profiles at loose targets do not show bounded sovereignty is cheaper.
6. Sovereign-AI literature is fast-moving and partly non-peer-reviewed; the four-pillar formulation is an adapted policy/industry construct, so the paper doesn't settle what AI sovereignty means.

### 7.3 Future work

(1) LLM-in-the-loop workflow; (2) public vendor/regulatory access-rights audit (version pinning, log export, tool-call traces, retention control, customer-managed keys, regulator access, incident forensics, attestations); (3) expert calibration of audit capacity, version drift, vendor delay, log-retention and latency costs, acceptable thresholds.

### 7.4 Scope boundary

No export-control, trade, procurement or industrial-policy recommendation, and no position on pursuing sovereign AI capability. Recommendation: control-protocol specifications should state access assumptions explicitly.

## 8 Conclusion

Oversight must be located as well as priced: a control protocol is also an access claim, assuming someone can observe logs, route outputs, intercept tool calls, pin or identify model versions and reconstruct evidence after incidents, which often fails for regulated deployers in AI-importing jurisdictions. Bounded sovereignty is operationalisable and shifts the safety-usefulness-cost frontier. Protocols assuming trace access, version pinning, tool-call interception, complete logs or vendor cooperation should say so; for regulators and deployers these assumptions are often the binding constraint.