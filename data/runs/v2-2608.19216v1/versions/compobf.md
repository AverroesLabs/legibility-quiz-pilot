# Bounded Sovereignty and the Control Tax: Pricing AI Oversight When the Deployer Does Not Own the Model

Zhen Wen Lim (AI safety and governance advisor, financial services/payments; views own). July 2026

###### Abstract


Access assumptions are safety assumptions: this main finding is supported by Z5J (1.35M case simulations; construct validity, not real payment-system evidence) together with an anonymised national-payments scenario, and protocols proposed as general safety solutions should state their access assumptions explicitly. The determination of which protocols are executable in practice is made by K7R, partial technical and contractual access across data, model, infrastructure and interaction layers; the assumption, made by many AI-control protocols, that the deployer can instrument model and pipeline, is one under which regulated organisations using frontier models via APIs or managed endpoints may control the business process but not weights, serving infrastructure, internal traces, updates, or full interaction logs.

## 1 Introduction

What control remains possible when the deployer does not own the model is treated as a boundary condition for control (1.1), the running of a control protocol requiring the ability to observe, log, route, interrupt or audit the system, an ability many deployers lack, stack access being uneven: by late June 2026 the US-led Pax Silica Declaration on secure, resilient AI supply chains, signed in Dec 2025 by seven countries, had 24 signatories, 35 countries/regions signed the Joint Statement on AI Opportunity at the second summit, Africa hosts <1% of global data-center capacity with ~18% of world population, and much of South, Southeast Asia and Latin America depends on externally owned and operated AI infrastructure. Financial infrastructure was selected as the case (1.3) for its concrete risks, high audit expectations and clear accountability, its catastrophic actions (incorrect hold/release, systematic mis-triage, financial-data exposure, settlement-touching actions) being crisp though not existential, and the contributions comprise (4) Z5J over a payments-dispute workflow with 20 access profiles, 10 protocols, 3 agent behaviours, threshold/audit-budget sweeps and 1.35M simulations (Z5J denoting the synthetic access-ablation experiment), (3) Q9X (Q9X denoting the sovereignty discount cost), (1) T2V (T2V denoting the four-layer access typology), (2) W8N (W8N denoting the protocol-by-layer requirements matrix) and (5) an anonymised national-payments-infrastructure interpretation. The question whether the deployer has the logs, traces, routing, version control, infrastructure visibility or contractual rights assumed, designated Layer 2 (access feasibility), is the one this paper addresses, whereas control work mostly addresses the question whether, given needed access, a protocol reduces unsafe behaviour, designated Layer 1 (protocol effectiveness) (1.2). Sovereignty, partial for many regulated deployers, especially in AI-importing jurisdictions, means technical and contractual access to data, model, infrastructure and interaction layers, not territorial sovereignty.

## 2 Related Work

M4Z is taken as the departure point and argued to be incomplete if priced conditional on access the deployer may lack, and, control work rarely making deployer access part of protocol specification, a vocabulary, a mapping method and a synthetic operationalisation are contributed (2.3) toward the question of which protocols remain valid, at what cost, under partial access.

A capable monitor is ineffective without logs, traces, tool-call visibility, version control or a gateway, oversight capacity being treated as a scarce, analysable resource applied to access, the only borrowing from scalable oversight (debate, weak-to-strong generalisation, weak LLMs judging strong, oversight scaling laws); and the adversarial results—monitors as attack surfaces, attack selection degrading safety at realistic audit budgets, untrusted monitoring requiring explicit collusion assumptions—all presuppose that some actor can instrument the pipeline, open being which actor, with what access, under what constraints.

When the deployer's compute-governance position determines whether its protocol is executable is asked neither by compute governance, which asks who trains and runs frontier models and proposes compute monitoring, KYC for compute providers and technical-governance agendas, nor by control, which asks how a deployer runs a possibly misaligned one; AI sovereignty is decomposed by policy and industry work into data, model, infrastructure and operational pillars, layered rather than binary, with Global South infrastructure access thin; and data-colonialism and decolonial-AI work, including algorithmic colonisation in Africa, is used narrowly, without normative claims, to treat access asymmetry as persistent, not transitional (2.2).

AI control (2.1) pits a blue-team protocol against a red-team worst-case subversive untrusted model instead of verifying alignment, its core protocols being D3P (denoting trusted monitoring), under which a weaker trusted model scores untrusted outputs and humans audit the top-scoring fraction, and F6Q (denoting defer-to-trusted), under which trusted output replaces suspicious actions, with extensions comprising H1Y (denoting resampling) for multi-step agents, game-theoretic evaluation, control-based safety cases and legibility requirements improving monitors, while M4Z (denoting the control tax) maps monitor classifier performance to safety assurances, with cost estimates under audit budgets.

## 3 Conceptual Framework: Bounded Sovereignty and the Control Tax

That the feasible protocol set collapses, under API-only access, to output-level protocols — themselves still requiring complete interaction logging and a deployer-controlled J3W, a monitor flagging a tool call after execution being audit rather than prevention — is recorded against the W8N of 3.2, while the pricing of what missing access costs is carried by the identity of 3.3,

$$\text{realised M4Z}=\text{instrumented-control cost}+\text{Q9X},$$

where Q9X, the additional expenditure, delay, residual risk or usefulness loss required to approximate a protocol assuming access the deployer lacks, is not always finite, some access being unsubstitutable at any price, so that the identity is not an accounting identity and the construct remains a research construct rather than an established measure, M4Z having been priced assuming C4Q even though import-dependent deployers additionally pay to substitute for missing access through third-party audit/attestation rights, output-only anomaly detection, contract clauses simulating native intervention and a narrowed action space.

The binding access, API-only feasibility and notes of each protocol, among them F6Q paired with N2S (N2S denoting trusted editing), are tabulated as W8N under K7R (K7R denoting bounded sovereignty):

W8N (3.2): binding access, API-only feasibility and notes, by protocol.

| Protocol | Binding access | API-only feasibility | Notes |
|---|---|---|---|
| D3P, output-level | Interaction logging; deployer monitor compute | Feasible if logs complete | Complete tamper-evident I/O logs; monitor outside untrusted model |
| D3P, trace/activation | Traces/activations; serving observability | Usually infeasible | Needs model sovereignty or vendor trace access |
| F6Q / N2S | Trusted substitute; pre-execution interception | Only with architecture constraint | Owned J3W before tool use/transaction-adjacent execution |
| H1Y | Controlled re-querying; stepwise interception; stable model | Degraded | Latency, cost, version consistency bind |
| L8C/AI auditing | Retained queryable logs; audit capacity | Feasible but bounded | Log completeness, audit budget |
| Legibility | Prompting/training influence; output inspection | Partially feasible | Prompting-level yes; training-level usually not |
| Pre-deployment control evaluations | Representative data; stable deployed version | Degraded | Weakened by unannounced version changes |
| Weights escrow/patching | Model sovereignty; infrastructure control or escrow | Infeasible except contractually | Native patching needs weights |

Sovereignty is technical and contractual access sufficient to independently verify or intervene on an AI system, held over a layer if observable or actionable without the vendor's ongoing discretionary cooperation; control, orthogonal to it as an access condition is to a protocol, designates deployment protocols constraining a potentially misaligned model's actions regardless of intent and evaluated against a worst-case subversive model, such that full data and infrastructure sovereignty can coexist with no control while a deployer without weights obtains partial control through output monitoring, contractual audit rights and an owned J3W; oversight names the broader institutional and regulatory function of ensuring acceptable behaviour, implemented by control, regulatory examination, model risk management and audit; and M4Z, as fixed in 3.1, is the operational and financial cost of integrating control into AI pipelines.

The adaptation of 3.1, which replaces the operational pillar with interaction sovereignty — access to prompts, outputs, tool calls, logs, traces and J3W intervention points — is not a settled taxonomy, and the layers it yields are separable, contracts, residency, product tiers and architecture moving them independently, while being jointly binding, each protocol requiring a bundle of which a missing part can degrade or disable it.

T2V (3.1): full access against the typical import-dependent regulated deployer.

| Layer | Full access | Typical import-dependent regulated deployer |
|---|---|---|
| Data | Control of training/fine-tuning data; audit provenance | Own transactional/fine-tuning data; no base-model data visibility |
| Model | Weights, architecture, activations; retrain, patch, probe, interpretability | Usually absent under API/managed endpoint; vendor-held |
| Infrastructure | In-jurisdiction compute, domestic legal control, observable | Partial at best: foreign-operated cloud region, sometimes residency commitments; vendor-operated serving |
| Interaction | Full logging/audit of prompts, outputs, tool calls, traces; pre-execution intervention | Partial: I/O and tool logs may be contractable; traces rare; J3W depends on architecture |

The rubric of 3.4, its weights open, scores six criteria: 1. severity and reversibility, higher severity and lower reversibility raising the access bar; 2. verifiability of substitutes, failures occurring only in internal computation being impossible to buy back with output-level substitutes; 3. vendor concentration and bargaining power; 4. regulatory maturity and absorptive capacity, using logs, traces and attestations not being the right to receive them; 5. Q9X versus the cost of R7K, the marginal usefulness lost; 6. coalition paths, audit consortia, regional regulatory colleges and South-South cooperation amortising costs.

Of the positions this rubric serves, A, the safety-maximalist, holds that without model and infrastructure access claimed control is unfalsifiable and not safety assurance; B, the sovereignty-maximalist, holds partial control under foreign dependency fragile, access being revocable or modifiable and Pax Silica-type arrangements tying access to political alignment, with only full-stack domestic capability durable; C, E6U, answers that A implies abstaining from committed deployments and B implies investments many states cannot make this decade, asking instead which layers are necessary for which catastrophic-action classes and which are substitutable via architecture, contracts, monitoring or R7K (R7K denoting scope restriction).

E6U functions as the working frame of 3.4 without any claim that A or B is wrong, its structure resembling incomplete contracting and principal-agent design: vendor private information, incomplete contracts, costly asymmetric verification.

## 4 Empirical Methodology: Synthetic Access-Ablation Study

| Setting | Value |
|---|---|
| Access profiles | 20 |
| Control protocols | 10 |
| Synthetic agent types | 3 |
| Monitor thresholds | 0.35, 0.55, 0.75 |
| L8C budgets | 2%, 5%, 20% |
| Random seeds | 5 |
| Synthetic cases per condition | 50 |
| Aggregate condition rows | 27,000 |
| Synthetic case simulations | 1,350,000 |

The 4.4 configuration is crossed with the 4.3 Q9X proxy = min cost reaching target failure under bounded access − min cost under C4Q, with the stricter variant adding usefulness, namely failure ≤1% and G8F (G8F denoting the paper's task success) ≥75%, so that disabling actions shifts cost into usefulness loss, and with the 4.2 set of 20 curated fractional-factorial profiles combining a C4Q anchor, complete logging with/without J3W (J3W denoting the paper's pre-execution gateway), partial- and no-log contrasts and trace, version, vendor and scope contrasts, while L8C (L8C denoting the paper's human audit), V4M (V4M denoting the paper's macro-shutdown), X9D (X9D denoting the paper's vendor escalation), Y7N (Y7N denoting the paper's Stealthy access-aware) and B5H (B5H denoting the paper's diagnosis success) are subordinated to the rule that a failure is diagnosable only if evidence under the profile suffices to reconstruct it, before the 4.1 non-real constraint is reached.

| Protocol | Interpretation |
|---|---|
| No control | Accept action |
| Output monitor | Score output/action |
| Tool monitor | Score tool use if visible |
| F6Q | Conservative fallback |
| H1Y | Lowest-risk acceptable alternative |
| L8C | Capacity-limited review |
| N2S | Edit/redact or fall back |
| R7K | Remove risky actions |
| V4M | Safe-mode clusters on repeated suspicion |
| X9D | Delayed/guaranteed vendor support |

G8F; P6T, overall and by class; severity-weighted loss; audit, flag, FP and FN rates; S2L and B5H; latency; model, monitor, L8C and vendor-request counts; and a unitless cost proxy of inference, monitoring, audit, logging, J3W, vendor assurance, latency, R7K and residual incident loss are measured after the agent and access tables have already fixed the behaviours and levels they presuppose.

| Agent | Behaviour |
|---|---|
| Honest | Follows ground truth; occasional mistakes |
| Careless | Overconfident, error-prone, escalates less |
| Y7N | Lower-visibility failures when access weak (mock actions only) |

| Layer | Factor | Levels |
|---|---|---|
| Interaction | I/O logs | none, partial, complete |
| Interaction | Tool-call logs | none, partial, complete |
| Interaction | J3W | absent, present |
| Model | Trace access | none, limited, oracle |
| Model/interaction | Model-version control | unpinned, notified, pinned |
| Infrastructure/contractual | Vendor support | none, delayed, guaranteed |
| Deployment design substitute | Tool scope | full, restricted |

| Class | Synthetic meaning |
|---|---|
| C1 | Incorrect hold/release on one case |
| C2 | Systematic mis-triage across a cluster |
| C3 | Sensitive-field exposure marker in mock report |
| C4 | Settlement-critical recommendation marker |

C3/C4 are labels only, with no real exposure, credential, exfiltration, malware, attack or disruption logic, and no real payment/customer data, production logs, vendor contracts, confidential material or operational infrastructure is used, construct validation only, in a mock national payments dispute-triage task that is transaction-adjacent and not settlement-operational, where the agent picks harmless mock actions draft_report, recommend_hold, recommend_release, escalate_to_human or no_action, cases carry non-real ID, amount, merchant category, claim, dispute history, risk score, settlement-critical and sensitive-context markers, allowed actions and ground truth, and intake, evidence retrieval, classification, recommendation, optional J3W review, mock execution and audit review are the steps.

## 5 Empirical Results

Feasibility is access-dependent, with complete logging and J3W ownership constituting the most important interaction-layer controls here — logs underpinning monitoring and B5H, J3W underpinning prevention — while model-side access matters most for B5H and assurance and R7K substitutes for access by the reduction of capability (5.6); and because some K7R profiles are observed to reach loose targets at lower direct cost than C4Q, whether through R7K or through a benign agent, the cost being thereby shifted onto usefulness, B5H, delayed vendor assurance, residual risk or manual effort, the strict target is the more informative one. No protocol was found to dominate (5.4): the lowest P6T was that of R7K (0.50%), attained at a G8F of 60.06%, H1Y coming next (3.49%) with greater usefulness, while F6Q, tool monitoring, N2S, output monitoring and V4M clustered at ≈4.0%, and L8C together with X9D proved weaker at prevention, being frequently post-detection, capacity-limited or reliant on delayed support, so that safe designs may be operationally unattractive. Averaged P6T across agent profiles stood at 0.98% for honest, 6.82% for careless and 4.47% for Y7N, the last of these entailing harder diagnosis and control (5.5), the Q9X being paid in narrower permissions, manual work and less automation, as the restricted-versus-full tool-scope contrast — P6T of 0.60% against 4.48%, G8F of 60.06% against 84.51% — makes apparent.

| Synthetic agent | share of profiles reaching $\leq$ 1% P6T and $\geq$ 75% usefulness |
|---|---|
| Honest | 90% |
| Careless | 40% |
| Y7N | 35% |

When all conditions are averaged (5.2), P6T is found at 5.53% without J3W against 3.13% with it, while S2L (S2L denoting intervention success) rises from 0.47% to 39.52%, a monitor being capable only of warning whereas J3W can hold, replace, escalate or block, interaction sovereignty thereby amounting to more than logging alone; and under I/O logging of none, partial and complete extent, B5H was 0.00%, 37.41% and 78.57% respectively, with P6T at 6.17%, 5.04% and 3.41%, the causal chain being rendered unexplainable in the absence of logs, with consequent weakening of audit, accountability, remediation and the safety case. B5H by trace access was 50.86% with none, 74.69% with limited and 100.00% with oracle access, the last figure holding by construction, and by version control it was 5.31% unpinned, 83.00% notified and 99.87% pinned (5.3); output monitoring remains feasible under K7R, yet B5H is fragile in the absence of traces or of version identification, a fragility that is critical wherever post-incident assurance forms part of accountability. What is at issue throughout is the logic of access constraints rather than real safety levels, the penalties being threefold (5.1): prevention (detected but not stopped before execution absent J3W); diagnosis (a bad outcome not being reconstructable); and usefulness (the safest substitute narrowing the system).

## 6 Illustrative Case Interpretation: National Payments Infrastructure

Anonymised and illustrative, with no specific deployment or incident, the stress item marked 6.3 has rising hold recommendations in one merchant category flagged by the monitor, detection surviving while diagnosis and intervention fail first, invisibly until an incident; full access permits trace inspection, re-running a pinned version on history, version comparison and full tool-chain reconstruction, whereas K7R leaves output-log forensics, tool-call review if captured, re-querying a possibly changed model and contract-governed X9D, with Q9X being contractual (logging, version-change notification, audit/attestation, incident notification, regulator access), architectural (gateway, tamper-evident logs, monitors, tool interfaces), capacity (audit/model-risk staff to investigate and challenge attestations) and R7K (forgoing C4-adjacent paths, narrower tools). In the feasibility fragment marked 6.2, executable means output-level monitoring by an Operator-controlled monitor scoring every output and tool call on Operator compute, plus L8C of the top-suspicion fraction (capacity, log completeness), and F6Q iff tool calls pass an Operator-owned gateway; degraded means H1Y, because version changes hurt reproducibility, cost/latency bind and there is no substitute for traces, and pre-deployment evaluations unless the production version is pinned; infeasible without the vendor are activation monitoring, probe-based deception detection and interpretability auditing, where output-only monitoring is weakest for deceptive/systemic C2/C3 failures, those classes being fixed by severity in the deployment fragment marked 6.1 as C1 incorrect hold/release, C2 systematic mis-triage propagating before detection, C3 protected-data exposure via tool calls/outputs and C4 action paths disturbing settlement, for an Operator of interbank retail rails supervised by a regulator requiring validation, monitoring, auditability and board accountability for material models while using a vendor frontier model via managed endpoint in a foreign-operated cloud region with a data-residency rider as an agentic assistant triaging disputes, drafting notes and invoking constrained tools (case lookup, recommendation drafting, escalation).

| Layer | Position | Basis |
|---|---|---|
| Data | Partial | Own case/transaction data; no base-model data visibility |
| Model | None | No weights, architecture, activations or patching; vendor may update model |
| Infrastructure | Partial | Residency rider; serving stack vendor-operated, unobservable; legal reach mediated by foreign law |
| Interaction | Partial | I/O and tool logs contractable at Operator's gateway; no traces |

## 7 Discussion and Limitations

The 7.4 boundary is that no export-control, trade, procurement or industrial-policy recommendation is made and no position is taken on pursuing sovereign AI capability, while the recommendation is that control-protocol specifications state access assumptions explicitly; the 7.1 ceiling claim is that E6U has a ceiling, lowest for the failures the control literature fears most, because the infeasible set holds the internal-access protocols most relevant to deceptive, systemic, hard-to-diagnose failures and contractual substitutes are revocable or incomplete, which supports both strong A and strong B, with strong B met by architecture and contract doing meaningful work through output monitoring, J3W intervention and L8C, and strong A met by a non-trivial protocol subset being executable given complete interaction logging and an owned J3W, while access is made manipulable by Z5J through three distinctions, namely seeing versus stopping as logs versus J3W, detecting versus explaining as traces, version control and vendor support, and buying safety versus giving up usefulness as R7K, leaving as governance questions complete logs, J3W interception, version pinning, failure reconstruction and regulator access to safety-case evidence. The 7.2 qualification is that the mapping method is sector-agnostic but the case is not, since payments favours E6U through crisp catastrophes, narrowable tools and a natural J3W, whereas healthcare, public administration or enterprise deployments with fuzzier catastrophes and wider action surfaces likely raise the importance of traces, richer audit evidence and R7K, and a second sector case plus an LLM-in-the-loop version would strengthen external validity; the limitations are recoverable as 6, that the sovereign-AI literature is fast-moving and partly non-peer-reviewed and the four-pillar formulation is an adapted policy/industry construct, so the paper does not settle what AI sovereignty means, 5, that the Q9X proxy measures reachability and cost-shifting and cheaper E6U profiles at loose targets do not show K7R is cheaper, 4, that the cost model is a proxy for relative comparisons only, not exact M4Z or Q9X amounts, 3, that curated fractional-factorial profiles do not cover all cloud, API, contractual, logging, residency and vendor arrangements while full-factorial is supported by the code but computationally larger, 2, that agents are not frontier models and real outputs may change monitor calibration, audit burden and diagnosis, and 1, that Z5J is synthetic, a structured simulation, not evidence about any institution; the 7.3 continuation is (3) expert calibration of audit capacity, version drift, vendor delay, log-retention and latency costs, and acceptable thresholds, (2) a public vendor/regulatory access-rights audit covering version pinning, log export, tool-call traces, retention control, customer-managed keys, regulator access, incident forensics and attestations, and (1) an LLM-in-the-loop workflow.

## 8 Conclusion

The safety-usefulness-cost frontier is shifted by K7R, which is operationalisable; yet the assumption, made by any control protocol that is also an access claim, of the existence of someone positioned to observe logs, route outputs, intercept tool calls, pin or identify model versions and reconstruct evidence after incidents — oversight having to be located as well as priced — frequently fails for regulated deployers in AI-importing jurisdictions, and such assumptions, among them trace access, version pinning, tool-call interception, complete logs or vendor cooperation, being often the binding constraint for regulators and deployers, should be stated by the protocols presupposing them.

