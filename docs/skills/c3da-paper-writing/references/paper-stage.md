# Paper Stage

The manuscript uses one explicit lifecycle state.

---

## 1. DRAFT_DURING_RESEARCH

This is the default and current CD-C3DA paper stage while experiments remain incomplete.

### May write

- paper outline;
- Introduction background;
- Related Work;
- task definition;
- established base framework;
- datasets;
- metrics;
- experimental protocol;
- verified historical motivation;
- candidate method technical definitions;
- experiment questions;
- placeholder result tables.

### Must remain provisional

- final method contribution;
- candidate module effectiveness;
- final performance claims;
- final main-results table;
- final ablation claims;
- multi-seed robustness claims;
- SOTA claims;
- final Abstract numbers;
- final Conclusion.

### Required behavior

Unresolved evidence goes into:

`.paper/pending_evidence.md`

Candidate claims go into:

`.paper/claim_evidence_ledger.md`

---

## 2. METHOD_STABILIZED

May enter only when Chat Sol determines that the central method has passed the required research Gates.

At this stage:

- core Method may be locked;
- method diagram may be finalized;
- technical contribution wording may stabilize;
- planned ablations may be finalized.

Still avoid final SOTA or robustness claims if formal runs are incomplete.

---

## 3. SUBMISSION_READY

May enter only when:

- final method is frozen;
- required FULL_RUN experiments are complete;
- required directions are complete;
- required seeds are complete;
- main baselines are verified;
- citations are verified;
- main tables are final;
- limitations are documented;
- current venue requirements are checked.

Only here may the manuscript lock:

- final Abstract;
- final Contributions;
- final Results;
- final Conclusion;
- SOTA-style claims;
- submission checklist.

Paper stage may only be promoted by an explicit research/writing decision.

Do not promote it automatically because a single experiment succeeds.
