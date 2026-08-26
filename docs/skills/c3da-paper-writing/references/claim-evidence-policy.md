# Claim-Evidence Policy

Maintain the paper claim ledger using:

| ID | Claim | Evidence | Class | Scope | Status | Allowed wording | Paper locations |
|---|---|---|---|---|---|---|---|

Status:

- SUPPORTED
- PARTIAL
- PROVISIONAL
- GAP
- REJECTED

---

## Rules

### SUPPORTED

May appear within the supported scope.

### PARTIAL

May appear only with bounded wording.

### PROVISIONAL

May appear only as a hypothesis, candidate method, or pending research question.

### GAP

Do not make the claim.

Record what evidence is required.

### REJECTED

Do not use as a positive contribution.

May appear as negative evidence or analysis where useful.

---

## Evidence hierarchy

Possible classes:

- FORMAL_RESULT
- QUICK_ABLATION
- DIAGNOSTIC
- REPRODUCED_BASELINE
- EXTERNAL_REPORTED
- PROVISIONAL

Diagnostic evidence cannot automatically support formal performance claims.

---

## Causal strength

Words requiring strong controlled evidence:

- proves
- causes
- eliminates
- solves
- consistently outperforms
- establishes that X is the sole cause

Safer mechanism language:

- indicates
- identifies
- suggests
- is consistent with
- is associated with
- is concentrated in
- supports investigating

---

## Example

```text
CLAIM:
Element absence is a dominant failure category in the analyzed held-out setting.

EVIDENCE:
Anchor-disjoint diagnostic.

CLASS:
DIAGNOSTIC

STATUS:
SUPPORTED_WITH_SCOPE

ALLOWED:
"Our diagnostics identify element absence as the dominant failure category in this setting."

FORBIDDEN:
"We prove that element absence is the universal cause of cross-domain ASTE failure."
```

---

## Pending candidate example

```text
CLAIM:
The syntactic graph adapter improves multi-triplet pseudo-label quality.

STATUS:
PROVISIONAL

EVIDENCE:
Pending.

REQUIRED:
- implementation entry PASS
- source-dev structural Gate
- target-unlabeled pseudo Gate
- formal downstream evaluation if promoted

ALLOWED NOW:
"We investigate whether explicit syntactic structure can improve multi-triplet pseudo-label modeling."

FORBIDDEN NOW:
"Our syntactic graph adapter improves multi-triplet recall."
```
