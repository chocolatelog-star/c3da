---
name: c3da-paper-writing
description: Use when drafting, revising, auditing, or preparing manuscript content for the CD-C3DA cross-domain ASTE project, including ACL, EMNLP, NAACL, EACL, ARR papers, rebuttals, tables, figures, claims, citations, and submission materials.
---

# C3DA Paper Writing

## Purpose

This skill governs scientific writing for the CD-C3DA cross-domain ASTE project.

It is designed for a project in which research, experiments, and manuscript writing proceed in parallel.

It does not decide research routes and does not replace the experiment protocol.

Research facts remain governed by the project source-of-truth files.

---

## Source of Truth

Before writing substantive scientific content, read in this order:

1. `AGENTS.md`
2. `.ai/PROJECT_STATE.md`
3. `.ai/CURRENT_TASK.md`
4. `.ai/DECISION_LOG.md`
5. `实验记录与模型索引_CN.md`
6. `03_CD-C3DA下一阶段改进计划_CN.md`
7. `docs/skills/c3da-experiment-workflow/SKILL.md`

The paper-writing state under `.paper/` is secondary.

If `.paper/` conflicts with authoritative project state, project state wins.

Never resolve conflicting experimental facts by guessing.

---

## Core Principle

CD-C3DA is written while the research is still progressing.

Therefore:

> Stable facts may be written now.
> Candidate mechanisms must carry their current research status.
> Missing evidence must remain explicit placeholders.
> Only evidence that passes the relevant research Gate may become a final scientific claim.

The manuscript must never get ahead of the evidence.

---

## Paper Stages

Exactly one paper stage should be recorded in `.paper/context.md`.

Allowed values:

### `DRAFT_DURING_RESEARCH`

Use while the final method and experimental results are still evolving.

This is the current stage.

Write stable sections and manuscript structure.

Do not freeze final performance claims.

### `METHOD_STABILIZED`

Use only after the central method has survived its required research Gates and is unlikely to change materially.

The Method section may be locked.

Some main results may still remain pending.

### `SUBMISSION_READY`

Use only after the final method, required formal experiments, baseline verification, robustness evidence, and submission materials are complete.

Only at this stage may the Abstract, final Contributions, main Results, Conclusion, and SOTA-style claims be considered locked.

See `references/paper-stage.md`.

---

## Research Roles

### Chat Sol

Owns:

- scientific narrative;
- paper story;
- contribution framing;
- claim strength;
- method interpretation;
- experiment interpretation;
- Related Work positioning;
- final scientific wording.

### Codex Luna

Owns:

- writing approved content into Markdown/LaTeX;
- formatting;
- mechanical updates;
- tables from verified evidence;
- citation entries after verification;
- paper-state synchronization.

Codex Luna must not independently strengthen scientific claims.

### Work Luna

Use for:

- large result tables;
- many runs;
- prediction files;
- paired analysis;
- bootstrap;
- claim-evidence consistency checks;
- table/text numerical audits.

### Codex Sol

Not part of normal manuscript writing.

Use only if a paper claim depends on complicated implementation semantics requiring high-risk engineering verification.

---

## Evidence Classes

All scientific evidence must be classified.

### `FORMAL_RESULT`

Completed experiment eligible for formal performance reporting.

### `QUICK_ABLATION`

Controlled single-variable experiment used for research screening or ablation evidence.

### `DIAGNOSTIC`

Mechanism-oriented or read-only evidence.

May support motivation and analysis.

Must not be presented as formal target-domain model performance.

### `REPRODUCED_BASELINE`

Baseline reproduced under the project protocol.

### `EXTERNAL_REPORTED`

Result quoted from an external publication.

### `PROVISIONAL`

Incomplete, unapproved, or still-running evidence.

Cannot support a final claim.

---

## Stable vs Provisional Content

During `DRAFT_DURING_RESEARCH`, divide manuscript content into two categories.

### Stable content

May be written normally when supported by current project state:

- task definition;
- cross-domain ASTE background;
- dataset descriptions;
- evaluation protocol;
- target-test isolation policy;
- established base pipeline;
- already verified historical evidence;
- Related Work;
- terminology;
- reproducibility protocol;
- established research motivation.

### Provisional content

Must be marked internally as provisional:

- candidate modules;
- currently running experiments;
- unresolved hypotheses;
- future ablations;
- unverified performance improvements;
- incomplete multi-seed results;
- tentative contribution wording.

Use markers such as:

`[PROVISIONAL METHOD]`

`[RESULT PENDING]`

`[EVIDENCE NEEDED]`

`[CITATION TO VERIFY]`

These markers must be removed only when the corresponding evidence is approved.

---

## Candidate Method Rule

A research candidate may be drafted before its experiment finishes.

For example:

`[PROVISIONAL METHOD — pending research Gate]`

Its technical definition may be written.

Its motivation may be written.

Its expected research question may be written.

Do not write that it:

- improves performance;
- solves the bottleneck;
- outperforms a baseline;
- reduces a failure mode;

until accepted evidence exists.

If the candidate later fails, remove it from the final Method section.

It may remain as:

- negative result;
- alternative design;
- diagnostic history;
- limitation;

only if scientifically useful.

---

## Claim-Evidence Gate

Every important manuscript claim must have:

```text
CLAIM:
EVIDENCE:
EVIDENCE_CLASS:
SCOPE:
STATUS:
ALLOWED_CERTAINTY:
FORBIDDEN_STRONGER_WORDING:
```

Allowed status:

- `SUPPORTED`
- `PARTIAL`
- `PROVISIONAL`
- `GAP`
- `REJECTED`

`GAP` and `PROVISIONAL` claims cannot appear as assertive conclusions.

See `references/claim-evidence-policy.md`.

---

## Non-Negotiable C3DA Rules

### Metrics

Strict raw F1 is the formal headline metric.

Fixed F1 is auxiliary.

Never replace raw F1 with fixed F1 in superiority claims.

### Six-direction scope

The long-term research objective concerns six transfer directions.

Do not convert:

- one successful direction;
- an average improvement;
- one seed;

into an all-direction superiority statement.

### Target-test isolation

Target-test gold is for final evaluation only.

It cannot justify:

- module selection;
- checkpoint selection;
- threshold selection;
- pseudo-label configuration;
- candidate selection;
- hyperparameter tuning;
- research-route choice.

Post-hoc target-test analysis must be labelled as post-hoc.

### Experiment identity

Always distinguish:

- `DIAGNOSTIC`
- `QUICK_ABLATION`
- `FULL_RUN`

### Failed routes

Failed routes must not be retroactively rewritten as successful components.

### Causal language

Use causal wording only when supported by a clean single-variable comparison.

Otherwise prefer:

- indicates;
- suggests;
- is consistent with;
- is associated with;
- is concentrated in.

---

## Writing During Research

Do not wait until all experiments finish before writing.

Instead use this loop:

```text
stable research fact
→ write manuscript section
→ register unresolved claim
→ run experiment
→ Chat Sol judges evidence
→ update claim ledger
→ promote / revise / remove manuscript claim
```

The paper should expose what evidence is still missing.

The paper must not invent that evidence.

---

## Section Rules

### Introduction

During research, Introduction may be drafted early.

It may include:

- task importance;
- established limitations;
- verified motivation;
- research question.

Do not lock final outcome claims until evidence is complete.

### Related Work

Write early.

Use literature search to test whether the proposed contribution is actually distinct.

Do not wait for final experiments.

### Method

Stable base architecture may be written early.

Candidate modules must be marked provisional until research Gates pass.

The final Method section must contain only surviving components.

### Experimental Setup

May be written early when protocol is stable.

### Results

Create the section structure early.

Use placeholders for unfinished evidence.

Do not fill pending cells with temporary or guessed values.

### Abstract

Do not lock numerical claims during `DRAFT_DURING_RESEARCH`.

Draft conceptual versions only.

### Contributions

May exist as provisional research contributions.

Final contribution wording must wait until the method and evidence stabilize.

### Conclusion

Do not lock until results stabilize.

---

## Citation Policy

Never invent a citation.

Never generate BibTeX from memory.

For important references verify:

- title;
- authors;
- year;
- venue;
- publication record;
- task and experimental setting.

Prefer:

1. ACL Anthology;
2. official proceedings/publisher;
3. arXiv;
4. reliable scholarly indexes as verification aids.

Unresolved references remain:

`[CITATION TO VERIFY]`

---

## Numerical Policy

Any manuscript number must resolve to an authoritative source.

Do not hard-code changing project baselines inside this Skill.

Always read the current project state and experiment index.

If two project files disagree:

stop and identify the conflict.

---

## Submission

ACL/ARR rules are time-sensitive.

Before submission always verify current official:

- template;
- anonymity requirements;
- page limit;
- Limitations requirements;
- Responsible NLP checklist;
- supplementary material rules.

Do not rely on an old version of this Skill for venue deadlines or formatting rules.

---

## Final Rule

During ongoing research:

> Write the scientific structure early.
> Write established facts normally.
> Mark candidate mechanisms as provisional.
> Keep missing results as explicit evidence slots.
> Promote a claim only after the project research protocol accepts its evidence.
