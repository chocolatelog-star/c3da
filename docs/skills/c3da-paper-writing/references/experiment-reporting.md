# Experiment Reporting

For manuscript-facing experiments record:

```text
RUN_ID
TYPE
DIRECTION
SEED_OR_SEEDS
CODE
PARENT_RUN
REUSE_DEPTH
MODEL_SELECTION_RULE
TARGET_TEST_USAGE
PRIMARY_METRICS
STRUCTURAL_METRICS
GATE
FINAL_STATUS
```

---

## DIAGNOSTIC

Supports:

- motivation;
- mechanism;
- error attribution;
- experimental design.

Does not by itself establish final target-domain model performance.

---

## QUICK_ABLATION

Supports controlled evaluation of one research variable.

May justify promotion or rejection.

Do not automatically call its score the formal project best.

---

## FULL_RUN

Used for formal end-to-end performance claims when it satisfies project protocol.

---

## While experiments are incomplete

Result sections may contain placeholders:

```text
[MAIN RESULT PENDING]
[ABLATION PENDING]
[MULTI-SEED PENDING]
[REVERSE-DIRECTION CONFIRMATION PENDING]
```

Do not replace placeholders using a temporary run simply to make the draft appear complete.

---

## Comparison provenance

Clearly distinguish:

- externally reported result;
- project-reproduced baseline;
- current project result.

Do not mix incomparable:

- datasets;
- directions;
- splits;
- metrics;
- supervision;
- evaluation scripts

without disclosure.
