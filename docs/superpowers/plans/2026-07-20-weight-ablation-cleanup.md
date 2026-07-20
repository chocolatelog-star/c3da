# Weight Ablation And Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Clean failed experiment artifacts and add parameterized final-stage weight ablation commands around the current best CD-C3DA flow.

**Architecture:** Keep the current best upstream fixed. Add minimal routing in `run_bgca_aste_stage1_pairs.py` and, only if required, `t5_aste_pipeline.py` so `pseudo_weight`, `augment_weight`, and complete-multi pseudo weights can be varied without overwriting previous outputs.

**Tech Stack:** Python, HuggingFace Transformers, T5, PowerShell, Git.

---

### Task 1: Cleanup Failed Artifacts

**Files:**
- Remove directories under `J:\nlp\CD-C3DA\runs` and `J:\nlp\CD-C3DA\.worktrees\complete-multitriplet-ablation\runs` that are documented failed experiment outputs.
- Preserve metric files and source code history when practical.

- [ ] List candidate failed experiment directories.
- [ ] Remove only directories explicitly tied to failed experiments.
- [ ] Record cleanup in `实验记录与模型索引_CN.md`.

### Task 2: Parameterize Best-Flow Weight Ablations

**Files:**
- Modify: `J:\nlp\CD-C3DA\run_bgca_aste_stage1_pairs.py`
- Modify if needed: `J:\nlp\CD-C3DA\t5_aste_pipeline.py`
- Test: `J:\nlp\CD-C3DA\test_run_bgca_stage1_pairs.py`

- [ ] Add arguments for final pseudo, augment, and complete-multi weights.
- [ ] Ensure result tags include the changed weights.
- [ ] Ensure resume status keys do not collide with old experiments.
- [ ] Add/adjust tests for parsing and tag isolation.

### Task 3: Validate And Document

**Files:**
- Modify: `J:\nlp\CD-C3DA\实验记录与模型索引_CN.md`

- [ ] Run focused unit tests.
- [ ] Run broader unit tests if time permits.
- [ ] Add complete CMD commands for RTX 3070 8GB settings.
- [ ] Commit changes to Git.
