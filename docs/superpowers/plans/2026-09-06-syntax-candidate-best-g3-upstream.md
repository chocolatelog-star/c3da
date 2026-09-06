# G3 上游句法候选选择实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在复用 G3 最佳上游产物的前提下，为 aspect/opinion 增强候选加入句法兼容排序、受控回退和完整审计，并支持 A1/A2/A3 三个下游变体。

**Architecture:** 在 `t5_aste_pipeline.py` 中实现纯函数式句法特征归一化、候选评分和审计计数；在现有 `build_augmentation_requests` 的候选入口接入通道开关，保留旧 selector 作为回退。`run_bgca_aste_stage1_pairs.py` 只负责参数传递、配置快照和下游结果汇总，不改上游产物。

**Tech Stack:** Python 3.10、argparse、现有 JSONL/parser/cache、pytest、Git。

---

### Task 1: 添加句法评分单元测试

**Files:**
- Create: `test_syntax_candidate_selection.py`
- Test: `CD-C3DA/t5_aste_pipeline.py`

- [ ] **Step 1: Write the failing tests**

测试覆盖：三类句法特征分别兼容时得分更高；部分兼容仍可选；没有可接受候选时返回回退标志；A1 只影响 aspect、A2 只影响 opinion、A3 同时影响两者；审计计数正确累加。

```python
from t5_aste_pipeline import rank_syntax_candidates

def test_syntax_rank_prefers_compatible_candidate():
    source = {"upos": "NOUN", "dependency_relation": "nsubj", "head_pos": "VERB"}
    candidates = [
        {"text": "bad", "upos": "ADJ", "dependency_relation": "amod", "head_pos": "NOUN"},
        {"text": "good", "upos": "NOUN", "dependency_relation": "nsubj", "head_pos": "VERB"},
    ]
    ranked, audit = rank_syntax_candidates(source, candidates, min_acceptable_score=1)
    assert ranked[0]["text"] == "good"
    assert audit["syntax_fallback"] is False

def test_syntax_rank_falls_back_when_no_candidate_is_acceptable():
    source = {"upos": "NOUN", "dependency_relation": "nsubj", "head_pos": "VERB"}
    candidates = [{"text": "bad", "upos": "ADV", "dependency_relation": "advmod", "head_pos": "ADJ"}]
    ranked, audit = rank_syntax_candidates(source, candidates, min_acceptable_score=3)
    assert ranked == candidates
    assert audit["syntax_fallback"] is True
```

- [ ] **Step 2: Run tests to verify failure**

Run: `python -m pytest test_syntax_candidate_selection.py -q`
Expected: FAIL because `rank_syntax_candidates` does not exist.

### Task 2: 实现 occurrence-level 句法候选排序和审计

**Files:**
- Modify: `CD-C3DA/t5_aste_pipeline.py` near augmentation helper functions and `build_augmentation_requests`.

- [ ] **Step 1: Add normalized feature helpers**

实现 `normalize_syntax_metadata`、`syntax_compatibility_score` 和 `rank_syntax_candidates`。评分分别统计 UPOS、dependency relation、head POS；缺失字段不加分也不判定不兼容；`min_acceptable_score` 控制明显不兼容候选的回退阈值。

- [ ] **Step 2: Add candidate-bank metadata preservation**

从现有 parser/cache 结果读取 occurrence-level metadata；同文本不同出现位置必须保留不同 `source_row_id`、span 和句法字段。缺失 parser/cache 字段时保留候选并将对应审计项记为 unavailable，而不是伪造值。

- [ ] **Step 3: Add channel-aware mode**

新增 `syntax_candidate_mode`，取值 `none`、`aspect`、`opinion`、`dual`。aspect replacement 只在 `aspect`/`dual` 开启；opinion replacement 只在 `opinion`/`dual` 开启。每次请求保存 `syntax_fallback`、`syntax_score` 和 `syntax_compatibility`。

- [ ] **Step 4: Add audit aggregation**

汇总候选请求数、句法前后候选数、三类兼容率、高兼容率、无兼容请求、回退请求和候选变化率，写入 augmentation summary 和 manifest。

- [ ] **Step 5: Run unit tests**

Run: `python -m pytest test_syntax_candidate_selection.py test_augment_quality_filters.py -q`
Expected: PASS。

### Task 3: 接通 runner 参数并保持下游 recipe 一致

**Files:**
- Modify: `CD-C3DA/run_bgca_aste_stage1_pairs.py` argument parser and all augmentation subprocess command builders.
- Modify: `CD-C3DA/t5_aste_pipeline.py` argument parser and augmentation summary.
- Test: `CD-C3DA/test_run_bgca_stage1_pairs.py`。

- [ ] **Step 1: Add CLI flags**

增加 `--syntax_candidate_mode`、`--syntax_min_acceptable_score`、`--syntax_high_compatibility_score`，默认关闭句法候选，保证历史命令行为不变。

- [ ] **Step 2: Forward flags to every augmentation invocation**

A1/A2/A3 命令分别传 `aspect`、`opinion`、`dual`；所有其它参数固定为当前最佳 recipe，且不传 `--structure_preserving_augmentation`。

- [ ] **Step 3: Add parameter propagation regression test**

断言 dry-run 生成的 augmentation command 包含句法模式和阈值，并保留 Final DANN=0.03、batch=16、accumulation=2、seed=1000。

- [ ] **Step 4: Run parser and regression tests**

Run: `python -m pytest test_run_bgca_stage1_pairs.py test_phase_a_inference_contract.py -q`
Expected: PASS。

### Task 4: 本地静态验证和 Git 提交

**Files:**
- Modify: `CD-C3DA/t5_aste_pipeline.py`
- Modify: `CD-C3DA/run_bgca_aste_stage1_pairs.py`
- Create: `CD-C3DA/test_syntax_candidate_selection.py`

- [ ] **Step 1: Run full focused test set**

Run: `python -m pytest test_syntax_candidate_selection.py test_augment_quality_filters.py test_run_bgca_stage1_pairs.py -q`

- [ ] **Step 2: Run syntax compilation**

Run: `python -m py_compile t5_aste_pipeline.py run_bgca_aste_stage1_pairs.py`

- [ ] **Step 3: Review diff and commit**

Run: `git diff --check; git add t5_aste_pipeline.py run_bgca_aste_stage1_pairs.py test_syntax_candidate_selection.py docs/superpowers/plans/2026-09-06-syntax-candidate-best-g3-upstream.md; git commit -m "feat: add syntax-aware augmentation candidates"`

### Task 5: 服务器同步和只读实验命令检查

- [ ] **Step 1: 同步精确提交到服务器 Git 工作树并提交**

服务器只接收本地已验证提交，不直接编辑未提交代码；同步后检查 `git status` 和 commit hash 一致。

- [ ] **Step 2: 生成 A1/A2/A3 三条单行命令**

三组均复用 `/root/autodl-tmp/CD-C3DA-runs/reproduce_G3_dann003_20260906/phase_a_upstream_adapter`，只改变 `--syntax_candidate_mode`，并保留 `--keep_intermediates`。

- [ ] **Step 3: 给出持续监视命令**

提供日志、GPU 和磁盘监视命令；不自动启动实验，等待用户执行。
