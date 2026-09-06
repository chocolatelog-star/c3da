# G0/G3 增强失败审计结果卡

任务：`C3DA_G0_G3_AUGMENTATION_FAILURE_AUDIT_V1`

## VALIDITY

`COMPLETE_FOR_AVAILABLE_FIELDS`。G0/G3 选中增强样本均能通过 `base_id` 映射回实际 pseudo parent；未使用 target-test gold。句法 POS/依存角色需要额外图字段，当前标记 `UNAVAILABLE`。

## PARENT_CHILD_MAPPING

```text
G0: COMPLETE, 116/116
G3: COMPLETE, 101/101
```

## EDITED_TRIPLET_VALIDITY

| 指标 | G0 | G3 |
|---|---:|---:|
| edited rows | 116 | 101 |
| edited triplet present in output | 87 (75.00%) | 69 (68.32%) |
| aspect channel validity | 72/87 (82.76%) | 55/74 (74.32%) |
| opinion channel validity | 15/29 (51.72%) | 14/27 (51.85%) |

这里的 validity 是代理口径：增强后 label 中存在预期 new_triplet；不是人工语义金标。

## UNTOUCHED_TRIPLET_RETENTION

| parent 类型 | G0 保留率 | G3 保留率 |
|---|---:|---:|
| multi parent | 11/22 = 50.00% | 5/11 = 45.45% |
| 3+ parent | 9/14 = 64.29% | 2/10 = 20.00% |
| all parents with untouched | 20/36 = 55.56% | 7/21 = 33.33% |

G3 在 3+ 多三元组父样本上出现严重结构丢失。

## TRIPLET_COUNT_PRESERVATION

```text
G0: preserved 78/116 (67.24%), decreased 4, increased 34
G3: preserved 60/101 (59.41%), decreased 8, increased 33
```

## UNPLANNED_TRIPLETS

```text
G0: 76 unplanned triplets, 57/116 rows (49.14%)
G3: 76 unplanned triplets, 58/101 rows (57.43%)
```

按通道：

```text
G0 aspect: 46 triplets / 36 rows；opinion: 30 / 21
G3 aspect: 54 triplets / 39 rows；opinion: 22 / 19
```

## CHANNEL_DIAGNOSIS

方面通道的编辑有效性高于观点通道：

```text
G0 aspect=82.76%，opinion=51.72%
G3 aspect=74.32%，opinion=51.85%
```

未编辑 triplet 保留率也显示观点通道更差：

```text
G0 aspect=15/21=71.43%，opinion=5/15=33.33%
G3 aspect=7/11=63.64%，opinion=0/10=0%
```

观点通道是明显的次级问题，但其影响与多三元组结构破坏交叉存在。

## NOISE_AMPLIFICATION

父伪标签事后质量分类及其增强行：

```text
G0: correct 53，partial 28，incorrect 35；incorrect parent descendants=35/116=30.17%
G3: correct 46，partial 19，incorrect 36；incorrect parent descendants=36/101=35.64%
```

当前每个 parent 最多产生一个选中增强后代，因此平均 descendants per parent=1.0；不存在同一 parent 被多次选中放大的证据。G3 的低质量 parent 占比更高，但这不是唯一故障。

## SYNTAX_COMPATIBILITY

真实 POS/依存角色不在当前增强 JSONL 中，`SYNTAX_MISMATCH_RATE=UNAVAILABLE`。NLI 和 model filter 只证明样本通过现有过滤代理，不能等价为句法兼容。

## PRIMARY_FAILURE

`A_STRUCTURE_BREAK`

主要证据是 G3 的 3+ parent untouched retention 仅20%，count preservation 为0%，unplanned row rate 为80%。这直接违反“replacement 不应改变其它 triplet 且总数应保持”的增强设计原则。

## SECONDARY_FAILURE

`D_OPINION_CHANNEL`。观点替换的编辑有效性约52%，G3 观点通道未编辑 triplet 保留率为0%；但不能忽略方面通道也有较高 unplanned rate。

## RECOMMENDED_AUGMENTATION_CHANGE

只优先改：`Multi-Triplet Structure Preservation`（多三元组结构保持）。

理由：它对应最直接、可量化且最严重的失败信号，尤其是 G3 的 3+ 句子。不要在本任务中自动修改算法；该建议仅供下一步 Chat 决策。

## KEY_EVIDENCE

1. parent-child mapping 完整：G0=116/116，G3=101/101。
2. G3 3+ parent 未编辑 triplet 保留率20%，G0为64.29%。
3. G3 3+ parent triplet-count preservation=0%，unplanned row rate=80%。
4. G3 overall edited validity=68.32%，低于 G0 的75.00%；unplanned row rate=57.43%，高于 G0 的49.14%。
5. G3 incorrect-parent augmentation ratio=35.64%，高于 G0 的30.17%，但每个 parent 平均只有1个后代，未发现重复放大。

`TARGET_TEST_GOLD_USED=NO`

`NEXT_ALLOWED_ACTION=RETURN_TO_CHAT_SOL`
