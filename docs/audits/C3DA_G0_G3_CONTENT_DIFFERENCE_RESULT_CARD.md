# G0/G3 内容差异审计结果卡

任务：`C3DA_G0_G3_CONTENT_DIFFERENCE_AUDIT_V1`

## 有效性

`COMPLETE_READ_ONLY_AUDIT`。读取的是刚刚完整重建中实际使用的 G0/G3 pseudo、增强和 final_train 文件；没有训练、没有改 selector/补全/增强逻辑，没有使用 target-test gold 做决策。hidden gold 仅用于 target-train 伪标签事后审计。

## OVERLAP

| 项目 | 数量 |
|---|---:|
| G0 unique triplets | 684 |
| G3 unique triplets | 665 |
| shared exact triplets | 506 |
| G0-only triplets | 178 |
| G3-only triplets | 159 |

这里是 unique exact triplet（去重后的精确方面-观点-情感三元组）统计；原始文件中的重复出现仍保留在行级统计中。

## G0_ONLY

```text
rows=144, triplets=271
single/multi/3plus rows=44/100/27
P/R/F1=46.13%/45.13%/45.62%
TP/FP/FN=125/146/152
multi P/R/F1=48.63%/47.65%/48.14%
3plus P/R/F1=51.85%/58.33%/54.90%
single F1=24.00%
unique aspects=182, unique opinions=155
mean sentence length=18.04 words
mean gold triplets=1.92
sentiment pos/neg/neu=201/62/8
```

## G3_ONLY

```text
rows=128, triplets=231
single/multi/3plus rows=47/81/22
single/multi/3plus F1=27.27%/44.26%/50.38%
P/R/F1=44.16%/40.16%/42.06%
TP/FP/FN=102/129/152
unique aspects=148, unique opinions=148
mean sentence length=19.00 words
mean gold triplets=1.98
sentiment pos/neg/neu=170/59/2
```

G0-only 的质量略高于 G3-only，尤其在 multi 和 3+ 结构上；G0-only 不是主要由新增 FP 构成。G0-only 的错误类型（wrong boundary/partial/duplicate 等）无法仅凭现有字段稳定分类，记为 `UNAVAILABLE`。

## COMPLEXITY

G3-only 句子略长，平均金标三元组略多，说明 G3-only 内容在句子和标注结构上略复杂；但差距有限，不能单独解释最终 F1 的2.10个百分点反转。

## COMPLETE_MULTI_AMPLIFICATION

逐伪标签后代追踪字段不足，G0-only/G3-only 被 complete_multi 修改、保留或新增的精确数量为 `UNAVAILABLE`。不能据此声称补全放大了某组噪声。

## AUGMENTATION_AMPLIFICATION

```text
G0 selected augmentation=116 rows / 182 triplets
G3 selected augmentation=101 rows / 151 triplets
G0 augmentation rows / pseudo rows=21.28%
G3 augmentation rows / pseudo rows=18.26%
```

G0 的增强相对放大比例更高，但 G0-only pseudo 到增强后代的精确映射为 `UNAVAILABLE`。

## FINAL_TRAIN_AMPLIFICATION

```text
G0 final_train=1565 rows / 2450 triplets / 644 multi rows / density=1.5655
G3 final_train=1558 rows / 2404 triplets / 612 multi rows / density=1.5430
```

G0 的总三元组、多三元组和密度均更高，不能归因于 G0 的 multi 供给不足。最终训练集总行数仅差7行。

## FINAL_TARGET

```text
G0 Raw P/R/F1=55.53%/49.69%/52.45%，TP/FP/FN=241/193/244
G3 Raw P/R/F1=57.40%/51.96%/54.55%，TP/FP/FN=252/187/233
```

## PRIMARY_PATTERN

`F`：多个因素共同存在，但证据最强的表述是“G0-only 内容质量虽略高，G3-only 内容略复杂且更可能提供适合目标域泛化的结构；最终模型对 G3 分布的吸收更好”。

## SECONDARY_PATTERN

`C` 不成立（G0 multi 供给并不弱）；`D/E` 部分成立（G0 最终目标域同时多11个 FN 和多6个 FP）。

## LIMITATIONS

无法可靠计算：G0-only/G3-only 到 complete_multi 的逐行保留率、增强后代归属、错误类型细分、target-specific 元素集合。因此不能把反转归因到某个单独下游阶段。

`NEED_CROSS_EXPERIMENT=NO`（本任务禁止交叉实验，当前审计证据也不足以提出必要性）。

`TARGET_TEST_GOLD_USED=NO`

`NEXT_ALLOWED_ACTION=RETURN_TO_CHAT_SOL`
