# CD-C3DA 跨域 ASTE 超越 BGCA 目标路线图

## 0. 当前推进重点：领域感知语义增强

六组 BGCA 风格跨域实验已经跑完，当前平均 `raw F1`（原始 F1）为 49.86，平均 `fixed F1`（修正 F1）为 51.59。主要短板仍然是 `recall`（召回率）偏低，尤其是 `laptop14 -> restaurant`（笔记本到餐馆）方向。

下一步不先扩大模型，也不先改 DANN（领域对抗）主体，contrastive learning（对比学习）也先放到后续单独消融。当前优先改数据增强：在已有 `domain_prefix_style=text`（文本式领域前缀）的基础上，把掩码增强里的观点词通道从随机成对替换改成 domain-aware semantic replacement（领域感知语义替换）。

当前保留并使用的领域前缀：

```text
text:    target domain: [laptop14] ; masked aspect edit: ...
```

新增观点词替换策略：

```text
--opinion_replacement_mode semantic_same_sentiment
```

它只在同一 sentiment（情感极性）内部替换 opinion（观点词），并优先选择目标域高精度伪标签里与当前 aspect（方面词）更相容的 opinion（观点词）。排序信号包括 target-domain opinion frequency（目标域观点词频率）、aspect-opinion co-occurrence（方面词-观点词共现）、target triplet count（目标域三元组共现）和 lexical similarity（词面相似度）。

判断标准：

```text
1. 先跑 rest16 -> laptop14 单组实验。
2. 与当前 text 前缀结果对比：raw F1=46.63，fixed F1=48.98。
3. 同时参考原六组基线同方向：raw F1=46.14，fixed F1=47.69。
4. 如果 raw F1 稳定提升，再扩展到六组跨域实验。
5. 如果新策略不如当前最好结果，先询问是否删除对应历史输出文件。
```

## 1. 项目目标

本项目的最终目标不是简单复现 C3DA，而是把 C3DA 改造成一个完整的跨域 ASTE 框架，并在跨域能力上超过 BGCA 约 1 到 2 个 F1 点。

目标任务是 ASTE：

```text
输入句子 -> 输出 (aspect, opinion, sentiment) 三元组
```

当前线性化格式：

```text
<pos> battery life <opinion> long ; <neg> screen <opinion> dark
```

严格跨域设定：

```text
source train/dev：有标签，只用于训练和验证
target train：训练时当作无标签，只用于生成伪标签和增强数据
target train gold：只允许做分析，不允许参与训练或选模型
target test：只用于最终评估
```

当前主要实验方向：

```text
rest16 -> laptop14
```

项目路径：

```text
J:\nlp\CD-C3DA
```

数据来源：

```text
J:\nlp\BGCA-master\data\aste\cross_domain
```

本地模型目录：

```text
J:\nlp\models
```

已经使用的主要模型：

```text
抽取器：J:\nlp\models\t5-base-py
生成器：J:\nlp\models\mrm8488-t5-base-finetuned-common_gen
NLI：J:\nlp\models\nli-deberta-v3-base-mnli-fever-anli
```

硬件约束：

```text
RTX 3070 8G
batch size 通常设为 1 或 2
gradient_accumulation_steps = 16
fp16
gradient_checkpointing
```

## 2. 当前项目进度

当前已经完成了从 C3DA 到完整跨域 ASTE 的第一版改造。

核心文件：

```text
J:\nlp\CD-C3DA\t5_aste_data.py
J:\nlp\CD-C3DA\t5_aste_pipeline.py
J:\nlp\CD-C3DA\t5_aste_augment.py
J:\nlp\CD-C3DA\t5_absa_train.py
J:\nlp\CD-C3DA\T5_C3DA_CROSS_DOMAIN_ASTE_README_CN.md
```

当前流程：

```text
1. prepare
   读取 BGCA ASTE 数据，构造 source train/dev、target unlabeled、target test。

2. extractor training
   用 t5-base-py 训练 text -> triplet 抽取器。

3. generator training
   用 common_gen T5 训练 label -> text 生成器。

4. pseudo
   用抽取器给 target train 生成伪三元组。

5. augment
   使用 C3DA 双通道生成增强数据，并用 aspect/opinion 一致性检查和 NLI filter 过滤。

6. final training
   用 source gold + target pseudo + C3DA augment 训练最终抽取器。

7. evaluate
   在 target test 上计算严格 micro-F1。
```

当前已经实现的模块：

```text
1. 完整 ASTE 三元组抽取
2. T5-only 抽取和生成
3. C3DA 双通道增强
4. label-invariant paraphrase prompt
5. aspect/opinion 必须出现在增强句子中
6. NLI filter 过滤 contradiction
7. target pseudo 质量分析
8. BGCA/TOL 风格的严格 micro-F1 计数
9. 数据来源加权训练
10. best checkpoint 选择，避免最后几轮过拟合
```

当前最好的实验结果：

```text
run_dir:
runs\t5_c3da_aste_rest16_to_laptop14_rsda_stage1

model:
models\final_weighted_ep5\best

precision: 0.4383
recall:    0.2495
micro-F1:  0.3180
TP: 135
FP: 173
FN: 406
```

之前对比结果：

```text
旧 v2 baseline:
precision 0.3357
recall    0.2662
F1        0.2969
TP 144, FP 285, FN 397

NLI 未加权:
precision 0.4013
recall    0.2218
F1        0.2857
TP 120, FP 179, FN 421

加权 ep5:
precision 0.4383
recall    0.2495
F1        0.3180
TP 135, FP 173, FN 406
```

当前结论：

```text
样本加权有效。
NLI filter 有助于降低 FP。
best checkpoint 有必要，最终抽取器在第 3 轮附近最好。
当前主要瓶颈仍然是 recall 偏低和 span 边界不准。
```

## 3. 之前参考过的论文和项目

### 3.1 原始 C3DA 论文

原始 C3DA 不是完整 ASTE，而是更接近：

```text
给定 aspect -> 判断 sentiment
```

原始 C3DA 的重要思想是双通道增强：

```text
AAC：aspect-aware channel
PAC：polarity-aware channel
```

在本项目里已经被改造成 ASTE 双通道：

```text
aspect_channel：
替换或引入目标域 aspect。

opinion_sentiment_channel：
把 opinion 和 sentiment 成对替换，避免只改 sentiment 造成三元组不一致。
```

注意：三元组里 opinion 和 sentiment 高度绑定，所以不应该单独替换 sentiment。

### 3.2 BGCA 项目

路径：

```text
J:\nlp\BGCA-master
```

BGCA 是本项目最重要的 baseline 和目标参考。

BGCA 的核心不是简单伪标签，而是双向生成：

```text
text -> label
label -> text
```

BGCA ASTE 脚本包含：

```text
t5-base
extraction-universal
25 epoch
init_tag english
data_gene
data_gene_extract
data_gene_extract_epochs 25
data_gene_epochs 25
use_same_model
data_gene_wt_constrained
model_filter
save_best
train_by_pair
```

BGCA 对本项目最值得借鉴的部分：

```text
1. extraction-universal 三元组线性化格式
2. init_tag english
3. constrained decoding
4. label -> text 数据生成
5. model_filter
6. edit-distance prediction fix
7. best checkpoint selection
8. 多阶段训练和多 pair 评估
```

当前 CD-C3DA 还缺 BGCA 的两个关键能力：

```text
1. BGCA-style model_filter
   generated sentence -> extractor -> predicted label
   如果 predicted label 和原始增强 label 不一致，就过滤。

2. edit-distance fix
   对预测出的 aspect/opinion 做原句词级修复。
   例如把 up 修成 set up，把不完整 opinion 修到原句中更接近的 span。
```

这两个模块很可能显著提高当前 F1。

### 3.3 TOL 项目

用户之前要求参考 `J:\nlp` 中其他 ABSA 或跨域项目，重点关注它们如何处理方面词抽取。

TOL/BGCA 类项目一般不是先外置一个独立方面词抽取器，再分类情感，而是把输出结构统一生成出来：

```text
ATE：抽 aspect
AOPE：抽 aspect-opinion pair
ASTE：抽 aspect-opinion-sentiment triplet
```

当前项目已经从 AESC/给定 aspect 分类，转向完整 ASTE 三元组生成，这是正确方向。

后续仍应参考 TOL/BGCA 的评估方式：

```text
严格 triplet exact match
重复预测惩罚 precision
必要时报告 raw F1 和 fixed F1
```

### 3.4 RSDA / 2024 Findings ACL 论文

用户给过：

```text
C:\Users\wwh\Desktop\2024.findings-acl.615.pdf
```

该论文启发了当前第一阶段的三个模块：

```text
1. NLI filter
2. label-invariant paraphrase prompt
3. pseudo-label quality analysis
```

RSDA 的核心思想可概括为：

```text
Me: text -> label
Mg: label -> text
target text -> pseudo label
pseudo label -> generated target text
NLI filter 过滤低质量生成文本
```

当前已经实现：

```text
NLI filter
label-invariant prompt
pseudo-label quality analysis
```

但当前 NLI filter 的局限是：

```text
NLI 只能过滤明显 contradiction。
NLI label 为 neutral 并不代表 ASTE 标签正确。
```

所以后续必须加 model_filter。

### 3.5 数据增强框架 PDF

用户曾给：

```text
C:\Users\wwh\Desktop\文档分类整理\跨域\数据增强框架.pdf
```

当时讨论的启发：

```text
可以用伪标签生成目标域训练信号。
可以用 NLI 或语义一致性过滤低质量数据。
增强不是越多越好，必须做质量控制。
```

当前项目已经证明：

```text
大量低质量增强数据会让 F1 从 29 掉到 22。
NLI + 加权能把 F1 拉到 31.8。
```

后续要继续强化：

```text
伪标签置信度
增强数据过滤
训练损失控制
```

### 3.6 领域特定片段感知 / GCN 论文

用户曾给：

```text
C:\Users\wwh\Desktop\文档分类整理\论文\s11761-024-00432-9_dual_智谱4Flash.pdf
```

当时讨论点：

```text
是否可以用领域特定片段感知替代简单高频词提取。
是否可以参考其中 GCN 来做领域图或词片段关系建模。
```

建议定位：

```text
短期不要直接上 GCN。
先把 BGCA-style baseline 补齐。
中期可以把领域特定片段感知用于：
  1. 构建 target aspect/opinion bank
  2. 筛选高置信目标域片段
  3. 指导 C3DA aspect_channel 替换
  4. 构建 domain-aware prompt
```

GCN 可以作为后续论文创新模块，但不应该在 baseline 未站稳时引入。

## 4. 当前主要问题诊断

### 4.1 F1 低于 BGCA 的原因

当前最优 F1 为 31.8，距离 BGCA 级别仍然很远。

原因不是单一超参，而是当前 CD-C3DA 还缺少 BGCA 的关键机制：

```text
1. 没有 edit-distance 修复，span 边界错误直接算错。
2. 没有 model_filter，NLI 不能保证 ASTE 标签正确。
3. 伪标签质量低，target pseudo F1 约 26。
4. 召回率低，测试集 gold triplets 541，当前只预测约 310。
5. neutral 类几乎抽不出来。
6. 生成器增强数据仍有语义漂移。
7. 训练轮数和训练策略还没有达到 BGCA 论文级设置。
```

### 4.2 当前错误类型

当前预测错误主要包括：

```text
1. 空预测
   gold 有三元组，但 pred 为空。

2. aspect span 不准
   gold: set up
   pred: up

3. opinion span 不准
   gold: not yet discovered
   pred: not yet

4. polarity 否定丢失
   gold: not enjoy
   pred: enjoy

5. 多三元组句子漏抽
   gold 有多个 triplet，pred 只抽一个。

6. target domain aspect 不稳定
   如 Windows 8、OS、keyboard、performance 等经常漏掉。
```

## 5. 总体实现路线

为了达到超过 BGCA 1 到 2 个 F1 点，建议分成五个大阶段推进。

```text
阶段 A：对齐 BGCA 评估和后处理
阶段 B：补齐 BGCA-style model_filter
阶段 C：提升伪标签召回和置信度建模
阶段 D：改造 C3DA 双通道增强质量
阶段 E：加入联合训练损失和领域片段感知创新
```

每个阶段都必须做消融验证，不能一次性全加。

## 6. 阶段 A：对齐 BGCA 评估和后处理

### 6.1 目标

先保证当前项目的评估方式可以和 BGCA 对齐。

当前项目已经实现 strict raw micro-F1，但还没有 BGCA 的 fixed_scores。

BGCA 会做：

```text
raw_scores
fixed_scores
```

其中 fixed_scores 会把生成结果中不在原句里的 aspect/opinion，用 edit distance 修回原句中的近似词。

### 6.2 要改的地方

建议新增文件：

```text
J:\nlp\CD-C3DA\t5_aste_postprocess.py
```

实现：

```text
1. recover_terms_with_editdistance(term, sentence_tokens)
2. fix_pred_triplets(pred_label, sentence)
3. evaluate raw and fixed F1
```

修改：

```text
J:\nlp\CD-C3DA\t5_aste_pipeline.py
```

在 `evaluate` 输出：

```text
aste_metrics_raw.json
aste_metrics_fixed.json
aste_predictions_raw_fixed.jsonl
```

### 6.3 验证方法

用当前最优模型直接评估，不需要重训：

```cmd
J: & cd J:\nlp\CD-C3DA & conda activate c3da & python t5_aste_pipeline.py evaluate --run_dir runs\t5_c3da_aste_rest16_to_laptop14_rsda_stage1 --model_path runs\t5_c3da_aste_rest16_to_laptop14_rsda_stage1\models\final_weighted_ep5\best --batch_size 2 --num_beams 4 --cuda 0
```

预期：

```text
fixed F1 应该高于 raw F1。
如果 fixed F1 提升明显，说明当前大量错误来自 span 边界不准。
```

### 6.4 阶段通过标准

```text
raw F1 保持 31.8 左右
fixed F1 有明显提升
错误文件中 set up/up、not yet discovered/not yet 等问题减少
```

## 7. 阶段 B：补齐 BGCA-style model_filter

### 7.1 目标

NLI filter 只判断句子间是否矛盾，不能判断 ASTE 标签是否正确。

BGCA-style model_filter 要判断：

```text
增强句子 generated text
-> 抽取器
-> predicted label
```

如果：

```text
predicted label != 增强时指定的 label
```

就删除该增强样本。

### 7.2 要改的地方

修改：

```text
J:\nlp\CD-C3DA\t5_aste_pipeline.py
```

在 `augment` 增加参数：

```text
--model_filter_path
--model_filter_batch_size
--model_filter_mode exact|fixed
```

流程：

```text
1. 生成增强数据
2. aspect/opinion 一致性过滤
3. NLI filter
4. model_filter
5. 写入 final_train
```

输出文件：

```text
c3da_two_channel_augmented_model_filter.jsonl
c3da_model_filter_analysis.json
c3da_model_filter_removed.jsonl
```

### 7.3 验证方法

先不重训，先看过滤统计：

```text
NLI 后增强数据：1107
model_filter 后剩余数量：待测
```

然后训练最终模型：

```text
source_weight = 1.0
pseudo_weight = 0.5
augment_weight = 0.2
num_train_epochs = 5
checkpoint_selection = best
```

### 7.4 阶段通过标准

```text
FP 继续下降
recall 不应明显下降
raw F1 > 31.8
fixed F1 进一步提升
```

如果 model_filter 过滤太狠，导致 recall 下降，则需要：

```text
只对 C3DA augment 做 model_filter
不要过滤 target pseudo
或者 model_filter_mode 使用 fixed
```

## 8. 阶段 C：提升伪标签召回和置信度建模

### 8.1 当前伪标签问题

当前 target pseudo 分析：

```text
target_rows: 906
pseudo_rows: 878
target_triplets_gold: 1456
pseudo_triplets: 974
pseudo_micro_f1: 0.2601
precision: 0.3244
recall: 0.2170
TP: 316
FP: 658
FN: 1140
```

问题：

```text
1. 伪标签 recall 很低。
2. neutral 几乎没有。
3. 多三元组句子漏抽严重。
4. 否定 opinion 容易丢失。
```

### 8.2 目标

提升 target pseudo 的质量，尤其是 recall。

### 8.3 可实现方法

#### 方法 1：多 checkpoint ensemble 伪标签

不要只用一个 extractor checkpoint。

使用：

```text
checkpoint-1
checkpoint-2
checkpoint-3
```

对 target train 生成多个 pseudo label。

合并策略：

```text
高精度模式：
只保留多个 checkpoint 都预测到的 triplet。

高召回模式：
取多个 checkpoint 的 union，再用 model_filter 或置信度过滤。
```

当前目标是提高 recall，所以建议：

```text
union + 后续过滤
```

#### 方法 2：多解码策略伪标签

对同一 extractor 使用：

```text
num_beams = 4
num_beams = 8
max_new_tokens = 128
```

合并多个预测结果。

之前单纯改变 evaluate 解码几乎没提升，但用于 pseudo-label union 可能有用。

#### 方法 3：confidence-aware pseudo loss

在 `target_pseudo_predictions_analysis.jsonl` 中记录每条伪标签置信度。

可用置信度来源：

```text
1. 生成序列平均 log probability
2. 多 checkpoint 一致性
3. model_filter 是否通过
4. NLI 是否支持
```

训练时：

```text
pseudo_weight = base_weight * confidence
```

### 8.4 验证方法

每次 pseudo 后必须看：

```text
target_pseudo_analysis.json
```

重点指标：

```text
pseudo_micro_f1
pseudo precision
pseudo recall
pseudo_triplets
empty_pseudo_rows
gold/pseudo sentiment distribution
```

目标：

```text
pseudo recall 从 0.217 提升到 0.30+
pseudo F1 从 0.260 提升到 0.32+
neutral 不再接近 0
```

## 9. 阶段 D：改造 C3DA 双通道增强质量

### 9.1 当前增强流程

当前增强：

```text
requests: 1744
generated: 1744
after_consistency_filter: 1311
NLI kept: 1107
NLI filtered contradiction: 204
```

增强数据分布：

```text
aspect_channel: 600
opinion_sentiment_channel: 507
```

问题：

```text
1. NLI label 为 neutral 的增强样本不一定 ASTE 正确。
2. 有些增强句子语义漂移。
3. 有些情感和 opinion 不匹配。
4. 有些 target aspect bank 来自错误伪标签。
```

### 9.2 目标

让 C3DA 增强从“能生成”变成“高质量生成”。

### 9.3 可实现方法

#### 方法 1：目标域 aspect/opinion bank 质量控制

当前 aspect bank 来自 pseudo labels。

后续应改成：

```text
只使用高置信 pseudo 中的 aspect/opinion
或者使用领域特定片段感知提取候选片段
```

参考之前用户给的领域特定片段感知论文。

短期可实现：

```text
aspect/opinion 至少出现 k 次
必须通过 extractor 多 checkpoint 一致性
必须出现在目标域原句中
```

#### 方法 2：增强 label 置信度

每个增强 request 继承其 base pseudo 的置信度。

训练时：

```text
augment_weight = 0.2 * base_pseudo_confidence
```

#### 方法 3：通道分开消融

不要总是两个通道一起训练。

必须分别跑：

```text
source + pseudo + aspect_channel
source + pseudo + opinion_sentiment_channel
source + pseudo + both channels
```

看哪个通道贡献更大。

### 9.4 验证方法

每次 augment 后看：

```text
c3da_augment_analysis.json
c3da_model_filter_analysis.json
final_train.jsonl 数据组成
```

最终比较：

```text
precision
recall
F1
TP/FP/FN
empty_pred_rows
pred_triplets
sentiment distribution
```

## 10. 阶段 E：联合训练损失和领域片段感知创新

### 10.1 当前损失函数

当前已经实现样本级加权：

```text
L_total = 1.0 * L_source + 0.5 * L_pseudo + 0.2 * L_aug
```

对应参数：

```text
--source_weight 1.0
--pseudo_weight 0.5
--augment_weight 0.2
```

这已经证明有效：

```text
F1 从 28.5 提升到 31.8
FP 明显下降
recall 回升
```

### 10.2 后续联合损失目标

最终可以设计：

```text
L_total =
  L_gold
  + λp * L_pseudo
  + λaug * L_aug
  + λcon * L_consistency
  + λdom * L_domain_fragment
```

解释：

```text
L_gold：
源域人工标注监督损失。

L_pseudo：
目标域伪标签监督损失。

L_aug：
C3DA 增强数据监督损失。

L_consistency：
原目标句子和增强句子的预测一致性约束。

L_domain_fragment：
领域片段感知约束，用于强化目标域 aspect/opinion 片段。
```

### 10.3 一致性损失建议

对同一个 base target sentence 和它的增强句子：

```text
base_text -> pseudo_label
aug_text -> aug_label
```

要求模型在共享 aspect/opinion/sentiment 上保持一致。

简化实现：

```text
先不做 hidden state KL。
先做 label-level consistency：
如果 aug label 中的 aspect/opinion/sentiment 和 base pseudo 可对齐，
则预测不一致时增加惩罚。
```

### 10.4 领域片段感知建议

参考用户给的领域特定片段感知论文，不建议一开始就上 GCN。

先做轻量版：

```text
1. 从 target train 中提取候选 noun phrase / domain phrase
2. 结合 pseudo label 和频次筛选领域片段
3. 形成 target fragment bank
4. 用于 aspect_channel 替换和 prompt 约束
```

后续再考虑：

```text
GCN domain graph
fragment-aspect-opinion graph
跨域片段对齐
```

## 11. 推荐实验顺序

为了避免混乱，必须按阶段做消融。

### 11.1 当前已完成基线

```text
A0: old v2 weak augment
F1 = 29.69

A1: NLI + no weighting
F1 = 28.57

A2: NLI + weighted loss + best checkpoint
F1 = 31.80
```

### 11.2 下一组必须跑的实验

第一组：评估对齐

```text
B1: A2 + edit-distance fixed evaluation
比较 raw F1 和 fixed F1
```

第二组：增强过滤

```text
C1: A2 + model_filter
C2: A2 + NLI + model_filter
C3: A2 + model_filter，不使用 NLI
```

第三组：通道消融

```text
D1: source + pseudo only
D2: source + pseudo + aspect_channel
D3: source + pseudo + opinion_sentiment_channel
D4: source + pseudo + both channels
```

第四组：伪标签召回

```text
E1: single checkpoint pseudo
E2: multi checkpoint union pseudo
E3: multi decode union pseudo
E4: union pseudo + model_filter
```

第五组：最终组合

```text
F1: best pseudo + best augment + weighted loss
F2: F1 + consistency loss
F3: F2 + domain fragment bank
```

## 12. 每阶段需要记录的指标

每次实验必须记录：

```text
precision
recall
micro-F1
TP
FP
FN
```

还必须记录：

```text
pred_triplets
gold_triplets
empty_pred_rows
multi_pred_rows
pred sentiment distribution
target_pseudo_analysis.json
c3da_augment_analysis.json
final_train 数据组成
best checkpoint epoch
```

对于伪标签阶段：

```text
pseudo_rows
pseudo_triplets
empty_pseudo_rows
exact_match_rows
pseudo precision
pseudo recall
pseudo F1
gold/pseudo sentiment distribution
```

对于增强阶段：

```text
requests
generated
after_consistency_filter
NLI kept
NLI contradiction removed
model_filter kept
augmentation_distribution
sentiment_distribution
```

## 13. 对新对话 AI 的接续说明

如果把本文档发给一个没有历史记忆的新 AI，请让它先做以下事情：

```text
1. 阅读 J:\nlp\CD-C3DA 当前代码。
2. 阅读 J:\nlp\CD-C3DA\T5_C3DA_CROSS_DOMAIN_ASTE_README_CN.md。
3. 阅读 J:\nlp\BGCA-master\code\eval_utils.py。
4. 阅读 J:\nlp\BGCA-master\code\run_utils.py 中 model_filter 和 data_gene 相关代码。
5. 不要重新推翻当前项目方向。
6. 不要把任务退回成给定 aspect 的情感分类。
7. 当前任务是完整 ASTE triplet extraction。
8. 当前最佳结果是 F1=31.80。
9. 下一步优先实现 edit-distance fixed evaluation 和 BGCA-style model_filter。
```

新 AI 需要理解：

```text
当前项目不是要照搬 BGCA。
当前项目目标是吸收 BGCA 的有效机制，同时保留 C3DA 双通道增强思想。
最终目标是 CD-C3DA 在跨域 ASTE 上超过 BGCA 1 到 2 个 F1 点。
```

## 14. 短期下一步建议

最推荐的下一步顺序：

```text
1. 实现 BGCA-style edit-distance fixed evaluation。
2. 用当前 final_weighted_ep5\best 直接评估 raw/fixed，不重训。
3. 实现 BGCA-style model_filter。
4. 重新 augment，重新 final weighted ep5 训练。
5. 跑通 channel ablation。
6. 再考虑 multi-checkpoint pseudo 和 consistency loss。
```

不要马上做：

```text
1. 大规模 GCN
2. 复杂领域图
3. 过早联合训练所有损失
4. 用 target train gold 参与训练
5. 用 target test 选模型
```

当前最有性价比的两个模块：

```text
1. edit-distance fixed evaluation
2. BGCA-style model_filter
```

这两个模块最贴近 BGCA，同时最可能解释为什么当前 F1 和 BGCA 差距很大。
