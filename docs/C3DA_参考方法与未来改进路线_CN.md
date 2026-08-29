# CD-C3DA 参考方法与未来改进路线

## 文档定位

本文档保存 CD-C3DA 项目的两类长期研究信息：

1. **方法参考与技术来源**：记录当前研究从哪些相关工作中获得了哪些具体启发，以及 CD-C3DA 与这些工作的区别。
2. **未来改进路线图**：记录当前主线之后已经讨论并确定优先级的候选研究方向。

重要原则：

> “参考某篇论文”不等于“当前方法直接复现该论文”。

当前方法是由文献启发、项目自身 failure diagnostics（失败诊断） 和逐阶段实验共同发展形成的。

未来路线中的候选模块也不等于已经批准执行或已经成为论文贡献。

最新研究状态仍必须优先读取：

```text
AGENTS.md
.ai/PROJECT_STATE.md
.ai/CURRENT_TASK.md
.ai/DECISION_LOG.md
实验记录与模型索引_CN.md
03_CD-C3DA下一阶段改进计划_CN.md
```

---

# 第一部分：相关论文与方法参考

## 1. BGCA（Bidirectional Generative Cross-Domain Adaptation，双向生成跨域适配）：跨域伪标签—数据增强基础范式

BGCA（Bidirectional Generative Cross-Domain Adaptation，双向生成跨域适配） 是当前项目最重要的基础比较工作之一，也提供了早期跨域生成式适配的整体范式。

其主要流程可以概括为：

```text
source labeled data
        ↓
text-to-label extractor
        ↓
target unlabeled text
        ↓
target pseudo labels
        ↓
label-to-text generation
        ↓
target-domain labeled samples
        ↓
source + target generated data
        ↓
final ASTE model
```

CD-C3DA 从 BGCA（Bidirectional Generative Cross-Domain Adaptation，双向生成跨域适配） 主要参考了：

* 利用源域监督训练抽取器；
* 在无标注目标域形成伪标签；
* 利用目标域知识构造训练数据；
* 将源域监督与目标域增强监督共同用于最终模型。

但是当前 CD-C3DA 已经不再是简单的 BGCA reproduction（BGCA 复现）。

后续项目增加或修改了：

* 更严格的伪标签/增强质量控制；
* NLI consistency（自然语言推理一致性）；
* extractor re-extraction（抽取器再抽取）；
* strict exact validation（严格精确验证）；
* conflict rejection（冲突拒绝）；
* target-anchored augmentation（目标锚定增强）；
* multi-triplet structure preservation（多三元组结构保持）；
* 当前正在研究的 syntactic RGAT pseudo formation。

因此 BGCA（Bidirectional Generative Cross-Domain Adaptation，双向生成跨域适配） 更准确的定位是：

> **CD-C3DA 的基础跨域数据增强范式与主要实验比较基线之一。**

不要在论文 Introduction 中把 BGCA（Bidirectional Generative Cross-Domain Adaptation，双向生成跨域适配） 写成唯一中心工作。

---

## 2. RSDA（Reliable Self-training Domain Adaptation，可靠自训练域适配）：伪标签误差传播、NLI 与增强可靠性

RSDA（Reliable Self-training Domain Adaptation，可靠自训练域适配） 对 CD-C3DA 的影响主要集中在：

```text
pseudo-label error propagation（伪标签错误传播）
+
quality filtering（质量筛选）
+
data diversity（数据多样性）
```

RSDA（Reliable Self-training Domain Adaptation，可靠自训练域适配） 强调：

> 如果目标域伪标签本身存在错误，这些错误可能继续进入后续生成数据，并被进一步传播。

这一问题与 CD-C3DA 后来观察到的核心 failure mode 高度相关：

```text
target element omission（目标元素缺失）
        ↓
incomplete pseudo labels（不完整伪标签）
        ↓
augmentation无法恢复缺失信息
```

### 主要参考点 1：NLI-based filtering（基于自然语言推理的筛选）

RSDA（Reliable Self-training Domain Adaptation，可靠自训练域适配） 使用自然语言推理判断增强文本和原始文本之间的一致性，用于过滤低质量增强数据。

CD-C3DA 后来保留了 NLI 思想，但构建了更严格的验证流程：

```text
semantic consistency
+
bidirectional NLI
+
extractor re-extraction（抽取器再抽取）
+
strict exact
+
conflict rejection（冲突拒绝）
+
untouched-triplet preservation
```

因此：

> NLI 质量控制受到 RSDA（Reliable Self-training Domain Adaptation，可靠自训练域适配） 启发，但当前 CD-C3DA 的结构验证范围更严格。

### 主要参考点 2：Composition / multi-information augmentation

RSDA（Reliable Self-training Domain Adaptation，可靠自训练域适配） 还使用标签组合等方式增加样本中的信息密度。

这一思路曾启发 CD-C3DA 研究：

```text
multi-triplet composition（多三元组组合）
complete triplet completion（完整三元组补全）
```

但是项目后续诊断发现：

> 主要瓶颈不是能否构造新的 triplet plan（三元组计划），而是 generator（生成器） 能否可靠将其 realization（文本实现） 成文本，同时不破坏已有 triplets。

因此 RSDA（Reliable Self-training Domain Adaptation，可靠自训练域适配）-style composition 属于：

> **重要历史启发，但不是当前正式增强方法。**

---

## 3. DAEGCN（Domain Adaptation Enhanced Graph Convolutional Network，域适配增强图卷积网络）：句法图辅助目标域伪标签形成

DAEGCN（Domain Adaptation Enhanced Graph Convolutional Network，域适配增强图卷积网络） 是当前 M1 句法图路线最重要的直接启发之一。

其核心研究思想包括：

```text
linguistic / syntactic graph（语言/句法图）
+
domain adaptation（域适配）
+
graph propagation（图传播）
+
target pseudo-label formation（目标伪标签形成）
```

它表明：

> 句法结构可以在跨域目标知识获取阶段被用于改善目标域伪标签形成。

这与当前 M1 的研究位置高度一致。

当前 CD-C3DA 的 M1 为：

```text
T5 Encoder
        ↓
word pooling（词级池化）
        ↓
external dependency / POS graph
        ↓
multi-head relational graph attention（图注意力）
        ↓
gated residual fusion（门控残差融合）
        ↓
T5 Decoder
        ↓
target pseudo labels
```

主要区别：

DAEGCN（Domain Adaptation Enhanced Graph Convolutional Network，域适配增强图卷积网络） 使用其自己的 GCN / graph architecture（图架构）；

CD-C3DA 当前采用：

```text
T5 contextual semantics
+
typed dependency/POS topology
+
multi-head relational graph attention（图注意力）
+
gated residual fusion（门控残差融合）
```

因此我们主要参考的是：

> **“利用句法图改善跨域 target pseudo formation”这一研究位置和机制思想。**

而不是直接复制其网络结构。

DAEGCN（Domain Adaptation Enhanced Graph Convolutional Network，域适配增强图卷积网络） 对使用 Stanza 等外部句法分析工具也提供了同类工作依据。

---

## 4. DASA（Dependency-Aware Sentiment Adaptation，依存关系感知情感适配）：Dependency/POS 图与 Graph Attention

DASA（Dependency-Aware Sentiment Adaptation，依存关系感知情感适配） 对当前图路线的另一个重要启发是：

```text
dependency information（依存关系信息）
+
POS information
+
graph attention（图注意力）
```

DASA（Dependency-Aware Sentiment Adaptation，依存关系感知情感适配） 一方面利用语言结构进行跨域数据增强，另一方面把 dependency / POS 图结构用于 ASTE 建模。

因此 DASA（Dependency-Aware Sentiment Adaptation，依存关系感知情感适配） 对 CD-C3DA 主要提供的是：

> **哪些语言结构值得建图，以及图注意力可以如何服务 aspect/opinion relation modeling。**

当前 M1 与 DASA（Dependency-Aware Sentiment Adaptation，依存关系感知情感适配） 的重要区别：

```text
DASA（Dependency-Aware Sentiment Adaptation，依存关系感知情感适配）:
graph进入最终ASTE extraction model

当前CD-C3DA M1:
graph只用于upstream pseudo extractor
```

当前正式 M1 不允许把图写成 final ASTE model 的组成部分。

未来是否将图再次用于 final ASTE，是后续独立研究变量。

---

## 5. C3DA（Cross-Channel Data Augmentation，跨通道数据增强）：多通道数据增强思想

原始 C3DA（Cross-Channel Data Augmentation，跨通道数据增强） 工作研究了 cross-channel data augmentation（跨通道数据增强），并设计多个增强 channel 处理 aspect / polarity 等信息。

它对当前项目的重要启发主要是：

> **复杂 ABSA 数据增强可以按不同情感元素拆分为不同 augmentation channel（增强通道）。**

但是当前 CD-C3DA 的实际增强方法与原始 C3DA（Cross-Channel Data Augmentation，跨通道数据增强） 存在显著区别。

当前正式增强是：

```text
real target-domain anchor sentence
        ├── aspect replacement channel（方面替换通道）
        ├── opinion replacement channel（观点替换通道）
        └── small neutral auxiliary branch
```

其中两个主要通道为：

### Aspect replacement

```text
(service, excellent, positive)
→
(food, excellent, positive)
```

替换 aspect，同时保持 opinion 和 sentiment polarity。

### Opinion replacement

```text
(service, slow, negative)
→
(service, terrible, negative)
```

替换 opinion，同时保持 aspect 和 sentiment polarity。

当前方法：

* 不是 polarity channel（极性通道）；
* 不是 dual generator（生成器）（双生成器）；
* 不是 cross-channel generation（跨通道生成）；
* 当前 target-anchored augmentation（目标锚定增强） 本身不调用 generator（生成器）。

所以原 C3DA（Cross-Channel Data Augmentation，跨通道数据增强） 更准确的作用是：

> **提供 channelized augmentation / multi-aspect augmentation 的思想启发。**

当前目标锚定的 aspect–opinion 双通道结构，则是在项目自身跨域 ASTE failure diagnostics（失败诊断） 基础上重新设计形成的。

---

## 6. AG-CDSA（Adversarial Graph-based Cross-Domain Sentiment Analysis，对抗图式跨域情感分析）：Pseudo Noise、Adversarial Adaptation 与增强动机

AG-CDSA（Adversarial Graph-based Cross-Domain Sentiment Analysis，对抗图式跨域情感分析） 对当前项目更多提供：

* pseudo-label noise（伪标签噪声） 的问题动机；
* adversarial adaptation（对抗适配） 与 target augmentation（目标增强） 可以组合的研究背景；
* 跨域 ABSA Introduction / Related Work 的参考；
* 目标伪标签错误可能影响后续增强质量的案例依据。

它不是：

* 当前 RGAT architecture 的直接来源；
* 当前 target-anchored dual-channel augmentation 的直接模板。

因此更适合在论文中作为：

> **跨域伪标签学习与数据增强相关工作。**

---

# 7. 当前 CD-C3DA 各模块与文献启发的对应关系

可以将当前技术来源概括为：

```text
BGCA（Bidirectional Generative Cross-Domain Adaptation，双向生成跨域适配）
↓
source → pseudo → target supervision → final model
基础跨域增强框架

RSDA（Reliable Self-training Domain Adaptation，可靠自训练域适配）
↓
pseudo error propagation
NLI filtering
multi-information augmentation
增强可靠性与历史multi增强启发

DAEGCN（Domain Adaptation Enhanced Graph Convolutional Network，域适配增强图卷积网络）
↓
graph + domain adaptation（域适配）
用于target pseudo formation
当前M1图模块研究位置的重要启发

DASA（Dependency-Aware Sentiment Adaptation，依存关系感知情感适配）
↓
dependency/POS
+
graph attention（图注意力）
当前句法图结构设计的重要启发

C3DA（Cross-Channel Data Augmentation，跨通道数据增强）
↓
channelized augmentation
multi-aspect augmentation
当前双通道设计的概念启发

AG-CDSA（Adversarial Graph-based Cross-Domain Sentiment Analysis，对抗图式跨域情感分析）
↓
pseudo noise
adversarial adaptation（对抗适配）
target augmentation（目标增强）
问题动机与相关工作参考
```

但 CD-C3DA 当前真正的方法形成还依赖项目自己的 diagnostic evidence（诊断证据）：

```text
target aspect/opinion coverage不足
+
multi-triplet element omission
+
generator（生成器） realization（文本实现） failure
+
training-end intervention repeatedly fails（训练末端干预反复失败）
```

因此当前方法不是简单拼接已有论文，而是：

> **文献机制启发 + 项目 failure diagnosis → 针对跨域 multi-triplet ASTE 重新形成的方法链。**

---

# 第二部分：未来改进路线图

## 1. 总体研究主线

项目当前已经停止继续优先探索：

```text
training-end weight tuning
loss modification
DANN coefficient search
selector微调
```

当前核心瓶颈已经转向：

```text
target-domain knowledge acquisition（目标域知识获取）
        ↓
target-domain knowledge utilization（目标域知识利用）
        ↓
final structured extraction（最终结构化抽取）
```

因此未来研究按照三个层次推进：

```text
Stage 1
Knowledge Acquisition
图增强目标域伪标签形成

Stage 2
Knowledge Utilization
改进target-anchored augmentation（目标锚定增强）

Stage 3
Knowledge Reuse / Structured Extraction
研究图结构是否需要进入最终ASTE
```

一次实验只能回答一个研究问题。

---

# 2. Stage 1 — Graph-Aware Target Pseudo Formation（图感知目标伪标签形成）

状态：

```text
CURRENT / APPROVED（当前/已批准）
```

当前任务：

```text
M1_SYNTACTIC_RGAT_PSEUDO_QUICK_ABLATION
```

核心结构：

```text
T5 Encoder
        ↓
word pooling（词级池化）
        ↓
dependency/POS typed graph
        ↓
multi-head relational graph attention（图注意力）
        ↓
gated residual fusion（门控残差融合）
        ↓
T5 Decoder
```

目的：

> 改善跨域条件下目标域 aspect、opinion 及其 relation 的发现，尤其针对 multi-triplet sentence 中的 element omission。

当前 graph 仅允许作用于：

```text
source extractor training
source-dev evaluation
target-unlabeled DANN
target pseudo inference
```

当前明确不进入：

```text
generator（生成器）
augmentation
NLI
candidate（候选） selection
final ASTE
final inference
```

当前第一研究问题：

> **Can syntactic structure improve target-domain pseudo-label supply and multi-triplet structural coverage?**

---

# 3. Stage 1 的两阶段实验逻辑

## Phase A

```text
graph extractor training
        ↓
source-dev
        ↓
target-unlabeled pseudo inference
        ↓
pseudo structural audit
```

先判断：

> 图是否真正改善 upstream supply。

如果 Phase A 不通过，则停止，不运行完整 downstream。

## Phase B

仅在 Phase A PASS 后：

```text
new pseudo
        ↓
pseudo selection
        ↓
existing augmentation
        ↓
NLI / exact / conflict validation
        ↓
candidate（候选） selection
        ↓
final training data
        ↓
non-graph final ASTE
        ↓
uptake audit
```

Phase B 回答：

> **Upstream pseudo improvement can actually be absorbed by the downstream ASTE system or not.**

注意：

```text
better pseudo supply
≠
better final uptake
```

必须分别验证。

---

# 4. Stage 2 — Graph-Constrained Dual-Channel Augmentation（图约束双通道增强）

状态：

```text
NEXT CANDIDATE（下一候选）
NOT YET APPROVED（尚未批准）
```

如果 Stage 1 证明 graph-aware pseudo formation 有价值，当前最优先的数据增强升级方向不是立即增加 triplet 数量，而是：

> **让现有 aspect/opinion 双通道增强真正利用目标域句法结构知识。**

当前增强：

```text
real target anchor
        ├── aspect replacement
        └── opinion replacement
```

候选升级：

```text
aspect replacement
+
opinion replacement
+
target-domain syntactic compatibility
+
relation-preservation validation
```

核心研究问题：

> **Can syntactic knowledge obtained during target pseudo formation be used to select structurally compatible augmentation candidate（候选）s?**

---

# 5. Stage 2A — Syntactic Role Compatibility

状态：

```text
NEXT CANDIDATE（下一候选）
```

当前替换主要判断：

```text
candidate（候选）是否是可靠target aspect/opinion
```

未来进一步加入：

```text
UPOS
dependency relation
head POS
local dependency neighborhood
aspect-opinion dependency path
```

形成 target-domain structural signature。

例如：

```text
waiter是target aspect
+
其真实target-domain syntactic role
与当前anchor aspect slot兼容
↓
才允许替换
```

Opinion channel 同理。

增强因此从：

```text
lexical replacement
```

升级为：

```text
structure-compatible lexical replacement
```

---

# 6. Stage 2B — Relation-Preserving Structural Validation

状态：

```text
NEXT CANDIDATE（下一候选）
```

当前结构验证已经包括：

```text
new element出现
old element消失
untouched triplets保留
new label完整
NLI consistency（自然语言推理一致性）
extractor re-extraction（抽取器再抽取）
strict exact
conflict rejection（冲突拒绝）
unplanned relation control
```

未来如果 graph route 成功，可以进一步验证：

```text
edited triplet
是否形成合理target-domain syntactic relation

+
untouched triplets
原有核心dependency structure是否保持
```

也就是将 structure preservation 从：

```text
label / element preservation
```

进一步提升到：

```text
syntactic relation preservation
```

这样形成完整链：

```text
graph discovers target structural knowledge
        ↓
augmentation uses target structural knowledge
        ↓
graph/structure-aware validation
        ↓
reliable target supervision
```

---

# 7. Stage 3 — Aspect–Opinion Pair Replacement（方面—观点对替换）

状态：

```text
SECONDARY CANDIDATE（次级候选）
NOT APPROVED（未批准）
```

当前 augmentation 一次只替换一个 element：

```text
(service, slow, negative)
→
(waiter, slow, negative)
```

未来可研究：

```text
(service, slow, negative)
→
(waiter, rude, negative)
```

即：

> 从高置信 target pseudo 中获得真实 target-domain aspect–opinion pair，并整体替换一个 triplet 内的 aspect/opinion。

潜在优势：

> 保留真实目标域 aspect–opinion collocation，而不是将新 element 与旧 element 人工组合。

但由于一次改变两个情感元素，其研究复杂度高于单元素 graph-constrained replacement。

因此优先级低于 Stage 2。

---

# 8. Stage 4 — Final ASTE RGAT（最终方面级情感三元组抽取关系图注意力网络）

状态：

```text
OPTIONAL CANDIDATE（可选候选）
NOT APPROVED（未批准）
```

如果 Stage 1 表明：

```text
pseudo supply improved
```

并且 downstream evidence 表明：

```text
最终模型仍存在明显structured extraction bottleneck（结构化抽取瓶颈）
```

则可以研究：

> 是否让同类 RGAT 也进入 final ASTE model。

正确实验必须保持训练数据完全相同：

```text
same pseudo
same augmentation
same final training data

CONTROL:
ordinary T5 final ASTE

TREATMENT:
T5 + RGAT final ASTE
```

研究问题：

> **Does syntactic graph modeling provide additional benefit during final structured extraction（最终结构化抽取）, beyond its upstream pseudo-label benefit?**

不能同时改变 pseudo 和 final architecture。

---

# 9. Stage 5 — Fresh Final RGAT vs Transferred RGAT（全新初始化与迁移复用）

状态：

```text
OPTIONAL LATER CANDIDATE（后续可选候选）
```

如果 Final RGAT 有效，还可以进一步区分：

## Fresh

```text
final T5 + RGAT
RGAT正常随机初始化
```

## Transfer / Reuse

```text
pseudo extractor训练好的RGAT
        ↓
transfer weights
        ↓
final ASTE继续训练
```

研究问题：

> **Is the learned upstream syntactic knowledge itself transferable to final ASTE training?**

只有比较：

```text
Fresh RGAT
vs
Transferred RGAT
```

才能严格支持：

> “reuse learned syntactic knowledge”

这一 claim（论文主张）。

否则只能证明：

> final graph architecture（图架构） 有效。

---

# 10. Final Graph 的代价

当前 upstream-only graph 有一个重要优势：

```text
graph only during target knowledge acquisition
        ↓
final inference remains ordinary T5
        ↓
no parser dependency（句法分析器依赖） at deployment
```

如果未来 graph 进入 final ASTE：

```text
test sentence
        ↓
Stanza
        ↓
dependency/POS
        ↓
RGAT
        ↓
T5
```

最终推理将依赖 parser。

因此未来是否采用 Final RGAT，需要比较：

> multi-triplet / final ASTE 收益是否值得增加 parser dependency（句法分析器依赖） 和 inference complexity（推理复杂度）。

---

# 11. Stage 6 — Complete Triplet Completion

状态：

```text
HIGH-RISK LONG-TERM CANDIDATE（高风险长期候选）
NOT CURRENT METHOD（非当前方法）
NOT APPROVED（未批准）
```

候选形式：

```text
{T1, T2}
→
{T1, T2, Tnew}
```

它是最直接解决：

```text
missing complete triplet
```

的方向。

但是此前实验已经证明真正瓶颈是：

```text
triplet plan（三元组计划）可以形成
        ↓
text realization（文本实现）困难
```

常见 failure：

```text
新triplet没有完整实现

或

实现新triplet但破坏旧triplet

或

产生unplanned relation
```

因此 complete triplet completion（完整三元组补全） 只有在出现：

> **真正不同于旧 composition / forced generation / local insertion 的 realization（文本实现） mechanism**

时才值得重新开启。

不能简单复活旧路线。

当前论文不能把 complete triplet completion（完整三元组补全） 写成当前方法或当前贡献。

---

# 12. 当前路线优先级总图

```text
CURRENT
Stage 1
Graph-Aware Pseudo Formation
        ↓
        ↓ if supported
        ↓
NEXT CANDIDATE（下一候选）
Stage 2
Graph-Constrained Dual-Channel Augmentation（图约束双通道增强）
        ↓
        ↓
SECONDARY CANDIDATE（次级候选）
Stage 3
Aspect–Opinion Pair Replacement（方面—观点对替换）
        ↓

OPTIONAL depending on evidence
Stage 4
Final ASTE RGAT（最终方面级情感三元组抽取关系图注意力网络）
        ↓
Stage 5
Fresh vs Transferred RGAT
        ↓

HIGH-RISK LONG TERM
Stage 6
Complete Triplet Completion
```

Stage 3 和 Stage 4 的实际先后顺序不预先锁死。

必须根据 Stage 1 / Stage 2 的结果决定。

例如：

如果发现：

```text
pseudo明显改善
但augmentation/final uptake不足
```

优先改：

```text
augmentation
```

如果发现：

```text
pseudo改善
augmentation也被吸收
但final model仍然multi recall不足
```

则 Final RGAT 更值得优先。

---

# 13. 三个核心研究问题

整个未来路线最终可以归纳为三个问题。

## Q1 — 能不能发现？

```text
target aspect / opinion / relation
是否能够被充分发现？
```

对应：

```text
Graph-Aware Pseudo Formation
```

## Q2 — 能不能可靠变成训练监督？

```text
被发现的target knowledge
能否在不破坏multi-triplet结构的情况下
转化为可靠augmentation？
```

对应：

```text
Graph-Constrained Target-Anchored Augmentation
```

## Q3 — 最终模型能不能充分学习复杂结构？

```text
有了更好的训练监督以后
final ASTE是否仍存在structured extraction bottleneck（结构化抽取瓶颈）？
```

对应：

```text
Final RGAT / graph reuse
```

因此完整因果链为：

```text
target-domain element discovery不足
        ↓
graph-aware pseudo formation
        ↓
better target knowledge
        ↓
target-anchored augmentation（目标锚定增强）
        ↓
structure-aware validation
        ↓
reliable target supervision
        ↓
final ASTE uptake
        ↓
如仍有structured extraction bottleneck（结构化抽取瓶颈）
        ↓
consider final RGAT
```

---

# 14. 研究纪律

未来路线必须遵守：

> **一次只改变一个研究变量。**

不能因为后面的候选已经讨论过，就一次同时实现：

```text
graph pseudo
+
graph-aware augmentation
+
pair replacement
+
final RGAT
+
triplet completion
```

任何新阶段都必须由前一阶段 evidence 决定是否开启。

---

# 15. 当前论文边界

当前可以作为主要候选贡献描述的是：

1. **Syntactic graph-aware target pseudo-label formation（目标伪标签形成）**
2. **Target-anchored aspect–opinion dual-channel augmentation with structure preservation**

以下当前不能作为已经完成贡献：

```text
graph-constrained augmentation
aspect-opinion pair replacement
final RGAT
RGAT weight reuse
complete triplet completion（完整三元组补全）
```

只有经过后续正式 Gate、真正进入最终方法以后，才能升级为论文 claim（论文主张）。

---

# 16. 一句话技术血缘与未来路线总结

当前 CD-C3DA 可以概括为：

```text
BGCA（Bidirectional Generative Cross-Domain Adaptation，双向生成跨域适配）
提供跨域 pseudo → target supervision 的基础范式

RSDA（Reliable Self-training Domain Adaptation，可靠自训练域适配）
启发 pseudo error control / NLI / multi-information augmentation

DAEGCN（Domain Adaptation Enhanced Graph Convolutional Network，域适配增强图卷积网络） + DASA（Dependency-Aware Sentiment Adaptation，依存关系感知情感适配）
启发 syntax graph 用于跨域结构知识建模

C3DA（Cross-Channel Data Augmentation，跨通道数据增强）
启发 channelized augmentation

+
CD-C3DA 自身 failure diagnostics（失败诊断）
发现：
target element absence
multi-triplet omission
generator（生成器） realization（文本实现） bottleneck

↓

当前：
Graph-Aware Pseudo Formation
+
Target-Anchored Aspect–Opinion Dual-Channel Augmentation

↓

未来：
Graph-Constrained Augmentation
→ Pair-Level Augmentation / Final RGAT
→ 如有新realization（文本实现）机制再考虑Complete Triplet Completion
```

---

## 文档维护规则

后续如果发生以下变化：

* 某个 candidate（候选） 被正式批准；
* 某个 future route（未来路线） 被实验关闭；
* Final RGAT 正式进入方法；
* augmentation 架构发生实质改变；
* complete triplet completion（完整三元组补全） 被重新批准；
* 新论文成为核心方法参考；

应同步更新本文档。

但：

> 当前正在执行什么实验、当前 Gate 状态、最新 commit（提交）、最新结果，不应主要维护在本文档中。

这些仍应写入：

```text
.ai/PROJECT_STATE.md
.ai/CURRENT_TASK.md
.ai/DECISION_LOG.md
实验记录与模型索引_CN.md
```

完成后请返回：

```text
FILE:
docs/C3DA（Cross-Channel Data Augmentation，跨通道数据增强）_参考方法与未来改进路线_CN.md

STATUS:
CREATED / UPDATED

COMMIT:
<commit（提交）>

WORKTREE:
CLEAN / DIRTY
```

不需要运行 GPU，不需要修改任何研究代码，不需要改变 CURRENT_TASK。
