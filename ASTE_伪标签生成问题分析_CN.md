# ASTE 伪标签生成问题分析

## 1. 当前伪标签生成流程

当前项目中的伪标签生成只针对目标域无标签训练集进行。以本次实验为例：

```text
source domain = rest16
target domain = laptop14
```

训练阶段中，`laptop14 train` 不使用人工标签，只保留句子文本：

```text
target_unlabeled.jsonl
```

伪标签生成流程如下：

```text
rest16 train gold
        |
        v
训练 T5 ASTE 抽取器 extractor
        |
        v
laptop14 train text only
        |
        v
extractor 生成 ASTE 三元组
        |
        v
target_pseudo.jsonl
```

其中 ASTE 是：

```text
Aspect Sentiment Triplet Extraction
方面-观点-情感三元组抽取
```

每条伪标签格式为：

```text
<pos> battery life <opinion> good
```

对应三元组：

```text
(battery life, good, positive)
```

当前伪标签生成阶段没有使用目标域人工标签，也没有使用 `target_test`。这一点符合跨域设置。

## 2. 本次实验中的伪标签结果

本次实验目录：

```text
runs\t5_c3da_aste_rest16_to_laptop14_v2
```

目标域无标签数据规模：

```text
target_unlabeled = 906
```

成功生成伪标签的样本数：

```text
target_pseudo = 901
```

说明抽取器基本能为大多数目标域句子生成可解析三元组。

但是三元组情感极性分布严重异常：

```text
target_pseudo triplets = 1168
positive = 1168
negative = 0
neutral = 0
```

也就是说，目标域伪标签全部坍塌成了 positive。

## 3. 典型伪标签样例

一些正向样本是合理的：

```json
{
  "text": "I charge it at night and skip taking the cord with me because of the good battery life .",
  "label": "<pos> battery life <opinion> good"
}
```

但大量负向表达也被预测成 positive：

```json
{
  "text": "When I finally had everything running with all my software installed I plugged in my droid to recharge and the system crashed .",
  "label": "<pos> system <opinion> crashed"
}
```

这里 `crashed` 明显是负向观点词，但模型预测为：

```text
<pos>
```

另一个例子：

```json
{
  "text": "I can barely use any usb devices because they will not stay connected properly .",
  "label": "<pos> usb devices <opinion> barely use"
}
```

`barely use` 在语义上也偏负向，但仍然被预测为 positive。

## 4. 问题一：情感极性坍塌

当前伪标签生成最大问题是：

```text
模型几乎只生成 <pos>
```

这会导致后续最终训练集中的目标域伪标签严重偏向 positive。

源域训练集本身的情感分布为：

```text
source_train triplets = 1393
positive = 1015
negative = 328
neutral = 50
```

目标域测试集真实分布为：

```text
target_test gold triplets = 541
positive = 364
negative = 114
neutral = 63
```

目标域真实测试集虽然 positive 较多，但仍然存在大量 negative 和 neutral。然而伪标签中：

```text
negative = 0
neutral = 0
```

这说明模型并不是单纯反映目标域分布，而是发生了生成偏置。

## 5. 问题二：目标域负向表达识别差

从预测结果看，模型对 laptop 领域的一些负向表达不敏感。例如：

```text
crashed
barely use
slow
not fix
not yet discovered
difficult
defective
```

这些词或短语在 laptop 领域中经常表示负面评价，但模型容易预测成：

```text
<pos>
```

这说明从 restaurant 到 laptop 的迁移中，模型虽然能抽取部分 aspect 和 opinion，但没有学好目标域中的情感极性映射。

## 6. 问题三：伪标签缺少质量控制

当前伪标签生成策略是：

```text
只要 extractor 生成了可解析三元组，就保存
```

没有进一步判断：

```text
1. 这个三元组的情感极性是否合理
2. opinion 是否和 sentiment 一致
3. aspect/opinion 是否真的来自原句
4. 目标域伪标签类别分布是否异常
5. 模型对该预测是否有足够置信度
```

因此，错误伪标签会直接进入后续训练。

例如：

```text
<pos> system <opinion> crashed
```

这种明显错误的伪标签如果进入最终训练，会强化模型的错误判断，让模型更倾向于把负向 opinion 也预测为 positive。

## 7. 对最终模型的影响

最终模型在 `laptop14 test` 上的预测也出现了同样问题：

```text
pred triplets = 429
positive = 429
negative = 0
neutral = 0
```

最终测试结果：

```text
precision = 0.3357
recall = 0.2662
micro-F1 = 0.2969
```

其中一个关键错误类型是情感极性错误。

例如：

```text
gold: <neg> delete key <opinion> not yet discovered
pred: <pos> delete key <opinion> not yet discovered
```

```text
gold: <neg> mountain lion <opinion> slow
pred: <pos> mountain lion <opinion> slow
```

这说明伪标签阶段的 positive 坍塌会继续影响最终模型，使最终模型也倾向于只预测 positive。

## 8. 和 BGCA 的区别

BGCA 中也会训练 text-to-label 抽取模型：

```text
target sentence -> pseudo triplets
```

这一点和当前项目类似。

但 BGCA 不会简单地把所有伪标签直接用于最终训练。它还会进行双向生成和模型过滤：

```text
target sentence -> pseudo label
pseudo label -> generated sentence
generated sentence -> extracted label
extracted label == pseudo label 才保留
```

也就是说，BGCA 有一致性过滤：

```text
伪标签需要经过生成-再抽取验证
```

当前项目的伪标签阶段没有这种过滤，因此更容易把错误伪标签全部保留下来。

## 9. 当前伪标签生成的结论

本次实验说明：

```text
当前 T5 extractor 可以生成目标域伪三元组，但情感极性严重坍塌。
```

它的主要问题不是完全抽不出 aspect/opinion，而是：

```text
1. 目标域伪标签全部变成 positive
2. negative 和 neutral 完全缺失
3. 明显负向 opinion 被标成 positive
4. 伪标签缺少质量过滤
5. 错误伪标签进一步影响最终模型
```

因此，当前伪标签可以作为一个初始 baseline 的组成部分，但不能认为质量可靠。

## 10. 后续改进方向

### 10.1 加入情感极性约束

在生成伪标签时加入更强的 sentiment control：

```text
<pos>
<neg>
<neu>
```

可以尝试：

```text
1. constrained decoding
2. sentiment prompt
3. 类别均衡采样
4. 对 negative/neutral 加权训练
```

目标是避免模型只生成 `<pos>`。

### 10.2 加入伪标签过滤

可以先加入规则过滤：

```text
1. opinion 是明显负向词时，不接受 <pos>
2. opinion 是明显正向词时，不接受 <neg>
3. aspect 和 opinion 必须出现在原句中
```

后续可以加入模型过滤：

```text
生成句子 -> 再抽取三元组 -> 一致才保留
```

也可以加入 NLI 过滤：

```text
句子是否蕴含该三元组表达的情感含义
```

### 10.3 控制伪标签类别分布

当前伪标签分布为：

```text
positive 100%
negative 0%
neutral 0%
```

可以设置基本分布检查：

```text
如果某类为 0，说明伪标签质量异常，不应直接进入最终训练。
```

也可以限制伪标签采样：

```text
positive 伪标签只取一部分
negative/neutral 通过更强策略补足
```

### 10.4 加入目标域情感词库

针对 laptop 领域，应该构建目标域 opinion bank：

```text
positive: fast, excellent, easy, stable
negative: slow, crashed, defective, difficult, not fix
neutral: average, ok, normal
```

伪标签生成后，用该词库进行校验或重标注辅助。

## 11. 建议记录方式

当前实验结果建议记录为：

```text
T5-C3DA-ASTE pseudo-label baseline
source = rest16
target = laptop14
seed = 1000
target pseudo samples = 901
target pseudo triplets = 1168
pseudo sentiment distribution = 100% positive
final micro-F1 = 29.69
```

同时必须注明：

```text
伪标签阶段发生 positive collapse，negative/neutral 缺失。
```

这个现象本身可以作为后续改进方法的动机。
