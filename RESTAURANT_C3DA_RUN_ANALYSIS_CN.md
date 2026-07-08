# Restaurant 数据集 C3DA 单次复现实验不足分析报告

本文档基于本次已经完成的实验日志、最终分类报告、混淆矩阵以及 T5 生成的增强数据文件，对模型完成任务时的不足、主要损失来源和后续可改进方向进行分析。

本次实验配置如下：

```text
数据集：Restaurant
最终分类模型：RoBERTa
生成模型：T5 + LoRA
随机种子：1000
生成器微调轮数：100 epoch
分类器训练轮数：15 epoch
分类器 batch size：8
使用增强数据：是
使用对比学习：是
```

## 1. 最终实验结果

日志中记录的最佳结果为：

```text
max_test_acc_overall = 0.868632707774799
max_f1_overall       = 0.8193910518339607
```

也就是说，本次单 seed 实验结果为：

```text
Accuracy = 86.86%
Macro-F1 = 81.94%
```

最终分类报告中三类表现如下：

```text
positive F1 = 0.9252
negative F1 = 0.8039
neutral  F1 = 0.7185
```

可以看出，模型对 positive 类判断最好，对 neutral 类判断最弱。

## 2. 混淆矩阵分析

最终混淆矩阵为：

```text
[[674  29  24]
 [ 13 164  19]
 [ 43  19 134]]
```

标签对应关系：

```text
0 = positive
1 = negative
2 = neutral
```

按行看真实标签，按列看预测标签：

```text
真实 positive：727 个，其中 674 个预测正确，29 个错判为 negative，24 个错判为 neutral
真实 negative：196 个，其中 164 个预测正确，13 个错判为 positive，19 个错判为 neutral
真实 neutral： 196 个，其中 134 个预测正确，43 个错判为 positive，19 个错判为 negative
```

最大错误来源是：

```text
neutral -> positive
```

也就是 neutral 样本经常被模型判断成 positive。

这说明模型存在明显的 positive 偏置，尤其是在情感表达较弱、语义模糊或没有明显正负倾向时，模型更倾向于给出 positive。

## 3. 类别不平衡问题

测试集类别分布为：

```text
positive: 727
negative: 196
neutral:  196
```

positive 类占测试集的大多数，约为 65%。

这会带来两个影响：

```text
1. Accuracy 会比较高，因为多数类 positive 容易撑起整体准确率。
2. Macro-F1 会更真实地暴露 minority class 的问题，尤其是 neutral 类。
```

因此，本次实验中：

```text
Accuracy = 86.86%
Macro-F1 = 81.94%
```

Accuracy 明显高于 neutral 类 F1，这说明模型的整体正确率不错，但类别均衡能力仍有不足。

## 4. 生成数据质量分析

生成增强数据文件为：

```text
dataset/Restaurants_corenlp/generate-t5-lora-100.json
```

统计结果如下：

```text
原始训练样本组数：3608
总生成句子数：14432
空生成句子数：0
唯一生成句子数：1914
重复生成句子额外数量：12518
```

这是本次实验暴露出的最重要问题之一。

虽然模型生成了 14432 条增强句子，但唯一句子只有 1914 条，说明生成数据存在严重重复。

重复最多的句子包括：

```text
出现 1668 次：The food is delicious and I highly recommend it.
出现 1577 次：The food is delicious and the service is prompt and professional.
出现 903 次： The food is very good too..
出现 456 次： The food is delicious and the prices are reasonable.
出现 277 次： The food is delicious and the staff is very attentive.
```

这说明 T5 生成器发生了较明显的模板化生成或模式坍缩，倾向于生成高频、泛化、偏 positive 的 restaurant 句子。

## 5. 主要损失来源

从训练日志看，训练后期模型在训练集上的 acc 已经接近：

```text
0.99 - 1.00
```

这说明模型不是“训练没学会”，而是已经较充分地拟合了训练数据。

但是测试集 F1 在后期仍然波动，并没有随着训练集 acc 接近 1 而持续提高。

因此，本次实验的主要损失来源是：

```text
泛化损失
```

而不是：

```text
优化失败
```

换句话说，模型能很好地记住训练数据和增强数据，但这些模式并不能完全稳定地迁移到测试集。

## 6. 增强数据带来的问题

训练日志中 `VanillaLoss` 和 `AugLoss` 在早期经常比较接近，说明增强数据确实参与了训练。

但是，由于增强数据中大量句子重复且偏向 positive 模板，增强数据可能强化了浅层模式，例如：

```text
food + delicious -> positive
service + professional -> positive
staff + attentive -> positive
price + reasonable -> positive
```

这些模式对 positive 类有帮助，但对 neutral 或复杂情感样本帮助有限。

甚至在某些情况下，过多正向模板会让模型在遇到弱情感或中性表达时也倾向于预测 positive。

## 7. 对比学习损失问题

虽然训练命令中启用了：

```text
--withCL
```

但是日志中 `CLLoss` 几乎一直为：

```text
0.0000
```

这说明本次实验中，对比学习项的实际贡献很弱。

可能原因包括：

```text
1. k = 1，每个样本只选 1 个增强样本，对比结构不够丰富。
2. 生成样本重复严重，导致正负样本区分度不足。
3. 当前 margin 下，正负样本表示已经满足约束，loss 很快变为 0。
4. 缺少 hard negative，负样本不够难。
```

这点很重要，因为 C3DA 理论上的优势之一就是通过对比学习提高多 aspect、多 polarity 场景下的鲁棒性。

但在这次实验中，对比学习信号几乎没有发挥出应有作用。

## 8. 当前模型的主要不足

### 8.1 neutral 类识别能力不足

neutral 类 F1 为：

```text
0.7185
```

neutral 类 recall 为：

```text
0.6837
```

说明真实 neutral 样本中，有相当一部分没有被识别出来。

其中最明显的问题是：

```text
neutral 被错判为 positive
```

这可能与类别不平衡、生成数据偏 positive、neutral 边界模糊有关。

### 8.2 生成数据重复严重

总生成句子数是：

```text
14432
```

但唯一句子只有：

```text
1914
```

实际有效增强规模远小于表面规模。

这会导致模型反复学习类似模板，而不是学习更多样化的 aspect-polarity 表达。

### 8.3 生成样本偏向 positive

高频生成句大多是 positive，例如：

```text
food is delicious
service is professional
staff is attentive
prices are reasonable
```

这会进一步放大 positive 类优势，削弱模型对 neutral 和 negative 的敏感性。

### 8.4 aspect 覆盖不均衡

生成句子大量集中在：

```text
food
service
staff
price
atmosphere
```

对长尾 aspect 的覆盖不足，例如：

```text
wine
seats
menu
quantity
design
specific dishes
```

这会限制模型在复杂多 aspect 场景下的鲁棒性。

### 8.5 情感保持不稳定

部分生成样本虽然语法通顺，但可能发生 aspect 或 polarity 漂移。

例如，源句可能是负向或中性表达，但生成句会变成泛化的 positive food/service 模板。

这类样本会给增强训练带来噪声。

### 8.6 后期存在过拟合迹象

训练后期训练集 acc 接近 1，但测试集 F1 波动明显。

说明模型后期已经在强拟合训练数据和增强数据，泛化收益有限。

## 9. 可改进方向与具体方法

### 9.1 对生成数据做精确去重

当前最直接的改进是对生成文件进行去重。

方法：

```text
对每组生成样本去重
对全局生成样本去重
删除完全相同的句子
```

预期作用：

```text
减少模板句反复参与训练
降低 positive 模板对模型的过度影响
提高增强数据有效多样性
```

### 9.2 对生成数据做语义去重

精确去重只能去掉完全一样的句子，但无法处理近似重复。

例如：

```text
The food is delicious and I highly recommend it.
The food is very delicious and I highly recommend this place.
```

可以使用句向量模型计算相似度：

```text
cosine similarity > 0.95 则视为近似重复
```

方法：

```text
使用 sentence-transformers/all-MiniLM-L6-v2 编码生成句
聚类或相似度过滤
每个簇只保留一个高质量样本
```

### 9.3 引入 NLI 过滤低质量样本

参考 RSDA 的方法，引入 NLI 过滤器。

设置：

```text
premise = 原始句子
hypothesis = 生成句子
```

如果 NLI 模型判断 contradiction 概率较高，则删除该生成样本。

推荐第一版规则：

```text
P(contradiction) >= 0.5 的样本删除
```

更严格规则：

```text
P(entailment) > 0.5 且 P(contradiction) < 0.2 才保留
```

但对于数据增强来说，生成句不一定必须严格蕴含原句，所以第一版建议只过滤高 contradiction 样本。

预期作用：

```text
减少 aspect 漂移
减少 polarity 漂移
过滤语义矛盾或明显不一致的生成句
```

### 9.4 平衡生成样本的情感极性

当前生成数据明显偏 positive。

可以控制每类生成样本数量：

```text
positive: negative: neutral 尽量接近
```

或者对 minority 类进行更强增强：

```text
negative 和 neutral 生成更多样本
positive 降采样
```

预期作用：

```text
提高 negative 和 neutral 的 F1
降低模型对 positive 的偏置
```

### 9.5 用领域特定片段引导生成

参考 DAEGCN 的领域特定片段感知方法，不再让 T5 随机选择 aspect，而是用目标域或当前域的高质量片段引导生成。

方法：

```text
提取 n-gram 片段
计算领域频率比
保留名词短语和形容词短语
用这些片段作为生成条件
```

示例：

```text
aspect: wine sentiment: negative
aspect: seats sentiment: neutral
aspect: menu sentiment: positive
```

预期作用：

```text
提高长尾 aspect 覆盖
减少 food/service 模板坍缩
增强多 aspect 表达能力
```

### 9.6 增强 neutral 样本生成质量

neutral 类是当前短板，所以需要专门优化。

方法：

```text
为 neutral 设计更明确的生成模板
对 neutral 生成结果做分类器置信度过滤
删除带明显 positive/negative 情感词的 neutral 生成句
```

例如 neutral 句应更像：

```text
The menu includes several seafood options.
The restaurant is located near the station.
The seats are arranged close to the window.
```

而不是：

```text
The food is delicious.
The service is terrible.
```

### 9.7 加强对比学习信号

当前 `CLLoss` 基本为 0，说明对比学习没有充分发挥作用。

可以尝试：

```text
1. 增大 k，例如 k=2 或 k=3
2. 调整 margin，例如 0.3、0.5、0.7
3. 引入 hard negative
4. 对不同 polarity 的同 aspect 样本构造负样本
5. 对同一句中的不同 aspect 构造更难的对比对
```

注意：

```text
k 增大会增加显存占用
```

3070 8GB 下建议先尝试：

```text
k=2
batch_size=4
```

### 9.8 加入早停机制

现在固定训练 15 epoch，但后期已经出现过拟合迹象。

可以加入 early stopping：

```text
如果连续 N 次评估 F1 没有提升，则停止训练
```

推荐：

```text
patience = 5 或 10
```

预期作用：

```text
减少过拟合
节省训练时间
避免后期模型退化
```

### 9.9 调整 log_step

当前：

```text
log_step = 5
```

每 5 个 step 就评估一次测试集，会明显拖慢训练。

如果后续实验较多，可以改为：

```text
log_step = 50
```

或：

```text
log_step = 100
```

这不会改变训练本身，只会减少评估频率。

### 9.10 建立生成样本质量评分机制

可以给每条生成样本打综合质量分：

```text
quality_score = 分类器置信度 + NLI一致性 + 去重惩罚 + aspect覆盖权重
```

然后只保留高质量样本。

这样比简单“全部使用生成样本”更稳。

## 10. 后续最推荐的改进优先级

如果按收益和实现难度排序，建议按以下顺序改：

### 第一优先级：生成数据去重

原因：

```text
当前重复问题非常严重，且实现简单。
```

### 第二优先级：NLI 过滤

原因：

```text
可以过滤语义矛盾、情感漂移和生成跑偏样本。
```

### 第三优先级：平衡 polarity

原因：

```text
当前模型明显偏 positive，neutral 表现较弱。
```

### 第四优先级：领域特定片段引导生成

原因：

```text
可以减少 food/service 模板坍缩，提高长尾 aspect 覆盖。
```

### 第五优先级：加强对比学习

原因：

```text
当前 CLLoss 接近 0，C3DA 的核心优势没有完全发挥。
```

## 11. 总结

本次实验已经成功跑通 Restaurant 单 seed 复现，结果为：

```text
Accuracy = 86.86%
Macro-F1 = 81.94%
```

但模型主要不足在于：

```text
1. neutral 类识别较弱。
2. 生成增强数据重复严重。
3. 生成样本偏向 positive 模板。
4. 长尾 aspect 覆盖不足。
5. 对比学习损失几乎为 0，作用较弱。
6. 后期训练存在过拟合迹象。
```

主要损失来源可以概括为：

```text
由类别不平衡、低多样性增强数据、生成噪声和对比信号不足共同导致的泛化损失。
```

最值得优先尝试的改进路线是：

```text
生成样本去重
+ NLI 过滤
+ polarity 平衡
+ 领域特定片段引导生成
+ 更有效的 hard negative 对比学习
```

