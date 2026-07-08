# C3DA 实验复现结果与后续改进总结

本文档用于保存当前 C3DA 复现实验的结果、暴露出的不足，以及后续可以围绕“跨域、多方面情感分析、数据增强质量控制”继续改进的思路。后面如果长对话导致上下文丢失，把这份文档重新发给我，我就可以快速恢复我们之前讨论过的实验状态和改进方向。

## 1. 当前项目复现状态

当前项目目录：

```text
J:\nlp\C3DA-main
```

模型统一下载目录：

```text
J:\nlp\models
```

当前已经准备好的本地模型：

```text
J:\nlp\models\mrm8488-t5-base-finetuned-common_gen
J:\nlp\models\roberta-base
J:\nlp\models\bert-base-uncased
```

当前 conda 环境检查结果：

```text
torch: 2.2.2+cu121
transformers: 4.39.3
cuda available: True
gpu: NVIDIA GeForce RTX 3070
```

这说明环境已经可以使用 GPU 跑实验，当前显卡为 RTX 3070，8GB 显存。后续实验参数需要优先考虑显存限制，不能直接把 batch size、生成数量或模型规模开得太大。

## 2. 当前已完成的实验

已经完成 Restaurant 数据集上的 C3DA + RoBERTa 单 seed 复现实验。

生成阶段命令：

```cmd
cd /d J:\nlp\C3DA-main && conda activate c3da && python generate.py --dataset restaurant --prompt_name lora --num_epoch 100 --batch_size 4 --num_workers 0 --cuda 0 --model_root J:\nlp\models
```

该阶段使用 T5 + LoRA 生成增强样本，生成结果文件为：

```text
J:\nlp\C3DA-main\dataset\Restaurants_corenlp\generate-t5-lora-100.json
```

分类训练阶段命令：

```cmd
cd /d J:\nlp\C3DA-main && conda activate c3da && python train.py --dataset restaurant --model_name roberta --generate_model t5 --prompt_name lora --ft_epoch 100 --withAugment --withCL --aug_loss_fac 1.0 --margin 0.5 --cl_loss_fac 2.0 --k 1 --num_epoch 15 --batch_size 8 --cuda 0 --model_root J:\nlp\models --seed 1000
```

该阶段使用 RoBERTa 作为最终情感分类器，并启用增强数据和对比学习。

保存的最好模型：

```text
J:\nlp\C3DA-main\C3DA\state_dict\roberta_restaurant_acc_0.8686_f1_0.8159
```

训练日志：

```text
J:\nlp\C3DA-main\C3DA\log\roberta-restaurant-s1000-2026-06-02_11-20-34.log
```

## 3. 当前复现实验结果

本次 Restaurant 单 seed 实验的最好结果为：

```text
max_test_acc_overall = 0.868632707774799
max_f1_overall       = 0.8193910518339607
```

可以理解为：

```text
Accuracy = 86.86%
Macro-F1 = 81.94%
```

三类情感的具体表现：

```text
positive F1 = 0.9252
negative F1 = 0.8039
neutral  F1 = 0.7185
```

混淆矩阵：

```text
[[674  29  24]
 [ 13 164  19]
 [ 43  19 134]]
```

标签含义：

```text
0 = positive
1 = negative
2 = neutral
```

从结果看，模型对 positive 类识别最好，对 neutral 类识别最弱。最大的错误来源是：

```text
真实 neutral -> 预测 positive
```

也就是说，模型在遇到中性、弱情感、表达模糊的样本时，容易偏向预测为 positive。

## 4. 当前模型主要不足

### 4.1 neutral 类识别能力偏弱

neutral 类的 F1 只有 0.7185，recall 为 0.6837。这说明真实 neutral 样本中，有相当一部分没有被识别出来。

从混淆矩阵看，196 个真实 neutral 样本中：

```text
134 个预测正确
43 个被错判为 positive
19 个被错判为 negative
```

其中 neutral -> positive 是最明显的问题。这可能来自三方面：

```text
1. Restaurant 数据集中 positive 样本数量最多，类别天然不平衡。
2. T5 生成的增强样本偏向正向模板。
3. neutral 本身边界模糊，很多句子容易被浅层词汇误导。
```

### 4.2 生成增强数据重复严重

当前生成文件统计显示：

```text
原始训练样本组数：3608
总生成句子数：14432
唯一生成句子数：1914
重复生成句子额外数量：12518
```

高频重复句子包括：

```text
1668 次：The food is delicious and I highly recommend it.
1577 次：The food is delicious and the service is prompt and professional.
903 次：The food is very good too..
```

这说明 T5 + LoRA 生成器出现了明显的模板化生成。表面上生成了 14432 条增强句子，但真正有效的多样性远远不够。

### 4.3 增强数据有 positive 偏置

高频生成句大多是 positive 风格，例如：

```text
food is delicious
service is professional
staff is attentive
prices are reasonable
```

这些句子对提升 positive 类可能有帮助，但会进一步强化模型的 positive 偏置，使模型在 neutral 或弱情感样本上更容易误判为 positive。

### 4.4 对比学习损失没有充分发挥作用

训练日志中 `CLLoss` 几乎一直是：

```text
0.0000
```

这说明虽然命令里启用了 `--withCL`，但对比学习项对训练贡献很弱。

可能原因：

```text
1. 当前 k=1，每个样本只取 1 个增强样本，对比结构不够丰富。
2. 增强样本重复严重，正负样本区分度不足。
3. margin=0.5 可能太容易满足，导致 CLLoss 很快变成 0。
4. 缺少 hard negative，负样本不够难。
```

### 4.5 后期存在过拟合和泛化不稳定

日志显示训练后期训练集准确率可以接近：

```text
0.99 - 1.00
```

但是测试集 F1 会明显波动，甚至在后面下降。这说明主要问题不是“模型没学会”，而是“学到的东西没有稳定泛化到测试集”。

因此，当前主要损失来源可以概括为：

```text
泛化损失 > 优化失败
```

## 5. 立刻可以做的工程改进

### 5.1 生成数据去重

第一优先级应该是对 `generate-t5-lora-100.json` 做去重。

建议分两层：

```text
1. 精确去重：完全相同的生成句只保留一次。
2. 语义去重：句子不同但语义高度相似的样本，只保留更高质量的一条。
```

在 RTX 3070 8GB 显存上，第一版建议先做精确去重，成本最低，也最容易验证效果。

预期作用：

```text
减少模板化增强样本反复污染训练。
降低 positive 生成模板带来的类别偏置。
让 AugLoss 关注更多不同表达。
```

### 5.2 对生成样本做类别平衡

当前生成句明显偏 positive，因此需要控制每个类别参与训练的增强样本数量。

可行方法：

```text
1. 每个 label 最多保留固定数量增强样本。
2. 对 negative 和 neutral 生成样本适当提高采样权重。
3. 对 positive 高频模板进行下采样。
```

建议先做简单版本：

```text
每个原始样本最多保留 k 条不重复增强句；
每个类别的增强样本总数设置上限。
```

### 5.3 增加生成质量过滤

生成样本不应该直接全部进入训练。建议增加过滤脚本，例如：

```text
filter_generated.py
```

第一版可以过滤：

```text
1. 空句子。
2. 和原句完全无关的句子。
3. 长度过短或过长的句子。
4. 重复度过高的模板句。
5. 明显不包含原 aspect 或领域相关信息的句子。
```

后面可以接入 NLI 模型做语义矛盾过滤。

### 5.4 改进对比学习

当前 `CLLoss=0` 说明对比学习没有真正提供有效训练信号。

可以尝试：

```text
1. 将 k 从 1 提高到 2 或 3。
2. 调整 margin，例如测试 0.2、0.5、0.8。
3. 构造 hard negative：aspect 相近但 polarity 不同的样本。
4. 使用去重后的增强样本再做对比学习。
```

考虑 RTX 3070 8GB 显存，建议先尝试：

```cmd
cd /d J:\nlp\C3DA-main && conda activate c3da && python train.py --dataset restaurant --model_name roberta --generate_model t5 --prompt_name lora --ft_epoch 100 --withAugment --withCL --aug_loss_fac 1.0 --margin 0.2 --cl_loss_fac 2.0 --k 2 --num_epoch 15 --batch_size 6 --cuda 0 --model_root J:\nlp\models --seed 1000
```

如果显存不够，把 `--batch_size 6` 改成 `--batch_size 4`。

### 5.5 加入早停和最佳模型保存说明

当前训练不是最后一轮最好，而是在训练过程中某个 step 最好。因此后续应该明确：

```text
最终汇报使用 max_f1_overall 对应的模型；
不是直接使用最后一轮模型。
```

如果继续改代码，可以加入：

```text
early stopping
patience
best epoch / best step 记录
```

这样能减少后期过拟合，也方便写论文实验部分。

## 6. 从 RSDA 借鉴的改进方法

你提到的“数据增强框架”可以作为 C3DA 跨域改进的重要参考，尤其是伪标签和 NLI 过滤。

### 6.1 用目标域无标签数据生成伪标签

跨域场景可以这样做：

```text
源域有标签数据 Ds
目标域无标签数据 Ut
先用源域训练一个初始 ABSA 模型
用该模型给目标域 Ut 打伪标签
得到目标域伪标注数据 Dt'
```

对于方面级情感分类，伪标签形式可以是：

```text
(sentence, aspect, polarity)
```

例如：

```text
The screen is bright but the battery dies quickly.
screen -> positive
battery -> negative
```

这样就可以让 T5 生成更贴近目标域的增强样本，而不是只在源域内部增强。

### 6.2 用 NLI 过滤低质量生成样本

NLI 过滤可以判断原句和生成句是否语义矛盾。

推荐第一版规则：

```text
premise = 原始目标域句子
hypothesis = 生成句子
如果 P(contradiction) 太高，则丢弃该生成样本
```

第一版阈值可以设置为：

```text
P(contradiction) < 0.5
```

更严格版本：

```text
P(entailment) > 0.5 且 P(contradiction) < 0.2
```

但是对数据增强来说，生成句不一定必须严格蕴含原句，所以更推荐先只过滤高 contradiction 样本。

### 6.3 NLI 过滤和 C3DA 置信度过滤结合

可以形成两级过滤：

```text
生成样本 -> NLI 过滤 -> 分类器置信度/熵过滤 -> 进入训练
```

NLI 负责过滤语义矛盾，熵或置信度负责过滤分类器不确定样本。

这样比单纯依赖 T5 生成更稳。

## 7. 从 DAEGCN 借鉴的改进方法

你提到的另一篇论文中有“领域特定片段感知”和 GCN，这两部分都可以参考，但建议分阶段使用。

### 7.1 用领域特定片段替代简单高频词

原始 C3DA 或类似方法中，如果只是抽高频词，容易抽到无意义词，例如：

```text
the
is
good
food
```

这些词不一定真正代表目标领域差异。

更好的做法是提取领域特定片段：

```text
screen resolution
battery life
customer service
delivery speed
food quality
price range
```

可以用类似下面的评分：

```text
score(fragment) = freq_target(fragment) / (freq_source(fragment) + epsilon)
```

含义是：在目标域高频、但在源域不那么高频的片段，更可能是目标域特定片段。

第一版实现建议：

```text
1. 从源域和目标域中抽取 n-gram，n=1,2,3。
2. 过滤停用词和纯功能词。
3. 优先保留名词、名词短语、形容词短语。
4. 计算 target/source 频率比。
5. 选 Top-K 作为目标域片段。
6. 用这些片段指导 T5 生成目标域增强样本。
```

这样可以让生成数据更贴近目标域，而不是生成泛化模板。

### 7.2 GCN 可以参考，但不建议第一阶段就上

GCN 的优势是可以建模词、方面词、句法依存、领域片段之间的结构关系。

它适合用来增强：

```text
1. 伪标签模型。
2. 方面词和情感词之间的结构建模。
3. 跨域片段传播。
```

但它也会带来额外复杂度：

```text
1. 需要依存句法解析。
2. 需要构图。
3. 需要改分类器结构。
4. 训练成本更高。
5. 对 RTX 3070 8GB 显存更不友好。
```

因此建议顺序是：

```text
第一阶段：去重 + 类别平衡 + NLI 过滤 + 领域片段抽取
第二阶段：加入更强的伪标签和 hard negative
第三阶段：再考虑 GCN 或图结构模块
```

## 8. 推荐的后续实验路线

### 8.1 路线 A：先把当前 Restaurant 实验做扎实

目标：先确认当前复现是否稳定。

建议跑 5 个 seed：

```text
1000
2000
3000
4000
5000
```

论文一般不是只看一个 seed，而是多个 seed 取平均结果。当前只有 seed=1000，只能说明单次实验结果。

### 8.2 路线 B：做增强数据去重实验

目标：验证重复增强样本是否伤害泛化。

实验对比：

```text
原始 C3DA 增强数据
精确去重后的增强数据
精确去重 + 类别平衡后的增强数据
```

主要看：

```text
Macro-F1 是否提升
neutral F1 是否提升
neutral -> positive 错误是否减少
CLLoss 是否不再长期为 0
```

### 8.3 路线 C：加入 NLI 过滤

目标：减少语义漂移、标签不一致、低质量生成样本。

需要额外下载 NLI 模型，仍然默认放到：

```text
J:\nlp\models
```

可选模型：

```text
roberta-large-mnli
MoritzLaurer/deberta-v3-base-mnli-fever-anli
```

考虑显存，第一版建议优先使用 base 级别模型，不建议直接上太大的模型。

### 8.4 路线 D：做跨域 C3DA

目标：从 in-domain ABSA 扩展到 cross-domain ABSA。

基本流程：

```text
源域有标签数据训练初始分类器
目标域无标签数据抽取领域特定片段
初始分类器给目标域样本打伪标签
T5 根据目标域片段和伪标签生成增强数据
NLI 过滤低质量样本
RoBERTa/BERT 使用源域标注数据 + 目标域伪标注增强数据训练
测试目标域
```

## 9. 当前最推荐优先做的改进

按性价比排序：

```text
1. 对生成数据做精确去重。
2. 对增强数据做类别平衡，尤其减少 positive 模板的重复输入。
3. 记录每个类别的增强样本数量和重复率。
4. 调整 k 和 margin，让 CLLoss 真正产生作用。
5. 加入 NLI 过滤低质量增强样本。
6. 用领域特定片段指导生成，而不是只依赖高频词。
7. 跨域时加入伪标签生成。
8. 最后再考虑 GCN。
```

如果要写成论文式创新点，可以概括为：

```text
一种面向跨域方面级情感分析的质量感知数据增强框架。

该框架在 C3DA 的生成增强和对比学习基础上，引入目标域伪标签、领域特定片段感知、NLI 语义一致性过滤以及增强样本类别平衡，以缓解跨域场景下的标签噪声、生成漂移、类别偏置和多方面情感混淆问题。
```

## 10. 可以写进论文或开题的改进点表达

### 改进点 1：伪标签驱动的目标域增强

原始 C3DA 主要在已有标注数据内部做增强。改进后可以使用源域模型对目标域无标签数据生成伪标签，使增强数据更贴近目标域分布。

### 改进点 2：领域特定片段感知

不再只依赖简单高频词，而是通过目标域与源域之间的频率差异提取领域特定片段，指导生成模型产生更符合目标域表达习惯的增强样本。

### 改进点 3：NLI 质量过滤

使用 NLI 模型过滤与原句语义矛盾的生成样本，减少生成模型带来的语义漂移和标签错误。

### 改进点 4：增强样本去重与类别平衡

针对当前实验中生成样本重复严重、positive 偏置明显的问题，对增强样本进行去重和类别平衡，提升 minority class，尤其是 neutral 类的识别能力。

### 改进点 5：更有效的对比学习样本构造

通过增加 k、构造 hard negative、调整 margin，让对比学习真正区分不同 aspect 和不同 polarity 的样本，而不是让 `CLLoss` 长期为 0。

### 改进点 6：可选的图结构增强

GCN 可以作为后续增强模块，用于建模方面词、情感词和领域片段之间的结构关系。但第一阶段不建议直接加入，因为实现复杂度和显存成本较高。

## 11. 后续从 CMD 继续实验的建议

如果只是继续复现 Restaurant 多 seed，直接按下面格式替换 seed：

```cmd
cd /d J:\nlp\C3DA-main && conda activate c3da && python train.py --dataset restaurant --model_name roberta --generate_model t5 --prompt_name lora --ft_epoch 100 --withAugment --withCL --aug_loss_fac 1.0 --margin 0.5 --cl_loss_fac 2.0 --k 1 --num_epoch 15 --batch_size 8 --cuda 0 --model_root J:\nlp\models --seed 2000
```

如果显存不够，把 batch size 降低：

```cmd
cd /d J:\nlp\C3DA-main && conda activate c3da && python train.py --dataset restaurant --model_name roberta --generate_model t5 --prompt_name lora --ft_epoch 100 --withAugment --withCL --aug_loss_fac 1.0 --margin 0.5 --cl_loss_fac 2.0 --k 1 --num_epoch 15 --batch_size 4 --cuda 0 --model_root J:\nlp\models --seed 2000
```

如果测试对比学习更强一点的版本：

```cmd
cd /d J:\nlp\C3DA-main && conda activate c3da && python train.py --dataset restaurant --model_name roberta --generate_model t5 --prompt_name lora --ft_epoch 100 --withAugment --withCL --aug_loss_fac 1.0 --margin 0.2 --cl_loss_fac 2.0 --k 2 --num_epoch 15 --batch_size 6 --cuda 0 --model_root J:\nlp\models --seed 1000
```

## 12. 一句话结论

当前 C3DA 复现实验已经跑通，Restaurant 单 seed 的结果是 `Accuracy=86.86%`、`Macro-F1=81.94%`。主要问题不是模型训练失败，而是生成增强数据质量不足、重复严重、positive 偏置明显、neutral 类识别弱、对比学习信号没有充分发挥。后续最值得做的改进是：生成数据去重、类别平衡、NLI 过滤、领域特定片段抽取、伪标签目标域增强，以及更有效的 hard negative 对比学习。
