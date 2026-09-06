# G3 上游句法候选选择设计

## 目标

复用当前 G3 最佳上游产物（Raw F1 55.88%），只改变下游数据增强阶段的候选元素选择方式，分别运行 A1（aspect 句法候选）、A2（opinion 句法候选）和 A3（双通道句法候选），验证句法兼容排序是否改善最终 ASTE（方面-情感-意见抽取）结果。

## 不变边界

- 不重新训练 extractor（提取器）。
- 不重新生成 selected pseudo（筛选伪标签）或 complete-multi（多三元组补全）产物。
- 不修改图结构、pseudo selector（伪标签选择器）或 replacement generation（替换生成）逻辑。
- 保持 `masked_mutual`（互相遮盖生成）、无领域前缀、`coupled_random`（耦合随机）意见替换。
- 保持 Final ASTE（最终方面-情感-意见抽取）训练参数：batch=16、梯度累积=2、seed=1000、学习率=3e-4、epoch=5、pseudo 权重=0.75、augmentation 权重=0.20、complete-multi 权重=0.25、sentiment contrastive=0.01、Final DANN=0.03、best checkpoint、beam=4、max_new_tokens=96。
- Structure Hard Filtering（结构硬过滤）保持关闭。

## 候选选择

候选保持 occurrence-level（出现位置级）元数据，包括文本、元素类型、情感、来源域、来源 row、来源句子、token span、UPOS/POS、dependency relation、head index 和 head POS。

对每个 replacement request（替换请求），比较原元素与候选元素的三类句法特征：UPOS、dependency relation、head POS。兼容项加分，明显不兼容项降低分数；不要求三项全部精确相等，部分兼容候选仍可参与排序。

当句法候选不足、没有可接受候选或句法信息缺失时，回退到当前 selector（选择器）。每次回退记录 `syntax_fallback=true`，并汇总回退率。A1 只作用于 aspect 通道，A2 只作用于 opinion 通道，A3 同时作用于两个通道。

## 审计输出

每个变体记录：候选请求数、句法前后候选数、UPOS/依存关系/head POS 兼容率、高兼容率、无兼容候选请求数、回退请求数和回退率、候选选择变化率。增强阶段继续记录生成数、旧过滤通过数、最终选中数、edited validity（编辑三元组有效率）、untouched retention（未编辑三元组保留率）、triplet-count preservation（三元组数量保持率）和 unplanned-triplet rate（非计划三元组率）。

## 数据流

现有 selected pseudo 和 complete-multi 作为只读输入，沿用现有 generator（生成器）、NLI（自然语言推断）和 model filter（模型过滤）。只有 replacement candidate ranking（替换候选排序）增加句法兼容项。增强产物、final_train（最终训练集）、预测、指标、manifest（清单）和配置快照写入各自 A1/A2/A3 目录，保留 parent-child mapping（父子样本映射）。

## 验证

先增加纯函数级测试，覆盖兼容评分、A1/A2/A3 通道开关和候选不足回退；再运行现有测试、Python 编译检查和参数传递检查。代码验证通过后才给出服务器实验命令。
