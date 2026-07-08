# T5-only 完整跨域 ABSA 改造说明

本文档说明当前项目如何从原始 C3DA 的“给定方面词做情感分类”改造成“完整跨域 ABSA / AESC”。

## 1. 任务定义变化

原始 C3DA 任务：

```text
输入：sentence + gold aspect
输出：positive / negative / neutral
```

这是 ASC，不负责方面词抽取。

当前 T5-only 改造后的任务：

```text
输入：sentence
输出：<pos> aspect ; <neg> aspect ; <neu> aspect
```

这是完整 AESC：

```text
Aspect Extraction + Aspect Sentiment Classification
```

例如：

```text
输入：The battery life is terrible but the screen is bright.
输出：<neg> battery life ; <pos> screen
```

## 2. 如何保留 C3DA 思想

原始 C3DA 的核心思想：

```text
1. T5 生成增强样本
2. AAC：方面通道增强
3. PAC：情感通道增强
4. 生成数据用于提升最终 ABSA 模型
5. 对增强样本做过滤和对比学习
```

T5-only 完整 ABSA 版本中的对应关系：

```text
extract channel：sentence -> aspect-polarity labels
generate channel：aspect-polarity labels -> sentence
```

也就是：

```text
Extractor T5：抽取目标域伪标签
Generator T5：根据伪标签生成目标域增强句
Final T5：用源域标注数据 + 目标域增强数据训练完整 AESC 模型
```

后续可以继续加入：

```text
NLI 过滤
生成去重
类别平衡
领域特定片段感知
多方面 aspect-polarity 组合生成
T5 一致性对比约束
```

## 3. 新增文件

```text
t5_absa_data.py
t5_absa_train.py
t5_absa_pipeline.py
```

作用：

```text
t5_absa_data.py      数据格式转换、标签解析、完整 AESC micro-F1
t5_absa_train.py     T5 seq2seq 微调脚本
t5_absa_pipeline.py  prepare / augment / evaluate 统一入口
```

## 4. 第一阶段：严格 Source-only 完整跨域 ABSA

先只训练 T5 extractor，不使用目标域伪标签增强。

### 4.1 准备数据

```cmd
cd /d J:\nlp\CD-C3DA && conda activate c3da && python t5_absa_pipeline.py prepare --source_dataset restaurant --target_dataset laptop --run_dir runs\t5_aesc_restaurant_to_laptop --dev_ratio 0.1 --seed 1000
```

### 4.2 训练 T5 extractor

RTX 3070 8GB 推荐参数：

```cmd
cd /d J:\nlp\CD-C3DA && conda activate c3da && python t5_absa_train.py --model_path J:\nlp\models\mrm8488-t5-base-finetuned-common_gen --train_file runs\t5_aesc_restaurant_to_laptop\extract_train.jsonl --dev_file runs\t5_aesc_restaurant_to_laptop\extract_dev.jsonl --output_dir runs\t5_aesc_restaurant_to_laptop\models\extractor --num_train_epochs 10 --per_device_train_batch_size 1 --per_device_eval_batch_size 2 --gradient_accumulation_steps 16 --fp16 --gradient_checkpointing --cuda 0 --seed 1000
```

### 4.3 评估目标域完整 AESC

```cmd
cd /d J:\nlp\CD-C3DA && conda activate c3da && python t5_absa_pipeline.py evaluate --run_dir runs\t5_aesc_restaurant_to_laptop --model_path runs\t5_aesc_restaurant_to_laptop\models\extractor\best --batch_size 2 --cuda 0
```

输出指标：

```text
precision
recall
micro_f1
```

其中 `micro_f1` 是完整 aspect+sentiment 匹配 F1。只有方面词和情感都对，才算正确。

## 5. 第二阶段：加入 C3DA 式 T5 双向增强

### 5.1 训练 T5 generator

```cmd
cd /d J:\nlp\CD-C3DA && conda activate c3da && python t5_absa_train.py --model_path J:\nlp\models\mrm8488-t5-base-finetuned-common_gen --train_file runs\t5_aesc_restaurant_to_laptop\generate_train.jsonl --dev_file runs\t5_aesc_restaurant_to_laptop\generate_dev.jsonl --output_dir runs\t5_aesc_restaurant_to_laptop\models\generator --num_train_epochs 10 --per_device_train_batch_size 1 --per_device_eval_batch_size 2 --gradient_accumulation_steps 16 --fp16 --gradient_checkpointing --cuda 0 --seed 1000
```

### 5.2 用目标域无标签数据生成伪标签和增强样本

```cmd
cd /d J:\nlp\CD-C3DA && conda activate c3da && python t5_absa_pipeline.py augment --run_dir runs\t5_aesc_restaurant_to_laptop --batch_size 2 --cuda 0 --seed 1000
```

### 5.3 训练最终 T5 AESC 模型

```cmd
cd /d J:\nlp\CD-C3DA && conda activate c3da && python t5_absa_train.py --model_path J:\nlp\models\mrm8488-t5-base-finetuned-common_gen --train_file runs\t5_aesc_restaurant_to_laptop\final_train.jsonl --dev_file runs\t5_aesc_restaurant_to_laptop\final_dev.jsonl --output_dir runs\t5_aesc_restaurant_to_laptop\models\final --num_train_epochs 10 --per_device_train_batch_size 1 --per_device_eval_batch_size 2 --gradient_accumulation_steps 16 --fp16 --gradient_checkpointing --cuda 0 --seed 1000
```

### 5.4 评估最终模型

```cmd
cd /d J:\nlp\CD-C3DA && conda activate c3da && python t5_absa_pipeline.py evaluate --run_dir runs\t5_aesc_restaurant_to_laptop --model_path runs\t5_aesc_restaurant_to_laptop\models\final\best --batch_size 2 --cuda 0
```

## 6. 后续消融顺序

建议按下面顺序逐步增加模块：

```text
B0 T5 source-only extractor
B1 T5 extractor + T5 generator pseudo augmentation
M1 B1 + exact deduplication
M2 M1 + class balance
M3 M2 + target-domain specific fragments
M4 M3 + NLI filtering
M5 M4 + multi-aspect label composition
M6 M5 + T5 bidirectional consistency filtering
M7 optional GCN / domain-specific fragment graph
```

## 7. 注意

当前 T5-only 流程已经是完整跨域 ABSA/AESC：

```text
模型测试时只输入目标域句子，不输入 gold aspect。
```

目标域 test 的标签只用于最后计算 micro-F1，不参与训练和模型选择。

