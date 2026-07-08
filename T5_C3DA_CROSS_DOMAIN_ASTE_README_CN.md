# T5-C3DA 完整跨域 ASTE 实验说明

这个项目把原始 C3DA 的“给定方面词做情感分类”改成完整 ASTE 任务：

```text
输入句子 -> 输出 (aspect, opinion, sentiment) 三元组
```

线性化输出格式如下：

```text
<pos> food <opinion> delicious ; <neg> service <opinion> slow
```

当前版本仍然保留 C3DA 的双通道思想，但第一阶段加入了三个质量控制模块：

- 抽取器使用 `J:\nlp\models\t5-base-py`，不再用 common_gen 模型做抽取起点。
- 生成器继续使用 `J:\nlp\models\mrm8488-t5-base-finetuned-common_gen`。
- 增强生成器使用 label-invariant prompt，要求生成句子必须包含标签里的 aspect 和 opinion。
- 增强阶段默认使用 NLI filter，模型路径是 `J:\nlp\models\nli-deberta-v3-base-mnli-fever-anli`。
- 伪标签阶段会输出质量分析文件，只用于分析，不参与训练和选模型。

## 严格跨域设定

- source train/dev：有标签源域数据，用于训练和验证。
- target train：训练流程中只当无标签数据，用于生成伪三元组。
- target train gold：只写入 `target_train_gold_analysis.jsonl` 做伪标签诊断，不参与训练、不参与选模型。
- target test：只在最后评估一次 micro-F1。
- final dev：固定使用 source dev gold，不从伪标签或增强数据里随机切验证集。

## 关键输出文件

- `target_pseudo.jsonl`：目标域无标签句子的伪三元组。
- `target_pseudo_analysis.json`：伪标签质量分析，包含 hidden gold 对比、情感分布、空预测数量、严格 micro-F1。
- `target_pseudo_predictions_analysis.jsonl`：每条目标域 train 样本的 gold/pseudo 对照，只用于分析。
- `c3da_generator_train.jsonl` / `c3da_generator_dev.jsonl`：生成器训练数据。
- `c3da_two_channel_requests.jsonl`：双通道增强请求。
- `c3da_two_channel_augmented.jsonl`：aspect/opinion 一致性过滤后的增强数据。
- `c3da_two_channel_augmented_nli.jsonl`：再经过 NLI filter 后的增强数据。
- `c3da_augment_analysis.json`：增强阶段统计，包含 NLI 过滤数量、增强通道分布、情感分布。
- `final_train.jsonl`：最终训练集，包含 source gold、target pseudo、过滤后的增强数据。
- `final_dev.jsonl`：最终验证集，固定为 source dev gold。
- `aste_metrics.json`：target test 上的 precision、recall、micro-F1、TP、FP、FN。
- `aste_predictions.jsonl`：target test 每条样本的 gold 和 pred。

## 从 CMD 开始的完整流程

进入项目：

```cmd
J: & cd J:\nlp\CD-C3DA & conda activate c3da
```

准备数据。建议新开一个 run 目录，不要覆盖旧结果：

```cmd
python t5_aste_pipeline.py prepare --source_dataset rest16 --target_dataset laptop14 --run_dir runs\t5_c3da_aste_rest16_to_laptop14_rsda_stage1 --seed 1000
```

训练第一阶段 ASTE 抽取器。这里使用 `t5-base-py`：

```cmd
python t5_absa_train.py --model_path J:\nlp\models\t5-base-py --train_file runs\t5_c3da_aste_rest16_to_laptop14_rsda_stage1\extract_train.jsonl --dev_file runs\t5_c3da_aste_rest16_to_laptop14_rsda_stage1\extract_dev.jsonl --output_dir runs\t5_c3da_aste_rest16_to_laptop14_rsda_stage1\models\extractor --num_train_epochs 10 --per_device_train_batch_size 1 --per_device_eval_batch_size 2 --gradient_accumulation_steps 16 --learning_rate 3e-4 --fp16 --gradient_checkpointing --cuda 0
```

训练 C3DA 生成器。这里继续使用 common_gen 模型：

```cmd
python t5_absa_train.py --model_path J:\nlp\models\mrm8488-t5-base-finetuned-common_gen --train_file runs\t5_c3da_aste_rest16_to_laptop14_rsda_stage1\c3da_generator_train.jsonl --dev_file runs\t5_c3da_aste_rest16_to_laptop14_rsda_stage1\c3da_generator_dev.jsonl --output_dir runs\t5_c3da_aste_rest16_to_laptop14_rsda_stage1\models\generator --num_train_epochs 10 --per_device_train_batch_size 1 --per_device_eval_batch_size 2 --gradient_accumulation_steps 16 --learning_rate 3e-4 --fp16 --gradient_checkpointing --cuda 0
```

用抽取器给目标域无标签训练集生成伪三元组，并输出伪标签质量分析：

```cmd
python t5_aste_pipeline.py pseudo --run_dir runs\t5_c3da_aste_rest16_to_laptop14_rsda_stage1 --model_path runs\t5_c3da_aste_rest16_to_laptop14_rsda_stage1\models\extractor\best --batch_size 2 --num_beams 4 --cuda 0
```

执行双通道增强。默认会先检查 aspect/opinion 是否出现在生成句子中，再用 NLI filter 过滤 contradiction：

```cmd
python t5_aste_pipeline.py augment --run_dir runs\t5_c3da_aste_rest16_to_laptop14_rsda_stage1 --generator_model_path runs\t5_c3da_aste_rest16_to_laptop14_rsda_stage1\models\generator\best --nli_model_path J:\nlp\models\nli-deberta-v3-base-mnli-fever-anli --batch_size 2 --nli_batch_size 8 --num_beams 1 --per_row 1 --seed 1000 --cuda 0
```

训练最终 ASTE 抽取器。这里仍然从 `t5-base-py` 起步，并启用数据来源加权：source gold 权重 1.0，target pseudo 权重 0.5，C3DA augment 权重 0.2。

```cmd
python t5_absa_train.py --model_path J:\nlp\models\t5-base-py --train_file runs\t5_c3da_aste_rest16_to_laptop14_rsda_stage1\final_train.jsonl --dev_file runs\t5_c3da_aste_rest16_to_laptop14_rsda_stage1\final_dev.jsonl --output_dir runs\t5_c3da_aste_rest16_to_laptop14_rsda_stage1\models\final_weighted --num_train_epochs 10 --per_device_train_batch_size 1 --per_device_eval_batch_size 2 --gradient_accumulation_steps 16 --learning_rate 3e-4 --source_weight 1.0 --pseudo_weight 0.5 --augment_weight 0.2 --checkpoint_selection best --fp16 --gradient_checkpointing --cuda 0
```

在目标域测试集上最终评估：

```cmd
python t5_aste_pipeline.py evaluate --run_dir runs\t5_c3da_aste_rest16_to_laptop14_rsda_stage1 --model_path runs\t5_c3da_aste_rest16_to_laptop14_rsda_stage1\models\final_weighted\best --batch_size 2 --num_beams 4 --cuda 0
```

## 消融建议

为了判断是哪一部分带来收益或下降，建议按下面顺序跑：

```text
1. source only
2. source + pseudo
3. source + pseudo + C3DA augment，不开 NLI
4. source + pseudo + C3DA augment，开启 NLI
```

当前主流程默认是第 4 种。

## 3070 8G 注意事项

默认命令已经按 3070 8G 设置：

```text
per_device_train_batch_size = 1
gradient_accumulation_steps = 16
fp16
gradient_checkpointing
```

如果显存不够：

- 把生成或评估的 `--batch_size 2` 改成 `--batch_size 1`。
- 把 `--num_beams 4` 改成 `--num_beams 1`。
- 把 NLI 的 `--nli_batch_size 8` 改成 `--nli_batch_size 4` 或 `2`。
