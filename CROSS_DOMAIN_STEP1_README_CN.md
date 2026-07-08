# CD-C3DA 第一步：跨域 Baseline 实验说明

本文档记录当前跨域改造的第一步：让训练脚本支持 `source_dataset -> target_dataset`。

## 1. 当前已实现内容

在 `train.py` 中新增了两个参数：

```text
--source_dataset
--target_dataset
```

现在可以做到严格跨域评估：

```text
用 source_dataset 的 train.json 切分 source-train / source-dev
用 source-train 训练
用 source-dev 选择最佳 checkpoint
训练结束后只在 target_dataset 的 test.json 上评估一次
```

例如：

```text
Restaurant -> Laptop
Laptop -> Restaurant
Restaurant -> Twitter
Twitter -> Restaurant
```

如果不传 `--source_dataset` 和 `--target_dataset`，脚本仍然兼容原来的 `--dataset` 单域训练方式。

## 2. 第一步实验目标

这一步不是为了直接做最强模型，而是先建立跨域基线：

```text
B0：Strict Source-only
只用源域标注数据训练和选模型，最后直接在目标域测试。
```

它用来回答：

```text
原始 RoBERTa ABSA 分类器在跨域时会掉到什么水平？
Restaurant -> Laptop 的 domain gap 有多大？
Laptop -> Restaurant 的 domain gap 有多大？
哪一类情感在跨域时最容易错？
```

## 3. 推荐先跑的命令

Restaurant -> Laptop：

```cmd
cd /d J:\nlp\CD-C3DA && conda activate c3da && python train.py --source_dataset restaurant --target_dataset laptop --model_name roberta --num_epoch 10 --batch_size 8 --dev_ratio 0.1 --cuda 0 --model_root J:\nlp\models --seed 1000
```

Laptop -> Restaurant：

```cmd
cd /d J:\nlp\CD-C3DA && conda activate c3da && python train.py --source_dataset laptop --target_dataset restaurant --model_name roberta --num_epoch 10 --batch_size 8 --dev_ratio 0.1 --cuda 0 --model_root J:\nlp\models --seed 1000
```

如果显存不够，把 batch size 改成 4：

```cmd
cd /d J:\nlp\CD-C3DA && conda activate c3da && python train.py --source_dataset restaurant --target_dataset laptop --model_name roberta --num_epoch 10 --batch_size 4 --dev_ratio 0.1 --cuda 0 --model_root J:\nlp\models --seed 1000
```

训练日志中：

```text
source_dev_acc / source_dev_f1
```

用于训练过程中的模型选择。

最终日志中：

```text
target_test_acc / target_test_f1
```

才是跨域目标域测试结果。

## 4. 后续逐步增加模块的顺序

建议按下面顺序做消融实验：

```text
B0 Source-only
B1 Source + original C3DA augmentation
M1 Source + target pseudo-label
M2 M1 + target-domain specific fragments
M3 M2 + cross-domain AAC/PAC generation
M4 M3 + deduplication and class balance
M5 M4 + NLI filtering
M6 M5 + multi-aspect aspect-polarity composition
M7 M6 + hard negative contrastive learning
M8 optional GCN
```

每次只增加一个核心模块，这样才能看出哪个模块真正有用。

## 5. 当前模型目录检查结果

`J:\nlp\models` 当前顶层目录大小大致为：

```text
bert-base-uncased                    3294.1 MB
roberta-base                         2684.8 MB
mrm8488-t5-base-finetuned-common_gen 2266.3 MB
facebook-bart-base                   2130.4 MB
stanza_resources                     1075.0 MB
```

确实有多下载的模型格式。

例如 `bert-base-uncased` 中包含：

```text
tf_model.h5
rust_model.ot
model.onnx
coreml
pytorch_model.bin
model.safetensors
flax_model.msgpack
```

当前 PyTorch/Transformers 训练通常只需要：

```text
config.json
tokenizer.json / vocab.txt / merges.txt / spiece.model
pytorch_model.bin 或 model.safetensors 二选一
special_tokens_map.json
tokenizer_config.json
```

不需要的候选包括：

```text
tf_model.h5
flax_model.msgpack
rust_model.ot
model.onnx
coreml
.cache
```

注意：我还没有删除任何文件。是否清理模型目录，需要你明确许可。

## 6. 下一步代码任务

第一步 baseline 跑通后，下一步建议实现：

```text
pseudo_label.py
```

功能：

```text
读取目标域 train.json 或 unlabeled.json
使用源域训练好的 RoBERTa 模型预测 polarity
保存 sentence、aspect、pseudo_label、confidence
只保留 confidence >= 阈值的样本
```

这就是把 RSDA 的伪标签思想接进 C3DA 的第一步。
