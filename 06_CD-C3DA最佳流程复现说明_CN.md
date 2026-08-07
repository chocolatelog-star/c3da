# CD-C3DA 最佳流程复现说明

## 1. 正式结论

当前代码已经两次证明能够完整从头复现历史最佳：

1. 历史代码边界复现 `historical_best_two_stage_v1`：重新训练提取器和生成器，重新生成伪标签与增强，再训练最终模型，得到 raw F1 48.93、fixed F1 50.21。
2. 当前代码原生十阶段复现 `native-best-v2-training-semantic`：不读取历史运行产物，黄金模型、数据、预测和指标全部匹配。
3. 2026-08-03 再次从 `master` 运行 `master-best-check-v1`，十阶段全部完成，再次得到完全相同指标。

因此，`48.93/50.21` 是当前正式可复现最佳，不是偶然保存的旧模型结果。

## 2. 权威位置

```text
主仓库：J:\nlp\CD-C3DA
主分支：master
主分支提交：d2f2a35
验收代码提交：558e4de
正式配方：configs\recipes\rest16_to_laptop14_best_v1.json
最新验收：runs\reproducible\rest16_to_laptop14_best_v1\master-best-check-v1
独立验收仓库：J:\nlp\CD-C3DA-native-best-rc-v1
```

## 3. 黄金结果

```text
raw precision = 0.5831202046035806
raw recall    = 0.4214417744916821
raw F1        = 0.4892703862660945
raw TP/FP/FN  = 228/163/313

fixed precision = 0.59846547314578
fixed recall    = 0.43253234750462105
fixed F1        = 0.502145922746781
fixed TP/FP/FN  = 234/157/307
```

## 4. 关键黄金哈希

| 产物 | SHA256 |
|---|---|
| 提取器 | `6AD985A7D61274B6553C65B305BE18BBA8618B25B98742F0594C5336A3925F3E` |
| 基础高精度伪标签 | `0536D99840054EE928B5FB746EC60326640C9A23C8A676A2A8D25DF3D8C15C84` |
| 生成器 | `0C93F7660E136862428AC23797339D0196047F8C2A1FADE8C99B7635F68CB1CE` |
| 增强语义 | `5A5B87707BFA6C2D6416AF7962C390207CF1FAC9AFEDD5B7B4799A4C4570B2FF` |
| 完整伪标签 | `F3C6E0CF841FA84DD3F522248B3C0214B9FD1CC469A991FE853E7AFDE58AB710` |
| 最终训练语义 | `CEE5C1245C7CE4928B86D7246E0F9F44CA89C1B9A24DECE6C37F554A86E565A4` |
| 最终模型 | `FC8BC8A4736E5CF4A0575C6C52A9349E34363E01556CC5D3397FDF0029AFAB1F` |
| 预测 | `66E34B17512690C94425E0D64626AF5E101158CB8F5F4DAA705C59D1E5B115A9` |

黄金哈希用于验收和定位第一次偏差，不得用来反向篡改或裁剪本次训练产物。

## 5. 正式配方要点

- 源域 `rest16`，目标域 `laptop14`，随机种子1000。
- 提取器25轮，选择最后检查点。
- 生成器8轮，选择最佳检查点。
- `masked_mutual` 双通道增强，最多150条，权重0.20。
- 基础高精度单三元组伪标签加严格完整双三元组，额外权重0.25。
- 最终伪标签权重0.65。
- 最终模型5轮，领域对抗0.03，情感对比0.01。
- 目标域训练标签隐藏；目标测试集只用于最终评估。

## 6. 运行命令模板

正式运行必须从干净的当前提交启动，并使用新的 `RunId`：

```cmd
cmd /c "J: && cd /d J:\nlp\CD-C3DA && conda activate c3da && powershell -NoProfile -ExecutionPolicy Bypass -File run_best_reproducible_pipeline.ps1 -RunId <新的运行ID> -OutputRoot J:\nlp\CD-C3DA\runs\reproducible -Cuda 0"
```

断点恢复时重复完全相同的命令和 `RunId`。若 Git 提交、配方或输入哈希不同，流程应拒绝恢复，不得手工修改清单绕过保护。

## 7. 不得使用的做法

- 不得把 `historical_best_two_stage_v1` 的伪标签、增强或模型复制进新运行。
- 不得把 `native-best-v2-training-semantic` 作为新实验上游。
- 不得把421、494或1499当作所有方向的固定配额。
- 不得因为进度条乱码中止训练；完整 UTF-8 输出保存在阶段日志。
- 不得把 `49.01/51.83` 诊断结果称为正式最佳，因为它复用了历史增强。
- 不得把生成器轮次扫描的较低结果误解为主线无法复现；轮次扫描属于不同实验分支和产物。

## 8. 验收顺序

1. 外部数据和模型哈希匹配。
2. 十阶段全部完成。
3. 模型与数据黄金哈希匹配。
4. 预测哈希匹配。
5. raw/fixed 指标及 TP/FP/FN 匹配。
6. `manifest.json`、`commands.jsonl`、`environment.json` 和日志完整。
