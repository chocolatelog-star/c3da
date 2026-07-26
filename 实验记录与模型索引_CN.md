# CD-C3DA 实验记录与模型索引

本文档是 `J:\nlp\CD-C3DA` 的当前实验总览。每次改代码或跑完实验时整体更新相关章节，不在末尾无限追加流水记录。

## 1. 当前状态与差距

| 项目 | 当前值 |
|---|---|
| 主攻跨域组合 | `rest16 -> laptop14` |
| BGCA 论文 label-to-text（标签到文本）F1 | **47.28** |
| 完整从头可复现最佳 | raw P/R/F1 = **58.31 / 42.14 / 48.93**；fixed F1 = **50.21** |
| 相对 BGCA raw F1 | **+1.65** |
| 数值诊断最高 | raw F1 **49.01** / fixed F1 **51.83**；复用了历史增强，只用于归因，不作为正式可复现主线 |
| 已完成工作 | 在真实历史代码边界下重新训练抽取器和生成器，重新生成伪标签与增强，重新组装最终训练集并训练最终模型；精确得到 48.93/50.21 |
| 当前工作 | 原生 GPU 验收已连续精确匹配抽取器、421 条基础伪标签、生成器、150 条增强和 494 条完整伪标签；第 8 阶段发现整文件哈希会把无关空审计字段误判为训练数据差异，已改为训练语义哈希并通过回归测试 |
| 当前候选分支 | `fix/native-best-training-semantic-v2`；校验修正提交 `8472884`，继承原生十阶段实现 `afc0d3d..5a57449` |
| `master` 状态 | `6815bb7`，保持当前已验证版本；原生候选通过完整 GPU 验收前不合并 |
| 当前首次待验证偏差 | `native-best-v1-5a57449` 在第 8 阶段停止：1499 条训练样本的有效训练字段完全一致，仅 34 条增强记录多两个值为 `null` 的审计字段；该运行作为失败现场保留，最终模型未开始训练 |
| 当前主要模型短板 | neutral（中性）召回接近 0；多三元组召回仍低；历史最佳 recall（召回率）仅 42.14 |
| 当前下一步 | 使用新运行编号从头执行 RTX 3070 十阶段验收；逐阶段核对黄金值，最终确认 raw 48.93、fixed 50.21 后再决定是否合并 `master` |

## 2. 当前最佳与 BGCA 对比

主指标使用 raw F1（原始 F1），fixed F1（修正 F1）仅作辅助分析。

| 方法 | 是否从头生成全部产物 | raw P | raw R | raw F1 | fixed F1 | 相对 BGCA raw F1 | 状态 |
|---|---|---:|---:|---:|---:|---:|---|
| BGCA 论文 label-to-text（标签到文本） | 是 | - | - | **47.28** | - | 0.00 | 论文基线 |
| 历史边界精确复现 | 是；本次全部重新训练和生成 | **58.31** | **42.14** | **48.93** | **50.21** | **+1.65** | 当前可复现最佳 |
| 当前代码原生候选 | 是；禁止读取旧运行产物 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 | 前 7 阶段已精确匹配；修正第 8 阶段校验后重新正式验收 |
| 历史增强混合诊断 | 否；复用历史增强 150 条 | 56.22 | 43.44 | **49.01** | **51.83** | **+1.73** | 数值诊断最高，不是正式主线 |
| 完整双三元组，无情感对比 | 是 | - | - | 48.01 | 50.37 | +0.73 | 证明 complete_multi2 有效 |
| 强确定性完整重跑 | 是 | 53.72 | 41.40 | 46.76 | 49.32 | -0.52 | 诊断实验 |
| 旧式随机完整重跑 | 是 | 54.04 | 39.56 | 45.68 | 47.86 | -1.60 | 复现了被覆盖后的另一轨迹 |
| 25 轮 last 生成器 + label-to-text 增强 | 是 | 55.47 | 41.22 | 47.30 | 50.37 | +0.02 | 不替代 8 轮 best 主线 |

## 3. 最佳流程和当前原生模块

当前最佳配方：`configs\recipes\rest16_to_laptop14_best_v1.json`。

| 阶段 | 模块与参数 | 当前运行内输出 |
|---|---|---|
| 1. prepare（准备） | 当前 `t5_aste_pipeline.py prepare`；seed 1000；label-to-text 生成器数据 | 源域抽取数据、生成器数据、目标域无标签数据和测试数据 |
| 2. extractor（抽取器） | `t5-base-py`；25 轮；last（最后轮次）；batch 1；梯度累积 16；fp16 | `model.safetensors` 权重 |
| 3. pseudo（伪标签） | beams 1；最多 128 token；hp1；距离 5 | 全量伪标签和本次实际筛出的高精度伪标签 |
| 4. generator（生成器） | T5-base；源域标签生成句子；8 轮；best（最佳验证轮次） | `model.safetensors` 权重 |
| 5. augment（增强） | masked_mutual（互相掩码）双通道；NLI；抽取器回抽；历史最佳兼容配置；最多 150 条 | 本次生成并筛选的增强数据 |
| 6. prepare_final（最终数据准备） | 当前代码重新准备下游数据；只在同一 `run_id` 内同步本次全量伪标签 | `final_data` 数据目录 |
| 7. complete_multi2（完整双三元组） | hp1 基础上补完整双三元组；距离 5；额外权重 0.25 | 本次实际通过的完整伪标签 |
| 8. build_final_train（组装最终训练集） | 源域 gold + 本次完整伪标签 + 本次增强；增强权重 0.20 | 最终训练集和开发集 |
| 9. final_train（最终训练） | 5 轮 best；伪标签权重 0.65；DANN 0.03；情感对比 0.01；source only；class balanced | 最终模型权重 |
| 10. evaluate（评估） | beams 4；最多 96 token；不使用约束解码 | raw/fixed 指标和 328 条预测 |

领域对抗学习没有取消：最终训练仍使用 `lambda_domain_adv=0.03`。情感对比学习也保留：`lambda_sentiment_contrastive=0.01`。

## 4. 精确复现证据

历史边界精确复现运行：

```text
runs\historical_best_two_stage_v1\rest16_to_laptop14
```

清单：

```text
runs\historical_best_two_stage_v1\rest16_to_laptop14\manifest.json
```

| 产物 | 黄金观察行数 | SHA256 或语义 SHA256 |
|---|---:|---|
| 抽取器 `model.safetensors` | - | `6AD985A7D61274B6553C65B305BE18BBA8618B25B98742F0594C5336A3925F3E` |
| 基础高精度伪标签 | 421 | `0536D99840054EE928B5FB746EC60326640C9A23C8A676A2A8D25DF3D8C15C84` |
| 生成器 `model.safetensors` | - | `0C93F7660E136862428AC23797339D0196047F8C2A1FADE8C99B7635F68CB1CE` |
| 增强 text+label（文本与标签） | 150，上限也是 150 | `5A5B87707BFA6C2D6416AF7962C390207CF1FAC9AFEDD5B7B4799A4C4570B2FF` |
| 完整伪标签 | 494 | `F3C6E0CF841FA84DD3F522248B3C0214B9FD1CC469A991FE853E7AFDE58AB710` |
| 最终训练集 | 1499 | 历史整文件 `4876753D...6A88`；训练语义 `CEE5C1245C7CE4928B86D7246E0F9F44CA89C1B9A24DECE6C37F554A86E565A4` |
| 最终模型 `model.safetensors` | - | `FC8BC8A4736E5CF4A0575C6C52A9349E34363E01556CC5D3397FDF0029AFAB1F` |
| raw/fixed 预测 | 328 | `66E34B17512690C94425E0D64626AF5E101158CB8F5F4DAA705C59D1E5B115A9` |

421、494、1499 是黄金观察值，不是筛选配额。新运行必须使用本次模型实际筛出的全部伪标签；禁止为匹配历史数量裁剪、补齐或读取旧文件。只有增强 150 是配方显式声明的 `selection_limit`（筛选上限）。

增强兼容审计结论：同一批历史模型过滤候选经过当前默认边界过滤后，150 条增强语义哈希为 `F1583596...E8AE8`；显式 `historical_best_v1` 配置为 `5A5B8770...0B2FF`，与黄金完全一致。差异来自后来新增的观点边界过滤和元数据字段，不来自请求随机顺序。

最终训练集校验结论：训练器实际读取 `input`、`target`、`sample_weight`、`augmentation`、`base_id` 和 `id`。历史文件与 `native-best-v1-5a57449` 文件在这六类字段、记录顺序和 1499 条行数上完全一致，训练语义哈希均为 `CEE5C124...65A4`。当前清单仍保存整文件 SHA256 用于审计，但黄金停止条件使用训练语义哈希，避免新增无关审计字段造成误报。

## 5. 可追溯历史

| 实验方向 | Git 提交或代码边界 | 运行目录/结果标签 | raw F1 | fixed F1 | 结论 | 清理状态 |
|---|---|---|---:|---:|---|---|
| 历史边界从头精确复现 | 上游 `9e78904` + 恢复兼容 `a7e7778`；下游 `8c7f6b4` + 命令兼容 `a7d1473` | `historical_best_two_stage_v1` | **48.93** | **50.21** | 全部模型、伪标签、增强和最终训练重新产生；当前正式基准 | 全部保留 |
| 当前代码原生迁移 | `feature/native-best-reproduction-v1`；`afc0d3d..5a57449` | 配方 `rest16_to_laptop14_best_v1` | 待完成 | 待完成 | 已完成来源隔离、命令归档、黄金校验、增强兼容和 Windows 输出修复 | 候选代码保留 |
| 原生 GPU 首次验收 | 候选 `a755300` | `native-best-v1-a755300` | 中断 | 中断 | 抽取器 16/1325 step 遇到 `UnicodeEncodeError`；非模型、显存或 CUDA 错误，日志与清单保留 | 保留失败现场，不删除 |
| 原生 GPU 第二次验收 | `5a57449` | `native-best-v1-5a57449` | 未进入评估 | 未进入评估 | 前 7 阶段黄金值全部匹配；第 8 阶段因 34 条记录新增空审计字段触发整文件哈希误报，最终训练未开始 | 保留失败现场，不删除 |
| 最终训练语义校验修正 | `fix/native-best-training-semantic-v2`；`8472884` | 新运行编号待启动 | 待跑 | 待跑 | 训练语义哈希覆盖六类实际训练字段，历史与新文件均为 `CEE5C124...65A4`；123 个单元测试和 44 个增强函数测试通过 | 候选代码保留 |
| 历史增强混合诊断 | `5057ef2` | `historical_augment_hybrid_seed1000` | **49.01** | **51.83** | 证明增强内容是性能差异主因，但复用历史增强 | 模型、指标、清单保留 |
| 固定训练集关闭强确定性 | `5057ef2` | `final_only_nondeterministic_seed1000` | 47.98 | 50.88 | 强确定性会掉分，但不是唯一原因 | 保留指标和模型 |
| 强确定性完整重跑 | `5057ef2` | `full_pipeline_seed_sweep_v1\seed1000` | 46.76 | 49.32 | 增强与历史仅重合 2/150 | 保留诊断产物 |
| 旧式随机完整重跑 | `5057ef2` | `full_pipeline_legacy_stochastic_v1` | 45.68 | 47.86 | 精确复现了 7 月 21 日覆盖后的轨迹，不是 48.93 轨迹 | 建议最终归档后删除低分模型，尚未删除 |
| complete_multi2_w025 | `62113b4` / `4258bc6` / `0332aee` | `complete_multi2_w025` | 48.01 | 50.37 | 完整双三元组是关键正向改动 | 保留指标 |
| hp2_dist5 | `869466a` | `strict_aug150...hp2_dist5` | 44.44 | 46.87 | 简单放宽伪标签增加噪声 | 建议删除模型，尚未删除 |
| mixed generator（三任务混合生成器） | `e7560c7` / `e320fab` / `925d596` | `mixed_generator_v1` | 44.07 | 46.06 | 混合任务削弱主生成目标 | 坏模型已删除，指标保留 |
| neutral 强增权 | `0c49ba6` / `ce7452e` / `e5f5d47` | `neutral_gain100_max200` | 43.18 | 45.76 | 未解决中性召回并伤害正负类 | 已删除或建议删除，指标保留 |
| pairing loss（配对损失） | `c1082ab` / `123ab39` / `6075ee0` | `pairing_encoder_l001` | 46.49 | 48.86 | 精确率提高但召回下降 | 已删除或建议删除，指标保留 |
| coverage loss（覆盖损失） | `cbeb965` / `e60ca8f` / `44997d4` | `coverage_encoder_l001` | 44.37 | 46.72 | 分类辅助头没有带来有效生成收益 | 坏模型已删除，指标保留 |
| complete_multi2_w035 | `c0b2730` | `complete_multi2_w035` | 45.74 | 47.02 | 双三元组权重过高 | 建议删除，尚未删除 |
| dynamic strict 3+ | `7f3724d` / `9e76a19` | `complete_dynamic3plus_v1` | 45.38 | 47.48 | 3+ 噪声和欠配对明显 | 建议删除，尚未删除 |
| 25 轮 last 生成器 | `68bc0d0` / `11ca672` / `5057ef2` | `bgca_generator25_last_v1` | 47.30 | 50.37 | 训练更久本身不优于 8 轮 best | 保留指标 |

任何“建议删除”项目在用户明确许可前都不执行删除。

## 6. 待改进

| 优先级 | 当前不足 | 改进目标 | 下一步改动 | 接受标准 |
|---|---|---|---|---|
| P0 | 当前代码原生流程已验证前 7 阶段，但尚未完成最终训练和评估 | 证明不依赖历史工作树也能精确复现 | 在训练语义校验修正分支用新运行编号从头运行十阶段并逐阶段核对 | 模型与预测哈希、TP/FP/FN、raw 48.93、fixed 50.21 全部一致 |
| P0 | 过去可跨目录复用上游产物，导致表面重跑实际混合 | 永久阻断混合产物 | 继续维护 `manifest.json`、输入/输出 SHA256 和同一 `run_id` 恢复门禁 | 任一跨目录或变更输入被测试和运行时拒绝 |
| P1 | neutral（中性）F1 接近 0 | 学到真实中性边界 | 先分离否定、缺失属性、弱情绪三类错误，再构造少量高质量样本 | neutral F1 提升且 pos/neg F1 不明显下降 |
| P1 | 多三元组 recall（召回率）仍低 | 提升完整抽取而不引入 3+ 噪声 | 保留 complete_multi2_w025，优化候选多样性与回抽一致性 | 多三元组 raw F1 和 recall 同升，总体 raw F1 不下降 |
| P1 | 48.93 主要依赖精确率 58.31，召回只有 42.14 | 提高召回同时控制 FP | 按 FN 结构设计伪标签/增强，不再单纯放宽数量 | recall 超过 42.14，precision 不低于 58.0，raw F1 超过 48.93 |
| P2 | 单方向有效不等于六组平均有效 | 迁移到其他跨域组合 | 原生复现通过后按同一配方结构运行其他组合，各自不继承本组合黄金数量 | 六组平均 raw F1 提升，所有组合保留独立清单和命令 |

暂不继续投入：hp2 简单放宽、中性损失强增权、mixed generator、coverage classification head、dynamic strict top ratio。

## 7. 运行入口与命令归档

历史边界精确复现使用过的完整 CMD（命令提示符）命令：

```cmd
cmd /c "J: && cd /d J:\nlp\CD-C3DA && conda activate c3da && powershell -NoProfile -ExecutionPolicy Bypass -File run_historical_best_two_stage.ps1 -SourceDataset rest16 -TargetDataset laptop14 -Seed 1000 -Cuda 0 -OutputRoot J:\nlp\CD-C3DA\runs\historical_best_two_stage_v1"
```

当前代码原生试运行入口：

```cmd
cmd /c "J: && cd /d J:\nlp\CD-C3DA && conda activate c3da && powershell -NoProfile -ExecutionPolicy Bypass -File run_best_reproducible_pipeline.ps1 -RunId native-best-dry-run-v1 -OutputRoot J:\nlp\CD-C3DA\runs\reproducible -Cuda 0 -DryRun"
```

训练语义校验修正后的完整 GPU 命令：

```cmd
cmd /c "J: && cd /d J:\nlp\CD-C3DA-native-best-rc-v1 && conda activate c3da && powershell -NoProfile -ExecutionPolicy Bypass -File run_best_reproducible_pipeline.ps1 -RunId native-best-v2-training-semantic -OutputRoot J:\nlp\CD-C3DA-native-best-rc-v1\runs\reproducible -Cuda 0"
```

正式命令不使用 `-AllowDirtyDiagnostic`；断电后重复同一命令和同一 `RunId` 恢复。新运行不得读取 `native-best-v1-5a57449` 或任何历史运行产物。

每次原生运行目录固定保存：

```text
manifest.json
run_command.cmd
commands.jsonl
environment.json
stage_status.json
RUN_RECORD_CN.md
logs\<stage>.log
```

清单还会在 prepare（准备）完成后恢复并保存配方中的源域、目标域、seed（随机种子）、配方路径与配方 SHA256，防止底层准备脚本写入同名清单后丢失运行身份。每个阶段的 `inputs` 显式包含其通过 `run_dir` 隐式读取的文件，因此断点恢复不只检查命令行中直接出现的路径。

Windows 控制台输出会按当前编码安全替换无法显示的单个进度字符，完整原始行仍按 UTF-8 写入阶段日志；控制台显示问题不得中断训练子进程。

## 8. Git、文档与清理规则

- 修改前创建新分支；`master` 只合并完整 GPU 验证后的当前最佳版本。
- 历史提交和历史工作树只用于审计，不作为正式运行输入。
- 每次实验记录 Git 提交、分支、完整命令、配方、环境、输入输出 SHA256、指标和清理状态。
- 跑完实验先更新第 1、2、6 节，再压缩更新历史表；不在末尾追加重复叙述。
- 新结果较差时保留指标、命令、清单和首次错误，标记“建议删除”；用户明确许可后才能删除文件。
- 查看旧代码优先使用隔离工作树，不能覆盖当前代码；正式训练入口不得调用历史工作树。
