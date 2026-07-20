# 动态严格 3+ 组合实验说明

## 目的

本轮改动用于验证：在当前最好流程 `hp1 + complete_multi2_w025 + DANN + sentiment contrastive` 的基础上，只少量补充严格动态筛选出的完整 `3+` 三元组伪标签，是否能进一步提升多三元组召回，同时避免普通动态多三元组带来的噪声。

## 代码改动

| 项目 | 内容 |
|---|---|
| 分支 | `feature/complete-multitriplet-ablation` |
| 新增参数 | `--complete_dynamic_extra_weight`、`--complete_dynamic_min_triplets` |
| 新增伪标签构建 | `select_complete_dynamic_pseudo` |
| 新增筛选逻辑 | 只接受 `dynamic_strict_high_precision_pseudo=True` 且三元组数量不少于 `complete_dynamic_min_triplets` 的完整动态样本 |
| 默认实验权重 | `complete_multi_extra_weight=0.25`，`complete_dynamic_extra_weight=0.20`，`complete_dynamic_min_triplets=3` |
| 输出标记 | `complete_multi2_w025_dynamic_strict3plus_dist5_w020` |

## 实验逻辑

1. 使用 `hp1_dist5` 作为基础高精度伪标签。
2. 使用 `complete_multi2_w025` 补充完整双三元组伪标签。
3. 额外生成 `dynamic_strict_dist5`，只保留完整通过的动态多三元组样本。
4. 从 `dynamic_strict_dist5` 中只补充 `3+` 三元组样本，权重设为 `0.20`。
5. 复用当前最好流程的增强策略、DANN 和最终阶段情感对比学习设置。

## 完整运行命令

```cmd
cmd /c "J: && cd /d J:\nlp\CD-C3DA\.worktrees\complete-multitriplet-ablation && conda activate c3da && python run_bgca_aste_stage1_pairs.py --output_root runs\bgca_aste_stage1_complete_dynamic3plus_v1 --pairs rest16:laptop14 --generator_prompt_style label_to_text --augment_prompt_style masked_mutual --domain_prefix_style text --complete_multi_extra_weight 0.25 --complete_dynamic_extra_weight 0.20 --complete_dynamic_min_triplets 3 --lambda_sentiment_contrastive 0.01 --sentiment_contrastive_source_only --sentiment_contrastive_class_balanced --eval_batch_size 2 --cuda 0 --seed 1000"
```

## 结果记录

实验完成后，把 `runs\bgca_aste_stage1_complete_dynamic3plus_v1\results_bgca_aste_stage1_complete_multi2_w025_dynamic_strict3plus_dist5_w020_sentiment_contrastive_l001_source_balanced_CN.md` 中的结果和当前最好版本对比，再整体同步到总实验记录。
