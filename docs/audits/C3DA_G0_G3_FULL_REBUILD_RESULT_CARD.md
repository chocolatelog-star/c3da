# G0/G3 完整下游重建结果卡

任务：`C3DA_G0_G3_FULL_PIPELINE_REBUILD_AND_FINAL_COMPARE_V1`

两组均使用现有 extractor checkpoint（提取器检查点），没有重训 extractor。stage status（阶段状态）显示 prepare、generator、augmentation、final_train 和 target evaluation 全部成功。

统一配置：`16×2`（有效 batch=32）、seed=1000、学习率 `3e-4`、最终训练 5 epoch、pseudo weight=0.75、augmentation weight=0.20、complete-multi extra=0.25、sentiment contrastive=0.01、Final DANN=`0.03`、checkpoint=`best`、beam=4、max_new_tokens=96。

| 组 | Selected pseudo F1 | pseudo rows/triplets | augmentation rows | final_train rows | Target Raw P/R/F1 | Fixed F1 |
|---|---:|---:|---:|---:|---:|---:|
| G0 | 61.96% | 545/818 | 116 | 1565 | 55.53/49.69/52.45% | 53.16% |
| G3 | 57.70% | 553/803 | 101 | 1558 | 57.40/51.96/54.55% | 55.63% |

结论：`UPSTREAM_ADVANTAGE_PRESERVED_TO_FINAL=NO`；`RANKING_REVERSAL_STILL_EXISTS=YES`。G0 的上游 Selected pseudo F1 高 4.26 个百分点，但 G3 的最终 Raw F1 高 2.10 个百分点，且 G3 的 precision（精确率）和 recall（召回率）都更高。

限制：本次命令启用了 `minimal_outputs`，因此 complete_multi 及大部分 augmentation/final_train JSONL 在完成后被清理；对应指标写为 `UNAVAILABLE`，没有据此猜测归因。该重建复现了历史最终分数，但不能定位反转首次发生的中间阶段。

TARGET_TEST_GOLD_USED：NO（目标测试金标未用于训练或选模）。

NEXT_ALLOWED_ACTION：`RETURN_TO_CHAT_SOL`
