# G0-G3 实际 Final Train 审计结果卡

任务：`C3DA_G0_G3_ACTUAL_FINAL_TRAIN_AUDIT_V1`

有效性：部分完成。服务器已重新连接并完成只读文件核对；没有训练、改模型或使用目标测试金标。由于历史清理，G1-G3 的增强和 final_train 原始 JSONL 不存在。

已确认的最终 Raw F1（原始三元组 F1）：G0 52.45%，G1 53.69%，G2 54.09%，G3 54.55%。

关键证据：四组实际 adapter pseudo 文件均存在，SHA256 和行/三元组统计已记录；它们与此前重新审计目录不是同一批文件。G0 Final ASTE 实际使用的 pseudo 路径已由组成摘要确认，G0 最终训练为 source 906、pseudo 545、augmentation 116、总计 1565。G1-G3 的 augmentation/final_train 原始文件已清理，无法计算其阶段统计或确定反转最早阶段。

当前归因：主因 `G`（实际下游产物/配方一致性尚未被完整证实），次因 `H`（证据不足）。不支持直接断言是 augmentation 破坏或 multi-triplet 供给导致。

下一步：`RETURN_TO_CHAT_SOL`。
