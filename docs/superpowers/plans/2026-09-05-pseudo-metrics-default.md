# 默认伪标签指标实现计划

> **For agentic workers:** 使用测试驱动方式逐步执行本计划。

**目标：** 让每次 G0-G4 上游实验自动保存 Raw、Qualified、High-precision、Selected 四层伪标签的 Precision、Recall、F1 和 TP/FP/FN。

**架构：** 复用项目现有的 `evaluate_selected_pseudo_against_hidden_gold` 评估实现；`chat_task_runner.py` 仅读取 `target_train_gold_analysis.jsonl` 做事后隐藏金标分析，并显式标记不参与训练、选模和参数选择。

**技术栈：** Python、JSONL、pytest、现有 ASTE 解析与 Raw/Fixed 评估器。

---

### 任务 1：锁定四层指标输出

**文件：** `test_chat_task_runner_pseudo_metrics.py`

- [ ] 测试四个阶段均输出 raw/fixed P/R/F1 与 TP/FP/FN。
- [ ] 运行测试并确认在实现前失败。

### 任务 2：实现统一审计

**文件：** `chat_task_runner.py`

- [ ] 读取每组运行目录中的隐藏分析金标。
- [ ] 对四层伪标签调用现有评估器。
- [ ] 写入 `hidden_gold_audit` 和 `pseudo[*].metrics`。
- [ ] 缺失金标时保留 `UNAVAILABLE`，禁止推断。

### 任务 3：验证和同步

- [ ] 运行新增测试、语法检查。
- [ ] 将已验证脚本同步到服务器，不启动训练。
- [ ] 用现有 G0-G3 产物执行审计，确认指标文件生成。
