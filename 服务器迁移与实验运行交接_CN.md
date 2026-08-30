# CD-C3DA 服务器迁移与实验运行交接

> 更新时间：2026-08-30 23:17（北京时间）
>
> 本文只保存服务器运行约束和当前产物位置，不保存SSH（安全外壳）密码。研究结论以`.ai/PROJECT_STATE.md`和正式实验索引为准。

## 1. 当前服务器

```text
SSH主机：connect.nmb1.seetacloud.com
SSH端口：48759
SSH用户：root
项目目录：/root/CD-C3DA
数据盘：/root/autodl-tmp
GPU：NVIDIA GeForce RTX 4090，约24 GB显存
```

服务器更换时只更新主机、端口、项目路径和环境，不在仓库记录密码。

## 2. 环境与目录

```text
Conda：/root/miniconda3
环境：c3da
Python：3.10
PyTorch：2.2.2+cu121
Transformers：4.39.3
Accelerate：0.28.0
Stanza：1.14.0

实验输出：/root/autodl-tmp/CD-C3DA-runs/
临时目录：/root/autodl-tmp/tmp/
```

所有新实验必须设置`TMPDIR=/root/autodl-tmp/tmp`，不得把新运行写入`/root/CD-C3DA/runs`。

## 3. 代码同步规则

本地完成实现、CPU（中央处理器）测试和Codex Sol（高级工程模型）复审后：

1. 提交并推送功能分支；
2. 服务器执行`git fetch origin`；
3. 切换到明确的功能分支和提交；
4. 检查`git status --porcelain`为空；
5. 记录`git rev-parse HEAD`；
6. 再启动用户批准的服务器命令。

禁止直接在服务器长期修改正式代码。若服务器上做了紧急修复，必须形成独立提交并回推GitHub（代码托管平台），随后同步回本地。

## 4. 已完成服务器实验

历史`rest16 -> laptop14` batch=100速度配置得到raw/fixed F1=44.18/45.98，低于历史保护基线48.93/50.21；它改变了batch、梯度累积和梯度检查点，不是正式复现，模型已清理。

元素感知RGAT（关系图注意力网络）Treatment-only（仅实验组）batch=1/4/8/16均已完成。关键结果：

```text
batch16 source-dev strict F1：56.58%
batch16 multi recall：48.11%
batch16 overall absence：69.77%
batch16 qualified total/multi：552/209
```

服务器仅保留batch16 best模型及四组指标、日志、伪标签分析和入口身份文件；batch1/4/8及旧Treatment模型副本已删除。

## 5. 当前服务器下一任务

只允许运行Focus-only（仅聚焦）与Coverage-only（仅覆盖）组件归因。固定：

```text
方向：laptop14 -> rest15
seed：1000
T5-base：固定
图层：1
图隐藏维：256
注意力头：4
DANN：0
Focus-only：lambda_focus=0.05，lambda_coverage=0
Coverage-only：lambda_focus=0，lambda_coverage=0.05
```

不得从batch16 best热启动；两组必须按同一原始初始化流程训练。当前功能分支为`codex/m1-element-aware-multi-triplet-rgat-v1`，已推送提交`135c8be`，但组件消融测试尚待提交和最终复审，故当前文档不提供正式GPU（图形处理器）启动命令。

## 6. 运行边界

- 由用户手动启动服务器实验，除非用户明确要求Codex代跑。
- 先运行CPU测试/入口审计，再运行GPU诊断。
- Phase A（阶段A）完成后停止。
- 禁止自行启动Phase B、增强、最终ASTE（方面级情感三元组抽取）或目标测试。
- 不得使用目标测试金标选择组件、阈值、损失、batch或检查点。
- 每个阶段必须记录进度、可恢复状态、命令、提交、模型/缓存哈希和最终判定。

## 7. 接手检查

接手服务器任务时依次检查：磁盘空间、GPU状态、当前进程、仓库分支/提交、工作区洁净度、输出目录、阶段状态和最新日志。GPU显存未占满不代表参数错误；不得因速度或利用率临时改变研究配置。

## 8. GPU并行、启停与交付门槛

- 当前4090约24 GB显存默认并行2个独立实验；只有单任务烟雾测试峰值不超过约7 GB、三任务预计总峰值后仍预留至少3 GB，且CPU、内存、磁盘与数据加载稳定时才并行3个。
- 并行不能改变正式配方。每个任务必须绑定独立`run_id`、输出目录、日志、检查点和临时子目录；禁止共享可写缓存或阶段状态。
- 服务器运行期间不执行`git pull`、切换分支或现场修改代码。需要新提交时，在本地完成测试、复审和推送，等待现有任务结束后再切换，或使用隔离工作树。
- 开启GPU模式前必须具备：已推送提交、干净工作区、冻结单行命令、唯一输出目录、恢复入口和5至10步烟雾测试方案。烟雾测试通过后从新正式目录运行。
- 若代码实现或复审预计还需30分钟以上且没有可运行实验，建议切换无GPU模式或关机；代码即将完成时必须给用户明确预计时间，避免GPU付费空闲。
- GPU正在运行时，本地继续论文写作、表格、引用、结果分析和下一任务CPU实现。论文截止关键路径见`03_CD-C3DA下一阶段改进计划_CN.md`。
