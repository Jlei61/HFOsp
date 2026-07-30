# Topic 5 RNN 训练充分性审计：交给执行 agent 的完整 prompt

下面代码块可直接复制给新的 Codex agent。

```text
你现在在仓库：

/home/honglab/leijiaxin/HFOsp

请先阅读并遵守仓库根目录 AGENTS.md。然后依次阅读：

1. docs/topic0_methodology_audits.md
2. docs/archive/topic5/rnn_stage_acceptance_and_training_sufficiency_2026-07-30.md
3. docs/archive/topic5/ordered_history_architecture_audit_v0_1_report_2026-07-29.md
4. docs/archive/topic5/minimal_sequence_kernel_closeout_v0_2_report_2026-07-30.md
5. docs/archive/topic5/constructive_event_generation_sufficiency_v0_1_report_2026-07-30.md
6. config/topic5_interictal_rank_distribution_v0_4.yaml
7. scripts/train_topic5_architecture_control_v0_1.py
8. scripts/train_topic5_interictal_rank_distribution.py
9. src/topic5_rank_distribution.py

## 核心科学问题

不要寻找“更强的 RNN”，也不要重新打开 axis/path/low-rank architecture zoo。

本任务只回答：

1. 当前 linear-state 的完整事件生成阴性，是否来自训练轮数不足或优化未收敛？
2. 是否来自 teacher forcing 与自由 rollout 输入分布不一致？

必须区分：

- optimization sufficiency：同一模型、同一 one-step loss 是否训练充分；
- objective sufficiency：one-step teacher forcing 是否足以支持 free-running generation。

禁止把任何结果写成“RNN 证明了癫痫机制”。

## 数据和封存合同

- 使用既有 34 人 masked contact-rank dataset；
- chronological train80 / heldout20 不变；
- 非参与触点保持 masked，不允许重新引入 phantom rank；
- A/B、physical axis、SOZ、IEI、seizure label 和 early-ictal target 均不得用于调参；
- 超参数选择只能读取 train80 内部的 development/inner-validation；
- outer heldout20 只能在配置完全冻结后读取一次；
- patient-first aggregation，不能让事件多的患者主导；
- 既有 checkpoint 和结果目录只读，不覆盖。

首先对数据 fingerprint、subject list、split 和 target seal 做只读审计。如果与既有合同不一致，停止并报告，不得猜测。

## Phase A：重建真实优化语义

先写审计脚本，明确并输出：

- 每个 LOSO fold 的 outer training patients；
- coverage cycles；
- updates per patient；
- 实际 optimizer step 数；
- 每次 update 覆盖的事件数；
- batch/chunk size；
- gradient accumulation 边界；
- gradient clipping 比例；
- shared model 与 heldout local-offset 的更新次数；
- teacher-forced unroll 的最大/中位 rank steps。

特别验证：正式训练中的 batch_size=1024 只是显存 chunk，还是会改变 optimizer update。

输出：

results/topic5_rnn_training_sufficiency_v0_1/input_audit/

以及机器可读：

TRAINING_SEMANTICS_AUDIT.json

## Phase B：纯收敛审计

模型固定为：

- LinearStateSequenceRNN；
- hidden size 32 为 primary；
- hidden size 64 为 capacity sensitivity；
- observation encoder、candidate mask、next-set/STOP loss 不变；
- 不增加新 contact-mixing 层；
- 不增加新 path/axis 参数。

建议用冻结 development/inner-validation 逐级筛选，避免直接跑完整笛卡尔积。

### B1：training-budget grid

primary：

- learning rate = 1e-3；
- shared coverage cycles = {1, 2, 4}；
- updates per patient = {8, 32}；
- hidden size = 32；
- seeds = {20260725, 20260726, 20260727}。

heldout local-offset calibration cycles 同时比较 {4, 8}，但 shared model 选择优先于 offset calibration。

### B2：learning-rate sensitivity

只在 B1 中最稳定的 training budget 上比较：

- learning rate = {3e-4, 1e-3, 3e-3}；
- AdamW primary；
- Adam 作为单一 optimizer sensitivity；
- weight decay = {0, 1e-4}，若 Adam 使用 0。

不要扩展 optimizer zoo，不使用 outer heldout 选择。

### B3：batch/chunk parity

在同一 seed 和相同 segment/update boundaries 下比较：

- chunk size = 512；
- chunk size = 1024。

如果梯度累计正确，两者参数和 validation NLL 应在数值容差内一致。该实验是工程 parity，不是超参数选择。

### 收敛判据

必须预先写入配置：

- validation contact-choice NLL 为 primary；
- total NLL 和 STOP 为 secondary；
- 连续两个 coverage cycles 的 patient-median validation improvement
  小于 0.002 nats/decision，视为 plateau；
- 同时报告 gradient clipping fraction、parameter update norm、
  train-validation gap 和 seed variance；
- 若 4 cycles 仍持续改善，不能宣称收敛，需扩展到 8 cycles，但只扩展当前最佳配置。

Phase B 结束后冻结唯一配置，并写：

HYPERPARAMETER_FREEZE.json

其中必须记录选择所读取的数据范围，确认 outer heldout 和 ictal target 未读取。

## Phase C：objective sufficiency

在 Phase B 冻结的模型和训练预算下，仅比较以下训练目标：

1. teacher_forced_one_step：
   当前真实 prefix 的 next-set/STOP loss；

2. self_fed_2step：
   第一步使用真实 prefix，第二步使用模型生成或采样的前一步；

3. self_fed_3step：
   最多 3 个连续 rollout steps；

4. scheduled_sampling：
   训练早期主要使用真实 prefix，随后按预先冻结 schedule 增加模型 prefix。

必须满足：

- 所有目标使用相同模型参数量；
- 相同 train events、patient weighting 和 optimizer budget；
- STOP 与 contact choice 分开记录；
- 模型自己生成的 contact 不得重复；
- 不能用 heldout suffix、A/B 或 ictal target 调 rollout loss 权重；
- loss 权重和 schedule 只能在 development/inner-validation 冻结。

Phase C primary endpoint：

- development/inner-validation 的 source-conditioned free rollout
  first-order transition correlation；
- suffix rank Wasserstein；
- suffix precedence correlation/MAE；
- participation MAE；
- event-length/STOP calibration。

one-step NLL 必须同时报告，防止 rollout 目标通过牺牲局部预测换取表面生成改善。

## Phase D：正式冻结确认

只有 Phase B/C 完全冻结后，运行：

- 34 人；
- 3 seeds；
- chronological outer heldout20；
- 当前 teacher-forced reference；
- 最佳 converged teacher-forced；
- 最佳 rollout-aware objective；
- static-only control。

每个条件共享：

- 相同 source rank；
- 相同随机数；
- 相同 rollout count；
- 相同 STOP 定义；
- 相同 eligible contact set。

统计顺序：

1. seed 内先算 metric；
2. 患者内合并 seeds；
3. 患者为统计单位；
4. Epilepsiae 与 Yuquan 分层，同时给 combined cohort；
5. 报效应量、bootstrap CI、正向患者数和 paired test。

不能只给 P 值。

## Go / no-go 解释

### 结果 1：更多训练只改善 validation one-step NLL，不改善自由生成

结论：

当前失败不是简单 under-training；teacher-forced local transition
不足以组成完整事件。

### 结果 2：rollout-aware objective 改善完整事件，但 one-step NLL 不恶化

结论：

上一版完整事件阴性受到 exposure bias / objective mismatch 限制。
更新生成结论，但仍不能升级为生物机制。

### 结果 3：增加 coverage cycles 后 one-step 和自由生成都显著改善

结论：

上一版正式模型训练预算不足；必须撤回“当前模型不足”的强表述，
用收敛配置重做正式结果。

### 结果 4：所有收敛和 rollout-aware 条件仍只改善局部 transition

结论：

可以较有把握地冻结：
短程局部顺序信息存在，但该最小 linear-state 模型不足以生成完整事件。

## 资源和运行要求

- 使用现有 conda/cuda 环境；
- GPU/CPU 可并行，但每个进程设置显存上限；
- 所有长任务必须用 tmux 或 nohup，标准输出和错误输出写入独立 log；
- 不允许覆盖既有结果；
- 每个 run 必须写 run_state.json、config snapshot、seed、git commit、
  hostname、CUDA/PyTorch 版本、峰值显存和运行时间；
- watcher 每 5–10 分钟检查 COMPLETE/FAILED/OOM/NaN；
- OOM 时优先减小 chunk size，不改变 update boundary 或科学合同；
- 网络断开后任务必须能从 manifest 恢复，不能依赖当前 shell。

结果根目录：

results/topic5_rnn_training_sufficiency_v0_1/

目录至少包括：

- input_audit/
- development/
- formal/
- analysis/
- figures/
- logs/

figures/ 必须包含中文 README.md。

## 测试

至少补充：

- chunk 512/1024 梯度累计等价；
- coverage cycle/update count 正确；
- inner-validation 与 outer heldout 隔离；
- ictal target seal；
- self-fed rollout 不重复 contact；
- STOP 后不再更新；
- identical seed 可复现；
- resume 不重复或遗漏 run；
- patient-first aggregation。

先运行相关定向测试，再运行所有 Topic 5 RNN tests。

## 交付

完成后给出：

1. 一句话科学结论；
2. 训练充分性是否关闭；
3. 每个配置的 validation 和 free-rollout 指标；
4. 训练曲线与收敛图；
5. teacher-forced vs rollout-aware 的患者级 paired 图；
6. 代表患者的 observed vs generated rank/precedence/participation 图；
7. Epilepsiae/Yuquan 分层结果；
8. 失败单元、OOM、NaN 和重启记录；
9. 可复现 manifest；
10. 对论文当前 RNN wording 的逐条修订建议。

最终先完成科学验收和文档，再按逻辑分批 commit。不要 push，除非用户明确要求。
```
