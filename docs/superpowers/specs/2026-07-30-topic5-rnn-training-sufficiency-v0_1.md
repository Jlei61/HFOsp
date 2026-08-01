# Topic 5 RNN 训练充分性与目标充分性冻结 spec v0.1

日期：2026-07-30
状态：`DESIGN_LOCKED`
上游合同：`docs/archive/topic5/rnn_stage_acceptance_and_training_sufficiency_2026-07-30.md` §7
执行 prompt：`docs/superpowers/plans/2026-07-30-topic5-rnn-training-sufficiency-agent-prompt.md`
执行报告：`docs/archive/topic5/rnn_training_and_objective_sufficiency_v0_1_report_2026-07-30.md`

## 0. 本轮唯一要回答的问题

上一轮的结论是：冻结的 teacher-forced `LinearStateSequenceRNN` 能改善局部一阶转移
统计，但不能自由生成真实的完整间期传播事件。本轮**只**问这个阴性来自哪里：

1. **optimization sufficiency**：同一模型、同一 one-step loss，训练轮数/优化分辨率
   是否已经足够（是否已经到 validation plateau）？
2. **objective sufficiency**：one-step teacher forcing 这个目标本身，是否足以支持
   free-running generation（是否存在 exposure bias）？

**明确不做**：不找"更强的 RNN"，不重开 axis / path / low-rank architecture zoo，
不新增 contact-mixing 层，不新增 path/axis 参数，不读 IEI / 发作倒计时。

**明确禁止的写法**：任何"RNN 证明了癫痫机制"的表述。本轮结论只关于
"当前最小 linear-state 模型在当前训练合同下能/不能做什么"。

## 1. 数据与封存合同（只读审计已通过）

| 项 | 冻结值 |
|---|---|
| 数据集 | `results/topic5_interictal_rank_distribution/dataset_v0_4` |
| 队列 | 34 人（Epilepsiae 18 / Yuquan 16） |
| 事件 | 864,163（train80 691,314 / heldout20 172,849） |
| 分层 | 逐患者 chronological first-80% / last-20%，**不变** |
| masked 来源 | `results/interictal_propagation_masked/per_subject`，`masked_local_ranks(ranks, bools)`，非参与触点 `group_id = -1` |
| 逐文件指纹 | `load_records()` 每次强制比对 `dataset_npz_sha256` |
| 封存字段 | `forbidden_inputs_present` 全 false：IEI / event_rate / time_to_seizure / seizure_seed / ictal_target / 字符串 ID |

**读取纪律**

- 超参数与训练目标的选择只能读 train80 内部的 development / inner-validation。
- outer heldout20 只在 Phase B/C 全部冻结后，于 Phase D 读取一次。
- A/B、physical axis、SOZ、IEI、seizure label、early-ictal target 全程不读。
- 既有 checkpoint 与结果目录只读，本轮所有产物写入新根目录。

## 2. Development 结构（两段式，已批准）

| 层 | 结构 | 用途 |
|---|---|---|
| **非-LOSO development** | 单一共享模型训练于全部 34 人 train80 的前 90%，验证于 train80 的后 10%（时间序） | B1 / B2 / B3 的廉价宽筛，产出收敛曲线 |
| **LOSO-development** | 33 人 train80-前90% 训 shared core → 第 34 人 train80-前90% 校准 local offset（core 冻结）→ 第 34 人 train80-后10% 评估 | B1c 结构确认 + offset cycles 比较 + Phase C |
| **正式 LOSO** | 33 人 train80 训 shared core → 第 34 人 train80 校准 offset → 第 34 人 **heldout20** 评估 | 仅 Phase D |

非-LOSO 的已知代价：每个患者自己的早期事件参与了训练，绝对指标偏乐观。因此它
只用于**相对排序**，最终 budget 必须在结构忠实的 LOSO-development 上复核。

## 3. Phase A：训练语义审计（只读，不重训）

对既有正式架构审计的每个 LOSO fold 输出：outer training patients、coverage cycles、
updates per patient、实际 optimizer step 数、每次 update 覆盖的事件数、batch/chunk
size、gradient accumulation 边界、gradient clipping 比例、shared 与 heldout offset
的更新次数、teacher-forced unroll 的最大/中位 rank steps。

**必须验证的命题**：正式训练中的 `--batch-size 1024` 只是显存 chunk，不改变
optimizer update 边界。代码层证据 + Phase B3 的数值 parity 共同构成结论。

产物：`results/topic5_rnn_training_sufficiency_v0_1/input_audit/TRAINING_SEMANTICS_AUDIT.json`

## 4. Phase B：纯收敛审计

模型固定 `LinearStateSequenceRNN`；observation encoder、candidate mask、
next-set/STOP loss 全部不变；hidden 32 为 primary，64 为 capacity sensitivity。

### 4.1 收敛判据（预先冻结）

- **primary**：validation **contact-choice NLL**（`decomposed_next_set_stop_loss`
  的 `event_contact_choice_nll`，单位 nats/decision，只在非终止决策上平均），
  按患者聚合后取 **patient median**。
- **secondary**：total NLL、STOP contribution。
- **plateau 判据**：连续两个 coverage cycles 的 patient-median validation
  improvement 均 `< 0.002` nats/decision。
- 同时报告：gradient clipping fraction、parameter update norm、train–validation gap、
  seed variance。
- 若 4 cycles 仍持续改善，**不得宣称收敛**，需把**当前最佳配置**扩展到 8 cycles
  （只扩展最佳配置）。

### 4.2 B1 training-budget grid

- learning rate = 1e-3（固定）
- shared coverage cycles ∈ {1, 2, 4}
- updates per patient ∈ {8, 32}
- hidden size = 32（primary），64（sensitivity）
- seeds ∈ {20260725, 20260726, 20260727}

**嵌套读出**：coverage cycles 是嵌套的——一次训练到 4 cycles，在每个 cycle 末尾
评估一次，即可精确得到 {1,2,4}。前提是 patient order RNG 流与独立短跑一致；
本 spec 要求一项测试断言"4-cycle run 的 cycle-1 快照 == 独立 1-cycle run"。

因此 B1 = updates{2} × hidden{2} × seeds{3} = 12 runs × 4 cycles。

heldout local-offset calibration cycles {4, 8} 的比较放到 B1c（需要 LOSO 结构），
且 **shared model 选择优先于 offset calibration**。

### 4.3 B2 learning-rate sensitivity

只在 B1 选出的最稳定 training budget 上：

- learning rate ∈ {3e-4, 1e-3, 3e-3}
- AdamW 为 primary，Adam 为唯一 optimizer sensitivity
- weight decay ∈ {0, 1e-4}；Adam 固定用 0

共 9 个 cell（AdamW×wd{0,1e-4}×lr{3} + Adam×wd0×lr{3}）。不扩展 optimizer zoo，
不使用 outer heldout 选择。

**两条选择纪律（防止 B2 越权）**：

1. B2 只在**已冻结的训练预算**（B1/B1x 选出的 coverage cycles）上比较，选择限制在最终
   轮次；否则一倍标准误规则可能借道"更省的轮数"把预算改回去。
2. **Adam 只是敏感性臂，不参与选择**。spec 指定 AdamW 为 primary，因此 Adam 的结果
   照常报告但不可被选中。

### 4.4 B3 batch/chunk parity（工程 parity，非超参数选择）

同一 seed、相同 segment/update boundaries 下比较 chunk size ∈ {512, 1024}。

代码层预期：`train_shared_coverage` 在整段上按 `len(chunk)/len(segment)` 加权累积
梯度，而 `next_set_stop_loss` 返回 **event-mean**，故加权和精确等于整段 event-mean，
两者参数与 validation NLL 应在浮点容差内一致。

**适用范围**：parity 只对 `teacher_forced_one_step` 目标成立。self-fed 目标在采样时
消耗随机数，chunk 边界会改变随机数消耗序列；这一点显式记录，不当作缺陷。

### 4.5 B1c LOSO-development 结构确认与冻结

B1/B2 的 top-2 budget × 3 seeds × 34 folds，在 LOSO-development 结构上复核排序；
同一次训练里把 heldout local offset 校准到 8 cycles，并在第 4 个 cycle 末尾快照，
从而**一次训练同时读出 calibration cycles 4 与 8**。

Phase B 结束后冻结唯一配置，写
`development/HYPERPARAMETER_FREEZE.json`，其中必须记录选择所读取的数据范围，
并确认 outer heldout 与 ictal target 未读取。

## 5. Phase C：objective sufficiency

在 Phase B 冻结的模型和训练预算下，只比较四个训练目标。

### 5.1 监督定义（输入替换，已批准）

**唯一改变的是喂进递归状态的 history token；监督目标、candidate mask 与分母
逐字不变。**

对第 t 个决策：

- target = 真实 next-set `S_t`（不变）
- candidate mask = `contact_mask & ~真实prefix`（不变）
- 递归状态的输入 token = 真实 `S_t` 或**模型自己上一步采样出的 contact**

因此四个目标共享逐字相同的 decision 集合、mask 与分母，one-step NLL 可以直接对比。

### 5.2 四个目标

| 目标 | 喂入 schedule（step 0 恒为真实首 rank set） |
|---|---|
| `teacher_forced_one_step` | 全部真实 |
| `self_fed_2step` | 块 `[GT, M]` 循环；最多 1 个连续模型步 |
| `self_fed_3step` | 块 `[GT, M, M]` 循环；最多 2 个连续模型步，即最多 3 个连续 rollout 决策 |
| `scheduled_sampling` | 每步 Bernoulli(p_c)，`p_c` 随 coverage cycle 线性上升，schedule 预先冻结 |

约束：

- 所有目标使用相同模型参数量、相同 train events、相同 patient weighting、
  相同 optimizer budget。
- 模型自采样时，候选集排除**模型自己已生成过的 contact**（不得重复）。
- 只在 `|S_t| == 1` 的步允许模型喂入；`|S_t| > 1`（rank tie）时回落到真实 token，
  以保持 `progress = |recruited| / n_contacts` 与真实路径逐步对齐。tie 比例
  ~3e-5..4e-4，回落率作为诊断记录。
- STOP 与 contact choice 分开记录。
- **shared core 与 held-out local offset 都用同一个目标训练**（端到端）。混合模型
  （core 用 rollout-aware、offset 用 teacher forcing）在任何一个目标下都没有被
  完整训练过，不是合法的比较对象。
- 不得用 heldout suffix、A/B 或 ictal target 调 rollout loss 权重；loss 权重与
  schedule 只在 development / inner-validation 冻结。

### 5.3 Primary endpoint（development / inner-validation）

source-conditioned free rollout（复用 `src/topic5_constructive_event_generator.py`
的 `source_conditioned_rollout`，条件 `full_constructive`）：

1. first-order transition correlation
2. suffix rank Wasserstein
3. suffix precedence correlation / MAE
4. participation MAE
5. event-length / STOP calibration

**one-step NLL 必须同时报告**，防止 rollout 目标以牺牲局部预测换取表面生成改善。

### 5.4 次要诊断：模型自身的 free rollout（`native_model`）

constructive generator 的采样分布是「train80 静态 log-participation + 冻结 ordered
residual」的复合；而 self-fed 训练目标让模型对**自己 next-contact head 的采样**稳健。
两者不是同一个分布，因此只用 constructive rollout 评价 rollout-aware 目标，会让阴性
结果留下"训练与评价采样分布不匹配"的解释口子。

因此每个 cell 额外产出一个 `native_model` rollout：同样揭示真实首 rank set、同样用
逐 (subject, seed) 冻结的 uniform 矩阵做 inverse-CDF 采样（因此跨条件精确配对），
但每一步的 contact 与 STOP 都来自模型自己的联合分布 `softmax([stop_logit, contact_logits])`。

- **primary 仍是 constructive rollout**（与上一轮同分母，可直接对比）；
- `native_model` 是**次要诊断**，用于判断阴性是否可归因于采样分布不匹配；
- 两者的逐患者配对对比也一并记录（`paired_generator_contrast`）。

Phase C 结束后写 `development/OBJECTIVE_FREEZE.json`。

## 6. Phase D：正式冻结确认

只有 Phase B/C 完全冻结后运行：34 人 × 3 seeds × chronological outer heldout20，
四个条件：

1. `current_teacher_forced_reference`（cycles=1、updates=8、既有正式预算）
2. `converged_teacher_forced`（Phase B 冻结预算）
3. `best_rollout_aware`（Phase C 冻结目标 + Phase B 冻结预算）
4. `static_only`（rollout 条件，`residual = 0`）

共享：相同 source rank、相同随机数（逐 (subject, seed) 的冻结 uniform 矩阵）、
相同 rollout count、相同 STOP 定义、相同 eligible contact set。

条件 1 额外承担一项复现检查：与既有归档 `results/topic5_ordered_history_architecture_audit`
的对应 cell 指标比对（既有目录只读）。

### 统计顺序（不得只给 P 值）

1. seed 内先算 metric；
2. 患者内合并 seeds；
3. 患者为统计单位；
4. Epilepsiae 与 Yuquan 分层，同时给 combined cohort；
5. 报效应量、bootstrap CI、正向患者数与 paired test。

## 7. Go / no-go 解释（预先冻结）

| 结果 | 结论 |
|---|---|
| 1. 更多训练只改善 validation one-step NLL，不改善自由生成 | 当前失败不是简单 under-training；teacher-forced local transition 不足以组成完整事件 |
| 2. rollout-aware objective 改善完整事件且 one-step NLL 不恶化 | 上一版完整事件阴性受 exposure bias / objective mismatch 限制；更新生成结论，但仍不能升级为生物机制 |
| 3. 增加 coverage cycles 后 one-step 与自由生成都显著改善 | 上一版正式模型训练预算不足；撤回"当前模型不足"的强表述，用收敛配置重做正式结果 |
| 4. 所有收敛与 rollout-aware 条件仍只改善局部 transition | 可较有把握冻结：短程局部顺序信息存在，但该最小 linear-state 模型不足以生成完整事件 |

## 8. 目录与运行要求

```
results/topic5_rnn_training_sufficiency_v0_1/
├── input_audit/     TRAINING_SEMANTICS_AUDIT.json + 逐 fold CSV
├── development/     b1_budget/ b2_learning_rate/ b3_chunk_parity/
│                    b1c_loso_confirm/ c_objectives/
│                    HYPERPARAMETER_FREEZE.json OBJECTIVE_FREEZE.json
├── formal/          34 × 3 seeds × 4 conditions
├── analysis/        患者级统计、分层、paired tests
├── figures/         含中文 README.md
└── logs/
```

- conda env `cuda_env`；每进程设 `torch.cuda.set_per_process_memory_fraction`。
- 长任务用 tmux/nohup，stdout 与 stderr 写独立 log。
- 不覆盖既有结果；每个 run 写 `run_state.json`、config snapshot、seed、git commit、
  hostname、CUDA/PyTorch 版本、峰值显存与运行时间。
- watcher 每 5–10 分钟检查 COMPLETE / FAILED / OOM / NaN。
- OOM 时**只减小 chunk size**，不改变 update boundary 或科学合同。
- 断线后可从 manifest 恢复：已有 `DONE.json` 的 cell 跳过；存在但未完成的 cell
  阻塞恢复并报错，不静默重跑。

## 9. 测试合同

`tests/test_topic5_training_sufficiency.py` 至少覆盖：

1. chunk 512/1024 梯度累计等价（teacher-forced）
2. coverage cycle / update count 正确（含 4-cycle run 的 cycle-1 快照 == 独立 1-cycle run）
3. inner-validation 与 outer heldout 隔离（split==2 永不被训练或评估触碰）
4. ictal target seal（manifest 与 run summary 均断言未读）
5. self-fed rollout 不重复 contact
6. STOP 后不再更新状态
7. identical seed 可复现
8. resume 不重复或遗漏 run
9. patient-first aggregation（事件多的患者不主导）

先跑定向测试，再跑全部 Topic 5 RNN 相关测试。
