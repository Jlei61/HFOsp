# Continuous Marked State H2b Cross-task Transfer v0.1：技术报告

## 1. 判决

本版本完成了 E384 的冻结状态跨任务仪器和 fail-closed 机器审计，但没有形成 checkpoint-available cohort 结果。预注册主量为 30 min conditional log loss `B_state − B_observation`；E384 患者内三 seed 中位为 `+0.156047`，方向不支持 state increment。30 min 的 persistent−memoryless 为 `+0.187587`；严格 wrong-time 子集 n=2 的 correct−wrong-time 为 `+0.862894`。工程状态为 `COMPLETE`，科学 claim eligibility 为 `false`。

## 2. 状态与 checkpoint inventory

状态源只包含连续背景与 IED timing/mark 任务。使用 R1.6 `epilepsiae_384` 稳定 seeds 1、3、4：

| seed | checkpoint SHA256 |
|---:|---|
| 1 | `4113acf91e71736a1f5e9ea64c78389f3c0869877aa9c9f3bec71794620cecb6` |
| 3 | `45eb3b8fe9ac81fcd31d9aa648923da26c6be58e00b78b3f779aa71e2c5b0069` |
| 4 | `9eb27ed9a3d563a7911e053b90440a18f61786dc0844d371618a79a5aa05fdc8` |

精确 result/checkpoint 路径来自 R1.6 machine audit，不用 glob 推断。三份 result 与 checkpoint hash 均复算一致；`formal_test_partition_opened=false`、`sealed_opened=false`。完整清单位于 `manifests/state_checkpoint_inventory.json`。

## 3. Phase 0：crosswalk、分母和排除

发作真值来自 Epilepsiae SQL。E384 的 15/15 发作唯一匹配，onset delta 全为 0 秒。development partition 含 5 次发作。

| lead (min) | 完整记录覆盖 | 新鲜 inference observation | 最终 state 可分析 | evidence support |
|---:|---:|---:|---:|---|
| 5 | 5 | 5 | 5 | sensitivity，报告时仍服从 30-min descriptive tier |
| 15 | 5 | 5 | 5 | sensitivity，报告时仍服从 30-min descriptive tier |
| 30 | 4 | 4 | 4 | descriptive case series，primary |
| 60 | 2 | 2 | 2 | descriptive |
| 120 | 1 | 0 | 0 | not estimable |

旧 training-guarded anchor 支持为 `2/2/2/2/0`，不用于 H2b 推理分母。Inference gate 从 hash-verified `train_stats.json` 读取 `minute_min_valid_contact_fraction=0.70`；E384 有 2,509 个 inference-usable minutes、3,871 个 causal anchors，其中 606 分钟因 seizure-label training guard 曾被排除，但满足 coverage/cache/artifact 合同，可在冻结后推理。inventory anchor time 与 unique coverage-row segment 已与 state reader 逐元素及数组 hash 对齐。

输出：

- `manifests/seizure_crosswalk.csv`
- `manifests/seizure_support_by_lead.csv`
- `manifests/exclusion_funnel.json`
- `manifests/state_checkpoint_inventory.json`

## 4. Phase 1：因果状态提取

每个 query 仅消费时间不晚于 anchor 的 observation。状态在每个 unique `coverage_segment_index` 起点从冻结初态重置；`coverage.session` 仅供 deterministic IED history 连续性使用，不允许跨 gap carry state。绝对时间、anchor time、donor time 均保存为 float64。

推理 observation 不使用带 seizure label 的 `guard_free` 或 `minute_usable`，但继续要求：记录覆盖、合法 session、raw cache 存在、有效接触比例 ≥0.70、IED causal inpainting 至少保留两个 anchor 前背景点。当前 observation age 硬门为 ≤30 s；30 min 四次主发作的 age 为 `10.383/2.875/4.626/1.991 s`。

每个 cache 同时保存：persistent state、memoryless observation code、当前 explicit observation 的 masked mean/SD、38 维 deterministic history、current contact mask 和 wrong-time donors。38 维 history 由 11 个基础变量加 9 个 IED mark contacts 的 previous participation、previous group rank 和 2 min participation trace 构成；字段名写入 manifest，禁止裁剪或补零。

wrong-time donor 在同患者、同 unique coverage row 内按 time-of-day、session position、recent IED/history 与 observation confounder 的标准化距离软排序，并显式避开发作/ictal/postictal 窗。30 min 主风险分析不以 wrong-time 可用性删分母；wrong-time 直接比较单独使用严格共享子集。

## 5. Phase 2：risk-set 和 probe

每个 seizure×lead×seed 一个 risk set，1 个 case 加 5 个 controls。controls 与 case 都必须有 ≤30 s 的当前 observation、位于同一 coverage segment、在相应 horizon 内无发作、不在 ictal/postictal 窗，且 horizon 不越过 segment end。同一患者内所有 arm 共享完全相同的 risk sets；optimizer seeds 使用相同 control anchor IDs。

Probe 是 intercept-free ridge conditional logistic regression。主表 arms：

1. `B_history`：38 维严格因果 IED history；
2. `B_observation`：history + 当前 explicit observation；
3. `B_state`：observation + 8 维 frozen persistent state；
4. `memoryless`：observation + 8 维当前窗 code。

`wrong_time` 在严格子表中加入。E384 30 min 只有 4 次发作，因此使用 descriptive leave-one-seizure-out 和预设最强 ridge，不作模型选择或 held-out cohort inference。5/15 min 完整纳入第 5 次 sensitivity-only 发作，但证据层仍由预注册 30 min 分母锁为 descriptive；不能借短 lead 支持数升级。

### 5.1 全部分母结果

下表为先在 E384 内对 seeds 1、3、4 取中位后的 conditional log loss；差值为负才支持前者。

| lead | seizures | history | observation | state | memoryless | state−observation | persistent−memoryless |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 5 | 1.448795 | 1.190269 | 1.139475 | 1.122864 | −0.050795 | +0.019750 |
| 15 | 5 | 1.632344 | 1.573450 | 1.500707 | 1.554550 | −0.072744 | −0.061791 |
| 30 | 4 | 1.619096 | 1.607637 | 1.763684 | 1.560423 | **+0.156047** | **+0.187587** |
| 60 | 2 | 1.310918 | 1.056543 | 1.111100 | 1.091706 | +0.054557 | +0.019394 |
| 120 | 0 | — | — | — | — | — | — |

5/15 min 的 state−observation 为负，但 30 min 主端点和 60 min 为正；5 min persistent 也不胜 memoryless。按冻结合同不能选 15 min 包装阳性。30 min `observation−history=−0.011459`，不足以在 n=4 上宣称稳定即时 signal；`state−history=+0.144588`。

### 5.2 wrong-time 严格子集

| lead | seizures | state−observation | persistent−memoryless | correct−wrong-time |
|---:|---:|---:|---:|---:|
| 5 | 4 | −0.063123 | +0.039384 | −0.019267 |
| 15 | 3 | −0.104960 | −0.244178 | −0.005811 |
| 30 | 2 | +0.762326 | +0.786288 | **+0.862894** |
| 60 | 2 | +0.054557 | +0.019394 | −0.027611 |

30 min 只有 2 次满足严格 donor 合同，不能独立裁决时间特异性。主表与子表的 normalized risk-set hash 均在 producer/CSV consumer 间一致。

### 5.3 真实 E384 time-label diagnostic

主表 30 min observed state−observation 为 `+0.156047`；100 次患者内 risk-set case-label permutation 的 null median `+0.004023`，2.5%–97.5% 为 `[−0.082749,+0.082241]`。该诊断说明真实主效应不是“隐藏的负增量”，但 E384 n=4、置换离散，不能当队列 p 值。

## 6. Phase 3：phenotype transfer

真实人体 leg 为 `NOT_ESTIMABLE_NO_USABLE_FROZEN_TARGET`：

- 无预先冻结的 E384 seizure subtype；
- 盲法 onset-contact registry 为 0/71；
- ictal cache 只覆盖 4 次 primary 中的 2 次；
- cache 是逐通道时间序列/AUC 数组，不是预先冻结的 seizure-level recruitment extent。

本轮没有根据 state 重新聚类或定义阈值，也没有用 SOZ、focus、模板端点或最高能量触点替代。边界见 `reports/e384_phenotype_availability.json`。

## 7. 三个承重 instrument checks

| check | 结果 | 解释边界 |
|---|---|---|
| positive synthetic | state−observation `−1.100422` | probe 能恢复已知 frozen-state increment |
| synthetic time-label permutation | 100 次；null median `−0.000027`，区间跨 0 | label 对应打乱后增量回到零附近 |
| real causality perturbation | query 后 10 个窗极端替换；13 字段逐位不变 | anchor 后数据没有进入 anchor 前 state |

Phenotype synthetic 同时恢复 continuous extent `−4.742903` 和 frozen subtype `−0.917019`。这些只证明仪器按合同工作，不是人体 H2b 证据。

## 8. R1.7 接入状态

只读 watcher 最终看到 `machine_audit.json=COMPLETE` 和 50 个 fit result；50 个 result hash、审计列出的 45 个 checkpoint hash 以及各 result 的 source-hash payload 均复核一致，`formal=false`、`sealed=false`，当前 HEAD 也与 upstream 一致。但 R1.7 工作树仍有 14 条未提交路径，`code_committed=false`，因此 release 仍冻结为 `UNAVAILABLE_NOT_USED`。H2b 没有加载这些模型、把它们作为分析输入或导入 R1.7 代码，也未运行 T2/H3/physical clock。以后必须对全部 checkpoint-available 患者运行；H1-stable 只能作为预定义解释层，不能成为选择 gate。

## 9. 机器审计与复现

`reports/machine_audit.json` 为 `COMPLETE`、failed checks 为空，同时明确：

- `scientific_claim_eligible=false`；
- `r1_7_integration_status=UNAVAILABLE_NOT_USED`；
- `formal=false`、`sealed=false`；
- paper-ready figures 未修改；
- seeds 在患者内取中位，不作为患者重复。

H2b 定向测试为 `54 passed, 4 warnings`；warnings 均来自 PyTorch Transformer 的 nested-tensor 提示，不改变数值或合同判定。`src/`、`scripts/` 与 `tests/` 另通过 `compileall`。

核心复现命令：

```bash
cd /tmp/hfosp_h2b_transfer_20260828
export OMP_NUM_THREADS=1

/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python \
  scripts/topic5_continuous_marked_state_h2b/build_inventory.py

/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python \
  scripts/topic5_continuous_marked_state_h2b/run_e384_pilot.py \
  --overwrite --n-permutations 100

/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python \
  scripts/topic5_continuous_marked_state_h2b/build_phenotype_availability.py

/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python \
  scripts/topic5_continuous_marked_state_h2b/build_instrument_validation.py

/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python \
  scripts/topic5_continuous_marked_state_h2b/audit_h2b.py
```

## 10. 失败、OOM 与不可估计项

- 无 OOM，无 GPU H2b job；全部状态提取在 CPU、`OMP_NUM_THREADS=1` 下运行。
- 集成中发现并修复三项 fail-closed 问题：旧 training guard 被误用于推理；history 真实 38 维与集成层 11 维假设不符；CSV 重读把 seizure ID 猜成整数导致语义 hash 漂移。
- 另修复两个会缩分母的过严条件：逐位相同 contact-mask hash 不是 observation availability 合同；wrong-time donor 不得作为主风险表 gate。
- 5/15-min 最初只运行 30-min primary 子集，最终已加入第 5 次 sensitivity-only 发作并从头重建 cache。
- 120 min 不可估计；真实 phenotype 不可估计；R1.7 cohort 未接入。

## 11. 允许与禁止的结论

允许：冻结间期状态的跨任务仪器通过；E384 descriptive pilot 在 30 min 不显示 latent-state 增量；cohort H2b 尚未检验。

禁止：seizure-susceptibility mechanism identified、state causes seizures、latent attractor identified、IED shapes seizure transition、clinical seizure predictor、formal held-out confirmation。
