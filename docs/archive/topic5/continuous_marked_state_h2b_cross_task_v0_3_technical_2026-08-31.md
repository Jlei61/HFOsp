# Continuous Marked State H2b Cross-task Transfer v0.3：技术验收

## 1. Scientific claim and boundary

本轮检验的问题是：完全不使用 seizure-risk outcome 训练的 frozen interictal state，能否在患者内未见发作上提供超出 recent IED history 与 current observation 的预测信息，并进一步描述 seizure-entry geometry。

最终裁决为：

> **PASS_GATED_NEGATIVE_CLOSEOUT_H2B_NOT_ESTABLISHED.**

这是 instrument/estimability closeout，不是 shared interictal–ictal state 的生物学阴性。v0.3 是看过 v0.2 development 结果后的修复性重设计，不是 independent confirmation。formal、sealed、H3、T2、physical clock 与 paper-ready figures 全程关闭。

术语固定如下：

| 术语 | 本报告定义 |
|---|---|
| `all_frozen` | checkpoint 可读且完成 A1 的全部患者 |
| `state_qualified` | 至少 3 个 seeds 联合通过 Q1–Q6 的患者 |
| T | `M2 persistent state` 相对 `M1 current observation` 的 held-out log-loss increment |
| M | outer-training residualized persistent history 相对 memoryless state 的 increment |
| decoder metric | frozen timing/contact/size decoder outputs 的标准化空间 |
| diagnostic-only | 只验证实现或灵敏度，不释放真实 H2b claim |

## 2. v0.2 acceptance

v0.2 已独立收口：防泄漏、checkpoint freeze、risk-set identity 和 nested `B_state − B_observation` 实现通过；唯一 primary patient 的增量位于 permutation null 内，persistent、memoryless 和 correct-time 对照均未形成支持。其安全定位是工程合格的 negative pilot，不能写成 H2b 阳性或生物学阴性。

v0.3 不复制修复前旧 H2b 的 `+0.4582 SD、21/27、p=0.00592`，也不把 v0.2 的少量方向性结果作为先验 gate。

## 3. Isolated execution and contracts

- worktree：`/tmp/hfosp_h2b_v03_strict_20260831`；
- branch：`codex/topic5-state-h2b-v0-3-strict-acceptance`；
- result root：`results/epi_prssm/continuous_marked_state/h2b_cross_task/v0_3/`；
- machine contract：`config/topic5_continuous_marked_state_h2b_v0_3.json`；
- frozen receipt：`analysis_contract.json`；
- claim policy receipt：`exploration_policy.json`。

较早的 `exploration_policy.json` 与“少 gate、多探索”附录保留为历史 provenance，但本次指定验收意见的 §9.3 与停止规则优先。当前有效 policy 为 `config/topic5_continuous_marked_state_h2b_v0_3_strict_acceptance_policy.json`：A1 无足够 `state_qualified` 时停止 A3–A8，A2 power 不足时不解释真实阴性，orchestrator 不允许 diagnostic override。最终机器审计另写 `strict_acceptance_policy.json` receipt。

本工作树未写入 R1.7 工作树，也未提交或终止 R1.7 队列。所有 CPU workers 设置 `OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1` 和 `NUMEXPR_NUM_THREADS=1`。75-cell A1 队列按实测 RSS 预算使用最多 8 workers；最后 10 cells 与 1500 个 assay replicates 均为 8-worker detached runs，零 OOM。full-grid 基础设施另按大患者实测 RSS 动态限流，其资源审计见第 7 节。

## 4. A0 attrition and denominator

冻结 inventory 共 85 cells。attrition reason 为：

| 原因 | cells |
|---|---:|
| checkpoint unavailable: `NONFINITE_GRADIENT` | 10 |
| checkpoint readable but no frozen seizure support in v0.2 | 24 |
| no complete primary 30-min coverage | 5 |
| v0.2 state cache available | 46 |

A1 的分母独立于 seizure support：75 个可读 checkpoint cells、16 位患者全部进入状态资格审查。H1-stable 只作解释层，不作筛选 gate。

E1084 和 E583 的 R1.2 design/embedding 缺失后重建。消费者侧 verifier 对每位患者检查：

- `full_design.npz` SHA256 与 frozen R1.7B cache manifest 一致；
- `explicit_embedding.npy` 与自身 manifest 一致；
- R1.3 `explicit_normalised.npy` SHA256 与 frozen R1.7B 一致；
- timing/mark history baseline tensors 与 5 个 frozen seeds 的 checkpoint 位级一致；
- sealed 未打开。

两位患者全部检查通过后才加入最后 10-cell A1 queue。

## 5. A1 interictal-only state qualification

### 5.1 Q1 non-collapse

Q1 在 frozen decoder-output metric 上计算 active dimension、effective rank、top-PC share、within-segment temporal shuffle 和 reset-phase variance。严格规则要求 active decoder dimensions ≥2、effective rank ≥2、top-PC share ≤0.95；单一 decoder axis 只记作 scalar candidate。

### 5.2 Q2 cross-window information

observer 在 sparse cut 后关闭；在 5/15/30/60 min 非重叠 interictal windows 中比较 persistent 与 memoryless 对未来 IED timing/mark likelihood 的增量。记录的未来 IED history 仅以 teacher-forced nuisance 进入，报告中明确不是 fully autonomous rollout。

### 5.3 Q3 generator contribution

分别记录 generator 与 correction 在 latent 和 decoder metric 中的 step norm，并进行 artificial reset recovery。Q3 要求窗口间运动不能几乎全部由当前 observer correction 重写。

### 5.4 Q4 time constant

在 decoder-output distance 中估计 empirical `tau_z`，同时记录 generator analytic slowest mode。仅 analytic time constant 较长但 empirical decay 不可辨识，不构成通过。

### 5.5 Q5 seed stability

比较 decoder-distance correlation、linear CKA 和 Procrustes-aligned decoder trajectory，并以 within-session permutation 为 null。raw latent Procrustes 只作诊断，不进入资格判定。

### 5.6 Q6 not-only-clock

在同患者、同 segment 的 5 min 非重叠 interictal anchors 上，分别预测下一窗口 IED timing 与 first-event marks。base 包含 current memoryless decoder、近期 IED history、time of day、session position、segment 与严格过去发作 nuisance；increment 为 persistent-minus-memoryless decoder。chronological ridge 仅用过去 folds。无事件窗口保留，跨 gap 窗口排除。sleep/medication metadata 不可验证，作为限制记录，不伪造 covariate。

### 5.7 Patient-level result

完整结果为 75 cells、16 patients、0 state-qualified：

| patient | seeds | Q1 | Q2 | Q3 | Q4 | Q5 | Q6 | joint Q1–Q6 | stratum |
|---|---:|---:|---:|---:|---:|---|---:|---:|---|
| epilepsiae_1073 | 5 | 0 | 0 | 0 | 0 | fail | 0 | 0 | collapsed/unusable |
| epilepsiae_1077 | 1 | 1 | 1 | 1 | 1 | fail | 0 | 0 | insufficient seeds |
| epilepsiae_1084 | 5 | 0 | 0 | 0 | 5 | pass | 0 | 0 | collapsed/unusable |
| epilepsiae_1125 | 5 | 0 | 5 | 4 | 5 | pass | 5 | 0 | scalar slow axis |
| epilepsiae_1146 | 5 | 1 | 5 | 0 | 5 | pass | 4 | 0 | scalar slow axis |
| epilepsiae_1150 | 5 | 0 | 5 | 5 | 5 | pass | 2 | 0 | scalar slow axis |
| epilepsiae_253 | 5 | 1 | 4 | 0 | 5 | pass | 0 | 0 | scalar slow axis |
| epilepsiae_442 | 5 | 0 | 5 | 0 | 5 | pass | 0 | 0 | scalar slow axis |
| epilepsiae_548 | 5 | 3 | 1 | 0 | 5 | pass | 0 | 0 | collapsed/unusable |
| epilepsiae_583 | 5 | 0 | 5 | 5 | 5 | pass | 0 | 0 | scalar slow axis |
| epilepsiae_635 | 5 | 0 | 1 | 3 | 5 | pass | 0 | 0 | collapsed/unusable |
| yuquan_liyouran | 5 | 0 | 5 | 0 | 5 | pass | 0 | 0 | scalar slow axis |
| yuquan_wangyiyang | 5 | 0 | 5 | 0 | 5 | pass | 0 | 0 | scalar slow axis |
| yuquan_xuxinyi | 5 | 0 | 5 | 0 | 5 | pass | 0 | 0 | scalar slow axis |
| yuquan_zhangbichen | 5 | 3 | 5 | 1 | 5 | pass | 0 | 0 | scalar slow axis |
| yuquan_zhaochenxi | 4 | 0 | 1 | 2 | 4 | pass | 0 | 0 | collapsed/unusable |

其中 Q1–Q4/Q6 数字是通过该项的 seed 数，Q5 是患者级 seed-stability 判定。任何患者都没有 3 个 seeds 联合通过全部承重项。

## 6. A2 semi-synthetic assay audit

### 6.1 Rejected implementation

首版 smoke 被作废并保留在 `assay/superseded_raw_latent_v1/`。P0 包括：

1. T 和 M 都读取同一 `relative_logloss_improvement`；
2. geometry 使用 raw latent，而不是 frozen decoder metric；
3. null threshold 与 null false-positive 在同一批 replicates 上计算；
4. 没有独立 lag-degradation estimand；
5. 在 null false-positive 约 20% 时仍 fallback 选择 K=5。

### 6.2 Corrected diagnostic assay

修复版 template 来自 E1125 seed 0 的 interictal D_state decoder trace，按同 segment 5 min grid 下采样。current observation 用纯间期 PCA，persistent/memoryless 使用同一 frozen decoder scaling。M 的 persistent-history residual 仅在 outer-training fold 内拟合。synthetic onset 保留真实 coverage、时钟支持、固定 seizure count、状态自相关和不跨 gap 的 30 min lead。

K∈{2,3,4,5} 使用偶数-seed null calibration 建立 T/M 95th-percentile thresholds，奇数-seed null 独立估计 false-positive；再按 joint `T+M+lag` power 选择 K。100-replicate smoke 选择 K=2。

| world | T detection | M detection | joint T+M+lag | geometry recovery |
|---|---:|---:|---:|---:|
| null | 0.05 | 0.06 | 0.02 | N/A |
| observation-only | 0.05 | 0.07 | 0.02 | N/A |
| persistent-state | 0.04 | 0.04 | 0.01 in main batch; K-selection joint power 0/100 | N/A |
| clock-confounded | 0.040 | 0.051 | 0.00 | N/A |
| basin gating | 0.05 | 0.07 | 0.02 | 0.99 |
| directed approach | 0.03 | 0.03 | 0.01 | 1.00 |
| abrupt transition | 0.11 | 0.08 | 0.05 | 0.99 |

该 smoke 证明 geometry statistic 能识别显式注入的三类模式，但 transfer instrument 对 persistent world 无 power。由于 A1 为空且 A2 smoke 已失败，没有运行 1000-replicate final acceptance；`type1_power_summary.json` 不存在，claim route 未释放。

## 7. A3–A8 route decision

严格 queue receipt 为：

```text
status = NOT_RELEASED_A1_OR_A2
n_state_qualified_patients = 0
final_assay_available = false
tasks_started = 0
```

因此：

- A3 nested full-grid hazard：未运行主路线；
- A4 prequential OOF：未运行主路线；
- A5 `tau_z` lag-response：未运行主路线；
- A6 OOS manifold/flow：未运行；
- A7 shuffled IED-objective ablation：未触发；
- A8 frozen phenotype bridge：未触发。

另行构建的 full-grid cache 只是一层不读取 seizure table 的状态提取基础设施。46/46 cells、10 位患者均完成，共 10,597 个唯一 5-min anchors、跨 seed 45,841 行。其患者集合仍来自预冻结的 v0.2 seizure-support inventory，因此正确表述是“在既有支持人群内，anchor 构造不依赖 outcome”，不能称为全队列 outcome-independent selection。它不释放 A3–A8，也不进入 H2b 结果。

实现层还要再区分两部分：full-grid query anchors、observer/generator 更新、deterministic IED history 和 persistent state 值均不读取 seizure outcome；`global_exclusions.csv` 只在这些状态算完后进入 `build_wrong_time_candidates()`，用于排除 ictal/postictal donor。因此 serialized cache 内的 wrong-time donor 索引接触了 seizure exclusion metadata，不能把整个 cache 文件称为完全 outcome-blind。该 donor 层未进入本次任何 probe 或科学结论。

历史与提前生成的 support-conditioned 产物位于：

- `quarantine/pre_gate_hazard_v1/`；
- `quarantine/support_conditioned_hazard_v2/`。

另一个提前生成的 E1125 seed 0 geometry cell 位于：

- `quarantine/support_conditioned_geometry_v1/`。

这些产物显式标记为 support-conditioned exploration，使用 v0.2 seizure-support query grid，不是完整 recorded-coverage grid。它们不进入活跃 `hazard/by_cell`、`geometry/by_cell`、patient-first inference、报告结论或 machine audit 的科学证据层。

在 full-grid 完成后，并发旧编排器又三次带 diagnostic override 运行 A3/A6/A8。对应 hazard、geometry、phenotype 各三套，共 9 个目录，分别以 `post_gate_*_exploratory_v1`、`v2` 和 `v3` 保存于 `quarantine/`。当前活跃结果树只保留严格 `hazard/QUEUE_STATUS.json`、`full_grid/FOLLOWUP_STATUS.json` 和 `geometry/ROUTE_STATUS.json`，均记录 A1/A2 gate 未释放；这 9 套结果不被 closeout 读取为科学证据。由于另一个共享任务在文件锁释放后会立即重跑旧编排器，交付时保留一个只持有 `.followup.lock` 的安全守卫；它不运行模型、不占 GPU。

full-grid state queue 为 revision v5：按 `4 GiB + 4 MiB × max_query_rows` 为每个 worker 估算，使用可用内存的 65% 上限，从配置的 8 workers 中选择 8 个；`OMP/MKL/OPENBLAS/NUMEXPR=1`。36 份 GNU-time RSS receipt 的最大峰值为 1.48 GiB；31 个本轮任务与 15 个既有任务合计 46/46，失败、retry 和 kernel OOM 均为 0。更早的非流式测量因单个大患者超过 50 GiB 被主动终止，未等到 kernel OOM；这正是改为流式缓存后再放满 worker 的依据。

## 8. Core implementation corrections

本轮确认并修复：

1. A1 queue 分母从 v0.2 seizure-support census 改为全部 readable checkpoints；
2. E1084/E583 upstream rebuild 必须有 v0.3 consumer-side equivalence receipt；
3. Q1 恢复多维 state gate，拒绝单一 decoder axis 冒充 persistent state；
4. Q6 使用非重叠 interictal 5 min windows，保留 no-event windows，避免 event-as-anchor 假 next-IED；
5. Q6 training fold 要求样本数超过 feature width，减少早期 fold 数值伪增量；
6. A1 aggregate 只有在完整 75-cell 分母时才释放 population manifests；
7. A2 T/M 分离，M residualization 仅在 outer training，geometry 改用 decoder metric；
8. A2 null calibration/evaluation 分离；
9. A3 queue 默认 fail closed，并拒绝 v0.2 query cache 冒充 full recorded grid；
10. 所有越过 gate 的旧结果进入 quarantine。
11. hazard 的 M4 改为在每个 outer-training fold 内估计 `ZP | ZM,O,C,H`，再用 held-out residual 比较 M4 与 M3；不再用与 T 几乎同构的简单 `persistent−memoryless` 特征。
12. prequential TRAIN/TEST rows 只有在各自 cutoff 时 horizon label 已可观测才进入评分，未来发作不再借由全时间轴 outcome 提前进入训练标签。
13. OOS geometry 的 PCA、basin 与 reference manifold 只在 previous-seizure cutoff 前、且距训练发作至少 120 min 的 clean interictal full-grid rows 上拟合；held-out trajectory 仅作投影。
14. full-grid follow-up 默认检查 A1/A2，当前必须返回 `NOT_RELEASED_A1_OR_A2, downstream_tasks=0`；diagnostic override 不再由编排器自动打开。
15. continuous phenotype bridge 同样默认 fail closed；旧 `seizure_idx` target mapping 只保留为未运行 scaffold，未作为通过 0 秒 onset crosswalk 的科学产物。

仍未把 5/15/60 min 统一为同一个 fitted hazard 导出的 horizon；当前代码把它们明确标成 separate development refits。由于 A1/A2 已停止科学路线，这不影响本次 gated closeout，但它仍是未来重新释放 A3 前必须补齐的 P1，而不是已完成项。

## 9. Reproducibility

关键命令：

```bash
python scripts/topic5_continuous_marked_state_h2b/verify_v03_upstream_rebuild.py \
  --subjects epilepsiae_1084 epilepsiae_583 \
  --result-root /home/honglab/leijiaxin/HFOsp/results/epi_prssm/continuous_marked_state/h2b_cross_task/v0_3

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
python scripts/topic5_continuous_marked_state_h2b/run_v03_instrument_queue.py \
  --cpu-workers 8 --n-null-permutations 100

python scripts/topic5_continuous_marked_state_h2b/aggregate_v03_instrument_smoke.py \
  --subjects <all-16-readable-subjects> --n-permutations 100

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
python scripts/topic5_continuous_marked_state_h2b/run_v03_assay_smoke.py \
  --replicates 100 --cpu-workers 8 --allow-unqualified-diagnostic

python scripts/topic5_continuous_marked_state_h2b/run_v03_full_grid_followup.py \
  --result-root /home/honglab/leijiaxin/HFOsp/results/epi_prssm/continuous_marked_state/h2b_cross_task/v0_3 \
  --expected-cells 46 --poll-seconds 1

python -m pytest -q tests/topic5_continuous_marked_state_h2b

python scripts/topic5_continuous_marked_state_h2b/build_v03_closeout.py \
  --test-log /home/honglab/leijiaxin/HFOsp/results/epi_prssm/continuous_marked_state/h2b_cross_task/v0_3/logs/pytest_topic5_h2b_v03_strict.log
```

最后一条严格 hazard queue 应返回 `NOT_RELEASED_A1_OR_A2 tasks=0`，不是 `COMPLETE` cells。

完整 source、checkpoint 和逐 cell SHA256 在 `reports/machine_audit.json`。全 scoped test 为 168 passed、5 warnings；warnings 均为 PyTorch Transformer nested-tensor 提示。

## 10. Final interpretation

当前 R1.7B checkpoint 中存在若干稳定 scalar/filter/generator 线索，但没有患者形成跨至少 3 seeds 的多维、跨窗口、nuisance-robust state instrument。修复后的 assay 又无法在当前 seizure support 下恢复 persistent-state transfer。因而最强安全结论是：

> 当前 checkpoint 的 H2b transfer utility 不可估计，H2b 未建立；下一步应先进行纯间期 slow/fast state redesign，而不是增加 seizure heads。

这不排除 shared interictal–ictal state，也不支持 attractor、机制、因果转换或临床预测措辞。
