# Topic 5 RNN motif / cross-state / theory v0.4 — Execution Plan

状态：**EXECUTING AGAINST LOCKED SPEC 2026-08-09**  
对应 spec：`docs/superpowers/specs/2026-08-08-topic5-rnn-motif-cross-state-theory-v0-4-design.md`

---

## 0. 执行原则

这不是继续扩 architecture zoo，也不是用 early-ictal target 调 RNN。
执行顺序固定为：

1. 冻结间期任务与 motif 矩阵；
2. 完成全部间期训练和自由生成；
3. 冻结模型生成场；
4. 再统一读取 early-ictal target；
5. 最后做有效影响和 matched lesion。

科学结果可以阴性；只有工程完整性问题会阻止下游。

---

## Milestone A｜合同、inventory 与复用审计

### A1. 新建运行配置

新增：

- `config/topic5_rnn_motif_cross_state_v0_4.yaml`
- `results/topic5_rnn_motif_cross_state_benchmark_v0_4/RUN_CONTRACT.json`
- `.../INPUT_MANIFEST.json`

配置写死：cohort、31 fits、splits、planes、H、model matrix、seeds、eta、训练参数、target metadata 路径、统计量。
同时写 `GEOMETRY_STATUS=RETROSPECTIVE_TEST_INFORMED_PROPAGATION_PLANE`；不得把 held-out 结果解释成
train-only axis discovery。

### A2. 只读 inventory

核对：

- 21 位 interictal patients / 31 fits；
- masked rank dataset 与 event-source index；
- 11 shared / 10 non-collinear；
- expected 15 primary early-ictal patients + E1146 supportive；
- exact contact-name join 的预计分母；
- target 文件 hash，但不反序列化数值数组。

输出：`PREFLIGHT_INVENTORY.json`。

### A3. v0.3 checkpoint reuse matrix

逐单元比较完整 hash/config。输出：

- `reuse_manifest.csv`；
- `reused` / `must_retrain` / `missing_seed`；
- 禁止仅凭目录名复用。

### A4. 建立干净、不可变的 execution worktree

当前 `scripts/plot_topic5_we_figures.py` 有既存未提交修改，不能在当前 worktree 正式运行。

1. spec/plan 锁定后，只提交本轮合同，不夹带该 figure 修改；
2. 从包含锁定合同的明确 base commit 新建 branch/worktree
   `codex/topic5-rnn-motif-cross-state-v0-4`；
3. 确认新 worktree `git status --porcelain` 为空；
4. 实现验收后，将 launcher、model、trainer、decoder、scorer、field builder 复制到
   `results/.../run_snapshot/`；
5. active run 只执行 snapshot，运行中禁止编辑；修改必须开新 run revision；
6. 每个 stage 保存 producer code/config/input-manifest hash 与 created time；
7. aggregate 前核对 freshness、cohort revision 和全部 producer hashes。

验收：preflight、cohort、hash、target metadata schema 全通过。

---

## Milestone B｜模型适配与最小单元测试

### B1. 扩展 arm schema

在不破坏 v0.3 接口的前提下实现：

- `M0_NO_REC`
- `M1_DENSE`
- `M2_UNIFORM_SET`
- `M3_FIXED_LOCAL`
- `M4_SPATIAL_GROWTH`
- `M5_SPATIAL_LOW`
- `M6_SPATIAL_MID`
- `M7_SPATIAL_HIGH`
- `M8_UNIFORM_COST_MID`
- `C_ORDER_SHUFFLED`
- `C_FULL_RANK_SHUFFLED`（一 seed sensitivity）

优先新建 v0.4 adapter/module，不直接重命名 v0.3 历史 arm。

### B2. fixed-local mask

测试：

- edge count 精确等于 10% resource；
- 无 self-loop；
- bidirectional Euclidean MST 已包含；
- min in/out degree ≥1；
- weak/strong component 均为 1；
- H-supported nodes 全在 main component；
- out-degree 最大差 ≤1；
- seed 只改变 weights，不改变 deterministic mask；
- mask 训练全程不变。

### B3. eta 与 wiring cost

测试 `0/0.01/0.03/0.10` 只改变指定损失项，其他 config 完全相同。M2/M4/M8/M6 的 pruning
必须逐 epoch 同构；只允许 regrowth proposal 与 eta 不同。GRU wiring magnitude 使用 gate-RMS，
并保存 init/mid/freeze/final 的 `eta*C_wire/L_task`。

### B4. common scoring contract

新增 contact-choice / STOP / length decomposition，并确认所有 models 使用相同 decisions、eligible masks 和 event denominator。

实现 `ROLLOUT_DECODER_CONTRACT.json`：冻结 recurrent checkpoint 后，用 train states 拟合共同 `4→16→C_p`
size head、validation early stopping；test rollout 固定 STOP-first、`p_stop=0.5`、predicted top-K、recruited mask、
deterministic ties、最大 `C_p` steps。回归测试必须证明 decoder 不读取 observed next-set size。

### B5. field builder tests

toy events 验证：

- `FIELD_CANONICAL_FULL` 包含 seed，严格复用 human/SNN field builder；
- `FIELD_SEED_REMOVED` 排除 seed；
- frequent-source contact 使用“非 seed events 数”作为显式分母，不被自动补零；
- absent contact 得分为 0；
- A/B labels 只在 post-hoc grouping 读取；
- common/contrast 公式；
- Q1 non-collinear 两 fits 先患者内平均；
- Q2 own_a/own_b 保留为 A/B candidates，直到 per-seizure maxAB 后才聚合；
- order shuffle 保留 event participation 与 split；
- 所有模型使用逐位相同的 common evaluation support；模型自己的 participation support 不能改变 primary eligibility。

### B6. effective-influence tests

小网络上比较 lag-1 probability Jacobian 与 finite pulse；pulse tissue-input norm 必须跨 contacts 相同。
lag-2/3 必须 open-loop、不输入真实未来、不穿过 argmax 反传。hidden Jacobian 跨 cell family 比较应报错。

验收：新增测试全过，既有 Topic 5 RNN 核心测试不回退。

---

## Milestone C｜小规模 smoke 与资源 benchmark

### C1. 三 fit smoke

固定使用：

- 一个 shared fit；
- 一个 own_a fit；
- 一个 thin fit。

每个只跑短预算，覆盖 leaky RNN 全 arm 与 GRU 五 core arms。

检查：shape、gradient、mask update/freeze、四个 snapshots、size-head decoder、multi-contact rollout、
convergence log、OOM、NaN、atomic DONE。M0 运行三 seed；只有训练轨迹、参数和输出逐位一致才标
`DETERMINISTIC_REUSE_ALLOWED`。

### C2. 并发 benchmark

从 6 workers 起，逐档测试 10 / 14 / 18；记录 GPU memory、CPU RAM、吞吐和 I/O。
选择吞吐不再增长前一档，保留至少 15% 显存余量。

不修改 batch size，不混用 CPU/GPU 训练结果。

### C3. runner

提供可断点续跑脚本：

`scripts/run_topic5_rnn_motif_cross_state_pipeline_v0_4.sh <stage> <workers>`

要求：nohup/tmux 友好、已有 DONE 跳过、失败写 `FAILED.json`、每 stage 独立 log。

---

## Milestone D｜正式 interictal benchmark

### D-Core. 全 cohort 核心模型

先完成 `M0/M1/M2/M3/M4/M6/M8/C_ORDER_SHUFFLED`，全部 seeds `0/1/2`：

```text
31 fits × 8 arms × 3 seeds = 744 units
```

这保证核心 2×2 factorial、边界参照和顺序对照最先完整。

### D-Dose. 空间成本剂量与 full-shuffle sensitivity

再完成：

```text
M5/M7: 31 fits × 2 arms × 3 seeds = 186 units
C_FULL_RANK_SHUFFLED: 31 fits × 1 seed = 31 units
```

M5/M7 是 dose sidecars，不改变 core factorial。

### D-Architecture. GRU replication

最后完成 `M0/M1/M2/M3/M6`，全部三 seeds：

```text
31 fits × 5 arms × 3 seeds = 465 units
```

所有三个 D stage 都必须在 target unseal 前完成，或被明确标为不可恢复的工程失败。合法 v0.3 final checkpoint
可以复用，但新的 size decoder 仍须拟合。

### D3. convergence audit

每批完成后自动汇总：

- complete / failed / OOM / NaN / non-converged；
- epochs、best epoch、最后 5 个 validation slopes；
- mask edge count；
- checkpoint/config hash。

所有新训练单元保存 `INIT/REWIRE_MID/MASK_FREEZE/FINAL`。复用 checkpoint 若缺少中间 snapshots，
不得事后伪造；training-trajectory primary 至少对 M6 true-order 与 C_ORDER_SHUFFLED 重新训练并保存四点，
其他复用单元标 `snapshot_missing_reused_checkpoint`。

non-converged 单元可按原上限从 checkpoint 继续，不能换学习率追结果。

### D4. Q1 scoring

输出：

- `interictal_per_event.csv`
- `interictal_per_fit_seed.csv`
- `interictal_per_patient.csv`
- `interictal_bootstrap.json`
- `task_adequacy_tiers.json`
- `accuracy_wiring_pareto.csv`
- `factorial_effects_interictal.json`
- `ROLLOUT_DECODER_CONTRACT.json`
- `FIT_TO_PATIENT_AGGREGATION_CONTRACT.json`

图：

- rollout raw patient points；
- recurrence/order gain waterfall；
- contact-choice vs STOP decomposition；
- dense-benefit retention；
- accuracy–wiring Pareto；
- leaky RNN vs GRU replication。

同时冻结 `minimum_dense_benefit=0.01` 和由 8 个 development fits 得到的 `delta_NI`；
`ADEQUATE_STRONG` 用 raw recurrence gain、rollout non-collapse 和 NLL non-inferiority，不由不稳定 retention ratio 单独决定。
对 M2/M4/M8/M6 计算 growth-at-zero、growth-at-mid、cost-uniform、cost-spatial 和 interaction 五项，
NLL 先转为 `-NLL` 统一“越大越好”的方向。

注意：这一步不读取 early-ictal target。

---

## Milestone E｜target-free model field freeze

### E1. 生成 held-out event rollouts

每 model×fit×seed 使用完全相同的 held-out events 与真实 seed rank set，保存：

- generated rank sets；
- STOP step；
- `FIELD_CANONICAL_FULL` contact scores；
- `FIELD_SEED_REMOVED` contact scores 与 per-contact non-seed denominator；
- A/B label provenance。

### E2. 聚合 model fields

两个 field endpoint 分别生成：

- `F_A`
- `F_B`
- `F_common`
- `F_contrast`
- generated participation support

并输出 empirical interictal fidelity、mode-collapse、seed stability、split-half stability。
primary evaluation support 从 exact-joined contact geometry 与 frozen empirical support 构造，所有模型逐位相同；
generated participation support 只作诊断。

对 non-collinear 患者保留 own_a→A、own_b→B；禁止在 per-seizure maxAB 前平均。shared 患者从同一 fit
的 post-hoc A/B event groups 建场。

### E3. shared-mode 特别审计

仅 n=11：

- 同一模型 TA/TB 生成相关；
- matched vs swapped template fidelity；
- A/B contrast retention；
- 不把两个起点造成的差异写成模型自发发现两类。

### E4. field manifest 与封存

写：

- `MODEL_FIELD_MANIFEST.json`
- `FIT_TO_PATIENT_AGGREGATION_CONTRACT.json`
- `PRIMARY_THEORY_SET.json`
- `MOTIF_DEFINITION.json`
- `TARGET_UNSEAL_AUTHORIZATION.json`

manifest 记录所有 field/contact/grid/hash/representative patient。写完后模型和 field builder 只读。

验收：所有工程有效模型均有完整 fields；early-ictal value read count 仍为 0。

---

## Milestone F｜early-ictal external benchmark

### F1. 统一 unseal

只允许一个 scorer 进程读取 target；先验证 manifest 时间戳和 hash，再解封。

### F2. R3 scoring

对 empirical reference 和每个 model 使用同一：

- event inventory；
- contact join；
- grid / sigma / support logic；
- mirror / maxAB；
- 5,000 all-contact permutations；
- patient-first aggregation。

同一 permutation mapping 跨模型共享。

### F3. common / contrast / seed sensitivity

同时输出：

- `FIELD_CANONICAL_FULL` primary maxAB；
- `FIELD_SEED_REMOVED` recurrence-specific key-secondary maxAB；
- common-field concordance；
- contrast fidelity；
- canonical-full vs seed-removed；
- within-shaft sensitivity denominator。

### F4. 统计与多模型比较

输出：

- `early_ictal_per_seizure.csv`
- `early_ictal_per_patient_model.csv`
- `early_ictal_model_contrasts.json`
- `factorial_effects_early_ictal.json`
- `early_ictal_conditional_on_interictal_fidelity.json`
- `early_ictal_null_matrices.npz`
- `target_access_audit.json`

严格区分：每模型超 null、模型间直接差、empirical reference gap。

除 raw paired contrasts 外，运行 patient-intercept model
`early margin ~ patient + interictal empirical-field fidelity + model`，用 10,000 patient-cluster bootstrap
和 patient-level permutation；不能因为某模型间期拟合更好就直接解释为跨状态 inductive bias。

读取 target 后不得重新训练或改图中代表患者。

---

## Milestone G｜effective computation 与理论分析

### G1. 读取预冻结 primary theory set

primary 固定为 M1/M2/M3/M4/M6/M8/C_ORDER_SHUFFLED，不由 early target 选择。
三轴 Pareto 只决定 M5/M7 是否增加 `TARGET_INFORMED_EXPLORATORY_THEORY`，不能改 primary。

### G2. effective influence

计算 lag-1 teacher-forced probability Jacobian，以及 lag-1/2/3 open-loop finite pulse probability response。
pulse 使用 patient train-input median norm 标准化，后续不输入真实 ranks。按：

- distance；
- same/cross shaft；
- A/B axis-aligned vs transverse；
- local vs long-range；
- seed/split stability

汇总。

### G3. primary motif summaries

主分析只保留 effective reach、local-backbone/long-range connector organization、matched lesion specificity。
谱半径、non-normality、path diversity 与 hidden modes 放 Supplementary，且不跨 cell family 直接比较。

### G4. candidate motif tests

测试“local backbone + sparse high-influence connector”是否：

- 超 matched geometry/proposal null；
- 与 patient propagation fidelity 相关；
- 与 wiring efficiency 相关；
- 跨 cell family 方向一致。

### G5. matched lesions

每患者每模型目标 500 matched draws，最低 200。删除 high-influence、connector、local-backbone 或 connector nodes；
按 spec 固定 calipers 匹配 edge/node count、weight、degree、length 与 spatial extent。低于 200 不做患者级 inference，
不得事后放宽 caliper。

评分只使用 frozen model 和 frozen held-out events。输出 lesion 对 NLL、rollout、length、A/B/common field 的影响；
lesion field 进入 frozen early scorer 只作 secondary `intact-lesioned` readout。

### G6. 结论审计

只有“结构富集 + 任务关系 + matched lesion”三者同向，才写“该 motif 更容易支持传播”。
否则降为 descriptive association。

---

## Milestone H｜Human–RNN–SNN common-observable table

### H1. 只读既有 SNN 产物

不重跑 SNN gate，不恢复 SNN 参数。只提取已有：

- opposite-source bidirectionality；
- spatial reach；
- interictal event field；
- early seizure-like recruitment field；
- perturbation readout。

### H2. 同层级比较

构建 `COMMON_OBSERVABLES.json/csv`，并明确 missing / not comparable 字段。
Human–RNN–SNN 的 field 行只使用 `FIELD_CANONICAL_FULL`；`FIELD_SEED_REMOVED` 作为 RNN recurrence-specific sidecar。
禁止 edge-to-edge 或 hidden-unit-to-neuron 映射。

---

## Milestone I｜Figure 6、报告与验收

### I1. Figure 6

六块：

- A connectivity motif ladder；
- B observed vs generated A/B example；
- C cohort interictal sufficiency；
- D fidelity–wiring Pareto；
- E canonical-full primary + seed-removed secondary ↔ early-ictal external benchmark；
- F effective motif + matched lesion。

每个统计 panel 使用 patient raw points；legend 尽量独立；字号按 `docs/figure_style_guide.md`；
不得靠大段文字补科学含义。

### I2. source data 与 README

每个 panel 都有 source CSV/JSON；`figures/README.md` 用中文逐图写“展示什么/关注什么”。

### I3. 最终报告

按三问写：

1. 哪些 motif 能学习间期传播；
2. 哪些冻结 field 复现 early-ictal 对应；
3. 哪些有效计算结构经 matched lesion 支持。

同时列出允许/禁止 wording、denominator、target history 和 cell-specific 边界。

### I4. 最终工程验收

- unit tests；
- relevant integration tests；
- full manifest/hash audit；
- target access audit；
- 0 silent failure；
- 图逐张目视检查；
- 干净 execution worktree，run_snapshot 与 producer hashes 全部一致；
- 分逻辑 commit，不自动 push。

---

## 预计资源与恢复策略

- 最坏约 1,426 training units（D-Core 744 + D-Dose 186 + full-shuffle 31 + GRU 465），
  实际会因 v0.3 合法 final-checkpoint 复用而减少；
- 建议并发以实测饱和点为准，预期 12–14 workers，而不是盲目开满；
- 单元级 checkpoint + `DONE.json`，网络波动或会话断开后从缺失单元续跑；
- 每 5 分钟 watcher 汇总 complete/failed/OOM/NaN/GPU memory；
- watcher 只监控和推进预定义 stage，不修改科学配置。

---

## 最短执行顺序

1. A：inventory + reuse audit；
2. B/C：实现、测试、smoke、资源 benchmark；
3. D-Core → D-Dose → D-Architecture：全部 target-free interictal benchmark；
4. E：冻结 A/B/common/contrast model fields；
5. F：一次性 early-ictal unseal 和评分；
6. G/H：effective motif、lesion、human/RNN/SNN common observables；
7. I：Figure 6、报告、验收、分批 commit。
