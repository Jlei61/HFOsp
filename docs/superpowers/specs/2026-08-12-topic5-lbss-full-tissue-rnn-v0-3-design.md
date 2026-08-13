# Topic 5.1 Full-tissue LBSS-RNN 科学设计 v0.3

> 状态：**执行完成，2026-08-13 closeout audit PASS**。v0.2 的 contact-dilated latent domain 已被审计为会混淆 local 与 nonlocal propagation；本版本只修复 latent tissue domain，并继续执行已经冻结的 LBSS matched experiment。正式结果见 `docs/archive/topic5/lbss_full_tissue_spatial_search_closeout_2026-08-12.md`。

## 1. 核心问题

在患者的冻结传播平面上，用大量 latent recurrent units 表示 contact 之间和 contact 直接观测范围之外的组织；SEEG contacts 仅通过局部 `H^T/H` 作为输入/读出端口。检验：

\[
\boxed{
\text{local recurrent tissue backbone}
+
\text{few task-selected nonlocal shortcuts}
}
\]

是否比 local-only、等容量 extra-local、固定随机 nonlocal 和 order-shuffle 更能生成患者间期远端传播，并在完全由间期数据训练和冻结后保留 early-ictal broadband energy field 的对应。

本版本的主张是模型计算充分性与跨状态空间一致性，不是恢复真实白质边或真实未采样组织活动。

## 2. v0.2 的 P0 修复

v0.2 的所有 nodes 均满足：

\[
\min_c\|r_i-r_c\|\leq3\sigma,
\]

因此不存在真正未被 contact 直接读出的 tissue state。v0.3 改为：

1. latent domain 由患者 contact cloud 在冻结传播平面中的几何包络定义，而不是由 `H` 的 support union 定义；
2. background tissue nodes 在该包络内近似均匀采样；
3. 每个 contact 额外配置最小局部 support nodes，保证 `H` 是邻域平均而不是单节点参数；
4. `H` 继续在 `3 sigma` 处严格截断，因此包络内部会自然产生 `H[:,i]=0` 的 unobserved latent nodes；
5. zero-H nodes 只能经 recurrent propagation 被驱动，不能直接看到 contact input 或直接进入 contact output。

## 3. Full-tissue latent domain

### 3.1 几何包络

对非退化二维 contact cloud，计算 convex hull，并将每条 hull half-space 向外平移：

\[
m_p=\max(3\sigma_p,d_{\mathrm{contact,NN,med}}).
\]

所得 offset convex envelope 为患者 latent interpolation domain。若 contact cloud 近共线，使用 PCA 主轴上的最小包络，并在主轴与横轴各扩展同一 `m_p`。该包络只表示冻结传播平面中的计算域，不解释为完整皮层解剖边界。

### 3.2 Node placement

节点分两部分，但进入同一个 recurrent state：

- `observation-support nodes`：每个 contact 周围半径 `0.5 sigma` 的三个 latent nodes；contact 本身不是 node；
- `background tissue nodes`：在完整包络候选网格上，以 farthest-point sampling 近似均匀放置。

背景 node 的目标间距：

\[
d_{bg}=\max(2\ \mathrm{mm},d_{\mathrm{contact,NN,med}}).
\]

背景数目按包络面积除以 `d_bg^2` 决定，最低 64；总 node 数最高 384。support nodes 与 background nodes 去重。若 zero-H nodes 少于 16 或低于总 nodes 的 10%，只允许增加包络内 zero-H background nodes，不删除已有 nodes、不改变包络、不得读取任何传播或发作结果。

### 3.3 Observation contract

\[
H_{ci}\propto\exp\left(-\frac{\|r_c-r_i\|^2}{2\sigma_p^2}\right),
\qquad H_{ci}=0\quad\text{if }\|r_c-r_i\|>3\sigma_p.
\]

每个 contact 的 H row 归一化为 1；每个 contact 至少直接观察 3 个 nodes。允许且要求存在 `H` column sum 为 0 的 nodes。输入只经 `H^T x_t`，输出只经 `H z_t`，不允许 contact-to-contact bypass。

## 4. 主模型矩阵

继续使用同一个 leaky RNN cell、同一个 local backbone 和同一个 added-edge budget：

| Arm | 固定 local backbone | 新增边 | 作用 |
|---|---|---|---|
| `L0_LOCAL_ONLY` | 是 | 无 | 完整组织平面上的局部传播基线 |
| `L1_LOCAL_PLUS_LEARNED_EXTRA_LOCAL` | 是 | K 条任务选择的 extra-local | 等容量、等优化控制 |
| `L2_LOCAL_PLUS_RANDOM_LR` | 是 | K 条固定随机 nonlocal | 任意 shortcut 控制 |
| `L3_LOCAL_PLUS_LEARNED_LR` | 是 | K 条任务选择的 nonlocal | 癫痫双尺度主模型 |
| `C_L3_ORDER_SHUFFLED` | 是 | K 条任务选择的 nonlocal | 有序间期传播信息控制 |

local backbone 仍为 symmetrized kNN candidate mask，两个方向权重独立。所有 nodes 必须位于一个 strongly connected component。K、candidate pools、SET、checkpoint、decoder、loss、split 与 v0.2 保持不变。

Wiring-economy dense/sparse/spatial-cost 矩阵不重跑为主线；其旧结果只作“多种 recurrence 可完成任务、空间成本可降低资源”的补充 benchmark。

## 5. Cohort 与分母

- contact-space 自监督 RNN 的间期可学习性：完整 34 位 K=2 cohort，沿用已验收结果；
- full-tissue LBSS spatial mechanism：具有可用二维传播平面的 21 位患者、31 fits；
- early-ictal benchmark：模型只在间期训练。空间机制分析必须先报告与 Figure 3D 17 人/167 seizures 的 exact join；2026-08-12 target-value-free metadata audit 冻结的实际交集为 **12 人/141 seizures**，exact contact support 为 8–16。缺少 full-tissue geometry 的 5 位患者必须进入 attrition table，不能静默排除；
- 不以 individual A/B axis 显著性筛患者。

## 6. Primary endpoints

### 6.1 Interictal

1. heldout next-contact NLL 与 STOP；
2. supplied first rank 后的 free-rollout rank correlation；
3. empirical interictal field fidelity；
4. local/intermediate/distal transition NLL；distal 使用新招募 contact 到当前传播前沿的最短距离；
5. true-order vs order-shuffle；
6. `L3-L0`、`L3-L1`、`L3-L2` 的患者级 paired contrasts。

### 6.2 Early-ictal frozen-field benchmark

所有 model fields 在 target access 前冻结。与 Figure 3 主分析同构的 external benchmark 使用 clinical onset 后 0–10 s phenotype-matched readout（strict broadband 为 1–150 Hz，gamma-nonbroadband 为冻结 gamma readout）、synchronized all-contact shuffle、mirror/maxAB 和 patient-first aggregation；其空间模型交集固定为 12 人/141 seizures。另行报告纯 1–150 Hz strict-broadband sensitivity（11 人/92 seizures）。E1146 主图空间场只使用 15 次 strict-broadband seizures 的 0–10 s、1–150 Hz energy，不混入 gamma readout。

- D1：canonical full field 的跨状态对应；
- D2：seed-removed field 的 `L3-L0/L1/L2` 增量，或 L3 nonlocal attenuation 的单调损害；
- full field 与 seed-removed field 并列报告，不能互相替代。

### 6.3 Mechanistic perturbation

对各 arm 自己实际拥有的 added edges 做 `25/50/75/100%` attenuation。主量为 distal-selectivity AUC 和 L3 nonlocal vs L3 matched-local subset 的 double dissociation。精确 edge overlap 仅作 secondary；承重对象是 contact-space effective influence、endpoint density、distal reach 与 perturbation response。

## 7. 必须报告的 geometry audit

每个 fit 保存并汇总：

- latent-domain area；
- total/support/background/zero-H node counts；
- zero-H fraction；
- H-support count per contact；
- contact-private node fraction；
- local-edge length distribution；
- local edges 穿过 zero-H tissue 的比例；
- local/extra-local/nonlocal pool size 与毫米尺度；
- graph strong connectivity；
- estimated dense-state GPU memory。

v0.2 与 v0.3 的 geometry audit 必须并列作图；不得只展示修复后漂亮的代表患者。

## 8. 结果解释边界

允许：

- full-tissue recurrent model 可从稀疏 SEEG readouts 学习患者特异间期传播；
- local recurrence 是否足够，或 task-selected nonlocal shortcuts 是否对 distal propagation 提供增量；
- frozen model field 与 early-ictal field 是否存在 target-free 对应。

不允许：

- latent nodes 是真实未记录神经元；
- learned shortcuts 是真实白质纤维；
- wiring economy 是癫痫网络形成机制；
- 一个阴性结果否定患者存在病理传播轴。

## 9. Target-free 空间连接搜索附加合同

若正式 v0.3 间期结果中 `L3` 未在 distal propagation 上同时优于
`L0/L1/L2`，不得直接解封 early-ictal target，也不得据此宣布空间先验无效。
先在固定的三个 development fits（E1084、E1146、Yuquan chengshuai）上，
仅使用间期 heldout 数据检查六个预定义因素：local density、added-edge
fraction、nonlocal cutoff、rewiring fraction、learning rate 和每个 tissue node
的 state dimension（1/2/4）。state dimension 改变内部动力学容量，但不能增加
contact-to-contact bypass，所有 matched arms 必须使用同一维度。

第一阶段是 13 个单因素配置、L3 only、3 fits × 3 seeds。选择指标依次为：

1. distal contact NLL；
2. overall contact NLL；
3. seed-removed free-rollout Spearman；
4. seed stability。

三个 seed 只作为优化重复，不能作为九个独立样本。每个 endpoint 必须先在每个
development fit 内对三个 seed 取中位数，再在三个 development fits 间取中位数并排序；
不得直接 pool 9 个 fit-seed 单元。

候选配置必须全部收敛且 checkpoint 位于 mask freeze 之后；相对基准的 overall
NLL 不得恶化超过 0.01 nats，rollout 不得恶化超过 0.02。单因素中只有 distal
NLL 改善超过 0.002 nats 的水平才可进入 joint candidate。随后选择最多两个
非基准配置，在相同的 5 arms、3 fits、3 seeds 上作 matched confirmation。

所有搜索配置、选择规则、代码 hash 和 development fits 必须在运行前写入
`SEARCH_CONTRACT.json`。搜索期间若发现任何 target authorization/access marker，
立即终止。只有 development confirmation 显示 `L3` 相对 L0/L1/L2 的 distal
增量一致，才允许将该配置冻结为新的 full-cohort confirmation；否则正式结论是
“当前搜索范围内未辨识出优于局部/等容量/随机 shortcut 的空间配置”，不能继续
无限扩网格追阳性。

一旦使用三位 development 患者选择配置，最终患者级统计必须同时报告：

- development-excluded spatial cohort 作为确认性结果；
- 全部 spatial cohort 作为描述性/支持性结果。

不从数据表中删除 development 患者，也不声称每位患者都必须出现同一 motif；只是
避免把用于选择配置的三位患者再次当作独立确认样本。

后处理固定分成两段：间期统计、完整/扰动 fields 和 manifests 先完成并写入
`INTERICTAL_POSTPROCESS_PRETARGET_COMPLETE.json`；默认在此暂停。只有空间模型
决策完成并留痕后，才显式使用 `--through-target` 进入统一 early-ictal benchmark。

若 development 选择出的新配置通过 development-excluded full-cohort confirmation，
不得用其结果覆盖原 v0.3 目录。必须建立独立 sibling artifact root，将已确认的 465 个
unit 作为只读 `per_fit` 输入，重新冻结该配置自己的 interictal summary、intact fields、
pathways 和 attenuated fields；这些 target-free manifests 全部完成后，才允许由该独立
root 解封同一个 early-ictal scorer。最终 Figure 与 closeout 必须跟随
`PRIMARY_ARTIFACT_POINTER.json`，不能误读原 v0.3 fields。
任何 checkpoint 重建必须逐单元读取 `density`、`added_fraction`、
`r_local_multiplier` 与 `state_dim`；不得用基准配置重建搜索后模型。
