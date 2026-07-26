# Topic 5 / Figure 6：symmetric-axis E/I system identification RNN v2.0

**日期**：2026-07-26
**状态**：待执行的冻结设计合同
**上游结论**：v0.7/v0.9/v1.0 已冻结为 bounded negative；不得继续 K sweep 或
event-persistent path-mode 优化。
**目的**：从间期 contact-rank 序列辨识一个患者特异、近似对称、轴向各向异性的共同
传播 scaffold，并检验不同事件起点是否足以产生正反传播；纯间期门通过后，才做冻结的
clinical-onset 发作期动态迁移。

## 1. 核心科学问题

> 大量间期 HFO 群体事件的触点顺序，能否由一个患者特异、近似对称且沿病理轴各向异性
> 的共同传播 scaffold 解释；事件间相反的传播方向是否主要由起始位置而非离散 path
> identity 决定；这一仅由间期数据学习的系统能否在不接触发作期训练信号的情况下，
> 根据 clinical-onset 附近最早招募触点预测后续发作招募？

模型对象固定为：

```text
shared symmetric scaffold
+ event-specific source
+ local excitation / restraint state
```

不是：

```text
K discrete paths
+ one persistent path identity per event
```

## 2. 研究层级与数据分母

### 2.1 SNN 参数恢复层

先使用 Topic 4 已验证的患者特异 E/I SNN 生成专用 benchmark。现有
`readout_*.json` 已保存逐事件 contact ranks，`figdata_*.npz` 已保存 montage、
真实 axis 和两个 source foci；但当前代表性产物经过工作点/事件选择，而且没有完整保存
可用于 E/I 对照的隐状态。因此不能直接把 Figure 4 代表性事件当作 v2.0 Gate 0。

必须新建未挑选、paired-seed 的 synthetic benchmark，并保存：

- 每个事件的 tie-aware contact-rank sets；
- contact coordinates、valid mask 和固定 observation noise；
- 真实无向 axis（符号任意）、长/短轴尺度比和 source arm；
- 事件级 contact E-envelope；
- 若引擎可导出，contact-level I-envelope 或 inhibitory-current proxy；
- generator config、engine fingerprint、seed 和未筛选事件清单。

SNN latent quantities只用于训练后的参数恢复审计，不进入模型损失。

### 2.2 人体纯间期层

- 数据集固定为 `results/topic5_interictal_rank_distribution/dataset_v0_4`。
- 34 位患者全部进入预测 inventory；不得回到 13 人候选表。
- 每位患者 chronological train80 / heldout20。
- masked rank、event independence、A/B 禁入、IEI 禁入和发作期 target seal 沿用
  v1.0。
- 当前审计中 25/34 患者全部候选触点有坐标，9/34 坐标不完整。34 人全部报告
  conditional prediction；三维物理轴恢复的 primary geometry subset 在运行前按
  `geometry_mapped == n_contacts` 冻结。其余患者进入 topology-only sensitivity，
  不得把 latent contact coordinate 写成物理三维轴。

### 2.3 发作期层

只有 Gate 0–3 全部通过后才建立发作期 inventory。不得沿用旧的
`candidate_target_patient` 13 人路由作为分母。所有 34 位患者逐一记录：

- 是否有 clinical-onset 时间；
- 是否有 clinical-onset contact set；
- 是否能得到与间期同名空间对齐的 ictal recruitment rank；
- 排除原因。

Epilepsiae 的 clinical onset 是主时间锚。Yuquan 或其他只有 EEG onset 的病例不得与
clinical-onset primary pool 混合；只能作为明确标记的 sensitivity。既有结果已提示
EEG-onset 不显著，不能用它替代 clinical-onset 主任务。

## 3. 输入与输出

### 3.1 输入

患者 \(p\) 的第 \(e\) 个事件表示为 rank-set 序列

\[
\mathcal E_e=(S_{e,1},\ldots,S_{e,T_e}),
\]

其中每个 \(S_{e,t}\) 可包含近同时触点。模型在 prefix \(S_{e,1:t}\) 下工作；第一组
触点定义该事件已经观察到的 source，不需要额外的 A/B 或 path label。

每个触点输入只允许包含：

- 是否已在 prefix 中参与；
- 当前 rank/prefix progress；
- train80 估计并固定的节点基线参与倾向 \(b_i\)；
- 坐标和 shaft topology；
- 当前可解释局部状态。

触点字符串 ID 不得作为模型特征。

### 3.2 输出

两个共同主输出：

1. 下一 rank set 与 STOP 的条件概率；
2. 给定当前 prefix 后，每个未观察触点的未来参与概率与剩余相对 rank distribution。

secondary：

- prefix-conditioned soft multistep rollout；
- 每触点 early/middle/late 概率；
- 完整自由 rollout 的 participation/rank distributions；
- latent excitation/restraint trajectory。

自由 rollout 不再作为阻断主门。它只能说明生成外推，不得反向挽救 conditional gate。

## 4. 模型合同

### 4.1 唯一跨触点算子

患者图固定写成

\[
W_p = \operatorname{norm}\!\left(
\alpha_{\mathrm{local}}K_{\mathrm{local},p}
+\alpha_{\mathrm{axis}}K_{\mathrm{axis},p}
\right), \qquad W_p=W_p^\top .
\]

对完整三维几何病例：

\[
K_{ij}^{\mathrm{axis}}=
\exp\left[
-\frac{d_{\parallel,ij}^2}{2\ell_\parallel^2}
-\frac{d_{\perp,ij}^2}{2\ell_\perp^2}
\right],
\qquad \ell_\parallel>\ell_\perp ,
\]

其中 \(d_\parallel\) 和 \(d_\perp\) 由单位轴 \(\mathbf u_p\) 分解触点距离。
\(\mathbf u_p\) 与 \(-\mathbf u_p\) 等价，所有恢复/稳定性统计使用 sign-invariant
比较。坐标先在患者内中心化，并用 train80 触点的 median non-zero pairwise distance
缩放；不得用 heldout event 或发作期量选择坐标尺度。

坐标不完整病例使用单一 latent axial coordinate + shaft-smoothness 的
topology-only fallback；这一层可参与预测，但不能用于三维物理轴 claim。

这里的“low-rank”来自单一轴、对称核和两类局部状态，而不是重新引入自由
\(UV^\top\)。实现可为节省显存对 \(W_p\) 作对称 eigentruncation，固定保留解释至少
99% operator Frobenius energy 的最小 rank；完整 kernel 才是科学定义。该 rank
不得 sweep 或作为模型选择变量。若 SNN Gate 0 中压缩版相对完整 kernel 的任一主终点
误差超过 1%，正式人体实验使用完整 kernel，并把 low-rank 仅保留为工程敏感性。

模型内不得出现：

- 任意 dense contact-to-contact recurrent matrix；
- 可绕过 \(W_p\) 的 contact-mixing GRU/attention；
- forward/reverse 两套独立参数；
- event-persistent path identity；
- 训练中使用 A/B、IEI、发作期值或 heldout20 构图。

### 4.2 起点和双向性

事件 source 由已观察的第一 rank set 表示。网络没有 direction latent label。
相反方向必须来自同一个 \(W_p\) 在不同 source 下的状态演化。模型可使用 source 到各
触点的相对轴坐标，但不能根据 source 选择另一套权重。

### 4.3 可解释局部状态

每个触点保留两个非负状态：

\[
E_{t+1}=(1-\tau_E)E_t+\tau_E\,f_E(W_p x_t,E_t),
\]

\[
I_{t+1}=(1-\tau_I)I_t+\tau_I\,f_I(E_t,x_t),
\]

\[
\operatorname{logit}h_{i,t}
=b_i+\beta_E E_{i,t}-\beta_I I_{i,t},
\qquad \beta_E,\beta_I\ge0 .
\]

状态的符号、作用方向和时间常数范围在模型中固定，禁止任意线性旋转混合。这里的
\(E/I\) 首先是“局部兴奋 drive / restraint”状态；只有同时满足以下条件，论文中才可
称其为 biological E/I-like state：

1. SNN benchmark 上未用真值训练却能恢复对应 E/I proxy；
2. no-restraint lesion 对 synthetic 和 human conditional endpoints 的影响方向一致；
3. 跨 seed 的状态轨迹稳定。

任一不满足，只能写成 excitation/refractory latent components，不能推断患者细胞级
E/I 机制。

### 4.4 节点基线隔离

\(b_i\) 只由 train80 估计。完整模型、isotropic control、axis shuffle 和状态 lesion
必须共用完全相同的 \(b_i\)。任何结构收益都必须来自 \(W_p\) 和状态转移，不能来自重新
拟合节点频率。

## 5. 自监督训练

损失固定为：

\[
\mathcal L=
\mathcal L_{\mathrm{next\ set}}
+\lambda_{\mathrm{stop}}\mathcal L_{\mathrm{stop}}
+\lambda_{\mathrm{future}}\mathcal L_{\mathrm{future\ participation}}
+\lambda_{\mathrm{rank}}\mathcal L_{\mathrm{remaining\ rank}}
+\lambda_{\mathrm{reg}}\mathcal R .
\]

- next-set 使用 multi-label set likelihood，不把同 rank 触点强行排序。
- future participation 对每个 prefix 预测后续是否出现。
- remaining rank 只在真实后续参与触点上计算，使用归一化剩余 rank distribution，
  不退化成仅预测 mean rank。
- 训练器首选 AdamW、gradient clipping、mixed precision 可选；所有 seed、覆盖次数、
  checkpoint 和日志必须保存。
- \(\lambda\)、rollout horizon、状态维度和时间常数范围只允许在 SNN Gate 0 中选择；
  进入人体 pilot 后全部冻结。
- 人体仍采用 LOSO shared dynamics + heldout-patient train80 system identification；
  heldout20 只评估。

## 6. 对照与消融

共同 primary controls：

1. `node_bias_no_history`：只有同一个 \(b_i\)；
2. `local_isotropic`：保留局部对称图，移除轴向各向异性；
3. `axis_shuffle`：保留节点、边密度、权重分布和 shaft 组成，破坏轴与触点的对应；
4. `no_restraint`：移除 \(I\) 对 hazard 的作用；
5. `asymmetric_upper_bound`：允许定向图，仅作容量/错配上限，不作为目标模型；
6. `dense_gru_upper_bound`：只作 engineering ceiling，输赢均不授权机制结论。

结构必要性以重新训练的 nested control 为 primary；对冻结完整模型做 in-place lesion
只作 secondary，避免把分布外损伤误当成必要性。

## 7. 分阶段硬门

### Gate 0：SNN 已知真值恢复

benchmark 至少包含：

- symmetric-anisotropic + source-left/source-right paired arms；
- symmetric-isotropic negative control；
- asymmetric directed-generator misspecification control（若现有 SNN guarded engine
  不支持定向核，使用独立的轻量 contact-graph generator；不得为这一负对照修改已验证
  SNN engine）；
- 有/无局部 restraint 的状态 control。

先做 3-seed smoke，再做至少 12 个 paired seeds confirm。confirm 必须同时满足：

1. **轴恢复**：anisotropic 条件的 median
   \(|\widehat{\mathbf u}\cdot\mathbf u_{\mathrm{true}}|\ge0.80\)，且高于 isotropic
   轴-null 的 95th percentile；
2. **各向异性恢复**：至少 80% paired seeds 正确给出
   \(\widehat\ell_\parallel/\widehat\ell_\perp>1\)，而 isotropic 条件的假阳性不超过
   1/12；
3. **同图双向性**：同一个冻结 \(W\) 在 left/right source 下的预测位移符号相反，
   且两侧 next-set 与 future-rank 均在至少 80% paired seeds 优于 isotropic control；
4. **结构必要性**：axis removal/shuffle 在 anisotropic 条件下使两个主终点均恶化，
   至少 80% seeds 同向；在 isotropic 条件下不产生伪“轴必要性”；
5. **错配识别**：directed generator 下 asymmetric upper bound 在两个主终点均稳定
   优于 symmetric model，证明测试能识别“对称假设不成立”；
6. **状态可解释性**：若导出真 I proxy，未监督状态与真实 contact-level E/I proxy 的
   patient/seed median Spearman \(|\rho|\ge0.50\)，且 restraint lesion 方向正确。
   未满足只封禁 E/I 生物学命名，不单独否决 scaffold Gate 0。

Gate 0 的任一 scaffold 条件失败：停止，不进入人体；优先判定模型不可辨识，不得在人
数据上继续调参。

### Gate 1：三位既有开发病例工程 pilot

固定使用 `epilepsiae_1073`、`epilepsiae_1146`、
`yuquan_chenziyang`，3 seeds。不得新增开发患者。通过条件：

- 训练覆盖、determinism、checkpoint、target seal 全部通过；
- 3 位中至少 2 位的 seed-median，在 next-set 和 future-rank 两个主终点上均优于
  `node_bias_no_history` 和 `local_isotropic`；
- 至少 6/9 patient-seed runs 同方向；
- 无 NaN/OOM，节点基线 fingerprint 在各 control 完全相同。

Gate 1 只决定工程可行性，不产生 cohort claim，也不用于改 Gate 2 阈值。

### Gate 2：34 人正式纯间期条件预测

- 34 heldout folds × 3 seeds；患者内先取 seed median。
- full model 相对 `node_bias_no_history`、`local_isotropic` 和 `axis_shuffle`，
  必须在 next-set 与 future-rank 两个主终点上同时满足：
  cohort median benefit \(>0\)、改善患者 \(>17/34\)、one-sided patient-level
  Wilcoxon 经全部 primary comparisons BH-FDR 后 \(q<0.05\)。
- no-restraint 为状态必要性；只有两个主终点均恶化才允许解释 restraint。
- symmetric model 相对 asymmetric upper bound 的 non-inferiority margin 在
  Gate 0 冻结为“保留至少 90% 的 full-vs-no-history benefit”。若不满足，仍可报告
  预测结果，但“近似对称 scaffold 足够”判为失败。
- 31 人 development-excluded sensitivity 并列报告，不替代 34 人主合同。

### Gate 3：轴稳定性与同一 scaffold 双向性

三维物理轴 primary subset 固定为 Gate 2 前
`geometry_mapped == n_contacts` 的患者：

- train80 split-half 轴 sign-invariant cosine / contact projection Spearman 的
  cohort median \(\ge0.70\)，并高于 shaft-preserving null（patient-level
  \(q<0.05\)）；
- 冻结模型后，才使用 heldout20 的数据驱动模板/传播轴作外部 read-back；A/B 仅在此
  后验验证步骤出现；
- 按第一 rank set 在 learned axis 两端分层，两端事件都必须保持 full-vs-isotropic
  的 next/future benefit，且后续招募的 signed axial displacement 随 source side
  翻转；
- 不允许 direction-specific weights。若只一侧通过，结论降为单向 scaffold。

Gate 2 或 Gate 3 失败：纯间期 bounded negative 收口，不打开发作期值。

### Gate 4：冻结的 clinical-onset 动态迁移

先做全 34 人 target inventory，再冻结 denominator 和 ictal-rank extractor。任务是：

1. 以 clinical-onset 时间锚附近最早的 1–2 个 ictal rank sets / 临床 onset contacts
   作为已观察 prefix；
2. 冻结所有间期参数和 \(W_p\)；
3. 预测后续触点是否招募及其剩余 recruitment rank；
4. 与同一节点基线的 no-history、local-isotropic 和 axis-shuffle readout 比较。

不得：

- 用发作期数据微调模型、轴、阈值或 loss 权重；
- 以 EEG onset 替代 clinical onset 后混入 primary；
- 把逐秒 `[0,10] s` 静态能量窗当成传播顺序；
- 把 retrospective prefix completion 写成 prospective seizure warning。

旧 clinical-onset `[0,10] s`、`1–150 Hz` 静态能量场只作为 secondary compatibility
readout，检验动态模型汇总场与已知 shared field 是否一致，不决定 Gate 4。

## 8. Figure 6 六块的冻结科学含义

| Panel | 科学问题 | 必须显示 |
|---|---|---|
| A | 假设是什么 | 一个对称轴图、不同 source、局部 E/restraint；明确无离散 path mode |
| B | 模型能否识别已知机制 | SNN 真轴 vs recovered axis、anisotropic/isotropic/directed controls |
| C | 人体事件中结构是否改善条件预测 | 34 人 next-set + future-rank patient-level effects |
| D | 轴是否独立、稳定且必要 | split-half、shaft-null、isotropic/axis-shuffle lesion |
| E | 同一 scaffold 是否解释两种方向 | source-side 分层、同一 \(W\)、signed displacement 翻转 |
| F | 能否跨状态迁移 | clinical-onset prefix → later recruitment；未过门则明确 target sealed |

每个 panel 都必须有独立统计或明确 gate，不能只画模型示意。图目录必须有中文
`figures/README.md`。

## 9. 允许与禁止的最终措辞

全部 Gate 0–3 通过后，允许：

- “Interictal rank sequences identify a patient-specific symmetric
  anisotropic scaffold.”
- “Opposing propagation is compatible with source reversal on a shared
  scaffold.”
- 若 Gate 4 通过：“The frozen interictal scaffold predicts later
  clinical-onset recruitment.”

仍禁止：

- RNN 证明了患者真实细胞级 E/I 机制；
- A/B 是模型训练标签或两个独立通路；
- retrospective ictal prefix completion 是提前预警；
- 单纯 AUC/NLL 更高等于机制成立；
- topology-only 病例恢复了三维物理轴。

## 10. 停止规则

- 不再运行 v0.7/v0.9/v1.0 新 seed、K 或超参数。
- Gate 0 不过，不进入人体。
- Gate 1 不过，只允许修明确 bug；不得在人 pilot 上重选 loss/结构。
- Gate 2/3 不过，不读取发作期值。
- clinical-onset rank extractor 或 denominator 无法冻结时，Gate 4 标记 blocked；
  静态能量场不能代替动态任务。
