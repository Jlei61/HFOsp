# Topic 5 RNN 部分整体综合报告与最终验收

日期：2026-07-28
机器可读验收：
`results/topic5_rnn_overall_acceptance/FINAL_ACCEPTANCE.json`

> **2026-07-30 更新**：本文件保留为完整模型谱系和上一版总验收。最新的阶段性结论、
> 训练充分性缺口与下一轮唯一允许任务见
> `rnn_stage_acceptance_and_training_sufficiency_2026-07-30.md`。完整事件生成阴性现仅限
> 当前 frozen teacher-forced realization；在 coverage-cycle 和 rollout-aware
> objective 审计完成前，不解释为任意 RNN 均无法学习完整传播。

# 审阅结论

## 1. 一句话判断

RNN 部分已经完成到可以正式收口的程度，但最终科学对象与最初设想不同。

当前真正成立的是：

\[
\boxed{
\text{稳定的患者特异间期 contact scaffold}
+
\text{最近 2--3 个 rank set 的有序短历史}
}
\]

尚未成立的是：

\[
\boxed{
\text{离散 path mode、正 low-rank mode、物理病理轴、competition/source 机制、}
\text{GRU-specific 发作迁移}
}
\]

因此，RNN 可以进入论文，定位应是：

> **target-blind 的间期序列自监督结果 + 跨状态静态 contact morphology 的内部验证 +
> 对多个机制化 RNN 假设的 bounded falsification。**

它不能承担“模型自动恢复 A/B 病理轴并预测发作传播”的主机制结论。

## 2. 完成程度

> **科学合同执行至各自冻结停止点：100/100**

> **整体论文验收：PASS，但限 supplementary bounded computational result**

完成的正式分支包括：

| 分支 | 正式规模 | target 状态 | 最终角色 |
|---|---:|---|---|
| full-history GRU 与非递归对照 | 34 人 × 3 seeds | 间期阶段未读 | 数据中是否存在可学习顺序 |
| unconstrained low-rank leaky RNN | 34 人 × 3 seeds × rank 0–4，510 folds | 未读 | low-rank sensitivity |
| persistent path-mode graph RNN | 34 人 × 3 seeds × 5 conditions，510 runs | 未读 | 离散 path 假设 falsification |
| symmetric linear observation model v2.2.1 | 22 人，66 formal runs | 未读 | 非负线性对称核 falsification |
| transition decomposition | 31 人；physical-axis 22 人 | 未读 | 定位经验 transition signal |
| competitive structured RNN v2.3 | 22 人 × 3 seeds × 5 conditions，330 models | 未读 | axis/competition/source 检验 |
| axis read-back v2.4 | axis-positive 9 人 | 选轴阶段未读 | axis construct validity |
| internal-state reduction | 34 人 × 3 seeds | direction 冻结后才读 | 顺序扰动与低维诊断 |
| fixed static early-ictal readout | 16 人、106 seizures | 已读且明确复用 | 同数据集内部跨状态验证 |
| H1/H2/H3 history necessity | 34 人 × 3 seeds，408 新模型 | 未读 | 冻结有效历史深度 |

所有正式分支均有患者级统计、seed 合并、输入 fingerprint 或 target seal。新完成的
H1/H2/H3 与 matched shuffle 共 204 个 formal fold logs，未发现 OOM、NaN、CUDA error
或 Traceback。

## 3. 这部分最初想回答什么

最初目标包含三层：

1. 间期 contact-rank 事件是否包含可学习的患者特异传播信息；
2. RNN 内部是否会自然浮现 A/B 双向传播、低维 path 或患者病理轴；
3. 间期学到的结构能否迁移到 clinical onset 后的 early-ictal energy field。

现在三层的答案已经分开：

| 科学问题 | 最终答案 |
|---|---|
| 间期 rank sequence 可不可学 | 可以 |
| 是否只需要静态 contact frequency | 不完全；最近 2–3 个 rank set 有额外顺序信息 |
| 是否需要整场 full history | 不需要 |
| 是否自然浮现可辨识正 low-rank mode | 当前参数化下没有 |
| 是否需要离散 A/B/path identity | 没有证据 |
| 是否恢复物理病理轴 | 没有 |
| 是否需要 delayed competition/source term | 没有 |
| 是否与 early-ictal static energy 有空间对应 | 有方向无关的内部数据集对应 |
| 该对应是否由 GRU 或 ordered history 特异产生 | 没有证据 |
| 是否预测发作传播或 clinical onset | 没有测试 |

## 4. 数据、输入和训练合同

### 4.1 间期输入

- cohort：34 人，18 名 Epilepsiae、16 名 Yuquan；
- 输入：一场群体间期事件的 masked contact rank-set sequence；
- 非参与触点不带入 phantom rank；
- split：chronological train80 / heldout20；
- self-supervised label：下一 rank set 或 STOP；
- A/B、IEI、SOZ、seizure label 和 early-ictal target 不进入间期训练。

这不是学习 A/B 标签，也不是重新做 KMeans。模型看到的是：

> 当前事件已经招募了哪些触点、以什么先后顺序到达，下一步最可能出现哪些触点。

### 4.2 普通 GRU

普通 GRU 允许自由 hidden mixing，作用是检验数据上限：

- 数据中是否有顺序信息；
- 大量事件能否汇总成 participation、rank 和 precedence 分布；
- 不预先规定病理轴或路径数量。

它适合作为 information probe，不天然具有机制可解释性。

### 4.3 有限历史 GRU

H1、H2、H3 只改变预测前重放的最近 rank-set 数：

- H1：最近 1 步；
- H2：最近 2 步；
- H3：最近 3 步；
- full：全部 prefix。

完整 prefix 仍用于 candidate mask，因此不会重新选择已经参与的触点；最终事件长度和
未来参与触点均不可见。

### 4.4 结构化 RNN

先后检验过：

- 离散 persistent path mode；
- 对称非负轴向 kernel；
- propagation/competition traces；
- source-conditioned direction；
- unconstrained low-rank recurrent update。

这些模型的共同目的不是刷 NLL，而是问：

> 当前提出的结构先验是否是数据中 transition information 的必要、可辨识解释。

## 5. 最终结果

### 5.1 静态间期 participation scaffold 可重复

对每个触点定义其参与事件的概率。train80 与 heldout20 的患者内 contact-wise
Spearman：

| 指标 | 结果 |
|---|---:|
| 患者数 | 34 |
| 中位 \(ρ\) | 0.893 |
| 95% CI | [0.868, 0.936] |
| 正值患者 | 34/34 |
| one-sided \(P\) | \(1.82\times10^{-7}\) |

相对 within-shaft circular null：

| 指标 | 结果 |
|---|---:|
| eligible | 33 |
| observed − null 中位数 | 0.685 |
| 95% CI | [0.536, 0.727] |
| 正值患者 | 33/33 |
| patient null \(P<0.05\) | 31/33 |

约 200 个事件时，30/34 名患者已经落在 full train80 estimate 的 Spearman 0.05
范围内；500 个事件时为 31/32。

**科学含义**：

> 间期事件中“哪些触点经常参与”是稳定的患者特异对象，不只是一次聚类结果。

**边界**：

> participation scaffold 不是完整 rank distribution，也不是物理轴或传播方向。

### 5.2 最近 2–3 步包含真正的顺序信息

| 比较 | 中位 NLL gain | 95% CI | 正值患者 | two-sided \(P\) |
|---|---:|---:|---:|---:|
| H2 over H1 | 0.0172 | [0.0117, 0.0240] | 32/34 | \(1.01\times10^{-6}\) |
| H3 over H2 | 0.0113 | [0.0089, 0.0154] | 29/34 | \(2.95\times10^{-8}\) |
| Full over H3 | -0.0010 | [-0.0051, 0.0035] | 16/34 | 0.436 |
| ordered H3 over matched H3 shuffle | 0.0261 | [0.0168, 0.0348] | 27/34 | 0.00361 |

另一个独立训练的 full GRU vs rank-shuffle 比较同样为正：

- median gain 0.0408；
- 27/34 为正；
- one-sided \(P=2.51\times10^{-4}\)。

但是 full GRU 没有超过 strongest nonrecurrent prefix model：

- median gain 0.0010；
- 17/34 为正；
- one-sided \(P=0.440\)。

**最终解释**：

> 事件顺序不是随机排列，真正有用的历史主要集中在最近 2–3 个 rank set；无界 full
> history 不是必要对象。

### 5.3 hidden state 低维，但低维本身不是机制证据

full-history GRU hidden state：

- effective rank 中位约 1.88；
- 2 PCs 保留约 85.5% variance；
- 4 PCs 保留约 96.5% variance；
- 跨 seed CKA 很高。

但 rank-shuffle GRU 同样低维、稳定。普通线性 probe 也会使用 static contact prior 和
prefix size。

更可靠的证据来自 matched prefix-order perturbation：

- ordered GRU order-shuffle NLL penalty 中位约 0.012；
- rank-shuffle GRU 约 0.0016；
- ordered-minus-shuffle penalty 中位约 0.0100；
- 32/34 为正，\(P=1.79\times10^{-8}\)。

因此可以说模型内部使用了 rank 顺序；不能把 PC1/PC2 直接命名为 excitation、
inhibition、propagation phase 或患者物理轴。

### 5.4 full-rank 与 unconstrained low-rank 结果

full-rank GRU 是数据可学性的参考，不是最终机制模型。其自由生成结果只有 10/34
患者同时落入 participation、rank 和 precedence 三项经验变异范围。

low-rank leaky RNN 扫描 rank 0–4：

- 34 人 × 3 seeds × 5 ranks，共 510 folds；
- 没有正 rank 通过预设 distribution gate；
- rank 1–4 没有稳定优于 rank 0；
- rank 1 contact loading 的 chance-adjusted 跨 seed similarity 仅约 0.044。

同时，rank 0 并非真正无递归模型：它仍有 32 个独立 diagonal decay 和记忆通道。
所以这一分支的正确定位是：

> **unconstrained low-rank sensitivity negative**，不能作为“低维机制不存在”的正式否定，
> 也不能把 rank 1–4 mode 解释为病例轴。

### 5.5 persistent path-mode 模型未通过

34 人 × 3 seeds × 5 conditions，共 510 runs：

- heldout next-set prediction 能利用 prefix；
- participation 与完整 rank distribution 的联合生成门失败；
- graph lesion 与 mode-collapse necessity 均失败；
- path posterior 不可辨识。

因此：

> 局部 prefix 可学，但离散、event-persistent path identity 不是当前数据支持的科学对象。

这也说明 RNN 不应退化为一个更复杂的 A/B 分类器。

### 5.6 symmetric-axis 与 competitive structured RNN 未通过

v2.2.1 的非负线性单状态 symmetric operator：

- Markov 21/22 优于 node bias；
- full 和 isotropic 各仅 1/22 优于 node bias；
- 说明 structural scaffold 不能直接等同于 observed next-contact operator。

transition decomposition 随后确认数据中存在：

- 跨局部几何的 transition signal；
- symmetric residual；
- multi-step ordered history；
- 小的 axis/source-conditioned residual。

这足以许可测试 v2.3，但不是机制确认。

v2.3 competitive structured RNN：

| Claim | 结果 |
|---|---|
| full vs node predictive adequacy | PASS，22/22 |
| ordered history vs instantaneous | PASS，18/22 |
| delayed competition | FAIL |
| matched physical-axis increment | FAIL |
| source-conditioned direction | FAIL |

模型约恢复 ordered-history Markov cohort-median benefit 的 58%，但没有恢复剩余 transition
structure。

最终结论不是“患者没有病理轴”，而是：

> 当前 tested observation mappings 没有把 empirical transition signal 唯一归因到
> symmetric axis、competition 或 source term。

### 5.7 RNN 没有自动恢复 A/B 病理轴

axis-positive 9 人中，RNN 选轴：

- seed-to-seed cosine 近 1，说明优化可重复；
- 与冻结 A/B shared axis 的 alignment margin 中位为 -0.205；
- 仅 2/9 为正；
- 95% CI 跨 0。

高 seed 稳定性只表示模型反复选到同一方向，不表示该方向是数据中的病理轴。A/B 轴也不是
所有患者的金标准，但当前结果同样没有给出独立的方向特异性证据。

22 人双侧 source 审计中，从轴两端到内部的 displacement 很常见，但 selected direction
没有优于其他候选方向，axis/source NLL benefit 也没有在两侧共同成立。它更容易由 contact
cloud 边界和 inward regression 解释。

### 5.8 early-ictal static contact correspondence 存在，但不是 GRU 特异

最终 fixed target：

- Epilepsiae strict clinical-onset；
- 16 人、106 seizures；
- clinical onset 后 `[0,10] s`；
- 1–150 Hz baseline-normalized contact energy。

该 target 已被前序 v2.5 使用，因此后续所有统计都是同数据集内部验证，不是独立复制。

固定 participation field 的正方向 signed correspondence 未成立：

| 指标 | n | 中位数 | 正值患者 | \(P\) |
|---|---:|---:|---:|---:|
| full GRU signed all-contact margin | 16 | 0.243 | 11/16 | 0.126 |

方向无关的 morphology correspondence 成立：

| field / null | n | 中位 margin | 正值患者 | \(P\) |
|---|---:|---:|---:|---:|
| full GRU / all-contact | 16 | 0.215 | 14/16 | 0.000153 |
| raw train80 / all-contact | 16 | 0.196 | 13/16 | 0.000656 |
| raw train80 / within-shaft | 16 | 0.079 | 12/16 | 0.00131 |
| raw train80 / geometry-smooth | 13 | 0.100 | 9/13 | 0.0341 |

full GRU 与 best regularized field 的 contact-wise Spearman 中位为 0.941。full GRU
相对 best regularized 和 rank-shuffle 的静态增量均未成立。

因此可以说：

> 间期病理 contact topography 与发作早期 broadband energy 在患者内具有相同或反向的
> 空间组织。

不能说：

> RNN 按正确正方向预测了哪些触点在发作早期能量更高，或 ordered history 被发作期重用。

## 6. 当前 RNN 结构是否合理

### 6.1 作为自监督 information probe：合理

普通 GRU 与 H1/H2/H3 的训练任务直接对应数据问题：

- 输入是真实 contact rank-set prefix；
- label 是下一 rank set/STOP；
- heldout20 不参与训练；
- matched rank-shuffle 排除了 participation-set 和容量解释；
- H1/H2/H3 明确定位了有效历史深度。

因此它足以支持“间期事件有短时序结构”。

### 6.2 作为最终动力学模型：不合理

full-history GRU：

- 没有超过 H3；
- 没有超过 strongest nonrecurrent prefix model；
- hidden state 低维但不具 ordered-specific 唯一性；
- 跨状态静态结果不具 GRU-specific increment。

所以 full GRU 不能被写成患者病理动力学本体。

### 6.3 当前结构化模型：工程合理，科学先验未获支持

path、axis、competition 和 source 模型均有清楚消融与停止规则，工程上可以验收。但数据
没有支持其关键结构必要性。

这类失败应保留，因为它限定了论文能说到哪里；不应继续通过增加 seeds、改 loss 或重新选
患者追阳性。

### 6.4 下一版是否还要做新 RNN

当前不建议继续新 RNN 训练。

如果未来出现真正独立的新患者或新 clinical-onset 队列：

- H3 应作为唯一 accepted sequence reference；
- 新可解释模型只需表达最近 2–3 步；
- 必须先超过 H3 或 strongest nonrecurrent baseline；
- 必须在 target-blind 条件下冻结 representation；
- 然后才能做独立 early-ictal confirmation。

在当前重复使用的 16 人 target 上继续扩模型，没有独立科学增量。

## 7. 对论文核心科学目标是否偏移

### 7.1 没有偏移的部分

- 输入始终是原始 SEEG 的 contact-rank 简化表示；
- 不以 A/B 为监督标签；
- 不使用 IEI；
- 间期阶段始终自监督；
- early-ictal 端始终使用 clinical onset 后静态 1–150 Hz energy field；
- 主要统计以患者为单位。

### 7.2 主动收窄的部分

原始目标希望从 RNN 内部解释 A/B 双向轴及其发作期重用。当前数据只支持：

- 更宽的患者特异 contact scaffold；
- 最近 2–3 步的有序历史；
- 同数据集上的方向无关跨状态 morphology。

这不是把任务改成容易阳性的分类问题，而是依据正式对照删除未被支持的机制层。

### 7.3 与论文现有经验结果的关系

RNN 阴性不否定已有真实数据中的 A/B 模板或 empirical early-ictal field 结果，因为：

- 模型 failure 只针对 tested parameterization；
- A/B 不是所有患者的金标准；
- empirical field 结果不依赖 RNN 成功。

RNN 新增的可靠信息是：

1. contact participation scaffold 可从独立间期事件稳定估计；
2. 事件内部真实顺序含有最近 2–3 步的增量信息；
3. 多种直观机制化 RNN 不能据此被自动确认为患者病理轴；
4. 跨状态对应首先是 static contact morphology，而不是 sequence replay。

## 8. 论文中的建议定位

### 8.1 建议保留

作为 Supplementary computational result，回答三件事：

1. 间期 rank events 的患者特异 participation scaffold 是否稳定；
2. 下一 contact 预测到底需要多长历史；
3. 这一 scaffold 与 early-ictal energy 的 correspondence 是否需要 GRU/axis/path。

### 8.2 不建议作为 Figure 6 中心机制

不应把 structured-axis RNN、path-mode RNN 或 low-rank mode 画成论文主机制模型，因为
其必要性门均失败。

若主文必须保留一处计算增量，最安全的一句话是：

> Interictal group events defined a reproducible patient-specific contact
> scaffold and contained order-dependent information over the latest two to
> three recruitment steps; however, neither unbounded recurrent history nor
> the tested path- and axis-constrained dynamics were required to explain the
> cross-state static contact morphology.

### 8.3 图的使用

当前两张 canonical 图分工：

1. `topic5_scaffold_reliability_history_necessity_v0_1`：
   - static reliability；
   - event-count saturation；
   - H1/H2/H3/full history；
   - matched H3 rank-shuffle。
2. `fig6_static_contact_topography`：
   - fixed early-ictal static target；
   - signed vs sign-free correspondence；
   - regularized baseline、teacher/free 与 confound；
   - GRU-specific static increment 的阴性边界。

旧 path-mode、v2.2、v2.3、axis/read-back 和 internal-state 图保留为 provenance 或方法
补充，不再并列竞争“当前 Figure 6”。

## 9. P0 / P1 关键问题

### P0

无。所有纳入最终结论的正式产物均存在，目标读取状态与各阶段合同一致。

### P1-1：early-ictal target 不是独立验证

同一 16 人/106 seizures 已被多轮读取。当前只能写 same-dataset internal validation。

### P1-2：sign-free 不是正方向预测

`abs(rho)` 同时接受相同和反向 contact ordering，不能写成“高间期参与必然对应高发作
能量”。

### P1-3：短历史阳性不能升级成 axis/path mechanism

H3 比 H2 好只说明最近三步含信息，不说明该信息由某一物理轴、A/B path 或 E/I state
产生。

### P1-4：low-rank 阴性不是干净的低维机制否定

rank 0 仍保留多条 diagonal memory channels，因此 low-rank sweep 只能作 sensitivity。

### P1-5：已有模型不再调参

继续调 full GRU、path count、axis kernel、competition decay 或 source term 会破坏当前
冻结边界，且没有新的独立 target 可验证。

## 10. 最小收口路线

1. 以本报告和机器可读 `FINAL_ACCEPTANCE.json` 作为 RNN 总入口；
2. manuscript-facing 只保留当前 static-contact + bounded-history 版本；
3. 旧模型文稿全部标 historical/provenance；
4. 不再启动当前 cohort 上的新 early-ictal RNN；
5. 等待真正独立 clinical-onset patient cohort 后，使用冻结 H3 和 fixed static field
   做确认。

## 11. 最终验收

```text
execution_integrity:
  PASS

static_interictal_contact_scaffold:
  SUPPORTED_TARGET_BLIND

short_ordered_history_H2_H3:
  SUPPORTED_TARGET_BLIND

unbounded_full_history:
  NOT_SUPPORTED

positive_low_rank_modes:
  NOT_SUPPORTED_BY_TESTED_PARAMETERIZATION

persistent_path_mode:
  NOT_SUPPORTED

physical_axis_competition_source_mechanism:
  NOT_SUPPORTED_BY_CURRENT_MODEL_FAMILIES

early_ictal_sign_free_static_morphology:
  SUPPORTED_WITHIN_REUSED_TARGET_DATASET

fixed_positive_cross_state_direction:
  NOT_ESTABLISHED

GRU_specific_static_transfer:
  NOT_ESTABLISHED

dynamic_seizure_prediction_or_replay:
  NOT_TESTED_AND_NOT_ALLOWED

paper_tier:
  SUPPLEMENTARY_BOUNDED_COMPUTATIONAL_RESULT
```

整体状态：

```text
ACCEPTED_AS_BOUNDED_SUPPLEMENTARY_COMPUTATIONAL_RESULT
```

## 12. 主要产物

- 总机器验收：
  `results/topic5_rnn_overall_acceptance/FINAL_ACCEPTANCE.json`
- 总验收构建脚本：
  `scripts/build_topic5_rnn_overall_acceptance.py`
- 本报告：
  `docs/archive/topic5/rnn_overall_integrated_acceptance_2026-07-28.md`
- 当前 manuscript-facing source：
  `docs/paper-draft/figure6_static_contact_topography_bounded_result.md`
- bounded-history 图：
  `results/topic5_interictal_scaffold_reliability_history_necessity/figures/`
- static cross-state 图：
  `results/paper-ready-figure/fig6_static_contact_topography/figures/`
- bounded-history 分报告：
  `docs/archive/topic5/interictal_scaffold_reliability_history_necessity_v0_1_report_2026-07-28.md`
- fixed static 分报告：
  `docs/archive/topic5/static_scaffold_fixed_readout_validation_v0_1_report_2026-07-28.md`
- path-mode formal：
  `docs/archive/topic5/persistent_path_mode_rnn_formal_result_2026-07-26.md`
- v2.3 structured formal：
  `docs/archive/topic5/symmetric_axis_competitive_propagation_rnn_v2_3_result_2026-07-27.md`
- internal-state reduction：
  `docs/archive/topic5/rnn_internal_state_reduction_v0_1_report_2026-07-28.md`
