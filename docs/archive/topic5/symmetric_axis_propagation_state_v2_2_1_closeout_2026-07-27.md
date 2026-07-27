# Symmetric-axis propagation-state RNN v2.2.1 closeout

> 日期：2026-07-27  
> 状态：按预注册 stop rule 完成并冻结  
> 结果根：`results/topic5_symmetric_axis_propagation_state_v2_2/`

## 1. 一句话结论

患者内间期 contact-rank 序列含有稳定的一阶 transition information，但把近似对称
的物理 scaffold 直接写成非负、线性、单状态的 next-contact propagation operator
不能捕获该信息；这否定的是当前观测映射，不是患者存在共享病理轴的可能性。

## 2. 执行状态

- 科学合同执行至预注册停止点：100%。
- 34 人 input inventory、3 人 development、31 人 sequence sensitivity、22 人
  physical-axis formal 均按冻结合同完成。
- Claim 1 predictive adequacy：`FAIL`。
- Claim 2 next-set：`FAIL`。
- Claim 2 future first-arrival：`FAIL`。
- Claim 3 random-axis specificity：`LOCKED_NOT_RUN`。
- Claim 4 shared scaffold：`LOCKED_NOT_RUN`。
- early-ictal transfer：
  `BLOCKED_INTERICTAL_GATE_AND_MISSING_SOURCE_METADATA`。
- target values 始终未读取。

Claim 3/4 没有运行是 stop rule 的正确执行，不是科学实验遗漏。

## 3. 66/66 训练审计

`formal/TRAINER_EPOCH_AUDIT.json` 已重建：

- development models audited：18/18；
- formal resolved configs audited：66/66；
- formal epochs：全部 200；
- heldout20 未用于 early stopping；
- target values read：false。

## 4. 同一 22 人评分合同下的模型比较

所有结果均先在患者内聚合，full/isotropic 再在 3 seeds 内取中位数。

| 模型相对 node-bias | 患者中位 NLL benefit | 95% bootstrap CI | 正效应患者 | one-sided Wilcoxon |
|---|---:|---:|---:|---:|
| empirical first-order Markov | +0.01078 | +0.00661, +0.01546 | 21/22 | 4.77e-7 |
| local-isotropic propagation | -0.01661 | -0.03270, -0.00875 | 1/22 | ≈1 |
| symmetric-axis propagation | -0.01657 | -0.03266, -0.00888 | 1/22 | ≈1 |

Markov 相对 full 的患者中位 benefit 为 +0.02897（22/22 为正，
one-sided Wilcoxon 2.38e-7）；相对 isotropic 为 +0.02871（22/22 为正，
2.38e-7）。

这正式回答了上一轮缺失的比较：失败不是仅由 axis term 引起，而是当前整个
nonnegative linear propagation-state model class 没有抓住 Markov 所含的 transition
signal。

## 5. 评分合同复算

`closeout_v2_2_1/SCORING_CONTRACT_AUDIT.json`：

- node-control NLL 最大复算误差：8.33e-17；
- full/isotropic checkpoint NLL 最大复算误差：5.87e-8；
- event、prefix、eligible contacts、tie-set likelihood 和 event-first normalization
  已核对一致；
- node/Markov 共享同一个 LOSO STOP；
- full/isotropic 共享同一 propagation-state/STOP model form。

full/isotropic 的 STOP 含 propagation-drive 项，而 node/Markov 使用冻结的
`c0 + c_n * seen_fraction` control。总 NLL 可以比较，但差异归因必须结合下面的
calibration decomposition，不能只看总分。

## 6. Calibration 解剖

冻结 checkpoint 的全部 heldout prefixes 显示：

- observed nonterminal next-set size 均值：1.00；
- full predicted conditional next-set size：1.655；
- local-isotropic：1.654；
- full 的 mean positive-contact NLL：1.960；
- mean negative-contact NLL：1.127；
- local-isotropic 对应为 1.958 和 1.126。

按 prefix step：

| Step | observed size | full predicted size | full negative-contact NLL |
|---|---:|---:|---:|
| 1 | 1.00 | 1.858 | 1.453 |
| 2 | 1.00 | 1.753 | 1.288 |
| 3 | 1.00 | 1.639 | 1.102 |
| 4+ | 1.00 | 1.368 | 0.665 |

因此当前模型的一个明确失配是：独立 Bernoulli hazards 在早期 prefix 同时抬高太多
eligible contacts，系统性高估下一 rank set 大小。full 与 isotropic 的校准几乎
完全重合，说明该失败不能归因于 axis 增量本身。

## 7. Operator identifiability

22 人、3 seeds 的患者内中位 descriptive audit：

- local/axis kernel Frobenius cosine：中位 0.979
  （IQR 0.963–0.998）；
- full/isotropic operator relative distance：中位 0.084
  （IQR 0.044–0.124）；
- learned axis 与 contact-cloud PCA1 的 absolute cosine：中位 0.869；
- 将 learned gamma 设为 0 后的 operator relative change：中位 0.083；
- gamma→0 后 heldout eligible-contact logit mean absolute change：中位仅 0.0052。

这说明许多患者中 local 与 axis kernel 高度共线，learned axis 又常接近植入点云
PCA1。跨 seed cosine 接近 1 只能说明优化重复；结合上面的共线性和很小的实际 logit
改变，不能称为 physical-axis identifiability。

## 8. Clinical-onset source 阻断

当前 metadata inventory 为 13 人、71 次发作，但 exact per-seizure
clinical-onset contact sets 仍为 0。已建立独立盲法标注 registry：

`results/topic5_clinical_onset_source_annotation_v0_1/`

它禁止使用 SOZ、患者级 focus、A/B source 或 energy-top contacts 补位。只有双人一致
或专家裁决且 exact contact join 成功的发作可进入 primary transfer。在人工标注完成
前，early-ictal bridge 继续保持 blocked。

## 9. 图

`closeout_v2_2_1/figures/v2_2_1_closeout_diagnostics.{png,pdf}`

- A：Markov / isotropic / axis 相对 node-bias；
- B：observed vs predicted next-set cardinality；
- C：kernel 共线性与有效 operator distance；
- D：learned axis–PCA1 与 gamma→0 的实际 heldout logit change。

该图是 bounded-negative supplementary diagnostic，不进入主文 Figure 6。

## 10. 安全与禁止口径

### 可以写

> Patient-specific interictal rank sequences contained robust first-order
> transition information. However, a nonnegative linear symmetric physical
> scaffold used directly as the sole next-contact propagation operator failed
> to capture this information.

### 不可以写

- 人体不存在共享病理轴；
- SNN 的近对称 scaffold 机制被否定；
- random-axis 或 cross-direction generalization 已失败；
- early-ictal transfer 阴性；
- seed 稳定证明了病理轴可辨识。

## 11. 下一步

不再调 v2.2 超参数。下一阶段按独立合同
`Interictal transition signal decomposition v0.1` 分解 Markov signal：

1. same-shaft 与欧氏局部距离；
2. symmetric 与 skew/directed residual；
3. 控制局部几何后的 physical-axis residual；
4. observed-source-conditioned directional component；
5. last-rank 与 ordered multi-step history。

只有跨 shaft、source-conditioned axis 和 multi-step history 同时在 heldout
患者中成立，才允许设计 v2.3 recurrent model。

