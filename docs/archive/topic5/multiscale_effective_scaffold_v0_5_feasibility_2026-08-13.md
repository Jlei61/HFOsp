# Topic 5.1 v0.5 多尺度有效传播 scaffold：target-free feasibility

日期：2026-08-13
状态：**历史 feasibility 记录，已被正式 full-parent builder supersede。** 本文的 5 位定向审计和
`26 patients / 40 fits` 只记录启动前可行性，不再是当前分母。正式 target-free builder 随后自动扫描
全部 34 位 K=2 parent，冻结为 **28 patients / 42 fits**；正式训练按终审合同扩展并完成
**531/531 units**。本文原始数值保留作 provenance，不得在后续报告中当作当前结果引用。

## 1. 为什么做这个 audit

v0.3 full-tissue spatial RNN 只有 21 人/31 fits，和 Figure 3 的 17 人 early-ictal parent cohort 相交为 12 人。缺少的 E1077、E1096、E1125、E139、E635 不是没有间期 rank 数据或二维 plane，而是 upstream cache builder 把 `joint contacts <8` 直接挡掉。

v0.5 不把这一工程阈值当作生物学排除。先在完全 target-free 条件下检查把下限降到 6 后，这五位是否仍满足同一个 full-tissue 计算合同。

## 2. 结果

- 5/5 patients 可恢复；
- 9/9 fits 可构造（E139 shared；其余 own-A/B）；
- joint contacts 为 6–7；
- 每 fit 64–79 tissue nodes；
- zero-H nodes 为 16–52，比例 0.232–0.658；
- local backbone 9/9 单一 strong component，最低 in/out degree 均 >=5；
- extra-local 与 nonlocal candidate pools 9/9 均超过 added-edge budget；
- `event_lag_raw` 5/5 完整，参与 contact 的有限比例为 1.0；
- 候选事件量为 5,111–140,337/患者，事件内 lag span 中位约 18.4–33.1 ms。

所以当前 projected spatial cohort 是：

```text
v0.3: 21 patients / 31 fits
v0.5: 26 patients / 40 fits
```

这不是正式 cohort census。后续 builder 必须对全部 masked-rank K=2 parent 自动应用
`min_joint_contacts=6`，逐患者输出 inclusion/exclusion reason，并证明没有其他满足规则者被遗漏。
只有 cache builder、训练 smoke 和全量 attrition audit 都通过后，Figure 3 的 17 人/167 seizures
才可正式写成拥有 spatial RNN field；不能只凭这份 5 人定向 audit 改分母。

## 3. 时间变量边界

`event_lag_raw` 来自 interictal `lagPatRaw`，表示单个 HFO 群体事件内 contact 的相对谱质量中心/到达时间代理。它可以用于检查“远距离 contact 是否比 local-wave baseline 预期更早”，但不能写成：

- 临床 ictal recruitment time；
- MUA recruitment；
- 真实轴突传导时延；
- inter-event interval。

v0.5 的 nonlocality index 必须以严格 cross-fitted local-wave baseline 和 out-of-fold residual 构造。
若数据充分但非负约束 slope 落在 0，标为 `LOCAL_WAVE_UNSUPPORTED` 并保留；只有距离范围、finite
latency、事件数不足或设计矩阵退化才标 `NOT_IDENTIFIABLE`。

## 4. 当前允许的下一步

可以实现并测试：

- 40-fit cache builder；
- latency sidecar；
- prefix-preserving suffix order control；
- prefix-template baseline；
- L2 macro matching、L1/L3 candidate-capacity matching；
- trajectory effective-gain audit；
- patient nonlocality index；
- 从头训练的 macro-matched random-nonlocal `L2m` control。

本文阶段曾估计正式补训预算至少 321 units；full-parent census 与严格重训/复用规则冻结后，正式
launch manifest 更新为 531 units 并已全部完成。frozen rewiring 仍只作 perturbation，不能替代
从头训练的 L2m。
Early-target unseal 等待 v0.5 spec/plan 终审锁定。E1 latency RNN、E2 state gain 和 E3
susceptibility factorial 均不在本阶段直接启动。

## 5. 产物

- `results/topic5_multiscale_effective_scaffold_v0_5/FEASIBILITY_AUDIT.json`；
- `RECOVERY_FIT_AUDIT.csv`；
- `FEASIBILITY_AUDIT_COMPLETE.json`；
- `figures/stage_a_v0_5_spatial_cohort_recovery.{png,pdf}`；
- `figures/README.md`；
- 3 项新增单测。
