# Continuous marked-state R0.1 正式验收记录

**验收日期：** 2026-08-24
**验收对象：** `results/epi_prssm/continuous_marked_state/r0_1/`
**验收等级：** development closeout / evidence package / R1 技术基础
**不是：** 已成功建立的连续生理隐状态模型，也不是 H1–H3 的最终验收。

## 1. 验收结论

R0.1 **通过限定范围验收并冻结**。本次在当前环境独立复跑：

- 27/27 项相关测试通过；
- `FINAL_PACKAGE_AUDIT.json` 为 `all_pass=true`；
- 15 个证据模块完整；
- 正式检验分区未打开；
- 本次复核 source package SHA-256：`48173c8e77f246d08bd75b93e25bac3191ec9266e177d3fd34cd6ebc68b09458`。

旧 `CURRENT_HANDOFF.md` 中的 `3121cd...a0692` 是较早交付快照的哈希，不再作为当前验收哈希。两者不得混用。

## 2. 科学上正式接受的内容

1. **H2a 是当前最强的 development predictive evidence。** 过去 IED 的空间表达包含可预测后续 IED repertoire 的信息；正确时刻状态、same-prefix continuation、较长事件跨度预测和患者特异图接线共同支持这一点。最强证据仍主要来自旧 event-history state，尚未由最终 raw-informed T1 复现。
2. **H1 只支持 predictive filter 层。** 连续背景观测可形成时刻特异的预测代码；当前尚未建立稳定的 controlled generative state，更未建立 autonomous generative model。
3. **H2b 只接受为探索性关联。** 361 次发作主层的冻结 readout 有偏移，但 203 次高可观测敏感性更弱、连续梯度不清楚、亚型交互未成立。
4. **H3-S0 接受为候选机制筛查。** 近期约 25–200 次 IED 的累积历史主要携带下一事件 STOP/termination/extent 信息；它不是当前单事件效应，也没有显示稳定 timing、contact order 或 same-prefix route shaping。
5. **没有识别出真实分钟时钟。** physical clock 未稳定胜过 event-count clock；事件数尺度是操作性 observation-count memory，不是生理时间常数。
6. **H3a generator edge 与 H3b 均未成立。** 当前 exposure 结果可能来自共同未观测状态，不能写成 IED 已塑造未来脑状态或发作模式。

## 3. 正式拒绝的措辞

- 已发现癫痫易感状态；
- raw Transformer 已被选为正确 observer；
- IED 已被证明塑造脑网络状态；
- 存在 3–30 分钟生理恢复时间常数；
- H3 改变了传播路线；
- 发作亚型对应不同 latent attractor；
- correction-off 阴性否定持续状态存在。

## 4. R0.1 停止项

以下工作在 R0.1 永久停止：

- 增加 H3-S0 时间尺度、exposure definition 或筛查网格；
- 将 R0.2 频谱预测扩到 34 人；
- 继续扫描 state dimension、Conformer depth 或频谱 horizon；
- 在最终 likelihood 和 observer 未正确完成前启动千级 T2 矩阵；
- 打开正式检验分区。

## 5. 下一版本的唯一入口

R1 必须先完成：

1. recorded-interval point-process timing likelihood；
2. exact tied-group unordered without-replacement sequential mark likelihood；
3. 确定性的显式 event-history baseline，不允许 free history RNN；
4. 在同一个 exact IED objective 上比较 explicit observer 与 zero-initialized raw-residual observer；
5. 通过合成恢复和小型 development pilot 后，才启动 6 人 T1；
6. T1 成立后才进入小规模 `N=100` T2 generator edge。

R1 的 H1 命名分层固定为：`predictive filter`、`time-specific predictive state estimate`、`controlled generative state`、`autonomous generative model`。`event-observed, raw-correction-off` 必须与 fully generative rollout 分开报告。

## 6. 权威证据

- `results/epi_prssm/continuous_marked_state/r0_1/manifests/FINAL_PACKAGE_AUDIT.json`
- `results/epi_prssm/continuous_marked_state/r0_1/manifests/HYPOTHESIS_EVIDENCE_CARD.json`
- `results/epi_prssm/continuous_marked_state/r0_1/reports/plain_report_2026-08-24.md`
- `results/epi_prssm/continuous_marked_state/r0_1/reports/technical_report_2026-08-24.md`

本文件是 R0.1 验收边界的唯一权威入口；旧 spec、plan 和 handoff 中与本文件冲突的“下一步”表述，以本文件为准。
