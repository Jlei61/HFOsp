# Fig3-B：间期时序场与发作早期能量场

### epilepsiae_1146_seizure_15_interictal_ictal_shared_field.png / .pdf

E1146 的冻结 shared plane 配对图。左侧为红色语义标题的 TA early-to-late timing field；右侧为 E1146 全部可用发作中 `shared_a_signed` 最高的 seizure 15，显示 clinical onset `0–10 s` 内 broadband `1–150 Hz` baseline-normalized power，使用 `magma_r`，且不做 rank 或 sign flip。两幅图严格复用同一 contact order、shared TA axis、transverse sign、TA support 与同一个 6 mm display kernel。

**关注点**：这是一个 representative-subject shared-field readout，用于连接间期传播轴与同次发作早期能量分布；不能单独解释为 replay、因果机制或 cohort 结论。

### epilepsiae_1146_seizure_15_interictal_ictal_shared_field_metadata.json

记录 raw seizure、临床窗、远端 baseline、频谱参数、冻结 fingerprint、A/B 匹配分数、逐触点原始值与显示归一化。重画时必须先通过 checkpoint score parity，不能从 ictal 值重拟合轴、平面、support 或 kernel。

**关注点**：两条 colorbar 分别恢复为真实 propagation rank 与 robust-z 数值；`magma_r` 令高 broadband power 为深色，与左图“早期为深色”的视觉方向一致。右图不是 contact rank，也不改变当前 maxAB 科学统计。
