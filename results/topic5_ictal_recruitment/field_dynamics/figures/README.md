# 发作内 field 动力学 pilot — 图说明

6 个几何干净 ECoG（`epilepsiae_{442,548,583,384,958,1084}`，narrow 基底）逐次发作的 field 指标随发作
进程（onset→offset）+ 终止前后的变化。**纯描述性 subject-level pilot，不下 cohort/机制结论。**

触点分四组（间期态一次定）：`source_core`（红=两间期模板各自最早传播触点的 compact core，假设病灶）、
`axial_mid`（金=两 source 之间、贴轴的走廊，**主检验对象**）、`non_axial`（蓝=离轴远）、`axis_end_noncore`
（灰=贴轴但靠端）。白/黑虚线 = source-A→source-B 轴。每窗算：场-轴对齐 maxAB、各组正质量占比
`positive_mass_share`、同步 median pairwise corr、场梯度方向漂移。band 默认 broadband（HFA 同存于 CSV）。

### \<ds\>_progress.png
横轴 = 发作进程 0→100%（每发作压到同一尺度）；灰细线 = 每次发作，黑粗线 = 分箱 median。四子图：场-轴
对齐(maxAB)、轴向中段正质量占比、非轴向正质量占比、同步。仅用 `band=bb` ∧ `ictal_fraction≥0.5` 的 onset 窗。
**关注点**：轴向中段占比是否随进程**下降**、非轴向是否**上升**、同步是否上升、对齐是否塌——以及这些是否
跨发作一致（灰线是否抱团）。

### \<ds\>_offset.png
按发作终止（eeg offset=0，红虚线）对齐的窗 `[-60,-30,-10,0,+30]s`，散点=各发作各窗、黑线=median；已排除
左缘早于 onset 的短发作窗（`pre_onset_overlap`）。
**关注点**：终止前 10–30s 是否有指标突变；offset 后（+30s）是否回落。不预设终止动力学，仅看有无变化。

### \<ds\>_seizure_heatmap.png
行=每次合格发作，列=进程分箱（0–100%），色值=轴向中段正质量占比。
**关注点**：同一 subject 不同发作的轴向中段走势是否一致（行与行是否相似），还是发作间高度异质。

### \<ds\>_geometry_qc.png
间期态四分区着色 + source-A→source-B 轴线（几何健康检查，无 field 背景）。标题含两侧是否单点
（`uncertA/B`）、两 source top2 距离、是否退化轴（`degen`）。
**关注点**：轴线是否真的穿过两 source、走廊（金）是否落在两 source 之间、是否退化（1084 两 source 重合
=单红点无轴=天然负控）、source 是否散到只能取单点（384/958）。

### per_seizure/\<ds\>/\<ds\>_szN.png
单次发作（仅 `eeg_duration≥40s`）一张 8-panel composite：上排 4 个进程快照（0/33/66/100% 的活动场，
viridis=窗内 mean robust-z 的 within-subject rank，叠四分区 + 轴）；下排 4 panel：各指标随进程、四组正质量
占比随进程、终止对齐 zoom、同步随进程。
**关注点**：单次发作内 field 是否保持同一轴还是方向漂移；轴向走廊 vs 非轴向的此消彼长；终止前后是否突变。

---
**已知边界（pilot 暴露的限制，非 bug）**：
- `source_core 恒亮` 前提只对 442/548 成立；**384 的间期 source 在发作时是冷的**（热区在另一根杆），958 全场普热，
  583 仅 7 通道。即"间期传播最早触点"≠"发作最热区"并非普适。
- 两 source **靠得近**时（958/单点 fallback）轴很短 → 走廊只剩极少触点（958 `axial_mid`=1、583=0）→ 这些 subject
  的"轴向中段"检验欠功率。axis 真正跨阵列、走廊有料的是 442/548。
- 1084 = 退化轴负控（两 source 重合，`degen=True`），不进任何主结论。
