# FCXR-LC6A 患者轴 E→I surround：canonical-seed bounded-negative

日期：2026-08-15

## 1. 一句话结论

我们确实把患者轴方向的有效两跳抑制周边做宽了，但在当前 legacy Z/H substrate 上，它没有创造一个可停留的中间高态：五条自然轨迹都从 returning IED 进入，随后继续升级到注册 saturation。

正式标签：`CANONICAL_SEED_AXIAL_REACH_FAMILY_NO_CARRIER`。

## 2. LC5v2.1 右删失格收口

唯一右删失格在 23 s onset；从 25 s exact state 继续后，于 27 s 达到 405.9 Hz，D=0.573、H=25.763，判 `ESCALATING_SATURATION`，没有 offset。25--26 s 曾在 reducer 路径失败，因此该段输入 hash 和诊断 trace 不可用；但 exact checkpoint 被恢复，28 个 classifier snapshot bundles 全部重放，注册 saturation 位于完整记录的 26--27 s 段。这个缺口限制第一续跑秒的细粒度诊断，不改变终局 saturation 裁决。

## 3. 这轮真正改了什么

只改变代码 `IE`，即生物学 E→I 的患者轴 reach；EE、I→E、I→I、权重、Z/H、两个 core、噪声与所有慢机制保持冻结。graph-only two-hop 审计显示 q 从 0.935 增至 1.490，surround/center 质量比从 1.73 增至 3.07，所以这不是“图没改到位”的假阴性。

改造是轴向的：垂轴 marginal 宽度从 0.9989 mm 只走到 0.9945 mm，两跳垂轴宽度同样保持在 1.414 mm 附近；source out-degree 的 CV/q95/q99/interior-edge 比值相对 C0 的偏差都在 ±10% 容差内（最大 −5.8%），对轴向与垂轴位置的 Spearman 偏移都 ≤0.005。spec 要求的垂轴 confound 与 out-degree 容差都通过。

## 4. 实际跑出来的不是五级阶梯

spec 预期 C0 的 `q_parallel^marginal` 约 0.66，实测是 0.934。因此预注册的 Q1=1.00 这一档落在了 C1 自己的 ±0.05 同 q 容差里。实际结构是：C0/C1/Q1 三张同 q 但连线微状态不同的图（互换边比例 20–21%），加上 Q2、Q3 两个真正的 reach 档。这三张同 q 图就是这一轮唯一的内建对照。

## 5. 五条自然轨迹

| 条件 | q(two-hop) | 同 q 对照带内 | onset | onset 前 IED | 全局峰值(100 ms) | baseline 代价 | 结局 |
|---|---:|---|---:|---:|---:|---|---|
| C0 | 0.935 | 是 | 11.0 s | 29 | 408.2 Hz | 无 | saturation |
| C1 | 0.959 | 是 | 10.0 s | 28 | 354.1 Hz | 无 | saturation |
| Q1 | 0.979 | 是 | 13.0 s | 34 | 361.0 Hz | 无 | saturation |
| Q2 | 1.285 | 否 | 12.0 s | 37 | 398.8 Hz | 有 | saturation |
| Q3 | 1.490 | 否 | 6.0 s | 21 | 395.7 Hz | 有 | saturation |

所有条件的 active area 都到 400 mm²，也就是 20×20 mm 全片 100%；近 refractory ceiling 的细胞比例很低（最大 0.23%，注册线 5%），但全局 1 s 均值仍跨过 250 Hz 的注册 saturation。因此这轮不是“有限面积内的健康 carrier”，而是全片 escalating high state。

**进入时刻**：三张同 q 图的 onset 已经自己散在 10–13 s。Q2 的 12 s 在这个带内，Q3 的 6 s 在带外。可以说的只有“最强那一档把进入提前到对照带之外”，不能说“Q1/Q2 推迟了进入”。

**进入之后**：五臂都在进入后 5–6 s 抵达各自的停机点（越过注册饱和线后再跑 1 s 停），对齐到进入时刻后逐秒 rate 的跨臂离散度最大 2.46×，而仅三张同 q 图之间就已经有 2.36×。换句话说，把两跳抑制周边加宽 60% 只改变了什么时候开始升级，没有改变升级本身。

**轴向 D-halo / front-speed 不可用**：这三个读数在本 substrate 上没有进入后的动态范围。`D_halo_lead_mm` 只在“第一个越过局部 rate 阈值的 1 s 窗”里有限，其数值由那一窗里已经点亮了多少面积决定（C1 10%→1.24 mm、Q2 7%→1.21 mm、Q1 17%→1.19 mm、Q3 70%→0.57 mm、C0 39%→0.01 mm），与 q 完全不成序；之后每一窗都 <0.03 mm，即不到一个 0.625 mm 的空间格。`D_halo_width` 在所有臂所有秒（含进入前）都是 18.1 mm，就是整片的宽度。因此不能写“更宽 E→I 加速了 D 耗竭 halo”。详见 `run_manifest.json` 的 `post_hoc_corrections`。

## 6. 短功能探针

八个探针位置全部满足预锁的亚阈值合同：分窗放电率差在每个空间格点上精确为零。其中五个位置的远场（|x|>1.5 mm）也精确为零——没有任何脉冲改变过。另外三个（C1:neutral_axis, Q1:neutral_axis, Q3:neutral_axis）远场非零，说明有脉冲在窗内挪了位置；它们正是唯一出现晚窗“零交叉”的三个位置。这三个位置与 reach 不成序（C0 与 Q2 为零，Q3 换到 core-adjacent 位点也为零），所以这一轮的配对探针**不支持任何关于 reach 依赖功能周边的说法**，只支持“探针确实是亚阈值的、直接足迹只落在被刺激的小块上、中心偏转由抑制驱动力而非被招募的兴奋构成”。

另外，探针基线是 C0 在 2100 ms 的精确状态：Z=1、U=M=0、X=1 精确成立，但 H 已经涨到均值 0.453，不是 spec 字面写的 H=0。

## 7. gain fork 的意义与边界

按预注册规则（boundedness margin 最大 + 表型最不同）选择 C0 与 Q2。必须注明这些 fork 采样的是什么状态：onset+2 s 时 C0 前 1 s 为 34.3 Hz、Q2 为 52.1 Hz，分别只有各自 1 s 峰值的 9% 和 15%——这是升级斜坡，不是高态。只有 C0 的 onset+6 s（前 1 s 177.6 Hz）真正采到了高 rate 状态，它给出非零响应。因此“C0 早期响应为零”只能读作“斜坡早期对这个弱输入没有可测响应”，不能读作“高态是惰性的”。

`high_state_dwell_s` 也需要注意：它按 `总时长 − onset` 计算，不是实测的高态时长，所以 spec §10.1 第 2 条的“高态至少持续 5 s”在实现里是观察时长而非高态时长。本轮五臂都因其他条件失败，这个宽松定义没有改变结论。

## 8. 可以说与不能说

可以说：在 canonical graph/noise 与锁定 legacy substrate 下，单独把患者轴 E→I reach 扫到 q≈1.5，没有打开 bounded carrier；进入之后的升级过程与 reach 无关；最强那一档把进入提前到同 q 对照带之外；Q2/Q3 带来 baseline tradeoff。

不能说：Mexican-hat 普遍无效；U 被否定；LC6A 测过 termination 或完整 lifecycle；更宽 reach 改变了 D 耗竭 halo 的几何；Q1/Q2 推迟了进入；gain fork 测到了高态的响应性；晚窗配对偏转是 reach 依赖的周边信号。LC6A 从设计上只测 carrier capability。

## 9. 下一机制分支

固定宽核把 800 条 E→I 输入从近处重新分配到远处，也削弱了局部 center。若继续，优先做 spec 已预留但未授权的 center-preserving two-component E→I kernel（70–75% legacy local + 25–30% wide axial），而不是继续扩单一 q 网格。若仍是全局 saturation，应转向 H source/transfer；不能再把问题包装成“刹车剂量不足”。

## 10. 工程与边界

graph legality、two-hop、functional、自然轨迹、两个 gain phenotype、四组主图和未触发 confirmation 均完成；六个 blessed engine hash 一致。C0 自然轨迹对参考路径 bitwise parity 通过（spike sha256 一致、rate 最大差 0 Hz）。无 carrier，所以 confirmation 不运行是合同结果，不是缺失实验。

本 archive 的 §4–§7 由 2026-08-15 的复审重写；被取代的旧表述与逐条证据记录在 `run_manifest.json` 的 `post_hoc_corrections`（六条：轴向 front 读数退化、进入时刻需要同 q 对照、gain fork 不是高态、晚窗功能偏转是脉冲时刻发散、boundedness margin 单位混用、探针基线 H≠0）。原始 per-arm JSON、图与哈希链未改动。

结果根：`results/topic4_sef_hfo/fcxr_lc6a_patient_axis_surround`。spec 的旧嵌套路径示例未被 runner 使用；这一纯路径偏差已在 `run_manifest.json` 留痕。

termination：`NOT_TESTED`。lifecycle：`NOT_TESTED`。
