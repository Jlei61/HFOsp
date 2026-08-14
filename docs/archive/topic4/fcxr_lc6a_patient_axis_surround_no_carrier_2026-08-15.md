# FCXR-LC6A 患者轴 E→I surround：canonical-seed bounded-negative

日期：2026-08-15

## 1. 一句话结论

我们确实把患者轴方向的有效两跳抑制周边做宽了，但在当前 legacy Z/H substrate 上，它没有创造一个可停留的中间高态：五条自然轨迹都从 returning IED 进入，随后继续升级到注册 saturation。

正式标签：`CANONICAL_SEED_AXIAL_REACH_FAMILY_NO_CARRIER`。

## 2. LC5v2.1 右删失格收口

唯一右删失格在 23 s onset；从 25 s exact state 继续后，于 27 s 达到 405.9 Hz，D=0.573、H=25.763，判 `ESCALATING_SATURATION`，没有 offset。25--26 s 曾在 reducer 路径失败，因此该段输入 hash 和诊断 trace 不可用；但 exact checkpoint 被恢复，28 个 classifier snapshot bundles 全部重放，注册 saturation 位于完整记录的 26--27 s 段。这个缺口限制第一续跑秒的细粒度诊断，不改变终局 saturation 裁决。

## 3. 这轮真正改了什么

只改变代码 `IE`，即生物学 E→I 的患者轴 reach；EE、I→E、I→I、权重、Z/H、两个 core、噪声与所有慢机制保持冻结。graph-only two-hop 审计显示 q 从 0.935 增至 1.490，所以这不是“图没改到位”的假阴性。

## 4. 五条自然轨迹

| 条件 | q(two-hop) | onset | onset前IED | 全局峰值 | D halo | baseline代价 | 结局 |
|---|---:|---:|---:|---:|---:|---|---|
| C0 | 0.935 | 11.0 s | 29 | 408.2 Hz | 0.01 mm | 无 | saturation |
| C1 | 0.959 | 10.0 s | 28 | 354.1 Hz | 1.24 mm | 无 | saturation |
| Q1 | 0.979 | 13.0 s | 34 | 361.0 Hz | 1.19 mm | 无 | saturation |
| Q2 | 1.285 | 12.0 s | 37 | 398.8 Hz | 1.21 mm | 有 | saturation |
| Q3 | 1.490 | 6.0 s | 21 | 395.7 Hz | 0.57 mm | 有 | saturation |

所有条件的 active area 都到 400 mm²；近 refractory ceiling 的细胞比例很低，但全局和局部 rate 仍跨过注册 saturation。因此这轮不是“有限面积内的健康 carrier”，而是全片 escalating high state。

Q1/Q2 把 onset 推迟到 13/12 s，Q3 却提前到 6 s；更宽 E→I reach 不是单向稳定旋钮。它可能先招募更远的 I 使用，继而在 wavefront 前方加速 D=1-Z 的耗竭。

## 5. gain fork 的意义

按预注册规则选择 C0 与 Q2。fork 只测 exact high-state snapshot 对弱局部输入是否仍有有限非零响应，不参与 boundedness 标签。即使存在很小的非零响应，也不能把已经升级到 saturation 的状态改称 carrier。

## 6. 可以说与不能说

可以说：在 canonical graph/noise 与锁定 legacy substrate 下，单独把患者轴 E→I reach 扫到 q≈1.5，没有打开 bounded carrier；Q2/Q3 还带来 baseline tradeoff。

不能说：Mexican-hat 普遍无效；U 被否定；LC6A 测过 termination 或完整 lifecycle。LC6A 从设计上只测 carrier capability。

## 7. 下一机制分支

固定宽核把 800 条 E→I 输入从近处重新分配到远处，也削弱了局部 center。若继续，优先做 spec 已预留但未授权的 center-preserving two-component E→I kernel（70–75% legacy local + 25–30% wide axial），而不是继续扩单一 q 网格。若仍是全局 saturation，应转向 H source/transfer；不能再把问题包装成“刹车剂量不足”。

## 8. 工程与边界

graph legality、two-hop、functional、自然轨迹、两个 gain phenotype、四组主图和未触发 confirmation 均完成；六个 blessed engine hash 一致。无 carrier，所以 confirmation 不运行是合同结果，不是缺失实验。

结果根：`results/topic4_sef_hfo/fcxr_lc6a_patient_axis_surround`。spec 的旧嵌套路径示例未被 runner 使用；这一纯路径偏差已在 `run_manifest.json` 留痕。

termination：`NOT_TESTED`。lifecycle：`NOT_TESTED`。
