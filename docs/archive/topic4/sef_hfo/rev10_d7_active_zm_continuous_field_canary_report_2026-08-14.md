# Topic 4 rev10-D7 active Z+M 连续场 canary 报告

## 裁定

```text
REV10D7_ACTIVE_ZM_FIELD_STABILIZATION_NOT_OBSERVED
/
SELECTION_NOT_AUTHORIZED
/
FIG4_NOT_AUTHORIZED
```

49 个不使用电极位置构造的低频连续场，在两张 development 网络上的 98/98 条 active Z+M 轨迹全部进入 runaway。当前冻结 Z/M reference 不能作为 data-driven SNN 的稳定机制 baseline，也不能进入 KMeans selection 或 Fig.4 confirmation。

## 实验合同

- 场：18×18 tensor cubic B-spline 上的 49 个 whole-sheet Fourier residual 候选；无 `K`、component 或 peak 参数。
- 机制：`V_th,i=V_th,0-h_i d_i`，固定 local spatial OU，加 `z+m` slow state。
- 固定参数：`tau_z=5000 ms`、`tau_adp=500 ms`、`eta_m=0.0074516`。
- 网络：新 seeds `1421,1422`，候选间 common random numbers。
- 每条轨迹名义时长 20 s；late runaway invalid，允许检测后提前停止。
- Edge exact no-op，`beta`、topology growth 和其他动态机制关闭。

## 工程验收

- controller：`SUCCESS`，commit `c2545895`。
- workers：98/98 JSON 与 98/98 NPZ 完整，runtime/config/manifest/array hash 全部一致。
- measured-RSS sentinel：`7,904,884 KiB`；完成后可用内存 `214,672,752 KiB`。
- controller 选择 9 workers；单 worker `MemoryHigh=20G`、`MemoryMax=24G`；无 OOM 或 worker failure。

## 科学结果

所有 49 个候选都在 2/2 网络上 runaway，共 98/98 条。runaway 出现在 `5.83–10.29 s`，因此先前短于 10 s 的运行不足以裁定该 slow-state reference 的稳定性。

轨迹在 runaway 前产生了大量 returned events，通用审计甚至可找到每网络 A/B 事件；但这些事件不能进入正式 KMeans 或患者几何比较，因为它们来自最终不可返回的轨迹。`REV10D6_NO_EVALUABLE_CONTINUOUS_FIELD_DIRECTION` 在数值上是对的，专用 D7 解释是：**安全门先失败，而不是没有可分群事件。**

## 结论边界

这排除了“在当前固定 `tau_adp=500 ms` Z/M reference 下，仅靠 49 个低频连续场改变 `h` 就能稳定网络”的解释。它不排除其他 slow-variable 方程、参数、连续场基底或机制，但此前 `tau_adp=500–2000 ms` 的 warm-field 校准也全部 runaway，因此不应继续围绕同一 Z/M family 调场或增加 optimizer。

本轮没有 Fig.4 阳性产物。D6.3 的同网络波形与 KMeans 图仍只能作为 diagnostic；它们不能覆盖独立网络复制失败。
