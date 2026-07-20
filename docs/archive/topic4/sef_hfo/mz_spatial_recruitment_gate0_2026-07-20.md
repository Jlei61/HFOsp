# 审阅结论：MZ spatial recruitment Gate 0 sentinel

日期：2026-07-20

工作线：`.worktrees/topic4-mz-divisive-lifecycle`

并行边界：本轮只重放已有 seed-1 artifact；未启动 SNN，未修改 `W_EE`、E→E kernel/delay、relay、conductance membrane 或另一 worktree。

## 1. 一句话判断

**Spatial recruitment 是 Stage B 缺失状态的 single-seed operational support 已成立，但正式 Gate 0 仍未通过。**局部强度单独使用会在普通 IED 时提前开门；空间均值能复刻 onset 后的晚 gate，且与独立二维 movie participation 一致，但当前只有 seed 1、没有 core-specific field，也不存在一个可锁定的瞬时 `rho` separation interval。

## 2. 完成程度

> **完成度：65/100**

已经完成：

- 从锁定 capture 重建 SNN 原始 `Psi(rE_fast)` 局部峰值、空间均值 `A_G` 和 `U_TG/T_G`；
- 在 `p_pool=1` 下做严格分解 `A_G=max(Psi)×rho_eff`；
- 用 causal onset 与完整 pre-onset history 比较 local-intensity-only 和 area-weighted persistence；
- 以 24×24 movie participation ratio 做独立二维主验证，48-bin axial 只作 span sensitivity；
- 把 movie/axial frame 的因果可用时刻从 frame start 修正为 frame end；
- 做 onset-seed vs recruited window、连续 250-ms blocks 和 intensity-matched frame sentinel；
- 输出图、CSV、JSON、NPZ、中文 figure README 与测试。

尚未完成：

- seeds 3/4 的同字段 spatial histories；
- exact core/surround occupancy 或同一 `z_G` field 的 `sum_sq/PR`；
- 可跨 seed 锁定的 `rho_on/rho_off` interval；
- P=1 uniform parity、P=2 core–surround 和 autonomous latch dynamics；
- same-low recovery、retrigger、front stall/annihilation。

扣分原因是当前 artifact 只能支持机制方向的 sentinel，不能把 single seed 的 proxy 当正式空间机制验收。

## 3. P0 / P1 关键问题

### P0：瞬时 `rho` 不能区分 IED 与 established state

`rho_eff=A_G/max(Psi)` 的完整 pre-onset 最大值为 `.3000`，而 onset+2 s 后 established window 的 Q25 只有 `.1579`。因此：

\[
\max rho_{pre} > Q_{25}(rho_{established}),
\]

不存在可把所有 returning IED 与 established state 分开的 memoryless 阈值。

**为什么严重**：若直接把 `rho>rho_r` 作为 M set gate，普通 IED 仍会提前积累 M，重新造成 prevention。

**怎么改**：下一节点保留 `local persistence AND spatial recruitment`，并把 latch set 与 Z-safe reset 分开；不得用这条 seed-1 trace反向挑一个成功阈值。

### P0：正式 Gate 0 缺 multiseed 与 core-specific artifact

当前只有 seed 1 保存 `A_G/rEfast_max/movie`。movie cell 还混合 core/surround，不能恢复 exact core occupancy；现有 capture 也没有同一 `z_G` field 的 `sum(z_G²)`，无法算 exact field participation ratio。

**怎么改**：正式 SNN 确认前，为 primary seeds 保存每步 `sum(z_G)`、`sum(z_G²)`、`max(z_G)`，以及 core/surround compact numerator；无需保存 full raster 或 full field。

### P1：`rho_eff` 是 soft support，不是物理 recruited area

只有当前 `p_pool=1` 时：

\[
rho_{eff}=\frac{\langle z_G\rangle}{\max z_G}
\]

可解释为 normalized `L1/Linf` support。它受 32×32 grid、Gaussian smoothing、局部 density、periodic boundary、Psi 饱和和单点峰值影响，也不能区分 compact patch、多个 islands 或 ring。

**怎么改**：机器字段和图统一称 `effective extent`；二维 movie PR 作为独立 proxy，axial PR 只说明沿预定轴的 span，不进入主判定。

### P1：输入 history 已受旧 divisive feedback 影响

本轮只证明“保留空间平均后能够解释 recorded late gate”，不是新 additive recovery 的 endogenous prediction。

**怎么改**：P=2 必须让 recruitment 内生生成；若需要 imposed `rho(t)` 或 replay 才能 set latch，机制停止。

## 4. 科学性问题

### 做对了什么

1. **局部强度和空间范围被操作性拆开**：局部 `rEfast_max` 从 onset-seed 到 recruited window 仅变化 `+2.64%`；`rho_eff` 增加 `.0994`（`+92.3%`），movie area-PR 增加 `.1139`（`+76.8%`）。
2. **强度匹配后空间变化仍在**：`|Delta Psi_peak|<=.02` 的 10 对 frame 中，median `Delta rho_eff=.0562`、`Delta movie-PR=.0623`。
3. **不是一个短帧偶然**：onset+1–3 s 的 8 个连续 250-ms block 都同时保持 extent 与 movie-PR 正向增加。
4. **独立二维 readout 一致**：post-onset `rho_eff` 与 movie PR 的 descriptive Spearman `r=.942`；500-ms block correlation median `.932`。逐 frame 自相关很强，因此不报告 iid p-value。
5. **timing 反证清楚**：同一 `tau_p=750 ms,p_r=.0722287` 下，local-intensity-only p 在记录 `421.7 ms` 已开门，比 causal macro onset 早 `13.56 s`；area-weighted p 在 onset 后 `2758.5 ms` 开门并以 `<8e-9` 误差复刻 saved `T_G`。

### 结果真正支持的动力学解释

当前数据支持：

\[
\text{local oscillatory intensity}
+\text{spatial extent}
+\text{persistence memory}
\rightarrow \text{late recovery permission}.
\]

它不支持：

\[
\text{single-frame }rho>rho_r
\rightarrow \text{recovery set}.
\]

因此下一版不是再加一条 current，而是让 additive M 的开门条件成为空间—时间联合状态；additive current 本身仍只负责移动 exit boundary。

## 5. 工程性问题

- 所有四个输入以 SHA-256 fail-closed；capture schema、seed、`dt=.1 ms`、`T=20 s`、`p_pool=1` 均验证；
- `dt` 从 JSON 读取，没有从 float32 time diff 反推；保存时钟只按一 ULP 做一致性检查；
- `U_TG` 重建最大误差 `1.74e-7`，`T_G` parity 最大误差 `7.86e-9`；
- movie/axial 值按 `[frame_start,frame_end)` 汇总，所有因果比较使用 frame end；
- 8 个新单元/导入测试通过；
- runner 单进程、BLAS 单线程，wall `3.57 s`，peak RSS `292232 kB`，无 swap/OOM；
- 结果图目录含中文 README，图已人工目视检查。

## 6. 最小修改路线

1. 冻结本 sentinel，不把 seed-1 proxy 当 `rho` 参数标定。
2. 实现通用 P-patch RHS，但先只跑 P=1/uniform parity；全域只有一对 shared `mu_G/S_G`。
3. P=1 复刻 Stage 0C RHS、low fixed point、A=0 cycle period 与 additive boundary 后，才开 P=2。
4. P=2 固定原 coupling，只检查 `LL→CL→CC→low`；core-only 前三次真实 return 必须 `dot m=0`。
5. latch set 使用 local persistence 与 neighborhood recruitment 的 AND；reset 使用 local-low + `z>=z_safe` + `p<=p_off`。
6. 只做 memoryless/latch、rho off/on、m off、cross-zone coupling off；若需要改 E→E weight/kernel/delay，停止并交给并行线。
7. 两区通过后才移植 P=32/64 coarse field，再补 primary-seed compact spatial capture。

## 7. 下一步建议

**GO 到 P=1 parity，再有条件 GO 到 P=2 cheap oracle；正式 spatial Gate 0 保持 OPEN。**

下一次必须回答的不是“空间相关性是否显著”，而是：固定 fast scaffold 下，内生的 surround recruitment 能否让 core latch 在至少三次真实 return 后才 set，并使 `dot D` 从 entry 的负方向翻为 exit 的正方向，最后无 reset 回到同一 low basin。

## 8. 产物

- 图：`results/topic4_sef_hfo/mz_spatial_recruitment_gate0/figures/mz_spatial_recruitment_gate0.png`
- summary：`results/topic4_sef_hfo/mz_spatial_recruitment_gate0/spatial_recruitment_gate0_summary.json`
- metrics：`results/topic4_sef_hfo/mz_spatial_recruitment_gate0/gate0_metrics.csv`
- spatial frames：`results/topic4_sef_hfo/mz_spatial_recruitment_gate0/spatial_validation_frames.csv`
- sentinel blocks：`results/topic4_sef_hfo/mz_spatial_recruitment_gate0/operational_sentinel_blocks.csv`
- traces：`results/topic4_sef_hfo/mz_spatial_recruitment_gate0/spatial_recruitment_gate0_traces.npz`
- config：`config/topic4_mz_spatial_recruitment_gate0.yaml`

设计与执行合同：`docs/superpowers/specs/2026-07-20-topic4-mz-persistence-gated-additive-spatial-lifecycle-design.md`。
