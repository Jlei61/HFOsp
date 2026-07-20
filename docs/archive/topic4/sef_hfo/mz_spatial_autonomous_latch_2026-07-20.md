# 审阅结论：autonomous regional Z–p–M latch

日期：2026-07-20

工作线：`.worktrees/topic4-mz-divisive-lifecycle`

独立性：本节点固定 current-based Stage 0C、M3B `K_EE/K_I`、shared global pool、E→E weight/kernel/delay。没有使用并行 `.worktrees/topic4-mz-conductance` 的 recurrent conductance、smooth saturation 或 relay `x`。

## 1. 一句话判断

**regional Z 已能由普通背景事件自主跨过真实空间 fold，并产生 pulse-free core–annulus recurrent activity；但当前 additive M 只能在 3 个 return 后安全退出，一旦保留第 4 个 return，持续 Z 耗竭就先把轨迹推离 validated CCO/transfer-support 区域。**因此本设计在预注册稳健性标准下是 clean no-go，不能继续靠细扫 `tau_m` 抢救；这不是对所有中间参数值的全局不存在定理。

## 2. 完成程度

> **完成度：100/100（本 autonomous latch gate）；约 78/100（最终可恢复时空 lifecycle 目标）**

已完成：

- 从 accepted interictal root 出发，不直接初始化到 post-fold；
- 固定 seed、state-independent 的 6 次 ordinary focal-event challenge；
- slow-off、Z-only、no-entry、fast-exit、four-return boundary、delayed-gate 六个因果 arm；
- regional core+annulus Z 与 M 坐标、bath zero-depletion mask；
- local occupancy、persistence、neighborhood recruitment 与 hybrid latch；
- base/half `dt=.125/.0625 ms`；
- fail-closed transfer/bound/finite tracking；
- 2×3 diagnostic figure、CSV、summary JSON、两套 NPZ traces 与中文 README。

尚未完成：合格且留有时间余量的 `>=4 returns + finite exit`、autonomous recovery/reset、early/late retrigger、coarse field wavefront、full SNN 与多 event-seed confirm。由于当前 gate 是 registered-margin clean no-go，这些不是漏跑，而是按 stop rule 冻结。

## 3. P0 / P1 关键问题

### P0：入口成立；固定 bath-resource 条件下 bath 未被招募

slow-off arm 中 6 次 20-ms、3-mV core pulse 均产生一次 core+annulus returning event，随后回到原间期；bath 无 return。Z 打开后：

- `tau_z_dep=7.5 s`：base/half-dt 均在 `11.000 s` 使共享 regional Z 跨过
  `z_R^SN=.8558315843`；
- `tau_z_dep=8.0 s`：最小 Z 约 `.85612/.85615`，保持间期；
- Z-only post-fold arm 在 17 s 内产生 14 个 pulse-free regional returns；
- 在预注册 zero-depletion mask 下，bath 始终 `z=.90`、0 returns，峰值低于 20 Hz。

所以当前缺口已经不是“模型里没有 spatial entry”，也不是“背景 pulse 直接造了 seizure”。最后一个外驱 pulse 在 `10.915–10.935 s`；分析从 `11.035 s` 开始，后续 returns 是无外驱 fast dynamics。

### P0：完整 lifecycle arm 不存在

primary tradeoff 在两套 dt 完全一致：

| arm | `p_on` | `tau_m_up` | pulse-free returns | 结果 |
|---|---:|---:|---:|---|
| fast exit | `.115` | `200 ms` | 3 | finite，17 s 尾窗回到 `<1.4 Hz` |
| four-return boundary | `.115` | `225 ms` | 4 | `13.91 s` 左右离开 transfer support |
| delayed gate | `.120` | `100 ms` | 4 | `13.61 s` 左右离开 transfer support |

fast-exit arm 最低 `z_R≈.82323/.82329`，最大共享 `m≈.2796/.2782`；four-return arm 在失败前最低 `z_R≈.82189/.82191`、最大 `m≈.3064/.3056`。把 latch 延后没有解决问题：它只让 Z 更深，尽管 M build 更快，仍先碰 support boundary。

225-ms arm 的第 4 次 annulus/core return 到 support escape 仅约 `18.9/25.3 ms`，两套 dt 一致。200–225 ms 之间不能被数学上穷尽排除，但任何“刚过第 4 return 就立刻退出”的窄解都会违反已锁定的 `<0.25 return` robustness margin，不能作为可分析、可移植的 lifecycle。

**怎么改**：关闭“同一个 Z-use law + additive M”这条精调路线。下一节点先限制 recruited-state 下 effective inhibitory loss 的可达范围，而不是继续增大 A 或缩小 `tau_m`。

### P1：这仍不是严格 zero-input spontaneous seizure

输入 schedule 由 seed `20260720` 的 shifted-exponential renewal 过程预先生成，所有 arm/dt 共用，且不读取状态；但 onset 前确实有普通背景 pulse。因此安全词是：

> **background-event-driven endogenous transition**

不能写“零输入自发 seizure”。真正 zero-input 需在完整噪声 SNN 中复验。

### P1：P=3 仍是 coarse spatial coordinate

core 与 equal-area annulus 共用 regional Z/M，是为了严格对应 frozen oracle 已证明的控制坐标；bath 仍保留 90.2% 真实面积。它证明 localized entry，以及**在 bath resource 被固定时** fast coupling 没有招募 far bath；后者不是 emergent containment，更不是连续 wavefront。若下一机制通过 P=3，field lift 必须取消手工共享区域坐标与 bath mask，改成逐点 inhibitory-use dynamics。

## 4. 科学性反思

### 4.1 哪些设计是正确的

1. **先解 fast geometry，再闭 slow loop**是正确的。当前能明确区分“没有 ictal state”和“有 state 但 slow trajectory 走不通”。
2. **Z 负责 entry、A 负责 exit**的符号正确：降低 Z 把 E nullcline 向高率方向移并穿过 localized real fold；提高 A 把它拉回低率侧。
3. **persistence AND neighborhood recruitment**修复了 memoryless sensor 的 prevention。单 IED annulus `p_max=.0487`、完整背景列 slow-off 为 `.0603`、near-critical no-entry 为 `.0858`，accepted fixed CCO 为 `.1227`；`p_on=.115` 在 fixed CCO 第 4 个 return 后 set。
4. **区域化而非全局化**是必要的。初版逐 patch Z 会让 core 先过 fold、annulus 留在 `.8596`；改成已验证的 shared regional coordinate 后，才得到真正 regional entry。bath 若参与同一 depletion law，则 CCO 轨迹反算会在约数秒内把 far bath 也推过 fold；primary zero-depletion bath mask 把这个混杂固定住，但也意味着当前不能把 bath-low 写成 emergent containment。
5. **fail-closed classifier**是必要的。旧临时输出中 return 消失后出现 NaN；若只看 event count，会把 support escape 误写成 termination。正式 producer 在第一次 support/bound/finite failure 即冻结该 fork并单独标红。

### 4.2 哪些设计不够、为什么不够

当前 slow fast system 为：

\[
\dot x=F(x;z,A),\qquad A=A_{max}m.
\]

令动态退出边界为 `A_exit(z)`。frozen 数值给出：

\[
A_{exit}(.855)\approx.02\ {\rm mV},\qquad
A_{exit}(.850)\approx.12\ {\rm mV},
\]

所以局部斜率约为：

\[
\frac{dA_{exit}}{dz}\approx-20\ {\rm mV}.
\]

定义到退出边界的距离：

\[
G=A-A_{exit}(z),
\]

则：

\[
\dot G=\dot A-A'_{exit}(z)\dot z.
\]

在 CCO 中 `dot z<0` 且 `A'_exit<0`，因此 M 追逐的是一个持续上移的 moving target。为 sparse IED accumulation 设定的 Z-use gain，在高占空比 CCO 中继续工作，导致：

- M 快：能追上，但在合格 ictal-duration gate 前就退出；
- M 稍慢：保留第 4 个 return，但 Z 已把 exit boundary 推到更高并先离开 support；
- 延后 latch：Z 更深，问题更严重；
- 加大 `Amax`：只会把尖锐边界移动，不修复 slow-flow geometry。

所以 additive M 的**方向**没有错；错的是假设“同一无下界 Z-use law 可以同时负责 sparse-event entry 和 recruited-state evolution”。

### 4.3 与 QI/JK M3/M4 分母路线的区别

QI/JK 的 recurrent-E divisor 改变 fast E-nullcline 的局部增益，打开了 bounded sustained third state，但没有建立回原间期的 slow exit。当前线不再缺 bounded target state：P3/frozen oracle 已找到 bounded localized CCO 与 additive exit boundary。当前失败发生在外层慢轨迹：Z 下冲让 exit boundary逃离 M。

因此不能照搬 QI/JK 再加一个 recurrent-E 分母：那会重复另一条线正在处理的 E→E bounded-branch 问题，也没有直接修复这里的 moving-boundary slow-flow 冲突。

### 4.4 当前分岔图景

现有证据支持：

\[
\text{LLL}\xrightarrow{\text{regional real fold}}\text{bounded CCO window}
\xrightarrow{\text{slow drift}}\text{exit or support escape}.
\]

入口临界模为纯实数，不支持 Hopf/torus 名称；内环周期随 Z 降低而缩短，符合 fold/SNIC-like bursting 工作假设。当前 no-go 正说明“大环套小环”要求外层慢流留在 bounded CCO corridor 内；只写一个方向正确的负电流不够。

## 5. 工程性问题

- 单进程、单 BLAS 线程、6 个 vectorized forks；两套 canonical traces 各约 2.4 MiB；
- 长轨迹在分配前估算 trace bytes，超过 256 MiB 直接拒绝；
- pulse onset/duration 必须与 dt 对齐；
- full state 检查覆盖 `rE/rI/sEE/sEI/sIE/sII/rE_fast/z/p/m/shared`；
- transfer support、state bound、finite 各有首次失败时间；失败 fork 不继续传播 NaN；
- external current default 为 exact zero，slow-off/旧 frozen paths byte-for-byte 不受影响；
- base/half-dt 六个 outcome label 全一致；
- 相关测试当前 24/24 通过，图已目视检查。

## 6. 最小修改路线

### 首选：depletable + nondepletable inhibitory reserve

把 effective inhibitory multiplier 写成：

\[
q_i=q_{res}+(q_0-q_{res})D_i,
\]

\[
\dot D_i=\frac{1-D_i}{\tau_{D,r}}-\frac{D_iU_i}{\tau_{D,d}},
\qquad
U_i=H_\epsilon((K_I*r_I)_i-s^{I,0}_i).
\]

它保留 sparse IED 的逐次 depletion，但让 recruited CCO 中 `q_i>=q_res`，从结构上给 `A_exit(q)` 一个有限上界。这里必须区分慢参数 `q_res` 与快系统实际看到的 `q`。在 frozen attractor 上周期平均后：

\[
\overline{\dot q}=
\frac{q_0-q}{\tau_{D,r}}-
\frac{(q-q_{res})\overline U(q,A)}{\tau_{D,d}},
\]

所以通常 `q` 的慢零流形位于 `q_res` 之上；不能把目标 hold coordinate 直接当作 floor 参数。

进一步续算也修正了“一维 q-support boundary”的想法。A=0 的 bounded CCO 至少延伸到 `q=.80`，而 autonomous four-return failure 位于 `(q,A)≈(.82189,.49023 mV)`；该点几乎贴住插值得到的 low-root fold `A_SN≈.49061 mV`。因此失败边界是二维 `(q,A)` geometry，不是单独一个 `q_support`。

第一步不是跑 lifecycle，而是做二维 frozen corridor continuation。只有找到宽度至少 `.005`、由至少 3 个节点确认的连续 `q` 区间，并在每个节点满足

\[
A_{exit}(q)+.02\ {\rm mV}<A_{fail}(q),
\]

才允许由周期平均 `\overline U` 反算 `q_res` 并闭 slow loop。单个精调 floor 不算通过。

### 条件第二臂：Abbott-like local dynamic threshold

若 reserve 防止 support escape 但只留下 tonic plateau，再加入：

\[
\tau_\phi\dot\phi_i=-\phi_i+\kappa r^E_i,
\qquad \mu^E_{eff}=\mu^E-A-\phi.
\]

它的目标不是代替 slow exit，而是限制连续 CCO 的峰值/占空比、降低前 4 returns 的累计 Z-use，并可能把 fast real high-gain response 改造成 relaxation-like bursting。若只造成 prevention、global Hopf 或低平台，则停止。

### 只作阳性控制：recruited-state Z clamp

用 recruited latch 直接关闭 Z depletion 几乎肯定能构造 lifecycle，但它容易把答案写进方程。只允许作“fast geometry 是否充分”的阳性对照，不能升为主机制。

## 7. 下一步建议

**GO：先做 inhibitory-reserve 二维 `(q,A)` frozen corridor oracle。**保持 E→E、conductance、global pool、Amax 全冻结；先测 established CCO 的 `A_exit(q)` 与首次 `A_fail(q)`，再判断是否存在连续、有余量的 safe strip。R0 通过后才记录周期平均 `U_CCO`、由 q-nullcline 反算 `q_res`；不能直接把候选 `q_hold` 当 floor 参数。

**NO-GO**：继续扫 `tau_m/p_on/Amax`、直接进入 field/SNN、恢复三条下游 workflow，或把 fast-M 的 3-return finite exit 写成完整 seizure termination。

## 8. 产物

- 图：`results/topic4_sef_hfo/mz_spatial_autonomous_latch/figures/mz_spatial_autonomous_latch.png`
- summary：`results/topic4_sef_hfo/mz_spatial_autonomous_latch/autonomous_latch_summary.json`
- outcomes：`results/topic4_sef_hfo/mz_spatial_autonomous_latch/autonomous_latch_outcomes.csv`
- traces：`results/topic4_sef_hfo/mz_spatial_autonomous_latch/autonomous_latch_dt{0p125,0p0625}_traces.npz`
- config：`config/topic4_mz_spatial_autonomous_latch.yaml`
- producer：`scripts/run_topic4_mz_spatial_autonomous_latch.py`
