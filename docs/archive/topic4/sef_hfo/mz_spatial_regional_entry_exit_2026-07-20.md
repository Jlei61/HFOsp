# 审阅结论：regional-Z entry fold 与 delayed additive exit

日期：2026-07-20

工作线：`.worktrees/topic4-mz-divisive-lifecycle`

独立性：本节点固定 current-based Stage 0C / M3B spatial operator，只改变 regional `z` 与 recruited-region additive `A` 两个控制坐标。没有修改 E→E weight/kernel/delay，没有使用并行 `topic4-mz-conductance` 线已接受的 recurrent conductance/smooth saturation，也没有加入 relay `x_j`。

> **2026-07-20 downstream addendum**：后续 autonomous regional Z–p–M 已完成并得到 clean no-go：Z 可自主跨本报告的 real fold，但 additive M 安全退出早于 4-return gate；保留第 4 个 return 时，轨迹先离开 transfer support。因此本 frozen geometry 仍成立，但它没有自动闭合 slow lifecycle。后续入口为 `docs/archive/topic4/sef_hfo/mz_spatial_autonomous_latch_2026-07-20.md`。

## 1. 一句话判断

**当前 fast spatial system 已具备一条可被慢变量穿越的完整入口–退出几何。**区域性 Z 耗竭通过一个局部实模 fold 把同一低态历史带入 bounded `CCO`；在该 cycle 已建立后，只对 core+annulus 打开 additive A，又能把它推回同一个 `LLL` basin。这个结果解锁一个最小 autonomous regional Z/M latch，但还不是完整自发 seizure lifecycle。

## 2. 完成程度

> **完成度：100/100（frozen regional entry/exit oracle）；约 70/100（最终时空 lifecycle 目标）**

已完成：

- 6D regional E/I equilibrium manifold 与多 seed root search；
- 去掉冻结 `z/p/m` 零模后的 23D fast Jacobian；
- 增广 fold solve `F=0, D_rFv=0, ||v||=1`；
- fold 非退化、transfer support、临界模态空间定位；
- low-history relabel 与四相位 warm-cycle basin forks；
- established-cycle event checkpoint、matched `A=0` twin、from-t0 prevention 与 low-history control；
- regional `A=(A,A,0)` 两个 depletion depths、base/half-dt；
- exit 后恢复 `(z=.90,A=0)` 的同 basin fork；
- PNG/PDF、五张 CSV、summary JSON、NPZ traces、中文 README 与单元测试。

尚未完成：autonomous `dot z/dot m`、噪声触发的 returning interictal events、slow passage、refractory/retrigger、full field wavefront 和完整 SNN 移植。

## 3. P0 / P1 关键问题

### P0 已关闭：入口不是“方向猜测”，而是实模 fold

区域控制坐标固定为：

\[
z=(z_R,z_R,0.90),\qquad A=(0,0,0).
\]

在 equilibrium manifold 上，rates 顺序为：

\[
x=(r^E_c,r^E_a,r^E_b,r^I_c,r^I_a,r^I_b).
\]

增广解得到：

\[
z_R^{SN}=0.8558315843,
\]

\[
r_E=(3.2652,2.2290,0.8674)\ {\rm Hz},\quad
r_I=(8.4363,7.2610,5.7455)\ {\rm Hz}.
\]

数值证据：

- augmented residual `1.14e-11`；
- `sigma_min(D_rF)=1.26e-12`；
- 23D fast leading eigenvalue `-2.10e-8 + 0i /ms`；
- `u^T F_z=-.02558`；
- `u^T D²F[v,v]=34.99`；
- E/I transfer support 三区全通过。

临界 rE 模态归一化为 `(core=.893, annulus=.422, bath=.0186)`；bath 仅为最大分量的 `2.08%`。所以这是 core–annulus localized real fold，不是 Hopf，也不是 bath/global mode。

更关键的是，fold 点三区 `rE<5 Hz`，recruitment sensor 尚未打开，`mu_G=S_G=0`。因此 entry fold 来自 local Z 对抑制 nullcline 的移动和固定空间 coupling；global divisor 在这里不创造入口，它在 fold 后承担 containment。

### P0 已关闭：A 的 exit 是 established-cycle exit，不是 prevention

每个 phase 先在 `A=0` 完成 core/annulus 至少 4 个 returns，再在同一 section state fork：

- twin 继续 `A=0`；
- test 才打开 `A=(A_R,A_R,0)`；
- bath 始终不加 A。

两个 depletion depths 均得到相同 base/half-dt bracket：

| `z_R` | low-root additive fold `A_SN` | delayed cycle persists | first registered delayed exit |
|---:|---:|---:|---:|
| `.855` | `.0127766 mV` | `A=0,.01` | `.02 mV` |
| `.850` | `.0886442 mV` | `A=0,.04,.08` | `.12 mV` |

四 phase × 两 dt 全一致。matched `A=0` 全部保持 bounded CCO；所以 `.02/.12` 是对 established orbit 的 exit leverage，不是从 t=0 压住 onset。

### P1 仍开放：slow vector field 尚未闭合

本结果只证明存在路径：

\[
LLL\xrightarrow{z_R\downarrow}CCO
\xrightarrow{A_R\uparrow}LLL.
\]

它没有证明当前 `dot z/dot m` 会沿正确方向、以正确时间顺序穿越两个边界。若 M 在 entry 前就升高，会变成 prevention；若 Z 在 CCO 中继续下降太快，`A_SN(z)` 会逃离 M 的追赶；若退出后 M 过快衰减，会立刻 retrigger。

**怎么改**：下一步只允许一个基于这两条数值边界锁定的 autonomous slow latch；不扫 E→E，不扩 `Amax`，不先上 SNN。

## 4. 科学性问题

### 4.1 同一低态历史确实跨过 entry boundary

保持 `.90` low root 的全部 fast history，只把 core+annulus 的 `z_R` 改到目标点：

- `.8560`：两套 dt 均保持 `LLL`；
- `.8555`：两套 dt 均进入 bounded `CCO`；
- 更深的 `.8550/.8500` 也进入 bounded CCO；
- `.85575` 显示 fold 附近的长周期 critical slowing，保留为 bottleneck diagnostic。

`LLL` 额外要求最终 `max|F_fast|<1e-8/ms`，避免把 saddle-node ghost 附近的长低尾误认成 fixed point。

### 4.2 fast 小环随 Z 深度有系统变化

从 `.855` 到 `.850`，bounded CCO period 由约 `1.146 s` 缩短到约 `.671 s`；同时 exit 所需 A 从 `.0128 mV` 增至 `.0886 mV`。这正是 slow–fast bursting 需要的几何：Z 越耗竭，cycle 越快、M 需要追赶的 exit boundary 越高。

但当前数值不支持 torus 解释：entry 临界特征值是纯实数，未见 complex pair crossing。最合适的工作假设是：

> 外层 Z/M 慢环包住内层 fast CCO；入口与退出靠 fold/SNIC-like 边界，而不是 torus bifurcation。

正式 SNIC 名称仍需 regional periodic-orbit continuation；当前安全词是 **real entry fold + bounded CCO + fold-aligned additive exit**。

### 4.3 空间模式是什么、还缺什么

已得到的空间 pattern 是：core 与等面积 annulus 周期性招募，90.2% far bath 无 returns。它不是全场同步，也不是均匀单节点振荡。

但空间尺度仍由 `core/equal-area annulus/bath` partition 预先定义；P3 fast gate 又显示 `CLL→CCL` 不成立。因此下一版必须解释“为什么 Z 会在 core 与 near annulus 联合下降”，不能偷偷改 E→E 把邻环硬拉进来。最自然的本线假设是：Z 由**到达该区域的抑制性突触使用量**耗竭，而不是只由本地 E spike 耗竭；由于 `s_EI=K_I r_I` 已含宽程 I 投射，core event 可以消耗 annulus 的 postsynaptic inhibitory resource，而不改变 E→E connectivity。

### 4.4 参数恢复回到同一个间期 basin

在 confirmed delayed exit 后，把所有 fork 恢复到 `(z=.90,A=0)` 再积分；`2 z × 4 phases × 2 dt =16/16` 均保持 `LLL`。这关闭了“只是掉进另一个 silent basin”的替代解释，但尚未测试 early/late retrigger。

## 5. 工程性问题

- 单进程、单 BLAS 线程；最大向量 forks 小于 96；
- canonical run wall 约 `18 min`，peak RSS 约 `0.35 GiB`，0 swap；
- root/fold 部分使用 6D manifold，避免把九个 frozen slow zero modes 混进 spectrum；
- fast spectrum 精确使用 packed indices `[0:21,30,31]`；
- regional additive 只写 core/annulus `m=A/1.6`，bath 恒为 0；
- exit checkpoint 按 return count，而非固定 wall-clock；
- low、cycle、ceiling、support failure 分层，避免旧 classifier 混类；
- 所有上游输入 SHA-256 fail-closed；
- 图已目视检查，entry branch、basin map、A fold/exit 和 matched exit traces 可读。

## 6. 最小修改路线

下一版锁为独立的 **regional-use Z / local-activity M slow latch**：

\[
\tau_z\dot z_j=(z_0-z_j)-u_z z_j\,\chi^I_j,
\qquad
\chi^I_j=H_\epsilon(s^{EI}_j-s^{EI}_{0,j}),
\]

\[
\tau_m\dot m_j=\chi^E_j(1-m_j)-\kappa_m(1-\chi^E_j)m_j,
\qquad
\chi^E_j=H_\epsilon(r^E_{fast,j}-\theta_m),
\]

\[
A_j=A_{max}m_j,qquad j\in\{core,annulus\},\quad A_{bath}=0.
\]

执行顺序：

1. 从 accepted interictal state 出发，先用低幅 stochastic/returning-event drive 验证普通 IED 后 `m` 不越过 `.0128 mV` prevention boundary。
2. 只从已有 SNN/Gate-0 event statistics 锁 `chiI/chiE` 的 threshold 与 slow timescale，不从 seizure outcome 反调。
3. 要求 `z_R` 在若干 returning events 后跨 `.8558316`，而 bath `z` 保持高；不允许直接初始化到 `.85` 冒充 onset。
4. onset 后要求至少 4 个 bounded CCO returns，随后 `A_R` 穿过与当时 `z_R` 对应的 `A_SN(z)`。
5. exit 后检验 Z 恢复、M 慢衰减、early retrigger suppression 与 late retrigger recovery。
6. P=3 成功后才进 coarse field；full SNN 最后做，不与 conductance/relay 线重复。

只允许一个 small factorial：`z-use gain × m-up timescale` 各 2–3 档，参数必须先由 event history 与已测边界换算；不做大网格。

## 7. 下一步建议

**GO：执行一个最小 autonomous regional Z/M latch。**这次 GO 的依据不是“看起来会回去”，而是 fast subsystem 已有：

1. 支撑良好的 localized real entry fold；
2. fold 后 bounded spatial CCO；
3. event-locked additive exit；
4. 参数恢复后的同一 LLL basin。

**仍然 NO-GO**：直接恢复三条下游 workflow、直接叫 spontaneous seizure、直接上 full SNN 或改 E→E。只有 autonomous slow trajectory、retrigger 与空间扩展门通过，才恢复 early-ictal、ecomode 与 interictal→ictal bridge 分析。

## 8. 产物

- 图：`results/topic4_sef_hfo/mz_spatial_regional_entry_exit/figures/mz_spatial_regional_entry_exit.png`
- summary：`results/topic4_sef_hfo/mz_spatial_regional_entry_exit/regional_entry_exit_summary.json`
- roots：`results/topic4_sef_hfo/mz_spatial_regional_entry_exit/regional_equilibrium_roots.csv`
- entry：`results/topic4_sef_hfo/mz_spatial_regional_entry_exit/regional_entry_outcomes.csv`
- exit：`results/topic4_sef_hfo/mz_spatial_regional_entry_exit/regional_exit_outcomes.csv`
- recovery：`results/topic4_sef_hfo/mz_spatial_regional_entry_exit/regional_recovery_outcomes.csv`
- config：`config/topic4_mz_spatial_regional_entry_exit.yaml`
