# heterogeneity — E 细胞异质性 (Track E / rate 均场层)

spec: `docs/superpowers/specs/2026-06-06-sef-hfo-pathology-parameter-mapping-design.md` §5.2
plan: `docs/superpowers/plans/2026-06-06-sef-hfo-lif-heterogeneity-first-round.md`

## 首轮 = 只收窄阈值分布 `Var(V_th,E)`（子结论，已封存）

- `optpoint.json` — 非空间工作点强制链 (Task 7)。三层 baseline(wide=1.5)/raw_narrow(0.5,同均值)/mean_matched(0.5,重配均值到 nuE) 的有效曲线斜率、曲率、**完整 2×2 E/I 闭环稳定性 `closed_loop_re_max`**（max Re λ, spec §2C 的「地图/诊断量」）。每层带 `closed_loop_converged/n_converged/n_modes`；未收敛报错（不报 stable/unstable）；非默认工作点先过 reset-knee 闸门。
- `patch.json` — 空间有限脉冲 + 事件分析 (Task 9)。两种 patch 位置（与种子同位 x=−3 / 下游 x=0），三层同上。自限标签 `label`、成核聚集 `frac_mass_in_patch`、`max_ext`。
- `margin.json` — 有限脉冲安全余量 sweep (Task 9b, spec §2C 真闸门 S = A_runaway − A_event)。**只变刺激幅度 A**，baseline(uniform wide=1.5) vs mean_matched(narrow=0.5 core)，both mean-matched 到 nuE。

**关注点（朴素话）**：阈值分布截断在复位电位以上（2026-06-07 修复——旧版无界分布会采到非物理区、出负发放率、被硬压成 0 导致非单调；旧对照测试是循环验证，绿灯无意义）。可用阈值差异幅度被 `V_TH−V_RESET=7 mV` 这条窄缝**结构性**卡住，锁定 wide=1.5/narrow=0.5。

**首轮子结论（只针对"阈值这一条"）**：收窄 E 阈值分布（mean-matched）使有效曲线变陡约 +13%（算出来的真信号，spec §7 只报方向）——但这点信号**传不到安全余量**：
1. 线性闭环余量基本不动（Δ max Re λ ≈ −0.0003）——这只是「地图」；
2. 有限脉冲：把刺激 A 从 8 推到 200，响应在 A≈80 已进入**饱和平台**（max_ext wide 0.239 / narrow 0.247，A80..A200 不再变；narrow 因 +13% 略高一点点）。在**这组已扫到饱和的平台里，整齐门槛和原始门槛都没推到失控**（`runaway_unreached_in_saturated_grid`）。
   - ⚠ 措辞纪律：这是"在已扫到饱和的网格里推不到失控"，**不是**数学意义上"任意大刺激永不失控"。
3. 非空对照：直接降低 core 门槛均值（让组织本身更易兴奋）确实能 runaway（`test_hetero_field_can_run_away_positive_control`）——所以"怎么推都自限"是有意义的结果，不是 integrator 永不失控。

**封存口径（不要写成"异质性下降无效"）**：**仅"阈值分布变窄"这一条，在均场 rate 模型里没有让有限扰动更容易把组织推到失控。** 完整的细胞异质性机制（不止阈值）尚未验完；有限细胞的"突然集体同步爆发"均场原理上看不见，最终要交给 SNN 的有限细胞实现。

## 下一步 = Rate Step3b：细胞异质性来源扩展（仍 rate 纪律，不上 SNN）

问题换成：如果"细胞更像"不只是阈值变窄，而是整条输入-输出曲线一起变像，会怎样？rate 层先做**非空间敏感度矩阵**找哪类旋钮真有杠杆，再做组合轴，**只有在非空间层真的改变余量的参数包**才放进空间 patch。
- 参数包：`V_th`(门槛) / `tau_m`(膜时间常数,响应速度) / `tau_ref`(不应期,高发放上限) / `sigma`(输入噪声,低发放区斜率)；`E_L`/baseline drive 只作工作点移动对照，不和"异质性下降"混读。
- 产物（规划中）：`sensitivity_matrix.json`（每个参数分布变窄后 斜率/曲率/闭环余量怎么变）→ `combo_*.json`（V_th+tau_m+tau_ref 同向；V_th+sigma）。

## Rate Step3b — 非空间敏感度矩阵 (sensitivity_matrix.json)

每个细胞参数的相对离散度按同一比例收窄 (CV 0.08→0.027)、mean-matched 到 nuE，看有效曲线斜率/曲率/闭环余量怎么变（方向算出来不预设）。结果（杠杆排序）：
- `V_th`(门槛)：斜率 **+11.9%**，闭环余量 Δ≈−0.0003；
- `sigma`(输入噪声)：斜率 **+5.8%**，余量 Δ≈−0.0001；
- `tau_m`(膜时间常数)：**0 杠杆**（斜率 +0.0%）；
- `tau_ref`(不应期)：**0 杠杆**（工作点率极低，离 1/tau_ref 天花板太远，预料之中）；
- 对照（移工作点 +1mV）：余量 Δ=+0.0124 → 仪器是活的，上面那些"余量不动"是真"没杠杆"不是测不出。

**结论**：单个内在细胞参数收窄，没有一个能在 rate 层推动安全余量。只有 V_th / sigma 能改曲线斜率（~+12% / +6%），但传不到余量。下一步只测唯一有意义的组合 **V_th + sigma**（两个有斜率杠杆的）；`V_th + tau_m + tau_ref` 因后两者 0 杠杆 ≈ 等于只收窄 V_th。只有组合若真的推动余量，才进空间 patch。
