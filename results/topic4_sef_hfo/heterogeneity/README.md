# heterogeneity — E 阈值异质性首轮 (Track E)

spec: `docs/superpowers/specs/2026-06-06-sef-hfo-pathology-parameter-mapping-design.md` §5.2
plan: `docs/superpowers/plans/2026-06-06-sef-hfo-lif-heterogeneity-first-round.md`

- `optpoint.json` — 非空间工作点强制链 (Task 7)。收窄 `Var(V_th,E)`，三层 baseline(wide=1.5)/raw_narrow(0.5,同均值)/mean_matched(0.5,重配均值到 nuE) 的有效曲线斜率、曲率、**完整 2×2 E/I 闭环稳定性 `closed_loop_re_max`**（max Re λ, spec §2C）。每层带 `closed_loop_converged/n_converged/n_modes`；未收敛报错（不报 stable/unstable）；非默认工作点先过 reset-knee 闸门。
- `patch.json` — 空间有限脉冲 + 事件分析 (Task 9)。两种 patch 位置（与种子同位 x=−3 / 下游 x=0），三层同上。每层报自限标签 `label`、成核聚集 `frac_mass_in_patch`、`max_ext`。`interpretation.d_*_pure` = mean_matched − baseline = 纯方差效应。

**关注点（朴素话）**：阈值分布截断在复位电位以上（2026-06-07 修复）；可用异质性幅度被 `V_TH−V_RESET=7 mV` 卡住（gap-limit, 结构性），锁定 wide=1.5/narrow=0.5。

- `margin.json` — 有限脉冲安全余量 S sweep (Task 9b)。**只变刺激幅度 A**，对 baseline(uniform wide=1.5) vs mean_matched(narrow=0.5 core, both mean-matched 到 nuE) 各扫出 `A_runaway`（label 从 self_limited 翻成 runaway/global 的最小 A）。这才是 spec §2C 的最终闸门 S = A_runaway − A_event。

**关注点（朴素话）**：阈值分布截断在复位电位以上（2026-06-07 修复）；可用异质性幅度被 `V_TH−V_RESET=7 mV` 卡住（gap-limit, 结构性），锁定 wide=1.5/narrow=0.5。

**首轮观测（rate-field 层，⚠ 结论以 margin.json 的 S 为准）**：收窄 E 阈值异质性（mean-matched）使有效增益升约 +13%（算出来的真信号，spec §7 只报方向）——
1. 线性闭环稳定余量基本不动（Δ max Re λ ≈ −0.0003，甚至略更稳）——但这是 spec §2C 说的「**地图/诊断量**」，不是闸门；
2. 空间有限脉冲在**单一幅度 A=8** 下 6 个条件都 self_limited_propagation（只说明 A_event=8 < A_runaway），纯方差成核效应很小且随 patch 位置变号（−0.06 / +0.10），物理自洽（+13% 增益 → 传播略远 max_ext +0.01~0.02）。

**真闸门 S（A_runaway sweep）= 结论依据**：所有动了的读数都朝「更易激」动（增益↑、max_ext↑），提示效应可能正落在 A_runaway 上 —— 故 Rich 方向是否在 rate-field 复现，取决于 S_narrow 是否 < S_wide（margin compression），不能只看固定 A 的 label。见 `margin.json`。
