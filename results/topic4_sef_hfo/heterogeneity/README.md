# heterogeneity — E 阈值异质性首轮 (Track E)

spec: `docs/superpowers/specs/2026-06-06-sef-hfo-pathology-parameter-mapping-design.md` §5.2
plan: `docs/superpowers/plans/2026-06-06-sef-hfo-lif-heterogeneity-first-round.md`

- `optpoint.json` — 非空间工作点强制链 (Task 7)。收窄 `Var(V_th,E)`，三层 baseline(wide=1.5)/raw_narrow(0.5,同均值)/mean_matched(0.5,重配均值到 nuE) 的有效曲线斜率、曲率、**完整 2×2 E/I 闭环稳定性 `closed_loop_re_max`**（max Re λ, spec §2C）。每层带 `closed_loop_converged/n_converged/n_modes`；未收敛报错（不报 stable/unstable）；非默认工作点先过 reset-knee 闸门。
- `patch.json` — 空间有限脉冲 + 事件分析 (Task 9)。两种 patch 位置（与种子同位 x=−3 / 下游 x=0），三层同上。每层报自限标签 `label`、成核聚集 `frac_mass_in_patch`、`max_ext`。`interpretation.d_*_pure` = mean_matched − baseline = 纯方差效应。

**关注点（朴素话）**：阈值分布截断在复位电位以上（2026-06-07 修复）；可用异质性幅度被 `V_TH−V_RESET=7 mV` 卡住（gap-limit, 结构性），锁定 wide=1.5/narrow=0.5。

**首轮总结论（rate-field 层，两条闸门一致的 null）**：收窄 E 阈值异质性（mean-matched）使有效增益升约 +13%（算出来的真信号，spec §7 只报方向）——但
1. 线性闭环稳定余量基本不动（Δ max Re λ ≈ −0.0003，甚至略更稳）；
2. 空间有限脉冲层**所有 6 个条件都保持 self_limited_propagation**，没翻成 runaway；成核聚集的纯方差效应很小且随 patch 位置变号（同位 −0.06 / 下游 +0.10），物理上自洽——+13% 增益表现为传播略远（max_ext +0.01~0.02），质量从种子处略微移到下游，不是质变。

**即 Rich 方向「异质性↓→更不稳/更危险」在 rate-field（均场）层没有稳健复现。** 这与 framework 预期一致：rate-field 只抓「粗版」（均场余量/传播），真正的有限尺寸**同步爆发**需要 SNN 对应版验证（本轮范围之外）。存在性移交 SNN。
