# heterogeneity — E 阈值异质性首轮 (Track E)

spec: `docs/superpowers/specs/2026-06-06-sef-hfo-pathology-parameter-mapping-design.md` §5.2
plan: `docs/superpowers/plans/2026-06-06-sef-hfo-lif-heterogeneity-first-round.md`

- `optpoint.json` — 非空间工作点强制链 (Task 7)。在 canonical 工作点收窄 `Var(V_th,E)`，比较 baseline(wide=1.5) / raw_narrow(0.5,同均值) / mean_matched(0.5,重配均值到 nuE) 三层的有效曲线斜率、曲率、**完整 2×2 E/I 闭环稳定性 `closed_loop_re_max`**（max Re λ, spec §2C, 不是 E→E proxy）。`interpretation.d_*_pure` = mean_matched − baseline = **纯方差效应**（去掉 Jensen 平均移动）。每层带 `closed_loop_converged/n_converged/n_modes`；脚本对未收敛的根搜索直接报错（不报 stable/unstable）；非默认工作点先过 reset-knee 闸门否则报错。

**关注点（朴素话）**：阈值分布必须截断在复位电位以上（否则采到非物理负发放率，见 2026-06-07 修复）。可用异质性幅度被 `V_TH−V_RESET=7 mV` 卡住（gap-limit），锁定 wide=1.5/narrow=0.5（reset-knee 贡献 <5%，干净）。**首轮结果**：收窄异质性使有效增益升约 +13%（算出来的真信号，spec §7 只报方向），**但线性闭环稳定余量基本不动（Δ max Re λ ≈ −0.0003，甚至略更稳）** —— Rich 方向「异质性↓→更不稳」在这个工作点的线性稳定性这条腿上**没复现**。这与 framework 2026-06-03 更正一致（LIF 工作点稳健稳定但非线性可激；线性余量不是作用点，有限脉冲余量才是）。存在性移交空间有限脉冲层（Task 8/9，待定）。
