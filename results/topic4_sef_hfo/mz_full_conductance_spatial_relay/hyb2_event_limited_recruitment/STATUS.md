# FCXR-HYB2 —— 最终状态：Gate B0 通过，Gate A0 全锁定输入不可判，预注册停机

最后更新 2026-08-01（A0 判决后），分支 `codex/topic4-fcxr-hyb2`（未 push / 未 merge）。
完整记录见 `docs/archive/topic4/sef_hfo/fcxr_hyb2_a0_undecidable_2026-08-01.md`；
Gate B0 的方法学订正见 `docs/archive/topic4/sef_hfo/fcxr_hyb2_gate_b0_2026-07-31.md`。

| 阶段 | 状态 |
|---|---|
| preflight | 通过 |
| calibration（seed1 / seed3，各 24 s） | 通过并锁定 |
| finalize（全局 `τ_R` + 逐 seed `Q_on`） | 通过；`Q_on` 为**预注册的 calibration 半段**值 |
| zaxis（离线 `S_Z` 重放） | 通过并锁定 |
| **Gate B0** | **两 seed 均 `BASELINE_PRACTICALLY_INVISIBLE`**（膜层判据；离线重判，未重跑 baseline） |
| **Gate A0 q50 初始点** | **`A0_CEILING_CONFOUNDED`** —— off 臂窗内已招募 930/1024 体素（90.8%） |
| **Gate A0 rev2：S25/S50/S75** | **`A0_UNDECIDABLE_ALL_LEVELS`** —— 三档 off 臂分别占据 923/925/924 个体素（90.1%/90.3%/90.2%），均越过 90% 天花板；执行器效力**未判定** |
| 12 格屏 / 七门 / M 臂 / 确认 | 未跑 |

锁定值：`τ_R` = 30.5411 ms；`Q_on` = 112.5047（seed1）/ 173.4290（seed3）；
`I_R,max` = 4.134151260609386；三档 `I_th_EI` = 37.49 / 13.61 / 3.66。

**B0 允许的表述**：在预注册的 calibration-half 阈值下，seed1 出现极稀少的 validation 激活
（20.34 s，占空比 1.077e-4）、seed3 未激活；两个 seed 均未观察到间期事件统计扰动。
**不得写**"一次都没启动"或"逐位证明完全隐形"。

**A0 停机理由**：rev2 在运行前锁定“从 S25/S50/S75 里取最弱 eligible 档；若全部碰天花板，
不得继续降低 Z、加长/平移窗口或修改 90% 门”。三档全部碰天花板，因此 12 格短屏未解锁。
**不得写**“执行器无效”或“得到/否定了可控发作 lifecycle”。
