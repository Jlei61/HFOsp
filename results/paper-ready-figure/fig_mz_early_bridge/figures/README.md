# MZ early-ictal bridge — paper-ready draft 图说明

### fig_mz_early_bridge.png / .pdf
论文候选草图（沿用旧 Figure 5 视觉语法，但**全部换成当前 MZ bridge 数据**）。上排两格是**同一 scaffold、
同一 seed、两次独立状态 replay**（不是一条连续轨迹）：左 = slow-off 安静态里一个**预先确定的留出间期事件**
（浅蓝窗）；右 = 只开 `z` 后**跨过失控阈值 t120 之前**的转变片段（粉窗 = `t_recruit` 后 0–50 ms 早期招募窗，
红虚线 = t120）。下排四格：TA（A→B）与 TB（B→A）**两个方向的间期时序模板都画出来**（不只画事后胜出方向；
viridis 紫=早）、pre-t120 早期能量场（Blues）、以及三 seed 的 observed maxAB（星）对各自 within-shaft null p95
（灰线）。maxAB「取哪个方向」只进入右下统计格，不写进案例场；已去掉神经元颗粒层（局部组织参与度审计未完成）。

**关注点**：TB 时序场「最早」的一端（高轴/sink 端）是否与能量场「最热」的一端对上（间期指纹是否预测早期招募
能量）；TA 只有约 7 个可读触点、TB 全 15 个——说明两个方向支撑不同，别把某个固定方向当稳定表型；右下三 seed
里 **seed3 的星正落在自己的 null 线上（弱）**，seed1/4 明显高过。

---

口径（这些从画布移到 README/metadata，不压在图上）：

- 失控 = operational runaway，是**模型代理，不是临床发作**；能量是 virtual-LFP 30–80 Hz 包络能量代理（任意
  单位 a.u.），不是临床宽带功率。
- 候选 `zA_q75_tz5000`，seeds 1/3/4，案例 seed = 1。分母 = **一块 E1146 底物 × 三个随机种子，不是三个患者**。
- 主统计量 = 方向无关的 maxAB；轴是**双向**的，哪端起始由噪声/状态决定（A→B、B→A 都合法），不作为成功判据。
- 精确数字见 `fig_mz_early_bridge_metadata.json`（每 seed 的 rho_a / rho_b / rho_maxab + within-shaft p + null p95）。
- 这是「看得见相关」的观测层结果，**非因果、非发作**；不覆盖旧 `fig5_snn_state_readout`，也不替代两张诊断图。
- 数据源：`results/topic4_sef_hfo/mz_early_field_bridge/per_seed/seed{1,3,4}/`。生产脚本：
  `scripts/paper_figures/plot_fig_mz_early_bridge_paper.py`（plotting-only，不重跑仿真）。
