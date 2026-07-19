# MZ early-field bridge — paper-ready draft 图说明

### fig_mz_early_bridge.png / .pdf
论文候选草图，严格沿用旧 Figure 5 的视觉语法，但全部换成当前 MZ bridge 数据。上方是 **seed1 z-only
native run 的一条连续 virtual-SEEG**，不是两次 replay 拼接：显示窗对应绝对时间 7500–9403.6 ms，图上把窗起点
平移为 0 ms。浅蓝窗是同一连续轨迹内的一个 returning interictal-like burst（绝对时间 7955–8016 ms，包含
`t_off+40 ms` 的完整 contact-readout 尾窗）；浅红窗是 `t_recruit` 后 0–50 ms 的 pre-t120 early-energy window，
红虚线是 operational runaway 的 t120。

蓝窗事件不是按目标能量挑选：固定规则是「slow-off 中事件数更多的方向里，选择 native run 在 `t_recruit` 前最后一个
合格事件」。本例为 TB/B→A，13 个触点可读；它与 slow-off 冻结 TB 模板在共享触点上的 Spearman 为 0.990。下方只保留
两个与上方阴影一一对应的连续场：左 = 蓝窗 exact event 的 30–80 Hz envelope-peak contact recruitment rank
（viridis 紫=早）；右 = 红窗的 pre-t120 virtual-LFP early-energy field（Blues 深=高）。两图均在固定 E1146
registered plane 上用 3 mm Gaussian contact-readout projection 渲染，并叠加完整 15-contact montage。灰色颗粒仅表示同一
固定底物的 E-neuron 几何位置，不编码局部招募；局部组织参与度审计未完成，因此不能把场热点写成其下方组织已直接招募。

**关注点**：左图最早的高轴端是否与右图最热端一致。这个 exact native event 的描述性
earliness–energy Spearman 为 0.912；正式 bridge 统计仍使用 held-out-validated slow-off 双向模板的 mirror-invariant
maxAB（seed1=0.945，within-shaft p=0.0004），不是由这张代表性案例图重新定义。

### fig_mz_cross_seed_transfer.png / .pdf
跨 seed 模板迁移（回答：间期时序模板是**固定 scaffold 的属性**，还是只是同 seed 噪声重放的巧合）。做法：用
seed i 的 slow-off 方向模板去预测 seed j 的 pre-t120 失控能量场（3×3，方向无关 maxAB，每格用**打乱目标能量 seed
在同一根杆内**的 within-shaft null 重算）。A = 3×3 迁移热图（行=模板 seed，列=能量 seed，星=within-shaft
p<0.05）；B = 同一能量 seed 下三个模板 seed 的 maxAB 点是否收敛成一簇；C = 9 个格子里 maxAB 与非相关指标
（场余弦、四分位能量对比）是否一起为正、随之增大。

**关注点**：热图**按列几乎恒定、按行才变**（模板 seed 之间散度约 0.007 ≪ 能量 seed 之间散度约 0.095，约 13 倍）
——说明 maxAB 由「预测哪个 seed 的能量」决定，而不是「用哪个 seed 的模板」；不同 seed 的模板收敛到同一预测，
指向**共同的固定 scaffold**，显著削弱「只是同噪声重放巧合」的替代解释。seed3 那一列仍然弱（绿、无星、贴 null
线）。表在 `../cross_seed_transfer.{json,csv}`。仍是观测层、非因果。

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
