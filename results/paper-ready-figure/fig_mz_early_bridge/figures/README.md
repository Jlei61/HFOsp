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
跨 seed 模板迁移诊断。用 seed i 的 slow-off 双向模板去预测 seed j 的 pre-t120 能量场；每格重算
mirror-invariant maxAB 和对应的 within-shaft null。A = 3×3 描述性迁移矩阵（行=模板 seed，列=目标能量 seed）；
B = 每个目标场下三个模板 seed 的 maxAB，短横线是每一个格子自己的 null p95，星号只标同 seed 对角格；
C = 对每个目标 seed 直接计算「同 seed 模板 maxAB − 两个外来模板 maxAB 的中位数」。

**关注点**：同一目标场下换模板 seed 只改变约 0.007，而三个目标场之间约 0.095；更直接的 matched 差值约
+0.002 / −0.002 / −0.010，没有描述性的 same-seed 优势，因此削弱了“只是同一噪声 replay”的解释。但 3×3
只有 **3 个目标场重复单位，不是 n=9**；而且本次 9/9 格子都由 B→A 分支赢得 maxAB，所以只能说**被调用的预测
分支可以跨 seed 迁移**，不能宣称两个方向都已证明为 seed-invariant scaffold 属性。表在
`results/topic4_sef_hfo/mz_early_field_bridge/cross_seed_transfer.{json,csv}`；此图是 exploratory diagnostic，
不进入主图推断。

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
