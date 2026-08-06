# M4-MZ discovery / P3 图说明（中文）

> tier = mechanism screen（探索性诊断图，非 paper-ready 主图）。E1146、spontaneous、seed=1、T=12s。

### mz_phenotype_map.png
参数网格上的 7 类表型分布：arm A（去抑制强度 `I_th` × 恢复时间 `τ_z`）、arm B（适应时间 `τ_adp` × 适应电流档）、arm C（z × m）。arm A 从弱去抑制的间期（蓝）过渡到强去抑制的失控（红），中档 + 最慢恢复出现唯一一格有界扩大（橙 exp-B，n=48）；arm B 只在间期 / 压制之间（从不扩大）；arm C 全间期。⚠️ arm C 的 "z weak/mid/strong" 是**目标档位标签**——实测去抑制没覆盖弱档，三档其实都塌到强 z（见最终 report §3 Q8 局限），别读成"弱 z 也测过了"。
**关注点**：去抑制→失控、适应→压制、z+m→回间期这三条主线；以及 arm C 塌角这个 caveat。

### mz_mechanism_traces.png
slow-off + 每臂代表格的时程（上 = 群体率 Hz，中 = 抑制效能 z，下 = 适应电流 η_m·m）。z-only 代表格里 z 缓慢耗竭、事件变密变高但仍 < 120Hz（有界；这是 seed=1 那格，注意 arm A **多数其实失控**，见 phenotype map）；m-only 里一次事件触发大适应电流把后续率压到 ~0（刹车）；z+m 里 z 轻度耗竭 + 适应尖峰 → 稀疏间期（m 抵消 z）。
**关注点**：z 是推（z↓ 则率↑）、m 是拉（适应↑ 则率↓）、z+m 相消回间期这条机制分解；红虚线 = 120Hz 失控判据。

### mz_spatial_recruitment.png
每个代表格的**峰值** E-active footprint（24×24，取时间最大），红圈 = source/sink 核、白虚线 = 病人传播轴。四格都是大片高招募——因为"时间取峰"会饱和（任何 regime 都有某刻局部高活动），所以此图**不区分表型**，只说明一件事：招募在所有 regime 都是**宽 / 全场**的、不是紧贴轴的行波（与均质衬底预期一致）。
**关注点**：招募的空间范围是宽场而非轴向——描述性，本分支不要求破轴；区分 interictal / runaway 请看 phenotype map + traces，不要用此图。
