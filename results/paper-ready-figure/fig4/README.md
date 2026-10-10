# 当前 Figure 4：A–I 重排版

[完整预览](figures/fig4-complete-layout-preview.png) · [PDF](figures/fig4-complete-layout.pdf) · [编号说明](panel_map.md)

按2026-10-09作者草图重排；B保留原第6–16阶段，C保留四core均值±标准差，D/E拆分角度响应与最新位置密度。旧E/F传播showcase不再展示，底排旧G–J改为F–I。横向留白收紧，B/C/D/E/H/I扩大到50×50 mm；D/H与E/I逐列对齐，F/G组合匹配B/C的整体边界。B两端无空白延伸、色条间距1.5 mm；A框位于(-8.5,0) mm，虚线避开电极。紧凑行距保留，右侧机制连接图仍待补充，整图待作者目视检查。

重建：`python scripts/paper_figures/build_fig4_compact_ai.py`；本次前一版保存在`results/paper-ready-figure/archive/2026-10-09_pre_tighter_columns_fig4/fig4`，旧A–J数据包保存在`results/paper-ready-figure/archive/2026-10-09_pre_compact_ai_fig4/fig4`。

A机制重绘候选v9（2026-10-10）：[预览](candidates/a_local_sampling_sigma_20261010/figures/fig4-complete-layout-preview.png)。仅在右侧Local sampling恢复绿色高斯采样权重，按实际σ=0.25 mm绘制，随距离淡出、不画硬边界圈。中央sheet不画采样范围、光晕或绿色填充，只用无填充灰色虚线框定位放大视野。下方两组真实120 ms波形和标签逐像素保持v8，左侧回路与B–I保持；新A待作者目视检查，正式布局仍为v4。
