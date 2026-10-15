# 当前 Figure 4：A–I 重排版

[完整预览](figures/fig4-complete-layout-preview.png) · [PDF](figures/fig4-complete-layout.pdf) · [编号说明](panel_map.md)

按2026-10-09作者草图重排；B保留原第6–16阶段，C保留四core均值±标准差，D/E拆分角度响应与最新位置密度。旧E/F传播showcase不再展示，底排旧G–J改为F–I。横向留白收紧，B/C/D/E/H/I扩大到50×50 mm；D/H与E/I逐列对齐，F/G组合匹配B/C的整体边界。B两端无空白延伸、色条间距1.5 mm；A框位于(-8.5,0) mm，虚线避开电极。紧凑行距保留，右侧机制连接图仍待补充，整图待作者目视检查。

重建：`python scripts/paper_figures/build_fig4_compact_ai.py`；本次前一版保存在`results/paper-ready-figure/archive/2026-10-09_pre_tighter_columns_fig4/fig4`，旧A–J数据包保存在`results/paper-ready-figure/archive/2026-10-09_pre_compact_ai_fig4/fig4`。

A机制重绘候选v10（2026-10-10）：[预览](candidates/a_left_circuit_spacing_20261010/figures/fig4-complete-layout-preview.png)。只调整左侧中央E/I连线及z↓节点：红色下行箭头和蓝色抑制竖线各向外移0.07示意单位，间距由0.08增至0.22单位（当前版面约0.66→1.80 mm）；z↓边框原生线宽由1.60减至0.60 pt，改为1.5/1.5细密虚线。原始回路按冻结绘图代码重建后与v9逐像素一致；修改后的完整拼版只有该细节足迹内的像素变化，中央、右侧、两组波形及B–I均保持v9。新A待作者目视检查，正式布局仍为v4。
