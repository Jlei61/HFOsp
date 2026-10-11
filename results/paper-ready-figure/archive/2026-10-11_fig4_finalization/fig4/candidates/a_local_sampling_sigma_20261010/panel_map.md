# Figure 4A 机制重绘候选：A–I

| Panel | 内容 |
|---|---|
| A | 仅右侧Local sampling显示实际sigma的绿色采样权重，中央只留无填充定位框 |
| B | 先验开发后的三类误差：原第6–16阶段194个工作点 |
| C | E1146一至四core最佳训练loss：4次优化重复均值±样本标准差 |
| D | E→E角度的Mean rank／Order／Participation误差响应 |
| E | 413个可评分配置中低J_joint前20%（83个）的位置密度 |
| F | 连续30–80 Hz虚拟接触活动及MTA／MTB事件标记 |
| G | 模型与患者的平均传播rank |
| H | 模型—患者触点交叉匹配矩阵 |
| I | 25位患者TA–MTA／TB–MTB模板相似度 |

A小框位于(-8.5,0) mm，虚线避开电极；相邻横向留白收紧，B/C/D/E/H/I均为50×50 mm，D/H、E/I两侧对齐，B/C整体与F/G组合对齐，紧凑行距保持。A只在右侧采样放大图显示sigma=0.25 mm绿色权重，中央只留无填充定位框；三个连续触点、局部回路与双burst保持，待作者目视检查；旧完整A–J包保存在归档，底排旧G/H/I/J对应新F/G/H/I。

重建：`python scripts/paper_figures/build_fig4_a_mechanism_candidate.py`。
