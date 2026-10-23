# Figure 2 当前 A–E 排版候选

2026-10-23。完整图和单图见 [figures/README.md](figures/README.md)，状态仍为 `PENDING_AUTHOR_VISUAL_REVIEW`。本次同步包含 A 的 2D projection 与箭头位置、放大的 C rose、统一的 A/D/E 坐标字号和 D/E 对齐，以及 B 的留出三维最近方向误差结果。

## 重建图件

在仓库根目录，使用包含 NumPy、SciPy、scikit-learn、Matplotlib、PyMuPDF 和 threadpoolctl 的 Python 环境；字体使用系统 Liberation Sans：

```bash
python scripts/paper_figures/build_fig2_projection_typography.py --output /tmp/fig2-rebuild --dpi 600
```

重建从随包的上一版 A/D 原生 PDF 与 B 冻结图开始，C 使用 E10/E14 的冻结方向缓存，E 使用同一来源的患者 CSV 和空间 null。全部必要绘图输入已随提交保存；不需要原工作区 `.worktrees/` 或外部 `/data/` 目录。输出会独立存放，不覆盖交付图。

## 结果与计算来源

B 的定义和统计单位见 [实验说明](../../../../../docs/fig2b_nearest_direction_holdout.md)；26 人逐患者 JSON、统计汇总、留出敏感性表及运行清单位于 `results/topic1/nearest_direction_holdout_20261023/`。B 为两折等权的三维最近模板夹角；C 的灰柱为第一折二维投影，红蓝实线为同折空间模板方向，不能把二维 rose 的角距离当作 B 的三维误差。

随包发布实验源码和已完成的验证结果。重新计算全队列事件级误差/训练模型需要原始事件、坐标及完整中间缓存；这些大体积数据不属于本次图件提交。原始路径和参数保留在运行清单中。
