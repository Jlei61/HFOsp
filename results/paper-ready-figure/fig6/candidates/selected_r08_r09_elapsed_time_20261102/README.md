# Figure 6：恢复仿真时间着色

[完整图](figures/fig6-complete-layout.png) · [PDF](figures/fig6-complete-layout.pdf) · [B](figures/fig6-panelb.png)

B 主图、小窗及方向箭头统一按原生仿真时间连续着色，色条为 Time (s)，范围 0–120 秒。保留现有完整记录、布局及 A/C/D/E，不再使用周期归一化或黑色覆盖。本包为本轮发布候选，正式指针及人工验收状态保持。

## 发布核验与导出

在仓库根目录运行：

```bash
python scripts/paper_figures/build_fig6_elapsed_time_snapshot.py
python scripts/paper_figures/build_fig6_elapsed_time_snapshot.py --output-dir /tmp/fig6-time-exports
```

上述入口仅用 Python 标准库核验 publication_manifest.json 中的文件并复制冻结 PNG/PDF/SVG，无需原始数据盘，不重跑仿真或统计。

## 数据与原生绘图来源

轨迹、显示波形、八个阶段的放电率场、匹配能量场及队列统计保留于包内 NPZ/JSON/CSV。A–D 来自 seed 9108404 的同一 120 秒原生记录，E 沿用历史队列结果；完整定义见 caption_zh.md。onset_comparisons 与 onset_stage_revision 保留此前匹配和阶段修订的来源记录，不代表另一个当前版本。

source/build_fig6_selected_r08_r09_elapsed_time.py 为本次原生 Matplotlib producer；source/ 同时保存其本地直接导入链和历史迭代源码。原生重绘依赖原工作区、先前候选及外部实验记录；这些环境路径保留用于来源追溯，不能把本包的冻结导出核验说成独立重跑仿真。可移植交付入口是上面的 build_fig6_elapsed_time_snapshot.py。

validation.json 与 visual_review.json 保存生成时的数值/图层检查和 PNG–PDF 一致性检查；publication_manifest.json 保存发布快照哈希。
