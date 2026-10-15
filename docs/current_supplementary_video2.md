# 当前 Supplementary Video 2

**作者确认定稿｜2026-10-15｜`paired_rows_20261015`｜`AUTHOR_ACCEPTED_FINAL`**

- [正式动画](../results/paper-ready-figure/supplementary-video-2.gif)
- [预览](../results/paper-ready-figure/supplementary-video-2-model-propagation/figures/supplementary-video-2-preview.png)
- [版本指针](../results/paper-ready-figure/supplementary-video-2-model-propagation/current_version.json) · [完整来源与作者确认](../results/paper-ready-figure/supplementary-video-2_metadata.json)

本视频展示模型间期传播的两个拟合条件。上排是 **Higher error / Poorer fit**，下排是 **Lower error / Better fit**；每个条件一整行，依次为 MTA 包络、MTB 包络、MTA 动态场、MTB 动态场，上下逐列对应。醒目条件标签替代内部实验编号，沿用原包络样式、15 触点顺序、模型几何与色标。

作者明确纠正并确认编号为 **Supplementary Video 2**；配套静态合图不编号为 Supplementary Figure 2。视频及预览与作者认可的候选逐字节相同，本次定稿不重新渲染、不重选事件、不重跑模型。

每个事件回放原 250 ms 窗，原生步长 2 ms，125 帧、20 fps，共 6.25 s。四个事件从各自窗口起点同步推进，标示时间各自相对最早有效质心，不能理解为同一绝对时间。包络固定 0–1，原生场固定 0–75 E cells / mm² / 2 ms。较好／较差仅指这两个配置的相对训练拟合误差；该视频展示代表事件，不新增独立验证或因果机制结论。

正式包位于 `results/paper-ready-figure/supplementary-video-2-model-propagation/`；`source/` 保留生成时元数据和脚本。验证及原样导出入口：

```bash
python scripts/paper_figures/build_supplementary_video2_current.py
python scripts/paper_figures/build_supplementary_video2_current.py --output-dir /tmp/supplementary-video-2-export
```

旧 E10/SZ3 围发作场动画及当时记录已[归档](../results/paper-ready-figure/archive/2026-10-15_supplementary_video2_replacement/README.md)，不再占用 Video 2 编号，也未自动编为 Video 3。后续修改输出独立修订；未经作者再次指定，不替换本版。
