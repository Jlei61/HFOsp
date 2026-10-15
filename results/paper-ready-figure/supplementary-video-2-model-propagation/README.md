# Supplementary Video 2：作者确认定稿

2026-10-15，`paired_rows_20261015`，`AUTHOR_ACCEPTED_FINAL`。正式 [GIF](figures/supplementary-video-2.gif) 与[预览](figures/supplementary-video-2-preview.png)均逐字节保留作者刚确认的版本；投稿入口为 [supplementary-video-2.gif](../supplementary-video-2.gif)。

上排为 Higher error / Poorer fit，下排为 Lower error / Better fit。每行依次为 MTA 包络、MTB 包络、MTA 动态场、MTB 动态场，上下逐列对应；保留原 15 触点顺序、0–1 包络色标、0–75 原生场色标及事件和模型几何。各事件完整 250 ms，2 ms 一步，共 125 帧、20 fps、6.25 s；时间各自相对最早有效质心，各窗同步起播不表示事件同时发生。

拟合好坏指两个配置的相对训练拟合误差，动画是同一模型来源的代表事件展示，不新增模型普适性或因果机制结论。静态合图只是未编号的配套材料，本次确认不更改 Supplementary Figure 2。

版本指针为 `current_version.json`，完整来源和作者确认记录见 `metadata.json`，生成时冻结记录见 `source/generation_metadata.json`。`python scripts/paper_figures/build_supplementary_video2_current.py` 验证正式文件，增加 `--output-dir <空目录>` 可逐字节导出；后续改动另开修订，不覆盖本版。
