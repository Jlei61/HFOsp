# Fig5参考布局修订（2026-10-11）

入口：[完整PDF](figures/fig5-complete-layout.pdf)；[预览](figures/fig5-complete-layout-preview.png)。按附件省去原栅格行，空间图上移，右下放已有dB统计，连续编号A–F。

原始PDF、数值定义和metadata保存在source/；单panel PDF与来源逐字节一致。A仅在合图隐藏重复时间刻度，不裁波形；所有图层等比例缩放。模型/患者例图保持z场，队列保持作者指定的dB统计，未用拼版修改读出定义或科学结论。

复现：`python scripts/paper_figures/build_fig5_reference_layout.py`（依赖PyMuPDF）。独立候选，当前正式入口未覆盖；作者目视验收待完成。
