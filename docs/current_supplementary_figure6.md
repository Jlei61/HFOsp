# 当前 Supplementary Figure 6

2026-11-03 更新。当前补图 S6 为**发作早期频谱特征与能量频带的选择**，资产 `early_seizure_spectral_features`，布局 `raw_endpoint_alignment_20261103`。当前版本记录见 [current_version.json](../results/paper-ready-figure/supplementary_onset_spectral_features/current_version.json)，状态仍为 `PENDING_AUTHOR_VISUAL_REVIEW`。

## 本次修改

- **A：** 23 名患者、324 次发作的类型分布；Broadband 与 Gamma 的名称、读出频段分行显示，分别为 1–150 Hz 和 30–80 Hz。低频类型的图内名称缩短为 Low bands，分类与 7 次发作（2.2%）保持不变。Low bands 下方补充 1–13 Hz；A 的标题和内容左移收紧，A/B 的标题字号、字母位置分别与 C/D 对齐。
- **B：** 用 E20 全部 17 次已分类发作的患者内饼图替换原 B。Broadband 与 Gamma 各 6 次（35.3%），Other 为 5 次（29.4%）；百分比完整位于扇区内，类型与发作数分行。B 饼图缩回紧凑尺寸并保持圆形，图例 Broadband 末字母与 D 第一行波形时间终点对齐。
- **C、D：** 保留 Y15/SZ3（D3）和 E20/SZ8（HRB1）的原始脑电及时频示例，D 的显示名称改为 Gamma。C、D 各有独立的 TFR (dB) 色条，范围均为 −14 至 14 dB；左侧色条与右图频率轴文字留有间距。以 C/D 第一行原始波形的时间终点作为两列右边界；A 的 38.3% 后括号对齐 C，B 图例 Broadband 末字母对齐 D。色条保留在 TFR 右侧，不作为列对齐参照。
- **E：** 四个频带子图各有自己的图例，图内左上角竖排、12 号字、紧凑白底。蓝色为 Y15/SZ3 Broadband，绿色为 E20/SZ8 Gamma；曲线、起始虚线及 0–10 s 区域均未被遮挡。E 四个子图按左右两组排列：第二幅 Gamma 图最右 20 刻度的文字末端对齐 C，第四幅最右 20 刻度末端对齐 D；四幅数据区等宽。

图内及中英文图注统一使用 Gamma；原分类字段 `gamma_nonbroadband`、缓存中的 `Gamma-dominant` 和分类标准保留原义。B 支持两类发作在 E20 内共存且占比最高，分母包含 Other；不将这一个患者的比例推广至全队列。

## 文件与报告

- [完整 PNG](../results/paper-ready-figure/supplementary_onset_spectral_features/figures/supp_fig6_onset_spectral_features.png) · [完整 PDF](../results/paper-ready-figure/supplementary_onset_spectral_features/figures/supp_fig6_onset_spectral_features.pdf)
- [A/B 局部图](../results/paper-ready-figure/supplementary_onset_spectral_features/figures/supp_fig6_panels_ab.png) · [E 局部图](../results/paper-ready-figure/supplementary_onset_spectral_features/figures/supp_fig6_panele_power_curves.png) · [TFR 局部图](../results/paper-ready-figure/supplementary_onset_spectral_features/figures/supp_fig6_tfr_panels.png)
- [中文图注](../results/paper-ready-figure/supplementary_onset_spectral_features/caption_zh.md) · [英文图注](../results/paper-ready-figure/supplementary_onset_spectral_features/caption_en.md) · [方法与分类定义](../results/paper-ready-figure/supplementary_onset_spectral_features/methods.md)
- [Supplementary 逐项检查报告](weekly_reports/2026-10-30_supplementary_review/2026-10-30_癫痫文章_Supplementary逐项检查.md) · [正文与主图对应审阅](weekly_reports/2026-10-30_supplementary_review/Supplementary_正文与主图对应审阅.md) · [Obsidian 打包版](weekly_reports/2026-10-30_supplementary_review_Obsidian.zip)

## 后续 agent 的修改入口

共享机器上的其他 worktree 先读 `/home/honglab/leijiaxin/HFOsp/docs/current_supplementary_figure6.md` 与主目录的版本记录。原生 producer 为 [build_supplementary_onset_spectral_features.py](../scripts/paper_figures/build_supplementary_onset_spectral_features.py)，运行 `python scripts/paper_figures/build_supplementary_onset_spectral_features.py`。数据沿用包内 `data/` 及 metadata 登记的分类来源。发布包带有两例的绘图缓存和分类表；原始分类目录不可用时，producer 自动读取包内同一分类表，可在独立 checkout 中重绘。重新从原始 EEG 提取仍需原始数据盘。

修改后同步包内中英文图注、检查报告 S6 节及 `captions/S6_zh.md`、`captions/S6_en.md`，并更新中文 LaTeX 与英文 Supplementary 图注合集。报告中的 `figures/current/S6*`、`asset_manifest.json`、`report_validation.json` 和 Obsidian ZIP 应对应同一版本。图像重绘后检查完整图与局部 PNG/PDF；单独修改图注时核对图文对应即可。

当前 S5 为患者传播截图，S6 为本页频谱图，S7 为多频带空间相关，S8 为模型传播对比。旧 RNN S6、历史旧 S6 编号以及主文 Figure 6 均不能替代本页资产。冻结论文正文保持不变。
